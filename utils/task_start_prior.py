"""从已合并任务起点权重提取谱位置，独立于固定 W_pre 缓存。"""
import json
import logging
import time

import torch

from utils.protect_position import permute_protect_mask


def count_matched_mask(score, reference):
    """逐 Q/K/V 匹配参照保护数量；并列分数按坐标索引选择。"""
    parts = []
    for value, original in zip(score.chunk(3), reference.chunk(3)):
        selected = torch.zeros_like(original).flatten()
        count = int(original.sum())
        if count:
            selected[value.flatten().argsort(descending=True, stable=True)[:count]] = 1
        parts.append(selected.reshape_as(original))
    return torch.cat(parts)


def correlation(first, second):
    first, second = first.float().flatten(), second.float().flatten()
    a, b = first - first.mean(), second - second.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if float(denominator) else None


def jaccard(first, second):
    first, second = first.bool(), second.bool()
    union = int((first | second).sum())
    return int((first & second).sum()) / union if union else 1.


@torch.no_grad()
def install_task_start_prior(module, reference):
    from models.attention import _normalize_score
    if module.task_start_score_task != module.cur_task:
        device = module.qkv.weight.device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metadata = module.last_svd_rank, module.last_svd_energy_coverage
        try:
            spectral = (module._soft_svd_importance if module.dual_mask_importance == 'soft_svd'
                        else module._svd_importance)(module.qkv.weight.detach())
            module.task_start_importance = _normalize_score(spectral).detach()
            module.task_start_svd_metadata = module.last_svd_rank, module.last_svd_energy_coverage
        finally:
            module.last_svd_rank, module.last_svd_energy_coverage = metadata
        module.task_start_permuted_importance = permute_protect_mask(
            module.task_start_importance, module.args.get('seed', 1993), module.layer_idx)
        module.task_start_score_task = module.cur_task
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        module.task_start_score_seconds = time.perf_counter() - started
    module.task_start_protect = count_matched_mask(module.task_start_importance, reference)
    module.core_reference_protect = module.task_start_protect
    module.core_permuted_protect = permute_protect_mask(
        module.task_start_protect, module.args.get('seed', 1993), module.layer_idx)
    module.core_diagnostic_positions = ('task_start_svd', 'task_start_svd_permuted')
    logging.info('TaskStartPrior %s', json.dumps(dict(task=module.cur_task, layer=module.layer_idx,
        relative_weight_drift=module.relative_weight_drift(),
        task_start_svd_rank=module.task_start_svd_metadata[0],
        task_start_svd_energy=module.task_start_svd_metadata[1],
        score_seconds=module.task_start_score_seconds,
        qkv_score_correlations=[correlation(a, b) for a, b in
            zip(module.w0_importance.chunk(3), module.task_start_importance.chunk(3))],
        qkv_reference_counts=[int(m.sum()) for m in reference.chunk(3)],
        qkv_task_start_counts=[int(m.sum()) for m in module.task_start_protect.chunk(3)],
        qkv_reference_jaccards=[jaccard(a, b) for a, b in
            zip(reference.chunk(3), module.task_start_protect.chunk(3))])))


@torch.no_grad()
def conflict_overlap(module, raw, actual):
    from models.attention import _exact_top_ratio_mask, _normalize_score
    magnitude = _normalize_score(raw.detach().abs())
    scores = {'magnitude': magnitude, 'wpre': _normalize_score(module.w0_importance.to(raw) * magnitude),
              'task_start': _normalize_score(module.task_start_importance.to(raw) * magnitude),
              'task_start_permuted': _normalize_score(module.task_start_permuted_importance.to(raw) * magnitude)}
    masks = {name: _exact_top_ratio_mask(score, module.dual_mask_conflict_ratio).bool()
             for name, score in scores.items()}
    return dict(score_mode=module.dual_mask_conflict_score_mode,
        importance_source=module.args.get('dual_mask_conflict_importance_source', 'wpre'),
        selected_coordinates=int(actual.sum()),
        actual_mask_jaccards={name: jaccard(actual, mask) for name, mask in masks.items()},
        task_start_vs_wpre_jaccard=jaccard(masks['task_start'], masks['wpre']),
        task_start_vs_permuted_jaccard=jaccard(masks['task_start'], masks['task_start_permuted']))
