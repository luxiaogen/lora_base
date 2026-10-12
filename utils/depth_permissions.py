"""用固定深度曲线安装更新权限，不引入训练损失或历史激活。"""
import json
import logging
import time

import torch


def depth_value(center, span, layer, layers):
    return center if layers == 1 else center + span * (1 - 2 * layer / (layers - 1))


def calibrate_center(scores, center=.90, span=.05):
    """只校准曲线中心；保留原重要性质量前缀选择的离散数量语义。"""
    prefixes = []
    for score in scores:
        flat = score.detach().float().flatten().clamp_min(0)
        prefixes.append((torch.sort(flat, descending=True).values.cumsum(0), flat.sum()))

    def count(prefix, total, coverage):
        if total <= 0 or coverage <= 0:
            return 0
        if coverage >= 1:
            return prefix.numel()
        return min(int(torch.searchsorted(prefix, coverage * total)) + 1, prefix.numel())

    target = sum(count(prefix, total, center) for prefix, total in prefixes)

    def evaluate(value):
        return sum(count(prefix, total, depth_value(value, span, layer, len(scores)))
                   for layer, (prefix, total) in enumerate(prefixes))

    best, actual = center, evaluate(center)
    low, high = span, 1 - span
    for _ in range(28):
        if actual == target:
            break
        candidate = (low + high) / 2
        measured = evaluate(candidate)
        if (abs(measured - target), abs(candidate - center)) < (abs(actual - target), abs(best - center)):
            best, actual = candidate, measured
        if measured < target:
            low = candidate
        else:
            high = candidate
    return best


@torch.no_grad()
def install_depth_permissions(modules, task, args, center=None):
    coverage_mode = args.get('dual_mask_depth_coverage', 'off')
    strength_mode = args.get('dual_mask_depth_strength', 'off')
    audit = args.get('dual_mask_depth_audit', False)
    if task == 0 or coverage_mode == strength_mode == 'off' and not audit:
        return center
    started = time.perf_counter()
    modules = list(modules)
    # before_task已经建立统一0.90参照；校准中心只在首个增量任务计算。
    reference_counts = [int(module.general_mask.sum()) for module in modules]
    if coverage_mode != 'off' and center is None:
        center = calibrate_center([module.w0_importance for module in modules])
    for layer, module in enumerate(modules):
        alpha = depth_value(.50, .25, layer, len(modules)) if strength_mode != 'off' else .50
        module.effective_protect_strength = alpha
        coverage = depth_value(center, .05, layer, len(modules)) if coverage_mode != 'off' else .90
        if coverage_mode != 'off':
            module.rebuild_dual_masks(coverage_override=coverage)
        if audit:
            score, mask = module.w0_importance, module.general_mask
            logging.info('DepthPermission %s', json.dumps(dict(
                task=task, layer=layer, layers=len(modules), coverage_mode=coverage_mode,
                strength_mode=strength_mode, coverage_target=coverage, alpha=alpha,
                s_rank=module.S_lora[task].r, p_rank=module.P_lora[task].r,
                reference_count=reference_counts[layer], protect_count=int(mask.sum()),
                qkv_protect_counts=[int(part.sum()) for part in mask.chunk(3)],
                protected_importance_fraction=float((score * mask).sum() / score.sum().clamp_min(1e-12)))))
    if audit:
        actual = sum(int(module.general_mask.sum()) for module in modules)
        logging.info('DepthPermissionBudget %s', json.dumps(dict(
            task=task, center=center if coverage_mode != 'off' else .90,
            reference_count=sum(reference_counts), protect_count=actual,
            count_residual=actual - sum(reference_counts), seconds=time.perf_counter() - started)))
    return center
