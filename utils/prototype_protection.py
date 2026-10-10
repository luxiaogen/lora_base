"""当前训练图片的原型分差敏感度；只构造任务内保护位置。"""
from contextlib import contextmanager, nullcontext
import json
import logging
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from utils.dual_mask_metrics import _prototype_holdout_mask
from utils.dualmask_core_audit import round_robin_indices
from utils.protect_position import permute_protect_mask


POSITIONS = ('wpre', 'prototype_high', 'prototype_shuffled', 'prototype_low', 'permuted')
MATCHED_POSITIONS = (0, 1, 2, 4)


def probe_indices(labels, fit_limit=128, holdout_limit=64):
    labels = np.asarray(labels)
    holdout = _prototype_holdout_mask(torch.as_tensor(labels), torch.arange(len(labels)), 5).numpy()
    def select(selected, limit):
        indices = np.flatnonzero(selected)
        chosen = round_robin_indices(labels[indices], np.unique(labels), limit)
        return indices[chosen].tolist()
    return select(~holdout, fit_limit), select(holdout, holdout_limit)


def shuffled_prototypes(prototypes, seed, task):
    generator = torch.Generator().manual_seed(int(seed) + 15485863 * int(task))
    order = torch.arange(len(prototypes))
    permutation = torch.randperm(len(prototypes), generator=generator)
    while torch.any(permutation == order):
        permutation = torch.randperm(len(prototypes), generator=generator)
    return prototypes.detach()[permutation.to(prototypes.device)], permutation


@contextmanager
def probe_state(learner, modules, anchor=False):
    network = learner._network.module if isinstance(learner._network, torch.nn.DataParallel) else learner._network
    modes = [(module, module.training) for module in network.modules()]
    flags = [(param, param.requires_grad) for param in network.parameters()]
    attributes = ('_pending_safe_residual_deltas', '_last_safe_residual_loss',
                  '_composed_forward_gate', '_p_direction_state', '_prototype_audit_position',
                  '_core_audit_position')
    transient = [(module, {name: getattr(module, name, None) for name in attributes}) for module in modules]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = learner._device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            with learner._pretrained_anchor_context() if anchor else nullcontext():
                yield network
    finally:
        for param, flag in flags:
            param.requires_grad_(flag)
        for module, training in modes:
            module.training = training
        for module, values in transient:
            for name, value in values.items():
                setattr(module, name, value)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def margin_values(features, prototypes, targets):
    logits = F.normalize(features, dim=1) @ F.normalize(prototypes.detach().to(features), dim=1).t()
    competitors = logits.detach().clone()
    competitors.scatter_(1, targets[:, None], -torch.inf)
    negative = competitors.argmax(1)
    margin = logits.gather(1, targets[:, None]) - logits.gather(1, negative[:, None])
    return margin.squeeze(1), logits.detach().argmax(1)


def estimate_scores(learner, modules, dataset, indices, prototypes, shuffled):
    weights = [module.qkv.weight for module in modules]
    scores = [[torch.zeros_like(weight, dtype=torch.float64 if weight.dtype == torch.float64 else torch.float32)
               for weight in weights] for _ in range(2)]
    correct, margin_sum = 0, 0.
    with probe_state(learner, modules, anchor=True) as network, torch.enable_grad():
        for weight in weights:
            weight.requires_grad_(True)
        for index in indices:
            _, inputs, label = dataset[index]
            features = network.extract_vector(inputs.unsqueeze(0).to(learner._device))
            target = torch.tensor([int(label)], device=features.device)
            for bank, prototype in enumerate((prototypes, shuffled)):
                margin, prediction = margin_values(features, prototype, target)
                gradients = torch.autograd.grad(margin.sum(), weights, retain_graph=bank == 0)
                for score, gradient in zip(scores[bank], gradients):
                    score.add_(gradient.detach().to(score).square())
                if bank == 0:
                    margin_sum += float(margin.detach())
                    correct += int(prediction.eq(target).sum())
    for bank in scores:
        for score in bank:
            score.div_(len(indices))
    return scores, dict(samples=len(indices), prototype_correct=correct,
                        mean_margin=margin_sum / len(indices))


@torch.no_grad()
def ranked_mask(score, reference, lowest=False):
    result = []
    for values, original in zip(score.chunk(3), reference.chunk(3)):
        order = values.flatten().argsort(descending=not lowest, stable=True)
        selected = torch.zeros_like(original, dtype=torch.bool).flatten()
        selected[order[:int(original.sum())]] = True
        result.append(selected.reshape_as(original))
    return torch.cat(result)


def prepare_positions(learner):
    modules = list(learner._iter_lora_modules())
    dataset = learner.w0_loader.dataset
    fit, holdout = probe_indices(dataset.labels)
    prototypes = torch.stack([learner._w0_class_means[i] for i in range(learner._total_classes)]).to(learner._device)
    shuffled, permutation = shuffled_prototypes(prototypes, learner.args['seed'], learner._cur_task)
    started = time.perf_counter()
    scores, information = estimate_scores(learner, modules, dataset, fit, prototypes, shuffled)
    for module, score, noise_score in zip(modules, *scores):
        reference = module.core_reference_protect.bool()
        module.prototype_position_masks = torch.stack((reference,
            ranked_mask(score, reference), ranked_mask(noise_score, reference),
            ranked_mask(score, reference, lowest=True),
            permute_protect_mask(reference, learner.args['seed'], module.layer_idx)))
        position = learner.args.get('dual_mask_protect_position', 'wpre')
        protect = module.prototype_position_masks[POSITIONS.index(position)]
        module.general_mask.copy_(protect)
        module.isolated_mask.copy_(~protect)
        counts = [int(part.sum()) for part in reference.chunk(3)]
        for name, mask in zip(POSITIONS, module.prototype_position_masks):
            union = (reference | mask).sum().clamp_min(1)
            logging.info('PrototypePositionMask %s', json.dumps(dict(task=learner._cur_task,
                layer=module.layer_idx, position=name, selected_position=position,
                qkv_counts=[int(part.sum()) for part in mask.chunk(3)], reference_counts=counts,
                reference_jaccard=float((reference & mask).sum() / union),
                spectral_mass=float((module.w0_importance * mask).sum() /
                                    module.w0_importance.sum().clamp_min(1e-12)))))
    del scores, score, noise_score
    # 独立20%图片检验选中位置是否捕获分差敏感度；只报告，不改变掩码。
    holdout_info = None
    if holdout:
        holdout_scores, holdout_info = estimate_scores(learner, modules, dataset, holdout, prototypes, shuffled)
        for module, true_score, false_score in zip(modules, *holdout_scores):
            for position, mask in zip(POSITIONS, module.prototype_position_masks):
                logging.info('PrototypePositionHoldout %s', json.dumps(dict(task=learner._cur_task,
                    layer=module.layer_idx, position=position, samples=len(holdout),
                    true_sensitive_mass=float((true_score * mask).sum() / true_score.sum().clamp_min(1e-30)),
                    shuffled_sensitive_mass=float((false_score * mask).sum() / false_score.sum().clamp_min(1e-30)))))
        del holdout_scores, true_score, false_score
    logging.info('PrototypePositionProbe %s', json.dumps(dict(task=learner._cur_task,
        source='current_train_only', fit_indices=fit, holdout_indices=holdout,
        class_permutation=permutation.tolist(), seconds=time.perf_counter() - started,
        holdout=holdout_info,
        mask_bytes=sum(m.prototype_position_masks.numel() * m.prototype_position_masks.element_size()
                       for m in modules), **information)))


def position_delta(module, delta, isolated, conflict_ratio, conflict_strength):
    if module._effective_gate_mode() == 'protect_only' or not module._conflict_gate_enabled(isolated):
        applied = torch.zeros_like(delta)
    else:
        _, applied = module._branch_conflict(delta, isolated, conflict_ratio)
    masks = module.prototype_position_masks
    with torch.no_grad():
        gates = []
        for index in MATCHED_POSITIONS:
            mask = masks[index].to(delta)
            strength = module.effective_protect_strength
            if not isolated and module.args.get('dual_mask_protection_rule', 'static') != 'static':
                from utils.protection_strength import protection_alpha
                strength = protection_alpha(module, delta, mask, 1 - conflict_strength * applied.to(delta))
            permission = 1 - mask if isolated else 1 - strength * mask
            gates.append(permission * (1 - conflict_strength * applied.to(delta)))
        shape = (3, delta.shape[0] // 3, delta.shape[1])
        norms = torch.stack([(delta.detach().float() * gate.float()).reshape(3, -1).norm(dim=1)
                             for gate in gates])
        target = norms.min(0).values
        scales = torch.where(norms > 0, target / norms.clamp_min(1e-30), torch.ones_like(norms))
        position = module._prototype_audit_position or module.args.get('dual_mask_protect_position', 'wpre')
        chosen = MATCHED_POSITIONS.index(POSITIONS.index(position))
        gate = gates[chosen] * scales[chosen].repeat_interleave(shape[1]).unsqueeze(1).to(delta)
    return delta * gate, gate, applied


def diagnose_positions(learner, dataset, epoch):
    """任务末、合并前的同状态四位置诊断；测试标签只用于报告。"""
    if learner._cur_task not in (1, 5, 9):
        return
    modules = list(learner._iter_lora_modules())
    indices = round_robin_indices(dataset.labels, range(learner._known_classes), 512)
    indices += round_robin_indices(dataset.labels, range(learner._known_classes, learner._total_classes), 128)
    loader = DataLoader(Subset(dataset, indices), batch_size=learner.batch_size, shuffle=False,
                        num_workers=0, generator=torch.Generator().manual_seed(1993))
    totals = {(partition, position): dict(count=0, correct=0, changed=0, margin=0.)
              for partition in ('old', 'new') for position in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted')}
    with probe_state(learner, modules) as network, torch.no_grad():
        for module in modules:
            for branch, isolated, gamma, unit in (('S', False, module._shared_gamma(learner._cur_task),
                    module.S_lora[learner._cur_task]), ('P', True, module.plora_gamma,
                    module.P_lora[learner._cur_task])):
                raw = unit.B_weight @ unit.A_weight
                candidates = []
                for position in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted'):
                    module._prototype_audit_position = position
                    candidates.append(gamma * module._safe_delta(raw, isolated))
                norms = torch.stack([u.float().reshape(3, -1).norm(dim=1) for u in candidates])
                residual = norms.max(0).values - norms.min(0).values
                for index, position in enumerate(('wpre', 'prototype_high', 'prototype_shuffled', 'permuted')):
                    for p, projection in enumerate(('Q', 'K', 'V')):
                        logging.info('PrototypePositionNorm %s', json.dumps(dict(task=learner._cur_task,
                            epoch=epoch, layer=module.layer_idx, branch=branch, projection=projection,
                            position=position, raw_norm=float((gamma * raw).chunk(3)[p].float().norm()),
                            effective_norm=float(norms[index, p]),
                            removed_norm=float((gamma * raw - candidates[index]).chunk(3)[p].float().norm()),
                            same_state_norm_residual=float(residual[p]))))
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(learner._device), targets.to(learner._device)
            reference_prediction = None
            for position in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted'):
                for module in modules:
                    module._prototype_audit_position = position
                logits = network.interface(inputs)
                positive = logits.gather(1, targets[:, None]).squeeze(1)
                negative = logits.clone()
                negative.scatter_(1, targets[:, None], -torch.inf)
                margin, prediction = positive - negative.max(1).values, logits.argmax(1)
                if reference_prediction is None:
                    reference_prediction = prediction
                for partition, selected in (('old', targets < learner._known_classes),
                                            ('new', targets >= learner._known_classes)):
                    row = totals[partition, position]
                    row['count'] += int(selected.sum())
                    row['correct'] += int(prediction[selected].eq(targets[selected]).sum())
                    row['changed'] += int(prediction[selected].ne(reference_prediction[selected]).sum())
                    row['margin'] += float(margin[selected].sum())
    for (partition, position), row in totals.items():
        count = row['count']
        logging.info('PrototypePositionDiagnostic %s', json.dumps(dict(task=learner._cur_task,
            epoch=epoch, partition=partition, position=position, scope='all_seen',
            source='test_only_fixed_order', norm_control='same_state_four_position_min',
            accuracy=100 * row['correct'] / count if count else None,
            mean_margin=row['margin'] / count if count else None, **row)))
