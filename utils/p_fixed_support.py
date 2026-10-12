"""预热后固定 P 的有效更新坐标；不剪 A/B，不增加训练损失。"""
import hashlib
import json
import logging

import torch


def select_support(raw, plastic, keep_ratio, random=False, seed=0):
    """分母为整层可塑坐标数；幅度并列时按原坐标索引选择。"""
    valid = plastic.bool().flatten().nonzero(as_tuple=True)[0]
    count = int(valid.numel() * keep_ratio)
    selected = torch.zeros_like(raw, dtype=torch.bool).flatten()
    if count:
        if random:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            order = torch.randperm(valid.numel(), generator=generator)[:count].to(valid.device)
        else:
            scores = raw.detach().abs().flatten()[valid]
            order = torch.argsort(scores, descending=True, stable=True)[:count]
        selected[valid[order]] = True
    return selected.reshape_as(raw)


def support_update(raw, plastic, support, mode):
    """均匀组匹配固定幅度支持在同一 BA 状态下的额外移除范数。"""
    plastic = plastic.to(raw)
    base = raw * plastic
    removed_mask = plastic.bool() & ~support.bool()
    with torch.no_grad():
        reference = (base.detach().float() * removed_mask).norm()
        denominator = base.detach().float().norm()
        coefficient = (reference / denominator.clamp_min(1e-30)).clamp(0, 1).to(raw)
    if mode == 'uniform_matched':
        gate = plastic * (1 - coefficient)
        applied = plastic.bool() & (coefficient > 0)
    else:
        gate = plastic * support.to(raw)
        applied = removed_mask
    return dict(base=base, safe=raw * gate, gate=gate, applied=applied.to(raw),
                coefficient=coefficient, reference_removed_norm=reference)


@torch.no_grad()
def update_support(module, completed_epochs):
    if module.cur_task == 0:
        return
    mode = module.args.get('p_fixed_support', 'off')
    audit = module.args.get('p_fixed_support_audit', False)
    if mode == 'off' and not audit:
        return
    unit = module.P_lora[module.cur_task]
    raw = unit.B_weight @ unit.A_weight
    plastic = (1 - module.general_mask).bool()
    warmup = int(module.args.get('p_fixed_support_warmup_epochs', 2))
    if module.p_fixed_support_mask is None and completed_epochs == warmup:
        ratio = float(module.args.get('p_fixed_support_keep_ratio', .5))
        seed = module.args.get('seed', 0)
        seed = seed[0] if isinstance(seed, (list, tuple)) else seed
        seed = int(seed) + 104729 * module.cur_task + 1009 * module.layer_idx
        support = select_support(raw, plastic, ratio, mode == 'random', seed)
        module.p_fixed_support_mask = support
        logging.info('PFixedSupportSelection %s', json.dumps(dict(
            task=module.cur_task, layer=module.layer_idx, mode=mode,
            capture_epoch=completed_epochs, first_training_epoch=completed_epochs + 1,
            keep_ratio=ratio, plastic_coordinates=int(plastic.sum()),
            retained_coordinates=int(support.sum()), random_seed=seed if mode == 'random' else None,
            mask_sha256=hashlib.sha256(support.cpu().numpy().tobytes()).hexdigest(),
            audit_only=mode == 'off')))
    if audit and completed_epochs in (1, warmup, warmup + 1, 5, 10, 20):
        audit_update(module, raw, plastic, completed_epochs)


@torch.no_grad()
def audit_update(module, raw, plastic, epoch, event='epoch'):
    mode = module.args.get('p_fixed_support', 'off')
    support = module.p_fixed_support_mask
    active = mode != 'off' and support is not None
    state = support_update(raw, plastic, support, mode) if active else None
    p_safe = state['safe'] if active else module._safe_delta(raw, True)
    shared = module.S_lora[module.cur_task]
    s_raw = shared.B_weight @ shared.A_weight
    s_safe = module._safe_delta(s_raw, False)
    gs, gp = float(module.slora_gamma), float(module.plora_gamma)
    base = raw * plastic
    actual_removed = (base - p_safe).float().norm()
    target = state['reference_removed_norm'] if active else None
    logging.info('PFixedSupportUpdate %s', json.dumps(dict(
        task=module.cur_task, layer=module.layer_idx, epoch=epoch, event=event,
        mode=mode, fixed_active=active, plastic_coordinates=int(plastic.sum()),
        support_coordinates=int(support.sum()) if support is not None else None,
        effective_nonzero_coordinates=int((p_safe != 0).sum()),
        s_raw_norm=float(gs * s_raw.float().norm()),
        s_effective_norm=float(gs * s_safe.float().norm()),
        s_removed_norm=float(gs * (s_raw - s_safe).float().norm()),
        p_raw_norm=float(gp * raw.float().norm()),
        p_plastic_norm=float(gp * base.float().norm()),
        p_effective_norm=float(gp * p_safe.float().norm()),
        p_removed_norm=float(gp * (raw - p_safe).float().norm()),
        extra_removed_norm=float(gp * actual_removed),
        reference_removed_norm=float(gp * target) if target is not None else None,
        matching_residual=float(gp * (actual_removed - target).abs()) if active and mode == 'uniform_matched' else None,
        uniform_coefficient=float(state['coefficient']) if active and mode == 'uniform_matched' else None,
        net_effective_norm=float((gs * s_safe + gp * p_safe).float().norm()))))
