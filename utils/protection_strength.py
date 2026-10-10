"""只改变S保护强度；更新量规则使用当前BA，不保存历史统计。"""
import torch


RULES = ('static', 'ncm_all', 'ncm_risk', 'warmup', 'cooldown', 'energy_equal', 'energy_half')


def protection_alpha(module, raw, protect, conflict):
    rule = module.args.get('dual_mask_protection_rule', 'static')
    fixed = min(max(module.effective_protect_strength, 0.), 1.)
    if module.cur_task == 0 or rule == 'static':
        return fixed
    if rule == 'ncm_all':
        return max(0., module.pretrained_competence - module.pretrained_old_overlap_risk)
    if rule == 'ncm_risk':
        return module.pretrained_old_overlap_risk
    if rule in ('warmup', 'cooldown'):
        progress = getattr(module, 'protection_progress', 1.)
        return .5 * progress if rule == 'warmup' else 1. - .5 * progress
    with torch.no_grad():
        value = raw.detach().float() * conflict.float()
        inside = (value * protect.float()).norm()
        outside = (value * (1 - protect.float())).norm()
        ratio = 1. if rule == 'energy_equal' else .5
        strength = (1 - ratio * outside / inside.clamp_min(1e-30)).clamp(0, 1)
        # B=0时保留软门和学习梯度；无保护区能量时不额外抑制。
        strength = torch.where(inside > 0, strength, torch.zeros_like(strength))
        return torch.where(inside + outside > 0, strength, value.new_tensor(fixed)).to(raw)


def protection_update(module, raw, conflict_ratio=None, conflict_strength=None, protect=None):
    protect = module.general_mask.to(raw) if protect is None else protect.to(raw)
    if conflict_strength is None:
        conflict_strength = module._conflict_parameters()[1]
    if module._effective_gate_mode() == 'protect_only' or not module._conflict_gate_enabled(False):
        selected = torch.zeros_like(raw)
    else:
        _, selected = module._branch_conflict(raw, False, conflict_ratio)
    conflict = 1 - conflict_strength * selected.to(raw)
    alpha = protection_alpha(module, raw, protect, conflict)
    permission = 1 - alpha * protect if module.dual_mask_s_protect_enabled else torch.ones_like(protect)
    base, gate = raw * permission, permission * conflict
    return dict(alpha=alpha, base=base, safe=raw * gate, gate=gate, applied=selected)
