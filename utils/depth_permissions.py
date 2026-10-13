# 固定分层B实现；训练行为由回归测试核验。
"""用固定深度曲线安装更新权限，不引入训练损失或历史激活。"""
import logging
import torch

def depth_value(center, span, layer, layers):
    return center if layers == 1 else center + span * (1 - 2 * layer / (layers - 1))

def calibrate_center(scores, center=0.9, span=0.05):
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
    target = sum((count(prefix, total, center) for prefix, total in prefixes))

    def evaluate(value):
        return sum((count(prefix, total, depth_value(value, span, layer, len(scores))) for layer, (prefix, total) in enumerate(prefixes)))
    best, actual = (center, evaluate(center))
    low, high = (span, 1 - span)
    for _ in range(28):
        if actual == target:
            break
        candidate = (low + high) / 2
        measured = evaluate(candidate)
        if (abs(measured - target), abs(candidate - center)) < (abs(actual - target), abs(best - center)):
            best, actual = (candidate, measured)
        if measured < target:
            low = candidate
        else:
            high = candidate
    return best

@torch.no_grad()
def install_depth_permissions(modules, task, args, center=None):
    if task == 0:
        return center
    modules = list(modules)
    reference_coverage = float(args.get('protect_mass_coverage', .90))
    span = float(args.get('depth_coverage_span', .05))
    if center is None:
        center = calibrate_center([module.w0_importance for module in modules], reference_coverage, span)
    for layer, module in enumerate(modules):
        module.effective_protect_strength = float(args.get('protect_strength', .5))
        coverage = depth_value(center, span, layer, len(modules))
        module.rebuild_dual_masks(coverage_override=coverage)
    logging.info('Task %s permissions => q_center %.9f, q_first %.6f, q_last %.6f, '
                 'alpha %.3f, S/P %s/%s, protected_coordinates %s', task, center,
                 modules[0].effective_energy_coverage, modules[-1].effective_energy_coverage,
                 modules[0].effective_protect_strength, modules[0].S_lora[task].r,
                 modules[0].P_lora[task].r, sum(int(m.general_mask.sum()) for m in modules))
    return center
