# 固定分层B实现；训练行为由回归测试核验。
"""用固定深度曲线安装更新权限，不引入训练损失或历史激活。"""
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
    if center is None:
        center = calibrate_center([module.w0_importance for module in modules])
    for layer, module in enumerate(modules):
        # 默认复现B；开启时只改变S软保护，P仍使用硬补集权限。
        module.effective_protect_strength = (depth_value(0.50, 0.25, layer, len(modules))
                                             if args.get('layerwise_alpha', False) else 0.5)
        coverage = depth_value(center, 0.05, layer, len(modules))
        module.rebuild_dual_masks(coverage_override=coverage)
    return center
