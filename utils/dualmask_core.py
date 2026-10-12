"""Permissions and same-state norm control for the DualMask attribution runs."""
import torch


def permission_gate(protect, strength, isolated, mode, s_protect=True):
    if mode == 'symmetric_hard':
        return 1 - protect
    if mode == 'symmetric_soft':
        return 1 - strength * protect
    if isolated:
        return 1 - protect
    return 1 - strength * protect if s_protect else torch.ones_like(protect)


@torch.no_grad()
def paired_min_gates(delta, first, second):
    shape = (3, delta.shape[0] // 3, delta.shape[1])
    value = delta.detach().float().reshape(shape)
    norms = torch.stack([(value * gate.reshape(shape)).flatten(1).norm(dim=1)
                         for gate in (first, second)])
    target = norms.min(dim=0).values
    # Identity at zero preserves the gradient of a zero-initialized B.
    scales = torch.where(norms > 0, target / norms.clamp_min(1e-30), torch.ones_like(norms))
    gates = tuple(gate * scale.repeat_interleave(shape[1]).unsqueeze(1).to(gate)
                  for gate, scale in zip((first, second), scales))
    return gates, norms, target, scales


@torch.no_grad()
def paired_layer_min_gates(delta, first, second):
    """整层统一缩放，保留保护区与区外的预算比例。"""
    value = delta.detach().float()
    norms = torch.stack([(value * gate).norm() for gate in (first, second)])
    target = norms.min()
    scales = torch.where(norms > 0, target / norms.clamp_min(1e-30), torch.ones_like(norms))
    gates = tuple(gate * scale.to(gate) for gate, scale in zip((first, second), scales))
    return gates, norms, target, scales


@torch.no_grad()
def prior_layer_min_gates(delta, spectrum, magnitude, permuted):
    """谱、权重幅度、置换三位置使用同一整层目标，只缩小更新。"""
    value = delta.detach().float()
    originals = (spectrum, magnitude, permuted)
    norms = torch.stack([(value * gate).norm() for gate in originals])
    target = norms.min()
    scales = torch.where(norms > 0, target / norms.clamp_min(1e-30), torch.ones_like(norms))
    gates = tuple(gate * scale.to(gate) for gate, scale in zip(originals, scales))
    return gates, norms, target, scales
