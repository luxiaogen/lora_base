import math

import torch


def global_topk_masks(tensors, retain_ratio):
    """Return exact-budget masks for the largest magnitudes across tensors."""
    tensors = list(tensors)
    if not tensors:
        return []

    retain_ratio = min(max(float(retain_ratio), 0.0), 1.0)
    sizes = [tensor.numel() for tensor in tensors]
    total = sum(sizes)
    keep = min(total, max(0, int(math.floor(total * retain_ratio))))
    masks = [torch.zeros_like(tensor, dtype=torch.bool) for tensor in tensors]
    if keep == 0:
        return masks
    if keep == total:
        return [torch.ones_like(tensor, dtype=torch.bool) for tensor in tensors]

    values = torch.cat([
        tensor.detach().abs().reshape(-1).float().cpu()
        for tensor in tensors
    ])
    selected = torch.topk(values, keep, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total, dtype=torch.bool)
    flat_mask[selected] = True

    offset = 0
    for index, (tensor, size) in enumerate(zip(tensors, sizes)):
        masks[index] = flat_mask[offset:offset + size].reshape_as(tensor).to(tensor.device)
        offset += size
    return masks
