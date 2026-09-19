"""Matched extra attenuation of safe P updates, shared by training and merge."""
import torch


def region_budget(safe_delta, conflict_mask, amount):
    # Detach selection/scales: gradients still flow through the gated safe delta.
    with torch.no_grad():
        delta, mask = safe_delta.detach().float(), conflict_mask.detach().float()
        c_norm = (delta * mask).norm()
        u_norm = (delta * (1 - mask)).norm()
        target = float(amount) * torch.minimum(c_norm, u_norm)
        tiny = torch.finfo(delta.dtype).tiny
        c_scale = (target / c_norm.clamp_min(tiny)).clamp(0, float(amount))
        u_scale = (target / u_norm.clamp_min(tiny)).clamp(0, float(amount))
    return c_norm, u_norm, target, c_scale, u_scale


def shrink_private_region(safe_delta, conflict_mask, mode, amount):
    if mode == "none" or amount == 0:
        return safe_delta
    _, _, _, c_scale, u_scale = region_budget(safe_delta, conflict_mask, amount)
    mask = conflict_mask.to(safe_delta)
    attenuation = c_scale * mask if mode == "conflict" else u_scale * (1 - mask)
    return safe_delta * (1 - attenuation.to(safe_delta))
