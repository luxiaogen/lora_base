"""Diagonal, identity-regularized transport of existing CA statistics."""
import torch


@torch.no_grad()
def fit_diagonal_transport(before, after):
    x, y = before.double(), after.double()
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    variance = xc.square().sum(0)
    # Ridge towards scale=1, with a data-scaled fixed strength (no grid search).
    ridge = variance.mean().clamp_min(1e-12)
    scale = ((xc * yc).sum(0) + ridge) / (variance + ridge)
    scale = scale.clamp(.5, 1.5)
    return scale, my - scale * mx


@torch.no_grad()
def transport_statistics(means, covariances, scale, offset, mean_only=False):
    scale, offset = scale.to(means), offset.to(means)
    mapped_means = means * scale + offset
    if mean_only:
        return mapped_means, covariances
    mapped_covariances = covariances * scale[None, :, None]
    mapped_covariances.mul_(scale[None, None, :])
    return mapped_means, mapped_covariances
