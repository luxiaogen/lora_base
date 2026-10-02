"""Signed local contributions to pretrained linear or joint Q/K relation changes.

These weight-space proxies are not old-task gradients or forgetting guarantees.
Previously merged changes and the fixed S contribution are context; only P coordinates are ranked.
"""
import torch


@torch.no_grad()
def gram_risk(anchor, delta, context, kind):
    current = anchor + context + delta
    if kind == 'input':
        reference = anchor.transpose(-2, -1) @ anchor
        change = current.transpose(-2, -1) @ current - reference
        gradient = 4 * (current @ change)
    else:
        reference = anchor @ anchor.transpose(-2, -1)
        change = current @ current.transpose(-2, -1) - reference
        gradient = 4 * (change @ current)
    normalizer = reference.square().sum((-2, -1)).clamp_min(1e-12)
    return change.square().sum((-2, -1)) / normalizer, gradient / normalizer[:, None, None]


@torch.no_grad()
def qk_risk(anchor, delta, context, heads, bias=None):
    current = anchor + context + delta
    if bias is not None:
        anchor = torch.cat((anchor, bias.unsqueeze(-1)), -1)
        current = torch.cat((current, bias.unsqueeze(-1)), -1)
    width = current.shape[-1]
    reference = anchor.reshape(3, heads, -1, width)
    active = current.reshape_as(reference)
    reference_kernel = reference[0].transpose(-2, -1) @ reference[1]
    change = active[0].transpose(-2, -1) @ active[1] - reference_kernel
    normalizer = reference_kernel.square().sum((-2, -1)).clamp_min(1e-12)
    gradient = torch.zeros_like(active)
    gradient[0] = 2 * (active[1] @ change.transpose(-2, -1)) / normalizer[:, None, None] / heads
    gradient[1] = 2 * (active[0] @ change) / normalizer[:, None, None] / heads
    return (change.square().sum((-2, -1)) / normalizer).mean(), gradient.reshape_as(current)[..., :delta.shape[-1]]


@torch.no_grad()
def functional_score(mode, base, anchor, context, heads, importance=None, bias=None):
    base, anchor, context = base.float(), anchor.float(), context.float()
    if mode == 'wpre_product':
        return importance.float() * base.abs(), None
    if mode in ('wpre_input', 'wpre_output'):
        risk, gradient = gram_risk(anchor, base, context, mode.split('_')[1])
    else:
        risk, gradient = qk_risk(anchor, base, context, heads, bias)
    contribution = base * gradient
    # Decreasing a coordinate with a positive contribution decreases this
    # local proxy to first order. Interacting finite deletions can differ.
    score = contribution.clamp_min(0)
    if mode in ('wpre_qk', 'task_qk'):
        score[2] = base[2].abs()  # V remains the magnitude control.
    return score, dict(risk=risk, positive_fraction=(contribution > 0).float().mean((-2, -1)))
