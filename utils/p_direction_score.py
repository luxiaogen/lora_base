"""P-only coordinate ranking in the immutable pretrained singular basis."""
import torch


@torch.no_grad()
def spectral_score(base, left, singular, right, signed=False):
    """Map weighted spectral energy back via squared basis loadings, not reconstruction."""
    projected = left.transpose(-2, -1) @ base.float() @ right
    energy = singular.square()
    energy = energy / energy.sum(-1, keepdim=True).clamp_min(1e-12)
    weight = (energy.unsqueeze(-1) + energy.unsqueeze(-2)) * .5
    risk = projected.square() * weight
    diagonal = projected.diagonal(dim1=-2, dim2=-1)
    positive = diagonal.clamp_min(0).square() * energy
    negative = diagonal.clamp_max(0).square() * energy
    total = risk.sum((-2, -1)).clamp_min(1e-12)
    fractions = torch.stack((positive.sum(-1), negative.sum(-1),
                            risk.sum((-2, -1)) - positive.sum(-1) - negative.sum(-1)), -1) / total.unsqueeze(-1)
    if signed:
        risk.diagonal(dim1=-2, dim2=-1).copy_(negative)
    score = left.square() @ risk @ right.square().transpose(-2, -1)
    return score, fractions, projected


@torch.no_grad()
def norm_matched_gate(base, score, reference, strength, plastic):
    """Match reference removed norm per Q/K/V; extend ranking only if beta would exceed 1."""
    gate = torch.ones_like(base)
    selected = torch.zeros_like(plastic, dtype=torch.bool)
    strengths = []
    for index in range(base.shape[0]):
        value = base[index].float().flatten()
        valid = plastic[index].bool().flatten().nonzero(as_tuple=True)[0]
        k = int(reference[index].bool().sum())
        target = strength * (base[index].float() * reference[index]).norm()
        alpha = value.new_zeros(())
        if k and target > 0:
            order = score[index].flatten()[valid].topk(k, sorted=False).indices
            chosen = valid[order]
            available = value[chosen].norm()
            if available < target:
                order = score[index].flatten()[valid].argsort(descending=True)
                cumulative = value[valid[order]].square().cumsum(0)
                extra_k = min(valid.numel(), int(torch.searchsorted(cumulative, target.square())) + 1)
                chosen = valid[order[:max(k, extra_k)]]
                available = value[chosen].norm()
            alpha = (target / available.clamp_min(1e-12)).clamp(0, 1)
            selected[index].view(-1)[chosen] = True
            gate[index].view(-1)[chosen] = (1 - alpha).to(gate)
        strengths.append(alpha)
    return gate, selected, torch.stack(strengths)
