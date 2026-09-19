"""Fixed-policy DualMask port. No task labels, replay, or trainable router.

This port deliberately leaves host ranks, classifiers and objectives alone.
It does not implement LoDA's competence/rank controller or anchor loss.
"""
import torch
from torch import nn


def normalize(score):
    score = score.float() - score.float().min()
    return score / score.max().clamp_min(1e-12)


def select(score, coverage, floor=0.0):
    flat = score.detach().float().flatten().clamp_min(0)
    if flat.sum() <= 0:
        return torch.zeros_like(score)
    values = flat.sort(descending=True).values
    count = int(torch.searchsorted(values.cumsum(0), coverage * values.sum()).item()) + 1
    count = min(len(values), max(count, int(len(values) * floor), 1))
    return (score >= values[count - 1]).to(score.dtype)


class DualMask(nn.Module):
    """Immutable spectral importance and complementary masks per projection."""
    def __init__(self, pretrained, role):
        super().__init__()
        if role not in {"shared", "private", "single"}:
            raise ValueError(role)
        self.role = role
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(pretrained.detach().float(), full_matrices=False)
            k = min(32, len(s))
            rows = (u[:, :k].square() * s[:k]).sum(1)
            cols = (vh[:k].T.square() * s[:k]).sum(1)
            importance = normalize(rows[:, None] * cols[None, :])
        self.register_buffer("importance", importance)
        self.register_buffer("protect", select(importance, 0.5))

    def forward(self, delta):
        conflict = select(normalize(self.importance * normalize(delta.detach().abs())), 0.5, 0.1)
        # single = protected part + complementary plastic part, with tied factors.
        gate = 1 - self.protect if self.role == "private" else 1 - 0.5 * self.protect
        return delta * gate.to(delta) * (1 - 0.5 * conflict.to(delta))
