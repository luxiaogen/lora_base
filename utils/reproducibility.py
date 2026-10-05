"""Keep A0's main-process RNG sequence after removing diagnostic forwards."""
import torch


def advance_loader_rng():
    # DataLoader(shuffle=False, generator=None) draws exactly one CPU base seed.
    torch.empty((), dtype=torch.int64).random_()
