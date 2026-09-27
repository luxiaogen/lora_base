"""Bounded, class-normalized emphasis on near-tie classifier examples."""
import copy
from contextlib import contextmanager

import torch


@torch.no_grad()
def boundary_weights(logits, targets):
    positive = logits.gather(1, targets[:, None]).squeeze(1)
    other = logits.scatter(1, targets[:, None], float('-inf')).max(1).values
    # Both sides of a boundary receive emphasis; distant wrong outliers do not.
    weights = 1 + torch.exp(-(positive - other).abs() / 0.1)
    for label in targets.unique():
        selected = targets == label
        weights[selected] /= weights[selected].mean()
    return weights


@contextmanager
def temporary_classifier(network):
    """Undo a CA-only fork including RNG, flags and classifier gradients."""
    pool = network.classifier_pool
    state = copy.deepcopy(pool.state_dict())
    flags = [(p, p.requires_grad, None if p.grad is None else p.grad.clone()) for p in pool.parameters()]
    modes = [(m, m.training) for m in network.modules()]
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        pool.load_state_dict(state)
        for p, requires_grad, grad in flags:
            p.requires_grad_(requires_grad)
            p.grad = grad
        for module, training in modes:
            module.training = training
