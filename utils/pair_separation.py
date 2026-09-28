"""Current-task sample relations; no old data, extra forward, or RNG calls."""
import logging

import torch
from torch.nn import functional as F


def pair_separation_loss(features, targets, margin=.1):
    features = F.normalize(features.float(), dim=1)
    similarities = features @ features.t()
    same = targets[:, None].eq(targets[None, :])
    positive = same & ~torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    negative = ~same
    positive_count = positive.sum(dim=1)
    valid = (positive_count > 0) & negative.any(dim=1)
    positive_mean = (similarities * positive).sum(dim=1) / positive_count.clamp_min(1)
    # -2 is below the cosine range, including rows with no negatives.
    hardest_negative = similarities.masked_fill(~negative, -2.).max(dim=1).values
    violation = F.relu(hardest_negative - positive_mean + margin)
    denominator = valid.sum().clamp_min(1)
    loss = (violation * valid).sum() / denominator
    return loss, {
        'pair_raw': loss.detach(),
        'pair_valid_ratio': valid.float().mean().detach(),
        'pair_active_ratio': ((violation > 0) & valid).sum().float().detach() / denominator,
    }


def log_pair_gradients(task_loss, named_params, pair_grads, task, epoch, batch, scope):
    """Sampled read-only comparison, retaining the graph for ordinary backward."""
    task_grads = torch.autograd.grad(task_loss, [p for _, p in named_params],
                                     retain_graph=True, allow_unused=True)
    for branch in ('S', 'P'):
        task_sq, pair_sq, dot = (task_loss.new_zeros(()) for _ in range(3))
        count = 0
        for (name, _), main, pair in zip(named_params, task_grads, pair_grads):
            if name != branch:
                continue
            count += 1
            if main is not None:
                task_sq += main.detach().float().square().sum()
            if pair is not None:
                pair_sq += pair.detach().float().square().sum()
            if main is not None and pair is not None:
                dot += (main.detach().float() * pair.detach().float()).sum()
        if count:
            main_norm, pair_norm = float(task_sq.sqrt()), float(pair_sq.sqrt())
            logging.info('PairSeparationGrad %s', {
                'task': task, 'epoch': epoch + 1, 'batch': batch + 1,
                'scope': scope, 'branch': branch, 'task_grad_norm': main_norm,
                'weighted_pair_grad_norm': pair_norm,
                'pair_to_task_ratio': pair_norm / main_norm if main_norm > 0 else None,
                'cosine': float(dot) / (main_norm * pair_norm)
                if main_norm > 0 and pair_norm > 0 else None,
            })
