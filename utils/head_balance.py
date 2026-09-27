"""Head-only global competition using detached current and old Gaussian features."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def sample_old_feature_pool(means, covariances, task_sizes, current_task, device, generator):
    """Task-local CPU pool; reuse CA's age-scaled means and full covariances."""
    pools = []
    class_id = 0
    for task, size in enumerate(task_sizes[:current_task]):
        for _ in range(size):
            mean = means[class_id].detach().float().to(device)
            mean = mean * (.9 + .1 * (task + 1) / (current_task + 1))
            factor = torch.linalg.cholesky(covariances[class_id].detach().float().to(device))
            # A private CPU stream does not consume training/augmentation/CA RNG.
            noise = torch.randn(256, mean.numel(), generator=generator).to(device)
            pools.append((noise @ factor.T + mean).cpu())
            class_id += 1
    return torch.stack(pools)


def head_balance_loss(new_features, local_targets, old_features, old_weights, new_weights, scale):
    """One old sample per old class; only new_weights receive auxiliary gradients."""
    n_old, n_new = old_weights.shape[0], new_weights.shape[0]
    weights = F.normalize(torch.cat([old_weights.detach(), new_weights]), dim=1)
    new_logits = scale * F.linear(F.normalize(new_features.detach(), dim=1), weights)
    old_logits = scale * F.linear(F.normalize(old_features.detach(), dim=1), weights)
    new_losses = F.cross_entropy(new_logits, local_targets + n_old, reduction='none')
    # Equal weight per represented new class, independent of within-batch imbalance.
    _, inverse, counts = torch.unique(local_targets, return_inverse=True, return_counts=True)
    new_ce = (new_losses / counts[inverse]).sum() / counts.numel()
    old_targets = torch.arange(n_old, device=old_features.device)
    old_ce = F.cross_entropy(old_logits, old_targets)
    loss = (n_new * new_ce + n_old * old_ce) / (n_old + n_new)
    return loss, {
        'head_balance_ce': loss.detach(),
        'head_balance_new_ce': new_ce.detach(),
        'head_balance_old_ce': old_ce.detach(),
        'head_balance_new_to_old': (new_logits.argmax(1) < n_old).float().mean().detach(),
        'head_balance_old_to_new': (old_logits.argmax(1) >= n_old).float().mean().detach(),
    }
