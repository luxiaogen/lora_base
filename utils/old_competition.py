"""Current-training-image hinge against frozen old cosine classifier weights."""
from torch.nn import functional as F


def old_competition_loss(features, current_logits, local_targets, old_weights, scale, detach_old=False):
    old_logits = F.linear(F.normalize(features, dim=1),
                          F.normalize(old_weights.detach(), dim=1))
    positive = current_logits.gather(1, local_targets[:, None]).squeeze(1)
    old_reference = old_logits.max(1).values
    if detach_old:
        old_reference = old_reference.detach()
    violation = old_reference - positive
    return scale * violation.relu().mean(), (violation > 0).float().mean().detach()
