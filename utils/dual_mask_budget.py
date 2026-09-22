"""Budget-matched mask selection helpers for DualMask experiments."""

from typing import Optional, Sequence

import torch


def select_global_budget_masks(
    scores: Sequence[torch.Tensor],
    reference_masks: Sequence[torch.Tensor],
    valid_masks: Optional[Sequence[torch.Tensor]] = None,
) -> list[torch.Tensor]:
    """Select one global Top-K mask with K matched to local reference masks."""
    if not scores or len(scores) != len(reference_masks):
        raise ValueError("scores and reference_masks must be non-empty and aligned")
    if valid_masks is not None and len(valid_masks) != len(scores):
        raise ValueError("valid_masks must align with scores")

    devices = {score.device for score in scores}
    if len(devices) != 1:
        raise ValueError("all scores must be on the same device")
    for score, reference in zip(scores, reference_masks):
        if score.shape != reference.shape:
            raise ValueError("score and reference mask shapes must match")

    flat_scores = torch.cat([score.detach().float().flatten() for score in scores])
    if valid_masks is None:
        flat_valid = torch.ones_like(flat_scores, dtype=torch.bool)
    else:
        for score, valid in zip(scores, valid_masks):
            if score.shape != valid.shape:
                raise ValueError("score and valid mask shapes must match")
        flat_valid = torch.cat([valid.detach().bool().flatten() for valid in valid_masks])

    requested = sum(int(mask.detach().bool().sum().item()) for mask in reference_masks)
    budget = min(requested, int(flat_valid.sum().item()))
    selected = torch.zeros_like(flat_scores, dtype=torch.bool)
    if budget > 0:
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]
        valid_scores = flat_scores[valid_indices]
        top_indices = torch.topk(valid_scores, budget, largest=True, sorted=False).indices
        selected[valid_indices[top_indices]] = True

    masks = []
    offset = 0
    for score in scores:
        count = score.numel()
        masks.append(
            selected[offset:offset + count]
            .reshape_as(score)
            .to(device=score.device, dtype=score.dtype)
        )
        offset += count
    return masks


def select_projection_budget_masks(
    scores: Sequence[torch.Tensor],
    reference_masks: Sequence[torch.Tensor],
    valid_masks: Optional[Sequence[torch.Tensor]] = None,
) -> list[torch.Tensor]:
    """Select Q/K/V budgets independently while sharing each budget across layers."""
    if not scores or len(scores) != len(reference_masks):
        raise ValueError("scores and reference_masks must be non-empty and aligned")
    if valid_masks is not None and len(valid_masks) != len(scores):
        raise ValueError("valid_masks must align with scores")
    if any(score.ndim < 1 or score.shape[0] % 3 != 0 for score in scores):
        raise ValueError("QKV score rows must be divisible by 3")

    score_parts = [score.chunk(3, dim=0) for score in scores]
    reference_parts = [mask.chunk(3, dim=0) for mask in reference_masks]
    valid_parts = (
        [mask.chunk(3, dim=0) for mask in valid_masks]
        if valid_masks is not None
        else None
    )
    selected_parts = [[] for _ in scores]
    for projection in range(3):
        projection_scores = [parts[projection] for parts in score_parts]
        projection_references = [parts[projection] for parts in reference_parts]
        projection_valid = (
            [parts[projection] for parts in valid_parts]
            if valid_parts is not None
            else None
        )
        projection_masks = select_global_budget_masks(
            projection_scores,
            projection_references,
            projection_valid,
        )
        for layer, mask in enumerate(projection_masks):
            selected_parts[layer].append(mask)

    return [torch.cat(parts, dim=0) for parts in selected_parts]
