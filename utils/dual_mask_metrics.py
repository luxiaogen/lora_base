from typing import Optional
import torch
import torch.nn.functional as F

def build_prototypes(features: torch.Tensor, targets: torch.Tensor):
    class_ids = torch.unique(targets, sorted=True)
    prototypes = []
    for class_id in class_ids:
        class_features = features[targets == class_id]
        prototypes.append(F.normalize(class_features.mean(dim=0), dim=0))
    return (torch.stack(prototypes), class_ids)

def prototype_accuracy(features: torch.Tensor, targets: torch.Tensor, prototypes: torch.Tensor, class_ids: torch.Tensor) -> float:
    logits = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    predictions = class_ids.to(logits.device)[logits.argmax(dim=1)]
    return float((predictions == targets).float().mean().item())

def _prototype_holdout_mask(targets: torch.Tensor, indices: torch.Tensor, holdout_mod: int) -> torch.Tensor:
    holdout_mod = max(2, int(holdout_mod))
    calibration = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
    for class_id in torch.unique(targets, sorted=True):
        positions = (targets == class_id).nonzero(as_tuple=True)[0]
        order = torch.argsort(indices[positions])
        positions = positions[order]
        if positions.numel() >= 2:
            calibration[positions[::holdout_mod]] = True
    return calibration
def split_prototype_competence(features: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor, holdout_mod: int=5, old_prototypes: Optional[torch.Tensor]=None, old_class_ids: Optional[torch.Tensor]=None):
    """Estimate W_pre competence from current samples and optional old prototypes."""
    calibration = _prototype_holdout_mask(targets, indices, holdout_mod)
    prototype_mask = ~calibration
    if calibration.any() and prototype_mask.any():
        train_prototypes, train_class_ids = build_prototypes(features[prototype_mask], targets[prototype_mask])
        if old_prototypes is not None and old_class_ids is not None:
            train_prototypes = torch.cat([old_prototypes.to(features), train_prototypes], dim=0)
            train_class_ids = torch.cat([old_class_ids.to(targets), train_class_ids], dim=0)
        competence = prototype_accuracy(features[calibration], targets[calibration], train_prototypes, train_class_ids)
    else:
        competence = 0.0
    full_prototypes, full_class_ids = build_prototypes(features, targets)
    return (competence, full_prototypes, full_class_ids)
