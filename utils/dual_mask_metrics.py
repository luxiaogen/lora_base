from typing import Optional

import math
import torch
import torch.nn.functional as F


def build_prototypes(features: torch.Tensor, targets: torch.Tensor):
    class_ids = torch.unique(targets, sorted=True) # 类别id
    prototypes = []
    for class_id in class_ids:
        class_features = features[targets == class_id]
        prototypes.append(F.normalize(class_features.mean(dim=0), dim=0))
    return torch.stack(prototypes), class_ids


def prototype_accuracy(
    features: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    class_ids: torch.Tensor,
) -> float: # 20%的样本数  [197,768] @ [768,20]
    logits = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    predictions = class_ids.to(logits.device)[logits.argmax(dim=1)]
    return float((predictions == targets).float().mean().item())


def prototype_margin_competence(features: torch.Tensor,targets: torch.Tensor,prototypes: torch.Tensor,class_ids: torch.Tensor,) -> float:
    """Return a continuous NCM competence from correct-vs-best-wrong margin."""
    logits = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    class_ids = class_ids.to(logits.device)
    correct = targets.to(logits.device).unsqueeze(1).eq(class_ids.unsqueeze(0))
    if logits.shape[1] <= 1:
        return 1.0
    positive = logits.masked_fill(~correct, float("-inf")).max(dim=1).values
    negative = logits.masked_fill(correct, float("-inf")).max(dim=1).values
    # Cosine margins lie in [-2, 2]; map that fixed range to [0, 1].
    return float(((positive - negative + 2.0) / 4.0).clamp(0.0, 1.0).mean().item())


def _prototype_holdout_mask(
    targets: torch.Tensor,
    indices: torch.Tensor,
    holdout_mod: int,) -> torch.Tensor:
    holdout_mod = max(2, int(holdout_mod))
    calibration = torch.zeros(targets.shape[0],dtype=torch.bool,device=targets.device,)
    for class_id in torch.unique(targets, sorted=True):
        positions = (targets == class_id).nonzero(as_tuple=True)[0]
        order = torch.argsort(indices[positions])
        positions = positions[order]
        if positions.numel() >= 2:
            calibration[positions[::holdout_mod]] = True
    return calibration

def split_prototype_ncm_diagnostics(
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    holdout_mod: int = 5,
    scale: float = 1.0,
):
    """Measure new-task NCM loss without gradients or old-class candidates."""
    calibration = _prototype_holdout_mask(targets, indices, holdout_mod)
    prototype_mask = ~calibration
    if not calibration.any() or not prototype_mask.any():
        return 0.0, 0.0

    prototypes, class_ids = build_prototypes(
        features[prototype_mask],
        targets[prototype_mask],
    )
    with torch.no_grad():
        logits = (
            float(scale)
            * F.normalize(features[calibration], dim=1)
            @ F.normalize(prototypes, dim=1).T
        )
        local_targets = torch.searchsorted(
            class_ids.to(targets.device),
            targets[calibration],
        ).to(logits.device)
        ncm_loss = float(F.cross_entropy(logits, local_targets).item())

    num_classes = int(class_ids.numel())
    if num_classes <= 1:
        plasticity_demand = 0.0
    else:
        plasticity_demand = min(
            max(ncm_loss / math.log(num_classes), 0.0),
            1.0,
        )
    return ncm_loss, plasticity_demand

def functional_merge_diagnostics(
    anchor_features: torch.Tensor,
    candidate_features: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    holdout_mod: int = 5,
    scale: float = 1.0,
    old_prototypes: Optional[torch.Tensor] = None,
    old_class_ids: Optional[torch.Tensor] = None,
):
    """Evaluate one merge candidate on a train-only deterministic holdout.

    Current-task utility uses prototypes built from the candidate features.
    Anchor damage uses fixed W0 prototypes, optionally including cached old
    classes, so the candidate cannot move the reference decision geometry.
    """
    if anchor_features.shape != candidate_features.shape:
        raise ValueError("anchor and candidate features must have the same shape")
  

    calibration = _prototype_holdout_mask(targets, indices, holdout_mod)
    prototype_mask = ~calibration
    if not calibration.any() or not prototype_mask.any():
        return {
            "current_accuracy": 0.0,
            "current_loss": 0.0,
            "anchor_reference_loss": 0.0,
            "anchor_candidate_loss": 0.0,
            "anchor_damage": 0.0,
        }

    candidate_prototypes, candidate_class_ids = build_prototypes(
        candidate_features[prototype_mask],
        targets[prototype_mask],
    )
    anchor_prototypes, anchor_class_ids = build_prototypes(
        anchor_features[prototype_mask],
        targets[prototype_mask],
    )
    if old_prototypes is not None and old_class_ids is not None:
        anchor_prototypes = torch.cat(
            [old_prototypes.to(anchor_features), anchor_prototypes], dim=0
        )
        anchor_class_ids = torch.cat(
            [old_class_ids.to(targets), anchor_class_ids], dim=0
        )

    def ncm_metrics(features, prototypes, class_ids):
        logits = (
            float(scale)
            * F.normalize(features[calibration], dim=1)
            @ F.normalize(prototypes, dim=1).T
        )

        class_ids = class_ids.to(targets.device)
        labels = targets[calibration]
        matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
        if not matches.any(dim=1).all():
            raise ValueError("holdout target is missing from prototype candidates")
        local_targets = matches.float().argmax(dim=1).to(logits.device)
        accuracy = float(
            (class_ids.to(logits.device)[logits.argmax(dim=1)] == labels.to(logits.device))
            .float()
            .mean()
            .item()
        )
        loss = float(F.cross_entropy(logits, local_targets).item())
        return accuracy, loss
       

    current_accuracy, current_loss = ncm_metrics(
        candidate_features,
        candidate_prototypes,
        candidate_class_ids,
    )
    _, anchor_reference_loss = ncm_metrics(
        anchor_features,
        anchor_prototypes,
        anchor_class_ids,
    )
    _, anchor_candidate_loss = ncm_metrics(
        candidate_features,
        anchor_prototypes,
        anchor_class_ids,
    )


    return {
        "current_accuracy": current_accuracy,
        "current_loss": current_loss,
        "anchor_reference_loss": anchor_reference_loss,
        "anchor_candidate_loss": anchor_candidate_loss,
        "anchor_damage": max(0.0, anchor_candidate_loss - anchor_reference_loss),
    }


def select_functional_merge_candidate(candidates, tolerance: float):
    eligible = [
        item for item in candidates
        if item["anchor_damage"] <= float(tolerance)
    ]
    if eligible:
        # Prefer current-task utility; a stronger safe beta wins exact ties.
        return min(
            eligible,
            key=lambda item: (
                item["current_loss"],
                -item["current_accuracy"],
                -item["beta"],
            ),
        )
    return min(candidates, key=lambda item: item["anchor_damage"])


def split_prototype_competence(
    features: torch.Tensor, # 训练集所有特征
    targets: torch.Tensor,
    indices: torch.Tensor,
    holdout_mod: int = 5,
    old_prototypes: Optional[torch.Tensor] = None,
    old_class_ids: Optional[torch.Tensor] = None,
    metric: str = "accuracy",
): # 检查:W_pre 提取出的特征是否已经按类别自然聚集
    """Estimate W_pre competence from current samples and optional old prototypes."""
    calibration = _prototype_holdout_mask(targets, indices, holdout_mod)

    prototype_mask = ~calibration  # 取反
    if calibration.any() and prototype_mask.any():
        # 使用约 80% 样本建立类别原型  949*0.8=759   -- 749    #
        train_prototypes, train_class_ids = build_prototypes(
            features[prototype_mask], targets[prototype_mask]
        ) 
        if old_prototypes is not None and old_class_ids is not None:
            train_prototypes = torch.cat([old_prototypes.to(features), train_prototypes], dim=0)
            train_class_ids = torch.cat([old_class_ids.to(targets), train_class_ids], dim=0)
        
        if metric == "accuracy":
            competence = prototype_accuracy(  # 表示 W_pre 在当前任务训练集内部 holdout 上原型分类准确率
                features[calibration],
                targets[calibration],
                train_prototypes,
                train_class_ids,
            )
        elif metric == "margin":
            competence = prototype_margin_competence(
                features[calibration],
                targets[calibration],
                train_prototypes,
                train_class_ids,
            )
    else:
        competence = 0.0

    # competence高 --> 类内特征集中 + 类间特征分离 + W_pre 本身已经适合当前任务  用 100%，保存下来用于后续 W_pre-only NCM 测试
    full_prototypes, full_class_ids = build_prototypes(features, targets)

    return competence, full_prototypes, full_class_ids
