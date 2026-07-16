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

""" 只使用当前任务的训练数据，估计原始预训练模型 W_pre 对当前任务的分类能力，得到一个 0～1 的 competence 分数  """
def split_prototype_competence(
    features: torch.Tensor, # 训练集所有特征
    targets: torch.Tensor,
    indices: torch.Tensor,
    holdout_mod: int = 5,
): # 检查:W_pre 提取出的特征是否已经按类别自然聚集
    holdout_mod = max(2, int(holdout_mod)) # 5:第 0、5、10、15... 个样本进入 holdout 20%
    calibration = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
    # 将每个类别的样本按索引排序，然后每隔 holdout_mod 个样本选一个进入 holdout
    for class_id in torch.unique(targets, sorted=True):
        positions = (targets == class_id).nonzero(as_tuple=True)[0]
        order = torch.argsort(indices[positions])
        positions = positions[order]
        if positions.numel() >= 2:
            calibration[positions[::holdout_mod]] = True

    prototype_mask = ~calibration  # 取反
    if calibration.any() and prototype_mask.any():    #
        train_prototypes, train_class_ids = build_prototypes(
            features[prototype_mask], targets[prototype_mask]
        ) # 使用约 80% 样本建立类别原型  949*0.8=759   -- 749
        competence = prototype_accuracy( # 表示 W_pre 在当前任务训练集内部 holdout 上原型分类准确率
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