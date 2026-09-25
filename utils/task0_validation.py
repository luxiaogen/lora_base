import random

import numpy as np
import torch
from torch.nn import functional as F


def evaluate_task0_holdout(
    network,
    loader,
    loss_function,
    device,
    cuda_devices,
    known_classes,
):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in network.modules()]
    loss_total = 0.0
    class_margin_total = 0.0
    correct, total = 0, 0
    feature_blocks, target_blocks = [], []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            network.eval()
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device) - known_classes
                output = network(inputs)
                logits = output['logits']
                loss_total += float(loss_function(logits, targets).item()) * len(targets)
                positive = logits.gather(1, targets[:, None]).squeeze(1)
                negative = logits.scatter(1, targets[:, None], float('-inf')).max(dim=1).values
                class_margin_total += float((positive - negative).sum().item())
                correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                total += len(targets)
                if 'features' in output:
                    feature_blocks.append(F.normalize(output['features'].float(), dim=1).cpu())
                    target_blocks.append(targets.cpu())
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training
    metrics = {
        'loss': loss_total / total,
        'accuracy': correct * 100.0 / total,
        'class_margin': class_margin_total / total,
    }
    if feature_blocks:
        features = torch.cat(feature_blocks)
        labels = torch.cat(target_blocks)
        classes = labels.unique(sorted=True)
        centroids = torch.stack([features[labels == label].mean(dim=0) for label in classes])
        residuals = torch.cat([
            features[labels == label] - centroid
            for label, centroid in zip(classes, centroids)
        ])
        metrics['class_variance'] = float(residuals.square().sum(dim=1).mean())
        if len(classes) > 1:
            normalized_centroids = F.normalize(centroids, dim=1)
            similarities = normalized_centroids @ normalized_centroids.t()
            similarities.fill_diagonal_(float('-inf'))
            metrics['min_centroid_margin'] = float(1.0 - similarities.max())
    return metrics
