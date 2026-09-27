"""Current-task classifier preparation, without changing the feature extractor."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from models.losses import AngularPenaltySMLoss


def collect_features(network, loader, device, task, known_classes):
    modes = [(m, m.training) for m in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    generator_state = loader.generator.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    features, labels = [], []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            for _, inputs, targets in loader:
                features.append(network.extract_vector(inputs.to(device), task_id=task).detach())
                labels.append(targets.to(device) - known_classes)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        loader.generator.set_state(generator_state)
        for module, training in modes:
            module.training = training
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def prototype_init(head, features, labels):
    normalized = F.normalize(features, dim=1)
    prototypes = torch.stack([normalized[labels == c].mean(0)
                              for c in range(head.weight.shape[0])])
    # Preserve the original cosine-head parameter scale (and effective SGD scale).
    head.weight.copy_(F.normalize(prototypes, dim=1) * head.weight.norm(dim=1, keepdim=True))


def fit_head(head, features, labels, epochs, batch_size, lr, weight_decay,
             scale, margin, seed):
    features = F.normalize(features.detach(), dim=1)
    was_trainable = head.weight.requires_grad
    head.weight.requires_grad_(True)
    optimizer = torch.optim.SGD([head.weight], lr=lr, momentum=.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AngularPenaltySMLoss(loss_type='cosface', s=scale, m=margin)
    generator = torch.Generator().manual_seed(seed)
    steps = 0
    try:
        for _ in range(epochs):
            order = torch.randperm(len(labels), generator=generator).to(labels.device)
            for indices in order.split(batch_size):
                logits = F.linear(features[indices], F.normalize(head.weight, dim=1))
                loss = criterion(logits, labels[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
            scheduler.step()
    finally:
        head.weight.grad = None
        head.weight.requires_grad_(was_trainable)
    # Optimizer/momentum are local; normal LoRA training starts with a fresh optimizer.
    return steps
