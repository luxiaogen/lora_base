import random

import numpy as np
import torch


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
    correct, total = 0, 0
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            network.eval()
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device) - known_classes
                logits = network(inputs)['logits']
                loss_total += float(loss_function(logits, targets).item()) * len(targets)
                correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                total += len(targets)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training
    return {
        'loss': loss_total / total,
        'accuracy': correct * 100.0 / total,
    }
