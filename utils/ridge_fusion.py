"""Legal score fusion: current-train calibration, label-free all-seen prediction."""
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch

from utils.frozen_readout import FrozenReadout


def fuse_scores(base, ridge, coefficient):
    if coefficient == 0:
        return base
    standardize = lambda scores: (scores - scores.mean(1, keepdim=True)) / scores.std(
        1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return standardize(base) + coefficient * standardize(ridge)


def select_coefficient(base, ridge, labels):
    """One predeclared rule, fitted only on withheld CURRENT training images."""
    prediction = lambda scores: scores.topk(1, dim=1).indices.squeeze(1)
    base_correct = prediction(base).eq(labels)
    rows = []
    for coefficient in (0., .025, .05, .1, .2):
        correct = prediction(fuse_scores(base, ridge, coefficient)).eq(labels)
        rescued = int((~base_correct & correct).sum())
        harmed = int((base_correct & ~correct).sum())
        rows.append(dict(coefficient=coefficient, rescued=rescued, harmed=harmed,
                         utility=rescued - 3 * harmed))
    # First-on-tie, including unchanged base, is fixed before test evaluation.
    return max(rows, key=lambda row: row['utility'])['coefficient'], rows


@torch.no_grad()
def collect_calibration(network, loader, device, anchor_context, weight):
    modes = [(module, module.training) for module in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    parts = []
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                base = network.interface(inputs)
                with anchor_context():
                    ridge = network.extract_vector(inputs) @ weight
                parts.append((base.cpu(), ridge.cpu(), targets.cpu()))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for module, training in modes:
            module.training = training
    return tuple(torch.cat([part[i] for part in parts]) for i in range(3))


class RidgeFusion(FrozenReadout):
    def __init__(self, dim, classes, class_size, tasks):
        super().__init__(dim, classes)
        self.class_size = class_size
        self.coefficient = 0.
        self.base_matrix = np.zeros((tasks, tasks))
        self.candidate_matrix = np.zeros((tasks, tasks))
        self.base_curve, self.candidate_curve = [], []

    def seal(self, task, directory, rows, calibration_n, regularizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        policy = dict(task=task, coefficient=self.coefficient, candidates=rows,
                      alpha=1., regularizer=regularizer, calibration_n=calibration_n,
                      selection_source='current_train_holdout_only' if task else 'unchanged_task0',
                      selection_objective='rescued - 3 * harmed; first-on-tie retains base',
                      test_labels_used_for_selection=False, true_task_id_used_for_prediction=False,
                      old_images_revisited=False, cumulative_readout_fit=self.samples,
                      statistics_bytes=self.storage_bytes)
        (directory / f'task_{task:02d}_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
        logging.info('RidgeFusionPolicy %s', json.dumps(policy))

    def report(self, task, base, candidate, targets, directory):
        known_classes = task * self.class_size
        group = lambda predictions: {name: float((predictions[mask] == targets[mask]).mean() * 100)
                                     if mask.any() else None for name, mask in dict(
                                         total=np.ones(len(targets), dtype=bool),
                                         old=targets < known_classes,
                                         new=targets >= known_classes).items()}
        base_groups, candidate_groups = group(base), group(candidate)
        self.base_curve.append(base_groups['total'])
        self.candidate_curve.append(candidate_groups['total'])
        for seen_task in range(task + 1):
            mask = targets // self.class_size == seen_task
            self.base_matrix[seen_task, task] = float((base[mask] == targets[mask]).mean() * 100)
            self.candidate_matrix[seen_task, task] = float((candidate[mask] == targets[mask]).mean() * 100)
        forgetting = lambda matrix: float((matrix[:task, :task + 1].max(1)
                                          - matrix[:task, task]).mean()) if task else None
        payload = dict(task=task, coefficient=self.coefficient, base=base_groups,
                       candidate=candidate_groups, base_average=float(np.mean(self.base_curve)),
                       candidate_average=float(np.mean(self.candidate_curve)),
                       base_forgetting=forgetting(self.base_matrix),
                       candidate_forgetting=forgetting(self.candidate_matrix),
                       rescued=int(((base != targets) & (candidate == targets)).sum()),
                       harmed=int(((base == targets) & (candidate != targets)).sum()),
                       switched=int((base != candidate).sum()),
                       inference_uses_labels=False, inference_uses_task_id=False)
        (Path(directory) / f'task_{task:02d}_result.json').write_text(json.dumps(payload, indent=2) + '\n')
        logging.info('RidgeFusionResult %s', json.dumps(payload))
        return payload
