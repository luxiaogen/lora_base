"""Fixed W_pre readouts: additive current-task statistics, no image replay."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class FrozenReadout:
    def __init__(self, dim, classes):
        self.gram = torch.zeros(dim, dim)
        self.class_sum = torch.zeros(dim, classes)
        self.samples = 0

    @property
    def storage_bytes(self):
        return (self.gram.numel() + self.class_sum.numel()) * self.gram.element_size()

    @torch.no_grad()
    def update(self, features, labels):
        x, y = features.cpu().float(), labels.cpu().long()
        self.gram.add_(x.T @ x)
        self.class_sum.index_add_(1, y, x.T)
        self.samples += len(y)

    def ncm_scores(self, features, seen_classes):
        # Normalizing the raw sum is equivalent to normalizing the raw mean.
        return F.normalize(features, dim=1) @ F.normalize(self.class_sum[:, :seen_classes], dim=0)

    def ridge_weight(self, alpha, seen_classes):
        dim = len(self.gram)
        regularizer = float(alpha) * float(self.gram.trace()) / dim
        matrix = self.gram.double() + regularizer * torch.eye(dim, dtype=torch.float64)
        weight = torch.linalg.solve(matrix, self.class_sum[:, :seen_classes].double())
        return weight.float(), regularizer


def select_alpha(stats, holdout_features, holdout_labels, seen_classes):
    """Called once on Task0 train holdout, before opening any test images."""
    rows = []
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
        weight, regularizer = stats.ridge_weight(alpha, seen_classes)
        prediction = (holdout_features @ weight).argmax(1)
        rows.append(dict(alpha=alpha, regularizer=regularizer,
                         accuracy=100. * float(prediction.eq(holdout_labels).double().mean())))
    return max(rows, key=lambda row: row['accuracy'])['alpha'], rows


def training_partition(targets, old_classes, seen_classes, outer_mod, inner_mod):
    """Split alpha-selection data; the final readouts refit on both returned subsets."""
    fit, holdout = [], []
    for label in range(old_classes, seen_classes):
        positions = np.flatnonzero(targets == label)
        if old_classes and outer_mod:
            positions = positions[np.arange(len(positions)) % outer_mod != 0]
        calibration = np.arange(len(positions)) % inner_mod == 0
        if len(positions) < 2:
            calibration[:] = False
        fit.extend(positions[~calibration])
        holdout.extend(positions[calibration])
    return np.asarray(fit, dtype=int), np.asarray(holdout, dtype=int)


def load_reference(path, targets):
    """Pair cached predictions only when every sample index and label matches."""
    path = Path(path)
    if not path.is_file():
        return None, None, 'cache_missing'
    with path.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['index']))
    indices = [int(row['index']) for row in rows]
    labels = [int(row['target']) for row in rows]
    if indices != list(range(len(targets))) or labels != targets.tolist():
        return None, None, 'sample_alignment_mismatch'
    base = torch.tensor([int(row['base_prediction']) for row in rows])
    anchor = torch.tensor([int(row['anchor_prediction']) for row in rows])
    return base, anchor, 'matched_indices_and_labels'


def readout_report(base_prediction, readout_prediction, targets, old_classes):
    """True labels report accuracy/union only; they never choose a prediction."""
    base = base_prediction.eq(targets)
    readout = readout_prediction.eq(targets)
    result = {}
    for name, mask in dict(total=torch.ones_like(base), old=targets < old_classes,
                           new=targets >= old_classes).items():
        n = int(mask.sum())
        accuracy = lambda correct: 100. * int((correct & mask).sum()) / n if n else None
        result[name] = dict(n=n, base_accuracy=accuracy(base), readout_accuracy=accuracy(readout),
                            oracle_accuracy=accuracy(base | readout),
                            rescued=int((~base & readout & mask).sum()),
                            harmed=int((base & ~readout & mask).sum()),
                            both_wrong=int((~base & ~readout & mask).sum()))
    return result
