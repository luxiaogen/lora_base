"""Read-only, all-seen-class test reporting; never used to select updates."""
import hashlib
import json
import logging
import random

import numpy as np
import torch
from torch.nn import functional as F


def collect_logits(network, loader, device):
    modes = [(m, m.training) for m in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    generator = getattr(loader, 'generator', None)
    generator_state = generator.get_state() if generator is not None else None
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    outputs, labels, indices = [], [], []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            for ids, inputs, targets in loader:
                outputs.append(network.interface(inputs.to(device)).detach().float().cpu())
                labels.append(targets.cpu())
                indices.append(torch.as_tensor(ids).cpu())
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if generator_state is not None:
            generator.set_state(generator_state)
        for module, training in modes:
            module.training = training
    return torch.cat(outputs), torch.cat(labels), torch.cat(indices)


def stage_metrics(logits, targets, known_classes, previous=None):
    predictions = logits.argmax(1)
    correct = predictions.eq(targets)
    positive = logits.gather(1, targets[:, None]).squeeze(1)
    margin = positive - logits.scatter(1, targets[:, None], float('-inf')).max(1).values
    # Raw cosine-logit CE, not the scaled CosFace training objective.
    ce = F.cross_entropy(logits, targets, reduction='none')
    groups = {'total': torch.ones_like(correct), 'old': targets < known_classes,
              'new': targets >= known_classes}
    cross = (targets < known_classes) != (predictions < known_classes)
    partition_predictions = predictions.clone()
    if known_classes > 0:
        old_samples = targets < known_classes
        partition_predictions[old_samples] = logits[old_samples, :known_classes].argmax(1)
        partition_predictions[~old_samples] = logits[~old_samples, known_classes:].argmax(1) + known_classes
    result = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        values = dict(n=n, accuracy=float(correct[mask].float().mean() * 100) if n else None,
                      margin=float(margin[mask].mean()) if n else None,
                      cosine_ce=float(ce[mask].mean()) if n else None)
        values.update(cross_partition_errors=int((cross & mask).sum()),
                      within_partition_errors=int((~correct & ~cross & mask).sum()),
                      partition_oracle_accuracy=float(partition_predictions[mask].eq(targets[mask]).float().mean()*100) if n else None,
                      partition_oracle_recovered=int((~correct & partition_predictions.eq(targets) & mask).sum()))
        if previous is not None:
            before = previous.argmax(1).eq(targets)
            values.update(corrected=int((~before & correct & mask).sum()),
                          broken=int((before & ~correct & mask).sum()),
                          prediction_changes=int((previous.argmax(1).ne(predictions) & mask).sum()),
                          logit_max_abs_change=float((logits[mask] - previous[mask]).abs().max()) if n else None)
        result[name] = values
    return result


class StageAudit:
    def __init__(self, task, known_classes):
        self.task, self.known_classes = task, known_classes
        self.previous = None
        self.previous_stage = None

    def record(self, stage, network, loader, device):
        logits, targets, indices = collect_logits(network, loader, device)
        digest = hashlib.sha256(torch.stack((indices, targets)).numpy().tobytes()).hexdigest()
        row = dict(task=self.task, stage=stage, previous_stage=self.previous_stage,
                   source='test_report_only', class_count=logits.shape[1], sample_sha256=digest,
                   metrics=stage_metrics(logits, targets, self.known_classes, self.previous))
        logging.info('StageAudit %s', json.dumps(row, allow_nan=False))
        self.previous, self.previous_stage = logits, stage
        return row
