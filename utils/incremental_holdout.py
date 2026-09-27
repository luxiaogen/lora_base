"""Read-only global-class validation of reserved training images."""
import hashlib
import json
import logging

import torch

from utils.stage_audit import collect_logits, stage_metrics


def record_holdout(network, loader, device, task, known_classes):
    logits, targets, indices = collect_logits(network, loader, device)
    row = dict(task=task, stage='post_ca', source='train_holdout',
               class_count=logits.shape[1],
               sample_sha256=hashlib.sha256(torch.stack((indices, targets)).numpy().tobytes()).hexdigest(),
               metrics=stage_metrics(logits, targets, known_classes))
    logging.info('IncrementalHoldout %s', json.dumps(row, allow_nan=False))
    return row
