"""Fixed-prefix classification diagnostics; test outputs never enter training."""
import json
import logging
from pathlib import Path

import torch


def compare_prefix(logits, targets, reference):
    """Inputs are aligned to the reference sample order by HistoryAudit."""
    count = reference.shape[1]
    before = reference.argmax(1).eq(targets)
    fixed = logits[:, :count].argmax(1).eq(targets)
    full = logits.argmax(1).eq(targets)
    return dict(
        n=len(targets), fixed_classes=count,
        reference_correct=int(before.sum()), fixed_correct=int(fixed.sum()),
        full_correct=int(full.sum()),
        fixed_broken=int((before & ~fixed).sum()),
        fixed_corrected=int((~before & fixed).sum()),
        expansion_only_errors=int((fixed & ~full).sum()),
    )


class HistoryAudit:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.references = {}

    def record(self, task, stage, logits, targets, indices):
        # Class plus dataset index identifies samples across growing test prefixes.
        lookup = {(int(y), int(i)): j for j, (y, i) in enumerate(zip(targets, indices))}
        rows = []
        for source_task, ref in self.references.items():
            positions = torch.tensor([lookup[(int(y), int(i))]
                                      for y, i in zip(ref['targets'], ref['indices'])])
            row = dict(task=task, stage=stage, reference_task=source_task,
                       source='test_report_only',
                       **compare_prefix(logits[positions], ref['targets'], ref['logits']))
            logging.info('HistoryAudit %s', json.dumps(row))
            rows.append(row)
        snapshot = dict(task=task, stage=stage, logits=logits, targets=targets, indices=indices)
        torch.save(snapshot, self.directory / f'task{task:02d}_{stage}.pt')
        if stage == 'post_ca':
            self.references[task] = snapshot
        return rows
