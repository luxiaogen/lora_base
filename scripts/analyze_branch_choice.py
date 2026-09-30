"""Analyze proxy/privileged-risk agreement within each trajectory and task."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return float((ranks[inverse][labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def read_steps(path):
    rows = []
    for line in path.read_text().splitlines():
        if 'BranchChoiceStep ' not in line:
            continue
        step = json.loads(line.split('BranchChoiceStep ', 1)[1])
        raw = step['candidates'][0]
        for index, candidate in enumerate(step['candidates'][1:], 1):
            row = dict(file=path.name, mode=step['mode'], scope=step['scope'], task=step['task'],
                       epoch=step['epoch'], batch=step['batch'], candidate=index,
                       selected=index == step['selected'],
                       joint_b_step_norm=candidate['joint_b_step_norm'],
                       target_joint_b_step_norm=step['target_joint_b_step_norm'])
            for metric in ('new_loss', 'old_loss', 'old_logit_shift', 'feature_shift'):
                row[metric + '_delta'] = candidate[metric] - raw[metric] if metric in raw else None
            for metric in ('old_logit_shift', 'feature_shift'):
                row[metric + '_raw'] = raw[metric]
            rows.append(row)
    return rows


def signal_summary(rows):
    privileged = [r for r in rows if r['old_loss_delta'] is not None]
    groups = sorted({(r['file'], r['task']) for r in privileged})
    result = []
    for file, task in groups:
        group = [r for r in privileged if r['file'] == file and r['task'] == task]
        harmful = np.asarray([r['old_loss_delta'] > 1e-6 for r in group])
        for signal in ('old_logit_shift', 'feature_shift'):
            delta = np.asarray([r[signal + '_delta'] for r in group])
            # The same predeclared risk cap used by legal selection; no fitting.
            tolerance = np.asarray([abs(r.get(signal + '_raw', 0.)) * .001 + 1e-14 for r in group])
            predicted = delta > tolerance
            result.append(dict(file=file, task=task, signal=signal, n=len(group),
                               harmful_count=int(harmful.sum()), auc=auc(harmful, delta),
                               true_positive=int((harmful & predicted).sum()),
                               false_negative=int((harmful & ~predicted).sum()),
                               false_positive=int((~harmful & predicted).sum()),
                               true_negative=int((~harmful & ~predicted).sum())))
    return result


def analyze_directory(directory):
    rows = [row for path in sorted(Path(directory).glob('*.log'))
            if not path.name.startswith('smoke_') for row in read_steps(path)]
    if rows:
        with (Path(directory) / 'candidate_steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = signal_summary(rows)
    (Path(directory) / 'signal_summary.json').write_text(json.dumps(summary, indent=2))
    print('Signal report:', Path(directory) / 'signal_summary.json', flush=True)
    print('Risk target is sampled OLD TRAIN CE, not test forgetting; tasks/trajectories are separate.', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    analyze_directory(parser.parse_args().directory)
