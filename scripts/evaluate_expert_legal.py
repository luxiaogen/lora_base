"""Fit all policies before reading test rows; test labels are reporting-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.expert_legal import candidate_specs, choose, fit_policy, outcomes, signal_view


def read_rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        records = list(reader)
        integer = ('index', 'target', 'base_prediction', 'anchor_prediction', 'task_id_prediction')
        return {key: np.array([row[key] for row in records], dtype=int if key in integer else float)
                for key in reader.fieldnames}


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(rows, switch, known_classes, name, task, source):
    rescued, harmed = outcomes(rows)
    prediction = np.where(switch, rows['anchor_prediction'], rows['base_prediction'])
    base_correct = rows['base_prediction'] == rows['target']
    oracle = base_correct | (rows['anchor_prediction'] == rows['target'])
    metrics = []
    for group, mask in (('total', np.ones(len(prediction), dtype=bool)),
                        ('old', rows['target'] < known_classes),
                        ('new', rows['target'] >= known_classes)):
        n = int(mask.sum())
        accuracy = lambda correct: float(correct[mask].mean() * 100) if n else None
        r, h = int((switch & rescued & mask).sum()), int((switch & harmed & mask).sum())
        metrics.append(dict(task=task, source=source, policy=name, group=group, n=n,
                            base_accuracy=accuracy(base_correct), accuracy=accuracy(prediction == rows['target']),
                            oracle_accuracy=accuracy(oracle), rescued=r, harmed=h,
                            switched=int((switch & mask).sum()),
                            neutral=int((switch & ~(rescued | harmed) & mask).sum()),
                            net_gain_pp=100. * (r - h) / n if n else None))
    return metrics


def evaluate(directory, output, calibration_source, expected_tasks=10):
    directory, output = Path(directory), Path(output)
    stems = sorted(directory.glob('task_*_test_report_only.json'))
    tasks = [int(stem.name.split('_')[1]) for stem in stems]
    if tasks != list(range(expected_tasks)):
        raise FileNotFoundError(f'Expected Task0..{expected_tasks - 1}, found {tasks}: {directory}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'policies').mkdir(exist_ok=True)
    all_metrics, source_hashes, selections = [], {}, []
    for task in tasks:
        # Task0 retains the original prediction. It has no withheld calibration images.
        source = 'current_train_seen_probe' if task == 0 else calibration_source
        train_path = directory / f'task_{task:02d}_{source}.csv'
        train = read_rows(train_path)
        metadata = json.loads(train_path.with_suffix('.json').read_text())
        source_hashes[train_path.name] = hashlib.sha256(train_path.read_bytes()).hexdigest()
        known_classes = metadata.get('known_classes', int(train['target'].min()))
        audit_mask = train['index'] % 3 == 2
        fit = {key: value[~audit_mask] for key, value in train.items()}
        audit = {key: value[audit_mask] for key, value in train.items()}
        specs = candidate_specs(extended='prototype_advantage' in train and 'entropy_advantage' in train)
        policies = [fit_policy(spec, fit) for spec in specs]
        best_name, best_utility = 'base', 0
        for policy in policies:
            switch = choose(policy, signal_view(audit)) if task else np.zeros(len(audit['target']), dtype=bool)
            metrics = report(audit, switch, known_classes, policy['name'], task, 'calibration_audit')
            utility = metrics[0]['rescued'] - 3 * metrics[0]['harmed']
            if utility > best_utility:
                best_name, best_utility = policy['name'], utility
            all_metrics.extend(metrics)
        payload = dict(task=task, calibration_source=source, selection_source='calibration_audit_only',
                       fit_n=len(fit['target']), audit_n=len(audit['target']),
                       fit_audit_index_overlap=len(set(fit['index']) & set(audit['index'])),
                       selection_objective='rescued - 3 * harmed; ties default to base / first fixed policy',
                       selected_policy=best_name, selected_audit_utility=best_utility, policies=policies,
                       test_used_for_fitting_or_rule_selection=False)
        # Seal fitted policies and the train-selected rule BEFORE opening test data.
        policy_path = output / 'policies' / f'task_{task:02d}.json'
        policy_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        selections.append(dict(task=task, selected_policy=best_name, audit_utility=best_utility,
                               policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest()))
        test_path = directory / f'task_{task:02d}_test_report_only.csv'
        test = read_rows(test_path)
        source_hashes[test_path.name] = hashlib.sha256(test_path.read_bytes()).hexdigest()
        view = signal_view(test)
        switches = {policy['name']: choose(policy, view) for policy in policies}
        switches['base'] = np.zeros(len(test['target']), dtype=bool)
        if task == 0:
            switches = {name: switches['base'] for name in switches}
        switches['train_selected'] = switches[best_name]
        for name, switch in switches.items():
            metrics = report(test, switch, known_classes, name, task, 'test_report_only')
            all_metrics.extend(metrics)
            if name in ('base', 'train_selected'):
                for row in metrics:
                    print('ExpertLegalTask', json.dumps(row), flush=True)
    write_csv(output / 'task_metrics.csv', all_metrics)
    totals = [row for row in all_metrics if row['source'] == 'test_report_only' and row['group'] == 'total']
    names = list(dict.fromkeys(row['policy'] for row in totals))
    summary = []
    for name in names:
        rows = [row for row in totals if row['policy'] == name]
        last_groups = {row['group']: row for row in all_metrics
                       if row['source'] == 'test_report_only' and row['task'] == tasks[-1] and row['policy'] == name}
        summary.append(dict(policy=name, tasks=len(rows),
                            average=sum(row['accuracy'] for row in rows) / len(rows),
                            average_gain_pp=sum(row['net_gain_pp'] for row in rows) / len(rows),
                            last=rows[-1]['accuracy'], last_gain_pp=rows[-1]['net_gain_pp'],
                            old=last_groups['old']['accuracy'], new=last_groups['new']['accuracy'],
                            last_rescued=rows[-1]['rescued'], last_harmed=rows[-1]['harmed']))
    write_csv(output / 'summary.csv', summary)
    result = dict(completed_tasks=tasks, full_t10_completed=tasks == list(range(10)),
                  candidate_count=len(names) - 2, test_used_for_fitting_or_rule_selection=False,
                  task1_plus_calibration_expert_unseen=calibration_source == 'current_train_holdout',
                  task0_calibration_expert_unseen=False,
                  calibration_old_images_accessed=False, true_task_id_used_for_prediction=False,
                  calibration_covers_old_distribution=False, task0_prediction='unchanged_base',
                  inference_backbone_forwards=2, source_directory=str(directory.resolve()),
                  calibration_source=calibration_source, source_sha256=source_hashes,
                  train_selections=selections, summary=summary)
    run_path = directory.parent / 'run.json'
    result['source_run_metadata'] = json.loads(run_path.read_text()) if run_path.exists() else None
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in summary:
        print('ExpertLegal', json.dumps(row), flush=True)
    print('All predeclared rules (not a test-selected winner):', output / 'summary.csv', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration-source', choices=('current_train_seen_probe', 'current_train_holdout'),
                        default='current_train_holdout')
    parser.add_argument('--expected-tasks', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.directory, args.output, args.calibration_source, args.expected_tasks)


if __name__ == '__main__':
    main()
