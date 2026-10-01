"""Summarize this diagnostic only; test reports do not tune routing thresholds."""
import argparse
import csv
import json
from pathlib import Path


def summarize(directory, expected_tasks=None):
    directory = Path(directory)
    reports = [json.loads(path.read_text())
               for path in sorted(directory.glob('task_*_test_report_only.json'))]
    table = []
    for report in reports:
        for group, values in report['groups'].items():
            row = dict(task=report['task'], group=group,
                       **{key: values[key] for key in ('n', 'base_accuracy', 'anchor_accuracy',
                           'oracle_accuracy', 'oracle_gain_pp', 'task_id_ceiling_accuracy',
                           'rescued', 'harmed', 'neutral_disagreement')})
            for signal, metrics in values['signals'].items():
                for key in ('rescue_vs_harm_auc', 'fixed_rule_accuracy', 'selected_rescued',
                            'selected_harmed', 'selected_neutral', 'net_gain_pp'):
                    row[f'{signal}_{key}'] = metrics.get(key)
            table.append(row)
    if not reports:
        print('No test reports yet:', directory)
        return None
    total = [row for row in table if row['group'] == 'total']
    metrics = ('base_accuracy', 'anchor_accuracy', 'oracle_accuracy', 'task_id_ceiling_accuracy',
               'margin_advantage_fixed_rule_accuracy', 'proposal_advantage_fixed_rule_accuracy')
    summary = dict(completed_tasks=[row['task'] for row in total],
                   full_t10_completed=[row['task'] for row in total] == list(range(10)),
                   test_used_for_threshold_fitting=False,
                   average={key: sum(row[key] for row in total) / len(total) for key in metrics},
                   last={key: total[-1][key] for key in metrics})
    train_tasks = [json.loads(path.read_text())['task']
                   for path in sorted(directory.glob('task_*_current_train_seen_probe.json'))]
    summary['current_train_probe_tasks'] = train_tasks
    summary['expected_reports_complete'] = (summary['completed_tasks'] == list(range(expected_tasks))
                                           and train_tasks == list(range(expected_tasks))
                                           and all((directory / f'task_{task:02d}_{source}.csv').exists()
                                                   for task in range(expected_tasks)
                                                   for source in ('test_report_only', 'current_train_seen_probe'))) \
                                          if expected_tasks is not None else None
    with (directory / 'task_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('Per-task Old/New, rescued/harmed counts, and AUC:', directory / 'task_metrics.csv')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-tasks', type=int)
    args = parser.parse_args()
    summary = summarize(args.directory, args.expected_tasks)
    raise SystemExit(0 if summary is not None and summary['expected_reports_complete'] is not False else 1)
