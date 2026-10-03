"""Summarize measured full-T10 evidence, keeping each machine separate."""
import argparse
import ast
import json
import math
from pathlib import Path
import re

from analyze_protect_position import METRICS, json_rows, write_csv


SPEC = Path(__file__).resolve().parent / 'sweeps/imgr10_core_evidence_night.json'
BASE = SPEC.parent / 'imgr10_protect_position_3090.json'
FACTOR_KEYS = set(key for machine in ('3090', '5090')
    for row in json.loads(SPEC.read_text())[machine] for key in row['overrides'])


def read_run(directory, record):
    path = directory / record['mode']
    content = (path / 'training.log').read_text()
    snapshot = json.loads((path / 'run.json').read_text())
    tasks = [ast.literal_eval(row) for row in re.findall(r'\[trainer.py\] => CNN: (\{[^\n]+\})', content)]
    averages = re.findall(r'\[trainer.py\] => Average Accuracy: ([\d.]+)', content)
    forgetting = re.findall(r'\[trainer.py\] => Forgetting:\s*([\d.-]+)', content)
    summary = dict(record, tasks_reported=len(tasks),
        full_t10_completed=len(tasks) == 10 and record.get('exit_code') == 0 and snapshot['phase'] == 'formal',
        runtime_error=bool(re.search(r'Traceback \(most recent call last\)|CUDA out of memory|Loss\s+(?:nan|inf)\b', content)))
    summary.update({key: None for key in METRICS})
    if tasks:
        summary.update(Task0=tasks[0]['total'], Last=tasks[-1]['total'], Old=tasks[-1]['old'], New=tasks[-1]['new'])
    if len(tasks) > 1:
        summary.update(StageOld=sum(row['old'] for row in tasks[1:]) / (len(tasks) - 1),
                       StageNew=sum(row['new'] for row in tasks[1:]) / (len(tasks) - 1))
    if averages:
        summary['Average'] = float(averages[-1])
    if forgetting:
        summary['Forgetting'] = float(forgetting[-1])
    summary['valid_performance'] = (summary['full_t10_completed'] and not summary['runtime_error']
        and all(summary[key] is not None and math.isfinite(summary[key]) for key in METRICS))
    rows = {}
    for name, marker in (('epochs', 'CoreEpochUpdate'), ('updates', 'ProtectionPositionUpdate'),
                         ('masks', 'ProtectionPositionMask'), ('diagnostics', 'CorePositionDiagnostic'),
                         ('costs', 'CoreCost'), ('storage', 'CoreStorage')):
        rows[name] = [dict(mode=record['mode'], **row) for row in json_rows(content, marker)]
    rows['tasks'] = [dict(mode=record['mode'], task=i, **row) for i, row in enumerate(tasks)]
    expected_epochs = {(task, epoch, layer, branch, projection)
        for task in range(10) for epoch in range(1, 21) for layer in range(12)
        for branch in (('S',) if task == 0 else ('S', 'P')) for projection in ('Q', 'K', 'V')}
    expected_diagnostics = {(task, epoch, partition) for task in (1, 5, 9)
        for epoch in (1, 5, 10, 20) for partition in ('old', 'new')}
    summary['epoch_records_complete'] = (len(rows['epochs']) == len(expected_epochs)
        and {(r['task'], r['epoch'], r['layer'], r['branch'], r['projection']) for r in rows['epochs']} == expected_epochs)
    summary['position_diagnostics_complete'] = (len(rows['diagnostics']) == len(expected_diagnostics)
        and {(r['task'], r['epoch'], r['partition']) for r in rows['diagnostics']} == expected_diagnostics
        and all(0 < r['count'] <= (512 if r['partition'] == 'old' else 128)
                and r['norm_control'] == 'same_state_paired_min' for r in rows['diagnostics']))
    return summary, snapshot, rows


def matching_issues(snapshots, machine):
    reference = next(iter(snapshots.values()))
    spec = json.loads(SPEC.read_text())
    issues = []
    ignored = FACTOR_KEYS | {'prefix'}
    for mode, snapshot in snapshots.items():
        for field in ('code_revision', 'source_sha256', 'machine', 'phase', 'software', 'hardware'):
            if snapshot.get(field) != reference.get(field):
                issues.append(dict(mode=mode, field=field))
        for key in reference['effective_config'].keys() | snapshot['effective_config'].keys():
            if key not in ignored and reference['effective_config'].get(key) != snapshot['effective_config'].get(key):
                issues.append(dict(mode=mode, field='config.' + key))
        expected = dict(json.loads(BASE.read_text())['common_overrides'])
        expected.update(spec['common_overrides'])
        expected.update(next(row['overrides'] for row in spec[machine] if row['name'] == mode))
        expected['wandb_group'] = 'imgr10_core_evidence_' + machine
        for key, value in expected.items():
            if snapshot['effective_config'].get(key) != value:
                issues.append(dict(mode=mode, field='wrong_factor.' + key))
        for field, value in (('machine', machine), ('phase', 'formal')):
            if snapshot.get(field) != value:
                issues.append(dict(mode=mode, field='wrong_' + field))
    return issues


def contrasts(complete, machine):
    pairs = [('R1', 'R2'), ('R5', 'R6'), ('R0', 'R3'), ('R0', 'R4')] if machine == '3090' else [
        ('S0', name) for name in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7')]
    result = {first + '_minus_' + second: {key: complete[first][key] - complete[second][key] for key in METRICS}
              for first, second in pairs if first in complete and second in complete}
    if machine == '5090' and all(mode in complete for mode in ('S0', 'S1', 'S2', 'S3')):
        result['gates_penalty_2x2'] = {}
        for key in METRICS:
            a, b, c, d = [complete[name][key] for name in ('S0', 'S1', 'S2', 'S3')]
            result['gates_penalty_2x2'][key] = dict(gates_effect=((a + b) - (c + d)) / 2,
                conflict_penalty_effect=((a + c) - (b + d)) / 2, interaction=(a - b) - (c - d))
    return result


def draw(directory, complete, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for mode, summary in complete.items():
            for ax, old, new, title in ((axes[0], 'Old', 'New', 'Final task'),
                                       (axes[1], 'StageOld', 'StageNew', 'Task1-9 mean')):
                ax.scatter(summary[old], summary[new], s=50)
                ax.annotate(mode, (summary[old], summary[new]), xytext=(4, 4), textcoords='offset points')
                ax.set(xlabel='Old accuracy (%)', ylabel='New accuracy (%)', title=title)
        for suffix in ('png', 'pdf'):
            fig.savefig(directory / ('old_new.' + suffix), dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for mode in complete:
            values = [row['total'] for row in rows['tasks'] if row['mode'] == mode]
            ax.plot(range(len(values)), values, marker='.', label=mode)
        ax.set(xlabel='Task', ylabel='All-seen accuracy (%)')
        ax.legend(ncol=4)
        fig.savefig(directory / 'task_curves.png', dpi=180)
        plt.close(fig)
    if rows['epochs']:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for mode in complete:
            for ax, branch in zip(axes, ('S', 'P')):
                selected = [row for row in rows['epochs'] if row['mode'] == mode and row['branch'] == branch and row['task'] > 0]
                points = sorted({(row['task'], row['epoch']) for row in selected})
                groups = [[row['effective_norm'] for row in selected if (row['task'], row['epoch']) == point] for point in points]
                means = [sum(values) / len(values) for values in groups]
                ax.plot(range(len(points)), means, label=mode)
                ax.set(xlabel='Task1-9 epochs in order', ylabel='Mean effective update norm', title=branch)
        axes[0].legend(ncol=3)
        fig.savefig(directory / 'norm_trajectories.png', dpi=180)
        plt.close(fig)
    if rows['diagnostics']:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for mode in complete:
            for ax, partition in zip(axes, ('old', 'new')):
                selected = [row for row in rows['diagnostics'] if row['mode'] == mode and row['partition'] == partition and row['count']]
                ax.plot(range(len(selected)), [row['wpre_margin'] - row['permuted_margin'] for row in selected], marker='.', label=mode)
                ax.set(xlabel='Task1/5/9, epoch1/5/10/20 checkpoints', ylabel='Wpre minus permuted margin', title=partition)
                ax.axhline(0, color='grey', linewidth=.8)
        axes[0].legend(ncol=3)
        fig.savefig(directory / 'position_margin.png', dpi=180)
        plt.close(fig)
    if complete and rows['costs']:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        modes = list(complete)
        bottom = [0.0] * len(modes)
        for stage in sorted({row['stage'] for row in rows['costs']}):
            values = [sum(row['seconds'] for row in rows['costs'] if row['mode'] == mode and row['stage'] == stage) / 60 for mode in modes]
            ax.bar(modes, values, bottom=bottom, label=stage)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(ylabel='Measured minutes', title='Measured stages; diagnostics shown separately')
        ax.legend(fontsize=7)
        fig.savefig(directory / 'stage_costs.png', dpi=180)
        plt.close(fig)
    if complete and rows['storage']:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        modes = list(complete)
        final = {mode: next((row for row in reversed(rows['storage']) if row['mode'] == mode and row['stage'] == 'post_ca_merged'), {}) for mode in modes}
        bottom = [0.0] * len(modes)
        for group in sorted({key for row in final.values() for key in row.get('groups_bytes', {})}):
            values = [final[mode].get('groups_bytes', {}).get(group, 0) / 2 ** 20 for mode in modes]
            ax.bar(modes, values, bottom=bottom, label=group)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(ylabel='Live tensor MiB', title='Post-CA merged storage; includes CA covariances')
        ax.legend(fontsize=7)
        fig.savefig(directory / 'storage_costs.png', dpi=180)
        plt.close(fig)
    selected = [mode for mode in ('R0', 'R3', 'R4') if mode in complete]
    if len(selected) == 3:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, keys in zip(axes, (('Average', 'Last'), ('StageOld', 'StageNew'))):
            for offset, key in enumerate(keys):
                ax.bar([i + offset * .35 for i in range(3)], [complete[m][key] for m in selected], width=.35, label=key)
            ax.set(xticks=[i + .175 for i in range(3)], xticklabels=['Asymmetric', 'Both soft', 'Both hard'], ylabel='Accuracy (%)')
            ax.legend()
        fig.savefig(directory / 'permissions.png', dpi=180)
        plt.close(fig)
    if all(mode in complete for mode in ('S0', 'S1', 'S2', 'S3')):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, key in zip(axes, ('Average', 'Last')):
            values = [[complete[m][key] for m in ('S3', 'S2')], [complete[m][key] for m in ('S1', 'S0')]]
            ax.imshow(values, cmap='viridis')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(values[i][j], '.3f'), ha='center', va='center', color='white')
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Penalty off', 'Penalty on'],
                   yticklabels=['Gates off', 'Gates on'], title=key + ' (%)')
        fig.savefig(directory / 'gates_penalty_2x2.png', dpi=180)
        plt.close(fig)


def summarize(directory, machine, records):
    summaries, snapshots = [], {}
    all_rows = {name: [] for name in ('epochs', 'updates', 'masks', 'diagnostics', 'costs', 'storage', 'tasks')}
    for record in records:
        if record['status'] == 'time_budget_pending':
            summaries.append(record)
            continue
        summary, snapshot, rows = read_run(directory, record)
        summaries.append(summary)
        snapshots[record['mode']] = snapshot
        for name in all_rows:
            all_rows[name].extend(rows[name])
    write_csv(directory / 'results.csv', summaries)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, rows in all_rows.items():
        write_csv(directory / (name + '.csv'), rows)
    issues = matching_issues(snapshots, machine) if snapshots else []
    complete = {row['mode']: row for row in summaries if row.get('valid_performance')}
    baseline = complete.get('R0' if machine == '3090' else 'S0')
    if baseline is not None:
        issues += [dict(mode=mode, field='Task0_startpoint_accuracy') for mode, row in complete.items()
                   if row['Task0'] != baseline['Task0']]
    comparisons = contrasts(complete, machine) if not issues else {}
    (directory / 'contrasts.json').write_text(json.dumps(dict(machine=machine, matching_issues=issues,
        completed_pairs=comparisons, limitation='same-state norm control does not equalize training trajectories'), indent=2) + '\n')
    lines = ['# ' + machine + ' real T10 evidence', '',
        '| Run | Status | Tasks | Average | Last | Old | New | StageOld | StageNew | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        values = [row['mode'], row['status'], str(row.get('tasks_reported', 0))]
        values += [str(row.get(key, '')) for key in ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Complete matched contrasts (first minus second, pp)', '',
        '| Contrast | Average | Last | Old | New | StageOld | StageNew | Forgetting |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, values in comparisons.items():
        if name != 'gates_penalty_2x2':
            lines.append('| ' + name + ' | ' + ' | '.join(format(values[key], '.4f') for key in METRICS) + ' |')
    if 'gates_penalty_2x2' in comparisons:
        lines += ['', '## Gates × conflict penalty effects (pp)', '',
                  '| Metric | Gates effect | Penalty effect | Interaction |', '|---|---:|---:|---:|']
        for key, value in comparisons['gates_penalty_2x2'].items():
            lines.append('| ' + key + ' | ' + ' | '.join(format(value[k], '.4f') for k in
                ('gates_effect', 'conflict_penalty_effect', 'interaction')) + ' |')
    if machine == '5090':
        lines += ['', '## Simplification candidates (not equivalence)', '',
                  'S1/S4/S6/S7 are candidates only. Complete matched differences appear above.',
                  'Do not retain a New increase bought by a larger Old loss. No automatic baseline replacement.']
    lines += ['', 'Config/source matching issues: ' + json.dumps(issues), '',
        'Task0 uses the original unmasked path. Formal inference uses all seen classes.',
        'Unstarted budget entries are pending, not negative results. Only complete matched pairs have contrasts.',
        'Same-state diagnostics use fixed test samples only; no labels are used to update parameters.',
        'The 3090 and 5090 reports are independent. Single-seed differences do not establish significance.',
        'Mechanism telemetry completeness is separate from performance validity; see results.json.',
        'Same-state equal norms do not guarantee equal norms across separately trained trajectories.',
        'Cost stages include data loading/evaluation where executed inside them; inference includes the existing Wpre-NCM report.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    draw(directory, complete, all_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    summarize(args.directory, manifest['machine'], json.loads((args.directory / 'queue.json').read_text()))
