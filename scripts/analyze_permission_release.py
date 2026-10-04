"""Matched full-run summaries for permissions and compact task controls."""
import argparse
import json
from pathlib import Path

from analyze_core_evidence import read_run, METRICS
from analyze_protect_position import json_rows, write_csv


SPEC = Path(__file__).resolve().parent / 'sweeps/imgr10_permission_release_night.json'
BASE = SPEC.parent / 'imgr10_protect_position_3090.json'


def matching_issues(snapshots, machine):
    spec = json.loads(SPEC.read_text())
    factors = {key for row in spec[machine] for key in row['overrides']}
    reference = next(iter(snapshots.values()))
    issues = []
    for mode, snapshot in snapshots.items():
        for field in ('code_revision', 'source_sha256', 'machine', 'phase', 'software', 'hardware'):
            if snapshot.get(field) != reference.get(field):
                issues.append(dict(mode=mode, field=field))
        for key in reference['effective_config'].keys() | snapshot['effective_config'].keys():
            if key not in factors | {'prefix'} and reference['effective_config'].get(key) != snapshot['effective_config'].get(key):
                issues.append(dict(mode=mode, field='config.' + key))
        expected = dict(json.loads(BASE.read_text())['common_overrides'])
        expected.update(spec['common_overrides'])
        expected.update(next(row['overrides'] for row in spec[machine] if row['name'] == mode))
        expected['wandb_group'] = 'imgr10_permission_release_' + machine
        for key, value in expected.items():
            if snapshot['effective_config'].get(key) != value:
                issues.append(dict(mode=mode, field='wrong_factor.' + key))
        for field, value in (('machine', machine), ('phase', 'formal')):
            if snapshot.get(field) != value:
                issues.append(dict(mode=mode, field='wrong_' + field))
    return issues


def contrasts(complete, machine):
    pairs = [('A1', name) for name in ('A0', 'A2', 'A3', 'A5', 'A6')]
    pairs += [('A0', 'A4'), ('A5', 'A4')]
    if machine == '5090':
        pairs = [('B1', 'B0'), ('B2', 'B1'), ('B3', 'B2'), ('B4', 'B3'), ('B4', 'B5')]
    result = {a + '_minus_' + b: {key: complete[a][key] - complete[b][key] for key in METRICS}
              for a, b in pairs if a in complete and b in complete}
    cells = {'B0': (0, 0, 0), 'B6': (1, 0, 0), 'B7': (0, 1, 0), 'B8': (0, 0, 1),
             'B9': (1, 1, 0), 'B10': (1, 0, 1), 'B11': (0, 1, 1), 'B1': (1, 1, 1)}
    if machine == '5090' and all(name in complete for name in cells):
        result['fixed_controls_factorial_descriptive'] = {
            label: {key: sum((1 if code[index] else -1) * complete[name][key]
                            for name, code in cells.items()) / 4 for key in METRICS}
            for index, label in enumerate(('coverage_effect', 'protect_strength_effect', 'rank_effect'))}
        for i, j, label in ((0, 1, 'coverage_strength_interaction'), (0, 2, 'coverage_rank_interaction'),
                            (1, 2, 'strength_rank_interaction')):
            result['fixed_controls_factorial_descriptive'][label] = {
                key: sum((1 if code[i] == code[j] else -1) * complete[name][key]
                         for name, code in cells.items()) / 2 for key in METRICS}
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
                ax.scatter(summary[old], summary[new], s=45)
                ax.annotate(mode, (summary[old], summary[new]), xytext=(4, 4), textcoords='offset points')
                ax.set(xlabel='Old accuracy (%)', ylabel='New accuracy (%)', title=title)
        fig.savefig(directory / 'old_new.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for mode in complete:
            data = [row for row in rows['tasks'] if row['mode'] == mode]
            ax.plot([row['task'] for row in data], [row['total'] for row in data], marker='.', label=mode)
        ax.set(xlabel='Task', ylabel='All-seen accuracy (%)')
        ax.legend(ncol=4, fontsize=8)
        fig.savefig(directory / 'task_curves.png', dpi=180)
        plt.close(fig)
    selected = [r for r in rows['release_updates'] if r['mode'] in complete and r['reference_norm'] > 0]
    if selected:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for mode in sorted({row['mode'] for row in selected}):
            data = [row for row in selected if row['mode'] == mode]
            epochs = sorted({(r['task'], r['epoch']) for r in data})
            values = [sum(r['effective_norm'] / r['reference_norm'] for r in data
                          if (r['task'], r['epoch']) == epoch) /
                      sum((r['task'], r['epoch']) == epoch for r in data) for epoch in epochs]
            ax.plot(range(len(values)), values, label=mode)
        ax.axhline(1, color='grey', linestyle='--')
        ax.set(xlabel='Incremental training epoch', ylabel='Effective / same-state hard-permission norm')
        ax.legend()
        fig.savefig(directory / 'release_norms.png', dpi=180)
        plt.close(fig)
    diagnostics = [r for r in rows['release_diagnostics'] if r['mode'] in complete and r['count']]
    if diagnostics:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for partition in ('old', 'new'):
            modes = sorted({row['mode'] for row in diagnostics})
            values = []
            for mode in modes:
                data = [r for r in diagnostics if r['mode'] == mode and r['partition'] == partition]
                values.append(sum(r['released_margin'] - r['reference_margin'] for r in data) / len(data))
            ax.plot(modes, values, marker='o', label=partition)
        ax.axhline(0, color='grey', linestyle='--')
        ax.set(ylabel='Released minus hard-reference margin', title='Fixed test samples; report only')
        ax.legend()
        fig.savefig(directory / 'release_margin.png', dpi=180)
        plt.close(fig)
    costs = [r for r in rows['costs'] if r['mode'] in complete]
    if costs:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        modes = sorted(complete)
        bottoms = [0.] * len(modes)
        for stage in sorted({r['stage'] for r in costs}):
            values = [sum(r['seconds'] for r in costs if r['mode'] == mode and r['stage'] == stage) / 60 for mode in modes]
            ax.bar(modes, values, bottom=bottoms, label=stage)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set(ylabel='Recorded stage time (min)')
        ax.legend(fontsize=7, ncol=3)
        fig.savefig(directory / 'costs.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        values = [max(r['cuda_peak_allocated_bytes'] for r in costs if r['mode'] == mode) / 2**20 for mode in modes]
        ax.bar(modes, values)
        ax.set(ylabel='Peak allocated CUDA memory (MiB)', title='Recorded task-reset cumulative peaks')
        fig.savefig(directory / 'gpu_memory.png', dpi=180)
        plt.close(fig)
    storage = [r for r in rows['storage'] if r['mode'] in complete]
    if storage:
        modes = sorted({row['mode'] for row in storage})
        last = {mode: [r for r in storage if r['mode'] == mode][-1]['groups_bytes'] for mode in modes}
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        bottoms = [0.] * len(modes)
        for group in sorted({key for values in last.values() for key in values}):
            values = [last[mode].get(group, 0) / 2**20 for mode in modes]
            ax.bar(modes, values, bottom=bottoms, label=group)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set(ylabel='Last recorded live tensor memory (MiB)')
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(directory / 'storage.png', dpi=180)
        plt.close(fig)


def summarize(directory, machine, records):
    summaries, snapshots = [], {}
    rows = {name: [] for name in ('tasks', 'epochs', 'updates', 'masks', 'diagnostics', 'costs',
                                 'storage', 'release_masks', 'release_updates', 'release_train', 'release_diagnostics')}
    for record in records:
        if record.get('exit_code') is None:
            continue
        summary, snapshot, data = read_run(directory, record)
        summaries.append(summary)
        snapshots[record['mode']] = snapshot
        content = (directory / record['mode'] / 'training.log').read_text()
        for name, marker in (('release_masks', 'PPermissionRelease'), ('release_updates', 'PPermissionUpdate'),
                             ('release_train', 'PPermissionTrainDiagnostic'), ('release_diagnostics', 'PPermissionTestDiagnostic')):
            data[name] = [dict(mode=record['mode'], **r) for r in json_rows(content, marker)]
        release = snapshot['effective_config'].get('p_permission_release', 'off') != 'off'
        summary['release_mask_records_complete'] = (len(data['release_masks']) == 9 * 3 * 12) if release else None
        expected = {(task, epoch, partition) for task in (1, 5, 9) for epoch in (1, 5, 10, 20)
                    for partition in ('old', 'new')}
        summary['release_diagnostics_complete'] = (len(data['release_diagnostics']) == len(expected)
            and {(r['task'], r['epoch'], r['partition']) for r in data['release_diagnostics']} == expected) if release else None
        for name, values in data.items():
            rows.setdefault(name, []).extend(values)
    complete = {row['mode']: row for row in summaries if row['valid_performance']}
    issues = matching_issues(snapshots, machine) if snapshots else []
    comparison = contrasts(complete, machine) if not issues else {}
    write_csv(directory / 'results.csv', summaries)
    for name, data in rows.items():
        write_csv(directory / (name + '.csv'), data)
    (directory / 'results.json').write_text(json.dumps(dict(machine=machine, runs=summaries,
        matching_issues=issues, contrasts=comparison, queue=records), indent=2) + '\n')
    labels = {r['name']: r['label'] for r in json.loads(SPEC.read_text())[machine]}
    report = ['# ' + machine + ' permission-release results', '',
              'Only complete formal T10 runs are performance evidence. One seed; descriptive comparisons.', '',
              '| Mode | Change | Average | Last | Old | New | StageOld | StageNew | Forgetting |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, row in complete.items():
        report.append('| ' + mode + ' | ' + labels[mode] + ' | ' + ' | '.join(f'{row[k]:.3f}' for k in
            ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')) + ' |')
    report += ['', 'Matching issues: ' + json.dumps(issues), '',
               'Same-state norm control does not equalize different training trajectories.',
               'Training utility is not old-class safety. Test diagnostics never select settings.', '',
               '```json', json.dumps(comparison, indent=2), '```']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    draw(directory, complete if not issues else {}, rows)
    return dict(runs=summaries, matching_issues=issues, contrasts=comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    args = parser.parse_args()
    summarize(args.directory, args.machine, json.loads((args.directory / 'queue.json').read_text()))


if __name__ == '__main__':
    main()
