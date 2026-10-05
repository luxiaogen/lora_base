"""Matched compact-DualMask evidence; no smoke scores or cross-machine contrasts."""
import argparse
import json
from pathlib import Path

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import write_csv


SPEC = Path(__file__).resolve().parent / 'sweeps/imgr10_compact_structure_night.json'
BASE = SPEC.parent / 'imgr10_protect_position_3090.json'
PAIRS = {
    '3090': [('R1', 'R0'), ('R2', 'R1'), ('R3', 'R1'), ('R4', 'R5'),
             ('R0', 'R6'), ('R4', 'R0'), ('R5', 'R6'), ('R7', 'R3')],
    '5090': [('S1', 'S0'), ('S2', 'S1'), ('S3', 'S0'), ('S4', 'S0'),
             ('S5', 'S3'), ('S5', 'S4'), ('S2', 'S4'), ('S2', 'S6'), ('S7', 'S2')],
}


def matching_issues(snapshots, machine):
    if not snapshots:
        return []
    spec = json.loads(SPEC.read_text())
    factors = {key for row in spec[machine] for key in row['overrides']}
    reference = next(iter(snapshots.values()))
    issues = []
    for name, snapshot in snapshots.items():
        for field in ('code_revision', 'source_sha256', 'machine', 'phase', 'software', 'hardware'):
            if snapshot.get(field) != reference.get(field):
                issues.append(dict(mode=name, field=field))
        config = snapshot['effective_config']
        for key in reference['effective_config'].keys() | config.keys():
            if key not in factors | {'prefix'} and config.get(key) != reference['effective_config'].get(key):
                issues.append(dict(mode=name, field='config.' + key))
        expected = dict(json.loads(BASE.read_text())['common_overrides'])
        expected.update(spec['common_overrides'])
        expected.update(next(row['overrides'] for row in spec[machine] if row['name'] == name))
        expected['wandb_group'] = 'imgr10_compact_structure_' + machine
        for key, value in expected.items():
            if config.get(key) != value:
                issues.append(dict(mode=name, field='wrong_factor.' + key))
        for field, value in (('machine', machine), ('phase', 'formal'), ('mode', name)):
            if snapshot.get(field) != value:
                issues.append(dict(mode=name, field='wrong_' + field))
    return issues


def contrasts(complete, machine):
    result = {a + '_minus_' + b: {key: complete[a][key] - complete[b][key] for key in METRICS}
              for a, b in PAIRS[machine] if a in complete and b in complete}
    if machine == '5090' and all(name in complete for name in ('S0', 'S3', 'S4', 'S5')):
        result['ranking_strength_2x2'] = {}
        for key in METRICS:
            product_fixed, product_adaptive, magnitude_fixed, magnitude_adaptive = [
                complete[name][key] for name in ('S0', 'S3', 'S4', 'S5')]
            result['ranking_strength_2x2'][key] = dict(
                magnitude_effect=((magnitude_fixed + magnitude_adaptive) - (product_fixed + product_adaptive)) / 2,
                adaptive_strength_effect=((product_adaptive + magnitude_adaptive) - (product_fixed + magnitude_fixed)) / 2,
                interaction=(magnitude_adaptive - magnitude_fixed) - (product_adaptive - product_fixed))
    return result


def draw(directory, complete, rows, machine):
    if not complete:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, name):
        for suffix in ('png', 'pdf'):
            fig.savefig(directory / (name + '.' + suffix), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, summary in complete.items():
        for ax, old, new, title in ((axes[0], 'Old', 'New', 'Final task'),
                                   (axes[1], 'StageOld', 'StageNew', 'Task1-9 mean')):
            ax.scatter(summary[old], summary[new])
            ax.annotate(mode, (summary[old], summary[new]), xytext=(3, 3), textcoords='offset points')
            ax.set(xlabel='Old accuracy (%)', ylabel='New accuracy (%)', title=title)
    save(fig, 'old_new')
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for mode in complete:
        selected = [row for row in rows['tasks'] if row['mode'] == mode]
        ax.plot([row['task'] for row in selected], [row['total'] for row in selected], marker='.', label=mode)
    ax.set(xlabel='Task', ylabel='All-seen accuracy (%)')
    ax.legend(ncol=4)
    save(fig, 'task_curves')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, branch in zip(axes, ('S', 'P', 'Single')):
        for mode in complete:
            selected = [row for row in rows['epochs'] if row['mode'] == mode and row['branch'] == branch and row['task'] > 0]
            points = sorted({(row['task'], row['epoch']) for row in selected})
            values = [sum(row['effective_norm'] for row in selected if (row['task'], row['epoch']) == point) /
                      sum((row['task'], row['epoch']) == point for row in selected) for point in points]
            if points:
                ax.plot(range(len(points)), values, label=mode)
        ax.set(xlabel='Task1-9 epochs in order', ylabel='Mean effective update norm', title=branch)
        if ax.lines:
            ax.legend(fontsize=8)
    save(fig, 'norm_trajectories')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in complete:
        for ax, partition in zip(axes, ('old', 'new')):
            selected = [row for row in rows['diagnostics'] if row['mode'] == mode and row['partition'] == partition]
            ax.plot(range(len(selected)), [row['wpre_margin'] - row['permuted_margin'] for row in selected], marker='.', label=mode)
            ax.set(xlabel='Task1/5/9 x epoch1/5/10/20', ylabel='Wpre minus permuted margin', title=partition)
            ax.axhline(0, color='grey', linewidth=.8)
    axes[0].legend(ncol=4, fontsize=8)
    save(fig, 'position_margin')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    modes = list(complete)
    bottom = [0.0] * len(modes)
    for stage in sorted({row['stage'] for row in rows['costs']}):
        values = [sum(row['seconds'] for row in rows['costs'] if row['mode'] == mode and row['stage'] == stage) / 60 for mode in modes]
        axes[0].bar(modes, values, bottom=bottom, label=stage)
        bottom = [a + b for a, b in zip(bottom, values)]
    final = {mode: next((row for row in reversed(rows['storage']) if row['mode'] == mode and row['stage'] == 'post_ca_merged'), {}) for mode in modes}
    bottom = [0.0] * len(modes)
    for group in sorted({key for row in final.values() for key in row.get('groups_bytes', {})}):
        values = [final[mode].get('groups_bytes', {}).get(group, 0) / 2 ** 20 for mode in modes]
        axes[1].bar(modes, values, bottom=bottom, label=group)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set(ylabel='Measured minutes', title='Training / diagnostics / other stages')
    axes[1].set(ylabel='Live tensor MiB', title='Merged storage, including CA covariance')
    for ax in axes:
        if ax.containers:
            ax.legend(fontsize=6)
    save(fig, 'costs')
    if machine == '3090':
        selected = [mode for mode in ('R0', 'R1', 'R2', 'R3', 'R7') if mode in complete]
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for offset, key in enumerate(('Average', 'Last')):
            ax.bar([i + offset * .35 for i in range(len(selected))], [complete[m][key] for m in selected], width=.35, label=key)
        ax.set(xticks=[i + .175 for i in range(len(selected))], xticklabels=selected,
               ylabel='Accuracy (%)', title='Branch organization and joint/separate gating')
        ax.legend()
        save(fig, 'branch_organization')


def summarize(directory, machine, records):
    spec = json.loads(SPEC.read_text())
    labels = {row['name']: row['label'] for row in spec[machine]}
    summaries, snapshots = [], {}
    all_rows = {name: [] for name in ('epochs', 'updates', 'masks', 'diagnostics', 'costs', 'storage', 'tasks')}
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            summaries.append(dict(record, label=labels[record['mode']]))
            continue
        summary, snapshot, rows = read_run(directory, record)
        summary['label'] = labels[record['mode']]
        summaries.append(summary)
        snapshots[record['mode']] = snapshot
        for name in all_rows:
            all_rows[name].extend(rows[name])
    write_csv(directory / 'results.csv', summaries)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, rows in all_rows.items():
        write_csv(directory / (name + '.csv'), rows)
    issues = matching_issues(snapshots, machine)
    complete = {row['mode']: row for row in summaries if row.get('valid_performance')}
    reference = complete.get('R0' if machine == '3090' else 'S0')
    if reference:
        issues += [dict(mode=mode, field='Task0_startpoint_accuracy') for mode, row in complete.items()
                   if row['Task0'] != reference['Task0']]
    comparisons = contrasts(complete, machine) if not issues else {}
    (directory / 'contrasts.json').write_text(json.dumps(dict(machine=machine, matching_issues=issues,
        completed_pairs=comparisons), indent=2) + '\n')
    columns = ['mode', 'label', 'status', 'tasks_reported', 'Task0'] + list(METRICS)
    lines = ['# ' + machine + ' compact DualMask evidence', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in summaries:
        lines.append('| ' + ' | '.join(str(row.get(key, '')) for key in columns) + ' |')
    lines += ['', '## Matched contrasts: first minus second (pp)', '',
              '| Contrast | ' + ' | '.join(METRICS) + ' |', '|---|' + '---:|' * len(METRICS)]
    for name, values in comparisons.items():
        if name != 'ranking_strength_2x2':
            lines.append('| ' + name + ' | ' + ' | '.join(format(values[key], '.4f') for key in METRICS) + ' |')
    if 'ranking_strength_2x2' in comparisons:
        lines += ['', '## Original-count ranking x adaptive strength (S0/S3/S4/S5)', '',
                  'S4/S5 still use the product score to determine the original count.',
                  '```json', json.dumps(comparisons['ranking_strength_2x2'], indent=2), '```']
    lines += ['', 'Matching issues: ' + json.dumps(issues), '',
        'Incomplete pairs have no attribution result. Unstarted groups are pending, not failures.',
        'Single seed does not prove equivalence or significance. No automatic baseline replacement.',
        'The single branch preserves rank104 and raw gradient scale; it also changes joint vs separate gating.',
        'Same-state paired norms do not equalize separately trained trajectories; see epochs.csv.',
        'Fixed-control runs retain competence evaluation; no elimination of that cost is claimed.',
        'Sample diagnostics use test labels only for reporting. No old training images or task-ID inference.',
        'Storage excludes optimizer/transient workspaces and includes CA covariance. Activity is in storage.csv.',
        'Mechanism completeness is reported separately from performance validity in results.json.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    draw(directory, complete, all_rows, machine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    summarize(args.directory, manifest['machine'], json.loads((args.directory / 'queue.json').read_text()))
