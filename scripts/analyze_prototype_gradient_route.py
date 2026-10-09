"""原型梯度分工三组证据；缺失单元和指纹差异不产生配对结论。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows
from analyze_tail_update import write_csv


def paired_results(records, issues):
    complete = {r['mode'].split('_seed')[0]: r for r in records
                if r.get('valid_performance')} if not issues else {}
    return [dict(comparison=first + '_minus_' + second,
                 **{k: complete[first][k] - complete[second][k] for k in METRICS})
            for first, second in (('B', 'A'), ('B', 'C'), ('C', 'A'))
            if first in complete and second in complete]


def summarize_saved(directory):
    import run_tail_update as runner
    manifest = json.loads((directory / 'manifest.json').read_text())
    runner.SPEC = runner.ROOT / manifest['sweep_spec']
    records = json.loads((directory / 'queue.json').read_text())
    results, tasks, batches, updates, snapshots, issues = [], [], [], [], [], []
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False))
            continue
        row, snapshot, measured = read_run(directory, record)
        text = (directory / record['mode'] / 'training.log').read_text()
        expected = dict(json.loads((runner.ROOT / runner.spec()['3090']['config']).read_text()),
                        **runner.settings_for('3090', record['mode']))
        difference = [k for k, v in expected.items() if k != 'prefix'
                      and snapshot['effective_config'].get(k) != v]
        if difference or snapshot['machine'] != '3090' or snapshot['phase'] != 'formal':
            issues.append(dict(mode=record['mode'], fields=difference, kind='identity_or_config'))
        seen = {(int(t), int(e)) for t, e in re.findall(
            r'LoRA learning rates: task=(\d+), epoch=(\d+)', text)}
        epochs = {(t, e) for t in range(10) for e in range(1, 21)}
        route_rows = json_rows(text, 'PrototypeRouteBatch')
        row.update(training_epochs_complete=seen == epochs,
                   route_epochs_complete={(r['task'], r['epoch']) for r in route_rows}
                        == {(t, e) for t, e in epochs if t > 0},
                   code_revision=snapshot['code_revision'], std=None,
                   complete_three_seeds=False)
        route_valid = all(r['samples'] > 0 and 0 <= r['prototype_correct'] <= r['samples']
            and r['mode'] == expected['dual_mask_gradient_route']
            and r['s_samples'] + r['p_samples'] == r['samples'] * (2 if r['mode'] == 'all' else 1)
            and (r['s_samples'] == r['p_samples'] == r['samples'] if r['mode'] == 'all'
                 else r['s_samples'] == r['prototype_correct'])
            and all(math.isfinite(r[k]) for k in ('probe_ms', 'S_full_grad_norm',
                  'P_full_grad_norm', 'S_assigned_grad_norm', 'P_assigned_grad_norm'))
            for r in route_rows)
        row['valid_performance'] &= row['training_epochs_complete'] and row['route_epochs_complete'] and route_valid
        if route_rows:
            count = sum(r['samples'] for r in route_rows)
            row.update(prototype_correct_ratio=sum(r['prototype_correct'] for r in route_rows) / count,
                s_sample_ratio=sum(r['s_samples'] for r in route_rows) / count,
                p_sample_ratio=sum(r['p_samples'] for r in route_rows) / count,
                probe_seconds=sum(r['probe_ms'] for r in route_rows) / 1000)
            for branch in ('S', 'P'):
                for kind in ('full', 'assigned'):
                    key = branch + '_' + kind + '_grad_norm'
                    row[key + '_mean'] = sum(r[key] for r in route_rows) / len(route_rows)
        delta_rows = json_rows(text, 'TailUpdate')
        row['peak_allocated_bytes'] = max((r['peak_allocated_bytes'] for r in delta_rows), default=0)
        results.append(row)
        snapshots.append(snapshot)
        tasks.extend(measured['tasks'])
        batches.extend(dict(r, experiment=record['mode']) for r in route_rows)
        updates.extend(dict(r, experiment=record['mode']) for r in delta_rows)
    if snapshots:
        reference = snapshots[0]
        for snapshot in snapshots[1:]:
            for field in ('code_revision', 'source_sha256', 'software', 'hardware'):
                if snapshot[field] != reference[field]:
                    issues.append(dict(mode=snapshot['mode'], field=field))
            for key in reference['effective_config'].keys() | snapshot['effective_config'].keys():
                if key not in ('prefix', 'dual_mask_gradient_route') and (
                        reference['effective_config'].get(key) != snapshot['effective_config'].get(key)):
                    issues.append(dict(mode=snapshot['mode'], field='config.' + key))
    pairs = paired_results(results, issues)
    for name, rows in (('results', results), ('pairs', pairs), ('matching_issues', issues)):
        (directory / (name + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), rows)
    for name, rows in (('tasks', tasks), ('route_batches', batches), ('tail_updates', updates)):
        write_csv(directory / (name + '.csv'), rows)
    complete = [r for r in results if r.get('valid_performance')]
    if complete:
        draw(directory, complete, tasks, batches, updates)


def draw(directory, complete, tasks, batches, updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row in complete:
        name = row['mode']
        selected = [r for r in tasks if r['mode'] == name]
        axes[0].plot([r['task'] for r in selected], [r['total'] for r in selected], '.-', label=name)
        axes[1].scatter(row['StageOld'], row['StageNew'])
        axes[1].annotate(name, (row['StageOld'], row['StageNew']), fontsize=8)
    axes[0].set(xlabel='Task', ylabel='All-seen accuracy (%)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Task1-9 mean Old (%)', ylabel='Task1-9 mean New (%)')
    fig.savefig(directory / 'performance.png', dpi=180)
    fig.savefig(directory / 'performance.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for row in complete:
        name = row['mode']
        selected = [r for r in batches if r['experiment'] == name]
        points = sorted({(r['task'], r['epoch']) for r in selected})
        for ax, key in zip(axes[:2], ('S_assigned_grad_norm', 'P_assigned_grad_norm')):
            groups = [[r[key] for r in selected if (r['task'], r['epoch']) == point] for point in points]
            ax.plot(range(len(points)), [sum(g) / len(g) for g in groups], label=name)
            ax.set(xlabel='Task1-9 epochs', ylabel=key)
        selected = [r for r in updates if r['experiment'] == name]
        points = sorted({(r['task'], r['epoch']) for r in selected})
        groups = [[r['effective_norm'] for r in selected if (r['task'], r['epoch']) == p] for p in points]
        axes[2].plot(range(len(points)), [sum(g) / len(g) for g in groups], label=name)
        axes[2].set(xlabel='Fixed sampled task/epochs', ylabel='Mean effective update norm')
    axes[0].legend(fontsize=8)
    fig.savefig(directory / 'gradient_and_update.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
