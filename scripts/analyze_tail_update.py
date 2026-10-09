"""Same-machine paired seed comparisons; incomplete seed sets never produce means."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows, write_csv as write_nonempty_csv


def write_csv(path, rows):
    if rows:
        write_nonempty_csv(path, rows)
    else:
        path.write_text('')


def paired_results(complete, seeds):
    pairs = [('C', 'M'), ('C', 'T'), ('C', 'U'), ('M', 'O'),
             ('M40', 'M'), ('C40', 'C'), ('C40', 'M40')]
    result = []
    for first, second in pairs:
        available = [s for s in seeds if (first, s) in complete and (second, s) in complete]
        row = dict(comparison=first + '_minus_' + second, paired_seeds=available,
                   complete_planned_seeds=bool(seeds) and available == seeds,
                   complete_three_seeds=len(seeds) == 3 and available == seeds)
        row['per_seed'] = [dict(seed=s, **{k: complete[first, s][k] - complete[second, s][k] for k in METRICS}) for s in available]
        if row['complete_planned_seeds']:
            for key in METRICS:
                values = [r[key] for r in row['per_seed']]
                row[key + '_mean' if len(values) > 1 else key] = statistics.fmean(values)
                row[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
            if row['complete_three_seeds'] and first == 'C' and second == 'M':
                changes = row['per_seed']
                row['practical_followup_threshold_met'] = (
                    row['Average_mean'] >= .2 and row['Last_mean'] >= 0
                    and sum(r['Average'] > 0 for r in changes) >= 2
                    and min(r['Average'] for r in changes) >= -.2 and row['StageOld_mean'] >= -.1)
        if available:
            result.append(row)
    return result


def summarize(directory, machine, records):
    import run_tail_update as runner
    summaries, tasks, telemetry, snapshots, issues = [], [], [], [], []
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            summaries.append(record)
            continue
        row, snapshot, measured = read_run(directory, record)
        variant, seed = runner.identity(record['mode'])
        row.update(variant=variant, seed=seed, code_revision=snapshot['code_revision'])
        text = (directory / record['mode'] / 'training.log').read_text()
        config = snapshot['effective_config']
        seen = {(int(t), int(e)) for t,e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', text)}
        expected = {(t,e) for t in range(10) for e in range(1, (config['init_epoch'] if t == 0 else config['epochs']) + 1)}
        row['training_epochs_complete'] = seen == expected
        row['valid_performance'] &= row['training_epochs_complete']
        rows = [dict(r, mode=record['mode'], variant=variant, seed=seed) for r in json_rows(text, 'TailUpdate')]
        row['telemetry_records'] = len(rows)
        row['peak_allocated_bytes'] = max((r['peak_allocated_bytes'] for r in rows), default=0)
        expected_config = dict(json.loads((runner.ROOT / runner.spec()[machine]['config']).read_text()),
                               **runner.settings_for(machine, record['mode']))
        differences = [k for k,v in expected_config.items() if k != 'prefix' and config.get(k) != v]
        if differences or snapshot['machine'] != machine or snapshot['phase'] != 'formal':
            issues.append(dict(mode=record['mode'], fields=differences + ['identity'] if differences else ['identity']))
        summaries.append(row)
        tasks.extend(dict(r, variant=variant, seed=seed) for r in measured['tasks'])
        telemetry.extend(rows)
        if row['valid_performance']:
            snapshots.append(snapshot)
    for snapshot in snapshots[1:]:
        for field in ('code_revision', 'source_sha256', 'software', 'hardware'):
            if snapshot[field] != snapshots[0][field]:
                issues.append(dict(mode=snapshot['mode'], field=field))
    complete = {(r['variant'], r['seed']):r for r in summaries if r.get('valid_performance')} if not issues else {}
    seeds = runner.spec()['seeds']
    aggregates = []
    for variant in sorted({r.get('variant') for r in summaries if r.get('variant')}):
        available = [s for s in seeds if (variant,s) in complete]
        row = dict(variant=variant, completed_seeds=available,
                   complete_planned_seeds=bool(seeds) and available == seeds,
                   complete_three_seeds=len(seeds) == 3 and available == seeds)
        if row['complete_planned_seeds']:
            for key in METRICS:
                values = [complete[variant,s][key] for s in seeds]
                row[key + '_mean' if len(values) > 1 else key] = statistics.fmean(values)
                row[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(row)
    pairs = paired_results(complete, seeds)
    for name, value in (('results', summaries), ('aggregate', aggregates), ('pairs', pairs), ('matching_issues', issues)):
        (directory / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), value)
    write_csv(directory / 'tasks.csv', tasks)
    write_csv(directory / 'tail_updates.csv', telemetry)
    if machine == '3090' and runner.spec()[machine]['dataset'] == 'imgr10':
        historical_original(directory, snapshots)
    draw(directory, complete, pairs, telemetry)


def historical_original(directory, snapshots):
    import run_tail_update as runner
    path = runner.ROOT / runner.spec()['historical_original_3090']
    report = dict(source_directory=str(path), status='unavailable', records=[])
    if path.exists():
        records = json.loads((path / 'queue.json').read_text())
        for record in records:
            if not record['mode'].startswith('imgr10_seed') or record['status'] != 'completed':
                continue
            row, snapshot, _ = read_run(path, record)
            problems = []
            if not row['valid_performance']:
                problems.append('unhealthy_or_incomplete')
            current = snapshots[0] if snapshots else None
            if current:
                for field in ('software', 'hardware'):
                    if snapshot[field] != current[field]:
                        problems.append(field)
            hashes = {p:h for p,h in snapshot['source_sha256'].items() if p.startswith(('models/', 'methods/', 'utils/')) or p in ('main.py','trainer.py')}
            if any(not (runner.ROOT / p).exists() or hashlib.sha256((runner.ROOT / p).read_bytes()).hexdigest() != h for p,h in hashes.items()):
                problems.append('training_source_changed')
            # New telemetry is a protocol change even if output-equivalence tests pass.
            if not snapshot['effective_config'].get('dual_mask_tail_audit', False):
                problems.append('different_diagnostic_protocol')
            row.update(comparison_status='historical_reference_only' if problems else 'verified_reference',
                       matching_issues=problems, log_sha256=hashlib.sha256((path / record['mode'] / 'training.log').read_bytes()).hexdigest())
            report['records'].append(row)
        report['status'] = 'checked_no_automatic_baseline_rerun'
    (directory / 'historical_original.json').write_text(json.dumps(report, indent=2) + '\n')


def draw(directory, complete, pairs, telemetry):
    if not complete:
        for name in ('paired_accuracy', 'old_new', 'intervention_and_gate_change'):
            for extension in ('png', 'pdf'):
                (directory / (name + '.' + extension)).unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
    for pair in pairs:
        values = pair['per_seed']
        ax.plot([r['seed'] for r in values], [r['Average'] for r in values], 'o-', label=pair['comparison'])
    ax.axhline(0, color='grey', lw=.8)
    ax.set(xlabel='Seed / class order', ylabel='Paired Average difference (pp)')
    if pairs:
        ax.legend(fontsize=8)
    fig.savefig(directory / 'paired_accuracy.png', dpi=180)
    fig.savefig(directory / 'paired_accuracy.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    for (variant,seed), row in complete.items():
        ax.scatter(row['StageOld'], row['StageNew'])
        ax.annotate(variant + ':' + str(seed), (row['StageOld'],row['StageNew']), fontsize=7)
    ax.set(xlabel='Task1-9 mean Old (%)', ylabel='Task1-9 mean New (%)')
    fig.savefig(directory / 'old_new.png', dpi=180)
    fig.savefig(directory / 'old_new.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4), constrained_layout=True)
    for (variant,seed) in complete:
        selected = [r for r in telemetry if r['variant'] == variant and r['seed'] == seed]
        if not selected:
            continue
        # Aggregate layers/projections at the same sampled point, never as independent seeds.
        points = sorted({(r['task'],r['epoch'],r['step']) for r in selected})
        for ax, key in zip(axes, ('removed_norm','gate_sample_change')):
            values = [[r[key] for r in selected if (r['task'],r['epoch'],r['step']) == p and r[key] is not None] for p in points]
            ax.plot([i for i,v in enumerate(values) if v], [statistics.fmean(v) for v in values if v], label=variant + ':' + str(seed), lw=.8)
            ax.set(xlabel='Fixed sampled training points', ylabel=key)
    axes[0].legend(fontsize=6, ncol=3)
    fig.savefig(directory / 'intervention_and_gate_change.png', dpi=180)
    fig.savefig(directory / 'intervention_and_gate_change.pdf')
    plt.close(fig)


def summarize_saved(directory):
    import run_tail_update as runner
    manifest = json.loads((directory / 'manifest.json').read_text())
    runner.SPEC = runner.ROOT / manifest.get('sweep_spec','scripts/sweeps/tail_update.json')
    summarize(directory, manifest['machine'], json.loads((directory / 'queue.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summarize_saved(args.directory)
