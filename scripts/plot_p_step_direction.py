"""Export paired-run metrics and sampled-step diagnostics from original logs.

No inferred error bars. First-order gains are predictions with a frozen mask,
not measured validation gains, and mask-switch norm is not old-task damage.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_logs(paths):
    runs = []
    for path in paths:
        run = None
        for line in Path(path).read_text(errors='replace').splitlines():
            if '=> overrides: ' in line:
                settings = {}
                for item in ast.literal_eval(line.split('=> overrides: ', 1)[1]):
                    key, value = item.split('=', 1)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    settings[key] = value
                prefix = settings.get('prefix', '')
                run = dict(source=str(Path(path).resolve()), settings=settings,
                           label=settings.get('p_step_direction', prefix),
                           stages=[], steps=[], counts=[], probes=[], seconds=0., forgetting=None,
                           reported_average=None, reported_last=None, errors=[])
                runs.append(run)
            if run is None:
                continue
            if '=> CNN: {' in line:
                run['stages'].append(ast.literal_eval(line.split('=> CNN: ', 1)[1]))
            if 'PStepDirection {' in line:
                run['steps'].append(json.loads(line.split('PStepDirection ', 1)[1]))
            if 'PStepSummary {' in line:
                run['counts'].append(ast.literal_eval(line.split('PStepSummary ', 1)[1]))
            if 'PStepProbe {' in line:
                run['probes'].append(ast.literal_eval(line.split('PStepProbe ', 1)[1]))
            if '=> Time:' in line:
                run['seconds'] += float(line.split('=> Time:', 1)[1])
            if '=> Forgetting:' in line:
                run['forgetting'] = float(line.split('=> Forgetting:', 1)[1].split()[0])
            if '=> Average Accuracy:' in line:
                run['reported_average'] = float(line.split('=> Average Accuracy:', 1)[1])
            if '=> Last Accuracy:' in line:
                run['reported_last'] = float(line.split('=> Last Accuracy:', 1)[1])
            if 'Traceback (most recent call last)' in line or 'CUDA out of memory' in line:
                run['errors'].append(line)
    return runs


def summary(run):
    stages = run['stages']
    counts = run['counts']
    expected = run['settings'].get('max_tasks', 10)
    complete = (len(stages) == expected and run['reported_last'] is not None and not run['errors'])
    row = dict(source=run['source'], prefix=run['settings'].get('prefix'), mode=run['label'],
               seed=str(run['settings'].get('seed')), tasks=len(stages), expected_tasks=expected,
               protocol='T10' if expected == 10 and run['settings'].get('epochs', 20) == 20 else 'short-run',
               complete=complete, average=run['reported_average'], last=run['reported_last'],
               task0=stages[0]['total'] if stages else None,
               old=stages[-1]['old'] if stages else None, new=stages[-1]['new'] if stages else None,
               old_mean_t1_on=float(np.mean([s['old'] for s in stages[1:]])) if len(stages) > 1 else None,
               new_mean_t1_on=float(np.mean([s['new'] for s in stages[1:]])) if len(stages) > 1 else None,
               forgetting=run['forgetting'], minutes=run['seconds'] / 60,
               sampled_layer_steps=sum(s['layers'] for s in counts),
               norm_accepted=sum(s['norm_accepted'] for s in counts),
               conflict_accepted=sum(s['conflict_accepted'] for s in counts),
               applied=sum(s['applied'] for s in counts))
    return row


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def render(runs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    summaries = [summary(run) for run in runs]
    write_csv(directory / 'summary.csv', summaries)
    write_csv(directory / 'steps.csv', [dict(source=r['source'], run_label=r['label'], **s)
                                       for r in runs for s in r['steps']])
    write_csv(directory / 'curves.csv', [dict(source=r['source'], run_label=r['label'], task=t,
                                             total=s['total'], old=s['old'], new=s['new'])
                                        for r in runs for t, s in enumerate(r['stages'])])
    probes = [dict(source=r['source'], prefix=r['settings'].get('prefix'), run_label=r['label'], task=p['task'], epoch=p['epoch'],
                   proposal=name, loss=m['loss'], accuracy=m['accuracy'], margin=m['margin'],
                   loss_delta=m['loss'] - p['metrics']['reference']['loss'])
              for r in runs for p in r['probes'] for name, m in p['metrics'].items()]
    write_csv(directory / 'training_batch_probes.csv', probes)
    # An outer launcher log contains short smoke runs BEFORE the full queue.
    # Never mix their trajectories or step telemetry into full-run figures.
    full = [r for r in runs if summary(r)['protocol'] == 'T10']
    plotted_runs = full if full else runs
    comparable = [r for r in plotted_runs if summary(r)['complete']]
    if comparable:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
        for run in comparable:
            stages, label = run['stages'], run['label']
            axes[0].plot(range(len(stages)), [s['total'] for s in stages], marker='o', ms=3, label=label)
            axes[1].plot(range(1, len(stages)), [s['old'] for s in stages[1:]], marker='o', ms=3, label=label)
            axes[2].plot(range(1, len(stages)), [s['new'] for s in stages[1:]], marker='o', ms=3, label=label)
        for ax, title in zip(axes, ('All seen classes', 'Old classes', 'New classes')):
            ax.set(xlabel='Task', ylabel='Accuracy (%)', title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(('T10' if full else 'Short-run diagnostic') + '; one seed; no uncertainty estimate')
        save(fig, directory, 'accuracy_curves')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for run in comparable:
            row = summary(run)
            for ax, x, y in ((axes[0], row['new_mean_t1_on'], row['old_mean_t1_on']),
                             (axes[1], row['new'], row['old'])):
                ax.scatter(x, y, s=55, label=run['label'])
        for ax, title in zip(axes, ('Task1 onward: mean Old/New', 'Final task: Old/New')):
            ax.set(xlabel='New accuracy (%)', ylabel='Old accuracy (%)', title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        save(fig, directory, 'old_new')

    diagnostic_runs = [r for r in plotted_runs if r['steps']]
    if diagnostic_runs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for run in diagnostic_runs:
            for proposal in ('norm', 'conflict'):
                rows = [s for s in run['steps'] if s['proposal'] == proposal and s['accepted']
                        and s['reference_norm'] > 1e-12 and s['reference_conflict_norm'] > 1e-12]
                if rows:
                    axes[0].scatter([s['effective_norm'] / s['reference_norm'] for s in rows],
                                    [s['conflict_norm'] / s['reference_conflict_norm'] for s in rows],
                                    s=8, alpha=.3, label=run['label'] + '/' + proposal)
            mode = run['settings'].get('p_step_direction')
            proposal = 'reference' if mode == 'baseline' else mode
            rows = [s for s in run['steps'] if s['proposal'] == proposal and s['effective_norm'] > 1e-12]
            axes[1].scatter([s['mask_jaccard'] for s in rows],
                            [s['switch_norm'] / s['effective_norm'] for s in rows],
                            s=8, alpha=.3, label=run['label'])
        axes[0].axhline(1, color='black', lw=.8, ls='--')
        axes[0].set(xlabel='Effective step norm / original SGD norm', ylabel='Conflict norm / original SGD conflict norm',
                    title='Accepted proposals (shadow and applied)')
        axes[1].set(xlabel='Before/after mask Jaccard', ylabel='Switch norm / effective step norm',
                    title='Mask changes; ratio may exceed 1')
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        save(fig, directory, 'step_diagnostics')

        fig, axes = plt.subplots(1, len(diagnostic_runs), figsize=(4 * len(diagnostic_runs), 5),
                                 squeeze=False, layout='constrained')
        grids = []
        for run in diagnostic_runs:
            mode = run['settings'].get('p_step_direction')
            proposal = 'reference' if mode == 'baseline' else mode
            rows = [s for s in run['steps'] if s['proposal'] == proposal and s['effective_norm'] > 1e-12]
            grid = np.full((12, 3), np.nan)
            for layer in range(12):
                for j, projection in enumerate(('Q', 'K', 'V')):
                    values = [s['switch_norm'] / s['effective_norm'] for s in rows
                              if s['layer'] == layer and s['projection'] == projection]
                    if values:
                        grid[layer, j] = np.log10(max(float(np.mean(values)), 1e-8))
            grids.append(grid)
        finite = np.concatenate([g[np.isfinite(g)] for g in grids])
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (-8., 0.)
        for ax, run, grid in zip(axes[0], diagnostic_runs, grids):
            im = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', vmin=vmin, vmax=max(vmin + .01, vmax), cmap='viridis')
            ax.set(xticks=range(3), xticklabels=['Q', 'K', 'V'], yticks=range(12), ylabel='Layer', title=run['label'])
        fig.colorbar(im, ax=axes.ravel().tolist(), label='log10(mean switch/effective norm); missing cells blank')
        save(fig, directory, 'mask_switch_heatmap')

    if probes:
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        for run in plotted_runs:
            for name in ('norm', 'conflict'):
                rows = [p for p in probes if p['source'] == run['source'] and p['prefix'] == run['settings'].get('prefix') and p['proposal'] == name]
                ax.scatter([p['task'] + p['epoch'] / 25 for p in rows], [p['loss_delta'] for p in rows],
                           s=16, alpha=.6, label=run['label'] + '/' + name)
        ax.axhline(0, color='black', ls='--', lw=.8)
        ax.set(xlabel='Task + epoch/25', ylabel='Loss(candidate) - loss(original SGD)',
               title='Actual same-training-batch probe; not validation; lower is better')
        ax.legend(fontsize=7)
        save(fig, directory, 'training_batch_probe')

    manifest = dict(sources=[r['source'] for r in runs], runs=summaries,
                    plotted_prefixes=[r['settings'].get('prefix') for r in plotted_runs],
                    figures=[p.name for p in directory.glob('*.pdf')],
                    interpretation='Single seed, no error bars. Predicted gain assumes a fixed gate. '
                    'Switch norm is not evidence of old-task damage. Compare runs on the same machine. '
                    'Norm matching tolerance: 0.98 to 1.0001, per projection. Incomplete runs excluded from accuracy figures.')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', nargs='+')
    args = parser.parse_args()
    runs = parse_logs(args.logs)
    groups = {r['settings'].get('wandb_group') for r in runs}
    assert len(groups) <= 1, 'Render each machine separately; wandb groups differ'
    if args.labels:
        assert len(args.labels) == len(runs)
        for run, label in zip(runs, args.labels):
            run['label'] = label
    render(runs, args.out)
    for run in runs:
        print(json.dumps(summary(run)))
    print('Figures and source CSV:', args.out.resolve())


if __name__ == '__main__':
    main()
