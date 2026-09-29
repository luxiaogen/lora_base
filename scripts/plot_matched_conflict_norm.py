"""Plot completed, same-GPU selective/uniform T10 logs without invented results."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_run(path):
    text = path.read_text()
    curves = re.findall(r'=> CNN top1 curve: (\[.*?\])', text)
    curve = ast.literal_eval(curves[-1])
    if len(curve) != 10:
        raise ValueError(f'{path}: incomplete T10 ({len(curve)} tasks)')
    metrics = [ast.literal_eval(v) for v in re.findall(r'=> CNN: (\{.*?\})', text)]
    metrics = metrics[-10:]
    if len(metrics) != 10:
        raise ValueError(f'{path}: expected ten CNN metric records')
    return {'average': sum(curve) / 10, 'last': curve[-1],
            'old': metrics[-1]['old'], 'new': metrics[-1]['new'],
            'stage_old': sum(m['old'] for m in metrics[1:]) / 9,
            'stage_new': sum(m['new'] for m in metrics[1:]) / 9,
            'forgetting': float(re.findall(r'=> Forgetting:\s*([\d.]+)', text)[-1]),
            'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    baseline, candidate = read_run(args.baseline), read_run(args.candidate)
    checks = [json.loads(s.split('MatchedConflictNorm ', 1)[1])
              for s in args.candidate.read_text().splitlines() if 'MatchedConflictNorm {' in s]
    checks = [c for c in checks if c['task'] > 0]
    expected = {(t, layer, branch) for t in range(1, 10) for layer in range(12) for branch in ('S', 'P')}
    keys = [(c['task'], c['layer'], c['branch']) for c in checks]
    if len(checks) != 216 or set(keys) != expected:
        raise ValueError('Expected unique Task1–9 × 12 layers × S/P norm telemetry')
    if not all(math.isfinite(c[k]) for c in checks for k in ('reference_removed_norm', 'actual_removed_norm', 'base_norm')):
        raise ValueError('Non-finite norm telemetry')
    error = max(abs(c['reference_removed_norm'] - c['actual_removed_norm']) /
                max(c['base_norm'], 1e-12) for c in checks)
    if error > 1e-5:
        raise ValueError(f'Norm matching error / base norm: {error}')
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7), layout='constrained')
    for ax, keys, title in zip(axes, [('stage_new', 'stage_old'), ('new', 'old')],
                               ['Mean over Task1–9', 'Final Task9']):
        for result, label, color, marker in [(baseline, 'Selective (baseline)', '#2878B5', 'o'),
                                             (candidate, 'Uniform (norm-matched)', '#D97928', 's')]:
            ax.scatter(result[keys[0]], result[keys[1]], color=color, marker=marker, s=70, label=label)
        ax.set(xlabel='New accuracy (%)', ylabel='Old accuracy (%)', title=title)
        ax.margins(.35)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Does selection help beyond reducing update magnitude?')
    fig.supxlabel(f'3090, ImageNet-R T10, seed 1993; single runs, no significance claim.\n'
                  f'Selective: Avg {baseline["average"]:.3f}, Last {baseline["last"]:.2f}, F {baseline["forgetting"]:.3f} | '
                  f'Uniform: Avg {candidate["average"]:.3f}, Last {candidate["last"]:.2f}, F {candidate["forgetting"]:.3f}', fontsize=9)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.out / f'matched_norm_old_new.{ext}', dpi=220)
    (args.out / 'matched_norm_source.json').write_text(json.dumps({
        'baseline': baseline, 'candidate': candidate, 'max_norm_error_over_base': error,
        'status': 'render_only', 'matching': 'counterfactual selective removal on the same current update; not identical trajectories',
        'visualspec': {'schema': 'scientificfigure.visualspec.v2', 'x': 'new', 'y': 'old', 'uncertainty': None}
    }, indent=2))


if __name__ == '__main__':
    main()
