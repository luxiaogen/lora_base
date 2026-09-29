"""Rebuild the coverage figure from raw logs; no cross-GPU pooling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract(path):
    groups = {}
    coverage = None
    pattern = re.compile(r'Task (\d+) layer (\d+) P applied merge diagnostic:.*?removed_ratio=([\d.]+)')
    for line in path.read_text().splitlines():
        setting = re.search(r'=> dual_mask_conflict_ratio: ([\d.]+)', line)
        if setting:
            coverage = float(setting[1])
        match = pattern.search(line)
        if match and int(match[1]) > 0:
            groups.setdefault(coverage, {})[(int(match[1]), int(match[2]))] = float(match[3])
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs=2, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.7), layout='constrained')
    for path, machine, color, marker in zip(args.logs, ('3090', '5090'), ('#2878B5', '#D97928'), ('o', 's')):
        groups = extract(path)
        xs, ys = [], []
        for coverage, values in sorted(groups.items()):
            if len(values) != 108:
                raise ValueError(f'{path}: coverage={coverage} has {len(values)}/108 task-layer entries')
            mean = sum(values.values()) / len(values)
            xs.append(coverage * 100)
            ys.append(mean * 100)
            for (task, layer), value in values.items():
                rows.append((machine, coverage, task, layer, value, str(path)))
        ax.plot(xs, ys, marker=marker, color=color, label=machine, linewidth=1.8, markersize=7)
    ax.axhline(50, color='#999999', linestyle='--', linewidth=1, label='Uniform attenuation at beta = 0.5')
    ax.annotate('10% selected coordinates\n~42.3% removed-update norm ratio', xy=(10, 42.29),
                xytext=(27, 27), fontsize=10, arrowprops={'arrowstyle': '->', 'color': '#555555'})
    ax.set(xlabel='Selected conflict coordinates (% of full QKV matrix)',
           ylabel='Removed P-update norm ratio (%)',
           xlim=(-3, 103), ylim=(-2, 57), title='Coverage is not suppression strength')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.supxlabel('Denominator: P-update norm after plastic gating, before conflict gating.\nImageNet-R T10, seed 1993; Task1–9 × 12-layer mean, GPUs kept separate.\nLines are guides, not extra measurements.', fontsize=8)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.out / f'coverage_suppression.{ext}', dpi=220)
    plt.close(fig)
    with (args.out / 'coverage_source.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(('gpu', 'coverage', 'task', 'layer', 'removed_ratio', 'source'))
        writer.writerows(rows)
    manifest = {'status': 'render_only', 'strategy': 'raw_data', 'representation': 'semantic_vector',
                'statistic': 'equal-weight mean over 108 task-layer entries; no independent-seed uncertainty',
                'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.logs],
                'limitation': 'SciPlot runner absent locally; source log ratios rounded to four decimals',
                'visualspec': {'schema': 'scientificfigure.visualspec.v2', 'x': 'coverage * 100',
                               'y': 'mean(removed_ratio) * 100', 'series': 'gpu', 'uncertainty': None}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
