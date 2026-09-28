#!/usr/bin/env python3
"""CPU-only diagnostics from cached real features (not a training method)."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.feature_geometry import analyze_geometry, normalize, top_confused_pairs


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def save_figure(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180)
    fig.savefig(out / f'{name}.pdf')
    plt.close(fig)


def report(cache, out, pairs=3, per_class=100, seed=1993):
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((cache / 'metadata.json').read_text())
    with np.load(cache / 'features.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    r = analyze_geometry(arrays['train_features'], arrays['train_labels'],
                         arrays['test_features'], arrays['test_labels'], arrays['heads'])
    labels = arrays['test_labels']
    n_classes = len(arrays['heads'])
    counts = np.bincount(labels, minlength=n_classes)
    boundaries = np.cumsum(metadata['increments'])
    tasks = np.searchsorted(boundaries, np.arange(n_classes), side='right')
    old_count = sum(metadata['increments'][:metadata['task']])
    wrong = r['prediction'] != labels
    summary = dict(accuracy=r['accuracy'], checkpoint_last_accuracy=metadata['checkpoint_last_accuracy'],
                   accuracy_difference_pp=r['accuracy'] - metadata['checkpoint_last_accuracy'],
                   train_samples=len(arrays['train_labels']), test_samples=len(labels),
                   new_to_old=int(np.sum(wrong & (labels >= old_count) & (r['prediction'] < old_count))),
                   old_to_new=int(np.sum(wrong & (labels < old_count) & (r['prediction'] >= old_count))),
                   nearest_train_center_accuracy=float(100 * np.mean(r['nearest_center_prediction'] == labels)),
                   mean_margin=float(r['margin'].mean()), tsne_seed=seed,
                   plotting_versions=dict(numpy=np.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__),
                   checkpoint=metadata['checkpoint'], checkpoint_sha256=metadata['checkpoint_sha256'],
                   note='Final post-CA checkpoint only. Old train images used offline for diagnosis; not a replay method. Test-selected pairs are descriptive, not validation.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_csv(out / 'confusion_counts.csv', ['true_class'] + list(range(n_classes)),
              ([c] + r['confusion'][c].tolist() for c in range(n_classes)))
    write_csv(out / 'classes.csv', ['class', 'original_class', 'task', 'n_train', 'n_test',
              'accuracy_pct', 'dispersion_rms_chord', 'nearest_center', 'nearest_center_chord',
              'separation_over_dispersion', 'own_head_cosine', 'competitor_head',
              'competitor_head_cosine', 'center_head_margin', 'test_mean_margin'],
              ([c, metadata['class_order'][c], tasks[c], r['train_counts'][c], counts[c],
                100 * r['confusion'][c, c] / counts[c] if counts[c] else np.nan,
                r['dispersion'][c], r['nearest_center'][c], r['nearest_center_distance'][c],
                r['nearest_center_distance'][c] / max(r['dispersion'][c], 1e-12),
                r['own_head_cosine'][c], r['competitor_head'][c], r['competitor_head_cosine'][c],
                r['own_head_cosine'][c] - r['competitor_head_cosine'][c],
                r['margin'][labels == c].mean() if counts[c] else np.nan] for c in range(n_classes)))
    write_csv(out / 'samples.csv', ['test_index', 'true_class', 'predicted_class', 'margin', 'nearest_train_center'],
              zip(arrays['test_ids'], labels, r['prediction'], r['margin'], r['nearest_center_prediction']))
    np.savez_compressed(out / 'geometry.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    rates = r['confusion'] / np.maximum(counts[:, None], 1) * 100
    im = ax.imshow(rates, vmin=0, vmax=100, cmap='Blues', interpolation='nearest')
    for b in boundaries[boundaries < n_classes]:
        ax.axhline(b - .5, color='#999999', lw=.5)
        ax.axvline(b - .5, color='#999999', lw=.5)
    ticks = np.r_[0, boundaries[boundaries < n_classes]]
    ax.set(xticks=ticks, yticks=ticks, xlabel='Predicted class (incremental ID)',
           ylabel='True class (incremental ID)', title=f'All-seen confusion | accuracy {r["accuracy"]:.2f}%')
    fig.colorbar(im, ax=ax, label='Within-true-class percentage (%)')
    save_figure(fig, out, '01_confusion')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    im = axes[0].imshow(r['center_similarity'], vmin=-1, vmax=1, cmap='coolwarm')
    axes[0].set(xlabel='Class', ylabel='Class',
                xticks=ticks, yticks=ticks,
                title=f'Train-center cosine similarity ({arrays["heads"].shape[1]}-D space)')
    fig.colorbar(im, ax=axes[0], shrink=.8, label='Cosine similarity')
    x, y = r['own_head_cosine'], r['competitor_head_cosine']
    scatter = axes[1].scatter(x, y, c=tasks, cmap='viridis', s=22, alpha=.8)
    finite = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
    limits = [finite.min() - .02, finite.max() + .02]
    axes[1].plot(limits, limits, '--', color='#555555', lw=1)
    axes[1].set(xlim=limits, ylim=limits, xlabel='Center vs its own classifier head (cosine)',
                ylabel='Center vs strongest other head (cosine)',
                title='Above diagonal: center favors another head')
    fig.colorbar(scatter, ax=axes[1], shrink=.8, ticks=np.unique(tasks), label='Task ID')
    save_figure(fig, out, '03_center_head_alignment')

    selected = top_confused_pairs(r['confusion'], pairs)
    # Pair selection is fixed by symmetric error count, ties resolved by class IDs.
    write_csv(out / 'selected_pairs.csv', ['class_a', 'class_b', 'a_to_b', 'b_to_a'],
              ((a, b, r['confusion'][a, b], r['confusion'][b, a]) for a, b in selected))
    tsne_rows = []
    if selected:
        fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3.4 * len(selected)),
                                 squeeze=False, layout='constrained')
        for row, (a, b) in enumerate(selected):
            rng = np.random.default_rng(seed)
            indices = np.concatenate([np.sort(rng.choice(np.flatnonzero(labels == c),
                         min(per_class, counts[c]), replace=False)) for c in (a, b)])
            features = normalize(arrays['test_features'][indices])
            features = PCA(n_components=min(50, len(features) - 1, features.shape[1]),
                           svd_solver='full').fit_transform(features)
            for col, requested in enumerate((10, 30)):
                perplexity = min(requested, max(1, (len(indices) - 1) / 3))
                # n_iter defaults to1000 across both older/newer supported sklearn versions.
                xy = TSNE(n_components=2, perplexity=perplexity, init='pca',
                          learning_rate='auto', random_state=seed).fit_transform(features)
                ax = axes[row, col]
                for c, color in ((a, '#0072B2'), (b, '#D55E00')):
                    chosen = labels[indices] == c
                    ax.scatter(*xy[chosen].T, s=18, color=color, alpha=.7,
                               label=f'Class {c} (T{tasks[c]})')
                error = wrong[indices]
                ax.scatter(*xy[error].T, s=35, marker='x', color='black', lw=.9, label='Wrong (all-seen)')
                ax.set(title=f'{a} vs {b} | perplexity={perplexity:g} | n={len(indices)}',
                       xticks=[], yticks=[], xlabel='t-SNE 1', ylabel='t-SNE 2')
                ax.legend(fontsize=8, loc='best')
                tsne_rows.extend([a, b, perplexity, int(arrays['test_ids'][idx]), int(labels[idx]),
                                  int(r['prediction'][idx]), float(point[0]), float(point[1])]
                                 for idx, point in zip(indices, xy))
        fig.suptitle('Most confused class pairs | independent projections, not comparable distances', fontsize=11)
        save_figure(fig, out, '02_tsne_pairs')
    write_csv(out / 'tsne_coordinates.csv', ['class_a', 'class_b', 'perplexity', 'test_index',
              'true_class', 'prediction', 'x', 'y'], tsne_rows)
    (out / 'visualspec.json').write_text(json.dumps({
        'schema': 'scientificfigure.visualspec.v2', 'source': str(cache.resolve()),
        'strategy': 'raw_data', 'status': 'render_only',
        'figures': {'01_confusion': 'true class rows, prediction columns; row percentage; fixed0–100',
                    '02_tsne_pairs': 'normalized features -> PCA<=50 -> t-SNE; top symmetric-error pairs; per-class capped random samples',
                    '03_center_head_alignment': 'train normalized-feature center cosine matrix; own vs competitor head cosine'},
        'outputs': ['png', 'pdf', 'csv', 'npz', 'json'],
        'limitations': ['No CA-before weights', 'No causal claim from t-SNE',
                        'Separate pair/perplexity embeddings do not share coordinates',
                        'Skill renderer unavailable; custom matplotlib renderer; real-data visual QA pending']}, indent=2))
    (out / 'README.md').write_text(
        '# Offline feature diagnosis\n\n'
        'No optimization or continual-learning result is produced. Train centers use all seen training images, '
        'including historical ones, solely for offline analysis. Test-selected pairs are descriptive.\n\n'
        'Features and heads are L2-normalized. Class center = normalized mean of normalized train features. '
        'Dispersion = RMS Euclidean chord distance to that center; separation = nearest other center chord distance. '
        'Margin = correct-head cosine minus highest incorrect-head cosine. '
        'Nearest-center accuracy is an offline diagnostic using re-extracted old train data, not an eligible CIL score.\n\n'
        'Check summary.json accuracy against checkpoint accuracy (rounded log values may differ slightly). '
        'If materially different, resolve loading/data/backend before interpreting figures. '
        'Every class is in classes.csv; every test sample is in samples.csv. Sample indices map to sample_paths.json. '
        'No before/after CA claim: this checkpoint is post-CA only. '
        'If there are no confused pairs, the t-SNE figure is omitted. '
        'Do not select methods or hyperparameters from these test visualizations.\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Figures and tables: {out}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', default=None)
    cli = parser.parse_args()
    cache = Path(cli.cache)
    report(cache, Path(cli.out) if cli.out else cache / 'figures')
