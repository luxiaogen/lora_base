import argparse
import os
from pathlib import Path

import numpy as np
import torch


def _load_matplotlib():
    mpl_config_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    os.makedirs(mpl_config_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    return plt, Line2D, FancyArrowPatch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _topk_indices(values, k):
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(int(k), flat.size))
    idx = np.argpartition(flat, flat.size - k)[flat.size - k :]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return idx


def _sample_indices(indices, max_points, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= max_points:
        return indices
    return rng.choice(indices, size=max_points, replace=False)


def _coords_from_indices(indices, shape):
    rows, cols = np.unravel_index(indices, shape)
    x = cols.astype(np.float32) / max(shape[1] - 1, 1)
    y = 1.0 - rows.astype(np.float32) / max(shape[0] - 1, 1)
    return x, y


def _block_mean(matrix, out_h=80, out_w=80):
    h, w = matrix.shape
    out_h = min(out_h, h)
    out_w = min(out_w, w)
    row_bins = np.array_split(np.arange(h), out_h)
    col_bins = np.array_split(np.arange(w), out_w)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i, rows in enumerate(row_bins):
        for j, cols in enumerate(col_bins):
            out[i, j] = matrix[np.ix_(rows, cols)].mean()
    return out


def _energy(matrix, mask):
    denom = max(float(mask.sum()), 1.0)
    return float(((matrix ** 2) * mask).sum() / denom)


def _find_default_output(snapshot_path, output):
    if output:
        return Path(output)
    snapshot_path = Path(snapshot_path)
    return snapshot_path.with_suffix(".png")


def plot_snapshot(snapshot_path, output=None, max_points=260, seed=0):
    plt, Line2D, FancyArrowPatch = _load_matplotlib()
    rng = np.random.default_rng(seed)

    payload = torch.load(snapshot_path, map_location="cpu")
    general_mask = _to_numpy(payload["general_mask"])
    isolated_mask = _to_numpy(payload["isolated_mask"])
    conflict_mask = _to_numpy(payload["conflict_mask"])
    raw_delta = _to_numpy(payload["raw_delta"])
    safe_delta = _to_numpy(payload["safe_delta"])

    raw_abs = np.abs(raw_delta)
    safe_abs = np.abs(safe_delta)
    suppressed_abs = np.clip(raw_abs - safe_abs, 0.0, None)
    protected = (general_mask > 0.5).astype(np.float32)
    plastic = (isolated_mask > 0.5).astype(np.float32)
    conflict = (conflict_mask > 0.5).astype(np.float32)
    protected_or_conflict = np.logical_or(protected > 0.5, conflict > 0.5)

    raw_top = _topk_indices(raw_abs, max_points * 3)
    raw_safe_idx = raw_top[~protected_or_conflict.reshape(-1)[raw_top]]
    raw_conflict_idx = raw_top[protected_or_conflict.reshape(-1)[raw_top]]
    raw_safe_idx = _sample_indices(raw_safe_idx, max_points, rng)
    raw_conflict_idx = _sample_indices(raw_conflict_idx, max_points, rng)

    kept_idx = _topk_indices(safe_abs, max_points)
    suppressed_idx = _topk_indices(suppressed_abs, max_points)

    raw_protected = _energy(raw_delta, protected)
    raw_plastic = _energy(raw_delta, plastic)
    safe_protected = _energy(safe_delta, protected)
    safe_plastic = _energy(safe_delta, plastic)
    total_removed = 100.0 * (
        1.0 - float((safe_delta ** 2).sum()) / max(float((raw_delta ** 2).sum()), 1e-12)
    )

    protected_map = _block_mean(protected, 80, 80)
    plastic_map = 1.0 - protected_map

    fig = plt.figure(figsize=(16.6, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], hspace=0.34, wspace=0.20)
    task = payload.get("task", "?")
    layer = payload.get("layer", "?")
    fig.suptitle(
        "Dual-Mask LoRA Snapshot: Task {}, Layer {}".format(task, layer),
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    def setup_space(ax, title):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("input hidden dimension", fontsize=9)
        ax.set_ylabel("qkv output dimension", fontsize=9)
        for yy, label in [(1.0 / 3.0, "K"), (2.0 / 3.0, "V")]:
            ax.axhline(yy, color="#BBBBBB", linewidth=0.8, linestyle="--")
        ax.text(0.015, 0.17, "Q", fontsize=9, color="#666666")
        ax.text(0.015, 0.50, "K", fontsize=9, color="#666666")
        ax.text(0.015, 0.84, "V", fontsize=9, color="#666666")

    def draw_partition(ax):
        ax.imshow(
            plastic_map,
            cmap="Blues",
            vmin=0,
            vmax=1,
            alpha=0.18,
            extent=[0, 1, 0, 1],
            origin="upper",
            interpolation="nearest",
        )
        ax.imshow(
            protected_map,
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.38,
            extent=[0, 1, 0, 1],
            origin="upper",
            interpolation="nearest",
        )
        ax.contour(
            protected_map,
            levels=[0.5],
            colors=["#B00020"],
            linewidths=1.1,
            extent=[0, 1, 0, 1],
            origin="upper",
        )

    ax1 = fig.add_subplot(gs[0, 0])
    setup_space(ax1, "1) W0 space partition\nred = protected, blue = plastic")
    draw_partition(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    setup_space(ax2, "2) Raw LoRA update before mask\nred = overlap/conflict")
    draw_partition(ax2)
    x_safe, y_safe = _coords_from_indices(raw_safe_idx, raw_delta.shape)
    x_conf, y_conf = _coords_from_indices(raw_conflict_idx, raw_delta.shape)
    ax2.scatter(x_safe, y_safe, s=18, color="#1F77B4", alpha=0.62)
    ax2.scatter(x_conf, y_conf, s=22, color="#D62728", alpha=0.78)

    ax3 = fig.add_subplot(gs[0, 2])
    setup_space(ax3, "3) After dual mask\nremaining safe update")
    draw_partition(ax3)
    x_keep, y_keep = _coords_from_indices(kept_idx, safe_delta.shape)
    ax3.scatter(x_keep, y_keep, s=18, color="#1F77B4", alpha=0.72)

    ax4 = fig.add_subplot(gs[0, 3])
    setup_space(ax4, "4) Suppressed update\nremoved before merge")
    draw_partition(ax4)
    x_sup, y_sup = _coords_from_indices(suppressed_idx, safe_delta.shape)
    ax4.scatter(x_sup, y_sup, s=23, color="#D62728", alpha=0.78)
    for x, y in zip(x_sup[:: max(1, len(x_sup) // 24)], y_sup[:: max(1, len(y_sup) // 24)]):
        ax4.plot([x - 0.012, x + 0.012], [y - 0.012, y + 0.012], color="#7A0015", lw=0.9)
        ax4.plot([x - 0.012, x + 0.012], [y + 0.012, y - 0.012], color="#7A0015", lw=0.9)

    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.axis("off")
    ax5.set_title("5) Weight combination rule", fontsize=11, fontweight="bold", pad=8)
    flow = [
        ("raw\nBA", "raw LoRA"),
        ("protect\ngate", "1 - M_W0"),
        ("conflict\ngate", "1 - M_conf"),
        ("safe\ndelta", "masked BA"),
        ("merge", "W = W + safe delta"),
    ]
    xs = [0.08, 0.29, 0.50, 0.71, 0.91]
    for i, (sym, name) in enumerate(flow):
        box_w = 0.125 if i < 4 else 0.17
        ax5.text(
            xs[i],
            0.62,
            sym,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            transform=ax5.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB"),
        )
        ax5.text(
            xs[i],
            0.33,
            name,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#555555",
            transform=ax5.transAxes,
        )
        if i < len(flow) - 1:
            ax5.add_patch(
                FancyArrowPatch(
                    (xs[i] + box_w / 2 + 0.025, 0.62),
                    (xs[i + 1] - box_w / 2 - 0.025, 0.62),
                    transform=ax5.transAxes,
                    arrowstyle="->",
                    mutation_scale=13,
                    linewidth=1.4,
                    color="#444444",
                )
            )
    ax5.text(
        0.50,
        0.08,
        "safe delta = BA * (1 - M_W0) * (1 - M_conf)",
        ha="center",
        fontsize=10,
        color="#333333",
        transform=ax5.transAxes,
    )

    ax6 = fig.add_subplot(gs[1, 2])
    labels = ["protected\nregion", "plastic\nregion"]
    before = [raw_protected, raw_plastic]
    after = [safe_protected, safe_plastic]
    scale = max(max(before), max(after), 1e-12)
    before = [value / scale for value in before]
    after = [value / scale for value in after]
    xpos = np.arange(2)
    width = 0.34
    ax6.bar(xpos - width / 2, before, width, color="#4C78A8", label="before mask")
    ax6.bar(xpos + width / 2, after, width, color="#F58518", label="after mask")
    ax6.set_xticks(xpos)
    ax6.set_xticklabels(labels)
    ax6.set_ylim(0, 1.15)
    ax6.set_ylabel("relative update energy")
    ax6.set_title("6) Clear before/after contrast", fontsize=11, fontweight="bold")
    ax6.legend(frameon=False, fontsize=9)
    ax6.grid(axis="y", alpha=0.25)
    for i, (b, a) in enumerate(zip(before, after)):
        ax6.text(i - width / 2, b + 0.04, "{:.2f}".format(b), ha="center", fontsize=9)
        ax6.text(i + width / 2, a + 0.04, "{:.2f}".format(a), ha="center", fontsize=9)

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis("off")
    legend_items = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#FF6B6B", alpha=0.45, markersize=12, label="protected W0 region"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#EAF5FF", markersize=12, label="plastic region"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1F77B4", markersize=8, label="kept LoRA update"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D62728", markersize=8, label="suppressed conflict"),
    ]
    ax7.legend(handles=legend_items, loc="upper left", frameon=False, fontsize=10)
    message = (
        "\nKey message:\n"
        "LoRA can still learn in plastic regions,\n"
        "but updates overlapping important W0\n"
        "entries are removed before merging.\n\n"
        "Total removed: {:.1f}%\n\n"
        "Final merge:\n"
        "W_new = W_old + safe_delta"
    ).format(total_removed)
    ax7.text(
        0.02,
        0.62,
        message,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F8F8F8", edgecolor="#CCCCCC"),
    )

    fig.text(
        0.5,
        0.025,
        "Generated from a saved dual-mask snapshot before LoRA B weights are cleared.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    out_path = _find_default_output(snapshot_path, output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def iter_snapshots(path):
    path = Path(path)
    if path.is_file():
        yield path
        return
    for snapshot in sorted(path.rglob("layer_*.pt")):
        yield snapshot


def main():
    parser = argparse.ArgumentParser(description="Plot dual-mask LoRA visualization snapshots.")
    parser.add_argument("--snapshot", required=True, help="A .pt snapshot file or directory.")
    parser.add_argument("--output", default=None, help="Output png path. Only used for a single file.")
    parser.add_argument("--output_dir", default=None, help="Output directory when --snapshot is a directory.")
    parser.add_argument("--max_points", type=int, default=260)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    snapshots = list(iter_snapshots(args.snapshot))
    if not snapshots:
        raise FileNotFoundError("No snapshot files found under {}".format(args.snapshot))

    if len(snapshots) == 1 and args.output_dir is None:
        out = plot_snapshot(
            snapshots[0],
            output=args.output,
            max_points=args.max_points,
            seed=args.seed,
        )
        print(out)
        return

    output_dir = Path(args.output_dir or "visualizations/dual_mask_figures")
    for snapshot in snapshots:
        name = "{}_{}.png".format(snapshot.parent.name, snapshot.stem)
        out = output_dir / name
        plot_snapshot(snapshot, output=out, max_points=args.max_points, seed=args.seed)
        print(out)


if __name__ == "__main__":
    main()
