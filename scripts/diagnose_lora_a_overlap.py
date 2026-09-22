#!/usr/bin/env python3
"""Measure finite-dimensional overlap between independently initialized LoRA A matrices."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


METRICS = (
    "raw_cross_fro",
    "unit_cross_fro",
    "unit_cross_rms",
    "mean_abs_row_cosine",
    "max_abs_row_cosine",
    "subspace_overlap",
    "max_canonical_cosine",
)


def make_lora_a(init_name, tasks, rank, din, generator):
    """Match the repository's Kaiming or QR initialization with A shaped [task, rank, din]."""
    if init_name == "kaiming":
        matrices = torch.empty(tasks, rank, din)
        # The repository initializes every [rank, din] task matrix separately.
        # For a=sqrt(5), Kaiming uniform is exactly U(-1/sqrt(din), 1/sqrt(din)).
        # Initializing the stacked 3-D tensor with torch.nn.init directly would
        # incorrectly use rank * din as fan_in.
        bound = 1.0 / math.sqrt(din)
        matrices.uniform_(-bound, bound, generator=generator)
        return matrices
    if init_name == "qr":
        random_matrix = torch.randn(tasks, din, rank, generator=generator)
        q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        return q.transpose(1, 2).contiguous()
    raise ValueError("init_name must be kaiming or qr")


def pairwise_overlap(matrices, include_spectral=True):
    """Return pair metrics for all distinct task pairs in one layer and branch."""
    tasks, rank, _ = matrices.shape
    if tasks < 2:
        raise ValueError("at least two task matrices are required")

    row_unit = F.normalize(matrices.float(), p=2, dim=-1)
    row_cross = torch.einsum("trd,usd->turs", row_unit, row_unit)
    raw_cross = torch.einsum("trd,usd->turs", matrices.float(), matrices.float())

    q, _ = torch.linalg.qr(matrices.float().transpose(1, 2), mode="reduced")
    canonical = torch.einsum("tdr,uds->turs", q, q)

    pair_i, pair_j = torch.triu_indices(tasks, tasks, offset=1)
    row_pairs = row_cross[pair_i, pair_j]
    raw_pairs = raw_cross[pair_i, pair_j]
    canonical_pairs = canonical[pair_i, pair_j]

    result = {
        "raw_cross_fro": raw_pairs.square().sum(dim=(-2, -1)).sqrt(),
        "unit_cross_fro": row_pairs.square().sum(dim=(-2, -1)).sqrt(),
        "unit_cross_rms": row_pairs.square().mean(dim=(-2, -1)).sqrt(),
        "mean_abs_row_cosine": row_pairs.abs().mean(dim=(-2, -1)),
        "max_abs_row_cosine": row_pairs.abs().amax(dim=(-2, -1)),
        "subspace_overlap": canonical_pairs.square().sum(dim=(-2, -1)) / rank,
    }
    if include_spectral:
        result["max_canonical_cosine"] = torch.linalg.svdvals(canonical_pairs)[..., 0]
    else:
        result["max_canonical_cosine"] = torch.full_like(result["subspace_overlap"], float("nan"))
    return result


def within_matrix_overlap(matrices):
    row_unit = F.normalize(matrices.float(), p=2, dim=-1)
    gram = row_unit @ row_unit.transpose(-1, -2)
    rank = matrices.shape[1]
    off_diagonal = ~torch.eye(rank, dtype=torch.bool).unsqueeze(0)
    values = gram.masked_select(off_diagonal).reshape(matrices.shape[0], -1).abs()
    return {
        "intra_mean_abs_row_cosine": values.mean(dim=1),
        "intra_max_abs_row_cosine": values.max(dim=1).values,
    }


def summarize(values):
    values = torch.cat([value.detach().float().flatten().cpu() for value in values])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def evaluate_dimension(init_name, din, rank, tasks, layers, branches, seeds, include_spectral):
    pair_values = {name: [] for name in METRICS}
    intra_values = {
        "intra_mean_abs_row_cosine": [],
        "intra_max_abs_row_cosine": [],
    }
    by_branch = {
        branch: {name: [] for name in METRICS}
        for branch in branches
    }

    init_offset = 0 if init_name == "kaiming" else 10_000_000
    for seed in seeds:
        for layer in range(layers):
            for branch_index, branch in enumerate(branches):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(seed) + init_offset + din * 10_000 + layer * 101 + branch_index * 10_003
                )
                matrices = make_lora_a(init_name, tasks, rank, din, generator)
                pair_metrics = pairwise_overlap(matrices, include_spectral=include_spectral)
                intra_metrics = within_matrix_overlap(matrices)
                for name, value in pair_metrics.items():
                    pair_values[name].append(value)
                    by_branch[branch][name].append(value)
                for name, value in intra_metrics.items():
                    intra_values[name].append(value)

    aggregate = {name: summarize(values) for name, values in pair_values.items()}
    aggregate.update({name: summarize(values) for name, values in intra_values.items()})
    return {
        "init": init_name,
        "din": din,
        "rank": rank,
        "tasks": tasks,
        "layers": layers,
        "branches": list(branches),
        "seeds": list(seeds),
        "theory": {
            "row_cosine_rms": 1.0 / math.sqrt(din),
            "mean_abs_row_cosine": math.sqrt(2.0 / (math.pi * din)),
            "random_subspace_overlap": rank / din,
        },
        "aggregate": aggregate,
        "by_branch": {
            branch: {name: summarize(values) for name, values in metrics.items()}
            for branch, metrics in by_branch.items()
        },
    }


def print_report(report, main_din):
    print("# LoRA A finite-dimensional overlap diagnostic")
    print()
    print(
        "Primary metrics: unit_cross_rms approximates the LoRI A_s A_t^T premise; "
        "subspace_overlap is the average squared canonical cosine."
    )
    print()
    print(
        "| init | din | cross RMS | theory 1/sqrt(d) | subspace overlap | theory r/d | "
        "intra-row mean | max row cosine p95 | max canonical p95 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["main"]:
        aggregate = item["aggregate"]
        print(
            "| {init} | {din} | {cross:.4f} | {cross_theory:.4f} | {subspace:.4f} | "
            "{subspace_theory:.4f} | {intra:.4f} | {row_max:.4f} | {canonical:.4f} |".format(
                init=item["init"],
                din=item["din"],
                cross=aggregate["unit_cross_rms"]["mean"],
                cross_theory=item["theory"]["row_cosine_rms"],
                subspace=aggregate["subspace_overlap"]["mean"],
                subspace_theory=item["theory"]["random_subspace_overlap"],
                intra=aggregate["intra_mean_abs_row_cosine"]["mean"],
                row_max=aggregate["max_abs_row_cosine"]["p95"],
                canonical=aggregate["max_canonical_cosine"]["p95"],
            )
        )
    print()
    print("## Dimension sweep (Kaiming)")
    print()
    print("| din | cross RMS | theory | subspace overlap | theory r/d |")
    print("|---:|---:|---:|---:|---:|")
    for item in report["dimension_sweep"]:
        aggregate = item["aggregate"]
        marker = " **" if item["din"] == main_din else ""
        print(
            "| {din}{marker} | {cross:.4f} | {cross_theory:.4f} | {subspace:.4f} | {subspace_theory:.4f} |".format(
                din=item["din"],
                marker=marker,
                cross=aggregate["unit_cross_rms"]["mean"],
                cross_theory=item["theory"]["row_cosine_rms"],
                subspace=aggregate["subspace_overlap"]["mean"],
                subspace_theory=item["theory"]["random_subspace_overlap"],
            )
        )


def build_report(args):
    main = [
        evaluate_dimension(
            init_name=init_name,
            din=args.din,
            rank=args.rank,
            tasks=args.tasks,
            layers=args.layers,
            branches=args.branches,
            seeds=args.seeds,
            include_spectral=True,
        )
        for init_name in ("kaiming", "qr")
    ]
    sweep = [
        evaluate_dimension(
            init_name="kaiming",
            din=din,
            rank=args.rank,
            tasks=args.tasks,
            layers=args.sweep_layers,
            branches=("S",),
            seeds=args.seeds,
            include_spectral=False,
        )
        for din in args.sweep_din
        if din >= args.rank
    ]
    return {
        "metadata": {
            "purpose": "Initialization-only diagnostic; no dataset, training, or test labels used.",
            "a_shape": [args.rank, args.din],
            "orientation_note": "Repository A is [rank, din]; paper A is [din, rank].",
            "task0_note": "Task0 S-A is trainable in DualMask; this report measures initialization only.",
            "initialization_note": (
                "Kaiming samples each repository-shaped [rank, din] task matrix from "
                "U(-1/sqrt(din), 1/sqrt(din)); cosine and subspace metrics are scale invariant."
            ),
        },
        "main": main,
        "dimension_sweep": sweep,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--din", type=int, default=768)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--branches", nargs="+", default=["S", "P"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1993, 1996, 1997])
    parser.add_argument("--sweep-din", nargs="+", type=int, default=[256, 512, 768, 1024, 2048, 4096])
    parser.add_argument("--sweep-layers", type=int, default=3)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rank <= 0 or args.din < args.rank or args.tasks < 2:
        raise ValueError("require din >= rank > 0 and tasks >= 2")
    report = build_report(args)
    print_report(report, args.din)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report: {args.out}")


if __name__ == "__main__":
    main()
