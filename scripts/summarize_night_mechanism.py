"""Summarize completed per-run logs; no GPU, dataset, or retraining required."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re


def parse_log(path):
    text = path.read_text(errors="replace")
    def last(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None
    params = [tuple(map(int, row)) for row in re.findall(
        r"LoRA-stage trainable scalars: LoRA=(\d+), classifier=(\d+), total=(\d+)", text)]
    ca = list(map(int, re.findall(r"CA-stage optimized classifier scalars: (\d+)", text)))
    curves = re.findall(r"CNN top1 curve: (\[[^\n]+\])", text)
    curve = ast.literal_eval(curves[-1]) if curves else []
    grouped = re.findall(r"=> CNN: (\{[^\n]+\})", text)
    grouped = ast.literal_eval(grouped[-1]) if grouped else {}
    records = []
    budgets = []
    for line in text.splitlines():
        if "UpdateOverlap {" in line:
            records.append(json.loads(line.split("UpdateOverlap ", 1)[1]))
        if "AppliedConflictBudget {" in line:
            budgets.append(json.loads(line.split("AppliedConflictBudget ", 1)[1]))
    row = {"log": str(path), "tasks": len(curve),
           "complete": len(curve) == 10 and last(r"(?<!NCM )Average Accuracy: ([\d.]+)") is not None,
           "average": last(r"(?<!NCM )Average Accuracy: ([\d.]+)"),
           "last": last(r"(?<!NCM )Last Accuracy: ([\d.]+)"),
           "old": grouped.get("old"), "new": grouped.get("new"),
           "forgetting": last(r"Forgetting: ([\d.]+)"),
           "seconds": last(r"Total experiment time: ([\d.]+)s"),
           "diagnostic_seconds": sum(r.get("diagnostic_seconds", 0) for r in records),
           "task0_trainable": params[0][2] if params else None,
           "incremental_trainable_min": min((p[2] for p in params[1:]), default=None),
           "incremental_trainable_max": max((p[2] for p in params[1:]), default=None),
           "ca_classifier_min": min(ca, default=None), "ca_classifier_max": max(ca, default=None)}
    return row, records, budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, overlap, budgets = [], [], []
    for path in args.logs:
        row, records, masks = parse_log(path)
        rows.append(row)
        overlap.extend({"log": str(path), **r} for r in records)
        budgets.extend({"log": str(path), **r} for r in masks)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, records in (("overlap", overlap), ("applied_budgets", budgets)):
        with (args.output / (name + ".jsonl")).open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))
    print("Saved summary.csv, overlap.jsonl, applied_budgets.jsonl to", args.output)


if __name__ == "__main__":
    main()
