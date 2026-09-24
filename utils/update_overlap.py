"""Read-only, merge-time diagnostics of actual trained QKV updates.

Spectral alignment is the cosine between input-side Gram matrices D.T @ D.
It equals normalized projector overlap for equal-rank, flat-spectrum updates,
but is energy-weighted in general: it is NOT A-row cosine or principal angles.
Only temporary CPU/disk history is kept; no model buffers or GPU history.
"""

import json
import logging
from pathlib import Path
import tempfile
import time

import torch


def cosine(left, right):
    denominator = left.norm() * right.norm()
    if denominator.item() <= 1e-12:
        return None  # A zero update has no direction, not an orthogonal direction.
    return float(((left * right).sum() / denominator).clamp(-1, 1))


def describe(delta):
    delta = delta.detach().to(device="cpu", dtype=torch.float32).clone()
    parts = delta.reshape(3, delta.shape[0] // 3, delta.shape[1])
    grams = parts.transpose(1, 2) @ parts
    trace = grams.diagonal(dim1=-2, dim2=-1).sum(-1)
    energy = grams.square().sum((1, 2))
    effective_ranks = [float(t.square() / e) if e > 1e-24 else None
                       for t, e in zip(trace, energy)]
    return {"delta": parts, "grams": grams, "effective_rank": effective_ranks}


class UpdateOverlapRecorder:
    def __init__(self, directory, layer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.layer = int(layer)
        self.cache = tempfile.TemporaryDirectory(prefix=f"layer{layer}_", dir=directory)
        self.history = {"S": [], "P": []}
        self.output = directory / (Path(self.cache.name).name + ".jsonl")
        logging.info("Update overlap scalars: %s; temporary tensors: %s", self.output, self.cache.name)

    @torch.no_grad()
    def record(self, task, branch, raw, safe):
        started = time.perf_counter()
        current = {"raw": describe(raw), "safe": describe(safe)}
        raw_norm = float(current["raw"]["delta"].norm())
        safe_norm = float(current["safe"]["delta"].norm())
        removed_norm = float((current["raw"]["delta"] - current["safe"]["delta"]).norm())
        common = {"task": int(task), "layer": self.layer, "branch": branch}
        records = [{**common, "kind": "update", "raw_norm": raw_norm,
                    "safe_norm": safe_norm, "removed_norm": removed_norm,
                    "removed_ratio": removed_norm / raw_norm if raw_norm > 1e-12 else None,
                    "raw_effective_rank_qkv": current["raw"]["effective_rank"],
                    "safe_effective_rank_qkv": current["safe"]["effective_rank"]}]
        for old_task, path in self.history[branch]:
            previous = torch.load(path, map_location="cpu", weights_only=True)
            for stage in ("raw", "safe"):
                left, right = previous[stage], current[stage]
                records.append({
                    **common, "kind": "pair", "old_task": old_task, "stage": stage,
                    "cosine": cosine(left["delta"], right["delta"]),
                    "cosine_qkv": [cosine(a, b) for a, b in zip(left["delta"], right["delta"])],
                    "spectral_alignment_qkv": [cosine(a, b) for a, b in zip(left["grams"], right["grams"])],
                })
        path = Path(self.cache.name) / f"{branch}_task{task}.pt"
        torch.save(current, path)
        self.history[branch].append((int(task), path))
        records[0]["diagnostic_seconds"] = time.perf_counter() - started
        with self.output.open("a", encoding="utf-8") as stream:
            for record in records:
                line = json.dumps(record, allow_nan=False)
                stream.write(line + "\n")
                logging.info("UpdateOverlap %s", line)

    def close(self):
        self.cache.cleanup()  # Only tensors in this recorder's unique temp directory.
        self.history = {"S": [], "P": []}
