"""Experiment checks and provenance belong in the launcher, not training code."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def check_spec(spec):
    assert spec["seeds"] == [1993]
    common = spec["common_overrides"]
    assert common["disable_fused_sdpa"] is True
    assert common["init_epoch"] == common["epochs"] == 20
    assert common["ca_epochs"] == 5
    assert common["max_tasks"] == common["total_sessions"] == 10
    assert common["dual_mask_reg_weight"] == .01
    assert common["dual_mask_task0_gate_mode"] == "unmasked"
    assert common["dual_mask_conflict_merge_mode"] == "suppress"
    for variant in spec["variants"]:
        settings = {**common, **variant["overrides"]}
        assert "data_path" not in settings
        assert 0 <= settings.get("dual_mask_conflict_local_fraction", .5) <= 1
        assert settings["dual_mask_private_conflict_mode"] == "global"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    check_spec(spec)
    config = json.loads((ROOT / spec["datasets"][0]["config"]).read_text())
    data_path = Path(config["data_path"])
    assert data_path.is_dir(), f"JSON data_path does not exist: {data_path}"
    print("Dataset from machine JSON:", data_path)
    sources = ["models/attention.py", "methods/dlora.py", "utils/dual_mask_budget.py",
               "utils/update_overlap.py", "trainer.py", "main.py"]
    snapshot = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "base_config": config, "spec": spec,
        "note": "Extra --set arguments supplied to the shell are recorded by main.py in each training log.",
    }
    directory = ROOT / spec["log_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("preflight_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    output.write_text(json.dumps(snapshot, indent=2) + "\n")
    print("Experiment fingerprint:", output)


if __name__ == "__main__":
    main()
