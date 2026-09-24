import json
from pathlib import Path
import shlex
import subprocess
import unittest

from scripts.night_mechanism_preflight import check_spec
from scripts.summarize_night_mechanism import parse_log
import tempfile


ROOT = Path(__file__).resolve().parents[1]


class NightScriptTests(unittest.TestCase):
    def test_summary_keeps_stage_counts_separate_and_does_not_use_ncm_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text("\n".join([
                "CNN top1 curve: [90, 80]", "Average Accuracy: 85.0", "Last Accuracy: 80.0",
                "W_pre-only NCM Average Accuracy: 60.0", "W_pre-only NCM Last Accuracy: 55.0",
                "LoRA-stage trainable scalars: LoRA=100, classifier=10, total=110",
                "LoRA-stage trainable scalars: LoRA=200, classifier=10, total=210",
                "CA-stage optimized classifier scalars: 20",
            ]))
            row, _, _ = parse_log(path)
            self.assertEqual(row["average"], 85.)
            self.assertEqual(row["last"], 80.)
            self.assertFalse(row["complete"])
            self.assertEqual(row["incremental_trainable_min"], 210)
            self.assertEqual(row["ca_classifier_max"], 20)

    def spec(self, name):
        spec = json.loads((ROOT / "scripts/sweeps" / (name + ".json")).read_text())
        check_spec(spec)
        return spec

    def test_3090_six_group_factorial_and_branch_controls(self):
        spec = self.spec("imgr10_gate_reg_overlap_3090")
        self.assertTrue(spec["common_overrides"]["dual_mask_update_overlap"])
        self.assertTrue(spec["common_overrides"]["dual_mask_conflict_energy_adaptive"])
        self.assertTrue(spec["common_overrides"]["dual_mask_conflict_old_overlap_adaptive"])
        keys = ["dual_mask_conflict_reg_enabled", "dual_mask_s_conflict_enabled", "dual_mask_p_conflict_enabled"]
        rows = [tuple(v["overrides"][k] for k in keys) for v in spec["variants"]]
        self.assertEqual(rows, [(True, True, True), (False, True, True),
                                (True, False, False), (False, False, False),
                                (False, True, False), (False, False, True)])
        for v in spec["variants"]:
            self.assertEqual(set(v["overrides"]), set(keys))

    def test_5090_only_adds_four_rho_runs(self):
        spec = self.spec("imgr10_fixed_rho_5090")
        common = spec["common_overrides"]
        self.assertFalse(common["dual_mask_conflict_reg_enabled"])
        self.assertFalse(common["dual_mask_conflict_energy_adaptive"])
        self.assertFalse(common["dual_mask_conflict_old_overlap_adaptive"])
        self.assertEqual([v["overrides"]["dual_mask_conflict_local_fraction"] for v in spec["variants"]], [0, .5, .25, .75])
        old = json.loads((ROOT / "scripts/sweeps/imgr10_conflict_score_t10_seed1993_5090.json").read_text())
        baseline = {**json.loads((ROOT / "exps/dlora/imgr10.json").read_text()), **old["common_overrides"]}
        for key, value in common.items():
            if key in baseline and key not in {"wandb_group", "wandb_tags"}:
                self.assertEqual(value, baseline[key], key)

    def test_generated_shells_match_specs_and_run_outside_repo(self):
        for name in ("imgr10_gate_reg_overlap_3090", "imgr10_fixed_rho_5090"):
            spec = self.spec(name)
            script = ROOT / "scripts" / ("9_24_" + name + ".sh")
            subprocess.run(["bash", "-n", str(script)], check=True)
            output = subprocess.check_output(["bash", str(script), "--dry-run"], cwd="/tmp", text=True)
            commands = output.replace("\\\n", "").splitlines()
            self.assertEqual(len(commands), len(spec["variants"]))
            for line, variant in zip(commands, spec["variants"]):
                tokens = shlex.split(line)
                settings = dict(tokens[i+1].split("=", 1) for i, t in enumerate(tokens) if t == "--set")
                self.assertEqual(json.loads(settings["seed"]), [1993])
                self.assertNotIn("data_path", settings)
                for key, value in {**spec["common_overrides"], **variant["overrides"]}.items():
                    try:
                        actual = json.loads(settings[key])
                    except json.JSONDecodeError:
                        actual = settings[key]
                    self.assertEqual(actual, value, key)
                self.assertIn("${TIMESTAMP}", settings["prefix"])


if __name__ == "__main__":
    unittest.main()
