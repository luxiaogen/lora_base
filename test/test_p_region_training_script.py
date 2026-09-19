import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/9_19_p_region_train_merge_5090_overnight.sh"


class TrainingSweepTests(unittest.TestCase):
    def dry_run(self, budget="0"):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, DRY_RUN="1", BUDGET_SECONDS=budget,
                       ESTIMATED_RUN_SECONDS="3600", LOG_DIR=directory)
            result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

    def test_full_matrix_and_no_data_path_override(self):
        output = self.dry_run()
        commands = [shlex.split(line) for line in output.splitlines() if line.startswith("DRYRUN python ")]
        self.assertEqual(len(commands), 12)
        actual = []
        for tokens in commands:
            self.assertEqual(tokens[tokens.index("--config")+1], "exps/dlora/imgr10.json")
            overrides = dict(tokens[i+1].split("=", 1) for i, t in enumerate(tokens) if t == "--set")
            self.assertNotIn("data_path", overrides)
            self.assertEqual(overrides["total_sessions"], "10")
            self.assertEqual(overrides["ca_epochs"], "5")
            self.assertEqual(overrides["init_epoch"], "20")
            self.assertEqual(overrides["epochs"], "20")
            self.assertEqual(overrides["rank"], "64")
            self.assertEqual(overrides["dual_mask_p_region_diagnostic"], "false")
            self.assertEqual(overrides["dual_mask_conflict_merge_mode"], "suppress")
            actual.append((json.loads(overrides["seed"])[0], overrides["dual_mask_p_region_train_mode"],
                           float(overrides["dual_mask_p_region_train_amount"])))
        core = [("none", 0.5), ("conflict", 0.5), ("nonconflict", 0.5)]
        expected = [(1993, m, a) for m, a in core + [("conflict", 0.25), ("nonconflict", 0.25), ("none", 0.5)]]
        expected += [(s, m, a) for s in (1996, 1997) for m, a in core]
        self.assertEqual(actual, expected)
        self.assertIn("Finished 12/12 runs; passed=12; FAILED=0", output)

    def test_budget_stop_does_not_claim_completion(self):
        output = self.dry_run("1")
        self.assertIn("BUDGET STOP: attempted 0/12; passed=0; deferred 12", output)
        self.assertNotIn("DRYRUN python", output)
        self.assertNotIn("Finished 12/12", output)

    def test_specs_match_executable_matrix(self):
        output = self.dry_run()
        commands = [shlex.split(line) for line in output.splitlines() if line.startswith("DRYRUN python ")]
        planned = []
        for phase in ("screen", "confirm"):
            spec = json.loads((ROOT / f"scripts/sweeps/p_region_train_{phase}.json").read_text())
            for seed in spec["seeds"]:
                for variant in spec["variants"]:
                    planned.append(dict(spec["common_overrides"], **variant["overrides"], seed=[seed]))
        for command, plan in zip(commands, planned):
            overrides = dict(command[i+1].split("=", 1) for i, t in enumerate(command) if t == "--set")
            for key, value in plan.items():
                encoded = json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value
                self.assertEqual(overrides[key], encoded)


if __name__ == "__main__":
    unittest.main()
