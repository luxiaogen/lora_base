import json
import re
import shlex
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def commands(path):
    source = path.read_text()
    blocks = re.findall(r"(?ms)^    python main\.py.*?2>&1 \| tee", source)
    parsed = []
    for block in blocks:
        tokens = shlex.split(block.replace("\\\n", " "))
        parsed.append({
            tokens[i + 1].split("=", 1)[0]: tokens[i + 1].split("=", 1)[1]
            for i, token in enumerate(tokens[:-1])
            if token == "--set"
        })
    return source, parsed


class GlobalConflictScriptTests(unittest.TestCase):
    def test_script_is_a_matched_task0_to_task2_pair(self):
        script = ROOT / "scripts/9_22_imgr10_global_conflict_screen.sh"
        source, runs = commands(script)
        self.assertEqual(len(runs), 2)
        self.assertNotIn('\ncd "$(dirname "$0")"\n', source)
        subprocess.run(["bash", "-n", str(script)], check=True)

        layer = next(run for run in runs if run["dual_mask_conflict_granularity"] == "layer")
        model = next(run for run in runs if run["dual_mask_conflict_granularity"] == "model")
        ignored = {"prefix", "dual_mask_conflict_granularity", "wandb_tags"}
        self.assertEqual(
            {key: value for key, value in layer.items() if key not in ignored},
            {key: value for key, value in model.items() if key not in ignored},
        )
        for run in runs:
            self.assertEqual(run["seed"], "[1993]")
            self.assertEqual(run["max_tasks"], "3")
            self.assertEqual(run["total_sessions"], "10")
            self.assertEqual(run["task0_checkpoint_resume"], "")
            self.assertNotIn("data_path", run)

    def test_spec_has_one_seed_and_one_changed_factor(self):
        spec = json.loads(
            (ROOT / "scripts/sweeps/imgr10_global_conflict_screen.json").read_text()
        )
        self.assertEqual(spec["seeds"], [1993])
        self.assertEqual(len(spec["variants"]), 2)
        self.assertEqual(
            {variant["overrides"]["dual_mask_conflict_granularity"] for variant in spec["variants"]},
            {"layer", "model"},
        )

    def test_smoke_reaches_task1_without_changing_the_t10_split(self):
        script = ROOT / "scripts/9_22_imgr10_global_conflict_smoke.sh"
        source, runs = commands(script)
        self.assertEqual(len(runs), 1)
        self.assertNotIn('\ncd "$(dirname "$0")"\n', source)
        subprocess.run(["bash", "-n", str(script)], check=True)
        run = runs[0]
        self.assertEqual(run["max_tasks"], "2")
        self.assertEqual(run["total_sessions"], "10")
        self.assertEqual(run["init_epoch"], "1")
        self.assertEqual(run["epochs"], "1")
        self.assertEqual(run["ca"], "false")
        self.assertEqual(run["dual_mask_conflict_granularity"], "model")
        self.assertNotIn("data_path", run)


if __name__ == "__main__":
    unittest.main()
