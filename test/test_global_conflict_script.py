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

    def test_projection_script_is_a_matched_task0_to_task2_pair(self):
        script = ROOT / "scripts/9_22_imgr10_projection_conflict_screen_3090.sh"
        source = script.read_text()
        subprocess.run(["bash", "-n", str(script)], check=True)
        self.assertEqual(source.count("\nrun_variant \\\n"), 2)
        self.assertIn("imgr10_layer_budget_seed1993_3090", source)
        self.assertIn("imgr10_projection_budget_seed1993_3090", source)
        self.assertIn("--set 'seed=[1993]'", source)
        self.assertIn("--set max_tasks=3", source)
        self.assertIn("--set total_sessions=10", source)
        self.assertNotIn("--set data_path=", source)

    def test_projection_spec_has_one_seed_and_one_changed_factor(self):
        spec = json.loads(
            (ROOT / "scripts/sweeps/imgr10_projection_conflict_screen_3090.json").read_text()
        )
        self.assertEqual(spec["seeds"], [1993])
        self.assertEqual(len(spec["variants"]), 2)
        self.assertEqual(
            {
                variant["overrides"]["dual_mask_conflict_granularity"]
                for variant in spec["variants"]
            },
            {"layer", "projection"},
        )

    def test_projection_t10_scripts_keep_each_seed_pair_on_one_machine(self):
        cases = [
            (
                ROOT / "scripts/9_23_imgr10_projection_conflict_t10_seed1993_3090.sh",
                [("[1993]", "layer"), ("[1993]", "projection")],
            ),
            (
                ROOT
                / "scripts/9_23_imgr10_projection_conflict_t10_seeds1996_1997_5090.sh",
                [
                    ("[1996]", "layer"),
                    ("[1996]", "projection"),
                    ("[1997]", "layer"),
                    ("[1997]", "projection"),
                ],
            ),
        ]
        for script, expected_order in cases:
            with self.subTest(script=script.name):
                source, runs = commands(script)
                subprocess.run(["bash", "-n", str(script)], check=True)
                self.assertNotIn('\ncd "$(dirname "$0")"\n', source)
                self.assertEqual(
                    [
                        (run["seed"], run["dual_mask_conflict_granularity"])
                        for run in runs
                    ],
                    expected_order,
                )
                ignored = {
                    "seed",
                    "prefix",
                    "dual_mask_conflict_granularity",
                    "wandb_group",
                    "wandb_tags",
                }
                reference = {
                    key: value for key, value in runs[0].items() if key not in ignored
                }
                for run in runs:
                    self.assertEqual(
                        {key: value for key, value in run.items() if key not in ignored},
                        reference,
                    )
                    self.assertEqual(run["max_tasks"], "10")
                    self.assertEqual(run["total_sessions"], "10")
                    self.assertEqual(run["task0_checkpoint_resume"], "")
                    self.assertNotIn("data_path", run)

    def test_projection_t10_specs_cover_three_seeds_with_two_variants(self):
        spec_3090 = json.loads(
            (
                ROOT
                / "scripts/sweeps/imgr10_projection_conflict_t10_seed1993_3090.json"
            ).read_text()
        )
        spec_5090 = json.loads(
            (
                ROOT
                / "scripts/sweeps/imgr10_projection_conflict_t10_seeds1996_1997_5090.json"
            ).read_text()
        )
        self.assertEqual(spec_3090["seeds"], [1993])
        self.assertEqual(spec_5090["seeds"], [1996, 1997])
        for spec in (spec_3090, spec_5090):
            self.assertEqual(
                [
                    variant["overrides"]["dual_mask_conflict_granularity"]
                    for variant in spec["variants"]
                ],
                ["layer", "projection"],
            )


if __name__ == "__main__":
    unittest.main()
