import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MaskComparisonScriptTests(unittest.TestCase):
    def commands(self, machine, script_name):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script_name), "--dry-run"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        commands = [line for line in result.stdout.splitlines()
                    if line.startswith("python main.py")]
        for command in commands:
            self.assertIn("--config exps/dlora/imgr10.json", command)
            self.assertIn("seed=\\[1993\\]", command)
            self.assertIn("ca_epochs=5", command)
            self.assertIn("max_tasks=10", command)
            self.assertIn("dual_mask_conflict_reg_enabled=false", command)
            self.assertIn("dual_mask_conflict_ratio=0.1", command)
            self.assertNotIn("data_path=", command)
            self.assertNotIn("checkpoint_resume=/", command)
            self.assertIn(f"wandb_group=imgr10_mask_comparison_{machine}", command)
        return commands

    def test_3090_compares_three_granularities_at_one_budget(self):
        commands = self.commands("3090", "9_24_imgr10_granularity_3090.sh")
        self.assertEqual(len(commands), 3)
        for command, granularity in zip(commands, ("layer", "projection", "model")):
            self.assertIn(f"dual_mask_conflict_granularity={granularity}", command)
            self.assertIn("dual_mask_conflict_budget_multiplier=1.0", command)
            self.assertIn("dual_mask_conflict_score_mode=conflict", command)

    def test_5090_compares_budget_and_same_budget_score(self):
        commands = self.commands("5090", "9_24_imgr10_sparsity_5090.sh")
        self.assertEqual(len(commands), 4)
        for command, multiplier in zip(commands, ("0.5", "1.0", "1.5", "1.0")):
            self.assertIn("dual_mask_conflict_granularity=layer", command)
            self.assertIn(f"dual_mask_conflict_budget_multiplier={multiplier}", command)
        for command in commands[:3]:
            self.assertIn("dual_mask_conflict_score_mode=conflict", command)
        self.assertIn("dual_mask_conflict_score_mode=magnitude", commands[3])


if __name__ == "__main__":
    unittest.main()
