import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LoRINightScriptTests(unittest.TestCase):
    def check_machine(self, machine, expected):
        script = ROOT / "scripts" / f"9_23_imgr10_lori_night_{machine}.sh"
        result = subprocess.run(
            ["bash", str(script), "--dry-run"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        commands = [line for line in result.stdout.splitlines()
                    if line.startswith("python main.py")]
        self.assertEqual(len(commands), 3)
        for command, variant in zip(commands, expected):
            self.assertIn("--config exps/dlora/imgr10.json", command)
            self.assertIn("ca_epochs=5", command)
            self.assertIn("max_tasks=10", command)
            self.assertIn("dual_mask_conflict_reg_enabled=true", command)
            self.assertIn("seed=\\[1993\\]", command)
            self.assertNotIn("data_path=", command)
            self.assertIn(f"prefix=imgr10_{variant}_seed1993_{machine}", command)
        return commands

    def test_3090_has_baseline_plastic_optimization_and_model_granularity(self):
        commands = self.check_machine("3090", ["layer", "plastic_norm_matched", "model"])
        self.assertIn("dual_mask_private_conflict_mode=plastic_norm_matched", commands[1])
        self.assertIn("dual_mask_conflict_granularity=model", commands[2])

    def test_5090_has_baseline_and_two_ratio_endpoints(self):
        commands = self.check_machine("5090", ["layer", "ratio005", "ratio020"])
        self.assertIn("dual_mask_conflict_ratio=0.05", commands[1])
        self.assertIn("dual_mask_conflict_ratio=0.20", commands[2])


if __name__ == "__main__":
    unittest.main()
