import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
SWEEP_DIR = SCRIPT_DIR / "sweeps"


class BranchStrengthSweepTests(unittest.TestCase):
    def _load(self, name):
        return json.loads((SWEEP_DIR / name).read_text())

    def test_s_protect_followup_is_paired_for_two_seeds(self):
        spec = self._load("9_22_imgr10_s_protect_followup.json")

        self.assertEqual(spec["seeds"], [1996, 1997])
        self.assertEqual(
            [variant["name"] for variant in spec["variants"]],
            ["a_baseline", "d_s_protect_off"],
        )
        self.assertIs(spec["variants"][0]["overrides"]["dual_mask_s_protect_enabled"], True)
        self.assertIs(spec["variants"][1]["overrides"]["dual_mask_s_protect_enabled"], False)
        self.assertEqual(spec["common_overrides"]["dual_mask_conflict_strength"], 0.5)
        self.assertEqual(spec["common_overrides"]["dual_mask_private_conflict_strength"], 0.5)

    def test_p_strength_sweep_changes_only_private_beta(self):
        spec = self._load("9_22_imgr10_p_strength.json")

        self.assertEqual(spec["seeds"], [1993, 1996, 1997])
        self.assertEqual(spec["common_overrides"]["dual_mask_conflict_strength"], 0.5)
        self.assertEqual(
            [variant["overrides"]["dual_mask_private_conflict_strength"] for variant in spec["variants"]],
            [0.25, 0.5, 0.75],
        )
        for variant in spec["variants"]:
            self.assertNotIn("dual_mask_conflict_strength", variant["overrides"])

    def test_protocol_and_data_path_are_fixed(self):
        for name in (
            "9_22_imgr10_s_protect_followup.json",
            "9_22_imgr10_p_strength.json",
        ):
            with self.subTest(name=name):
                spec = self._load(name)
                common = spec["common_overrides"]
                self.assertEqual(spec["datasets"][0]["config"], "exps/dlora/imgr10.json")
                self.assertNotIn("data_path", common)
                self.assertEqual(common["init_epoch"], 20)
                self.assertEqual(common["epochs"], 20)
                self.assertEqual(common["rank"], 64)
                self.assertIs(common["ca"], True)
                self.assertEqual(common["ca_epochs"], 5)
                self.assertEqual(common["dual_mask_task0_gate_mode"], "unmasked")

    def test_generated_scripts_do_not_override_machine_local_data_path(self):
        for name in (
            "9_22_imgr10_s_protect_followup_3090.sh",
            "9_22_imgr10_p_strength_5090.sh",
            "9_22_imgr10_p_strength_smoke_5090.sh",
        ):
            with self.subTest(name=name):
                script = (SCRIPT_DIR / name).read_text()
                self.assertIn('cd "$(dirname "$0")/.."', script)
                self.assertNotIn("data_path=", script)
                self.assertNotIn("/mnt/", script)
                self.assertNotIn("/home/", script)


if __name__ == "__main__":
    unittest.main()
