import json
import unittest
from pathlib import Path

import torch

from models.attention import Attention_LoRA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
SWEEP_DIR = PROJECT_ROOT / "scripts" / "sweeps"


class BranchGateSweepTests(unittest.TestCase):
    def _load(self, name):
        return json.loads((SWEEP_DIR / name).read_text())

    def test_seed1993_order_and_single_factor_overrides(self):
        spec = self._load("9_21_imgr10_branch_gate_seed1993.json")
        variants = spec["variants"]

        self.assertEqual(spec["seeds"], [1993])
        self.assertEqual(
            [variant["name"] for variant in variants],
            [
                "a_baseline",
                "b_p_conflict_off",
                "c_all_conflict_off",
                "d_s_protect_off",
            ],
        )
        self.assertNotIn("data_path", spec["common_overrides"])
        self.assertEqual(
            variants[1]["overrides"]["dual_mask_private_conflict_mode"],
            "none",
        )
        self.assertEqual(
            variants[2]["overrides"]["dual_mask_conflict_strength"],
            0.0,
        )
        self.assertIs(
            variants[3]["overrides"]["dual_mask_s_protect_enabled"],
            False,
        )

    def test_followup_is_only_p_conflict_off(self):
        spec = self._load("9_21_imgr10_p_conflict_off_followup.json")

        self.assertEqual(spec["seeds"], [1996, 1997])
        self.assertEqual(len(spec["variants"]), 1)
        self.assertEqual(
            spec["common_overrides"]["dual_mask_private_conflict_mode"],
            "none",
        )
        self.assertIs(
            spec["common_overrides"]["dual_mask_s_protect_enabled"],
            True,
        )
        self.assertEqual(
            spec["common_overrides"]["dual_mask_conflict_strength"],
            0.5,
        )
        self.assertNotIn("data_path", spec["common_overrides"])

    def test_all_full_runs_keep_the_reference_protocol(self):
        for name in (
            "9_21_imgr10_branch_gate_seed1993.json",
            "9_21_imgr10_p_conflict_off_followup.json",
        ):
            with self.subTest(name=name):
                common = self._load(name)["common_overrides"]
                self.assertEqual(common["init_epoch"], 20)
                self.assertEqual(common["epochs"], 20)
                self.assertEqual(common["rank"], 64)
                self.assertIs(common["ca"], True)
                self.assertEqual(common["ca_epochs"], 5)
                self.assertEqual(common["dual_mask_task0_gate_mode"], "unmasked")
                self.assertEqual(common["dual_mask_anchor_reg_weight"], 10.0)

    def test_generated_scripts_keep_order_and_do_not_override_data_path(self):
        primary = (SCRIPT_DIR / "9_21_imgr10_branch_gate_seed1993_3090.sh").read_text()
        followup = (SCRIPT_DIR / "9_21_imgr10_p_conflict_off_followup_3090.sh").read_text()
        smoke = (SCRIPT_DIR / "9_21_imgr10_p_conflict_off_smoke_3090.sh").read_text()

        positions = [
            primary.index("Starting imgr10_a_baseline_seed1993"),
            primary.index("Starting imgr10_b_p_conflict_off_seed1993"),
            primary.index("Starting imgr10_c_all_conflict_off_seed1993"),
            primary.index("Starting imgr10_d_s_protect_off_seed1993"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(
            followup.index("Starting imgr10_b_p_conflict_off_seed1996"),
            followup.index("Starting imgr10_b_p_conflict_off_seed1997"),
        )
        for script in (primary, followup, smoke):
            self.assertIn('cd "$(dirname "$0")/.."', script)
            self.assertNotIn("data_path=", script)
            self.assertNotIn("/mnt/", script)
            self.assertNotIn("/home/", script)

    def test_s_protect_off_changes_only_shared_branch_gate(self):
        def build(enabled):
            module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
            module._init_params(
                {
                    "use_slora": True,
                    "use_plora": True,
                    "rank": 2,
                    "dual_mask_importance": "svd",
                    "dual_mask_general_ratio": 0.5,
                    "dual_mask_svd_rank": 2,
                    "dual_mask_conflict_ratio": 0.0,
                    "dual_mask_conflict_strength": 0.0,
                    "dual_mask_s_protect_enabled": enabled,
                    "lora_A_init": "kaiming",
                }
            )
            module.cur_task = 1
            module.general_mask.zero_()
            module.general_mask[:, :2] = 1.0
            module.effective_protect_strength = 0.5
            return module

        protect_on = build(True)
        protect_off = build(False)
        delta = torch.ones_like(protect_on.qkv.weight)

        self.assertFalse(
            torch.equal(
                protect_on._safe_delta(delta, isolated=False),
                protect_off._safe_delta(delta, isolated=False),
            )
        )
        self.assertTrue(
            torch.equal(
                protect_on._safe_delta(delta, isolated=True),
                protect_off._safe_delta(delta, isolated=True),
            )
        )

    def test_zero_conflict_strength_stays_zero_under_overlap_adaptation(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(
            {
                "use_slora": True,
                "use_plora": True,
                "rank": 2,
                "dual_mask_importance": "svd",
                "dual_mask_general_ratio": 0.5,
                "dual_mask_svd_rank": 2,
                "dual_mask_conflict_ratio": 0.1,
                "dual_mask_conflict_strength": 0.0,
                "dual_mask_conflict_old_overlap_adaptive": True,
                "lora_A_init": "kaiming",
            }
        )
        module.set_pretrained_old_overlap_risk(1.0)

        _, strength = module._conflict_parameters()

        self.assertEqual(strength, 0.0)

    def test_two_machine_specs_preserve_paired_comparisons(self):
        spec_3090 = self._load("9_23_imgr10_branch_gate_3090.json")
        spec_5090_diag = self._load("9_23_imgr10_branch_gate_5090_diag.json")
        spec_5090_b = self._load("9_23_imgr10_branch_gate_5090_b1997.json")

        self.assertEqual(spec_3090["seeds"], [1993, 1996])
        self.assertEqual(
            [variant["name"] for variant in spec_3090["variants"]],
            ["a_baseline", "b_p_conflict_off"],
        )
        self.assertEqual(spec_5090_diag["seeds"], [1993])
        self.assertEqual(
            [variant["name"] for variant in spec_5090_diag["variants"]],
            ["a_baseline", "c_all_conflict_off", "d_s_protect_off"],
        )
        self.assertEqual(spec_5090_b["seeds"], [1997])
        self.assertEqual(
            [variant["name"] for variant in spec_5090_b["variants"]],
            ["a_baseline", "b_p_conflict_off"],
        )

    def test_two_machine_generated_order_and_paths(self):
        script_3090 = (SCRIPT_DIR / "9_23_imgr10_branch_gate_3090_runs.sh").read_text()
        script_5090_diag = (SCRIPT_DIR / "9_23_imgr10_branch_gate_5090_diag_runs.sh").read_text()
        script_5090_b = (SCRIPT_DIR / "9_23_imgr10_branch_gate_5090_b1997_runs.sh").read_text()
        smoke_5090 = (SCRIPT_DIR / "9_23_imgr10_all_conflict_off_smoke_5090.sh").read_text()

        order_3090 = [
            "Starting imgr10_a_baseline_seed1993",
            "Starting imgr10_b_p_conflict_off_seed1993",
            "Starting imgr10_a_baseline_seed1996",
            "Starting imgr10_b_p_conflict_off_seed1996",
        ]
        positions_3090 = [script_3090.index(item) for item in order_3090]
        self.assertEqual(positions_3090, sorted(positions_3090))

        order_5090_diag = [
            "Starting imgr10_a_baseline_seed1993",
            "Starting imgr10_c_all_conflict_off_seed1993",
            "Starting imgr10_d_s_protect_off_seed1993",
        ]
        positions_5090_diag = [script_5090_diag.index(item) for item in order_5090_diag]
        self.assertEqual(positions_5090_diag, sorted(positions_5090_diag))

        self.assertLess(
            script_5090_b.index("Starting imgr10_a_baseline_seed1997"),
            script_5090_b.index("Starting imgr10_b_p_conflict_off_seed1997"),
        )

        for script in (script_3090, script_5090_diag, script_5090_b, smoke_5090):
            self.assertIn('cd "$(dirname "$0")/.."', script)
            self.assertNotIn("data_path=", script)
            self.assertNotIn("/mnt/", script)
            self.assertNotIn("/home/", script)

    def test_two_machine_full_runs_keep_reference_protocol(self):
        for name in (
            "9_23_imgr10_branch_gate_3090.json",
            "9_23_imgr10_branch_gate_5090_diag.json",
            "9_23_imgr10_branch_gate_5090_b1997.json",
        ):
            with self.subTest(name=name):
                common = self._load(name)["common_overrides"]
                self.assertEqual(common["init_epoch"], 20)
                self.assertEqual(common["epochs"], 20)
                self.assertEqual(common["rank"], 64)
                self.assertIs(common["ca"], True)
                self.assertEqual(common["ca_epochs"], 5)
                self.assertEqual(common["dual_mask_task0_gate_mode"], "unmasked")
                self.assertEqual(common["dual_mask_anchor_reg_weight"], 10.0)
                self.assertNotIn("data_path", common)


if __name__ == "__main__":
    unittest.main()
