import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import Attention_LoRA  # noqa: E402
from utils.dual_mask_budget import (  # noqa: E402
    select_global_budget_masks,
    select_projection_budget_masks,
)


class GlobalBudgetSelectionTests(unittest.TestCase):
    @staticmethod
    def _make_attention(granularity="model", **overrides):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        args = {
            "use_slora": True,
            "use_plora": True,
            "dual_mask_importance": "svd",
            "dual_mask_general_ratio": 0.4,
            "dual_mask_svd_rank": 2,
            "dual_mask_conflict_ratio": 0.25,
            "dual_mask_conflict_strength": 0.5,
            "dual_mask_conflict_energy_adaptive": False,
            "dual_mask_conflict_energy_ratio_floor": True,
            "dual_mask_private_conflict_mode": "global",
            "dual_mask_conflict_granularity": granularity,
            "dual_mask_competence_adaptive": False,
            "dual_mask_protect_strength_mode": "legacy_linear",
            "lora_A_init": "kaiming",
        }
        args.update(overrides)
        module._init_params(args)
        module.cur_task = 1
        module.w0_importance.fill_(1.0)
        return module

    def test_global_selection_matches_reference_budget(self):
        scores = [
            torch.tensor([[1.0, 4.0], [2.0, 3.0]]),
            torch.tensor([[8.0, 7.0], [6.0, 5.0]]),
        ]
        references = [
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        ]

        masks = select_global_budget_masks(scores, references)

        self.assertEqual(sum(int(mask.sum()) for mask in masks), 3)
        self.assertEqual(int(masks[0].sum()), 0)
        self.assertEqual(int(masks[1].sum()), 3)

    def test_global_selection_respects_valid_coordinates(self):
        scores = [torch.tensor([[100.0, 4.0], [3.0, 2.0]])]
        references = [torch.tensor([[1.0, 1.0], [0.0, 0.0]])]
        valid = [torch.tensor([[0.0, 1.0], [1.0, 1.0]])]

        masks = select_global_budget_masks(scores, references, valid)

        self.assertEqual(int(masks[0].sum()), 2)
        self.assertEqual(masks[0][0, 0].item(), 0.0)

    def test_projection_selection_matches_each_projection_reference_budget(self):
        scores = [
            torch.tensor([[9.0], [8.0], [1.0], [2.0], [3.0], [4.0]]),
            torch.tensor([[7.0], [6.0], [5.0], [4.0], [3.0], [2.0]]),
        ]
        references = [
            torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [1.0]]),
            torch.tensor([[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]),
        ]

        masks = select_projection_budget_masks(scores, references)

        for projection in range(3):
            selected = sum(int(mask.chunk(3, dim=0)[projection].sum()) for mask in masks)
            reference = sum(int(mask.chunk(3, dim=0)[projection].sum()) for mask in references)
            self.assertEqual(selected, reference)
        self.assertEqual(sum(int(mask.sum()) for mask in masks), 5)

    def test_projection_selection_rejects_non_qkv_rows(self):
        with self.assertRaisesRegex(ValueError, "divisible by 3"):
            select_projection_budget_masks(
                [torch.ones(4, 2)],
                [torch.ones(4, 2)],
            )

    def test_attention_uses_branch_specific_global_mask(self):
        module = self._make_attention()
        shared = torch.zeros_like(module.qkv.weight)
        private = torch.zeros_like(module.qkv.weight)
        shared[0, 0] = 1.0
        private[1, 0] = 1.0
        module.set_global_conflict_masks(shared, private)
        delta = torch.ones_like(module.qkv.weight)

        _, shared_selected = module._branch_conflict(delta, isolated=False)
        _, private_selected = module._branch_conflict(delta, isolated=True)

        self.assertTrue(torch.equal(shared_selected, shared))
        self.assertTrue(torch.equal(private_selected, private))

    def test_global_mask_controls_training_and_merge_delta(self):
        module = self._make_attention()
        module.general_mask.zero_()
        shared = torch.zeros_like(module.qkv.weight)
        shared[0, 0] = 1.0
        module.set_global_conflict_masks(shared, torch.zeros_like(shared))
        delta = torch.ones_like(module.qkv.weight)

        training_delta = module._safe_delta(
            delta,
            isolated=False,
            conflict_ratio=0.25,
            conflict_strength=0.5,
        )
        merge_delta = module._compose_merge_delta(
            delta,
            isolated=False,
            conflict_ratio=0.25,
            conflict_strength=0.5,
        )

        expected = delta.clone()
        expected[0, 0] = 0.5
        self.assertTrue(torch.equal(training_delta, expected))
        self.assertTrue(torch.equal(merge_delta, expected))

    def test_layer_mode_falls_back_to_existing_local_selection(self):
        module = self._make_attention(granularity="layer")
        self.assertEqual(module.global_s_conflict_mask.numel(), 0)
        self.assertEqual(module.global_p_conflict_mask.numel(), 0)
        delta = torch.arange(module.qkv.weight.numel(), dtype=torch.float32).reshape_as(module.qkv.weight)

        expected_score, expected_mask = module._joint_conflict(delta, conflict_ratio=0.25)
        actual_score, actual_mask = module._branch_conflict(
            delta,
            isolated=False,
            conflict_ratio=0.25,
        )

        self.assertTrue(torch.equal(actual_score, expected_score))
        self.assertTrue(torch.equal(actual_mask, expected_mask))

    def test_projection_mode_is_accepted(self):
        module = self._make_attention(granularity="projection")
        self.assertEqual(module.dual_mask_conflict_granularity, "projection")

    def test_scaled_budget_is_relative_to_actual_reference_mask(self):
        delta = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        baseline = self._make_attention(
            granularity="layer",
            dual_mask_conflict_energy_adaptive=True,
        )
        baseline.w0_importance.copy_(torch.linspace(0.1, 1.0, 48).reshape(12, 4))
        _, reference = baseline._joint_conflict(delta)
        reference_k = int(reference.sum().item())
        self.assertGreater(reference_k, 0)

        for multiplier in (0.5, 1.0, 1.5):
            module = self._make_attention(
                granularity="layer",
                dual_mask_conflict_energy_adaptive=True,
                dual_mask_conflict_budget_multiplier=multiplier,
            )
            module.w0_importance.copy_(baseline.w0_importance)
            _, selected = module._joint_conflict(delta)
            expected_k = min(delta.numel(), int(reference_k * multiplier + 0.5))
            self.assertEqual(int(selected.sum().item()), expected_k)
            if multiplier == 1.0:
                self.assertTrue(torch.equal(selected, reference))

    def test_magnitude_only_uses_same_budget_but_different_score(self):
        delta = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        baseline = self._make_attention(granularity="layer")
        magnitude = self._make_attention(
            granularity="layer",
            dual_mask_conflict_score_mode="magnitude",
        )
        importance = torch.ones_like(delta)
        importance[:, 2:] = 0.01
        baseline.w0_importance.copy_(importance)
        magnitude.w0_importance.copy_(importance)
        _, reference = baseline._joint_conflict(delta)
        _, selected = magnitude._joint_conflict(delta)
        self.assertEqual(int(selected.sum().item()), int(reference.sum().item()))
        self.assertFalse(torch.equal(selected, reference))

    def test_scaled_budget_respects_valid_mask_and_unmasked_task0(self):
        module = self._make_attention(
            granularity="layer",
            dual_mask_conflict_budget_multiplier=1.5,
        )
        module.dual_mask_task0_gate_mode = "unmasked"
        delta = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        valid = torch.zeros_like(delta)
        valid[:, :2] = 1
        _, selected = module._joint_conflict(delta, valid_mask=valid)
        self.assertEqual(int((selected * (1 - valid)).sum().item()), 0)
        module.cur_task = 0
        self.assertTrue(torch.equal(module._safe_delta(delta, isolated=False), delta))

    def test_private_merge_diagnostic_uses_applied_projection_mask(self):
        module = self._make_attention(granularity="projection")
        module.general_mask.zero_()
        module.general_mask[0, 0] = 1.0
        shared = torch.zeros_like(module.qkv.weight)
        private = torch.zeros_like(module.qkv.weight)
        private[0, 0] = 1.0  # Protected: selected, but P cannot update it.
        private[1, 0] = 1.0  # Plastic: selected and suppressed.
        module.set_global_conflict_masks(shared, private)
        raw = torch.ones_like(module.qkv.weight)
        safe = module._compose_merge_delta(raw, True, 0.25, 0.5)
        before = module.qkv.weight.detach().clone()

        stats = module._private_merge_diagnostic(raw, safe, 0.25, 0.5)

        self.assertEqual(stats["selected"], 2)
        self.assertEqual(stats["selected_plastic"], 1)
        self.assertEqual(stats["selected_active"], 1)
        self.assertAlmostEqual(stats["plastic_overlap"], 0.5)
        self.assertAlmostEqual(stats["removed_norm"], 0.5)
        self.assertAlmostEqual(stats["removed_ratio"], 0.5 / 47**0.5)
        self.assertEqual(stats["qkv_plastic_overlap"], (0.5, 0.0, 0.0))
        self.assertEqual(stats["qkv_selected"], (2, 0, 0))
        self.assertEqual(stats["merge_error"], 0.0)
        self.assertTrue(torch.equal(module.qkv.weight, before))
        self.assertTrue(module.global_conflict_masks_active)

    def test_after_task_merges_global_gated_update_once_and_releases_masks(self):
        module = self._make_attention()
        module.before_task(1)
        with torch.no_grad():
            module.S_lora[1].A.weight.fill_(0.25)
            module.S_lora[1].B.weight.fill_(1.0)
            module.P_lora[1].B.weight.zero_()
        shared = torch.zeros_like(module.qkv.weight)
        shared[0, 0] = 1.0
        module.set_global_conflict_masks(shared, torch.zeros_like(shared))

        before = module.qkv.weight.detach().clone()
        raw_shared = module.slora_gamma * (
            module.S_lora[1].B_weight.detach() @ module.S_lora[1].A_weight.detach()
        )
        raw_private = module.plora_gamma * (
            module.P_lora[1].B_weight.detach() @ module.P_lora[1].A_weight.detach()
        )
        ratio, strength = module._conflict_parameters()
        expected_update = module._compose_merge_delta(
            raw_shared,
            isolated=False,
            conflict_ratio=ratio,
            conflict_strength=strength,
        ) + module._compose_merge_delta(
            raw_private,
            isolated=True,
            conflict_ratio=ratio,
            conflict_strength=strength,
        )

        module.after_task(1)

        self.assertTrue(torch.allclose(module.qkv.weight, before + expected_update))
        self.assertIsNone(module.S_lora[1])
        self.assertIsNone(module.P_lora[1])
        self.assertFalse(module.global_conflict_masks_active)
        self.assertEqual(module.global_s_conflict_mask.numel(), 0)

    def test_after_task_logs_private_applied_mask_before_merge(self):
        module = self._make_attention(granularity="projection")
        module.before_task(1)
        module.general_mask.zero_()
        module.general_mask[0, 0] = 1.0
        with torch.no_grad():
            module.S_lora[1].B.weight.zero_()
            module.P_lora[1].A.weight.fill_(0.25)
            module.P_lora[1].B.weight.fill_(1.0)
        private = torch.zeros_like(module.qkv.weight)
        private[0, 0] = 1.0
        private[1, 0] = 1.0
        module.set_global_conflict_masks(torch.zeros_like(private), private)

        with self.assertLogs(level="INFO") as captured:
            module.after_task(1)

        diagnostic = next(
            line for line in captured.output if "P applied merge diagnostic" in line
        )
        self.assertIn("selected=2, selected_plastic=1, selected_active=1", diagnostic)
        self.assertIn("merge_error=0.000e+00", diagnostic)


if __name__ == "__main__":
    unittest.main()
