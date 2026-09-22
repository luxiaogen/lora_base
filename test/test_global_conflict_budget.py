import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import Attention_LoRA  # noqa: E402
from utils.dual_mask_budget import select_global_budget_masks  # noqa: E402


class GlobalBudgetSelectionTests(unittest.TestCase):
    @staticmethod
    def _make_attention(granularity="model"):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params({
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
        })
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


if __name__ == "__main__":
    unittest.main()
