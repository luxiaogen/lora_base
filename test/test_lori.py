import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import Attention_LoRA, FrozenA_TrainableB  # noqa: E402
from utils.lori import global_topk_masks  # noqa: E402


class LoRITests(unittest.TestCase):
    @staticmethod
    def make_args(**overrides):
        args = {
            "use_slora": True,
            "use_plora": True,
            "lori_s_enabled": True,
            "dual_mask_enabled": True,
        }
        args.update(overrides)
        return args

    def test_global_topk_masks_select_exact_budget_across_tensors(self):
        tensors = [
            torch.tensor([[1.0, -8.0], [3.0, 2.0]]),
            torch.tensor([[7.0, 4.0], [-6.0, 5.0]]),
        ]

        masks = global_topk_masks(tensors, retain_ratio=0.25)

        self.assertEqual(sum(mask.sum().item() for mask in masks), 2)
        self.assertTrue(masks[0][0, 1])
        self.assertTrue(masks[1][0, 0])

    def test_global_topk_masks_keep_exact_count_when_values_tie(self):
        tensors = [torch.ones(2, 5)]

        masks = global_topk_masks(tensors, retain_ratio=0.3)

        self.assertEqual(masks[0].sum().item(), 3)

    def test_sparse_unit_masks_forward_and_gradients(self):
        unit = FrozenA_TrainableB(
            dim_in=2,
            dim_out=2,
            r=2,
            A_init=torch.eye(2),
            B_init=torch.zeros(2, 2),
        )
        mask = torch.tensor([[True, False], [False, True]])
        unit.set_lori_mask(mask)
        with torch.no_grad():
            unit.B.weight.fill_(1.0)

        output = unit(torch.tensor([[2.0, 3.0]]))
        output.sum().backward()
        unit.apply_lori_gradient_mask()

        self.assertTrue(torch.equal(output, torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.equal(
            unit.B.weight.grad,
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        ))

    def test_reset_for_sparse_stage_zeros_b_and_keeps_a(self):
        unit = FrozenA_TrainableB(
            dim_in=2,
            dim_out=2,
            r=2,
            A_init=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            B_init=torch.ones(2, 2),
        )
        initial_a = unit.A.weight.detach().clone()

        unit.reset_for_lori_sparse(torch.eye(2, dtype=torch.bool))

        self.assertTrue(torch.equal(unit.A.weight, initial_a))
        self.assertEqual(unit.B.weight.count_nonzero().item(), 0)
        self.assertTrue(unit.lori_sparse_active)

    def test_sparse_optimizer_step_keeps_unselected_b_entries_zero(self):
        unit = FrozenA_TrainableB(
            dim_in=2,
            dim_out=2,
            r=2,
            A_init=torch.eye(2),
            B_init=torch.ones(2, 2),
        )
        mask = torch.tensor([[True, False], [False, True]])
        unit.reset_for_lori_sparse(mask)
        optimizer = torch.optim.SGD(
            [unit.B.weight],
            lr=0.1,
            momentum=0.9,
            weight_decay=0.01,
        )

        unit(torch.tensor([[2.0, 3.0]])).sum().backward()
        unit.apply_lori_gradient_mask()
        optimizer.step()

        self.assertEqual(unit.B.weight[~mask].count_nonzero().item(), 0)
        self.assertEqual(unit.B.weight[mask].count_nonzero().item(), 2)

    def test_lori_freezes_task0_a_and_trains_b(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(self.make_args())
        module.before_task(task=0)

        module.set_task_and_stage(task=0, layer_idx=0)

        self.assertFalse(module.S_lora[0].A.weight.requires_grad)
        self.assertTrue(module.S_lora[0].B.weight.requires_grad)

    def test_dual_mask_off_returns_raw_update_for_private_branch(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(self.make_args(dual_mask_enabled=False))
        delta = torch.randn_like(module.qkv.weight)

        safe_delta = module._safe_delta(delta, isolated=True)

        self.assertTrue(torch.equal(safe_delta, delta))


if __name__ == "__main__":
    unittest.main()
