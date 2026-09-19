import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.attention import Attention_LoRA
from utils.p_region_training import shrink_private_region


def make_attention(mode="none", amount=0.5):
    module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=3)
    module._init_params(dict(use_slora=True, use_plora=True, lora_A_init="kaiming",
                            slora_gamma=0.5, plora_gamma=0.75,
                            dual_mask_task0_gate_mode="unmasked",
                            dual_mask_conflict_energy_adaptive=True,
                            dual_mask_conflict_energy_ratio_floor=True,
                            dual_mask_p_region_train_mode=mode, dual_mask_p_region_train_amount=amount))
    return module


class PrivateRegionTrainingTests(unittest.TestCase):
    def test_matched_budget_support_and_gradient(self):
        mask = torch.tensor([[1., 1., 0., 0.]])
        delta = torch.tensor([[1., -2., 3., -4.]], requires_grad=True)
        for amount in (0.25, 0.5):
            c = shrink_private_region(delta, mask, "conflict", amount)
            u = shrink_private_region(delta, mask, "nonconflict", amount)
            expected = amount * min((delta * mask).norm(), (delta * (1-mask)).norm())
            torch.testing.assert_close((delta-c).norm(), expected)
            torch.testing.assert_close((delta-u).norm(), expected)
            torch.testing.assert_close(c * (1-mask), delta * (1-mask))
            torch.testing.assert_close(u * mask, delta * mask)
            grad = torch.autograd.grad(c.sum(), delta, retain_graph=True)[0]
            # Budget coefficients are stop-gradient, not trainable normalizers.
            torch.testing.assert_close(grad, c.detach() / delta.detach())
            self.assertTrue(((grad >= 1-amount) & (grad <= 1)).all())
            u_grad = torch.autograd.grad(u.sum(), delta, retain_graph=True)[0]
            torch.testing.assert_close(u_grad, u.detach() / delta.detach())

    def test_zero_initialization_has_finite_nonzero_gradient(self):
        for mode in ("conflict", "nonconflict"):
            delta = torch.zeros(2, 4, requires_grad=True)
            mask = torch.tensor([[1., 0., 1., 0.]]).expand_as(delta)
            result = shrink_private_region(delta, mask, mode, 0.5)
            result.sum().backward()
            self.assertTrue(torch.equal(delta.grad, torch.ones_like(delta)))
            for empty_mask in (torch.zeros_like(mask), torch.ones_like(mask)):
                x = torch.randn_like(delta)
                self.assertTrue(torch.equal(x, shrink_private_region(x, empty_mask, mode, 0.5)))

    def test_none_and_zero_amount_exact(self):
        x = torch.randn(3, 4)
        mask = torch.ones_like(x)
        self.assertIs(x, shrink_private_region(x, mask, "none", 0.5))
        self.assertIs(x, shrink_private_region(x, mask, "conflict", 0.0))

    def test_invalid_settings(self):
        for mode, amount in (("typo", 0.5), ("conflict", -0.1), ("conflict", 1.1), ("conflict", float("nan"))):
            with self.assertRaises(ValueError):
                make_attention(mode, amount)
        for unsupported in ({"dual_mask_conflict_merge_mode": "none"},
                            {"dual_mask_safe_residual_enabled": True},
                            {"dual_mask_functional_merge_calibration": True}):
            with self.assertRaises(ValueError):
                module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=3)
                module._init_params(dict(use_slora=True, use_plora=True,
                                         dual_mask_p_region_train_mode="conflict", **unsupported))

    def test_zero_amount_forward_merge_is_exact_baseline(self):
        torch.manual_seed(5)
        base = make_attention()
        base.before_task(0); base.after_task(0); base.before_task(1)
        with torch.no_grad():
            base.P_lora[1].B.weight.normal_(0, 0.1)
        x = torch.randn(3, 2, 4)
        for mode in ("conflict", "nonconflict"):
            candidate = copy.deepcopy(base)
            candidate.dual_mask_p_region_train_mode = mode
            candidate.dual_mask_p_region_train_amount = 0.0
            reference = copy.deepcopy(base)
            candidate.eval(); reference.eval()
            self.assertTrue(torch.equal(reference(x, 1), candidate(x, 1)))
            reference.after_task(1); candidate.after_task(1)
            self.assertTrue(torch.equal(reference.qkv.weight, candidate.qkv.weight))

    def test_task0_and_shared_branch_unchanged(self):
        torch.manual_seed(4)
        base = make_attention()
        for mode in ("conflict", "nonconflict"):
            candidate = copy.deepcopy(base)
            candidate.dual_mask_p_region_train_mode = mode
            delta = torch.randn(12, 4)
            base.before_task(0); candidate.before_task(0)
            self.assertTrue(torch.equal(base._safe_delta(delta, False), candidate._safe_delta(delta, False)))
            base.cur_task = candidate.cur_task = 1
            self.assertTrue(torch.equal(base._safe_delta(delta, False), candidate._safe_delta(delta, False)))

    def test_multi_task_optimizer_smoke_and_merge_equivalence(self):
        for mode in ("none", "conflict", "nonconflict"):
            torch.manual_seed(123)
            module = make_attention(mode)
            head = nn.Linear(4, 2)
            x, y = torch.randn(6, 3, 4), torch.tensor([0, 1, 0, 1, 0, 1])
            anchor = module.pretrained_weight.clone()
            for task in range(3):
                module.before_task(task)
                module.set_task_and_stage(task, 0)
                optimizer = torch.optim.SGD([p for p in list(module.parameters()) + list(head.parameters()) if p.requires_grad], lr=0.01)
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(head(module(x, task)[:, 0]), y)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    unit = module.S_lora[task] if task == 0 else module.P_lora[task]
                    self.assertTrue(torch.isfinite(unit.B.weight.grad).all())
                    self.assertGreater(unit.B.weight.grad.norm().item(), 0)
                    optimizer.step()
                module.eval()
                before = module(x, task).detach()
                module.after_task(task)
                after = module(x, task).detach()
                torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
                self.assertIsNone(module.P_lora[task])
                self.assertIsNone(module.S_lora[task])
                self.assertTrue(torch.equal(anchor, module.pretrained_weight))


if __name__ == "__main__":
    unittest.main()
