"""固定的是有效 BA 的坐标，B 仍学习；前向和合并使用同一支持。"""
import copy
import re
import unittest

import torch
from torch.nn import functional as F

from utils.p_fixed_support import select_support, support_update


class PFixedSupportTests(unittest.TestCase):
    def make(self, mode='magnitude', task=1):
        from test.test_p_conflict_freeze import PConflictFreezeTests
        module = PConflictFreezeTests().make_attention(freeze=0, task=task)
        module.args.update(p_fixed_support=mode, p_fixed_support_warmup_epochs=2,
                           p_fixed_support_keep_ratio=.5, p_fixed_support_audit=True,
                           dual_mask_private_conflict_mode='global')
        module.dual_mask_task0_gate_mode = 'unmasked'
        module.S_lora.append(None)
        module.P_lora.append(None)
        module.n_tasks = 3
        return module

    def test_exact_plastic_count_stable_ties_and_zero_scores(self):
        raw = torch.tensor([[0., -4., 4., 3.], [9., 2., 1., 8.]])
        plastic = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
        selected = select_support(raw, plastic, .5)
        self.assertEqual(selected.sum().item(), 2)
        self.assertEqual(selected.flatten().nonzero().flatten().tolist(), [1, 2])
        zeros = select_support(torch.zeros_like(raw), plastic, .5)
        self.assertEqual(zeros.flatten().nonzero().flatten().tolist(), [1, 2])
        self.assertFalse((selected & ~plastic).any())
        self.assertFalse(select_support(raw, torch.zeros_like(plastic), .5).any())
        self.assertTrue(torch.equal(select_support(raw, plastic, 1), plastic))

    def test_random_count_reproducibility_and_rng(self):
        raw = torch.arange(48).reshape(12, 4).float()
        plastic = raw.remainder(3) != 0
        state = torch.random.get_rng_state().clone()
        first = select_support(raw, plastic, .5, random=True, seed=73)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, select_support(raw, plastic, .5, random=True, seed=73)))
        self.assertEqual(first.sum(), select_support(raw, plastic, .5).sum())
        self.assertFalse((first & ~plastic).any())

    def test_uniform_matches_removed_norm_and_handles_zero(self):
        raw = torch.tensor([[1., -2., 4.], [3., 5., -6.]], requires_grad=True)
        plastic = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
        support = select_support(raw, plastic, .5)
        hard = support_update(raw, plastic, support, 'magnitude')
        uniform = support_update(raw, plastic, support, 'uniform_matched')
        torch.testing.assert_close((hard['base'] - hard['safe']).norm(),
                                   (uniform['base'] - uniform['safe']).norm())
        self.assertFalse(uniform['coefficient'].requires_grad)
        self.assertTrue(((uniform['safe'].abs() <= raw.abs()) | ~plastic).all())
        self.assertEqual(uniform['safe'][~plastic].abs().sum(), 0)
        zero = torch.zeros_like(raw, requires_grad=True)
        state = support_update(zero, plastic, support, 'uniform_matched')
        state['safe'].sum().backward()
        torch.testing.assert_close(zero.grad, plastic.float())

    def test_warmup_fixed_support_and_b_keeps_learning(self):
        module = self.make()
        raw = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
        before = module._safe_delta(raw, True).detach().clone()
        module.update_p_fixed_support(1)
        self.assertIsNone(module.p_fixed_support_mask)
        torch.testing.assert_close(module._safe_delta(raw, True), before)
        module.update_p_fixed_support(2)
        saved = module.p_fixed_support_mask.clone()
        expected = raw * (1 - module.general_mask) * saved
        torch.testing.assert_close(module._safe_delta(raw, True), expected)
        unit = module.P_lora[1]
        initial = unit.B_weight.detach().clone()
        F.linear(torch.ones(2, 4), module._safe_delta(raw, True)).sum().backward()
        self.assertGreater(unit.B_weight.grad.abs().sum(), 0)
        self.assertFalse(unit.A_weight.requires_grad)
        with torch.no_grad():
            unit.B_weight.add_(.01 * unit.B_weight.grad)
        module.update_p_fixed_support(3)
        self.assertTrue(torch.equal(saved, module.p_fixed_support_mask))
        self.assertFalse(torch.equal(initial, unit.B_weight))
        after = module._safe_delta(unit.B_weight @ unit.A_weight, True)
        self.assertEqual(after[~saved].abs().sum(), 0)
        self.assertFalse(torch.equal(expected.detach(), after.detach()))

    def test_default_task0_s_branch_and_rng_unchanged(self):
        module = self.make('off')
        reference = copy.deepcopy(module)
        reference.args.pop('p_fixed_support')
        state = torch.random.get_rng_state().clone()
        module.update_p_fixed_support(2)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        raw = torch.randn_like(module.qkv.weight)
        for isolated in (False, True):
            torch.testing.assert_close(module._safe_delta(raw, isolated),
                reference._safe_delta(raw, isolated), rtol=0, atol=0)
        routed = copy.deepcopy(module)
        routed.args['p_fixed_support'] = 'magnitude'
        torch.testing.assert_close(routed._safe_delta(raw, False),
            reference._safe_delta(raw, False), rtol=0, atol=0)
        for mode in ('magnitude', 'random', 'uniform_matched'):
            task0 = self.make(mode, task=0)
            task0.update_p_fixed_support(2)
            self.assertIsNone(task0.p_fixed_support_mask)
            torch.testing.assert_close(task0._safe_delta(raw, False), raw)

    def test_zero_b_warmup_has_gradient(self):
        module = self.make()
        unit = module.P_lora[1]
        with torch.no_grad():
            unit.B_weight.zero_()
        module._masked_unit_forward(torch.ones(2, 4), unit, True).sum().backward()
        self.assertGreater(unit.B_weight.grad.abs().sum(), 0)

    def test_forward_merge_same_and_merge_once_for_all_modes(self):
        for mode in ('magnitude', 'random', 'uniform_matched'):
            module = self.make(mode)
            module.update_p_fixed_support(2)
            x = torch.randn(3, 4)
            expected = module.qkv(x) + module._contrib_from_units(x, 1)
            anchor = module.pretrained_weight.clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv(x), expected, atol=2e-6, rtol=2e-6)
            merged = module.qkv.weight.clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)
            self.assertIsNone(module.p_fixed_support_mask)
            self.assertIsNone(module.P_lora[1])
            torch.testing.assert_close(module.pretrained_weight, anchor, rtol=0, atol=0)
            self.assertEqual(module._contrib_from_units(x, 1).abs().sum(), 0)
            module.before_task(2)
            self.assertIsNone(module.p_fixed_support_mask)

    def test_merge_diagnostics_use_actual_support_not_original_conflict_gate(self):
        for mode in ('magnitude', 'random', 'uniform_matched'):
            module = self.make(mode)
            module.update_p_fixed_support(2)
            raw = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
            state = support_update(raw, 1 - module.general_mask, module.p_fixed_support_mask, mode)
            expected = 1 - float(state['safe'].detach().norm() / state['base'].detach().norm())
            with self.assertLogs(level='INFO') as messages:
                module.after_task(1)
            diagnostic = next(s for s in messages.output if 'P applied merge diagnostic' in s)
            self.assertLess(float(re.search(r'merge_error=([\deE.+-]+)', diagnostic)[1]), 1e-6)
            self.assertIn('selected=' + str(int(state['applied'].sum())) + ',', diagnostic)
            self.assertAlmostEqual(float(module.last_private_conflict_gate_suppression), expected, places=6)


if __name__ == '__main__':
    unittest.main()
