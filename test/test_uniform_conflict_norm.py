import copy
import unittest

import torch
from torch.nn import functional as F

from test import test_p_conflict_freeze


class UniformConflictNormTests(unittest.TestCase):
    def make(self, task=1):
        module = test_p_conflict_freeze.PConflictFreezeTests().make_attention(freeze=0, task=task)
        module.dual_mask_task0_gate_mode = 'unmasked'
        module.dual_mask_uniform_norm_matched = True
        return module

    def test_same_removed_norm_both_branches_and_detached_strength(self):
        module = self.make()
        selective = copy.deepcopy(module)
        selective.dual_mask_uniform_norm_matched = False
        for isolated in (False, True):
            raw = torch.arange(48, dtype=torch.float32).reshape(12, 4).requires_grad_()
            base, mask = module._merge_base_and_conflict(raw, isolated, .1)
            actual = module._safe_delta(raw, isolated)
            reference = selective._safe_delta(raw, isolated)
            torch.testing.assert_close((base - actual).norm(), (base - reference).norm())
            alpha = module._uniform_conflict_strength(base, mask, .5)
            self.assertFalse(alpha.requires_grad)
            actual.sum().backward()
            self.assertIsNotNone(raw.grad)
            if isolated:
                self.assertTrue(torch.equal(actual[:, 3], torch.zeros(12)))

    def test_zero_and_task0(self):
        module = self.make()
        zero = torch.zeros(12, 4)
        self.assertTrue(torch.equal(module._safe_delta(zero, True), zero))
        module = self.make(task=0)
        raw = torch.randn(12, 4)
        self.assertTrue(torch.equal(module._safe_delta(raw, False), raw))

    def test_forward_merge_identity(self):
        module = self.make()
        x = torch.randn(7, 4)
        before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
        module.after_task(1)
        torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
