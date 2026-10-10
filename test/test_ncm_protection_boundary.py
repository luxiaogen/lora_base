"""保护强度端点：只移除S软保护或将其变为硬限制。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test import test_ncm_fixed_protection as fixed_tests
from test import test_ncm_direct_strengths as direct_tests


class NCMProtectionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self, variant):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_protection_boundary_3090.json'
        return runner.settings_for('3090', variant + '_seed1993')

    def test_only_alpha_changes_and_old_queue_is_preserved(self):
        reference = fixed_tests.NCMFixedProtectionTests().settings()
        for name, alpha in (('A0', 0.), ('A1', 1.)):
            candidate = self.settings(name)
            differences = {key for key in candidate.keys() | reference.keys()
                           if candidate.get(key) != reference.get(key)}
            self.assertEqual(differences, {'dual_mask_fixed_protect_strength', 'wandb_group'})
            self.assertEqual(candidate['dual_mask_fixed_protect_strength'], alpha)
            runner.validate_settings('3090')
        self.assertEqual(runner.modes('3090'), ['A0_seed1993', 'A1_seed1993'])
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_protection_sensitivity_3090.json'
        self.assertEqual(runner.modes('3090'), ['A025_seed1993', 'A075_seed1993'])
        runner.validate_settings('3090')

    def test_wrong_boundary_recipe_is_rejected(self):
        self.settings('A0')
        original = runner.settings_for
        for override in ({'dual_mask_fixed_protect_strength': .5},
                         {'dual_mask_ncm_conflict_mode': 'direct'}):
            with patch.object(runner, 'settings_for', side_effect=lambda *a, **kw:
                              dict(original(*a, **kw), **override)):
                with self.assertRaises(ValueError):
                    runner.validate_settings('3090')

    def test_task0_matches_reference_including_rng_and_gradient(self):
        for name in ('A0', 'A1'):
            reference = fixed_tests.NCMFixedProtectionTests()
            with patch.object(reference, 'settings', return_value=self.settings(name)):
                reference.test_task0_rng_forward_and_gradients_match_demand_reference()

    def test_endpoint_permissions_learning_gradient_and_merge(self):
        for name, alpha in (('A0', 0.), ('A1', 1.)):
            module = direct_tests.NCMDirectStrengthTests().module()
            module.args.update(self.settings(name), seed=1993)
            module.set_pretrained_competence(.72, 0.)
            module.set_pretrained_old_overlap_risk(.08)
            module.before_task(1)
            self.assertEqual(module.effective_protect_strength, alpha)
            self.assertEqual(module._conflict_parameters(), (.1, .54))
            module.general_mask.zero_()
            module.general_mask[-1, -1] = 1.
            delta = torch.arange(1, module.qkv.weight.numel() + 1,
                                 dtype=torch.float32).reshape_as(module.qkv.weight)
            s = module._safe_delta(delta, False)
            p = module._safe_delta(delta, True)
            self.assertAlmostEqual(s[-1, -1].item(),
                                   delta[-1, -1].item() * (1 - alpha) * .46, places=5)
            self.assertEqual(p[-1, -1].item(), 0.)
            x = torch.randn(3, 4)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertTrue(torch.isfinite(unit.B_weight.grad).all())
                self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
            with torch.no_grad():
                module.S_lora[1].B_weight.fill_(.01)
                module.P_lora[1].B_weight.fill_(.02)
            before = module._contrib_from_units(x, 1)
            weight = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(torch.nn.functional.linear(x, module.qkv.weight - weight),
                                       before, rtol=1e-4, atol=1e-7)
            merged = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
