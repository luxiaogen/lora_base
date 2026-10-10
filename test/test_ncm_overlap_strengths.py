"""与直接R_old候选相比，仅恢复原0.5*(1+R_old)冲突强度。"""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test import test_ncm_direct_strengths as direct_tests


class NCMOverlapStrengthTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_overlap_strengths_3090.json'
        return runner.settings_for('3090', 'OVERLAP_seed1993')

    def test_only_conflict_strength_changes(self):
        reference = direct_tests.NCMDirectStrengthTests().settings()
        candidate = self.settings()
        self.assertEqual({k for k in reference.keys() | candidate.keys()
                          if reference.get(k) != candidate.get(k)},
                         {'dual_mask_ncm_conflict_mode', 'wandb_group'})
        self.assertIsNone(candidate['dual_mask_fixed_conflict_strength'])
        self.assertEqual(candidate['dual_mask_ncm_conflict_mode'], 'scaled')
        self.assertIsNone(candidate['dual_mask_fixed_protect_strength'])
        self.assertEqual(runner.modes('3090'), ['OVERLAP_seed1993'])
        runner.validate_settings('3090')

    def test_original_overlap_formula_and_protection_remains_accuracy(self):
        module = direct_tests.NCMDirectStrengthTests().module()
        module.args.update(self.settings(), seed=1993)
        module.set_pretrained_competence(.72, 0.)
        module.before_task(1)
        for risk in (0., .08, .1913, 1.):
            module.set_pretrained_old_overlap_risk(risk)
            self.assertEqual(module._conflict_parameters(), (.1, min(.5 * (1 + risk), 1.)))
        self.assertAlmostEqual(module.effective_protect_strength, .72)
        self.assertEqual(module.effective_energy_coverage, .9)
        self.assertEqual(module.P_lora[1].r, 64)

    def test_scaled_forward_and_merge_use_same_strength(self):
        module = direct_tests.NCMDirectStrengthTests().module()
        module.args.update(self.settings(), seed=1993)
        module.set_pretrained_competence(.72, 0.)
        module.set_pretrained_old_overlap_risk(.08)
        module.before_task(1)
        module.general_mask.zero_()
        module.general_mask[-1, -1] = 1.
        delta = torch.arange(1, module.qkv.weight.numel() + 1, dtype=torch.float32).reshape_as(module.qkv.weight)
        shared = module._safe_delta(delta, False)
        self.assertAlmostEqual(shared[-1, -1].item(), delta[-1, -1].item() * .28 * .46, places=5)
        with torch.no_grad():
            module.S_lora[1].B_weight.fill_(.01)
            module.P_lora[1].B_weight.fill_(.02)
        x = torch.randn(3, 4)
        expected = module._contrib_from_units(x, 1)
        original = module.qkv.weight.detach().clone()
        module.after_task(1)
        actual = torch.nn.functional.linear(x, module.qkv.weight - original)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
        merged = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
