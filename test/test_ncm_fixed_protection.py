"""固定保护强度，保留原 NCM 竞争风险冲突公式。"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test import test_ncm_demand_restore as demand_tests
from test import test_ncm_overlap_strengths as overlap_tests
from test import test_ncm_direct_strengths as direct_tests
from test import test_dualmask_boundary_fixes as boundary_tests


class NCMFixedProtectionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_fixed_protection_3090.json'
        return runner.settings_for('3090', 'FIXED_seed1993')

    def test_fixed_alpha_is_the_only_effective_control_change(self):
        overlap = overlap_tests.NCMOverlapStrengthTests().settings()
        demand = demand_tests.NCMDemandRestoreTests().settings()
        candidate = self.settings()
        difference = lambda reference: {k for k in reference.keys() | candidate.keys()
                                       if reference.get(k) != candidate.get(k)}
        self.assertEqual(difference(overlap), {'dual_mask_fixed_protect_strength', 'wandb_group'})
        self.assertEqual(difference(demand), {'dual_mask_fixed_protect_strength',
                                           'dual_mask_ncm_direct_strengths', 'wandb_group'})
        self.assertEqual(runner.modes('3090'), ['FIXED_seed1993'])
        runner.validate_settings('3090')

    def test_incremental_probe_skips_demand_but_keeps_competition_risk(self):
        prepare = boundary_tests.function_from_source((ROOT / 'methods/dlora.py').read_text(), '_prepare_w0_prototypes')
        learner, module = boundary_tests.BoundaryFixTests().prototype_fixture(True)
        learner.args.update(self.settings(), seed=1993)
        module._init_params(learner.args)
        module.cur_task = 1
        with patch('utils.dual_mask_metrics.split_prototype_ncm_diagnostics',
                   side_effect=AssertionError('增量阶段不应计算D_t')):
            prepare(learner, None)
        self.assertIsNone(learner._w0_plasticity_demand)
        module.before_task(1)
        self.assertEqual(module.effective_protect_strength, .5)
        self.assertEqual(module.effective_energy_coverage, .9)
        self.assertEqual(module.P_lora[1].r, 64)
        self.assertEqual(module._conflict_parameters(),
                         (.1, min(.5 * (1 + learner._w0_old_overlap_risk), 1.)))

    def test_task0_rng_forward_and_gradients_match_demand_reference(self):
        first = direct_tests.NCMDirectStrengthTests().module()
        first.args.update(demand_tests.NCMDemandRestoreTests().settings(), seed=1993)
        second = copy.deepcopy(first)
        second.args.update(self.settings(), seed=1993)
        state = torch.get_rng_state().clone()
        first.set_pretrained_competence(.8, .25)
        first.before_task(0)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        second.set_pretrained_competence(.8, .25)
        second.before_task(0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(first.effective_protect_strength, second.effective_protect_strength)
        x = torch.randn(3, 4)
        outputs = [m._contrib_from_units(x, 0) for m in (first, second)]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for output in outputs:
            output.sum().backward()
        torch.testing.assert_close(first.S_lora[0].B_weight.grad,
                                   second.S_lora[0].B_weight.grad, rtol=0, atol=0)
        prepare = boundary_tests.function_from_source((ROOT / 'methods/dlora.py').read_text(), '_prepare_w0_prototypes')
        learner, _ = boundary_tests.BoundaryFixTests().prototype_fixture(True)
        learner._cur_task = 0
        learner.args.update(self.settings(), seed=1993)
        with patch('utils.dual_mask_metrics.split_prototype_ncm_diagnostics',
                   return_value=(.5, .25)) as diagnostic:
            prepare(learner, None)
            diagnostic.assert_called_once()

    def test_permissions_zero_b_gradient_and_merge_use_fixed_alpha(self):
        module = direct_tests.NCMDirectStrengthTests().module()
        module.args.update(self.settings(), seed=1993)
        module.set_pretrained_competence(.72, 0.)
        module.set_pretrained_old_overlap_risk(.08)
        module.before_task(1)
        module.general_mask.zero_()
        module.general_mask[-1, -1] = 1.
        delta = torch.arange(1, module.qkv.weight.numel() + 1, dtype=torch.float32).reshape_as(module.qkv.weight)
        self.assertAlmostEqual(module._safe_delta(delta, False)[-1, -1].item(),
                               delta[-1, -1].item() * .5 * .46, places=5)
        self.assertEqual(module._safe_delta(delta, True)[-1, -1].item(), 0.)
        x = torch.randn(3, 4)
        module._contrib_from_units(x, 1).sum().backward()
        for unit in (module.S_lora[1], module.P_lora[1]):
            self.assertTrue(torch.isfinite(unit.B_weight.grad).all())
            self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
        with torch.no_grad():
            module.S_lora[1].B_weight.fill_(.01)
            module.P_lora[1].B_weight.fill_(.02)
        expected = module._contrib_from_units(x, 1)
        original = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(torch.nn.functional.linear(x, module.qkv.weight - original),
                                   expected, rtol=1e-4, atol=1e-7)
        merged = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
