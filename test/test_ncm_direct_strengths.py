"""固定位置与容量，只将 NCM 两种准确率用于增量更新强度。"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test import test_ncm_controller_restore as restore_tests
from test import test_dualmask_boundary_fixes as boundary_tests


class NCMDirectStrengthTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_direct_strengths_3090.json'
        return runner.settings_for('3090', 'DIRECT_seed1993')

    def module(self):
        helper = restore_tests.NCMControllerRestoreTests()
        module = helper.module(False)
        module.args.update(self.settings())
        module.args['seed'] = 1993
        module._init_params(module.args)
        return module

    def test_recipe_is_fixed_m90_with_two_direct_strengths(self):
        reference = restore_tests.NCMControllerRestoreTests().settings(False)
        candidate = self.settings()
        self.assertEqual({k for k in reference.keys() | candidate.keys()
                          if reference.get(k) != candidate.get(k)},
                         {'dual_mask_fixed_protect_strength', 'dual_mask_fixed_conflict_strength',
                          'dual_mask_ncm_direct_strengths', 'wandb_group'})
        self.assertEqual(candidate['dual_mask_fixed_coverage'], .9)
        self.assertEqual(candidate['dual_mask_private_rank'], 64)
        self.assertIsNone(candidate['dual_mask_fixed_protect_strength'])
        self.assertIsNone(candidate['dual_mask_fixed_conflict_strength'])
        self.assertEqual(runner.modes('3090'), ['DIRECT_seed1993'])
        runner.validate_settings('3090')

    def test_incremental_probe_omits_loss_and_uses_new_accuracy(self):
        prepare = boundary_tests.function_from_source((ROOT / 'methods/dlora.py').read_text(), '_prepare_w0_prototypes')
        learner, module = boundary_tests.BoundaryFixTests().prototype_fixture(True)
        learner.args.update(dual_mask_ncm_direct_strengths=True,
                            dual_mask_plasticity_adaptive=True)
        module._init_params(dict(learner.args,
            dual_mask_competence_adaptive=True, dual_mask_plasticity_adaptive=True,
            dual_mask_protect_strength_mode='competence'))
        module.cur_task = 1
        with patch('utils.dual_mask_metrics.split_prototype_ncm_diagnostics',
                   side_effect=AssertionError('incremental D_t must not be calculated')):
            prepare(learner, None)
        self.assertIsNone(learner._w0_plasticity_demand)
        self.assertEqual(module.effective_protect_strength, learner._w0_competence_new)
        self.assertEqual(module._conflict_parameters()[1], learner._w0_old_overlap_risk)

    def test_task0_keeps_loss_controller_rng_forward_and_gradient(self):
        first = restore_tests.NCMControllerRestoreTests().module(False)
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
        learner.args.update(dual_mask_ncm_direct_strengths=True, dual_mask_plasticity_adaptive=True)
        with patch('utils.dual_mask_metrics.split_prototype_ncm_diagnostics',
                   return_value=(.5, .25)) as diagnostic:
            prepare(learner, None)
            diagnostic.assert_called_once()

    def test_risk_zero_fraction_and_full_range(self):
        module = self.module()
        module.cur_task = 1
        for risk in (0., .08, .2, 1.):
            module.set_pretrained_old_overlap_risk(risk)
            self.assertEqual(module._conflict_parameters(), (.1, risk))
        module.args['dual_mask_fixed_conflict_strength'] = .5
        self.assertEqual(module._conflict_parameters(), (.1, .5))

    def test_hand_computed_permissions_and_single_merge(self):
        module = self.module()
        module.set_pretrained_competence(.72, 0.)
        module.set_pretrained_old_overlap_risk(.08)
        module.before_task(1)
        module.general_mask.zero_()
        module.general_mask[0, 0] = 1.
        module.global_s_conflict_mask = torch.zeros_like(module.general_mask)
        module.global_s_conflict_mask[0, 0] = 1.
        module.global_p_conflict_mask = module.global_s_conflict_mask.clone()
        delta = torch.ones_like(module.qkv.weight, requires_grad=True)
        shared = module._safe_delta(delta, False)
        private = module._safe_delta(delta, True)
        self.assertAlmostEqual(shared[0, 0].item(), .28 * .92, places=6)
        self.assertEqual(private[0, 0].item(), 0.)
        self.assertEqual(private[1, 0].item(), 1.)
        shared.sum().backward()
        self.assertGreater(float(delta.grad.norm()), 0.)
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
        actual = torch.nn.functional.linear(x, module.qkv.weight - original)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
        merged = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)

    def test_reuses_single_seed_analyzer(self):
        self.settings()
        with tempfile.TemporaryDirectory() as tmp, patch(
                'analyze_prototype_position.summarize_saved') as summarize:
            runner.summarize_safely(Path(tmp), '3090', [])
            summarize.assert_called_once_with(Path(tmp))


if __name__ == '__main__':
    unittest.main()
