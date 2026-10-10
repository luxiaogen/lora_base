"""固定覆盖、容量和冲突公式，只恢复D_t对保护强度的调节。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test import test_ncm_overlap_strengths as overlap_tests
from test import test_ncm_direct_strengths as direct_tests
from test import test_dualmask_boundary_fixes as boundary_tests


class NCMDemandRestoreTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_demand_restore_3090.json'
        return runner.settings_for('3090', 'DEMAND_seed1993')

    def test_only_demand_control_changes(self):
        reference = overlap_tests.NCMOverlapStrengthTests().settings()
        candidate = self.settings()
        self.assertEqual({k for k in reference.keys() | candidate.keys()
                          if reference.get(k) != candidate.get(k)},
                         {'dual_mask_ncm_direct_strengths', 'wandb_group'})
        self.assertFalse(candidate['dual_mask_ncm_direct_strengths'])
        self.assertEqual(candidate['dual_mask_fixed_coverage'], .9)
        self.assertEqual(candidate['dual_mask_private_rank'], 64)
        self.assertEqual(candidate['dual_mask_ncm_conflict_mode'], 'scaled')
        self.assertEqual(runner.modes('3090'), ['DEMAND_seed1993'])
        runner.validate_settings('3090')

    def test_restored_protection_keeps_fixed_coverage_rank_and_conflict(self):
        module = direct_tests.NCMDirectStrengthTests().module()
        module.args.update(self.settings(), seed=1993)
        module.set_pretrained_competence(.72, .3)
        module.before_task(1)
        self.assertAlmostEqual(module.effective_protect_strength, .504)
        self.assertEqual(module.effective_energy_coverage, .9)
        self.assertEqual(module.P_lora[1].r, 64)
        for risk in (0., .08, .1913, 1.):
            module.set_pretrained_old_overlap_risk(risk)
            self.assertEqual(module._conflict_parameters(), (.1, min(.5 * (1 + risk), 1.)))

    def test_real_incremental_probe_uses_demand(self):
        prepare = boundary_tests.function_from_source((ROOT / 'methods/dlora.py').read_text(), '_prepare_w0_prototypes')
        learner, module = boundary_tests.BoundaryFixTests().prototype_fixture(True)
        learner.args.update(self.settings())
        learner.args['seed'] = 1993
        module._init_params(learner.args)
        module.cur_task = 1
        with patch('utils.dual_mask_metrics.split_prototype_ncm_diagnostics',
                   return_value=(.5, .3)) as diagnostic:
            prepare(learner, None)
            diagnostic.assert_called_once()
        self.assertEqual(learner._w0_plasticity_demand, .3)
        self.assertAlmostEqual(module.effective_protect_strength, learner._w0_competence_new * .7)
        self.assertEqual(module._conflict_parameters()[1], min(.5 * (1 + learner._w0_old_overlap_risk), 1.))


if __name__ == '__main__':
    unittest.main()
