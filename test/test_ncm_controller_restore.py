"""仅恢复三个原有控制量；冲突强度和其余 M90 配方保持不变。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
from test.test_global_conflict_budget import GlobalBudgetSelectionTests


class NCMControllerRestoreTests(unittest.TestCase):
    def setUp(self):
        self.original_spec = runner.SPEC
        self.addCleanup(setattr, runner, 'SPEC', self.original_spec)

    def settings(self, restored):
        runner.SPEC = ROOT / ('scripts/sweeps/imgr10_ncm_controller_restore_3090.json'
                              if restored else 'scripts/sweeps/imgr10_prototype_position.json')
        return runner.settings_for('3090', 'NCM_seed1993' if restored else 'R0_seed1993')

    def test_only_three_training_fields_change(self):
        reference, candidate = self.settings(False), self.settings(True)
        differences = {k for k in reference.keys() | candidate.keys()
                       if reference.get(k) != candidate.get(k)}
        self.assertEqual(differences, {'dual_mask_fixed_coverage',
            'dual_mask_fixed_protect_strength', 'dual_mask_private_rank', 'wandb_group'})
        self.assertIsNone(candidate['dual_mask_fixed_coverage'])
        self.assertIsNone(candidate['dual_mask_fixed_protect_strength'])
        self.assertEqual(candidate['dual_mask_private_rank'], 0)
        self.assertEqual(candidate['dual_mask_fixed_conflict_strength'], .5)
        self.assertEqual(candidate['dual_mask_gradient_route'], 'off')
        self.assertEqual(runner.modes('3090'), ['NCM_seed1993'])
        runner.validate_settings('3090')
        command, _ = runner.command_for('3090', 'NCM_seed1993', Path('/tmp/example/NCM_seed1993'))
        self.assertNotIn('data_path', ' '.join(command))

    def module(self, restored):
        args = dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                    **self.settings(restored))
        args.pop('dual_mask_conflict_granularity')
        args['seed'] = args['seed'][0]  # main.py 在创建模型前将种子列表展开为标量。
        return GlobalBudgetSelectionTests._make_attention('layer', **args)

    def test_original_controller_formula_and_fixed_conflict_priority(self):
        module = self.module(True)
        module.rank = 64
        module.set_pretrained_competence(.8, .25)
        self.assertAlmostEqual(module.pretrained_control_competence, .6)
        self.assertAlmostEqual(module.effective_energy_coverage, .85)
        self.assertAlmostEqual(module.effective_protect_strength, .6)
        self.assertEqual(module.current_private_rank, 35)
        for risk in (0., .2, 1.):
            module.set_pretrained_old_overlap_risk(risk)
            self.assertEqual(module._conflict_parameters(), (.1, .5))
        module.args['dual_mask_fixed_conflict_strength'] = None
        self.assertEqual(module._conflict_parameters(), (.1, 1.))

    def test_task0_rng_output_and_gradients_unchanged(self):
        module = self.module(False)
        reference = copy.deepcopy(module)
        reference.args.update(self.settings(True))
        reference.args['seed'] = 1993
        reference.dual_mask_private_rank = 0
        for model in (module, reference):
            model.set_pretrained_competence(.8, .25)
        state = torch.get_rng_state().clone()
        module.before_task(0)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        reference.before_task(0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        x = torch.randn(3, 4)
        first, second = [m._contrib_from_units(x, 0) for m in (module, reference)]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.sum().backward()
        second.sum().backward()
        torch.testing.assert_close(module.S_lora[0].B_weight.grad,
                                   reference.S_lora[0].B_weight.grad, rtol=0, atol=0)

    def test_reuses_position_evidence_analyzer(self):
        self.settings(True)
        with tempfile.TemporaryDirectory() as tmp, patch(
                'analyze_prototype_position.summarize_saved') as summarize:
            runner.summarize_safely(Path(tmp), '3090', [])
            summarize.assert_called_once_with(Path(tmp))


if __name__ == '__main__':
    unittest.main()
