"""仅改变固定保护强度，核对两组配方及实际门控。"""
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
import analyze_prototype_position as analyzer
from test import test_ncm_fixed_protection as fixed_tests
from test import test_ncm_direct_strengths as direct_tests


class NCMProtectionSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)

    def settings(self, variant):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_ncm_protection_sensitivity_3090.json'
        return runner.settings_for('3090', variant + '_seed1993')

    def test_only_alpha_changes_and_queue_runs_two_new_groups(self):
        reference = fixed_tests.NCMFixedProtectionTests().settings()
        for variant, alpha in (('A025', .25), ('A075', .75)):
            candidate = self.settings(variant)
            difference = {k for k in reference.keys() | candidate.keys()
                          if reference.get(k) != candidate.get(k)}
            self.assertEqual(difference, {'dual_mask_fixed_protect_strength', 'wandb_group'})
            self.assertEqual(candidate['dual_mask_fixed_protect_strength'], alpha)
            runner.validate_settings('3090')
            smoke = runner.settings_for('3090', variant + '_seed1993', True)
            self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
        self.assertEqual(runner.modes('3090'), ['A025_seed1993', 'A075_seed1993'])

    def test_validation_rejects_changed_alpha_or_conflict_formula(self):
        self.settings('A025')
        original = runner.settings_for
        for changed in ({'dual_mask_fixed_protect_strength': .5},
                        {'dual_mask_ncm_conflict_mode': 'direct'}):
            with patch.object(runner, 'settings_for',
                              side_effect=lambda *args, **kwargs: dict(original(*args, **kwargs), **changed)):
                with self.assertRaises(ValueError):
                    runner.validate_settings('3090')

    def test_task0_rng_forward_and_gradients_unchanged(self):
        for variant in ('A025', 'A075'):
            reference = fixed_tests.NCMFixedProtectionTests()
            # 复用相同的Task0前向、梯度、RNG及原D_t路径回归检查。
            with patch.object(reference, 'settings', return_value=self.settings(variant)):
                reference.test_task0_rng_forward_and_gradients_match_demand_reference()

    def test_actual_alpha_permissions_gradients_and_merge(self):
        for variant, alpha in (('A025', .25), ('A075', .75)):
            module = direct_tests.NCMDirectStrengthTests().module()
            module.args.update(self.settings(variant), seed=1993)
            module.set_pretrained_competence(.72, 0.)
            module.set_pretrained_old_overlap_risk(.08)
            module.before_task(1)
            self.assertEqual(module.effective_protect_strength, alpha)
            self.assertEqual(module.effective_energy_coverage, .9)
            self.assertEqual(module.P_lora[1].r, 64)
            self.assertEqual(module._conflict_parameters(), (.1, .54))
            module.general_mask.zero_()
            module.general_mask[-1, -1] = 1.
            delta = torch.arange(1, module.qkv.weight.numel() + 1,
                                 dtype=torch.float32).reshape_as(module.qkv.weight)
            self.assertAlmostEqual(module._safe_delta(delta, False)[-1, -1].item(),
                                   delta[-1, -1].item() * (1 - alpha) * .46, places=5)
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

    def test_summary_accepts_planned_alpha_difference_but_not_other_changes(self):
        configs = {name: dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                             **self.settings(name)) for name in ('A025', 'A075')}
        for unexpected_change in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090',
                    sweep_spec='scripts/sweeps/imgr10_ncm_protection_sensitivity_3090.json')))
                records = [dict(mode=name + '_seed1993', status='completed') for name in configs]
                (directory / 'queue.json').write_text(json.dumps(records))
                for record in records:
                    run = directory / record['mode']
                    run.mkdir()
                    (run / 'training.log').write_text('\n'.join(
                        'LoRA learning rates: task={}, epoch={}'.format(task, epoch)
                        for task in range(10) for epoch in range(1, 21)))

                def read_run(_, record):
                    config = dict(configs[record['mode'].split('_seed')[0]])
                    if unexpected_change and record['mode'].startswith('A075'):
                        config['slora_gamma'] = .4
                    row = dict(record, valid_performance=True, Task0=97.1,
                               **{key: 1. for key in analyzer.METRICS})
                    snapshot = dict(mode=record['mode'], machine='3090', phase='formal',
                        effective_config=config, code_revision='same', source_sha256={},
                        software={}, hardware={})
                    return row, snapshot, {key: [] for key in ('tasks', 'epochs', 'costs', 'storage')}

                with patch.object(analyzer, 'read_run', side_effect=read_run), patch.object(analyzer, 'draw'):
                    analyzer.summarize_saved(directory)
                issues = json.loads((directory / 'matching_issues.json').read_text())
                if unexpected_change:
                    self.assertIn('config.slora_gamma', [row.get('field') for row in issues])
                else:
                    self.assertEqual(issues, [])


if __name__ == '__main__':
    unittest.main()
