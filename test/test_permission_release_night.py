import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_permission_release as analysis
import run_permission_release_night as night
import run_core_evidence_night as engine


class NightTests(unittest.TestCase):
    def test_exact_order_protocol_and_single_factors(self):
        self.assertEqual([r['name'] for r in night.variants('3090')], ['A' + str(i) for i in range(7)])
        self.assertEqual([r['name'] for r in night.variants('5090')], ['B' + str(i) for i in range(12)])
        for machine in ('3090', '5090'):
            for variant in night.variants(machine):
                values = night.settings_for(machine, variant['name'])
                for key, expected in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20,
                        ca_epochs=5, dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
                        save_task_weights=False, plora_train_a=False, p_old_gradient_oracle=False,
                        branch_choice_mode='off', wpre_distill_weight=0, old_model_distill_weight=0,
                        ridge_fusion_enabled=False, plora_a_init_mode='off').items():
                    self.assertEqual(values[key], expected, (machine, variant['name'], key))
                smoke = night.settings_for(machine, variant['name'], True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 2))
                self.assertFalse(smoke['save_task_weights'])
        a = night.settings_for('3090', 'A1')
        b = night.settings_for('3090', 'A5')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'p_permission_position'})
        b = night.settings_for('3090', 'A6')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'p_permission_norm_match'})

    def test_factorial_complete_and_incomplete_pairs_have_no_effect(self):
        values = {name: {key: value for key in analysis.METRICS} for name, value in
                  zip(('B0', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B1'), (0, 1, 2, 3, 3, 4, 5, 6))}
        result = analysis.contrasts(values, '5090')['fixed_controls_factorial_descriptive']
        self.assertEqual(result['coverage_effect']['Average'], 1)
        self.assertEqual(result['protect_strength_effect']['Average'], 2)
        self.assertEqual(result['rank_effect']['Average'], 3)
        self.assertEqual(result['coverage_strength_interaction']['Average'], 0)
        del values['B1']
        self.assertNotIn('fixed_controls_factorial_descriptive', analysis.contrasts(values, '5090'))
        self.assertEqual(analysis.contrasts({'A0': {key: 1 for key in analysis.METRICS}}, '3090'), {})

    def test_fingerprint_mismatch_disables_comparisons(self):
        snapshots = {name: dict(code_revision='revision', source_sha256={'a': 'hash'},
            machine='3090', phase='formal', software={'torch': 'same'}, hardware={'GPU': 'same'},
            effective_config=night.settings_for('3090', name)) for name in ('A0', 'A1')}
        self.assertEqual(analysis.matching_issues(snapshots, '3090'), [])
        changed = copy.deepcopy(snapshots)
        changed['A1']['effective_config']['p_permission_position'] = 'permuted'
        self.assertIn(dict(mode='A1', field='wrong_factor.p_permission_position'), analysis.matching_issues(changed, '3090'))
        changed = copy.deepcopy(snapshots)
        changed['A1']['source_sha256']['a'] = 'different'
        self.assertIn(dict(mode='A1', field='source_sha256'), analysis.matching_issues(changed, '3090'))

    def test_pending_and_smoke_are_not_formal_results(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = [dict(mode='A0', status='time_budget_pending')]
            report = analysis.summarize(directory, '3090', records)
            self.assertEqual(report['runs'], [])
            self.assertEqual(report['contrasts'], {})
            self.assertIn('time_budget_pending', (directory / 'results.json').read_text())

    def test_cli_dry_run_uses_new_settings_without_gpu_or_outputs(self):
        old = (engine.SPEC, engine.settings_for, engine.command_for, engine.summarize, engine.EXTRA_SOURCE_PATHS)
        try:
            with patch.object(sys, 'argv', ['run', '--machine', '3090', '--mode', 'dry-run']), \
                    patch.object(night.subprocess, 'check_output', return_value='revision'), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(night.main(), 0)
            self.assertIn('p_permission_release=benefit', output.getvalue())
            self.assertIn('epochs=2', output.getvalue())
            self.assertIn('REAL FULL T10', output.getvalue())
        finally:
            engine.SPEC, engine.settings_for, engine.command_for, engine.summarize, engine.EXTRA_SOURCE_PATHS = old


if __name__ == '__main__':
    unittest.main()
