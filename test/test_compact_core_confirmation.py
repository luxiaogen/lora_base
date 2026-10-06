import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_compact_core_confirmation as analysis
import run_compact_core_confirmation as night
import run_compact_structure_night as previous


class ConfirmationQueueTests(unittest.TestCase):
    def test_only_three_new_t10_runs_and_exact_factors(self):
        self.assertEqual([v['name'] for v in night.variants('3090')], ['C10', 'C00'])
        self.assertEqual([v['name'] for v in night.variants('5090')], ['O'])
        for name, source in (('C10', 'R0'), ('C00', 'R3')):
            original = previous.settings_for('3090', source)
            actual = night.settings_for('3090', name)
            differences = {key for key in original.keys() | actual.keys()
                           if original.get(key) != actual.get(key)}
            self.assertEqual(differences, {'dual_mask_s_conflict_enabled',
                                          'dual_mask_p_conflict_enabled', 'wandb_group'})
            self.assertFalse(actual['dual_mask_s_conflict_enabled'])
            self.assertFalse(actual['dual_mask_p_conflict_enabled'])
        m, original = night.settings_for('5090', 'M'), night.settings_for('5090', 'O')
        self.assertEqual({key for key in m if m[key] != original[key]}, {
            'dual_mask_fixed_coverage', 'dual_mask_fixed_protect_strength',
            'dual_mask_fixed_conflict_strength', 'dual_mask_private_rank',
            'dual_mask_conflict_exact_topk', 'dual_mask_conflict_score_mode', 'dual_mask_reg_weight'})
        self.assertEqual(original['dual_mask_reg_weight'], .01)
        self.assertEqual(original['dual_mask_private_rank'], 0)
        self.assertIsNone(original['dual_mask_fixed_coverage'])
        for machine in ('3090', '5090'):
            for variant in night.variants(machine):
                settings = night.settings_for(machine, variant['name'])
                self.assertEqual((settings['seed'], settings['init_epoch'], settings['epochs'],
                                  settings['ca_epochs'], settings['dual_mask_anchor_reg_weight']),
                                 ([1993], 20, 20, 5, 2.5))
                self.assertFalse(settings['save_task_weights'])
                self.assertTrue(settings['disable_fused_sdpa'])
                self.assertEqual(settings['dual_mask_task0_gate_mode'], 'unmasked')

    def test_dry_run_does_not_query_gpu_or_write(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(night.analysis, 'current_identity', side_effect=AssertionError), \
                patch.object(night.analysis, 'load_references', side_effect=AssertionError), \
                patch.object(night.queue, 'execute_queue', return_value=0) as execute:
            path = Path(temp) / 'absent'
            self.assertEqual(night.start('3090', ['C10', 'C00'], path, 'revision', 'dry-run'), 0)
            self.assertFalse(path.exists())
            self.assertEqual(execute.call_args.kwargs['mode'], 'dry-run')

    def test_reference_mismatch_prevents_any_training(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(night.analysis, 'current_identity', return_value={}), \
                patch.object(night.analysis, 'load_references', return_value=([], {}, {})), \
                patch.object(night.analysis, 'reference_issues', return_value=['training_source']), \
                patch.object(night.queue, 'execute_queue') as execute:
            self.assertRaises(ValueError, night.start, '3090', ['C10'], Path(temp), 'rev')
            execute.assert_not_called()
            self.assertIn('training_source', (Path(temp) / 'reference_validation.json').read_text())

    def test_resume_verifies_completed_evidence_and_skips_it(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(night.analysis, 'read_run', return_value=(
                    dict(valid_performance=True), dict(code_revision='rev'), {})), \
                patch.object(night.analysis, 'completed_run_issues', return_value=[]):
            path = Path(temp)
            (path / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='rev')))
            (path / 'queue.json').write_text(json.dumps([
                dict(mode='C10', status='completed', exit_code=0)]))
            night.check_resume(path, '3090', 'rev', {})
            self.assertRaises(ValueError, night.check_resume, path, '5090', 'rev', {})
            with patch.object(night.analysis, 'completed_run_issues', return_value=['data_path']):
                self.assertRaises(ValueError, night.check_resume, path, '3090', 'rev', {})

    def test_resume_checks_smoke_evidence_instead_of_trusting_status(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(night.analysis, 'read_run', return_value=(
                    {}, dict(code_revision='rev'), {})) as read, \
                patch.object(night.analysis, 'completed_run_issues', return_value=['software']) as check:
            path = Path(temp)
            (path / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='rev')))
            (path / 'smoke_queue.json').write_text(json.dumps([
                dict(mode='C10', status='completed', exit_code=0)]))
            self.assertRaises(ValueError, night.check_resume, path, '3090', 'rev', {})
            self.assertEqual(read.call_args.args[1]['mode'], 'smoke_C10')
            self.assertTrue(check.call_args.kwargs['smoke'])

    def test_engine_bindings_are_restored_after_failure(self):
        original = (night.queue.analysis, night.queue.engine.command_for)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(night.queue, 'execute_queue', side_effect=RuntimeError('failure')):
            self.assertRaises(RuntimeError, night.start, '3090', ['C10'], Path(temp), 'rev', 'dry-run')
        self.assertEqual((night.queue.analysis, night.queue.engine.command_for), original)


class ConfirmationAnalysisTests(unittest.TestCase):
    def snapshot(self, machine, mode, revision='new'):
        return dict(machine=machine, phase='formal', mode=mode, code_revision=revision,
                    effective_config=night.settings_for(machine, mode),
                    source_sha256={'main.py': 'main', 'trainer.py': 'trainer',
                                   'models/attention.py': 'attention',
                                   'scripts/new_queue.py': revision},
                    software={'torch': 'same'}, hardware={'uuid': 'same'})

    def test_cross_revision_reuse_requires_identical_training_sources(self):
        snapshots = {name: self.snapshot('3090', name, 'old' if name == 'C11' else 'new')
                     for name in ('C11', 'C01', 'C10', 'C00')}
        self.assertEqual(analysis.matching_issues(snapshots, '3090'), [])
        bad = copy.deepcopy(snapshots)
        bad['C00']['source_sha256']['models/attention.py'] = 'changed'
        self.assertIn({'mode': 'C00', 'field': 'training_source_sha256'},
                      analysis.matching_issues(bad, '3090'))
        for field in ('software', 'hardware'):
            bad = copy.deepcopy(snapshots)
            bad['C10'][field] = {'different': True}
            self.assertIn({'mode': 'C10', 'field': field}, analysis.matching_issues(bad, '3090'))

    def test_unexpected_config_and_wrong_factor_are_rejected(self):
        snapshots = {name: self.snapshot('5090', name) for name in ('M', 'O')}
        self.assertEqual(analysis.matching_issues(snapshots, '5090'), [])
        for key, value in (('ca_epochs', 10), ('data_path', 'different'),
                           ('dual_mask_reg_weight', 0), ('seed', [4])):
            bad = copy.deepcopy(snapshots)
            bad['O']['effective_config'][key] = value
            self.assertTrue(analysis.matching_issues(bad, '5090'), key)

    def test_exact_two_by_two_and_missing_cell(self):
        complete = {name: {key: value for key in analysis.METRICS}
                    for name, value in (('C00', 1), ('C01', 3), ('C10', 5), ('C11', 9))}
        effects = analysis.contrasts(complete, '3090')['permissions_suppression_2x2']['Average']
        self.assertEqual(effects, dict(permission_effect=5, suppression_effect=3, interaction=2))
        del complete['C00']
        self.assertNotIn('permissions_suppression_2x2', analysis.contrasts(complete, '3090'))
        results = {name: {key: value for key in analysis.METRICS}
                   for name, value in (('M', 7), ('O', 8))}
        self.assertEqual(analysis.contrasts(results, '5090')['M_minus_O']['Average'], -1)

    def test_reference_health_source_set_and_current_identity(self):
        snapshot = self.snapshot('5090', 'M', analysis.REFERENCE_REVISION)
        snapshot['mode'] = 'S2'
        identity = dict(training_source_sha256=analysis.training_hashes(snapshot),
                        software=snapshot['software'], hardware=snapshot['hardware'],
                        effective_config=snapshot['effective_config'])
        summary = dict(mode='M', valid_performance=True, epoch_records_complete=True,
                       position_diagnostics_complete=True)
        with patch.object(analysis.previous, 'settings_for', return_value=snapshot['effective_config']):
            self.assertEqual(analysis.reference_issues([summary], {'M': snapshot}, '5090', identity), [])
            for field in ('valid_performance', 'epoch_records_complete', 'position_diagnostics_complete'):
                bad = dict(summary, **{field: False})
                self.assertTrue(analysis.reference_issues([bad], {'M': snapshot}, '5090', identity))
            changed = copy.deepcopy(identity)
            changed['training_source_sha256']['models/new.py'] = 'new'
            self.assertTrue(analysis.reference_issues([summary], {'M': snapshot}, '5090', changed))

    def test_summary_rechecks_raw_health_and_reference_revision(self):
        reference = self.snapshot('5090', 'M', analysis.REFERENCE_REVISION)
        reference['mode'] = 'S2'
        candidate = self.snapshot('5090', 'O')
        summary = dict(mode='M', status='completed', valid_performance=True,
                       epoch_records_complete=True, position_diagnostics_complete=True,
                       Task0=97, **{key: 87 for key in analysis.METRICS})
        measured = {name: [] for name in ('epochs', 'updates', 'masks', 'diagnostics',
                                          'costs', 'storage', 'tasks')}
        cases = [(None, None), ('epoch_records_complete', None),
                 ('position_diagnostics_complete', None), (None, 'revision'), (None, 'mode')]
        for unhealthy, wrong_reference in cases:
            with self.subTest(unhealthy=unhealthy, wrong_reference=wrong_reference), \
                    tempfile.TemporaryDirectory() as temp:
                old = copy.deepcopy(reference)
                if wrong_reference:
                    old['code_revision' if wrong_reference == 'revision' else 'mode'] = 'wrong'
                new = dict(summary, mode='O')
                if unhealthy:
                    new[unhealthy] = False
                with patch.object(analysis, 'load_references', return_value=(
                        [summary], {'M': old}, copy.deepcopy(measured))), \
                        patch.object(analysis, 'read_run', return_value=(new, candidate, measured)), \
                        patch.object(analysis.plotter, 'draw'):
                    analysis.summarize(Path(temp), '5090', [dict(mode='O', status='completed')])
                result = json.loads((Path(temp) / 'contrasts.json').read_text())
                if unhealthy or wrong_reference:
                    self.assertTrue(result['matching_issues'])
                    self.assertEqual(result['completed_pairs'], {})
                else:
                    self.assertEqual(result['matching_issues'], [])
                    self.assertIn('M_minus_O', result['completed_pairs'])


class ConfirmationGateTests(unittest.TestCase):
    def make(self, permission):
        import test.test_dualmask_core as existing
        module = existing.CorePolicyTests().make(
            dual_mask_permission_mode='asymmetric' if permission else 'symmetric_soft',
            dual_mask_mechanism_audit=True)
        module.effective_protect_strength = .5 if permission else 0
        module.dual_mask_s_conflict_enabled = False
        module.dual_mask_p_conflict_enabled = False
        module.dual_mask_conflict_reg_enabled = False
        return module

    def test_gate_off_forward_and_merge_for_both_permissions(self):
        import torch
        from torch.nn import functional as F
        from utils.dualmask_core import permission_gate
        for permission in (True, False):
            module = self.make(permission)
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_()
            x = torch.randn(5, 4)
            expected = torch.zeros(5, 12)
            for isolated, unit, gamma in ((False, module.S_lora[1], module.slora_gamma),
                                          (True, module.P_lora[1], module.plora_gamma)):
                raw = unit.B_weight @ unit.A_weight
                gate = permission_gate(module.general_mask, module.effective_protect_strength,
                                       isolated, module.args['dual_mask_permission_mode'])
                expected += F.linear(x, gamma * raw * gate)
            torch.testing.assert_close(module._contrib_from_units(x, 1), expected)
            before = F.linear(x, module.qkv.weight, module.qkv.bias) + expected
            module.after_task(1)
            torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias),
                                       atol=4e-6, rtol=1e-5)
            self.assertIsNone(module.S_lora[1])
            self.assertIsNone(module.P_lora[1])


if __name__ == '__main__':
    unittest.main()
