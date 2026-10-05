import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_compact_structure as analysis
import run_compact_structure_night as night


class CompactQueueTests(unittest.TestCase):
    def test_exact_matrix_and_exploration_disabled(self):
        for machine, prefix in (('3090', 'R'), ('5090', 'S')):
            self.assertEqual([row['name'] for row in night.variants(machine)], [prefix + str(i) for i in range(8)])
            for row in night.variants(machine):
                settings = night.settings_for(machine, row['name'])
                expected = dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20, ca_epochs=5,
                    rank=64, dual_mask_private_rank=40, dual_mask_reg_weight=0,
                    dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True, save_task_weights=False,
                    dual_mask_fixed_coverage=.825, slora_lr_multiplier=1, plora_lr_multiplier=1,
                    late_weight_average_epochs=0, sp_staged_s_epochs=0, plora_a_init_mode='off',
                    p_conflict_freeze_epoch=0, plora_train_a=False, wpre_distill_weight=0,
                    old_model_distill_weight=0, p_old_gradient_oracle=False, branch_choice_mode='off',
                    ridge_fusion_enabled=False, p_hard_zero_mode='off', p_permission_release='off',
                    p_step_direction='off', pair_separation_weight=0, p_direction_score='off',
                    dual_mask_composed_conflict='off', ca_cov_shrinkage=0, ca_stats_transport=False)
                for key, value in expected.items():
                    self.assertEqual(settings[key], value, (machine, row['name'], key))
                smoke = night.settings_for(machine, row['name'], True)
                self.assertEqual([smoke[key] for key in ('max_tasks', 'init_epoch', 'epochs', 'ca_epochs')], [2, 1, 1, 1])
        f, m = night.settings_for('5090', 'S0'), night.settings_for('5090', 'S2')
        self.assertEqual((f['dual_mask_conflict_exact_topk'], f['dual_mask_conflict_score_mode']), (False, 'conflict'))
        self.assertEqual((m['dual_mask_conflict_exact_topk'], m['dual_mask_conflict_score_mode']), (True, 'magnitude'))
        self.assertEqual(m['dual_mask_conflict_ratio'], .1)
        self.assertEqual(night.settings_for('3090', 'R2')['dual_mask_branch_layout'], 'single')
        self.assertEqual(night.settings_for('3090', 'R7')['dual_mask_fixed_protect_strength'], 0)
        self.assertIsNone(night.settings_for('5090', 'S3')['dual_mask_fixed_conflict_strength'])
        self.assertTrue(night.settings_for('5090', 'S6')['dual_mask_uniform_norm_matched'])

    def test_budget_starts_before_smokes_and_active_run_finishes(self):
        clock, calls = [0], []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            clock[0] += 20 if smoke else 50
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(night.engine, 'run', side_effect=run), \
                patch.object(night.analysis, 'summarize'), patch.object(night.time, 'monotonic', side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            self.assertEqual(night.execute_queue('3090', ['R0', 'R1', 'R2'], path, 'rev', hours=80 / 3600), 0)
            self.assertEqual(calls, [('R0', True), ('R1', True), ('R2', True), ('R0', False)])
            self.assertGreater(clock[0], 80)
            self.assertEqual([r['status'] for r in json.loads((path / 'queue.json').read_text())],
                             ['completed', 'time_budget_pending', 'time_budget_pending'])

    def test_deadline_during_smokes_does_not_start_formal(self):
        clock, calls = [0], []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            clock[0] += 101
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(night.engine, 'run', side_effect=run), \
                patch.object(night.analysis, 'summarize'), patch.object(night.time, 'monotonic', side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            self.assertEqual(night.execute_queue('3090', ['R0', 'R1'], path, 'rev', hours=100 / 3600), 0)
            self.assertEqual(calls, [('R0', True)])
            self.assertTrue(all(r['status'] == 'time_budget_pending' for r in json.loads((path / 'queue.json').read_text())))

    def test_analysis_error_does_not_abort_fixed_queue(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(night.engine, 'run',
                side_effect=lambda m, n, d, r, smoke=False, dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=0, Average=0)), \
                patch.object(night.analysis, 'summarize', side_effect=TypeError('duplicate mode')), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            self.assertEqual(night.execute_queue('5090', ['S0', 'S1'], path, 'rev', mode='t10'), 0)
            self.assertEqual(len(json.loads((path / 'queue.json').read_text())), 2)
            self.assertEqual(len((path / 'analysis_errors.jsonl').read_text().splitlines()), 2)

    def test_training_failures_pause_without_retry(self):
        for smoke_failure in (True, False):
            calls = []
            def run(m, n, d, r, smoke=False, dry_run=False):
                calls.append((n, smoke))
                return dict(mode=n, status='failed' if smoke == smoke_failure else 'completed',
                            exit_code=7 if smoke == smoke_failure else 0, minutes=1)
            with tempfile.TemporaryDirectory() as temp, patch.object(night.engine, 'run', side_effect=run), \
                    patch.object(night.analysis, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(night.execute_queue('3090', ['R0', 'R1'], Path(temp), 'rev'), 7)
                self.assertEqual(calls, [('R0', True)] if smoke_failure else [('R0', True), ('R1', True), ('R0', False)])

    def test_resume_skips_completed_t10_and_successful_smokes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(night.engine, 'run',
                side_effect=lambda m, n, d, r, smoke=False, dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=0)) as run, \
                patch.object(night.analysis, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            (path / 'queue.json').write_text(json.dumps([dict(mode='R0', status='completed', exit_code=0)]))
            (path / 'smoke_queue.json').write_text(json.dumps([dict(mode='R1', status='completed', exit_code=0)]))
            self.assertEqual(night.execute_queue('3090', ['R0', 'R1'], path, 'rev'), 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[1], 'R1')
            self.assertNotIn('smoke', run.call_args.kwargs)

    def test_dry_run_does_not_create_files_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(night.engine.subprocess, 'check_output', side_effect=AssertionError), \
                contextlib.redirect_stdout(io.StringIO()), patch.object(night.engine, 'command_for', side_effect=night.command_for):
            path = Path(temp) / 'not-created'
            self.assertEqual(night.execute_queue('3090', ['R0'], path, 'rev', mode='dry-run'), 0)
            self.assertFalse(path.exists())

    def test_resume_requires_real_completed_evidence_and_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='rev')))
            (path / 'queue.json').write_text(json.dumps([dict(mode='R0', status='completed', exit_code=0)]))
            self.assertRaises(ValueError, night.check_resume, path, '5090', 'rev')
            self.assertRaises(OSError, night.check_resume, path, '3090', 'rev')
            config = json.loads((night.ROOT / 'exps/dlora/imgr10.json').read_text())
            config.update(night.settings_for('3090', 'R0'))
            snapshot = dict(machine='3090', code_revision='rev', phase='formal', effective_config=config,
                source_sha256={'models/attention.py': hashlib.sha256((night.ROOT / 'models/attention.py').read_bytes()).hexdigest()})
            snapshot['effective_config']['prefix'] = 'timestamp_specific_original_run'
            measured = dict(tasks_reported=10, runtime_error=False, **{key: 90 for key in analysis.METRICS})
            with patch.object(night.analysis, 'read_run', return_value=(measured, snapshot, {})):
                night.check_resume(path, '3090', 'rev')
                snapshot['source_sha256']['models/attention.py'] = 'wrong'
                self.assertRaises(ValueError, night.check_resume, path, '3090', 'rev')


class CompactAnalysisTests(unittest.TestCase):
    def snapshot(self, machine, mode):
        return dict(code_revision='rev', source_sha256={'a': 'hash'}, machine=machine, mode=mode,
            phase='formal', software={'torch': 'same'}, hardware={'uuid': 'same'},
            effective_config=night.settings_for(machine, mode))

    def test_all_factors_and_common_protocol_checked(self):
        snapshots = {n: self.snapshot('3090', n) for n in ('R0', 'R1', 'R2', 'R5')}
        self.assertEqual(analysis.matching_issues(snapshots, '3090'), [])
        for key, value in (('dual_mask_branch_layout', 'dual'), ('rank', 32), ('seed', [5]),
                           ('dual_mask_conflict_exact_topk', False)):
            bad = copy.deepcopy(snapshots)
            bad['R2']['effective_config'][key] = value
            self.assertIn(dict(mode='R2', field='wrong_factor.' + key), analysis.matching_issues(bad, '3090'))
        bad = copy.deepcopy(snapshots)
        bad['R5']['source_sha256'] = {'a': 'other'}
        self.assertIn(dict(mode='R5', field='source_sha256'), analysis.matching_issues(bad, '3090'))

    def test_partial_pairs_and_correct_2x2(self):
        complete = {mode: {key: value for key in analysis.METRICS}
                    for mode, value in (('S0', 1), ('S3', 2), ('S4', 4), ('S5', 8))}
        result = analysis.contrasts(complete, '5090')
        self.assertNotIn('S2_minus_S1', result)
        self.assertEqual(result['ranking_strength_2x2']['Average'],
                         dict(magnitude_effect=4.5, adaptive_strength_effect=2.5, interaction=3))
        self.assertNotIn('ranking_strength_2x2', analysis.contrasts({'S0': complete['S0']}, '5090'))

    def test_pending_and_smoke_do_not_become_performance_and_mode_collision_safe(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(analysis, 'draw'):
            root = Path(temp)
            path = root / 'R2'
            path.mkdir()
            snapshot = self.snapshot('3090', 'R2')
            snapshot['phase'] = 'smoke'
            (path / 'run.json').write_text(json.dumps(snapshot))
            content = "[trainer.py] => CNN: {'total': 90, 'old': 80, 'new': 90}\n" * 2
            content += '[trainer.py] => Average Accuracy: 90.0\n[trainer.py] => Forgetting: 1.0\n'
            content += 'CoreCost ' + json.dumps(dict(mode='permission', task=1, stage='training', seconds=1)) + '\n'
            (path / 'training.log').write_text(content)
            records = [dict(mode='R2', status='completed', exit_code=0), dict(mode='R0', status='time_budget_pending')]
            analysis.summarize(root, '3090', records)
            results = json.loads((root / 'results.json').read_text())
            self.assertFalse(results[0]['valid_performance'])
            self.assertEqual(json.loads((root / 'contrasts.json').read_text())['completed_pairs'], {})
            self.assertIn('R2', (root / 'costs.csv').read_text())

    def test_single_epoch_telemetry_expectations_and_actual_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'R2'
            path.mkdir()
            snapshot = self.snapshot('3090', 'R2')
            (path / 'run.json').write_text(json.dumps(snapshot))
            content = "[trainer.py] => CNN: {'total': 90, 'old': 80, 'new': 90}\n" * 10
            content += '[trainer.py] => Average Accuracy: 90.0\n[trainer.py] => Forgetting: 1.0\n'
            for task in range(10):
                for epoch in range(1, 21):
                    for layer in range(12):
                        for projection in ('Q', 'K', 'V'):
                            row = dict(task=task, epoch=epoch, layer=layer, branch='S' if task == 0 else 'Single',
                                       projection=projection, effective_norm=1)
                            content += 'CoreEpochUpdate ' + json.dumps(row) + '\n'
            (path / 'training.log').write_text(content)
            analysis.summarize(root, '3090', [dict(mode='R2', status='completed', exit_code=0)])
            self.assertTrue(json.loads((root / 'results.json').read_text())[0]['epoch_records_complete'])
            for filename in ('old_new.png', 'norm_trajectories.pdf', 'branch_organization.png', 'position_margin.png', 'report.md'):
                self.assertTrue((root / filename).exists())


if __name__ == '__main__':
    unittest.main()
