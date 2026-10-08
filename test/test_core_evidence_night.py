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
import analyze_core_evidence as analysis
import run_core_evidence_night as night


class QueueTests(unittest.TestCase):
    def test_fixed_matrix_and_settings(self):
        self.assertEqual([v['name'] for v in night.variants('3090')], ['R' + str(i) for i in range(7)])
        self.assertEqual([v['name'] for v in night.variants('5090')], ['S' + str(i) for i in range(8)])
        for machine in ('3090', '5090'):
            for variant in night.variants(machine):
                settings = night.settings_for(machine, variant['name'])
                for key, value in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20,
                    ca_epochs=5, dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
                    save_task_weights=False, plora_train_a=False, p_old_gradient_oracle=False,
                    branch_choice_mode='off', old_model_distill_weight=0, wpre_distill_weight=0,
                    plora_a_init_mode='off', ridge_fusion_enabled=False).items():
                    self.assertEqual(settings[key], value, (machine, variant['name'], key))
                smoke = night.settings_for(machine, variant['name'], True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
                command, _ = night.command_for(machine, variant['name'], Path('/tmp/test/formal'))
                self.assertEqual(command[1:4], ['main.py', '--config', 'exps/dlora/imgr10.json'])
                self.assertIn('save_task_weights=false', command)

    def test_budget_includes_smokes_and_does_not_stop_active_run(self):
        clock = [0]
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            clock[0] += 20 if smoke else 50
            return dict(mode=name, status='completed', exit_code=0, minutes=50 / 60)
        with tempfile.TemporaryDirectory() as temp, patch.object(night, 'run', side_effect=run), \
                patch.object(night, 'summarize'), patch.object(night.time, 'monotonic', side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            code = night.execute_queue('3090', ['R0', 'R1', 'R2'], path, 'revision', hours=80 / 3600)
            self.assertEqual(code, 0)
            self.assertEqual(calls, [('R0', True), ('R1', True), ('R2', True), ('R0', False)])
            records = json.loads((path / 'queue.json').read_text())
            self.assertEqual([r['status'] for r in records], ['completed', 'time_budget_pending', 'time_budget_pending'])
            self.assertGreater(clock[0], 80)  # The active full run was allowed to finish.

    def test_failures_pause_without_retry_or_starting_formal_after_failed_smoke(self):
        for smoke_failure in (True, False):
            calls = []
            def run(machine, name, directory, revision, smoke=False, dry_run=False):
                calls.append((name, smoke))
                fail = smoke == smoke_failure
                return dict(mode=name, status='failed' if fail else 'completed', exit_code=7 if fail else 0, minutes=1)
            with tempfile.TemporaryDirectory() as temp, patch.object(night, 'run', side_effect=run), \
                    patch.object(night, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(night.execute_queue('3090', ['R0', 'R1'], Path(temp), 'revision'), 7)
                self.assertEqual(calls, [('R0', True)] if smoke_failure else [('R0', True), ('R1', True), ('R0', False)])

    def test_deadline_also_stops_new_smokes(self):
        clock = [0]
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            clock[0] += 101
            return dict(mode=name, status='completed', exit_code=0, minutes=101 / 60)
        with tempfile.TemporaryDirectory() as temp, patch.object(night, 'run', side_effect=run), \
                patch.object(night, 'summarize'), patch.object(night.time, 'monotonic', side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            self.assertEqual(night.execute_queue('3090', ['R0', 'R1'], path, 'revision', hours=100 / 3600), 0)
            self.assertEqual(calls, [('R0', True)])
            self.assertEqual(json.loads((path / 'smoke_queue.json').read_text())[-1]['status'], 'time_budget_pending')
            self.assertTrue(all(r['status'] == 'time_budget_pending' for r in json.loads((path / 'queue.json').read_text())))

    def test_completed_runs_do_not_stop_based_on_accuracy(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(night, 'run',
                side_effect=lambda m, n, d, r, smoke=False, dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=0, Average=0)), \
                patch.object(night, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)
            self.assertEqual(night.execute_queue('5090', ['S0', 'S1'], path, 'revision', mode='t10'), 0)
            self.assertEqual(len(json.loads((path / 'queue.json').read_text())), 2)

    def test_dry_run_creates_no_files_and_uses_no_gpu(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(night.subprocess, 'check_output', side_effect=AssertionError), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp) / 'unused'
            self.assertEqual(night.execute_queue('3090', ['R0'], path, 'revision', mode='dry-run'), 0)
            self.assertFalse(path.exists())

    def test_real_runner_records_pid_and_rejects_nonfinite_smoke(self):
        class Process:
            pid = 12345
            def __init__(self, content):
                self.stdout = content.splitlines(keepends=True)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def wait(self):
                return 0
        content = ("[trainer.py] => CNN: {'total': 80.0, 'old': 79.0, 'new': 81.0}\n" * 2
                   + '[trainer.py] => Average Accuracy: 80.0\n[trainer.py] => Forgetting: 5.0\n')
        for extra, expected in (('', 0), ('Loss nan\n', 2)):
            with tempfile.TemporaryDirectory() as temp, patch.object(night.subprocess, 'Popen', return_value=Process(content + extra)), \
                    patch.object(night.subprocess, 'check_output', return_value='0,fixture-GPU,fixture'), \
                    patch.object(night.importlib.metadata, 'version', return_value='fixture'), \
                    patch.object(night.platform, 'platform', return_value='fixture'), \
                    contextlib.redirect_stdout(io.StringIO()):
                path = Path(temp)
                record = night.run('3090', 'R0', path / 'smoke_R0', 'fixture', smoke=True)
                self.assertEqual(record['exit_code'], expected)
                active = json.loads((path / 'active.json').read_text())
                self.assertEqual(active['training_pid'], 12345)
                self.assertEqual(active['exit_code'], expected)
                self.assertEqual(json.loads((path / 'smoke_R0/run.json').read_text())['phase'], 'smoke')

    def test_snapshot_records_bounded_effective_strength_not_raw_override(self):
        class Process:
            pid = 12345
            stdout = ["[trainer.py] => CNN: {'total': 80., 'old': 79., 'new': 81.}\n"] * 2 + [
                '[trainer.py] => Average Accuracy: 80.\n[trainer.py] => Forgetting: 5.\n']
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def wait(self):
                return 0
        command_for = night.command_for
        for raw, bounded in ((1.2, 1.), (-.2, 0.)):
            def command(*args):
                argv, settings = command_for(*args)
                settings['dual_mask_fixed_protect_strength'] = raw
                argv += ['--set', 'dual_mask_fixed_protect_strength=' + str(raw)]
                return argv, settings
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp, \
                    patch.object(night, 'command_for', side_effect=command), \
                    patch.object(night.subprocess, 'Popen', return_value=Process()), \
                    patch.object(night.subprocess, 'check_output', return_value='fixture-GPU'), \
                    patch.object(night.importlib.metadata, 'version', return_value='fixture'), \
                    patch.object(night.platform, 'platform', return_value='fixture'), \
                    contextlib.redirect_stdout(io.StringIO()):
                path = Path(temp) / 'smoke_R0'
                night.run('3090', 'R0', path, 'fixture', smoke=True)
                snapshot = json.loads((path / 'run.json').read_text())
                self.assertEqual(snapshot['effective_config']['dual_mask_fixed_protect_strength'], bounded)
                self.assertIn('dual_mask_fixed_protect_strength=' + str(raw), snapshot['command'])


class AnalysisTests(unittest.TestCase):
    def snapshot(self, mode):
        return dict(code_revision='revision', source_sha256={'a': 'hash'}, machine='3090', phase='formal',
            software={'torch': 'same'}, hardware={'uuid': 'same'}, effective_config=night.settings_for('3090', mode))

    def test_all_expected_factors_and_common_protocol_checked(self):
        snapshots = {n: self.snapshot(n) for n in ('R0', 'R1', 'R2', 'R5')}
        self.assertEqual(analysis.matching_issues(snapshots, '3090'), [])
        for mode, key, value in (('R0', 'dual_mask_position_norm_match', 'paired_min'),
                ('R1', 'dual_mask_conflict_score_mode', 'magnitude'),
                ('R2', 'dual_mask_permission_mode', 'symmetric_hard'),
                ('R5', 'save_task_weights', True)):
            bad = copy.deepcopy(snapshots)
            bad[mode]['effective_config'][key] = value
            self.assertIn(dict(mode=mode, field='wrong_factor.' + key), analysis.matching_issues(bad, '3090'))
        bad = copy.deepcopy(snapshots)
        for snapshot in bad.values():
            snapshot['effective_config']['seed'] = [1996]
        self.assertTrue(analysis.matching_issues(bad, '3090'))

    def test_incomplete_pairs_have_no_attribution_and_2x2_signs(self):
        rows = {name: {key: value for key in analysis.METRICS}
                for name, value in zip(('S0', 'S1', 'S2', 'S3'), (8, 6, 5, 4))}
        self.assertNotIn('gates_penalty_2x2', analysis.contrasts({k: rows[k] for k in ('S0', 'S1', 'S2')}, '5090'))
        result = analysis.contrasts(rows, '5090')['gates_penalty_2x2']['Average']
        self.assertEqual(result, dict(gates_effect=2.5, conflict_penalty_effect=1.5, interaction=1))

    def test_smoke_partial_and_nan_are_not_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            run_dir = path / 'R0'
            run_dir.mkdir()
            for tasks, phase, health, valid in ((2, 'smoke', '', False), (9, 'formal', '', False),
                    (10, 'formal', 'Loss nan', False), (10, 'formal', '', True)):
                snapshot = self.snapshot('R0')
                snapshot['phase'] = phase
                (run_dir / 'run.json').write_text(json.dumps(snapshot))
                text = '\n'.join("[trainer.py] => CNN: {'total': 80.0, 'old': 79.0, 'new': 81.0}" for _ in range(tasks))
                text += '\n[trainer.py] => Average Accuracy: 80.0\n[trainer.py] => Forgetting: 5.0\n' + health
                (run_dir / 'training.log').write_text(text)
                result, _, _ = analysis.read_run(path, dict(mode='R0', status='completed', exit_code=0))
                self.assertEqual(result['valid_performance'], valid)
                self.assertFalse(result['position_diagnostics_complete'])

    def test_plots_generated_only_for_available_measured_groups(self):
        rows = {key: [] for key in ('tasks', 'epochs', 'diagnostics', 'costs', 'storage')}
        for modes, expected in ((('R0', 'R3', 'R4'), 'permissions.png'),
                               (('S0', 'S1', 'S2', 'S3'), 'gates_penalty_2x2.png')):
            complete = {mode: {key: 80 + i * .1 for key in analysis.METRICS} for i, mode in enumerate(modes)}
            rows['tasks'] = [dict(mode=mode, total=80, task=task) for mode in modes for task in range(10)]
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                analysis.draw(path, complete, rows)
                self.assertTrue((path / expected).exists())
                self.assertTrue((path / 'old_new.pdf').exists())


if __name__ == '__main__':
    unittest.main()
