from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze_branch_choice import analyze_directory, auc, signal_summary
from scripts.run_branch_choice_night import (
    command_for, estimate_seconds, fits_budget, main, settings_for,
)


ROOT = Path(__file__).resolve().parents[1]


class BranchNightTests(unittest.TestCase):
    def spec(self, machine):
        return json.loads((ROOT / f'scripts/sweeps/imgr10_branch_choice_{machine}.json').read_text())

    def test_machine_assignment_single_seed_original_recipe_no_path_or_checkpoints(self):
        for machine, scope, hours in (('3090', 'p', 6), ('5090', 'sp', 8)):
            spec = self.spec(machine)
            self.assertEqual(spec['scope'], scope)
            self.assertEqual(spec['hours'], hours)
            self.assertEqual(spec['seeds'], [1993])
            for variant in spec['variants']:
                config = settings_for(spec, variant)
                self.assertEqual(config['seed'], [1993])
                self.assertEqual(config['dual_mask_anchor_reg_weight'], 2.5)
                self.assertEqual(config['ca_epochs'], 5)
                self.assertEqual(config['epochs'], 20)
                self.assertEqual(config['init_epoch'], 20)
                self.assertEqual(config['dual_mask_conflict_granularity'], 'layer')
                self.assertTrue(config['disable_fused_sdpa'])
                self.assertFalse(config['save_task_weights'])
                self.assertFalse(config['p_old_gradient_oracle'])
                self.assertEqual(config['p_step_direction'], 'off')
                self.assertEqual(config['p_hard_zero_mode'], 'off')
                self.assertNotIn('data_path', config)
                command, _ = command_for(spec, variant, 'test')
                self.assertEqual(command[:4], [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json'])

    def test_budget_prediction_and_skip_do_not_use_performance(self):
        job = self.spec('3090')['variants'][1]
        self.assertAlmostEqual(estimate_seconds(job, {(3, 5, 'audit'): 1800}, 130), 8280)
        self.assertAlmostEqual(estimate_seconds(job, {(10, 5, 'oracle'): 6000}, 130), 6900)
        self.assertTrue(fits_budget(3600, 6900, 6))
        self.assertFalse(fits_budget(20000, 6900, 6))

    def test_smoke_uses_short_route_without_accuracy_assertions(self):
        spec = self.spec('5090')
        config = settings_for(spec, spec['variants'][1], True)
        self.assertEqual((config['max_tasks'], config['epochs'], config['ca_epochs']), (2, 1, 1))
        source = (ROOT / 'scripts/run_branch_choice_night.py').read_text()
        self.assertNotIn('assert ', source)
        self.assertNotIn('Average Accuracy', source)

    def test_queue_records_all_budget_skips_including_tail_and_all_skip(self):
        spec = self.spec('3090')
        for hours in (0, 1):
            clock = [0.]
            def fake_run(spec, variant, directory, smoke, dry_run):
                if not smoke:
                    clock[0] += 3300
                return 0, 0. if smoke else 3300.
            with patch('sys.argv', ['night', '3090', '--hours', str(hours)]), \
                    patch('scripts.run_branch_choice_night.Path.read_text', return_value=json.dumps(spec)), \
                    patch('scripts.run_branch_choice_night.Path.mkdir'), \
                    patch('scripts.run_branch_choice_night.Path.write_text', autospec=True) as write, \
                    patch('scripts.run_branch_choice_night.subprocess.check_output', return_value='test'), \
                    patch('scripts.run_branch_choice_night.time.monotonic', side_effect=lambda: clock[0]), \
                    patch('scripts.run_branch_choice_night.run', side_effect=fake_run), \
                    patch('scripts.analyze_branch_choice.analyze_directory'), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            recorded = [json.loads(call.args[1]) for call in write.call_args_list
                        if call.args[0].name == 'queue.json'][-1]
            self.assertEqual(len(recorded), 7)
            self.assertEqual(recorded[-1]['status'], 'budget_skipped')
            self.assertEqual(recorded[0]['status'], 'completed' if hours else 'budget_skipped')

    def test_dry_run_prints_seven_full_or_screen_jobs_without_running_training(self):
        for machine in ('3090', '5090'):
            result = subprocess.run([sys.executable, 'scripts/run_branch_choice_night.py', machine, '--dry-run'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('Command: '), 11)
            self.assertIn('oracle_dense_t3', result.stdout)
            self.assertIn('FAILED=0', result.stdout)

    def test_auc_and_summary_keep_trajectories_and_tasks_separate(self):
        self.assertAlmostEqual(auc([False, True], [0., 1.]), 1.)
        self.assertAlmostEqual(auc([False, True], [1., 0.]), 0.)
        self.assertAlmostEqual(auc([False, True], [1., 1.]), .5)
        self.assertIsNone(auc([True, True], [0., 1.]))
        rows = [dict(file='audit.log', task=1, old_loss_delta=d,
                     old_logit_shift_delta=d, feature_shift_delta=-d) for d in (-.2, .3)]
        rows += [dict(file='oracle.log', task=1, old_loss_delta=-.1,
                      old_logit_shift_delta=.1, feature_shift_delta=.1)]
        summary = signal_summary(rows)
        a = next(r for r in summary if r['file'] == 'audit.log' and r['signal'] == 'old_logit_shift')
        self.assertEqual((a['true_positive'], a['false_negative'], a['auc']), (1, 0, 1.))
        b = next(r for r in summary if r['file'] == 'audit.log' and r['signal'] == 'feature_shift')
        self.assertEqual(b['auc'], 0.)
        self.assertEqual(len(summary), 4)

    def test_actual_candidate_log_parses_and_exports_csv_without_dense_state(self):
        import torch
        from test.test_branch_step_choice import Network, BranchChoiceTests
        from utils.branch_step_choice import apply_branch_choice
        net = Network()
        inputs = torch.ones(3, 2)
        row = apply_branch_choice(net, BranchChoiceTests().snapshots(net), inputs,
                                  torch.tensor([2, 3, 2]), (inputs, torch.tensor([0, 1, 0])),
                                  2, 20., 'audit', 'sp')
        step = dict(task=1, epoch=1, batch=1, mode='audit', scope='sp', **row)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            directory = Path(folder)
            (directory / 'audit.log').write_text('INFO BranchChoiceStep ' + json.dumps(step) + '\n')
            summary = analyze_directory(directory)
            csv = (directory / 'candidate_steps.csv').read_text()
            self.assertEqual(len(csv.splitlines()), 5)
            self.assertIn('target_joint_b_step_norm', csv)
            self.assertEqual(len(summary), 2)
            self.assertTrue((directory / 'signal_summary.json').exists())


if __name__ == '__main__':
    unittest.main()
