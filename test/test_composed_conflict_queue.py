import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('composed_queue', ROOT / 'scripts/run_composed_conflict.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ComposedQueueTests(unittest.TestCase):
    def test_six_real_t10_jobs_with_baseline_protocol_and_no_teachers_or_checkpoints(self):
        self.assertEqual(len(queue.MODES), 6)
        for mode in queue.MODES:
            config = queue.settings_for(mode)
            for key, expected in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20, ca_epochs=5,
                disable_fused_sdpa=True, dual_mask_anchor_reg_weight=2.5, save_task_weights=False,
                dual_mask_composed_conflict=mode, dual_mask_conflict_score_mode='conflict',
                dual_mask_conflict_granularity='layer', dual_mask_conflict_budget_multiplier=1,
                dual_mask_private_conflict_mode='global', dual_mask_conflict_merge_mode='suppress',
                dual_mask_s_conflict_enabled=True, dual_mask_p_conflict_enabled=True,
                dual_mask_s_protect_enabled=True, dual_mask_reg_weight=.01, plora_train_a=False,
                wpre_distill_weight=0., old_model_distill_weight=0, ridge_fusion_enabled=False,
                two_expert_oracle_diagnostic=False, branch_choice_mode='off', p_old_gradient_oracle=False,
                p_hard_zero_mode='off', dual_mask_uniform_norm_matched=False,
                dual_mask_safe_residual_enabled=False, p_conflict_strength_warmup=False,
                p_conflict_freeze_epoch=0, dual_mask_vis=False, incremental_holdout=False,
                task0_validation_enabled=False).items():
                self.assertEqual(config[key], expected, (mode, key))
            self.assertNotIn('data_path', config)
            self.assertNotIn('pretrained_path', config)

    def test_single_variable_mode_changes_across_controls(self):
        base = queue.settings_for('net')
        for mode in queue.MODES:
            config = queue.settings_for(mode)
            config['dual_mask_composed_conflict'] = 'net'
            self.assertEqual(config, base)

    def test_smoke_is_explicitly_one_epoch_not_a_formal_result(self):
        config = queue.settings_for('net', smoke=True)
        self.assertEqual((config['max_tasks'], config['init_epoch'], config['epochs'], config['ca_epochs']), (2, 1, 1, 1))
        command, _ = queue.command_for('net', Path('test_run'))
        self.assertEqual(command[1:4], ['main.py', '--config', 'exps/dlora/imgr10.json'])
        self.assertIn('dual_mask_composed_conflict=net', command)

    def test_metrics_include_only_trainer_reports_and_actual_gate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.log'
            path.write_text("[trainer.py] => CNN: {'total': 82.6, 'old': 82.1, 'new': 87.1}\n"
                '[trainer.py] => Average Accuracy: 87.214\n'
                '[trainer.py] => W_pre-only NCM Average Accuracy: 61.122\n'
                '[trainer.py] => Forgetting: 5.1000   Backward: -5.1\n')
            metrics, rows = queue.read_metrics(path)
            self.assertEqual(metrics['Average'], 87.214)
            self.assertEqual(metrics['New'], 87.1)
            self.assertEqual(metrics['tasks_reported'], 1)
            self.assertEqual(metrics['Forgetting'], 5.1)
            self.assertEqual(rows, [])

    def test_complete_queue_report_exports_csv_and_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / 'net'
            path.mkdir()
            telemetry = dict(task=1, layer=0, mode='net', density=.15, removed_ratio=.4,
                net_to_gross_norm=.8, independent_amplified_coordinates=12,
                independent_reversed_coordinates=3, shared_amplified_coordinates=0,
                shared_reversed_coordinates=0, branch_sum_error=1e-8)
            (path / 'training.log').write_text(
                "[trainer.py] => CNN: {'total': 82.6, 'old': 82.1, 'new': 87.1}\n" * 10 +
                '[trainer.py] => Average Accuracy: 87.214\n[trainer.py] => Forgetting: 5.100\n' +
                'ComposedConflict ' + json.dumps(telemetry) + '\n')
            queue.summarize(directory, [dict(mode='net', status='completed', minutes=80.)])
            report = json.loads((directory / 'results.json').read_text())[0]
            self.assertEqual(report['tasks_reported'], 10)
            self.assertEqual(report['Average'], 87.214)
            self.assertEqual(report['independent_reversed_coordinates'], 3)
            self.assertTrue((directory / 'comparison.png').is_file())
            self.assertTrue((directory / 'gate_diagnostics.csv').is_file())
            self.assertTrue((directory / 'results.csv').is_file())


if __name__ == '__main__':
    unittest.main()
