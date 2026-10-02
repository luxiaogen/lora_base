import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('direction_queue', ROOT / 'scripts/run_p_direction_score.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class DirectionQueueTests(unittest.TestCase):
    def test_three_groups_change_only_p_scoring(self):
        base = queue.settings_for('coordinate')
        for mode in queue.MODES:
            config = queue.settings_for(mode)
            self.assertEqual(config.pop('p_direction_score'), mode)
            self.assertEqual(config, {k: v for k, v in base.items() if k != 'p_direction_score'})
            for key, expected in dict(max_tasks=10, epochs=20, init_epoch=20, ca_epochs=5,
                dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
                dual_mask_composed_conflict='off', dual_mask_conflict_granularity='layer',
                dual_mask_private_conflict_mode='global', dual_mask_conflict_merge_mode='suppress',
                dual_mask_s_conflict_enabled=True, dual_mask_p_conflict_enabled=True,
                dual_mask_uniform_norm_matched=False, dual_mask_safe_residual_enabled=False,
                p_step_direction='off', p_hard_zero_mode='off', branch_choice_mode='off',
                p_old_gradient_oracle=False, p_conflict_freeze_epoch=0,
                p_conflict_strength_warmup=False, plora_train_a=False,
                wpre_distill_weight=0, ridge_fusion_enabled=False, save_task_weights=False,
                incremental_holdout=False, stage_audit=False).items():
                self.assertEqual(config[key], expected)
            self.assertNotIn('data_path', config)
            self.assertNotIn('pretrained_path', config)

    def test_smoke_is_separate_from_t10(self):
        settings = queue.settings_for('signed', smoke=True)
        self.assertEqual((settings['max_tasks'], settings['epochs'], settings['init_epoch'], settings['ca_epochs']), (2, 1, 1, 1))
        command, _ = queue.command_for('signed', Path('preview'))
        self.assertIn('p_direction_score=signed', command)

    def test_report_reads_actual_norms_and_complete_task_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'signed').mkdir()
            row = dict(task=1, layer=0, mode='signed', selected_k=[1, 2, 3], reference_k=[1, 2, 3],
                strength=[.5, .5, .5], reference_removed_norm=[1, 2, 3], actual_removed_norm=[1, 2, 3],
                base_norm=[2, 4, 6], spectral_fractions=[[.1, .2, .7]] * 3, merge_error=0)
            (directory / 'signed/training.log').write_text(
                "[trainer.py] => CNN: {'total': 82.6, 'old': 82.1, 'new': 87.1}\n" * 10 +
                '[trainer.py] => Average Accuracy: 87.214\nPDirectionScore ' + json.dumps(row) + '\n')
            queue.summarize(directory, [dict(mode='signed', status='completed', exit_code=0, minutes=70)])
            summary = json.loads((directory / 'results.json').read_text())[0]
            self.assertEqual(summary['tasks_reported'], 10)
            self.assertEqual(summary['max_relative_norm_error'], 0)
            self.assertTrue((directory / 'direction_diagnostics.csv').is_file())


if __name__ == '__main__':
    unittest.main()
