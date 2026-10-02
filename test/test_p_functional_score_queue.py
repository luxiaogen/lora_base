import json
from pathlib import Path
import tempfile
import unittest

from test.test_p_direction_score_queue import queue
from utils.p_score_diagnostic import MODES


class FunctionalQueueTests(unittest.TestCase):
    def setUp(self):
        self.previous = queue.SPEC
        queue.SPEC = queue.ROOT / 'scripts/sweeps/imgr10_p_functional_score_3090.json'

    def tearDown(self):
        queue.SPEC = self.previous

    def test_six_groups_change_only_score_and_preserve_full_protocol(self):
        expected = queue.settings_for('coordinate')
        for mode in MODES:
            settings = queue.settings_for(mode)
            self.assertEqual(settings.pop('p_direction_score'), mode)
            self.assertEqual(settings, {k: v for k, v in expected.items() if k != 'p_direction_score'})
            self.assertEqual(settings['max_tasks'], 10)
            self.assertEqual(settings['dual_mask_anchor_reg_weight'], 2.5)
            self.assertEqual(settings['dual_mask_reg_weight'], .01)
            self.assertTrue(settings['p_score_counterfactual_report'])
            self.assertFalse(settings['save_task_weights'])
            self.assertEqual(settings['memory_size'], 0)
            self.assertFalse(settings['p_old_gradient_oracle'])
            self.assertNotIn('data_path', settings)
        self.assertEqual(tuple(v['name'] for v in json.loads(queue.SPEC.read_text())['variants']), MODES)

    def test_old_completed_control_diff_is_diagnostic_only(self):
        old = json.loads(self.previous.read_text())['common_overrides']
        new = queue.settings_for('coordinate')
        new.pop('p_direction_score')
        new.pop('p_score_counterfactual_report')
        new['wandb_group'] = old['wandb_group']
        self.assertEqual(new, old)

    def test_new_smoke_commands_are_real_training_with_separate_prefixes(self):
        for mode in MODES:
            command, settings = queue.command_for(mode, Path('new_smoke'), True)
            self.assertEqual(settings['max_tasks'], 2)
            self.assertEqual(settings['epochs'], 1)
            self.assertIn('p_direction_score=' + mode, command)
            self.assertIn('main.py', command)

    def test_mechanism_csv_is_measured_and_not_a_training_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'coordinate').mkdir()
            metrics = {group: dict(n=64, accuracy=90., margin=.2, corrected=2, broken=1)
                       for group in ('total', 'old', 'new')}
            row = dict(task=1, applied_mode='wpre_qk', reference_mode='coordinate', source='test_report_only_fixed_weights',
                       sample_sha256='same_samples', metrics=metrics, max_relative_removed_norm_error=0.,
                       layers=[dict(ungated={key: 1. for key in MODES[2:]}, gated={key: .5 for key in MODES[2:]})])
            (directory / 'coordinate/training.log').write_text('PScoreCounterfactual ' + json.dumps(row) + '\n')
            queue.summarize(directory, [dict(mode='coordinate', status='completed', exit_code=0, minutes=75)])
            content = (directory / 'mechanism_interventions.csv').read_text()
            self.assertIn('old_corrected', content)
            self.assertIn('old_broken', content)
            self.assertIn('reference_mode', content)
            self.assertIn('test_report_only_fixed_weights', content)


if __name__ == '__main__':
    unittest.main()
