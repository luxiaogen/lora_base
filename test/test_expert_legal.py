import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.data_manager import DataManager
from utils.expert_legal import candidate_specs, choose, fit_policy, signal_view
from utils.two_expert_oracle import expert_rows
from scripts.evaluate_expert_legal import evaluate


class ExpertLegalTests(unittest.TestCase):
    def rows(self):
        return dict(index=np.arange(12), target=np.zeros(12, dtype=int),
                    base_prediction=np.array([1, 0, 1] * 4),
                    anchor_prediction=np.array([0, 1, 2] * 4),
                    proposal_advantage=np.array([.8, .2, .1] * 4),
                    margin_advantage=np.array([.4, .1, -.1] * 4),
                    base_uncertainty=np.array([-.02, -.04, -.06] * 4))

    def test_signal_view_excludes_ground_truth_and_true_task(self):
        rows = self.rows()
        rows['task_id_prediction'] = np.arange(12)
        view = signal_view(rows)
        self.assertNotIn('target', view)
        self.assertNotIn('task_id_prediction', view)
        self.assertNotIn('index', view)
        spec = dict(name='fixed', kind='fixed', signal='proposal_advantage', threshold=.3)
        policy = fit_policy(spec, rows)
        other = copy.deepcopy(rows)
        other['target'][:] = 9
        other['task_id_prediction'][:] = 9
        self.assertTrue(np.array_equal(choose(policy, view), choose(policy, signal_view(other))))

    def test_cost_threshold_and_ties_include_whole_groups(self):
        rows = self.rows()
        spec = dict(name='threshold', kind='utility', signal='proposal_advantage', cost=3)
        policy = fit_policy(spec, rows)
        selected = choose(policy, signal_view(rows))
        self.assertEqual(np.flatnonzero(selected).tolist(), [0, 3, 6, 9])
        self.assertGreater(policy['threshold'], .2)
        rows['target'][:] = 8
        self.assertIsNone(fit_policy(spec, rows)['threshold'])

    def test_quantile_uses_only_training_scores(self):
        rows = self.rows()
        spec = dict(name='tail', kind='quantile', signal='proposal_advantage', quantile=.9)
        before = fit_policy(spec, rows)
        rows['target'][:] = 7
        self.assertEqual(before, fit_policy(spec, rows))

    def test_ridge_utility_and_empty_calibration_are_conservative(self):
        rows = self.rows()
        spec = dict(name='ridge', kind='ridge', features='core', cost=3)
        policy = fit_policy(spec, rows)
        self.assertEqual(choose(policy, signal_view(rows)).shape, (12,))
        disabled = fit_policy(spec, {key: value[:0] for key, value in rows.items()})
        self.assertFalse(choose(disabled, signal_view(rows)).any())

    def test_candidate_names_unique_and_extended_rules_require_new_columns(self):
        core = candidate_specs(extended=False)
        extended = candidate_specs(extended=True)
        self.assertGreaterEqual(len(core), 30)
        self.assertGreater(len(extended), len(core))
        self.assertEqual(len({item['name'] for item in extended}), len(extended))

    def test_extra_confidence_signals_are_label_free(self):
        base = torch.tensor([[.6, .4, .1], [.1, .6, .4]])
        anchor = base.flip(1)
        rows = expert_rows(base, anchor, torch.tensor([0, 1]), torch.arange(2), torch.arange(3), 1)
        other = expert_rows(base, anchor, torch.tensor([2, 0]), torch.arange(2), torch.arange(3), 1)
        for key in ('prototype_advantage', 'entropy_advantage'):
            self.assertTrue(torch.equal(rows[key], other[key]))
            self.assertTrue(torch.isfinite(rows[key]).all())

    def test_holdout_leaves_task0_intact_and_excludes_all_train_consumers(self):
        def setup(manager, *args):
            manager._train_data = np.arange(60)
            manager._train_targets = np.repeat(np.arange(6), 10)
            manager._test_data = np.arange(100, 106)
            manager._test_targets = np.arange(6)
            manager._class_order = list(range(6))
            manager._train_trsf = manager._test_trsf = manager._common_trsf = []
            manager.use_path = False
        with patch.object(DataManager, '_setup_data', setup):
            manager = DataManager('fake', False, 1993, 2, 2,
                                  {'two_expert_calibration_holdout_mod': 5})
        self.assertEqual(manager.get_dataset([0, 1], 'train', 'test').images.tolist(), list(range(20)))
        self.assertEqual(len(manager._holdout_targets), 8)
        held = manager.get_incremental_holdout([2, 3])
        self.assertEqual(held.labels.tolist(), [2, 2, 3, 3])
        for mode in ('train', 'test'):
            train = manager.get_dataset([2, 3], 'train', mode)
            self.assertFalse(set(train.images) & set(held.images))
        self.assertEqual(manager.get_dataset(range(6), 'test', 'test').images.tolist(), list(range(100, 106)))

    def test_task0_data_rng_and_small_initialization_match_with_holdout_on_off(self):
        def setup(manager, *args):
            manager._train_data = np.arange(60)
            manager._train_targets = np.repeat(np.arange(6), 10)
            manager._class_order = np.random.permutation(6).tolist()
            manager._train_trsf = manager._test_trsf = manager._common_trsf = []
            manager.use_path = False
        results = []
        for settings in ({}, {'two_expert_calibration_holdout_mod': 5}):
            np.random.seed(1993)
            torch.manual_seed(1993)
            with patch.object(DataManager, '_setup_data', setup):
                manager = DataManager('fake', True, 1993, 2, 2, settings)
            results.append((manager.get_dataset([0, 1], 'train', 'test').images,
                            np.random.rand(4), torch.nn.Linear(4, 3).weight.detach().clone()))
        self.assertTrue(np.array_equal(results[0][0], results[1][0]))
        self.assertTrue(np.array_equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def write_cache(self, folder, source):
        rows = self.rows()
        for task in range(2):
            for name in ('test_report_only', 'current_train_seen_probe' if task == 0 else source):
                stem = Path(folder) / f'task_{task:02d}_{name}'
                with stem.with_suffix('.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(rows)
                    writer.writerows(zip(*rows.values()))
                stem.with_suffix('.json').write_text(json.dumps(dict(task=task, source=name,
                    groups=dict(old=dict(n=0), new=dict(n=12)),
                    probe_is_unseen_holdout=source == 'current_train_holdout')))

    def test_cache_evaluation_fits_before_reading_test_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_cache(folder, 'current_train_holdout')
            output = Path(folder) / 'evaluation'
            result = evaluate(folder, output, 'current_train_holdout', expected_tasks=2)
            policies = json.loads((output / 'policies/task_01.json').read_text())
            self.assertFalse(result['test_used_for_fitting_or_rule_selection'])
            self.assertTrue(result['task1_plus_calibration_expert_unseen'])
            self.assertFalse(result['task0_calibration_expert_unseen'])
            self.assertEqual(policies['selection_source'], 'calibration_audit_only')
            self.assertEqual(policies['fit_audit_index_overlap'], 0)
            self.assertTrue((output / 'task_metrics.csv').exists())
            self.assertTrue((output / 'summary.csv').exists())
            self.assertTrue((output / 'manifest.json').exists())
            self.assertEqual(result['completed_tasks'], [0, 1])

    def test_test_labels_cannot_change_policies_or_train_selected_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_cache(folder, 'current_train_seen_probe')
            evaluate(folder, Path(folder) / 'before', 'current_train_seen_probe', expected_tasks=2)
            for path in Path(folder).glob('task_*_test_report_only.csv'):
                lines = path.read_text().splitlines()
                # target is the second column; predictions and scores do not change.
                changed = [lines[0]] + [','.join([line.split(',')[0], '9'] + line.split(',')[2:])
                                       for line in lines[1:]]
                path.write_text('\n'.join(changed) + '\n')
            evaluate(folder, Path(folder) / 'after', 'current_train_seen_probe', expected_tasks=2)
            for task in range(2):
                first = json.loads((Path(folder) / f'before/policies/task_{task:02d}.json').read_text())
                second = json.loads((Path(folder) / f'after/policies/task_{task:02d}.json').read_text())
                self.assertEqual(first, second)

    def test_missing_cache_reports_fail_in_analysis_not_training(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_cache(folder, 'current_train_seen_probe')
            (Path(folder) / 'task_01_test_report_only.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate(folder, Path(folder) / 'out', 'current_train_seen_probe', expected_tasks=2)

    def test_launchers_are_single_t10_and_keep_machine_paths_and_weights_off(self):
        for machine, holdout in (('3090', 20), ('5090', 10)):
            output = subprocess.check_output(['bash', f'scripts/10_01_imgr10_expert_legal_{machine}.sh',
                                             '--dry-run'], text=True)
            self.assertEqual(output.count('Command: '), 2)
            self.assertIn(f'two_expert_calibration_holdout_mod={holdout}', output)
            self.assertIn('save_task_weights=false', output)
            self.assertIn('dual_mask_anchor_reg_weight=2.5', output)
            self.assertNotIn('--set data_path=', output)
            self.assertNotIn('--set device=', output)
            self.assertNotIn('--set pretrained_path=', output)


if __name__ == '__main__':
    unittest.main()
