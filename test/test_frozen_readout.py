import csv
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class FrozenReadoutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.frozen_readout'),
                             'Fixed-feature readout is not implemented yet')
        self.api = importlib.import_module('utils.frozen_readout')

    def test_streaming_statistics_equal_one_shot_without_retained_samples(self):
        x = torch.tensor([[1., 2.], [3., 4.], [2., 0.]])
        y = torch.tensor([0, 1, 0])
        stats = self.api.FrozenReadout(2, 3)
        stats.update(x[:2], y[:2])
        stats.update(x[2:], y[2:])
        torch.testing.assert_close(stats.gram, torch.tensor([[14., 14.], [14., 20.]]))
        torch.testing.assert_close(stats.class_sum, torch.tensor([[3., 3., 0.], [2., 4., 0.]]))
        self.assertEqual(stats.samples, 3)
        self.assertEqual(stats.storage_bytes, (2 * 2 + 2 * 3) * 4)
        self.assertFalse(any(isinstance(v, list) for v in vars(stats).values()))

    def test_ridge_matches_direct_regularized_least_squares(self):
        x = torch.tensor([[1., 0.], [1., 1.], [0., 2.]])
        y = torch.tensor([0, 0, 1])
        stats = self.api.FrozenReadout(2, 2)
        stats.update(x, y)
        weight, regularizer = stats.ridge_weight(0.1, 2)
        self.assertAlmostEqual(regularizer, 0.35)
        expected = torch.linalg.solve(torch.tensor([[2.35, 1.], [1., 5.35]], dtype=torch.float64),
                                      torch.tensor([[2., 0.], [1., 2.]], dtype=torch.float64))
        torch.testing.assert_close(weight, expected.float())

    def test_ncm_uses_raw_feature_means_not_means_of_normalized_features(self):
        stats = self.api.FrozenReadout(2, 3)
        stats.update(torch.tensor([[10., 0.], [0., 1.], [0., 2.]]), torch.tensor([0, 0, 1]))
        scores = stats.ncm_scores(torch.eye(2), 2)
        torch.testing.assert_close(scores, torch.tensor([[10. / 101**0.5, 0.],
                                                         [1. / 101**0.5, 1.]]))
        self.assertEqual(tuple(scores.shape), (2, 2))  # Never predict unseen classes.

    def test_alpha_selection_uses_task0_holdout_and_does_not_update_statistics(self):
        stats = self.api.FrozenReadout(2, 2)
        stats.update(torch.eye(2), torch.tensor([0, 1]))
        before = stats.gram.clone()
        alpha, rows = self.api.select_alpha(stats, torch.eye(2), torch.tensor([0, 1]), 2)
        self.assertEqual(alpha, rows[0]['alpha'])  # Deterministic first-on-tie.
        self.assertTrue(all(row['accuracy'] == 100. for row in rows))
        torch.testing.assert_close(stats.gram, before)
        self.assertEqual(stats.samples, 2)

    def test_holdout_can_select_a_nondefault_alpha(self):
        stats = self.api.FrozenReadout(2, 2)
        stats.update(torch.tensor([[1., 0.], [1., 1.], [0., 2.]]), torch.tensor([0, 0, 1]))
        alpha, rows = self.api.select_alpha(stats, torch.tensor([[1., 2.45]]), torch.tensor([1]), 2)
        self.assertEqual(alpha, 1.)
        self.assertEqual([r['accuracy'] for r in rows], [0., 0., 0., 0., 100.])

    def test_partitions_reconstruct_nested_current_only_holdouts(self):
        targets = np.repeat(np.arange(4), 6)
        fit, holdout = self.api.training_partition(targets, 2, 4, 3, 2)
        self.assertEqual(fit.tolist(), [14, 17, 20, 23])
        self.assertEqual(holdout.tolist(), [13, 16, 19, 22])
        fit, holdout = self.api.training_partition(targets, 0, 2, 3, 2)
        self.assertEqual(fit.tolist(), [1, 3, 5, 7, 9, 11])
        self.assertEqual(holdout.tolist(), [0, 2, 4, 6, 8, 10])

    def test_reporting_separates_legal_readout_from_label_oracle(self):
        target = torch.tensor([0, 0, 1, 1])
        report = self.api.readout_report(torch.tensor([0, 1, 1, 0]),
                                         torch.tensor([1, 0, 1, 0]), target, 1)
        self.assertEqual(report['total']['base_accuracy'], 50.)
        self.assertEqual(report['total']['readout_accuracy'], 50.)
        self.assertEqual(report['total']['oracle_accuracy'], 75.)
        self.assertEqual(report['total']['rescued'], 1)
        self.assertEqual(report['total']['harmed'], 1)
        self.assertEqual(report['old']['readout_accuracy'], 50.)

    def test_cached_reference_aligns_ids_and_never_pairs_mismatched_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'target', 'base_prediction', 'anchor_prediction'])
                writer.writerows([[1, 1, 0, 1], [0, 0, 0, 1]])
            base, anchor, status = self.api.load_reference(path, torch.tensor([0, 1]))
            self.assertEqual(status, 'matched_indices_and_labels')
            self.assertEqual(base.tolist(), [0, 0])
            self.assertEqual(anchor.tolist(), [1, 1])
            base, _, status = self.api.load_reference(path, torch.tensor([1, 0]))
            self.assertIsNone(base)
            self.assertEqual(status, 'sample_alignment_mismatch')

    def test_missing_reference_allows_standalone_readout(self):
        base, anchor, status = self.api.load_reference(Path('/not/a/cache.csv'), torch.tensor([0]))
        self.assertIsNone(base)
        self.assertIsNone(anchor)
        self.assertEqual(status, 'cache_missing')


if __name__ == '__main__':
    unittest.main()
