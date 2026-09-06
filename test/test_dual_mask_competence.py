import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dual_mask_metrics import (  # noqa: E402
    split_prototype_competence,
    split_prototype_ncm_diagnostics,
)


class CompetenceTests(unittest.TestCase):
    def test_ncm_plasticity_demand_distinguishes_confidence(self):
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        indices = torch.arange(8)
        separated = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        ambiguous = separated.clone()
        ambiguous[[0, 2, 4, 6]] = torch.tensor([1.0, 1.0])

        separated_loss, separated_demand = split_prototype_ncm_diagnostics(
            separated,
            targets,
            indices,
            holdout_mod=2,
            scale=10.0,
        )
        ambiguous_loss, ambiguous_demand = split_prototype_ncm_diagnostics(
            ambiguous,
            targets,
            indices,
            holdout_mod=2,
            scale=10.0,
        )

        self.assertGreater(ambiguous_loss, separated_loss)
        self.assertGreater(ambiguous_demand, separated_demand)
        self.assertGreaterEqual(separated_demand, 0.0)
        self.assertLessEqual(ambiguous_demand, 1.0)

    def test_holdout_prototype_competence_uses_accuracy(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.6],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.6, 0.8],
                [0.0, 1.0],
            ]
        )
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        indices = torch.arange(8)

        competence, _, _ = split_prototype_competence(
            features, targets, indices, holdout_mod=2
        )
        self.assertEqual(competence, 1.0)

    def test_all_seen_candidates_include_old_prototypes(self):
        features = torch.tensor(
            [
                [0.8, 0.6],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        targets = torch.ones(4, dtype=torch.long)
        indices = torch.arange(4)

        new_only, _, _ = split_prototype_competence(
            features, targets, indices, holdout_mod=2
        )
        all_seen, _, _ = split_prototype_competence(
            features,
            targets,
            indices,
            holdout_mod=2,
            old_prototypes=torch.tensor([[0.8, 0.6]]),
            old_class_ids=torch.tensor([0]),
        )

        self.assertEqual(new_only, 1.0)
        self.assertEqual(all_seen, 0.5)

    def test_holdout_prototype_margin_is_continuous(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])
        indices = torch.arange(4)

        competence, _, _ = split_prototype_competence(
            features, targets, indices, holdout_mod=2, metric="margin"
        )

        self.assertAlmostEqual(competence, 0.75, places=6)

if __name__ == "__main__":
    unittest.main()
