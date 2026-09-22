import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose_lora_a_overlap.py"
SPEC = importlib.util.spec_from_file_location("diagnose_lora_a_overlap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoRIAOverlapTests(unittest.TestCase):
    def test_kaiming_sampling_is_deterministic_and_has_expected_shape(self):
        first_generator = torch.Generator().manual_seed(1993)
        second_generator = torch.Generator().manual_seed(1993)
        first = MODULE.make_lora_a("kaiming", 3, 4, 16, first_generator)
        second = MODULE.make_lora_a("kaiming", 3, 4, 16, second_generator)
        self.assertEqual(tuple(first.shape), (3, 4, 16))
        self.assertTrue(torch.equal(first, second))
        self.assertLessEqual(float(first.abs().max()), 1 / 16**0.5)

    def test_kaiming_scale_matches_per_task_repo_initialization(self):
        generator = torch.Generator().manual_seed(1993)
        matrices = MODULE.make_lora_a("kaiming", 64, 8, 128, generator)
        expected_variance = 1 / (3 * 128)
        self.assertLess(
            abs(float(matrices.var(unbiased=False)) - expected_variance),
            expected_variance * 0.05,
        )

    def test_known_disjoint_subspaces_have_zero_overlap(self):
        matrices = torch.zeros(2, 2, 4)
        matrices[0, 0, 0] = 1.0
        matrices[0, 1, 1] = 1.0
        matrices[1, 0, 2] = 1.0
        matrices[1, 1, 3] = 1.0

        metrics = MODULE.pairwise_overlap(matrices)

        for value in metrics.values():
            self.assertEqual(float(value.item()), 0.0)

    def test_random_subspace_overlap_tracks_rank_over_dimension(self):
        generator = torch.Generator().manual_seed(1993)
        matrices = MODULE.make_lora_a("kaiming", 20, 4, 128, generator)
        metrics = MODULE.pairwise_overlap(matrices, include_spectral=False)
        observed = float(metrics["subspace_overlap"].mean().item())
        self.assertLess(abs(observed - 4 / 128), 0.01)


if __name__ == "__main__":
    unittest.main()
