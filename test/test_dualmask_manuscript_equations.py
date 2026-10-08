"""Fixed-tensor checks for the original O manuscript, not efficacy checks."""
from types import SimpleNamespace
import unittest

import torch

from models.attention import _energy_coverage_mask, _energy_coverage_with_ratio_floor_mask, _normalize_score
from test import test_global_conflict_budget
from utils.dual_mask_metrics import build_prototypes


class ManuscriptEquationTests(unittest.TestCase):
    def make(self):
        module = test_global_conflict_budget.GlobalBudgetSelectionTests._make_attention('layer',
            dual_mask_conflict_energy_adaptive=True, dual_mask_conflict_reg_enabled=True,
            dual_mask_protect_strength_mode='competence')
        module.w0_importance.copy_(torch.arange(48).reshape(12, 4).float() / 47)
        module.general_mask.copy_((module.w0_importance > .6).float())
        module.isolated_mask.copy_(1 - module.general_mask)
        module.effective_protect_strength = .6
        return module

    def test_protection_is_prefix_but_conflict_retains_boundary_ties(self):
        score = torch.tensor([[4., 2., 2., 0.]])
        protect = _energy_coverage_mask(score, .75)
        self.assertEqual(int(protect.sum()), 2)
        self.assertEqual(float((protect * score).sum()), 6.)
        conflict = _energy_coverage_with_ratio_floor_mask(score, .5, .75)
        torch.testing.assert_close(conflict, torch.tensor([[1., 1., 1., 0.]]))
        floor = _energy_coverage_with_ratio_floor_mask(torch.tensor([[9., 1., .5, .1]]), .5, .5)
        torch.testing.assert_close(floor, torch.tensor([[1., 1., 0., 0.]]))
        torch.testing.assert_close(_energy_coverage_with_ratio_floor_mask(torch.zeros(4), .5, .5), torch.zeros(4))

    def test_forward_gate_and_unscaled_regularizer(self):
        module = self.make()
        delta = torch.arange(1, 49).reshape(12, 4).float() / 50
        importance = module.w0_importance
        score = _normalize_score(importance * _normalize_score(delta.abs()))
        conflict = _energy_coverage_with_ratio_floor_mask(score, .25, .5)
        for isolated in (False, True):
            region = module.isolated_mask if isolated else 1 - .6 * module.general_mask
            gate = region * (1 - .5 * conflict)
            unscaled = gate * delta
            torch.testing.assert_close(module._safe_delta(delta, isolated), unscaled)
            for gamma in (.5, .75):
                torch.testing.assert_close(module._safe_delta(gamma * delta, isolated), gamma * unscaled)
            expected = ((importance + score) * unscaled.square()).mean()
            unit = SimpleNamespace(A_weight=torch.eye(4), B_weight=delta)
            torch.testing.assert_close(module._joint_conflict_regularization(unit, isolated), expected)

    def test_prototype_mean_precedes_normalization(self):
        features = torch.tensor([[10., 0.], [0., 1.], [0., 3.]])
        actual, classes = build_prototypes(features, torch.tensor([0, 0, 1]))
        expected = torch.tensor([[10., 1.], [0., 1.]])
        expected = torch.nn.functional.normalize(expected, dim=1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(classes, torch.tensor([0, 1]))

    def test_controller_clips_and_rounds_and_strength_is_capped(self):
        module = self.make()
        module.dual_mask_competence_adaptive = True
        module.dual_mask_plasticity_adaptive = True
        module.set_pretrained_competence(1.2, .25)
        self.assertAlmostEqual(module.pretrained_control_competence, .75)
        self.assertAlmostEqual(module.effective_energy_coverage, .7 + .25 * .75)
        self.assertAlmostEqual(module.effective_protect_strength, .75)
        self.assertEqual(module.current_private_rank, max(1, round(module.rank * (1 - .75 * .75))))
        module.dual_mask_conflict_old_overlap_adaptive = True
        module.dual_mask_conflict_strength = .9
        module.set_pretrained_old_overlap_risk(2.)
        self.assertEqual(module._conflict_parameters()[1], 1.)


if __name__ == '__main__':
    unittest.main()
