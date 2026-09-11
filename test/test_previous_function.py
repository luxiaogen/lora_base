import unittest

import torch

from methods.dlora import selective_previous_function_loss
from test import test_cross_task_training as training_tests


class PreviousFunctionTests(unittest.TestCase):
    def test_selects_largest_old_function_drift_and_only_updates_student(self):
        student = torch.tensor([
            [3.0, 0.0],
            [0.0, 3.0],
            [2.8, 0.2],
            [0.2, 2.8],
        ], requires_grad=True)
        teacher = torch.tensor([
            [0.0, 3.0],
            [3.0, 0.0],
            [2.8, 0.2],
            [0.2, 2.8],
        ], requires_grad=True)

        loss, metrics = selective_previous_function_loss(student, teacher, ratio=0.5, scale=1.0)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(metrics["selected_count"], 2)
        self.assertAlmostEqual(float(metrics["selected_ratio"]), 0.5)
        self.assertGreater(student.grad[:2].abs().sum().item(), 0.0)
        self.assertEqual(student.grad[2:].abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_zero_ratio_is_rejected(self):
        logits = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            selective_previous_function_loss(logits, logits, ratio=0.0, scale=1.0)

    def test_extra_training_loss_routes_previous_function_gradient_to_features(self):
        learner = training_tests.CrossTaskTrainingTests().make_learner('task_local', task=1)
        learner.args.update({
            'dual_mask_previous_function_enabled': True,
            'dual_mask_previous_function_weight': 1.0,
            'dual_mask_conflict_ratio': 0.5,
            'dual_mask_reg_weight': 0.0,
            'dual_mask_anchor_reg_enabled': False,
        })
        output = learner._network(torch.eye(4))
        output['features'].retain_grad()
        teacher_logits = torch.flip(
            learner._network(output['features'].detach(), fc_only=True)[:, :2],
            dims=(1,),
        )

        loss = learner._extra_training_loss(
            output=output,
            batch_context={'previous_old_logits': teacher_logits},
        )
        loss.backward()

        self.assertGreater(output['features'].grad.norm().item(), 0.0)
        self.assertIn('previous_function', learner._last_training_loss_metrics)
        self.assertGreater(float(learner._last_training_loss_metrics['previous_function']), 0.0)


if __name__ == "__main__":
    unittest.main()
