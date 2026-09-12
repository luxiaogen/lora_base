import unittest

import torch

from utils.task_routing import predict_with_task_evidence


class TaskScoreRoutingTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [0.80, 0.65, 0.74, 0.20, 0.68, 0.40],
            [0.60, 0.10, 0.75, 0.72, 0.50, 0.20],
        ])
        self.task_sizes = [2, 2, 2]

    def test_global_mode_matches_flat_argmax(self):
        classes, tasks = predict_with_task_evidence(
            self.logits, self.task_sizes, "global"
        )

        self.assertTrue(torch.equal(classes, self.logits.argmax(dim=1)))
        self.assertEqual(tasks.tolist(), [0, 1])

    def test_centered_max_routes_by_within_head_confidence(self):
        classes, tasks = predict_with_task_evidence(
            self.logits, self.task_sizes, "centered_max"
        )

        self.assertEqual(tasks.tolist(), [1, 0])
        self.assertEqual(classes.tolist(), [2, 0])

    def test_top1_top2_preserves_selected_head_class_argmax(self):
        classes, tasks = predict_with_task_evidence(
            self.logits, self.task_sizes, "top1_top2"
        )

        self.assertEqual(tasks.tolist(), [1, 0])
        self.assertEqual(classes.tolist(), [2, 0])
        for row, (class_id, task_id) in enumerate(zip(classes, tasks)):
            start = sum(self.task_sizes[:task_id])
            stop = start + self.task_sizes[task_id]
            self.assertEqual(class_id.item(), start + self.logits[row, start:stop].argmax().item())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "classification_inference_mode"):
            predict_with_task_evidence(self.logits, self.task_sizes, "unknown")


if __name__ == "__main__":
    unittest.main()
