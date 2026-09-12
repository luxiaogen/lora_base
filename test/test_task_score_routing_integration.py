import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods.dlora import Learner


class LogitNetwork(nn.Module):
    def interface(self, inputs):
        return inputs


class TaskScoreRoutingIntegrationTests(unittest.TestCase):
    def test_eval_uses_configured_task_evidence_without_true_task_id(self):
        learner = Learner.__new__(Learner)
        learner._network = LogitNetwork()
        learner._device = torch.device("cpu")
        learner.class_num = 2
        learner.task_sizes = [2, 2, 2]
        learner.classification_inference_mode = "centered_max"
        learner.topk = 1
        logits = torch.tensor([
            [0.80, 0.65, 0.74, 0.20, 0.68, 0.40],
            [0.60, 0.10, 0.75, 0.72, 0.50, 0.20],
        ])
        targets = torch.tensor([2, 0])
        loader = DataLoader(TensorDataset(torch.arange(2), logits, targets), batch_size=2)

        classes, _, observed_targets, tasks, _ = learner._eval_cnn(loader)

        self.assertEqual(classes.tolist(), [2, 0])
        self.assertEqual(tasks.tolist(), [1, 0])
        self.assertEqual(observed_targets.tolist(), targets.tolist())


if __name__ == "__main__":
    unittest.main()
