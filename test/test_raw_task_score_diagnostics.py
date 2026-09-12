import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods.dlora import Learner


class LogitNetwork(nn.Module):
    def interface(self, inputs):
        return inputs


class RawTaskScoreDiagnosticTests(unittest.TestCase):
    def test_logs_aggregate_and_per_task_score_distributions(self):
        learner = Learner.__new__(Learner)
        learner._network = LogitNetwork()
        learner._device = torch.device("cpu")
        learner._cur_task = 1
        learner.task_sizes = [2, 2]
        logits = torch.tensor([
            [0.90, 0.10, 0.70, 0.20],
            [0.60, 0.30, 0.80, 0.10],
            [0.75, 0.20, 0.65, 0.40],
            [0.55, 0.50, 0.70, 0.20],
        ])
        targets = torch.tensor([0, 2, 0, 2])
        learner.test_loader = DataLoader(
            TensorDataset(torch.arange(len(targets)), logits, targets),
            batch_size=2,
        )
        learner._network.train()
        torch.manual_seed(17)
        rng_state = torch.random.get_rng_state().clone()

        with self.assertLogs(level="INFO") as captured:
            learner._log_raw_task_score_diagnostics()

        output = "\n".join(captured.output)
        self.assertTrue(learner._network.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertIn("Raw task score diagnostic Task 1 (test-only)", output)
        self.assertIn("pairwise_auc=", output)
        self.assertIn("correct_q10=", output)
        self.assertIn("wrong_max_q90=", output)
        self.assertIn("margin_q50=", output)
        self.assertIn("true_task=0", output)
        self.assertIn("true_task=1", output)


if __name__ == "__main__":
    unittest.main()
