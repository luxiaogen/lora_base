import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods.dlora import Learner


class LogitNetwork(nn.Module):
    def interface(self, inputs):
        return inputs

    def extract_vector(self, inputs):
        return inputs


class RawTaskScoreDiagnosticTests(unittest.TestCase):
    def test_logs_aggregate_and_per_task_score_distributions(self):
        learner = Learner.__new__(Learner)
        learner._network = LogitNetwork()
        learner._device = torch.device("cpu")
        learner._cur_task = 1
        learner.task_sizes = [2, 2]
        learner.args = {
            "classification_ncm_task_evidence_diagnostics": True,
            "dual_mask_conflict_ratio": 0.1,
        }
        learner._class_means = torch.eye(4)
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
        self.assertIn("NCM task evidence Task 1 (test-only)", output)
        self.assertIn("recoverable=", output)
        self.assertIn("introduced=", output)
        self.assertIn("oracle_union=", output)
        self.assertIn("Ambiguous NCM switch Task 1 (test-only)", output)
        self.assertIn("global_error_coverage=", output)
        self.assertIn("switch_delta=", output)

    def test_ncm_intersection_counts_recovery_and_damage(self):
        learner = Learner.__new__(Learner)
        learner._cur_task = 1
        learner.args = {"dual_mask_conflict_ratio": 0.25}
        values = {
            "target_tasks": torch.tensor([0, 1, 0, 1]),
            "predicted_tasks": torch.tensor([0, 0, 1, 1]),
            "task_scores": torch.tensor([
                [0.8, 0.2],
                [0.50, 0.49],
                [0.4, 0.7],
                [0.2, 0.8],
            ]),
        }
        ncm_tasks = torch.tensor([0, 1, 1, 0])

        with self.assertLogs(level="INFO") as captured:
            learner._log_ncm_task_evidence(values, ncm_tasks)

        output = "\n".join(captured.output)
        self.assertIn("both_correct=1(25.00%)", output)
        self.assertIn("recoverable=1(25.00%)", output)
        self.assertIn("introduced=1(25.00%)", output)
        self.assertIn("both_wrong=1(25.00%)", output)
        self.assertIn("oracle_union=75.00", output)
        self.assertIn("switch_accuracy=75.00, switch_delta=+25.00", output)


if __name__ == "__main__":
    unittest.main()
