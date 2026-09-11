import unittest

import torch
from torch import nn
from torch.nn import functional as F

from methods import dlora


class TokenEncoder(nn.Module):
    def forward(self, inputs, task_id=None):
        return inputs[:, None, :]


class TinyBiasClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.tensor([
            [0.80, 0.10, 0.90, 0.00],
            [0.10, 0.80, 0.00, 0.90],
            [0.20, 0.10, 0.80, 0.00],
            [0.10, 0.20, 0.00, 0.80],
        ])
        self.classifier_pool = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            self.classifier_pool[0].weight.copy_(logits[:, :2].T)
            self.classifier_pool[1].weight.copy_(logits[:, 2:].T)
        self.register_buffer("task_logit_bias", torch.zeros(2))
        self.use_task_logit_bias = False
        self.numtask = 2

    def forward(self, inputs, fc_only=False):
        return torch.cat([head(inputs) for head in self.classifier_pool], dim=1)


class TaskBiasCalibrationTests(unittest.TestCase):
    def test_fitted_bias_corrects_task_routing_without_changing_within_task_classes(self):
        logits = torch.tensor([
            [0.80, 0.10, 0.90, 0.00],
            [0.10, 0.80, 0.00, 0.90],
            [0.20, 0.10, 0.80, 0.00],
            [0.10, 0.20, 0.00, 0.80],
        ])
        targets = torch.tensor([0, 1, 2, 3])
        fit = getattr(dlora, "fit_task_logit_bias", lambda *_: torch.zeros(2))

        bias = fit(logits, targets, [2, 2], 20.0)
        class_bias = bias.repeat_interleave(torch.tensor([2, 2]))
        calibrated = logits + class_bias

        self.assertEqual(calibrated.argmax(1).tolist(), targets.tolist())
        self.assertEqual(logits[:, :2].argmax(1).tolist(), calibrated[:, :2].argmax(1).tolist())
        self.assertEqual(logits[:, 2:].argmax(1).tolist(), calibrated[:, 2:].argmax(1).tolist())
        self.assertAlmostEqual(float(bias.mean()), 0.0, places=6)

    def test_global_interface_adds_one_shared_bias_per_task_only_when_enabled(self):
        network = dlora.MANet.__new__(dlora.MANet)
        nn.Module.__init__(network)
        network.image_encoder = TokenEncoder()
        network.classifier_pool = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            network.classifier_pool[0].weight.copy_(torch.eye(4)[:2])
            network.classifier_pool[1].weight.copy_(torch.eye(4)[2:])
        network.numtask = 2
        network.register_buffer("task_logit_bias", torch.tensor([0.25, -0.10]))
        network.use_task_logit_bias = False
        inputs = torch.eye(4)

        raw = network.interface(inputs)
        expected = torch.cat([
            F.linear(F.normalize(inputs), F.normalize(head.weight))
            for head in network.classifier_pool
        ], dim=1)
        self.assertTrue(torch.equal(raw, expected))
        network.use_task_logit_bias = True
        calibrated = network.interface(inputs)

        self.assertTrue(torch.allclose(calibrated[:, :2] - raw[:, :2], torch.full((4, 2), 0.25)))
        self.assertTrue(torch.allclose(calibrated[:, 2:] - raw[:, 2:], torch.full((4, 2), -0.10)))
        self.assertEqual(raw[:, :2].argmax(1).tolist(), calibrated[:, :2].argmax(1).tolist())
        self.assertEqual(raw[:, 2:].argmax(1).tolist(), calibrated[:, 2:].argmax(1).tolist())

    def test_calibration_stage_changes_only_bias_and_restores_rng(self):
        learner = dlora.Learner.__new__(dlora.Learner)
        learner.args = {"seed": 1993, "scale": 20.0}
        learner._network = TinyBiasClassifier()
        learner._device = torch.device("cpu")
        learner._multiple_gpus = [learner._device]
        learner._cur_task = 1
        learner._total_classes = 4
        learner.task_sizes = [2, 2]
        learner._class_means = torch.eye(4)
        learner._class_covs = torch.eye(4).repeat(4, 1, 1) * 1e-6
        parameters = {name: value.detach().clone() for name, value in learner._network.named_parameters()}
        torch.manual_seed(7)
        expected_rng = torch.get_rng_state().clone()

        stage = getattr(learner, "_stage3_task_bias_calibration", lambda *_, **__: None)
        stage(task_size=2)

        self.assertFalse(torch.equal(learner._network.task_logit_bias, torch.zeros(2)))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in learner._network.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]), name)
            self.assertIsNone(value.grad, name)


if __name__ == "__main__":
    unittest.main()
