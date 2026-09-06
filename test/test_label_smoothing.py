import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.losses import AngularPenaltySMLoss  # noqa: E402


class LabelSmoothingTests(unittest.TestCase):
    def test_zero_smoothing_matches_legacy_cosface(self):
        cosine_logits = torch.tensor(
            [[0.8, 0.2, -0.1], [0.1, 0.7, 0.3]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 1])
        scale = 20.0
        margin = 0.1

        numerator = scale * (
            torch.diagonal(cosine_logits.transpose(0, 1)[targets]) - margin
        )
        excluded = torch.cat(
            [
                torch.cat((cosine_logits[i, :y], cosine_logits[i, y + 1 :]))
                .unsqueeze(0)
                for i, y in enumerate(targets)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(
            torch.exp(scale * excluded),
            dim=1,
        )
        expected = -(numerator - torch.log(denominator)).mean()

        loss = AngularPenaltySMLoss(
            loss_type="cosface",
            s=scale,
            m=margin,
            label_smoothing=0.0,
        )(cosine_logits, targets)

        self.assertTrue(torch.equal(loss, expected))

    def test_smoothed_cosface_uses_margin_adjusted_logits(self):
        cosine_logits = torch.tensor(
            [[0.8, 0.2, -0.1], [0.1, 0.7, 0.3]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 1])
        scale = 20.0
        margin = 0.1
        epsilon = 0.05
        adjusted_logits = scale * cosine_logits.detach().clone()
        adjusted_logits[torch.arange(targets.numel()), targets] = scale * (
            cosine_logits.detach()[torch.arange(targets.numel()), targets]
            - margin
        )
        expected = F.cross_entropy(
            adjusted_logits,
            targets,
            label_smoothing=epsilon,
        )

        loss = AngularPenaltySMLoss(
            loss_type="cosface",
            s=scale,
            m=margin,
            label_smoothing=epsilon,
        )(cosine_logits, targets)
        loss.backward()

        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertTrue(torch.isfinite(cosine_logits.grad).all())

if __name__ == "__main__":
    unittest.main()