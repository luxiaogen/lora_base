import torch
from torch import nn


class AngularPenaltySMLoss(nn.Module):
    """Original CosFace expression; deliberately preserve its arithmetic order."""
    def __init__(self, s=20, m=0.1):
        super().__init__()
        self.s = 20.0 if not s else s
        self.m = 0.0 if not m else m

    def forward(self, wf, labels):
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([
            torch.cat((wf[i, :label], wf[i, label + 1:])).unsqueeze(0)
            for i, label in enumerate(labels)
        ], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        return -torch.mean(numerator - torch.log(denominator))
