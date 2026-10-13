"""参考配方的CosFace目标；保持原exp/log计算顺序。"""
import torch
from torch import nn


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', s=20., m=0.):
        super().__init__()
        self.s, self.m = 20.0 if not s else s, 0.0 if not m else m

    def forward(self, wf, labels):
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0)
                          for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
