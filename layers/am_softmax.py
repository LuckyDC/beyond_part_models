from __future__ import print_function
from __future__ import division

import torch

from torch import nn


class AMSoftmaxLoss(nn.Module):
    def __init__(self, scale, margin):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        y_onehot = torch.zeros_like(x, device=x.device)
        y_onehot.scatter_(1, y.data.view(-1, 1), self.margin)

        out = self.scale * (x - y_onehot)
        loss = self.loss(out, y)

        return loss
