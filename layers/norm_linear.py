import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizeLinear(nn.Module):
    def __init__(self, in_features, num_class):
        super(NormalizeLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_class, in_features))

    def forward(self, x):
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w)
