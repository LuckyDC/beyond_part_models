import torch

from torch import nn
import torch.nn.functional as F


class RWLayer(nn.Module):
    def __init__(self, temperature=1.0):
        super(RWLayer, self).__init__()

        self.temperature = temperature

    def forward(self, x):
        normed_x = F.normalize(x, p=2, dim=1)
        sim = torch.mm(normed_x, normed_x.transpose(0, 1))

        sim = torch.exp(sim / self.temperature)
        sim = sim / sim.sum(dim=1, keepdim=True)

        return torch.mm(sim, x)
