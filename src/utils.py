import numpy as np
from torch import nn

from torch.nn import functional as F


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


class OneHot(nn.Module):
    def __init__(self, d_Embedding):
        super().__init__()
        self.d_Embedding = d_Embedding

    def forward(self, x):
        return F.one_hot(x, self.d_Embedding).float()*5