"""Utilities for time encoding."""

from numpy import linspace as np_linspace
from torch import cos as torch_cos
from torch import device as torch_device
from torch import from_numpy as torch_from_numpy
from torch import nn
from torch import zeros as torch_zeros


# time encoding by GAT
class TimeEncoder(nn.Module):
    def __init__(self, out_dimension: int, device: torch_device):
        super().__init__()
        self.device = device
        self.out_dimension = out_dimension
        self.w = nn.Linear(1, out_dimension).to(self.device)

        self.w.weight = nn.Parameter(
            (torch_from_numpy(1 / 10 ** np_linspace(0, 9, out_dimension)))
            .float()
            .reshape(out_dimension, -1)
        )
        self.w.bias = nn.Parameter(
            torch_zeros(out_dimension, device=self.device).float()
        )

    def forward(self, t):
        output = torch_cos(self.w(t))
        return output
