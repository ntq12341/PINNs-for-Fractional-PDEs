"""Neural network for 1D time-dependent fractional ADE examples."""

from typing import List

import torch
import torch.nn as nn

from utils import exact_solution_torch, initial_condition_torch


class TimeFPINN(nn.Module):
    """fPINN ansatz enforcing u(-1,t)=u(1,t)=0 and u(x,0)=u_exact(x,0)."""

    def __init__(self, layer_sizes: List[int], alpha: float):
        super().__init__()
        self.alpha = alpha
        modules = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        rho = 1.0 - x**2
        return initial_condition_torch(x, self.alpha) + t * rho * self.net(xt)


class ExactSolution(torch.nn.Module):
    """Torch module wrapper so exact solution can be passed to operators."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return exact_solution_torch(x, t, self.alpha)
