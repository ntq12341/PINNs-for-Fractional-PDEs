"""Neural networks for 1D inverse fractional ADE experiments."""

from typing import List

import torch
import torch.nn as nn

from utils import exact_solution_torch, initial_condition_torch


class TimeFPINN(nn.Module):
    """Ansatz enforcing u(-1,t)=u(1,t)=0 and u(x,0)=u_exact(x,0)."""

    def __init__(self, layer_sizes: List[int], alpha_for_ic: float = 1.5):
        super().__init__()
        self.alpha_for_ic = alpha_for_ic
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
        return initial_condition_torch(x, self.alpha_for_ic) + t * rho * self.net(xt)


class ExactSolution(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        alpha = torch.as_tensor(self.alpha, dtype=xt.dtype, device=xt.device)
        return exact_solution_torch(x, t, alpha)
