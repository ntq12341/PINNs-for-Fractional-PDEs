"""Trainable PDE parameters with the transforms used in Section 4.2."""

import math

import torch
import torch.nn as nn


def atanh(x):
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


class InverseParameters(nn.Module):
    def __init__(
        self,
        case,
        init_alpha=1.4,
        init_gamma=0.6,
        init_c=0.8,
        init_velocity=0.2,
        fixed_alpha=2.0,
        fixed_gamma=1.0,
    ):
        super().__init__()
        self.case = case
        self.fixed_alpha = fixed_alpha
        self.fixed_gamma = fixed_gamma

        self.alpha0 = nn.Parameter(torch.tensor(atanh(2.0 * (init_alpha - 1.5)), dtype=torch.float64))
        self.gamma0 = nn.Parameter(torch.tensor(atanh(2.0 * (init_gamma - 0.5)), dtype=torch.float64))
        self.c0 = nn.Parameter(torch.tensor(math.log(init_c), dtype=torch.float64))
        self.v0 = nn.Parameter(torch.tensor(math.log(init_velocity), dtype=torch.float64))

    def values(self):
        device = self.c0.device
        if self.case == "tf":
            alpha = torch.tensor(self.fixed_alpha, dtype=torch.float64, device=device)
        else:
            alpha = 0.5 * torch.tanh(self.alpha0) + 1.5

        if self.case == "sf":
            gamma = torch.tensor(self.fixed_gamma, dtype=torch.float64, device=device)
        else:
            gamma = 0.5 * torch.tanh(self.gamma0) + 0.5

        c = torch.exp(self.c0)
        velocity = torch.exp(self.v0)
        return alpha, gamma, c, velocity

    def as_floats(self):
        alpha, gamma, c, velocity = self.values()
        return {
            "alpha": float(alpha.detach().cpu()),
            "gamma": float(gamma.detach().cpu()),
            "c": float(c.detach().cpu()),
            "velocity": float(velocity.detach().cpu()),
        }
