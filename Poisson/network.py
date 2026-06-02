"""Network and training utilities for the 1D fractional Poisson fPINN.

Section 4.1.1 uses the trial solution

    u_hat(x) = x * (1 - x) * NN(x; mu),

so the homogeneous boundary conditions are enforced exactly. The fractional
Laplacian in the residual is discretized by the shifted GL matrices in gl.py.
"""

from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.special import binom

from gl import get_gl_matrix
from utils import forcing_term_nonsmooth, forcing_term_smooth


class fPINN(nn.Module):
    """Feedforward tanh network with a boundary-condition factor rho(x)."""

    def __init__(
        self,
        layer_sizes: List[int],
        rho_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        modules: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.rho = rho_func if rho_func is not None else (lambda x: x * (1.0 - x))
        self._init_weights()

    def _init_weights(self):
        """Glorot/Xavier uniform initialization, suitable for tanh networks."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(x) * self.net(x)


def build_loss_poisson(
    model: fPINN,
    N: int,
    alpha: float,
    order: int,
    device: torch.device,
    smooth: bool = True,
) -> Callable[[], torch.Tensor]:
    """Build MSE(A_GL * u_hat(x_grid) - f(x_interior)).

    For Fig. 4 and Fig. 5 in Section 4.1.1, use smooth=True, alpha=1.5,
    and order in {1, 2, 3}. The paper sets lambda=N for this smooth example.
    """
    dtype = next(model.parameters()).dtype
    x_grid = torch.linspace(0.0, 1.0, N + 1, dtype=dtype, device=device).reshape(-1, 1)

    x_interior = np.linspace(1.0 / N, 1.0 - 1.0 / N, N - 1).reshape(-1, 1)
    forcing_fn = forcing_term_smooth if smooth else forcing_term_nonsmooth
    f_np = forcing_fn(x_interior, alpha)

    A_np = get_gl_matrix(N, alpha, order)
    A = torch.tensor(A_np.copy(), dtype=dtype, device=device)
    f = torch.tensor(f_np, dtype=dtype, device=device)

    def loss_fn() -> torch.Tensor:
        u_grid = model(x_grid)
        residual = A.mm(u_grid) - f
        return torch.mean(residual**2)

    return loss_fn


def build_loss_poisson_lambda(
    model: fPINN,
    N: int,
    alpha: float,
    lambda_gl: float,
    device: torch.device,
    smooth: bool = True,
) -> Callable[[], torch.Tensor]:
    """Build first-order shifted GL loss with auxiliary density lambda.

    This matches the setup used for the non-smooth fabricated solution in
    Section 4.1.1, where training points x_j=j/N and auxiliary GL points are
    controlled by lambda independently of N.
    """
    dtype = next(model.parameters()).dtype
    x_interior_np = np.linspace(1.0 / N, 1.0 - 1.0 / N, N - 1).reshape(-1, 1)
    forcing_fn = forcing_term_smooth if smooth else forcing_term_nonsmooth
    f = torch.tensor(forcing_fn(x_interior_np, alpha), dtype=dtype, device=device)
    cos_scale = 1.0 / (2.0 * np.cos(np.pi * alpha / 2.0))

    terms = []
    for xj in x_interior_np.reshape(-1):
        row_terms = []
        for distance, sign in ((xj, -1.0), (1.0 - xj, 1.0)):
            m = max(1, int(np.ceil(lambda_gl * distance)))
            h = distance / m
            for k in range(m + 1):
                point = xj + sign * (k - 1) * h
                if 0.0 <= point <= 1.0:
                    coeff = cos_scale * ((-1) ** k) * binom(alpha, k) / (h**alpha)
                    row_terms.append((point, coeff))
        terms.append(row_terms)

    def loss_fn() -> torch.Tensor:
        residuals = []
        for row_terms in terms:
            value = torch.zeros(1, 1, dtype=dtype, device=device)
            for point, coeff in row_terms:
                x = torch.tensor([[point]], dtype=dtype, device=device)
                value = value + coeff * model(x)
            residuals.append(value)
        residual = torch.cat(residuals, dim=0) - f
        return torch.mean(residual**2)

    return loss_fn


def train(
    model: fPINN,
    loss_fn: Callable[[], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    iterations: int,
    log_freq: int = 1000,
    log_iterations: Optional[set] = None,
) -> dict:
    """Train with Adam and return the final loss plus optional trace points."""
    model.train()
    trace = []
    log_set = set(log_iterations) if log_iterations else set()

    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if log_freq > 0 and it % log_freq == 0:
            print(f"  iter {it:7d}  loss = {loss.item():.3e}")

        if it in log_set:
            trace.append(
                {
                    "iteration": it,
                    "loss": float(loss_fn().detach().cpu()),
                }
            )

    return {"loss_final": float(loss_fn().detach().cpu()), "trace": trace}
