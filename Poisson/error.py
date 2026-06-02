"""Error metrics for Poisson experiments."""

import numpy as np
import torch

from utils import exact_solution_nonsmooth, exact_solution_poisson


def relative_l2_error(u_pred, u_exact):
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def mse_error(u_pred, u_exact):
    return float(np.mean((u_pred - u_exact) ** 2))


def max_abs_error(u_pred, u_exact):
    return float(np.max(np.abs(u_pred - u_exact)))


def evaluate_relative_l2(model, device, num_points, alpha, case):
    was_training = model.training
    model.eval()
    x = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
    x_tensor = torch.tensor(x, dtype=next(model.parameters()).dtype, device=device)
    with torch.no_grad():
        pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
    if was_training:
        model.train()

    if case == "smooth":
        exact = exact_solution_poisson(x)
    elif case == "nonsmooth":
        exact = exact_solution_nonsmooth(x, alpha)
    else:
        raise ValueError(f"Unknown case: {case}")

    return relative_l2_error(pred, exact)
