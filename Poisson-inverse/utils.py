"""Utilities for 1D inverse fractional ADE experiments."""

import numpy as np
import torch


DOMAIN_X = (-1.0, 1.0)
DOMAIN_T = (0.0, 1.0)


TRUE_PARAMS = {
    "alpha": 1.5,
    "gamma": 0.5,
    "c": 1.0,
    "velocity": 0.1,
}


def exact_solution_np(x, t, alpha=1.5):
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return x * np.maximum(1.0 - x**2, 0.0) ** (1.0 + alpha / 2.0) * np.exp(-t)


def exact_solution_torch(x, t, alpha):
    return x * torch.clamp(1.0 - x**2, min=0.0) ** (1.0 + alpha / 2.0) * torch.exp(-t)


def initial_condition_torch(x, alpha):
    return exact_solution_torch(x, torch.zeros_like(x), alpha)


def relative_l2(pred, exact):
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def sobol_points(n_points, seed=0):
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
        unit = sampler.random(n_points)
    except Exception:
        rng = np.random.default_rng(seed)
        unit = rng.random((n_points, 2))
    x = DOMAIN_X[0] + (DOMAIN_X[1] - DOMAIN_X[0]) * unit[:, 0]
    t = 1.0e-6 + (DOMAIN_T[1] - 1.0e-6) * unit[:, 1]
    return x, t


def final_points(n_points, seed=0):
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=1, seed=seed)
        unit = sampler.random(n_points).reshape(-1)
    except Exception:
        rng = np.random.default_rng(seed)
        unit = rng.random(n_points)
    x = DOMAIN_X[0] + (DOMAIN_X[1] - DOMAIN_X[0]) * unit
    t = np.ones_like(x) * DOMAIN_T[1]
    return x, t


def test_grid(nx=80, nt=80):
    xs = np.linspace(DOMAIN_X[0], DOMAIN_X[1], nx + 1)[1:-1]
    ts = np.linspace(DOMAIN_T[0], DOMAIN_T[1], nt + 1)[1:]
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return xx.reshape(-1), tt.reshape(-1)
