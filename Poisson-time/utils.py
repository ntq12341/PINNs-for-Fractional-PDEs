"""Manufactured solutions for the time-dependent 1D fractional ADE examples."""

import numpy as np
import torch


DOMAIN_X = (-1.0, 1.0)
DOMAIN_T = (0.0, 1.0)


def exact_solution_np(x, t, alpha):
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return x * np.maximum(1.0 - x**2, 0.0) ** (1.0 + alpha / 2.0) * np.exp(-t)


def exact_solution_torch(x, t, alpha):
    return x * torch.clamp(1.0 - x**2, min=0.0) ** (1.0 + alpha / 2.0) * torch.exp(-t)


def initial_condition_torch(x, alpha):
    t0 = torch.zeros_like(x)
    return exact_solution_torch(x, t0, alpha)


def relative_l2(u_pred, u_exact):
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def lattice_points(nx, nt):
    xs = np.linspace(DOMAIN_X[0], DOMAIN_X[1], nx + 1)[1:-1]
    ts = np.linspace(DOMAIN_T[0], DOMAIN_T[1], nt + 1)[1:]
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return xx.reshape(-1), tt.reshape(-1)


def scattered_points(n_points, seed=0):
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


def test_grid(nx=80, nt=80):
    xs = np.linspace(DOMAIN_X[0], DOMAIN_X[1], nx + 1)[1:-1]
    ts = np.linspace(DOMAIN_T[0], DOMAIN_T[1], nt + 1)[1:]
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return xx.reshape(-1), tt.reshape(-1)
