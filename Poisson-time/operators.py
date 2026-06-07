"""Fractional operators for 1D time-dependent fPINNs."""

import numpy as np
import torch
from scipy.special import binom, gamma as gamma_fn


def make_xt(x, t, dtype, device, requires_grad=False):
    xt = torch.tensor([[x, t]], dtype=dtype, device=device)
    xt.requires_grad_(requires_grad)
    return xt


def autograd_derivative(model, x, t, component, order=1):
    dtype = next(model.parameters()).dtype if len(list(model.parameters())) else torch.float64
    device = next(model.parameters()).device if len(list(model.parameters())) else torch.device("cpu")
    xt = make_xt(x, t, dtype, device, requires_grad=True)
    y = model(xt)
    grad = torch.autograd.grad(y, xt, torch.ones_like(y), create_graph=True)[0][:, component : component + 1]
    if order == 1:
        return grad
    grad2 = torch.autograd.grad(grad, xt, torch.ones_like(grad), create_graph=True)[0][:, component : component + 1]
    return grad2


def caputo_l1(model, x, t, gamma, lambda_t):
    if gamma == 1.0:
        return autograd_derivative(model, x, t, component=1, order=1)
    if t <= 0.0:
        raise ValueError("Caputo derivative is evaluated only for t > 0.")

    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float64
    device = params[0].device if params else torch.device("cpu")
    m = max(1, int(np.ceil(lambda_t * t)))
    dt = t / m
    coeffs = np.array([(ell + 1) ** (1.0 - gamma) - ell ** (1.0 - gamma) for ell in range(m)], dtype=np.float64)

    def u_at(k):
        return model(make_xt(x, k * dt, dtype, device))

    value = coeffs[0] * u_at(m) - coeffs[m - 1] * u_at(0)
    for k in range(1, m):
        value = value + (coeffs[m - k] - coeffs[m - k - 1]) * u_at(k)
    return value / (gamma_fn(2.0 - gamma) * dt**gamma)


def fractional_laplacian_gl(model, x, t, alpha, lambda_x):
    if alpha == 2.0:
        return -autograd_derivative(model, x, t, component=0, order=2)

    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float64
    device = params[0].device if params else torch.device("cpu")
    cos_scale = 1.0 / (2.0 * np.cos(np.pi * alpha / 2.0))
    value = torch.zeros(1, 1, dtype=dtype, device=device)

    for distance, sign in ((x + 1.0, -1.0), (1.0 - x, 1.0)):
        if distance <= 0.0:
            continue
        m = max(1, int(np.ceil(lambda_x * distance)))
        h = distance / m
        for k in range(m + 1):
            xk = x + sign * (k - 1) * h
            if -1.0 <= xk <= 1.0:
                coeff = cos_scale * ((-1) ** k) * binom(alpha, k) / (h**alpha)
                value = value + coeff * model(make_xt(xk, t, dtype, device))
    return value


def pde_operator(model, x, t, alpha, gamma, c, velocity, lambda_x, lambda_t):
    time_term = caputo_l1(model, x, t, gamma, lambda_t)
    space_term = fractional_laplacian_gl(model, x, t, alpha, lambda_x)
    adv_term = velocity * autograd_derivative(model, x, t, component=0, order=1)
    return time_term + c * space_term + adv_term


def precompute_forcing(exact_model, xs, ts, alpha, gamma, c, velocity, lambda_x, lambda_t):
    values = []
    for x, t in zip(xs, ts):
        f = pde_operator(exact_model, float(x), float(t), alpha, gamma, c, velocity, lambda_x, lambda_t)
        values.append(float(f.detach().cpu().reshape(-1)[0]))
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def build_residual_loss(model, xs, ts, forcing, alpha, gamma, c, velocity, lambda_x, lambda_t, device):
    dtype = next(model.parameters()).dtype
    forcing_t = torch.tensor(forcing, dtype=dtype, device=device)

    def loss_fn():
        residuals = []
        for x, t in zip(xs, ts):
            op = pde_operator(model, float(x), float(t), alpha, gamma, c, velocity, lambda_x, lambda_t)
            residuals.append(op)
        residual = torch.cat(residuals, dim=0) - forcing_t
        return torch.mean(residual**2)

    return loss_fn
