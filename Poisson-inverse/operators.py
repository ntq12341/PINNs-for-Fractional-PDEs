"""Differentiable PDE operators for 1D inverse fractional ADEs."""

import numpy as np
import torch
from scipy.special import binom, gamma as gamma_fn


def module_dtype_device(module):
    params = list(module.parameters())
    if params:
        return params[0].dtype, params[0].device
    return torch.float64, torch.device("cpu")


def make_xt(x, t, dtype, device, requires_grad=False):
    xt = torch.tensor([[x, t]], dtype=dtype, device=device)
    xt.requires_grad_(requires_grad)
    return xt


def derivative(model, x, t, component, order=1):
    dtype, device = module_dtype_device(model)
    xt = make_xt(x, t, dtype, device, requires_grad=True)
    y = model(xt)
    grad = torch.autograd.grad(y, xt, torch.ones_like(y), create_graph=True)[0][:, component : component + 1]
    if order == 1:
        return grad
    grad2 = torch.autograd.grad(grad, xt, torch.ones_like(grad), create_graph=True)[0][:, component : component + 1]
    return grad2


def caputo_l1(model, x, t, gamma, lambda_t):
    if float(gamma.detach().cpu()) > 0.999:
        return derivative(model, x, t, component=1, order=1)

    dtype, device = module_dtype_device(model)
    m = max(1, int(np.ceil(lambda_t * t)))
    dt = torch.tensor(t / m, dtype=dtype, device=device)

    def u_at(k):
        return model(make_xt(x, float(k) * float(dt.detach().cpu()), dtype, device))

    value = torch.zeros(1, 1, dtype=dtype, device=device)
    for k in range(m + 1):
        if k == 0:
            coeff = -((torch.tensor(m, dtype=dtype, device=device)) ** (1.0 - gamma) - (torch.tensor(m - 1, dtype=dtype, device=device)) ** (1.0 - gamma))
        elif k == m:
            coeff = torch.ones((), dtype=dtype, device=device)
        else:
            a = torch.tensor(m - k + 1, dtype=dtype, device=device) ** (1.0 - gamma)
            b = 2.0 * torch.tensor(m - k, dtype=dtype, device=device) ** (1.0 - gamma)
            c = torch.tensor(m - k - 1, dtype=dtype, device=device) ** (1.0 - gamma)
            coeff = a - b + c
        value = value + coeff * u_at(k)
    return value / (torch.exp(torch.lgamma(2.0 - gamma)) * dt**gamma)


def gl_weight(alpha, k, dtype, device):
    weight = torch.ones((), dtype=dtype, device=device)
    for j in range(1, k + 1):
        weight = weight * (1.0 - (alpha + 1.0) / float(j))
    return weight


def fractional_laplacian_gl(model, x, t, alpha, lambda_x):
    if float(alpha.detach().cpu()) > 1.999:
        return -derivative(model, x, t, component=0, order=2)

    dtype, device = module_dtype_device(model)
    cos_scale = 1.0 / (2.0 * torch.cos(torch.pi * alpha / 2.0))
    value = torch.zeros(1, 1, dtype=dtype, device=device)

    for distance, sign in ((x + 1.0, -1.0), (1.0 - x, 1.0)):
        if distance <= 0.0:
            continue
        m = max(1, int(np.ceil(lambda_x * distance)))
        h_value = distance / m
        h = torch.tensor(h_value, dtype=dtype, device=device)
        for k in range(m + 1):
            xk = x + sign * (k - 1) * h_value
            if -1.0 <= xk <= 1.0:
                coeff = cos_scale * gl_weight(alpha, k, dtype, device) / (h**alpha)
                value = value + coeff * model(make_xt(xk, t, dtype, device))
    return value


def pde_operator(model, x, t, alpha, gamma, c, velocity, lambda_x, lambda_t):
    time_term = caputo_l1(model, x, t, gamma, lambda_t)
    space_term = fractional_laplacian_gl(model, x, t, alpha, lambda_x)
    adv_term = velocity * derivative(model, x, t, component=0, order=1)
    return time_term + c * space_term + adv_term


def pde_operator_fixed_exact(exact_model, x, t, alpha, gamma, c, velocity, lambda_x, lambda_t):
    dtype, device = module_dtype_device(exact_model)
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
    c_t = torch.tensor(c, dtype=dtype, device=device)
    v_t = torch.tensor(velocity, dtype=dtype, device=device)
    return pde_operator(exact_model, x, t, alpha_t, gamma_t, c_t, v_t, lambda_x, lambda_t)


def precompute_forcing(exact_model, xs, ts, true_params, lambda_x, lambda_t):
    values = []
    for x, t in zip(xs, ts):
        val = pde_operator_fixed_exact(
            exact_model,
            float(x),
            float(t),
            true_params["alpha"],
            true_params["gamma"],
            true_params["c"],
            true_params["velocity"],
            lambda_x,
            lambda_t,
        )
        values.append(float(val.detach().cpu().reshape(-1)[0]))
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def build_inverse_loss(
    model,
    inv_params,
    xi1_x,
    xi1_t,
    f_values,
    xi2_x,
    xi2_t,
    h_values,
    lambda_x,
    lambda_t,
    w1,
    w2,
):
    dtype, device = module_dtype_device(model)
    f_t = torch.tensor(f_values, dtype=dtype, device=device)
    h_t = torch.tensor(h_values, dtype=dtype, device=device)
    xi2 = torch.tensor(np.column_stack([xi2_x, xi2_t]), dtype=dtype, device=device)

    def loss_fn():
        alpha, gamma, c, velocity = inv_params.values()
        residuals = []
        for x, t in zip(xi1_x, xi1_t):
            residuals.append(pde_operator(model, float(x), float(t), alpha, gamma, c, velocity, lambda_x, lambda_t))
        residual = torch.cat(residuals, dim=0) - f_t
        final_mismatch = model(xi2) - h_t
        return w1 * torch.mean(residual**2) + w2 * torch.mean(final_mismatch**2)

    return loss_fn
