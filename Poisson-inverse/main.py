"""Run 1D inverse fractional ADE experiments inspired by Section 4.2."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from network import ExactSolution, TimeFPINN
from operators import build_inverse_loss, precompute_forcing
from parameters import InverseParameters
from utils import TRUE_PARAMS, exact_solution_np, final_points, relative_l2, sobol_points, test_grid


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


CASE_TRUE_PARAMS = {
    "tf": {"alpha": 2.0, "gamma": 0.5, "c": 1.0, "velocity": 0.1},
    "sf": {"alpha": 1.5, "gamma": 1.0, "c": 1.0, "velocity": 0.1},
    "stf": {"alpha": 1.5, "gamma": 0.5, "c": 1.0, "velocity": 0.1},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows, fieldnames):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_solution(model, true_alpha, device):
    x, t = test_grid()
    xt = torch.tensor(np.column_stack([x, t]), dtype=next(model.parameters()).dtype, device=device)
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().reshape(-1)
    exact = exact_solution_np(x, t, true_alpha).reshape(-1)
    return relative_l2(pred, exact)


def weights(f_values, h_values):
    w1 = 100.0 / max(float(np.mean(f_values**2)), 1.0e-12)
    w2 = 1.0 / max(float(np.mean(h_values**2)), 1.0e-12)
    return w1, w2


def run_one(case, seed, args, device):
    set_seed(seed)
    true_params = CASE_TRUE_PARAMS[case]
    layers = [2] + [args.width] * args.depth + [1]

    xi1_x, xi1_t = sobol_points(args.n_residual, seed=seed)
    xi2_x, xi2_t = final_points(args.n_final, seed=seed + 1000)

    exact_model = ExactSolution(alpha=true_params["alpha"])
    f_values = precompute_forcing(exact_model, xi1_x, xi1_t, true_params, args.lambda_x, args.lambda_t)
    h_values = exact_solution_np(xi2_x, xi2_t, true_params["alpha"]).reshape(-1, 1)
    w1, w2 = weights(f_values, h_values)

    model = TimeFPINN(layers, alpha_for_ic=true_params["alpha"]).to(device)
    inv_params = InverseParameters(
        case,
        init_alpha=args.init_alpha,
        init_gamma=args.init_gamma,
        init_c=args.init_c,
        init_velocity=args.init_velocity,
        fixed_alpha=true_params["alpha"],
        fixed_gamma=true_params["gamma"],
    ).to(device)

    loss_fn = build_inverse_loss(
        model,
        inv_params,
        xi1_x,
        xi1_t,
        f_values,
        xi2_x,
        xi2_t,
        h_values,
        args.lambda_x,
        args.lambda_t,
        w1,
        w2,
    )
    optimizer = torch.optim.Adam(list(model.parameters()) + list(inv_params.parameters()), lr=args.lr)

    trajectory = []
    log_set = set(log_iterations(args.iterations, args.log_points))
    for it in range(1, args.iterations + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        if it in log_set:
            vals = inv_params.as_floats()
            trajectory.append(
                {
                    "case": case,
                    "seed": seed,
                    "iteration": it,
                    "loss": float(loss_fn().detach().cpu()),
                    **vals,
                }
            )

    final_vals = inv_params.as_floats()
    final_loss = float(loss_fn().detach().cpu())
    err_u = evaluate_solution(model, true_params["alpha"], device)
    summary = {
        "case": case,
        "seed": seed,
        "loss": final_loss,
        "error_u": err_u,
        "alpha_true": true_params["alpha"],
        "gamma_true": true_params["gamma"],
        "c_true": true_params["c"],
        "velocity_true": true_params["velocity"],
        "alpha": final_vals["alpha"],
        "gamma": final_vals["gamma"],
        "c": final_vals["c"],
        "velocity": final_vals["velocity"],
    }
    return summary, trajectory


def log_iterations(iterations, n_points):
    values = {1, iterations}
    if n_points <= 2:
        return sorted(values)
    for value in np.geomspace(1, iterations, n_points):
        values.add(int(round(value)))
    return sorted(v for v in values if 1 <= v <= iterations)


def build_parser():
    parser = argparse.ArgumentParser(description="Run 1D inverse fPINN experiments.")
    parser.add_argument("--cases", nargs="+", choices=["tf", "sf", "stf"], default=["tf", "sf", "stf"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--log-points", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-x", type=float, default=20)
    parser.add_argument("--lambda-t", type=float, default=10)
    parser.add_argument("--n-residual", type=int, default=32)
    parser.add_argument("--n-final", type=int, default=8)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--init-alpha", type=float, default=1.4)
    parser.add_argument("--init-gamma", type=float, default=0.6)
    parser.add_argument("--init-c", type=float, default=0.8)
    parser.add_argument("--init-velocity", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)

    summaries = []
    trajectories = []
    for case in args.cases:
        for seed in args.seeds:
            print(f"inverse case={case} seed={seed} iterations={args.iterations} lr={args.lr:g}")
            summary, trajectory = run_one(case, seed, args, device)
            summaries.append(summary)
            trajectories.extend(trajectory)

    write_csv(
        RESULTS_DIR / "inverse_summary.csv",
        summaries,
        [
            "case",
            "seed",
            "loss",
            "error_u",
            "alpha_true",
            "gamma_true",
            "c_true",
            "velocity_true",
            "alpha",
            "gamma",
            "c",
            "velocity",
        ],
    )
    write_csv(
        RESULTS_DIR / "inverse_trajectory.csv",
        trajectories,
        ["case", "seed", "iteration", "loss", "alpha", "gamma", "c", "velocity"],
    )
    print(f"Saved inverse results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
