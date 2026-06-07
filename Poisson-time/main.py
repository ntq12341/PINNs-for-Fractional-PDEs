"""Run time-dependent 1D fractional ADE fPINN experiments for Fig. 8/9 style plots."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from network import ExactSolution, TimeFPINN
from operators import build_residual_loss, precompute_forcing
from utils import exact_solution_np, lattice_points, relative_l2, scattered_points, test_grid


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


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


def evaluate(model, alpha, device, nx=80, nt=80):
    model.eval()
    x, t = test_grid(nx, nt)
    xt = torch.tensor(np.column_stack([x, t]), dtype=next(model.parameters()).dtype, device=device)
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().reshape(-1)
    exact = exact_solution_np(x, t, alpha).reshape(-1)
    return relative_l2(pred, exact)


def problem_params(case):
    if case == "fig8_space":
        return {"alpha": 1.5, "gamma": 1.0, "c": 1.0, "velocity": 0.0, "forcing": "WB"}
    if case == "fig8_time":
        return {"alpha": 2.0, "gamma": 0.5, "c": 1.0, "velocity": 0.0, "forcing": "WB"}
    if case == "fig9":
        return {"alpha": 1.5, "gamma": 0.5, "c": 1.0, "velocity": 0.1, "forcing": "BB"}
    raise ValueError(f"Unknown case: {case}")


def lambda_x_for_case(case, lambda_t, gamma):
    if case == "fig8_time":
        return lambda_t ** ((2.0 - gamma) / 2.0)
    return lambda_t


def training_points(case, lambda_t, n_scattered, seed, lattice):
    if lattice:
        nx = max(4, int(2 * lambda_t))
        nt = max(2, int(lambda_t))
        return lattice_points(nx, nt), f"lattice({nx}x{nt})"
    x, t = scattered_points(n_scattered, seed)
    return (x, t), f"scattered({n_scattered})"


def run_one(case, lambda_t, n_scattered, seed, layers, iterations, lr, device, lattice):
    params = problem_params(case)
    alpha = params["alpha"]
    gamma = params["gamma"]
    lambda_x = lambda_x_for_case(case, lambda_t, gamma)

    set_seed(seed)
    (xs, ts), point_type = training_points(case, lambda_t, n_scattered, seed, lattice)

    exact_model = ExactSolution(alpha)
    forcing = precompute_forcing(
        exact_model,
        xs,
        ts,
        alpha=alpha,
        gamma=gamma,
        c=params["c"],
        velocity=params["velocity"],
        lambda_x=lambda_x,
        lambda_t=lambda_t,
    )

    model = TimeFPINN(layers, alpha=alpha).to(device)
    loss_fn = build_residual_loss(
        model,
        xs,
        ts,
        forcing,
        alpha=alpha,
        gamma=gamma,
        c=params["c"],
        velocity=params["velocity"],
        lambda_x=lambda_x,
        lambda_t=lambda_t,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = float(loss_fn().detach().cpu())
    error = evaluate(model, alpha, device)
    return {
        "case": case,
        "point_type": point_type,
        "lambda_t": lambda_t,
        "lambda_x": lambda_x,
        "alpha": alpha,
        "gamma": gamma,
        "c": params["c"],
        "velocity": params["velocity"],
        "forcing": params["forcing"],
        "seed": seed,
        "loss": final_loss,
        "relative_l2": error,
    }


def run_cases(args, cases):
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    layers = [2] + [args.width] * args.depth + [1]
    rows = []

    for case in cases:
        for lambda_t in args.lambda_ts:
            for lattice in ([False, True] if case.startswith("fig8") else [False]):
                for seed in args.seeds:
                    print(
                        f"{case} lambda_t={lambda_t:g} lattice={lattice} "
                        f"seed={seed} iter={args.iterations} lr={args.lr:g}"
                    )
                    rows.append(
                        run_one(
                            case=case,
                            lambda_t=lambda_t,
                            n_scattered=args.n_scattered,
                            seed=seed,
                            layers=layers,
                            iterations=args.iterations,
                            lr=args.lr,
                            device=device,
                            lattice=lattice,
                        )
                    )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Run time-dependent fractional ADE fPINN experiments.")
    parser.add_argument("--only", choices=["fig8", "fig9", "both"], default="both")
    parser.add_argument("--lambda-ts", type=float, nargs="+", default=[10, 20, 40])
    parser.add_argument("--n-scattered", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    if args.only == "fig8":
        cases = ["fig8_space", "fig8_time"]
        out = RESULTS_DIR / "fig8_results.csv"
    elif args.only == "fig9":
        cases = ["fig9"]
        out = RESULTS_DIR / "fig9_results.csv"
    else:
        cases = ["fig8_space", "fig8_time", "fig9"]
        out = RESULTS_DIR / "time_results.csv"

    rows = run_cases(args, cases)
    fields = [
        "case",
        "point_type",
        "lambda_t",
        "lambda_x",
        "alpha",
        "gamma",
        "c",
        "velocity",
        "forcing",
        "seed",
        "loss",
        "relative_l2",
    ]
    write_csv(out, rows, fields)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
