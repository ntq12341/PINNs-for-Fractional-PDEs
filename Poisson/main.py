"""Run the Section 4.1.1 Poisson experiments and write CSV results.

This script performs the computational work for Fig. 4/Fig. 5 style results.
Plotting is handled separately by figures.py.
"""

import argparse
import random

import numpy as np
import torch

from error import evaluate_relative_l2
from network import build_loss_poisson, build_loss_poisson_lambda, fPINN
from results_io import (
    RESULTS_DIR,
    ensure_results_dir,
    save_convergence_results,
    save_fig5_trace,
    save_fig6_summary,
    save_fig7_summary,
    write_config,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_once(n, alpha, order, lr, iterations, seed, layers, device, eval_points, case, log_points=None):
    set_seed(seed)
    model = fPINN(layers).to(device)
    loss_fn = build_loss_poisson(model, N=n, alpha=alpha, order=order, device=device, smooth=(case == "smooth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trace = []
    log_set = set(log_points or [])

    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if it in log_set:
            trace.append(
                {
                    "iteration": it,
                    "loss": float(loss_fn().detach().cpu()),
                    "relative_l2": evaluate_relative_l2(model, device, eval_points, alpha, case),
                }
            )

    final_loss = float(loss_fn().detach().cpu())
    final_error = evaluate_relative_l2(model, device, eval_points, alpha, case)
    return final_loss, final_error, trace


def train_once_lambda(n, lambda_gl, alpha, lr, iterations, seed, layers, device, eval_points, case):
    set_seed(seed)
    model = fPINN(layers).to(device)
    loss_fn = build_loss_poisson_lambda(
        model,
        N=n,
        alpha=alpha,
        lambda_gl=lambda_gl,
        device=device,
        smooth=(case == "smooth"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = float(loss_fn().detach().cpu())
    final_error = evaluate_relative_l2(model, device, eval_points, alpha, case)
    return final_loss, final_error


def parse_lr_specs(specs, default_iterations):
    parsed = []
    for spec in specs:
        if ":" in spec:
            lr_text, iter_text = spec.split(":", 1)
            parsed.append((float(lr_text), int(iter_text)))
        else:
            parsed.append((float(spec), default_iterations))
    return parsed


def log_iterations(max_iterations):
    values = {1, max_iterations}
    for exp in range(0, int(np.ceil(np.log10(max_iterations))) + 1):
        base = 10**exp
        for mult in (1, 2, 4, 7):
            value = mult * base
            if 1 <= value <= max_iterations:
                values.add(value)
    return sorted(values)


def trace_paths(case):
    if case == "smooth":
        return RESULTS_DIR / "fig5_trace.csv", "fig5"
    raise ValueError(f"Unknown case: {case}")


def run_convergence_experiments(args, device, case):
    rows = []
    summary = []
    lr_specs = parse_lr_specs(args.learning_rates, args.iterations)
    figure_name = "fig4" if case == "smooth" else "fig6"

    for width in args.widths:
        for depth in args.depths:
            layers = [1] + [width] * depth + [1]
            for order in args.orders:
                for n in args.ns:
                    for lr, iterations in lr_specs:
                        errors = []
                        losses = []
                        for seed in args.seeds:
                            print(
                                f"{figure_name} case={case} width={width} depth={depth} "
                                f"order={order} N={n} lr={lr:g} iterations={iterations} seed={seed}"
                            )
                            loss, error, _ = train_once(
                                n=n,
                                alpha=args.alpha,
                                order=order,
                                lr=lr,
                                iterations=iterations,
                                seed=seed,
                                layers=layers,
                                device=device,
                                eval_points=args.eval_points,
                                case=case,
                            )
                            errors.append(error)
                            losses.append(loss)
                            rows.append(
                                {
                                    "figure": figure_name,
                                    "case": case,
                                    "width": width,
                                    "depth": depth,
                                    "order": order,
                                    "N": n,
                                    "dx": 1.0 / n,
                                    "learning_rate": lr,
                                    "iterations": iterations,
                                    "seed": seed,
                                    "loss": loss,
                                    "relative_l2": error,
                                }
                            )

                        best_idx = int(np.argmin(losses))
                        summary.append(
                            {
                                "case": case,
                                "width": width,
                                "depth": depth,
                                "order": order,
                                "N": n,
                                "dx": 1.0 / n,
                                "learning_rate": lr,
                                "iterations": iterations,
                                "mean_error": float(np.mean(errors)),
                                "std_error": float(np.std(errors)),
                                "best_loss_error": float(errors[best_idx]),
                                "best_loss": float(losses[best_idx]),
                            }
                        )

    save_convergence_results(case, rows, summary)


def run_trace_experiments(args, device, case):
    rows = []
    points = log_iterations(args.fig5_iterations)
    trace_path, figure_name = trace_paths(case)

    for width in args.widths:
        for depth in args.depths:
            layers = [1] + [width] * depth + [1]
            for n in args.fig5_ns:
                print(
                    f"{figure_name} case={case} width={width} depth={depth} "
                    f"N={n} lr={args.fig5_lr:g} iterations={args.fig5_iterations} seed={args.fig5_seed}"
                )
                _, _, trace = train_once(
                    n=n,
                    alpha=args.alpha,
                    order=3,
                    lr=args.fig5_lr,
                    iterations=args.fig5_iterations,
                    seed=args.fig5_seed,
                    layers=layers,
                    device=device,
                    eval_points=args.eval_points,
                    case=case,
                    log_points=points,
                )
                for item in trace:
                    rows.append({"case": case, "width": width, "depth": depth, "N": n, **item})

    save_fig5_trace(rows)


def run_fig6_experiments(args, device):
    rows = []
    lr_specs = parse_lr_specs(args.learning_rates, args.iterations)
    case = "nonsmooth"

    for width in args.widths:
        for depth in args.depths:
            layers = [1] + [width] * depth + [1]
            for lr, iterations in lr_specs:
                for mode, n_values, lambda_values in (
                    ("lambda_sweep", [args.fig6_fixed_n], args.fig6_lambdas),
                    ("N_sweep", args.fig6_ns, [args.fig6_fixed_lambda]),
                ):
                    for n in n_values:
                        for lambda_gl in lambda_values:
                            errors = []
                            losses = []
                            for seed in args.seeds:
                                print(
                                    f"fig6 mode={mode} width={width} depth={depth} "
                                    f"N={n} lambda={lambda_gl:g} lr={lr:g} iterations={iterations} seed={seed}"
                                )
                                loss, error = train_once_lambda(
                                    n=n,
                                    lambda_gl=lambda_gl,
                                    alpha=args.alpha,
                                    lr=lr,
                                    iterations=iterations,
                                    seed=seed,
                                    layers=layers,
                                    device=device,
                                    eval_points=args.eval_points,
                                    case=case,
                                )
                                losses.append(loss)
                                errors.append(error)
                            best_idx = int(np.argmin(losses))
                            rows.append(
                                {
                                    "mode": mode,
                                    "case": case,
                                    "width": width,
                                    "depth": depth,
                                    "N": n,
                                    "lambda": lambda_gl,
                                    "learning_rate": lr,
                                    "iterations": iterations,
                                    "mean_error": float(np.mean(errors)),
                                    "std_error": float(np.std(errors)),
                                    "best_loss_error": float(errors[best_idx]),
                                    "best_loss": float(losses[best_idx]),
                                }
                            )

    save_fig6_summary(rows)


def run_fig7_experiments(args, device):
    rows = []
    lr_specs = parse_lr_specs(args.learning_rates, args.iterations)
    case = "nonsmooth"

    configs = []
    for width in args.fig7_widths:
        for depth in args.fig7_depths:
            configs.append(("grid", width, depth))
    for depth in args.fig7_line_depths:
        configs.append(("narrow_depth_sweep", args.fig7_narrow_width, depth))
    for width in args.fig7_line_widths:
        configs.append(("shallow_width_sweep", width, args.fig7_shallow_depth))

    seen = set()
    for mode, width, depth in configs:
        key = (mode, width, depth)
        if key in seen:
            continue
        seen.add(key)
        layers = [1] + [width] * depth + [1]
        for lr, iterations in lr_specs:
            errors = []
            losses = []
            for seed in args.seeds:
                print(
                    f"fig7 mode={mode} width={width} depth={depth} "
                    f"N={args.fig7_n} lambda={args.fig7_lambda:g} "
                    f"lr={lr:g} iterations={iterations} seed={seed}"
                )
                loss, error = train_once_lambda(
                    n=args.fig7_n,
                    lambda_gl=args.fig7_lambda,
                    alpha=args.alpha,
                    lr=lr,
                    iterations=iterations,
                    seed=seed,
                    layers=layers,
                    device=device,
                    eval_points=args.eval_points,
                    case=case,
                )
                losses.append(loss)
                errors.append(error)
            best_idx = int(np.argmin(losses))
            rows.append(
                {
                    "mode": mode,
                    "case": case,
                    "width": width,
                    "depth": depth,
                    "N": args.fig7_n,
                    "lambda": args.fig7_lambda,
                    "learning_rate": lr,
                    "iterations": iterations,
                    "mean_error": float(np.mean(errors)),
                    "std_error": float(np.std(errors)),
                    "best_loss_error": float(errors[best_idx]),
                    "best_loss": float(losses[best_idx]),
                }
            )

    save_fig7_summary(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Run fPINN Poisson experiments and save CSV files.")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--cases", nargs="+", choices=["smooth", "nonsmooth"], default=["smooth"])
    parser.add_argument("--ns", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--learning-rates", nargs="+", default=["1e-3", "1e-4"])
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--fig5-ns", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--fig5-lr", type=float, default=1e-6)
    parser.add_argument("--fig5-iterations", type=int, default=100000)
    parser.add_argument("--fig5-seed", type=int, default=0)
    parser.add_argument("--fig6-fixed-n", type=int, default=100)
    parser.add_argument("--fig6-lambdas", type=float, nargs="+", default=[20, 40, 80, 160])
    parser.add_argument("--fig6-fixed-lambda", type=float, default=100)
    parser.add_argument("--fig6-ns", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--fig7-n", type=int, default=100)
    parser.add_argument("--fig7-lambda", type=float, default=200)
    parser.add_argument("--fig7-widths", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--fig7-depths", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--fig7-narrow-width", type=int, default=10)
    parser.add_argument("--fig7-line-depths", type=int, nargs="+", default=[2, 4, 8, 16, 24, 32, 40])
    parser.add_argument("--fig7-shallow-depth", type=int, default=2)
    parser.add_argument("--fig7-line-widths", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--eval-points", type=int, default=1000)
    parser.add_argument("--widths", type=int, nargs="+", default=[20])
    parser.add_argument("--depths", type=int, nargs="+", default=[4], help="Numbers of hidden layers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--only",
        choices=["fig4", "fig5", "fig6", "fig7", "smooth", "nonsmooth", "both"],
        default="both",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.orders = sorted(set(args.orders))
    invalid_orders = [order for order in args.orders if order not in (1, 2, 3)]
    if invalid_orders:
        raise ValueError(f"GL order must be 1, 2, or 3. Invalid values: {invalid_orders}")

    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)

    ensure_results_dir()
    write_config(RESULTS_DIR / "config.json", vars(args))

    requested = set()
    if args.only == "both":
        if "smooth" in args.cases:
            requested.update(["fig4", "fig5"])
        if "nonsmooth" in args.cases:
            requested.update(["fig6", "fig7"])
    elif args.only == "smooth":
        requested.update(["fig4", "fig5"])
    elif args.only == "nonsmooth":
        requested.update(["fig6", "fig7"])
    else:
        requested.add(args.only)

    if "fig4" in requested:
        run_convergence_experiments(args, device, "smooth")
    if "fig5" in requested:
        run_trace_experiments(args, device, "smooth")
    if "fig6" in requested:
        run_fig6_experiments(args, device)
    if "fig7" in requested:
        run_fig7_experiments(args, device)

    print(f"Saved CSV results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
