"""Generate Fig. 4 and Fig. 5 style plots for Section 4.1.1.

The paper solves the 1D fractional Poisson equation

    (-Delta)^(alpha/2) u(x) = f(x),  x in (0, 1),  u(0)=u(1)=0,

with alpha=1.5 and the smooth fabricated solution u=x^3(1-x)^3. For this
example, lambda=N, so the training grid and GL auxiliary grid coincide.

This script intentionally omits the FDM curves requested by the user, but keeps
the fPINN mean/std and lowest-loss curves for Fig. 4 and the loss/error traces
for Fig. 5.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from error import relative_l2_error
from network import build_loss_poisson, fPINN
from utils import exact_solution_poisson


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, device, num_points):
    was_training = model.training
    model.eval()
    x = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
    x_tensor = torch.tensor(x, dtype=next(model.parameters()).dtype, device=device)
    with torch.no_grad():
        pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
    if was_training:
        model.train()
    return relative_l2_error(pred, exact_solution_poisson(x))


def train_once(n, alpha, order, lr, iterations, seed, layers, device, eval_points, log_points=None):
    set_seed(seed)
    model = fPINN(layers).to(device)
    loss_fn = build_loss_poisson(model, N=n, alpha=alpha, order=order, device=device, smooth=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trace = []
    log_set = set(log_points or [])

    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if it in log_set:
            current_loss = float(loss_fn().detach().cpu())
            current_error = evaluate_model(model, device, eval_points)
            trace.append(
                {
                    "iteration": it,
                    "loss": current_loss,
                    "relative_l2": current_error,
                }
            )

    final_loss = float(loss_fn().detach().cpu())
    final_error = evaluate_model(model, device, eval_points)
    return final_loss, final_error, trace


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


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_figure4(args, device):
    rows = []
    summary = []
    lr_specs = parse_lr_specs(args.learning_rates, args.iterations)

    for order in args.orders:
        for n in args.ns:
            for lr, iterations in lr_specs:
                errors = []
                losses = []
                for seed in args.seeds:
                    print(f"fig4 order={order} N={n} lr={lr:g} iterations={iterations} seed={seed}")
                    loss, error, _ = train_once(
                        n=n,
                        alpha=args.alpha,
                        order=order,
                        lr=lr,
                        iterations=iterations,
                        seed=seed,
                        layers=args.layers,
                        device=device,
                        eval_points=args.eval_points,
                    )
                    errors.append(error)
                    losses.append(loss)
                    rows.append(
                        {
                            "figure": "fig4",
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

    write_csv(
        RESULTS_DIR / "fig4_raw.csv",
        rows,
        ["figure", "order", "N", "dx", "learning_rate", "iterations", "seed", "loss", "relative_l2"],
    )
    write_csv(
        RESULTS_DIR / "fig4_summary.csv",
        summary,
        ["order", "N", "dx", "learning_rate", "iterations", "mean_error", "std_error", "best_loss_error", "best_loss"],
    )
    plot_figure4(summary, args.orders, RESULTS_DIR / "fig4_convergence.png")


def plot_figure4(summary, orders, out_path):
    orders = list(orders)
    fig, axes = plt.subplots(1, len(orders), figsize=(4.5 * len(orders), 4), sharey=True)
    if len(orders) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for panel_idx, (ax, order) in enumerate(zip(axes, orders), start=1):
        data = [row for row in summary if row["order"] == order]
        configs = sorted({(row["learning_rate"], row["iterations"]) for row in data})

        for idx, (lr, iterations) in enumerate(configs):
            config_rows = sorted(
                [row for row in data if row["learning_rate"] == lr and row["iterations"] == iterations],
                key=lambda item: item["N"],
            )
            x = np.array([row["N"] for row in config_rows])
            y = np.array([row["mean_error"] for row in config_rows])
            yerr = np.array([row["std_error"] for row in config_rows])
            color = colors[idx % len(colors)]
            ax.plot(x, y, marker="o", color=color, label=f"lr={lr:g}, it={iterations:g}")
            ax.fill_between(x, np.maximum(y - yerr, 1e-16), y + yerr, color=color, alpha=0.2)

        best_rows = []
        for n in sorted({row["N"] for row in data}):
            candidates = [row for row in data if row["N"] == n]
            best_rows.append(min(candidates, key=lambda item: item["best_loss"]))
        ax.plot(
            [row["N"] for row in best_rows],
            [row["best_loss_error"] for row in best_rows],
            "k-",
            linewidth=2,
            label="lowest loss",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({chr(96 + panel_idx)}) GL order {order}")
        ax.set_xlabel("N (= lambda)")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("L2 relative error")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fig. 4 style: fPINN convergence without FDM curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_figure5(args, device):
    rows = []
    points = log_iterations(args.fig5_iterations)

    for n in args.fig5_ns:
        print(f"fig5 N={n} lr={args.fig5_lr:g} iterations={args.fig5_iterations} seed={args.fig5_seed}")
        _, _, trace = train_once(
            n=n,
            alpha=args.alpha,
            order=3,
            lr=args.fig5_lr,
            iterations=args.fig5_iterations,
            seed=args.fig5_seed,
            layers=args.layers,
            device=device,
            eval_points=args.eval_points,
            log_points=points,
        )
        for item in trace:
            rows.append({"N": n, **item})

    write_csv(RESULTS_DIR / "fig5_trace.csv", rows, ["N", "iteration", "loss", "relative_l2"])
    plot_figure5(rows, RESULTS_DIR / "fig5_loss_error.png")


def plot_figure5(rows, out_path):
    ns = sorted({row["N"] for row in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4), sharey=False)
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        data = sorted([row for row in rows if row["N"] == n], key=lambda item: item["iteration"])
        x = np.array([row["iteration"] for row in data])
        loss = np.array([row["loss"] for row in data])
        err = np.array([row["relative_l2"] for row in data])
        ax.plot(x, loss, "b-", label="MSE loss")
        ax.plot(x, err, "r--", label="L2 relative error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N = {n}")
        ax.set_xlabel("# iterations")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("value")
    fig.suptitle("Fig. 5 style: loss and relative error without FDM line")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Fig. 4/5 style plots for the 1D fractional Poisson fPINN.")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--ns", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--learning-rates", nargs="+", default=["1e-3", "1e-4"])
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--fig5-ns", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--fig5-lr", type=float, default=1e-6)
    parser.add_argument("--fig5-iterations", type=int, default=100000)
    parser.add_argument("--fig5-seed", type=int, default=0)
    parser.add_argument("--eval-points", type=int, default=1000)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", choices=["fig4", "fig5", "both"], default="both")
    return parser


def main():
    args = build_parser().parse_args()
    args.orders = sorted(set(args.orders))
    invalid_orders = [order for order in args.orders if order not in (1, 2, 3)]
    if invalid_orders:
        raise ValueError(f"GL order must be 1, 2, or 3. Invalid values: {invalid_orders}")

    torch.set_default_dtype(torch.float64)
    args.layers = [1] + [args.width] * args.depth + [1]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with (RESULTS_DIR / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.only in ("fig4", "both"):
        run_figure4(args, device)
    if args.only in ("fig5", "both"):
        run_figure5(args, device)

    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
