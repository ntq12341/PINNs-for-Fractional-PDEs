"""Plot Fig. 4-7 style PNG files from CSV results.

Run main.py first to generate:

    results/fig4_summary.csv
    results/fig5_trace.csv
    results/fig6_summary.csv
    results/fig7_summary.csv

This file only handles plotting. It does not train networks or write CSV data.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from results_io import RESULTS_DIR, read_csv


def plot_convergence(summary, orders, out_path, title):
    orders = list(orders)
    fig, axes = plt.subplots(1, len(orders), figsize=(4.5 * len(orders), 4), sharey=True)
    if len(orders) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for panel_idx, (ax, order) in enumerate(zip(axes, orders), start=1):
        data = [row for row in summary if row["order"] == order]
        if not data:
            raise ValueError(f"No Fig. 4 data found for GL order {order}. Run main.py first.")

        configs = sorted(
            {
                (row["case"], row["width"], row["depth"], row["learning_rate"], row["iterations"])
                for row in data
            }
        )
        for idx, (case, width, depth, lr, iterations) in enumerate(configs):
            config_rows = sorted(
                [
                    row
                    for row in data
                    if row["case"] == case
                    and row["width"] == width
                    and row["depth"] == depth
                    and row["learning_rate"] == lr
                    and row["iterations"] == iterations
                ],
                key=lambda item: item["N"],
            )
            x = np.array([row["N"] for row in config_rows])
            y = np.array([row["mean_error"] for row in config_rows])
            yerr = np.array([row["std_error"] for row in config_rows])
            color = colors[idx % len(colors)]
            label = f"{case}, w={width}, d={depth}, lr={lr:g}, it={iterations:g}"
            ax.plot(x, y, marker="o", color=color, label=label)
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_trace(rows, out_path, title):
    if not rows:
        raise ValueError("No Fig. 5 data found. Run main.py first.")

    ns = sorted({row["N"] for row in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4), sharey=False)
    if len(ns) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for ax, n in zip(axes, ns):
        n_rows = [row for row in rows if row["N"] == n]
        configs = sorted({(row["case"], row["width"], row["depth"]) for row in n_rows})
        for idx, (case, width, depth) in enumerate(configs):
            data = sorted(
                [
                    row
                    for row in n_rows
                    if row["case"] == case and row["width"] == width and row["depth"] == depth
                ],
                key=lambda item: item["iteration"],
            )
            x = np.array([row["iteration"] for row in data])
            loss = np.array([row["loss"] for row in data])
            err = np.array([row["relative_l2"] for row in data])
            color = colors[idx % len(colors)]
            label = f"{case}, w={width}, d={depth}"
            ax.plot(x, loss, "-", color=color, label=f"loss: {label}")
            ax.plot(x, err, "--", color=color, label=f"L2: {label}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N = {n}")
        ax.set_xlabel("# iterations")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig6(rows, out_path):
    if not rows:
        raise ValueError("No Fig. 6 data found. Run main.py first.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = plt.cm.tab10.colors
    panels = [
        ("lambda_sweep", axes[0], "lambda", "lambda", "(a) fixed N, varying lambda"),
        ("N_sweep", axes[1], "N", "N", "(b) fixed lambda, varying N"),
    ]

    for mode, ax, x_key, xlabel, title in panels:
        data = [row for row in rows if row["mode"] == mode]
        if not data:
            raise ValueError(f"No Fig. 6 data found for mode={mode}.")
        configs = sorted(
            {
                (row["width"], row["depth"], row["learning_rate"], row["iterations"])
                for row in data
            }
        )
        for idx, (width, depth, lr, iterations) in enumerate(configs):
            config_rows = sorted(
                [
                    row
                    for row in data
                    if row["width"] == width
                    and row["depth"] == depth
                    and row["learning_rate"] == lr
                    and row["iterations"] == iterations
                ],
                key=lambda item: item[x_key],
            )
            x = np.array([row[x_key] for row in config_rows])
            y = np.array([row["mean_error"] for row in config_rows])
            yerr = np.array([row["std_error"] for row in config_rows])
            color = colors[idx % len(colors)]
            label = f"w={width}, d={depth}, lr={lr:g}, it={iterations:g}"
            ax.plot(x, y, marker="o", color=color, label=label)
            ax.fill_between(x, np.maximum(y - yerr, 1e-16), y + yerr, color=color, alpha=0.2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("L2 relative error")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig7(rows, out_path):
    if not rows:
        raise ValueError("No Fig. 7 data found. Run main.py first.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    grid_rows = [row for row in rows if row["mode"] == "grid"]
    if not grid_rows:
        raise ValueError("No Fig. 7 grid data found.")
    widths = sorted({row["width"] for row in grid_rows})
    depths = sorted({row["depth"] for row in grid_rows})
    heatmap = np.full((len(widths), len(depths)), np.nan)
    for row in grid_rows:
        i = widths.index(row["width"])
        j = depths.index(row["depth"])
        heatmap[i, j] = row["mean_error"]

    im = axes[0].imshow(heatmap, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(depths)), depths)
    axes[0].set_yticks(range(len(widths)), widths)
    axes[0].set_xlabel("Depth")
    axes[0].set_ylabel("Width")
    axes[0].set_title("(a) depth-width sweep")
    fig.colorbar(im, ax=axes[0], label="mean L2 relative error")

    line_specs = [
        ("narrow_depth_sweep", "depth", "Narrow NN: varying depth"),
        ("shallow_width_sweep", "width", "Shallow NN: varying width"),
    ]
    colors = plt.cm.tab10.colors
    for idx, (mode, x_key, label) in enumerate(line_specs):
        data = sorted([row for row in rows if row["mode"] == mode], key=lambda item: item[x_key])
        if not data:
            continue
        x = np.array([row[x_key] for row in data])
        y = np.array([row["mean_error"] for row in data])
        yerr = np.array([row["std_error"] for row in data])
        color = colors[idx % len(colors)]
        axes[1].plot(x, y, marker="o", color=color, label=label)
        axes[1].fill_between(x, np.maximum(y - yerr, 1e-16), y + yerr, color=color, alpha=0.2)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Depth or width")
    axes[1].set_ylabel("L2 relative error")
    axes[1].set_title("(b) extreme architecture sweeps")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Plot Poisson fPINN figures from CSV results.")
    parser.add_argument("--orders", type=int, nargs="+", default=None)
    parser.add_argument("--only", choices=["fig4", "fig5", "fig6", "fig7", "smooth", "nonsmooth", "both"], default="both")
    return parser


def maybe_plot_convergence(csv_name, png_name, orders, title):
    csv_path = RESULTS_DIR / csv_name
    if not csv_path.exists():
        print(f"Skip {png_name}: missing {csv_path}")
        return
    summary = read_csv(csv_path)
    plot_orders = orders if orders is not None else sorted({row["order"] for row in summary})
    plot_convergence(summary, plot_orders, RESULTS_DIR / png_name, title)


def maybe_plot_trace(csv_name, png_name, title):
    csv_path = RESULTS_DIR / csv_name
    if not csv_path.exists():
        print(f"Skip {png_name}: missing {csv_path}")
        return
    trace = read_csv(csv_path)
    plot_trace(trace, RESULTS_DIR / png_name, title)


def main():
    args = build_parser().parse_args()
    requested = set()

    if args.only == "both":
        requested.update(["fig4", "fig5", "fig6", "fig7"])
    elif args.only == "smooth":
        requested.update(["fig4", "fig5"])
    elif args.only == "nonsmooth":
        requested.update(["fig6", "fig7"])
    else:
        requested.add(args.only)

    if "fig4" in requested:
        maybe_plot_convergence(
            "fig4_summary.csv",
            "fig4_convergence.png",
            args.orders,
            "Fig. 4 style: smooth fPINN convergence without FDM curves",
        )
    if "fig5" in requested:
        maybe_plot_trace(
            "fig5_trace.csv",
            "fig5_loss_error.png",
            "Fig. 5 style: smooth loss and relative error without FDM line",
        )
    if "fig6" in requested:
        csv_path = RESULTS_DIR / "fig6_summary.csv"
        if not csv_path.exists():
            print(f"Skip fig6_convergence.png: missing {csv_path}")
        else:
            plot_fig6(read_csv(csv_path), RESULTS_DIR / "fig6_convergence.png")
    if "fig7" in requested:
        csv_path = RESULTS_DIR / "fig7_summary.csv"
        if not csv_path.exists():
            print(f"Skip fig7_architecture.png: missing {csv_path}")
        else:
            plot_fig7(read_csv(csv_path), RESULTS_DIR / "fig7_architecture.png")

    print(f"Saved PNG figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
