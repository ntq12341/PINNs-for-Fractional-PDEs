"""Plot Fig. 8/9 style figures from Poisson-time CSV results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def read_csv(path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("lambda_t", "lambda_x", "alpha", "gamma", "c", "velocity", "loss", "relative_l2"):
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["point_type"], row["lambda_t"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (case, point_type, lambda_t), values in grouped.items():
        best = min(values, key=lambda item: item["loss"])
        errors = np.array([item["relative_l2"] for item in values], dtype=np.float64)
        summary.append(
            {
                "case": case,
                "point_type": point_type,
                "lambda_t": lambda_t,
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "best_error": best["relative_l2"],
                "best_loss": best["loss"],
            }
        )
    return summary


def plot_fig8(rows, out_path):
    summary = summarize(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    panels = [("fig8_space", axes[0], "(a) 1D space-fractional ADE"), ("fig8_time", axes[1], "(b) 1D time-fractional ADE")]
    colors = plt.cm.tab10.colors

    for case, ax, title in panels:
        data = [row for row in summary if row["case"] == case]
        if not data:
            continue
        point_types = sorted({row["point_type"] for row in data})
        for idx, point_type in enumerate(point_types):
            rows_pt = sorted([row for row in data if row["point_type"] == point_type], key=lambda item: item["lambda_t"])
            x = np.array([row["lambda_t"] for row in rows_pt])
            y = np.array([row["best_error"] for row in rows_pt])
            color = colors[idx % len(colors)]
            ax.plot(x, y, marker="o", color=color, label=point_type)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lambda_t = 1 / Delta t")
        ax.set_ylabel("L2 relative error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig9(rows, out_path):
    summary = summarize(rows)
    data = sorted([row for row in summary if row["case"] == "fig9"], key=lambda item: item["lambda_t"])
    if not data:
        raise ValueError("No fig9 rows found.")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    x = np.array([row["lambda_t"] for row in data])
    y = np.array([row["best_error"] for row in data])
    ax.plot(x, y, marker="o", label="fPINN, BB forcing")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda_t = 1 / Delta t")
    ax.set_ylabel("L2 relative error")
    ax.set_title("1D space-time-fractional ADE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Plot Poisson-time figures.")
    parser.add_argument("--only", choices=["fig8", "fig9", "both"], default="both")
    return parser


def main():
    args = build_parser().parse_args()
    if args.only in ("fig8", "both"):
        path = RESULTS_DIR / "fig8_results.csv"
        if path.exists():
            plot_fig8(read_csv(path), RESULTS_DIR / "fig8_accuracy.png")
        else:
            print(f"Skip Fig. 8: missing {path}")
    if args.only in ("fig9", "both"):
        path = RESULTS_DIR / "fig9_results.csv"
        if path.exists():
            plot_fig9(read_csv(path), RESULTS_DIR / "fig9_accuracy.png")
        else:
            print(f"Skip Fig. 9: missing {path}")
    print(f"Saved figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
