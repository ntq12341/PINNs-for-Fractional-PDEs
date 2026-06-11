"""Plot inverse parameter trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


TRUE_BY_CASE = {
    "tf": {"gamma": 0.5, "c": 1.0, "velocity": 0.1},
    "sf": {"alpha": 1.5, "c": 1.0, "velocity": 0.1},
    "stf": {"alpha": 1.5, "gamma": 0.5, "c": 1.0, "velocity": 0.1},
}


def read_csv(path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["iteration"] = int(row["iteration"])
        for key in ("loss", "alpha", "gamma", "c", "velocity"):
            row[key] = float(row[key])
    return rows


def plot_trajectories(rows, out_path):
    cases = [case for case in ("tf", "sf", "stf") if any(row["case"] == case for row in rows)]
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4), sharey=False)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        case_rows = [row for row in rows if row["case"] == case]
        seeds = sorted({row["seed"] for row in case_rows})
        params = list(TRUE_BY_CASE[case].keys())
        for seed in seeds:
            seed_rows = sorted([row for row in case_rows if row["seed"] == seed], key=lambda item: item["iteration"])
            x = [row["iteration"] for row in seed_rows]
            for param in params:
                y = [row[param] for row in seed_rows]
                ax.plot(x, y, label=f"{param}, seed={seed}")
        for param, true_value in TRUE_BY_CASE[case].items():
            ax.axhline(true_value, linestyle="--", linewidth=1, color="k", alpha=0.4)
        ax.set_title(case.upper())
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter value")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Plot inverse parameter trajectories.")
    return parser


def main():
    build_parser().parse_args()
    path = RESULTS_DIR / "inverse_trajectory.csv"
    rows = read_csv(path)
    plot_trajectories(rows, RESULTS_DIR / "inverse_trajectory.png")
    print(f"Saved figure to {RESULTS_DIR / 'inverse_trajectory.png'}")


if __name__ == "__main__":
    main()
