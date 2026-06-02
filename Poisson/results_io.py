"""CSV and config helpers for Poisson experiment outputs."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

FIG4_RAW_FIELDS = [
    "figure",
    "case",
    "width",
    "depth",
    "order",
    "N",
    "dx",
    "learning_rate",
    "iterations",
    "seed",
    "loss",
    "relative_l2",
]
FIG4_SUMMARY_FIELDS = [
    "case",
    "width",
    "depth",
    "order",
    "N",
    "dx",
    "learning_rate",
    "iterations",
    "mean_error",
    "std_error",
    "best_loss_error",
    "best_loss",
]
FIG5_TRACE_FIELDS = ["case", "width", "depth", "N", "iteration", "loss", "relative_l2"]
FIG6_SUMMARY_FIELDS = [
    "mode",
    "case",
    "width",
    "depth",
    "N",
    "lambda",
    "learning_rate",
    "iterations",
    "mean_error",
    "std_error",
    "best_loss_error",
    "best_loss",
]
FIG7_SUMMARY_FIELDS = [
    "mode",
    "case",
    "width",
    "depth",
    "N",
    "lambda",
    "learning_rate",
    "iterations",
    "mean_error",
    "std_error",
    "best_loss_error",
    "best_loss",
]

FLOAT_FIELDS = {
    "dx",
    "learning_rate",
    "loss",
    "relative_l2",
    "mean_error",
    "std_error",
    "best_loss_error",
    "best_loss",
    "lambda",
}
INT_FIELDS = {"order", "N", "iterations", "seed", "iteration"}
INT_FIELDS.update({"width", "depth"})


def ensure_results_dir(results_dir=RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return [coerce_row(row) for row in rows]


def coerce_row(row):
    converted = {}
    for key, value in row.items():
        if key in INT_FIELDS:
            converted[key] = int(value)
        elif key in FLOAT_FIELDS:
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def write_config(path, config):
    with path.open("w") as f:
        json.dump(config, f, indent=2, default=str)


def save_convergence_results(case, raw_rows, summary_rows):
    if case == "smooth":
        raw_path = RESULTS_DIR / "fig4_raw.csv"
        summary_path = RESULTS_DIR / "fig4_summary.csv"
    elif case == "nonsmooth":
        raw_path = RESULTS_DIR / "fig6_raw.csv"
        summary_path = RESULTS_DIR / "fig6_summary.csv"
    else:
        raise ValueError(f"Unknown case: {case}")
    write_csv(raw_path, raw_rows, FIG4_RAW_FIELDS)
    write_csv(summary_path, summary_rows, FIG4_SUMMARY_FIELDS)


def save_fig5_trace(rows):
    write_csv(RESULTS_DIR / "fig5_trace.csv", rows, FIG5_TRACE_FIELDS)


def save_fig6_summary(rows):
    write_csv(RESULTS_DIR / "fig6_summary.csv", rows, FIG6_SUMMARY_FIELDS)


def save_fig7_summary(rows):
    write_csv(RESULTS_DIR / "fig7_summary.csv", rows, FIG7_SUMMARY_FIELDS)
