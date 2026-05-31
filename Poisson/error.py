# error.py
import numpy as np


def relative_l2_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """L2 relative error: ||u_pred - u_exact|| / ||u_exact||"""
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def mse_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    return float(np.mean((u_pred - u_exact) ** 2))


def max_abs_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    return float(np.max(np.abs(u_pred - u_exact)))