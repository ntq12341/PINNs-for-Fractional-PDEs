# error.py
import numpy as np

def relative_l2_error(u_pred, u_exact):
    """u_pred, u_exact: numpy arrays"""
    return np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

def mse_error(u_pred, u_exact):
    return np.mean((u_pred - u_exact)**2)

def max_abs_error(u_pred, u_exact):
    return np.max(np.abs(u_pred - u_exact))