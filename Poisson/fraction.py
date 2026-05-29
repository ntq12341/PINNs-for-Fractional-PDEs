# fraction.py
import numpy as np
from scipy.special import comb, gamma

def gl_coefficients(alpha, k):
    """(-1)^k * C(alpha, k)"""
    return (-1)**k * comb(alpha, k)

def fractional_laplacian_1d(u_vals, alpha, dx):
    """
    Tính (-Δ)^{α/2} u tại các điểm lưới bên trong (1D) dùng GL bậc 1.
    u_vals: numpy array (N+1,) giá trị u tại các điểm lưới 0..N
    alpha: fractional order
    dx: bước lưới
    Trả về: numpy array (N-1,)
    """
    N = len(u_vals) - 1
    lap = np.zeros(N-1)
    cos_term = 1.0 / (2 * np.cos(np.pi * alpha / 2))
    for i in range(1, N):
        j = i
        sum_left = 0.0
        for k in range(0, j+1):
            idx = j - (k-1)
            if 0 <= idx <= N:
                sum_left += gl_coefficients(alpha, k) * u_vals[idx]
        sum_right = 0.0
        for k in range(0, N-j+1):
            idx = j + (k-1)
            if 0 <= idx <= N:
                sum_right += gl_coefficients(alpha, k) * u_vals[idx]
        lap[i-1] = cos_term * (sum_left + sum_right) / (dx**alpha)
    return lap

def caputo_l1(u_history, dt, gamma):
    """
    Xấp xỉ đạo hàm Caputo bậc gamma (0<gamma<1) bằng L1 scheme.
    u_history: list numpy array (các bước thời gian)
    dt: bước thời gian
    gamma: fractional order
    """
    m = len(u_history) - 1
    coeffs = np.array([(l+1)**(1-gamma) - l**(1-gamma) for l in range(m+1)])
    deriv = (1.0 / gamma(2-gamma) / (dt**gamma)) * (
        -coeffs[m-1]*u_history[0] + coeffs[0]*u_history[m] +
        sum((coeffs[m-k] - coeffs[m-k-1]) * u_history[k] for k in range(1, m))
    )
    return deriv