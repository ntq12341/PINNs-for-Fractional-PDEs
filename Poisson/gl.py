# gl.py
"""
Xây dựng ma trận GL (Grünwald–Letnikov) cho toán tử (-Δ)^{α/2} trên [0,1].

Công thức (Appendix A, bài báo):
    (-Δ)^{α/2} u(x_j) ≈ δ^α_{Δx,p} u(x_j)
                       = (Δx)^{-α} Σ_{k=0}^{...} (-1)^k C(α,k) u(x_j - (k-p)Δx)
                       + (Δx)^{-α} Σ_{k=0}^{...} (-1)^k C(α,k) u(x_j + (k-p)Δx)

Ba bậc hội tụ (Appendix A.2):
    bậc 1:  δ^α_{Δx,1}
    bậc 2:  (1-β) δ^α_{Δx,1} + β δ^α_{Δx,0}        với β = 1 - α/2
    bậc 3:  w1·δ^α_{Δx,1} + w0·δ^α_{Δx,0} + w_{-1}·δ^α_{Δx,-1}

Ma trận kích thước (N-1) × (N+1):
    - Hàng i  ↔  điểm lưới bên trong x_j = j/N, j = 1..N-1
    - Cột k   ↔  giá trị u tại x_k = k/N,  k = 0..N
"""
import functools
from typing import Tuple

import numpy as np
from scipy.special import binom


# ---------------------------------------------------------------------------
# Hàm cốt lõi: shifted GL matrix cho một giá trị p
# ---------------------------------------------------------------------------

def _shifted_gl_matrix(N: int, alpha: float, p: int) -> np.ndarray:
    """
    Xây dựng ma trận rời rạc cho δ^α_{Δx,p}.

    Tham số
    -------
    N     : số khoảng → lưới x_0..x_N, N-1 điểm bên trong
    alpha : bậc phân số (1 < alpha < 2)
    p     : độ dịch (p = 0, 1, hoặc -1)

    Trả về
    ------
    mat : np.ndarray shape (N-1, N+1), dtype float64
    """
    dx = 1.0 / N
    # Hệ số tổng quát: 1/(2 cos(πα/2) · Δx^α)
    scale = 1.0 / (2.0 * np.cos(np.pi * alpha / 2.0) * dx**alpha)

    # Tính trước tất cả hệ số binomial (-1)^k C(α,k)
    # Số hạng khác không giảm nhanh, cắt tại N+2 là đủ an toàn
    max_k = N + 2
    gl_coeff = np.array([(-1)**k * binom(alpha, k) for k in range(max_k)], dtype=np.float64)

    mat = np.zeros((N - 1, N + 1), dtype=np.float64)

    for row, j in enumerate(range(1, N)):           # j = 1..N-1
        for k in range(max_k):
            c = gl_coeff[k] * scale

            # Đóng góp phía trái: u(x_j - (k-p)·Δx) = u(x_{j-(k-p)})
            left_col = j - (k - p)
            if 0 <= left_col <= N:
                mat[row, left_col] += c

            # Đóng góp phía phải: u(x_j + (k-p)·Δx) = u(x_{j+(k-p)})
            right_col = j + (k - p)
            if 0 <= right_col <= N:
                mat[row, right_col] += c

    return mat


# ---------------------------------------------------------------------------
# Ma trận GL theo bậc hội tụ (1, 2, 3)
# ---------------------------------------------------------------------------

def build_gl_matrix(N: int, alpha: float, order: int) -> np.ndarray:
    """
    Trả về ma trận GL (N-1, N+1) xấp xỉ (-Δ)^{α/2} với bậc hội tụ `order`.

    order = 1 → O(Δx)
    order = 2 → O(Δx^2)
    order = 3 → O(Δx^3)
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order phải là 1, 2 hoặc 3, nhận được {order}")

    beta = 1.0 - alpha / 2.0          # β = 1 - α/2

    if order == 1:
        return _shifted_gl_matrix(N, alpha, p=1)

    if order == 2:
        # (-Δ)^{α/2} ≈ (1-β) δ_{p=1} + β δ_{p=0}
        return (
            (1.0 - beta) * _shifted_gl_matrix(N, alpha, p=1)
            + beta        * _shifted_gl_matrix(N, alpha, p=0)
        )

    # order == 3
    # Trọng số từ Appendix A.2:
    #   w1  = (11 - 6β)(1 - β) / 12
    #   w0  = (-6β² + 11β + 1) / 6
    #   w-1 = (6β + 1)(β - 1)  / 12
    w1  = ((11.0 - 6.0 * beta) * (1.0 - beta)) / 12.0
    w0  = (-6.0 * beta**2 + 11.0 * beta + 1.0) / 6.0
    wm1 = ((6.0 * beta + 1.0)  * (beta - 1.0)) / 12.0
    return (
        w1  * _shifted_gl_matrix(N, alpha, p=1)
        + w0  * _shifted_gl_matrix(N, alpha, p=0)
        + wm1 * _shifted_gl_matrix(N, alpha, p=-1)
    )


# ---------------------------------------------------------------------------
# Cache: tránh tính lại cùng (N, alpha, order) nhiều lần
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def get_gl_matrix(N: int, alpha: float, order: int) -> np.ndarray:
    """
    Phiên bản cache của build_gl_matrix.
    Gọi nhiều lần với cùng (N, alpha, order) chỉ tính một lần.
    """
    mat = build_gl_matrix(N, alpha, order)
    mat.flags.writeable = False        # đóng băng để cache an toàn
    return mat