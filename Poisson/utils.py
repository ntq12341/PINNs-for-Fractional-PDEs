# utils.py
"""
Nghiệm chính xác và forcing term cho bài toán 1D fractional Poisson (Section 4.1.1).

Nghiệm mẫu 1 (trơn):  u(x) = x^3 (1-x)^3
Nghiệm mẫu 2 (ít trơn): u(x) = x (1 - x^2)^{alpha/2}

Forcing term f(x) theo công thức (4.3) trong bài báo.
"""
import numpy as np
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Nghiệm trơn: u(x) = x^3 (1-x)^3
# ---------------------------------------------------------------------------

def exact_solution_smooth(x: np.ndarray) -> np.ndarray:
    """u(x) = x^3 (1-x)^3  — dùng cho Figure 4."""
    x = np.asarray(x, dtype=np.float64).flatten()
    return x**3 * (1.0 - x)**3


def forcing_term_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Forcing term tương ứng với u(x) = x^3 (1-x)^3.
    Công thức (4.3) trong bài báo:

        f(x) = 1 / (2 cos(πα/2)) * [
            Γ(4)/Γ(4-α)  * (x^{3-α} + (1-x)^{3-α})
          - 3Γ(5)/Γ(5-α) * (x^{4-α} + (1-x)^{4-α})
          + 3Γ(6)/Γ(6-α) * (x^{5-α} + (1-x)^{5-α})
          - Γ(7)/Γ(7-α)  * (x^{6-α} + (1-x)^{6-α})
        ]
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    c = 1.0 / (2.0 * np.cos(np.pi * alpha / 2.0))
    t1 = (gamma(4) / gamma(4 - alpha)) * (x**(3 - alpha) + (1 - x)**(3 - alpha))
    t2 = 3 * (gamma(5) / gamma(5 - alpha)) * (x**(4 - alpha) + (1 - x)**(4 - alpha))
    t3 = 3 * (gamma(6) / gamma(6 - alpha)) * (x**(5 - alpha) + (1 - x)**(5 - alpha))
    t4 = (gamma(7) / gamma(7 - alpha)) * (x**(6 - alpha) + (1 - x)**(6 - alpha))
    return (c * (t1 - t2 + t3 - t4)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Nghiệm ít trơn: u(x) = x (1 - x^2)^{alpha/2}
# ---------------------------------------------------------------------------

def exact_solution_nonsmooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """u(x) = x (1 - x^2)^{alpha/2}  — dùng cho Figure 6."""
    x = np.asarray(x, dtype=np.float64).flatten()
    return x * np.maximum(1.0 - x**2, 0.0) ** (alpha / 2.0)


def forcing_term_nonsmooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Forcing term tương ứng với u(x) = x (1-x^2)^{alpha/2}.
    Theo bài báo: f(x) = Γ(α+2) x.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    return (gamma(alpha + 2.0) * x).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Alias mặc định (Figure 4 dùng nghiệm trơn)
# ---------------------------------------------------------------------------

def exact_solution_poisson(x: np.ndarray) -> np.ndarray:
    return exact_solution_smooth(x)


def forcing_term_poisson(x: np.ndarray, alpha: float) -> np.ndarray:
    return forcing_term_smooth(x, alpha)


# ---------------------------------------------------------------------------
# Tạo dữ liệu training
# ---------------------------------------------------------------------------

def generate_training_data(N: int, alpha: float, smooth: bool = True):
    """
    Trả về (x_interior, f_interior, dx).
    x_interior: (N-1, 1)  — các điểm lưới bên trong [1/N, ..., (N-1)/N]
    f_interior: (N-1, 1)  — forcing term tại x_interior
    dx: bước lưới = 1/N
    """
    dx = 1.0 / N
    x = np.linspace(dx, 1.0 - dx, N - 1).reshape(-1, 1)
    f = forcing_term_smooth(x, alpha) if smooth else forcing_term_nonsmooth(x, alpha)
    return x, f, dx