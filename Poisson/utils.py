# utils.py
import numpy as np
from scipy.special import gamma

def exact_solution_poisson(x):
    """u(x) = x^3 (1-x)^3"""
    x = np.asarray(x).flatten()
    return x**3 * (1-x)**3

def forcing_term_poisson(x, alpha):
    """Công thức (4.3) trong bài báo"""
    x = np.asarray(x).flatten()
    # tránh chia cho 0 tại biên nếu alpha lẻ, nhưng ở đây alpha=1.5 an toàn
    term1 = (gamma(4)/gamma(4-alpha)) * (x**(3-alpha) + (1-x)**(3-alpha))
    term2 = 3*(gamma(5)/gamma(5-alpha)) * (x**(4-alpha) + (1-x)**(4-alpha))
    term3 = 3*(gamma(6)/gamma(6-alpha)) * (x**(5-alpha) + (1-x)**(5-alpha))
    term4 = (gamma(7)/gamma(7-alpha)) * (x**(6-alpha) + (1-x)**(6-alpha))
    f = (1/(2*np.cos(np.pi*alpha/2))) * (term1 - term2 + term3 - term4)
    return f.reshape(-1,1)

def generate_training_data(N, alpha):
    """Tạo điểm huấn luyện và forcing term cho 1D fractional Poisson"""
    dx = 1.0 / N
    x_train = np.linspace(dx, 1-dx, N-1).reshape(-1,1)
    f_train = forcing_term_poisson(x_train, alpha)
    return x_train, f_train, dx