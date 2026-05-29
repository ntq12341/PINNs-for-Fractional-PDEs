# network.py
import torch
import torch.nn as nn
import numpy as np
from fraction import fractional_laplacian_1d

class fPINN(nn.Module):
    def __init__(self, layers, rho_func=None):
        """
        layers: list số neuron mỗi lớp, ví dụ [1, 20, 20, 20, 1]
        rho_func: hàm ρ(x) để thỏa mãn BC, mặc định x(1-x) cho 1D.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(nn.Tanh())
        # self.layers kết hợp Linear và Tanh xen kẽ
        # Để đơn giản, dùng Sequential
        self.net = nn.Sequential()
        for i in range(len(layers)-1):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f'tanh_{i}', nn.Tanh())
        if rho_func is None:
            self.rho = lambda x: x * (1 - x)
        else:
            self.rho = rho_func

    def forward(self, x):
        # x: tensor shape (batch, 1)
        u_nn = self.net(x)
        u = self.rho(x) * u_nn
        return u

def build_loss_1d_poisson(model, x_train, f_train, N, alpha, dx, device):
    """
    Xây dựng hàm loss cho bài toán 1D fractional Poisson.
    Trả về hàm loss (có thể gọi với model) dùng cho training.
    """
    # Tạo lưới đầy đủ N+1 điểm (kể cả biên)
    x_grid = torch.linspace(0.0, 1.0, N+1, device=device).reshape(-1, 1)
    
    # Ma trận GL (N-1) x (N+1) bằng numpy, sau đó chuyển sang torch tensor
    L = np.zeros((N-1, N+1))
    cos_term = 1.0 / (2 * np.cos(np.pi * alpha / 2))
    for i in range(1, N):
        j = i
        # left
        for k in range(0, j+1):
            idx = j - (k-1)
            if 0 <= idx <= N:
                L[i-1, idx] += cos_term * ((-1)**k * comb(alpha, k)) / (dx**alpha)
        # right
        for k in range(0, N-j+1):
            idx = j + (k-1)
            if 0 <= idx <= N:
                L[i-1, idx] += cos_term * ((-1)**k * comb(alpha, k)) / (dx**alpha)
    L_torch = torch.tensor(L, dtype=torch.float32, device=device)
    f_train_torch = torch.tensor(f_train, dtype=torch.float32, device=device).reshape(-1,1)
    
    def loss_fn():
        u_grid = model(x_grid)                # (N+1, 1)
        lap = torch.mm(L_torch, u_grid)       # (N-1, 1)
        residual = lap - f_train_torch
        mse = torch.mean(residual**2)
        return mse
    return loss_fn

def train(model, loss_fn, optimizer, scheduler, iterations, log_freq=1000, device='cpu'):
    """
    Vòng lặp training.
    scheduler: có thể là None hoặc lr_scheduler
    """
    model.train()
    loss_history = []
    for it in range(iterations):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if it % log_freq == 0:
            loss_val = loss.item()
            loss_history.append(loss_val)
            print(f"Iter {it}, loss = {loss_val:.2e}")
    return loss_history