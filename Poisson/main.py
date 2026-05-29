# main.py
import torch
import numpy as np
from network import fPINN, build_loss_1d_poisson, train
from error import relative_l2_error
from utils import exact_solution_poisson, generate_training_data

# ===== THAM SỐ BÀI TOÁN =====
alpha = 1.5
N = 20                     # số điểm lưới (λ = N)
x_train, f_train, dx = generate_training_data(N, alpha)

# ===== THAM SỐ MẠNG =====
layers = [1, 20, 20, 20, 20, 1]   # input: x, output: u_NN
learning_rate = 1e-4
iterations = 100000

# ===== THIẾT BỊ =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER =====
model = fPINN(layers).to(device)
loss_fn = build_loss_1d_poisson(model, x_train, f_train, N, alpha, dx, device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)  # optional

# ===== TRAINING =====
print("Bắt đầu training...")
loss_hist = train(model, loss_fn, optimizer, scheduler, iterations, log_freq=2000, device=device)

# ===== ĐÁNH GIÁ =====
model.eval()
x_test = np.linspace(0, 1, 1000).reshape(-1,1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
with torch.no_grad():
    u_pred = model(x_test_tensor).cpu().numpy().flatten()
u_exact = exact_solution_poisson(x_test)
err = relative_l2_error(u_pred, u_exact)
print(f"Relative L2 error: {err:.2e}")

# # (Tùy chọn) Vẽ đồ thị
# import matplotlib.pyplot as plt
# plt.plot(x_test, u_exact, 'k-', label='Exact')
# plt.plot(x_test, u_pred, 'r--', label='fPINN')
# plt.legend()
# plt.show()