"""Run one fPINN solve for the smooth Poisson example in Section 4.1.1."""

import numpy as np
import torch

from error import relative_l2_error
from network import build_loss_poisson, fPINN, train
from utils import exact_solution_poisson


def main():
    alpha = 1.5
    N = 20
    order = 1

    layers = [1, 20, 20, 20, 20, 1]
    learning_rate = 1e-4
    iterations = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.float64)
    model = fPINN(layers).to(device)
    loss_fn = build_loss_poisson(model, N=N, alpha=alpha, order=order, device=device, smooth=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training 1D fractional Poisson fPINN...")
    history = train(model, loss_fn, optimizer, iterations, log_freq=2000)

    model.eval()
    x_test = np.linspace(0.0, 1.0, 1000).reshape(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float64, device=device)
    with torch.no_grad():
        u_pred = model(x_test_tensor).cpu().numpy().reshape(-1)

    u_exact = exact_solution_poisson(x_test)
    err = relative_l2_error(u_pred, u_exact)
    print(f"Final loss: {history['loss_final']:.3e}")
    print(f"Relative L2 error: {err:.3e}")


if __name__ == "__main__":
    main()
