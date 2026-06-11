# Poisson-inverse fPINN

This folder implements a 1D version of the inverse problems in Section 4.2 of the fPINNs paper.

The paper reports 9 inverse problems in Table 3:

```text
1D/2D/3D time-fractional ADE
1D/2D/3D space-fractional ADE
1D/2D/3D space-time-fractional ADE
```

This implementation covers the 1D cases:

```text
tf:
  time-fractional ADE
  identify gamma, c, velocity
  alpha is fixed to 2

sf:
  space-fractional ADE
  identify alpha, c, velocity
  gamma is fixed to 1

stf:
  space-time-fractional ADE
  identify alpha, gamma, c, velocity
```

The trainable parameters use the transforms from the paper:

```text
alpha = 0.5 tanh(alpha0) + 1.5
gamma = 0.5 tanh(gamma0) + 0.5
c = exp(c0)
velocity = exp(v0)
```

The inverse loss is:

```text
Loss = w1 * MSE(PDE residual at Xi1) + w2 * MSE(u_pred(x,T) - h_BB(x,T) at Xi2)
```

where:

```text
Xi1: residual/training points in Omega x (0,T), drawn from Sobol points
Xi2: final-time observation points in Omega x {T}, drawn from Latin hypercube points
```

## Files

```text
utils.py       exact solution, sampling, error metric
network.py     fPINN model and exact-solution wrapper
parameters.py  trainable inverse PDE parameters and transforms
operators.py   differentiable Caputo/GL/advection operators and inverse loss
main.py        run inverse experiments and write CSV
figures.py     plot parameter trajectories
```

## Run

Quick smoke test:

```powershell
python Poisson-inverse\main.py --cases tf sf stf --iterations 2 --n-residual 4 --n-final 2 --lambda-x 3 --lambda-t 3 --width 4 --depth 1
python Poisson-inverse\figures.py
```

More meaningful 1D run:

```powershell
python Poisson-inverse\main.py --cases tf sf stf --seeds 0 1 2 3 4 5 6 7 8 9 --iterations 100000 --n-residual 100 --n-final 20 --lambda-x 400 --lambda-t 200 --width 20 --depth 4
python Poisson-inverse\figures.py
```

The second command is expensive.

Outputs:

```text
Poisson-inverse\results\inverse_summary.csv
Poisson-inverse\results\inverse_trajectory.csv
Poisson-inverse\results\inverse_trajectory.png
```

## Notes

The GL coefficients are implemented with a torch recurrence so gradients can flow through `alpha`. The Caputo L1 coefficients are torch expressions so gradients can flow through `gamma`.

The forcing data are manufactured by applying the same discrete operators to the exact solution with the true PDE parameters.
