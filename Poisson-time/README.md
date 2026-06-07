# Poisson-time fPINN

This folder contains a first implementation of the time-dependent 1D fractional ADE examples from Section 4.1.2 of the fPINNs paper.

The manufactured solution is

```text
u(x,t) = x (1 - x^2)^(1 + alpha/2) exp(-t),  x in [-1,1], t in [0,1].
```

The fPINN ansatz enforces both boundary and initial conditions:

```text
u_hat(x,t) = u_exact(x,0) + t (1-x^2) NN(x,t)
```

Implemented operators:

```text
Caputo time derivative: L1 scheme
Space fractional Laplacian: first-order shifted GL
Advection term: autograd u_x
```

The forcing term is manufactured by applying the same discrete operators to the exact solution. This keeps the training residual consistent with the implemented Caputo/GL operators.

## Files

```text
utils.py      manufactured solution, point generation, errors
network.py    TimeFPINN model and exact-solution wrapper
operators.py  Caputo L1, GL space operator, PDE residual
main.py       run experiments and write CSV
figures.py    plot PNG figures from CSV
```

## Run

Quick Fig. 8 smoke test:

```powershell
python Poisson-time\main.py --only fig8 --lambda-ts 5 --n-scattered 8 --iterations 2 --width 4 --depth 1
python Poisson-time\figures.py --only fig8
```

Quick Fig. 9 smoke test:

```powershell
python Poisson-time\main.py --only fig9 --lambda-ts 5 --n-scattered 8 --iterations 2 --width 4 --depth 1
python Poisson-time\figures.py --only fig9
```

Outputs:

```text
Poisson-time\results\fig8_results.csv
Poisson-time\results\fig8_accuracy.png
Poisson-time\results\fig9_results.csv
Poisson-time\results\fig9_accuracy.png
```

The defaults are intentionally modest compared with the paper. Increase `--iterations`, `--lambda-ts`, `--n-scattered`, and `--seeds` for more meaningful runs.
