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

## Cases

`Poisson-time` currently runs three different cases.

Quick summary:

```text
fig8_space:
  space-fractional ADE
  alpha = 1.5, gamma = 1.0
  fractional in space, ordinary in time

fig8_time:
  time-fractional ADE
  alpha = 2.0, gamma = 0.5
  Caputo in time, ordinary in space

fig9:
  space-time-fractional ADE
  alpha = 1.5, gamma = 0.5
  Caputo in time and fractional in space
```

### `fig8_space`

This is the 1D space-fractional ADE used in Fig. 8(a).

```text
alpha = 1.5
gamma = 1.0
c = 1.0
velocity = 0.0
forcing = WB
```

Meaning:

```text
alpha = 1.5  -> fractional operator in space
gamma = 1.0  -> ordinary first-order time derivative
v = 0        -> no advection
WB forcing   -> manufactured forcing is known at the training points
```

Residual form:

```text
u_t + c (-Delta)^(alpha/2) u - f = 0
```

Operators used:

```text
u_t                      -> autograd in t
(-Delta)^(alpha/2) u     -> first-order shifted GL in x
```

### `fig8_time`

This is the 1D time-fractional ADE used in Fig. 8(b).

```text
alpha = 2.0
gamma = 0.5
c = 1.0
velocity = 0.0
forcing = WB
```

Meaning:

```text
alpha = 2.0  -> ordinary second-order spatial operator
gamma = 0.5  -> Caputo fractional derivative in time
v = 0        -> no advection
WB forcing   -> manufactured forcing is known at the training points
```

Residual form:

```text
D_t^gamma u + c (-Delta) u - f = 0
```

Operators used:

```text
D_t^gamma u   -> Caputo L1 scheme
(-Delta) u    -> autograd second derivative in x
```

### `fig9`

This is the 1D space-time-fractional ADE used in Fig. 9.

```text
alpha = 1.5
gamma = 0.5
c = 1.0
velocity = 0.1
forcing = BB
```

Meaning:

```text
alpha = 1.5  -> fractional operator in space
gamma = 0.5  -> Caputo fractional derivative in time
v = 0.1      -> advection is included
BB forcing   -> forcing is available only at scattered training points
```

Residual form:

```text
D_t^gamma u + c (-Delta)^(alpha/2) u + v u_x - f = 0
```

Operators used:

```text
D_t^gamma u              -> Caputo L1 scheme
(-Delta)^(alpha/2) u     -> first-order shifted GL in x
u_x                      -> autograd in x
```

## Training points and lambda parameters

For Fig. 8, the code runs both:

```text
scattered training points
lattice-like training points
```

For Fig. 9, the code runs only:

```text
scattered training points
```

The parameter `lambda_t` controls the number of auxiliary time points in the Caputo L1 scheme.

The parameter `lambda_x` controls the number of auxiliary spatial points in the GL operator.

The current mapping is:

```text
fig8_space:
  lambda_x = lambda_t

fig8_time:
  lambda_x = lambda_t^((2-gamma)/2)

fig9:
  lambda_x = lambda_t
```

## Files

```text
utils.py      manufactured solution, point generation, errors
network.py    TimeFPINN model and exact-solution wrapper
operators.py  Caputo L1, GL space operator, PDE residual
main.py       run experiments and write CSV
figures.py    plot PNG figures from CSV
```

## Run

Run Fig. 8:

```powershell
python Poisson-time\main.py --only fig8
python Poisson-time\figures.py --only fig8
```

This runs:

```text
fig8_space
fig8_time
```

Run Fig. 9:

```powershell
python Poisson-time\main.py --only fig9
python Poisson-time\figures.py --only fig9
```

This runs:

```text
fig9
```

Run all cases:

```powershell
python Poisson-time\main.py --only both
python Poisson-time\figures.py --only both
```

This runs:

```text
fig8_space
fig8_time
fig9
```

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
