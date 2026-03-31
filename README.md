# Kalman Filter — from scratch in NumPy

Implementation of the linear Kalman Filter and Extended Kalman Filter using only NumPy. 

---

## Structure

```
kalman_quant/
├── src/
│   ├── kalman.py          # core KF, RTS smoother
│   └── extended_kalman.py # EKF for nonlinear systems
├── main.py                       # four demos
├── visualise.py                  # plots
└── results/                      # saved figures and data
```

---

## The model

The filter assumes a linear state-space model:

```
x_k = F · x_{k-1} + w_k       w_k ~ N(0, Q)   process equation
z_k = H · x_k     + v_k       v_k ~ N(0, R)   measurement equation
```

`x` is the hidden state you want to estimate. `z` is the noisy observation you actually see. `F` encodes how the state evolves. `H` maps state to observation. `Q` and `R` are the noise covariances.

---

## Algorithm

Every timestep runs two steps.

**Predict** — project forward before seeing the new measurement:
```
x̂_k⁻ = F · x̂_{k-1}
P_k⁻  = F · P_{k-1} · Fᵀ + Q
```

**Update** — correct using the new measurement:
```
K     = P_k⁻ · Hᵀ · (H · P_k⁻ · Hᵀ + R)⁻¹
x̂_k   = x̂_k⁻ + K · (z_k - H · x̂_k⁻)
P_k   = (I - K·H) · P_k⁻ · (I - K·H)ᵀ + K·R·Kᵀ
```

`K` is the Kalman gain — the weight that balances trust in the prediction versus the measurement. When `R` is small (accurate sensor), `K` is large and the measurement dominates. When `Q` is small (accurate model), `K` is small and the prediction dominates.

The covariance update uses the Joseph form rather than the simpler `(I-KH)P` because it remains numerically stable when floating-point errors accumulate over long sequences.

---

## RTS Smoother

The forward filter only uses past information to estimate `x_k`. The Rauch-Tung-Striebel smoother runs a backward pass after filtering, using all future observations to improve each estimate. It always reduces uncertainty — `P_smoothed <= P_filtered` at every step.

```
G_k      = P_k · Fᵀ · (P_{k+1}⁻)⁻¹
x̂_k|T   = x̂_k + G_k · (x̂_{k+1|T} - x̂_{k+1}⁻)
P_k|T    = P_k  + G_k · (P_{k+1|T} - P_{k+1}⁻) · G_kᵀ
```

---

## Extended Kalman Filter

For nonlinear systems where `f` and `h` are arbitrary functions rather than matrices, the EKF linearises around the current estimate using first-order Taylor expansion (Jacobians). Jacobians can be supplied analytically or computed automatically via central differences.

```
x_k = f(x_{k-1}) + w_k
z_k = h(x_k)     + v_k

F_k = df/dx |_{x̂_{k-1}}    (linearise f)
H_k = dh/dx |_{x̂_k⁻}      (linearise h)
```

The predict and update equations then follow the same form as the linear filter, using `F_k` and `H_k` in place of the fixed `F` and `H`.

---

## Demos

**1. Signal extraction** — a latent AR(1) signal is corrupted by noise. The filter recovers the clean signal, reducing RMSE by ~57% versus the raw observations.

**2. Trend extraction** — a two-state model `[level, slope]` separates the underlying trend from noise in a sequence. The smoother extracts both the level and the instantaneous slope.

**3. Nonlinear tracking (EKF)** — a particle moves in a circle. The state `[x, y, vx, vy]` is nonlinear relative to the observations, so the EKF is used. Numerical Jacobians are computed automatically.


---

## Usage

```python
from src.kalman_filter import KalmanFilter
import numpy as np

kf = KalmanFilter(
    F  = np.array([[1., 1.], [0., 1.]]),  # state transition
    H  = np.array([[1., 0.]]),             # observation matrix
    Q  = np.diag([1e-4, 1e-6]),            # process noise
    R  = np.array([[0.09]]),               # measurement noise
    x0 = np.zeros(2),                      # initial state
    P0 = np.eye(2),                        # initial covariance
)

result = kf.filter(observations)   # forward pass
result = kf.smooth(result)         # backward pass (optional)

level = result.x_smoothed[:, 0]
slope = result.x_smoothed[:, 1]
```

```python
# Run demos and generate plots
python main.py
python visualise.py
```

---

## Dependencies

```
numpy
matplotlib   # visualise.py only
```

---