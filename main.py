"""
main.py
-------
Demos for the Kalman Filter library.

1. Signal extraction - recover a clean signal from noisy observations
2. Trend extraction - separate level and slope from a noisy sequence
3. Nonlinear tracking - EKF tracking a particle moving in a circle
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from kalman         import KalmanFilter
from extended_kalman import ExtendedKalmanFilter

rng = np.random.default_rng(42)
os.makedirs("results", exist_ok=True)

T = 500


def section(title):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


# ── 1. Signal extraction ──────────────────────────────────────────────────────
section("1. Signal Extraction")

# AR(1) latent signal + noise
signal_true = np.zeros(T)
for t in range(1, T):
    signal_true[t] = 0.97 * signal_true[t-1] + 0.5 * rng.standard_normal()
obs = signal_true + 1.5 * rng.standard_normal(T)

# KF: AR(1) state, direct observation
kf  = KalmanFilter(
    F=np.array([[0.97]]),
    H=np.array([[1.0]]),
    Q=np.array([[0.25]]),
    R=np.array([[2.25]]),
    x0=np.zeros(1),
    P0=np.array([[1.0]]),
)
res = kf.filter(obs.reshape(-1, 1))
res = kf.smooth(res)

signal_kf = res.x_smoothed[:, 0]
sigma_kf  = np.sqrt(res.P_smoothed[:, 0, 0])

print(f"  RMSE raw : {rmse(obs, signal_true):.4f}")
print(f"  RMSE KF  : {rmse(signal_kf, signal_true):.4f}")

np.savez("results/signal_extraction.npz",
         obs=obs, signal_true=signal_true,
         signal_kf=signal_kf, sigma_kf=sigma_kf)


# ── 2. Trend extraction ───────────────────────────────────────────────────────
section("2. Trend Extraction (local-linear model)")

# simulated sequence with drift + noise
trend  = np.cumsum(rng.normal(0, 0.01, T))
noisy  = trend + rng.normal(0, 0.3, T)

# state = [level, slope]
kf2 = KalmanFilter(
    F=np.array([[1., 1.], [0., 1.]]),
    H=np.array([[1., 0.]]),
    Q=np.diag([1e-4, 1e-6]),
    R=np.array([[0.09]]),
    x0=np.zeros(2),
    P0=np.eye(2),
)
res2  = kf2.filter(noisy.reshape(-1, 1))
res2  = kf2.smooth(res2)

level = res2.x_smoothed[:, 0]
slope = res2.x_smoothed[:, 1]
std   = np.sqrt(res2.P_smoothed[:, 0, 0])

print(f"  RMSE level vs trend: {rmse(level, trend):.4f}")
print(f"  Mean slope (x252):   {slope.mean()*252:.4f}")

np.savez("results/trend_extraction.npz",
         noisy=noisy, trend=trend, level=level, slope=slope,
         upper=level+2*std, lower=level-2*std)


# ── 3. Nonlinear tracking (EKF) ───────────────────────────────────────────────
section("3. Nonlinear Tracking (EKF — circular motion)")

# particle moves in a circle, we observe (x, y) + noise
omega = 0.05   # angular velocity
r     = 10.0   # radius

angles = omega * np.arange(T)
px_true = r * np.cos(angles)
py_true = r * np.sin(angles)
obs_x   = px_true + rng.normal(0, 1.0, T)
obs_y   = py_true + rng.normal(0, 1.0, T)
Z       = np.column_stack([obs_x, obs_y])

# state = [x, y, vx, vy]
def f(x):
    dt = 1.0
    return np.array([x[0] + dt*x[2],
                     x[1] + dt*x[3],
                     x[2],
                     x[3]])

def h(x):
    return np.array([x[0], x[1]])

ekf = ExtendedKalmanFilter(
    f=f, h=h,
    Q=np.diag([0.01, 0.01, 0.1, 0.1]),
    R=np.eye(2) * 1.0,
    x0=np.array([r, 0., 0., omega*r]),
    P0=np.eye(4),
)
ekf_res = ekf.filter(Z)

pos_kf = ekf_res.x_filtered[:, :2]
print(f"  RMSE (x,y) raw : {rmse(Z, np.column_stack([px_true, py_true])):.4f}")
print(f"  RMSE (x,y) EKF : {rmse(pos_kf, np.column_stack([px_true, py_true])):.4f}")

np.savez("results/nonlinear_tracking.npz",
         px_true=px_true, py_true=py_true,
         obs_x=obs_x, obs_y=obs_y,
         kf_x=pos_kf[:, 0], kf_y=pos_kf[:, 1])


