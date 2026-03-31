"""
visualise.py — simple plots, no styling
Run main.py first.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)


# Fig 1: Signal extraction
d = np.load("results/signal_extraction.npz")
t = np.arange(len(d["obs"]))

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle("Signal Extraction")

axes[0].plot(t, d["obs"],         label="Noisy obs", alpha=0.5)
axes[0].plot(t, d["signal_true"], label="True signal")
axes[0].plot(t, d["signal_kf"],   label="KF smoothed")
axes[0].fill_between(t,
    d["signal_kf"] - 2*d["sigma_kf"],
    d["signal_kf"] + 2*d["sigma_kf"],
    alpha=0.2, label="+-2 sigma")
axes[0].legend(); axes[0].grid(); axes[0].set_title("Signal recovery")

err_raw = abs(d["obs"]        - d["signal_true"])
err_kf  = abs(d["signal_kf"] - d["signal_true"])
axes[1].plot(t, err_raw, label=f"Raw RMSE={err_raw.mean():.3f}")
axes[1].plot(t, err_kf,  label=f"KF  RMSE={err_kf.mean():.3f}")
axes[1].legend(); axes[1].grid(); axes[1].set_title("Absolute error")

axes[2].plot(t, d["sigma_kf"], label="Posterior sigma")
axes[2].axhline(d["sigma_kf"].mean(), ls="--", label=f"Mean={d['sigma_kf'].mean():.4f}")
axes[2].legend(); axes[2].grid(); axes[2].set_title("Posterior uncertainty")

plt.tight_layout()
plt.savefig("results/fig1_signal_extraction.png", dpi=130, bbox_inches="tight")
plt.close()
print("fig1 done")


# Fig 2: Trend extraction
d = np.load("results/trend_extraction.npz")
t = np.arange(len(d["noisy"]))

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("Trend Extraction")

axes[0].plot(t, d["noisy"], alpha=0.4, label="Noisy")
axes[0].plot(t, d["trend"],            label="True trend")
axes[0].plot(t, d["level"],            label="KF level")
axes[0].fill_between(t, d["lower"], d["upper"], alpha=0.2, label="+-2 sigma")
axes[0].legend(); axes[0].grid(); axes[0].set_title("Level")

axes[1].plot(t, d["slope"], label="KF slope")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].legend(); axes[1].grid(); axes[1].set_title("Slope")

plt.tight_layout()
plt.savefig("results/fig2_trend_extraction.png", dpi=130, bbox_inches="tight")
plt.close()
print("fig2 done")


# Fig 3: Nonlinear tracking
d = np.load("results/nonlinear_tracking.npz")

fig, ax = plt.subplots(figsize=(7, 7))
fig.suptitle("EKF Nonlinear Tracking (circular motion)")

ax.plot(d["px_true"], d["py_true"],  label="True path", lw=2)
ax.scatter(d["obs_x"], d["obs_y"],   label="Noisy obs", s=4, alpha=0.3)
ax.plot(d["kf_x"],   d["kf_y"],      label="EKF estimate", lw=1.5)
ax.legend(); ax.grid(); ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("results/fig3_nonlinear_tracking.png", dpi=130, bbox_inches="tight")
plt.close()
print("fig3 done")


