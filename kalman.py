"""
kalman_filter.py
────────────────
Pure-NumPy implementation of the classical (linear) Kalman Filter.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalmanResult:
    """Stores the full filter trajectory."""
    x_filtered:     np.ndarray   # (T, n)  filtered state means
    P_filtered:     np.ndarray   # (T, n, n) filtered state covariances
    x_predicted:    np.ndarray   # (T, n)
    P_predicted:    np.ndarray   # (T, n, n)
    innovations:    np.ndarray   # (T, m)  y_k - H x_k^-
    S:              np.ndarray   # (T, m, m) innovation covariances
    K:              np.ndarray   # (T, n, m) Kalman gains
    log_likelihood: float        # total log-likelihood

    # optional RTS smoother outputs
    x_smoothed:     Optional[np.ndarray] = None
    P_smoothed:     Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core filter
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Linear Kalman Filter with optional RTS smoother.

    Parameters
    ──────────
    F  : (n, n)  State transition matrix
    H  : (m, n)  Observation matrix
    Q  : (n, n)  Process noise covariance
    R  : (m, m)  Measurement noise covariance
    B  : (n, k)  Control-input matrix  (optional, None → no control)
    x0 : (n,)    Initial state mean
    P0 : (n, n)  Initial state covariance
    """

    def __init__(
        self,
        F:  np.ndarray,
        H:  np.ndarray,
        Q:  np.ndarray,
        R:  np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        B:  Optional[np.ndarray] = None,
    ) -> None:
        self.F  = np.asarray(F,  dtype=float)
        self.H  = np.asarray(H,  dtype=float)
        self.Q  = np.asarray(Q,  dtype=float)
        self.R  = np.asarray(R,  dtype=float)
        self.x0 = np.asarray(x0, dtype=float).ravel()
        self.P0 = np.asarray(P0, dtype=float)
        self.B  = np.asarray(B,  dtype=float) if B is not None else None

        self.n = self.F.shape[0]   # state dimension
        self.m = self.H.shape[0]   # observation dimension

        self._validate()

    # ── validation ──────────────────────────────────────────────────────────

    def _validate(self) -> None:  #make sure variables are correct dimension
        assert self.F.shape  == (self.n, self.n),  "F must be (n×n)"
        assert self.H.shape  == (self.m, self.n),  "H must be (m×n)"
        assert self.Q.shape  == (self.n, self.n),  "Q must be (n×n)"
        assert self.R.shape  == (self.m, self.m),  "R must be (m×m)"
        assert self.x0.shape == (self.n,),         "x0 must be length n"
        assert self.P0.shape == (self.n, self.n),  "P0 must be (n×n)"

    # ── forward pass (predict → update) ─────────────────────────────────────

    def filter(
        self,
        Z:  np.ndarray,
        U:  Optional[np.ndarray] = None,
    ) -> KalmanResult:
        """
        Run the forward Kalman filter.

        Parameters
        ──────────
        Z : (T, m)  measurement sequence (NaN values are handled as missing)
        U : (T, k)  control-input sequence (optional)

        Returns
        ───────
        KalmanResult  (call .smooth() afterwards for RTS smoothing)
        """
        Z = np.atleast_2d(Z)
        if Z.shape[1] != self.m:
            Z = Z.T                     # tolerate (m, T) input
        T = Z.shape[0]

        # pre-allocate
        xp  = np.zeros((T, self.n))       # predicted means
        Pp  = np.zeros((T, self.n, self.n))
        xf  = np.zeros_like(xp)           # filtered means
        Pf  = np.zeros_like(Pp)
        inn = np.zeros((T, self.m))       # innovations
        S   = np.zeros((T, self.m, self.m))
        K   = np.zeros((T, self.n, self.m))
        ll  = 0.0                         # log-likelihood accumulator

        x = self.x0.copy()
        P = self.P0.copy()

        for k in range(T):
            # ── PREDICT ────────────────────────────────────────────────────
            u_k = U[k] if U is not None else None
            x, P = self._predict(x, P, u_k)

            xp[k] = x
            Pp[k] = P

            # ── UPDATE (skip if all measurements missing) ───────────────────
            z_k = Z[k]
            obs_idx = ~np.isnan(z_k)

            if obs_idx.any():
                x, P, inn_k, S_k, K_k, ll_k = self._update(x, P, z_k, obs_idx)
                inn[k, obs_idx] = inn_k
                S[k][np.ix_(obs_idx, obs_idx)] = S_k
                K[k][:, obs_idx] = K_k
                ll += ll_k

            xf[k] = x
            Pf[k] = P

        return KalmanResult(
            x_filtered=xf, P_filtered=Pf,
            x_predicted=xp, P_predicted=Pp,
            innovations=inn, S=S, K=K,
            log_likelihood=ll,
        )

    # ── predict step ─────────────────────────────────────────────────────────

    def _predict(self, x, P, u=None):
        x_pred = self.F @ x
        if u is not None and self.B is not None:
            x_pred += self.B @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    # ── update step ──────────────────────────────────────────────────────────

    def _update(self, x, P, z, obs_idx):
        H_k = self.H[obs_idx, :]
        R_k = self.R[np.ix_(obs_idx, obs_idx)]
        z_k = z[obs_idx]

        # innovation & covariance
        inn  = z_k - H_k @ x
        S_k  = H_k @ P @ H_k.T + R_k
        S_inv = np.linalg.inv(S_k)

        # Kalman gain
        K_k = P @ H_k.T @ S_inv

        # state update
        x_upd = x + K_k @ inn
        I_KH  = np.eye(self.n) - K_k @ H_k
        # Joseph form for numerical stability
        P_upd = I_KH @ P @ I_KH.T + K_k @ R_k @ K_k.T

        # log-likelihood contribution  (Gaussian)
        sign, logdet = np.linalg.slogdet(S_k)
        m_k = obs_idx.sum()
        ll_k = -0.5 * (m_k * np.log(2 * np.pi) + logdet + inn @ S_inv @ inn)

        return x_upd, P_upd, inn, S_k, K_k, ll_k

    # ── RTS smoother ─────────────────────────────────────────────────────────

    def smooth(self, result: KalmanResult) -> KalmanResult:
        """
        Rauch–Tung–Striebel (RTS) smoother — backward pass.

        Improves state estimates using all future information.
        Modifies result in-place and returns it.
        """
        T  = result.x_filtered.shape[0]
        xs = result.x_filtered.copy()
        Ps = result.P_filtered.copy()

        for k in range(T - 2, -1, -1):
            Pp_k1 = result.P_predicted[k + 1]
            G_k   = result.P_filtered[k] @ self.F.T @ np.linalg.inv(Pp_k1)

            xs[k] = result.x_filtered[k] + G_k @ (xs[k + 1] - result.x_predicted[k + 1])
            Ps[k] = result.P_filtered[k] + G_k @ (Ps[k + 1] - Pp_k1) @ G_k.T

        result.x_smoothed = xs
        result.P_smoothed = Ps
        return result

    # ── parameter estimation (EM) ─────────────────────────────────────────────

    def em_step(self, Z: np.ndarray, result: KalmanResult) -> dict:
        """
        One M-step of the EM algorithm for parameter estimation.
        Updates Q and R given the smoothed estimates.
        Returns dict of updated parameters.
        """
        if result.x_smoothed is None:
            raise ValueError("Run smooth() before em_step().")

        T  = Z.shape[0]
        xs = result.x_smoothed
        Ps = result.P_smoothed
        xp = result.x_predicted
        Pp = result.P_predicted

        # Cross-covariance  E[x_k x_{k-1}^T | Z_1:T]
        Q_new = np.zeros_like(self.Q)
        R_new = np.zeros_like(self.R)

        for k in range(1, T):
            G_k = result.P_filtered[k-1] @ self.F.T @ np.linalg.inv(Pp[k])
            cross = Ps[k] + np.outer(xs[k], xs[k-1]) + G_k @ (Ps[k] - np.outer(xs[k], xs[k]))
            err   = xs[k] - self.F @ xs[k-1]
            Q_new += np.outer(err, err) + Ps[k] - self.F @ G_k.T @ Ps[k-1]

        Q_new /= (T - 1)

        for k in range(T):
            z_k = Z[k]
            obs = ~np.isnan(z_k)
            if obs.any():
                H_k = self.H[obs, :]
                err  = z_k[obs] - H_k @ xs[k]
                R_new[np.ix_(obs, obs)] += (
                    np.outer(err, err) + H_k @ Ps[k] @ H_k.T
                )

        R_new /= T
        return {"Q": Q_new, "R": R_new}