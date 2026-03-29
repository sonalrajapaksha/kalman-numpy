"""
extended_kalman_filter.py
─────────────────────────
Extended Kalman Filter for nonlinear systems.

Model
─────
  x_k = f(x_{k-1}) + w_k     w_k ~ N(0, Q)
  z_k = h(x_k)     + v_k     v_k ~ N(0, R)

Jacobians of f and h can be supplied analytically or computed numerically.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class EKFResult:
    x_filtered:  np.ndarray   # (T, n)
    P_filtered:  np.ndarray   # (T, n, n)
    x_predicted: np.ndarray   # (T, n)
    P_predicted: np.ndarray   # (T, n, n)
    innovations: np.ndarray   # (T, m)
    log_likelihood: float


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear f and h.

    Parameters
    ----------
    f     : state transition function  x_{k-1} -> x_k
    h     : observation function       x_k     -> z_k
    Q     : (n, n)  process noise covariance
    R     : (m, m)  measurement noise covariance
    x0    : (n,)    initial state
    P0    : (n, n)  initial covariance
    F_jac : Jacobian of f (optional, falls back to numerical)
    H_jac : Jacobian of h (optional, falls back to numerical)
    eps   : step size for numerical Jacobian
    """

    def __init__(self, f, h, Q, R, x0, P0, F_jac=None, H_jac=None, eps=1e-5):
        self.f     = f
        self.h     = h
        self.Q     = np.asarray(Q,  dtype=float)
        self.R     = np.asarray(R,  dtype=float)
        self.x0    = np.asarray(x0, dtype=float).ravel()
        self.P0    = np.asarray(P0, dtype=float)
        self._F_jac = F_jac
        self._H_jac = H_jac
        self.eps   = eps
        self.n     = len(self.x0)
        self.m     = self.R.shape[0]

    def _numerical_jacobian(self, func, x, out_dim):
        J = np.zeros((out_dim, len(x)))
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += self.eps
            xm[i] -= self.eps
            J[:, i] = (func(xp) - func(xm)) / (2 * self.eps)
        return J

    def F_jacobian(self, x):
        if self._F_jac is not None:
            return self._F_jac(x)
        return self._numerical_jacobian(self.f, x, self.n)

    def H_jacobian(self, x):
        if self._H_jac is not None:
            return self._H_jac(x)
        return self._numerical_jacobian(self.h, x, self.m)

    def filter(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[1] != self.m:
            Z = Z.T
        T = Z.shape[0]

        xp_all  = np.zeros((T, self.n))
        Pp_all  = np.zeros((T, self.n, self.n))
        xf_all  = np.zeros((T, self.n))
        Pf_all  = np.zeros((T, self.n, self.n))
        inn_all = np.zeros((T, self.m))
        ll      = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for k in range(T):
            # predict
            Fk  = self.F_jacobian(x)
            x_p = self.f(x)
            P_p = Fk @ P @ Fk.T + self.Q

            xp_all[k] = x_p
            Pp_all[k] = P_p

            # update
            Hk  = self.H_jacobian(x_p)
            inn = Z[k] - self.h(x_p)
            S   = Hk @ P_p @ Hk.T + self.R
            S_inv = np.linalg.inv(S)
            K   = P_p @ Hk.T @ S_inv

            x = x_p + K @ inn
            IKH = np.eye(self.n) - K @ Hk
            P   = IKH @ P_p @ IKH.T + K @ self.R @ K.T

            xf_all[k]  = x
            Pf_all[k]  = P
            inn_all[k] = inn

            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (self.m * np.log(2 * np.pi) + logdet + inn @ S_inv @ inn)

        return EKFResult(
            x_filtered=xf_all, P_filtered=Pf_all,
            x_predicted=xp_all, P_predicted=Pp_all,
            innovations=inn_all, log_likelihood=ll,
        )