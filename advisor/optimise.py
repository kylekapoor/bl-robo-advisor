"""Constrained mean-variance optimisation.

Deliberately not CVXPY. The problem is 20 assets with a budget constraint and a
box, which SLSQP solves in milliseconds from a sensible warm start. A convex
solver here would be a dependency and a build step to buy nothing.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _prepare(mu, cov):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (len(mu), len(mu)):
        raise ValueError(f"cov must be {(len(mu), len(mu))}, got {cov.shape}")
    return mu, cov


def _solve(objective, n: int, max_weight: float, x0=None):
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.repeat(1.0 / n, n) if x0 is None else x0

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(result.x, 0.0, max_weight)
    total = w.sum()
    # SLSQP satisfies the budget to solver tolerance, not exactly; renormalise so
    # downstream P&L is not quietly computed on a 99.7%-invested book.
    return w / total if total > 0 else np.repeat(1.0 / n, n)


def max_sharpe(mu, cov, risk_free: float = 0.0, max_weight: float = 0.25) -> np.ndarray:
    mu, cov = _prepare(mu, cov)

    def negative_sharpe(w):
        excess = w @ mu - risk_free
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return -excess / vol

    return _solve(negative_sharpe, len(mu), max_weight)


def min_variance(cov, max_weight: float = 0.25) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    return _solve(lambda w: w @ cov @ w, len(cov), max_weight)


def equal_weight(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def portfolio_stats(w, mu, cov, risk_free: float = 0.0) -> dict:
    w, (mu, cov) = np.asarray(w, dtype=float), _prepare(mu, cov)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return {
        "expected_return": ret,
        "volatility": vol,
        "sharpe": (ret - risk_free) / vol if vol > 0 else 0.0,
        "concentration": float(np.sum(w ** 2)),  # Herfindahl; 1/n is fully diversified
        "max_weight": float(w.max()),
        "n_holdings": int((w > 1e-4).sum()),
    }
