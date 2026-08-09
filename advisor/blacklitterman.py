"""Black-Litterman posterior.

Plain mean-variance optimisation is famously unusable on real return estimates:
sample means are mostly noise, and the optimiser responds by putting 90% of the
book in whichever asset got lucky. Black-Litterman fixes this by starting from
the returns the market itself implies and only moving away from them in the
specific directions you have an opinion about, weighted by how sure you are.

That structure is what makes an LLM safe to attach here. The model does not
choose portfolio weights. It produces opinions, each with a confidence, and the
maths decides how much they are allowed to matter. A garbage view moves the
book a little; no views at all leaves you holding the market.

Reference: He & Litterman (1999), "The Intuition Behind Black-Litterman Model
Portfolios".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# How strongly the prior is trusted. Small tau means "the equilibrium is close
# to right"; the literature uses 0.01-0.05.
DEFAULT_TAU = 0.05
# Implied risk aversion of the representative investor. delta = (E[r]-rf)/sigma^2
# for the market portfolio, which historically lands near 2.5.
DEFAULT_RISK_AVERSION = 2.5


def equilibrium_returns(
    cov: np.ndarray, market_weights: np.ndarray, risk_aversion: float = DEFAULT_RISK_AVERSION
) -> np.ndarray:
    """Reverse-optimised excess returns: what the market must expect to hold w."""
    return risk_aversion * cov @ market_weights


def default_omega(cov: np.ndarray, P: np.ndarray, tau: float = DEFAULT_TAU) -> np.ndarray:
    """View uncertainty proportional to the prior variance of each view.

    This is the He-Litterman choice. It has the useful property that a view on a
    volatile spread is automatically treated as less certain than the same
    nominal view on a stable one.
    """
    return np.diag(np.diag(P @ (tau * cov) @ P.T))


@dataclass
class Posterior:
    mu: np.ndarray
    cov: np.ndarray
    prior_mu: np.ndarray
    n_views: int

    @property
    def tilt(self) -> np.ndarray:
        """How far each asset's expected return moved from equilibrium."""
        return self.mu - self.prior_mu


def posterior(
    cov: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
) -> Posterior:
    """Combine equilibrium returns `pi` with views `P @ mu = Q`.

    With no views this returns the prior unchanged, which is the whole safety
    story: an LLM that emits nothing leaves you holding the market portfolio.
    """
    cov = np.asarray(cov, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1)
    n = len(pi)

    if P is None or Q is None or len(np.atleast_2d(P)) == 0 or np.size(Q) == 0:
        return Posterior(mu=pi.copy(), cov=cov.copy(), prior_mu=pi.copy(), n_views=0)

    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).reshape(-1)
    if P.shape != (len(Q), n):
        raise ValueError(f"P must be {(len(Q), n)}, got {P.shape}")

    if omega is None:
        omega = default_omega(cov, P, tau)
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    # A view with zero uncertainty would be treated as gospel and blow up the
    # inverse; floor it.
    omega = omega + np.eye(len(Q)) * 1e-10

    tau_cov_inv = np.linalg.pinv(tau * cov)
    omega_inv = np.linalg.pinv(omega)

    precision = tau_cov_inv + P.T @ omega_inv @ P
    cov_post_scaled = np.linalg.pinv(precision)
    mu = cov_post_scaled @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

    return Posterior(mu=mu, cov=cov + cov_post_scaled, prior_mu=pi, n_views=len(Q))


def views_to_matrices(views, tickers) -> tuple:
    """Turn structured views into the (P, Q, confidence) triple BL needs.

    Absolute view:  "AAPL returns 3% over the horizon"    -> one 1 in the row
    Relative view:  "AAPL beats MSFT by 2%"               -> +1 and -1
    Views naming unknown tickers are dropped, not guessed at.
    """
    index = {t: i for i, t in enumerate(tickers)}
    rows, q, conf = [], [], []
    for v in views:
        if v.asset not in index:
            continue
        if v.versus is not None and v.versus not in index:
            continue
        row = np.zeros(len(tickers))
        row[index[v.asset]] = 1.0
        if v.versus is not None:
            row[index[v.versus]] = -1.0
        rows.append(row)
        q.append(v.expected_return)
        conf.append(v.confidence)
    if not rows:
        return None, None, None
    return np.vstack(rows), np.array(q), np.array(conf)


def omega_from_confidence(
    cov: np.ndarray, P: np.ndarray, confidence: np.ndarray, tau: float = DEFAULT_TAU
) -> np.ndarray:
    """Scale the default view uncertainty by stated confidence in (0, 1].

    confidence -> 1 means "trust this view as much as the prior"; confidence
    near 0 widens omega until the view barely registers.
    """
    base = np.diag(default_omega(cov, P, tau))
    confidence = np.clip(np.asarray(confidence, dtype=float), 0.01, 1.0)
    return np.diag(base / confidence)
