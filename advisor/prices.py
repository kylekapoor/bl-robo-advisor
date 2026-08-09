"""Price ingestion and the covariance estimate everything else depends on."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "prices.parquet"

TRADING_DAYS = 252

# A deliberately boring, liquid, sector-spread universe. Nothing here is a bet
# on the universe selection itself.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # tech
    "JPM", "BAC", "GS",                         # financials
    "JNJ", "UNH", "PFE",                        # healthcare
    "XOM", "CVX",                               # energy
    "PG", "KO", "WMT",                          # staples
    "CAT", "BA",                                # industrials
    "NEE", "LIN",                               # utilities / materials
]
BENCHMARK = "SPY"


def download(
    tickers=None, start="2015-01-01", end=None, cache: Path | None = CACHE
) -> pd.DataFrame:
    """Daily adjusted closes. Cached, because yfinance is rate-limited and rude."""
    tickers = list(tickers or DEFAULT_UNIVERSE)
    symbols = sorted(set(tickers) | {BENCHMARK})

    if cache and cache.exists():
        cached = pd.read_parquet(cache)
        if set(symbols).issubset(cached.columns):
            sliced = cached.loc[start:end, symbols]
            if not sliced.empty:
                return sliced

    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw.to_frame()
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if prices.empty:
        raise RuntimeError("yfinance returned nothing -- check network or ticker list")

    if cache:
        cache.parent.mkdir(exist_ok=True)
        prices.to_parquet(cache)
    return prices


def returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods).dropna(how="all")


def shrunk_covariance(rets: pd.DataFrame, shrinkage: float = 0.15) -> pd.DataFrame:
    """Sample covariance pulled toward its diagonal.

    The sample covariance of 20 assets from a few hundred observations is
    ill-conditioned, and a mean-variance optimiser will happily exploit the
    noise in its smallest eigenvalues. Shrinking toward the diagonal costs a
    little accuracy and removes most of that failure mode.

    ponytail: fixed shrinkage rather than Ledoit-Wolf's analytic intensity.
    Swap in sklearn.covariance.LedoitWolf if the constant ever starts mattering.
    """
    sample = rets.cov()
    target = np.diag(np.diag(sample.to_numpy()))
    blended = (1 - shrinkage) * sample.to_numpy() + shrinkage * target
    return pd.DataFrame(blended, index=sample.index, columns=sample.columns)


def annualise_cov(daily_cov: pd.DataFrame) -> pd.DataFrame:
    return daily_cov * TRADING_DAYS


def market_weights(tickers) -> np.ndarray:
    """Equilibrium weights for the Black-Litterman prior.

    ponytail: equal weight, not market cap. yfinance only exposes *current*
    market cap, and using today's caps to weight a 2016 portfolio is look-ahead
    bias -- the exact error this project is supposed to be careful about. Equal
    weight is the honest, if blunt, alternative. Wire in a point-in-time cap
    series (CRSP, Sharadar) to do this properly.
    """
    return np.repeat(1.0 / len(tickers), len(tickers))
