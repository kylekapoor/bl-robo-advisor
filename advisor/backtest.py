"""Walk-forward backtest with an explicit control arm.

The obvious way to evaluate "LLM picks stocks" is to run it and look at the
return. That number is worthless on its own, for two reasons:

1. The model knows the future. It was trained on text written after the period
   being tested. Restricting its *inputs* to filings dated before each rebalance
   removes the mechanical leak but not what the weights already encode.
2. Black-Litterman with sane constraints is a decent portfolio construction
   method by itself. Some of any outperformance is the machinery, not the views.

So every run reports several arms on identical data, costs and constraints. The
only interesting question is whether `llm` beats `equilibrium` and `shuffled`,
and `shuffled` is the one that matters: it takes the real views and permutes
which asset each applies to, destroying the information while keeping the count,
magnitudes and confidences identical. If `llm` cannot beat its own shuffle, the
views carry nothing and the run is a demonstration of plumbing.

`lstm` and `minvar` are a matched pair on the same principle: identical
objective and constraints, differing only in whether the covariance uses
forecast or trailing volatilities. Any gap between them belongs to the
volatility model.

One structural note worth knowing before reading the table: `equilibrium` and
`equal` are the *same portfolio*, not a coincidence. Reverse optimisation is
self-inverting -- max_sharpe(delta @ Sigma @ w_mkt, Sigma) returns w_mkt for any
Sigma -- and `market_weights` is equal weight here to avoid look-ahead bias. So
the equilibrium arm is equal weight wearing a hat, and it is the honest baseline
the view-driven arms have to beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import blacklitterman as bl
from . import optimise, prices as px, volatility

TRADING_DAYS = px.TRADING_DAYS
# One-way cost per unit of turnover. 10 bps is generous for liquid US large caps
# and deliberately not zero -- a strategy that only wins before costs has lost.
COST_PER_TURNOVER = 0.0010
COV_WINDOW_DAYS = 504  # two years of daily returns


@dataclass
class Result:
    name: str
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    views_used: list = field(default_factory=list)
    vol_log: list = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    def since(self, start) -> "Result":
        """The same run, measured only from `start` onward.

        The portfolio still walks forward continuously through the whole period;
        this only changes the window the statistics are computed over. That is
        what makes a held-out evaluation honest: the strategy is not restarted
        with hindsight, it is simply scored on a stretch that was never used to
        choose anything.
        """
        cut = pd.Timestamp(start)
        equity = self.equity.loc[self.equity.index >= cut]
        if equity.empty:
            return self
        return Result(
            name=self.name,
            equity=equity / equity.iloc[0],
            weights=self.weights.loc[self.weights.index >= cut],
            turnover=self.turnover.loc[self.turnover.index >= cut],
            views_used=self.views_used,
            vol_log=self.vol_log,
        )

    def stats(self, benchmark: pd.Series | None = None) -> dict:
        r = self.returns
        years = len(r) / TRADING_DAYS
        total = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
        vol = r.std() * np.sqrt(TRADING_DAYS)
        peak = self.equity.cummax()
        out = {
            "total_return": total,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": cagr / vol if vol > 0 else 0.0,
            "max_drawdown": float((self.equity / peak - 1).min()),
            "avg_turnover": float(self.turnover.mean()),
            "n_views": int(sum(len(v) for v in self.views_used)),
        }
        if benchmark is not None:
            aligned = benchmark.reindex(r.index).pct_change().dropna()
            active = (r - aligned).dropna()
            te = active.std() * np.sqrt(TRADING_DAYS)
            out["tracking_error"] = te
            out["information_ratio"] = (active.mean() * TRADING_DAYS / te) if te > 0 else 0.0
        return out


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "QE") -> list:
    """Last trading day of each period, excluding any without enough history."""
    marks = pd.Series(index=index, data=1).resample(freq).last().index
    return [index[index <= m][-1] for m in marks if (index <= m).sum() > COV_WINDOW_DAYS]


def _weights_for(
    arm: str,
    rets_window: pd.DataFrame,
    tickers: list,
    view_batch=None,
    max_weight: float = 0.25,
    rng: np.random.Generator | None = None,
    vol_log: list | None = None,
):
    """Portfolio weights for one arm at one rebalance date."""
    n = len(tickers)
    if arm == "equal":
        return optimise.equal_weight(n), []

    cov_daily = px.shrunk_covariance(rets_window[tickers])

    if arm == "lstm":
        # Keep the correlations from history, replace the variances with a
        # forecast. Variance is what moves; correlation is comparatively stable,
        # and forecasting a full 20x20 matrix would need far more data.
        result = volatility.forecast(rets_window[tickers])
        annual = px.annualise_cov(cov_daily)
        forecast_cov = volatility.rebuild_covariance(annual, result.volatility).to_numpy()
        if vol_log is not None:
            vol_log.append({"source": result.source, "val_mae": result.val_mae,
                            "baseline_mae": result.baseline_mae})

        # Minimum variance, not max-Sharpe on reverse-optimised returns.
        #
        # The first version did the latter and produced weights *identical* to
        # the equilibrium arm at every rebalance -- because reverse optimisation
        # is self-inverting: max_sharpe(delta @ Sigma @ w_mkt, Sigma) returns
        # w_mkt for ANY Sigma. The covariance cancels, so the forecast could not
        # possibly matter and the arm measured nothing at all. Three arms came
        # back byte-identical, which is what gave it away.
        #
        # Minimum variance depends on Sigma and nothing else, so a better
        # variance estimate is exactly what it can exploit. That is also where
        # volatility forecasting earns its keep in practice.
        return optimise.min_variance(forecast_cov, max_weight=max_weight), []

    cov = px.annualise_cov(cov_daily).to_numpy()

    if arm == "minvar":
        # The control for `lstm`. Same objective, same constraints, trailing
        # sample covariance instead of a forecast one -- so any difference
        # between the two arms is attributable to the volatility model and
        # nothing else.
        return optimise.min_variance(cov, max_weight=max_weight), []

    w_mkt = px.market_weights(tickers)
    pi = bl.equilibrium_returns(cov, w_mkt)

    views = list(view_batch.views) if view_batch else []
    if arm == "equilibrium":
        views = []
    elif arm == "shuffled" and views:
        # Keep everything except which assets the view is *about*: same count,
        # same magnitudes, same confidences, and crucially the same relative-vs-
        # absolute structure. Collapsing relative views to absolute ones here
        # would make the control a different kind of portfolio rather than the
        # same portfolio with the information removed, and the comparison would
        # no longer isolate the model's stock selection.
        rng = rng or np.random.default_rng(0)
        shuffled = []
        for view in views:
            if view.versus is not None:
                a, b = rng.choice(tickers, size=2, replace=False)
                shuffled.append(view.model_copy(update={"asset": str(a), "versus": str(b)}))
            else:
                a = rng.choice(tickers)
                shuffled.append(view.model_copy(update={"asset": str(a)}))
        views = shuffled

    P, Q, conf = bl.views_to_matrices(views, tickers)
    omega = bl.omega_from_confidence(cov, P, conf) if P is not None else None
    post = bl.posterior(cov, pi, P, Q, omega)

    w = optimise.max_sharpe(post.mu, post.cov, max_weight=max_weight)
    return w, views


def run(
    prices_df: pd.DataFrame,
    arm: str = "equilibrium",
    tickers: list | None = None,
    view_provider=None,
    max_weight: float = 0.25,
    freq: str = "QE",
    seed: int = 0,
) -> Result:
    """Walk forward through `prices_df`, rebalancing on `freq`.

    `view_provider(as_of, tickers) -> ViewBatch | None` is called once per
    rebalance and only ever sees the date, never future prices.
    """
    tickers = list(tickers or [c for c in prices_df.columns if c != px.BENCHMARK])
    rets = prices_df.pct_change().dropna()
    dates = rebalance_dates(rets.index, freq)
    if not dates:
        raise ValueError("not enough history for a single rebalance")

    rng = np.random.default_rng(seed)
    equity, weight_log, turnover_log, views_log, vol_log = [], {}, {}, [], []
    value, current = 1.0, np.zeros(len(tickers))

    equity_index, equity_values = [], []
    for i, start in enumerate(dates):
        window = rets.loc[:start].tail(COV_WINDOW_DAYS)
        batch = view_provider(start.date(), tickers) if view_provider else None

        target, used = _weights_for(arm, window, tickers, batch, max_weight, rng,
                                    vol_log=vol_log)
        views_log.append([v.model_dump() for v in used])

        turnover = float(np.abs(target - current).sum())
        value *= 1 - turnover * COST_PER_TURNOVER
        current = target
        weight_log[start] = pd.Series(target, index=tickers)
        turnover_log[start] = turnover

        stop = dates[i + 1] if i + 1 < len(dates) else rets.index[-1]
        held = rets.loc[start:stop, tickers].iloc[1:]
        for day, row in held.iterrows():
            value *= 1 + float(row.to_numpy() @ current)
            equity_index.append(day)
            equity_values.append(value)

    return Result(
        name=arm,
        equity=pd.Series(equity_values, index=pd.DatetimeIndex(equity_index)),
        weights=pd.DataFrame(weight_log).T,
        turnover=pd.Series(turnover_log),
        views_used=views_log,
        vol_log=vol_log,
    )


def benchmark_equity(prices_df: pd.DataFrame, like: pd.Series) -> pd.Series:
    """Buy-and-hold SPY on the same calendar, for a fair comparison."""
    spy = prices_df[px.BENCHMARK].reindex(like.index).ffill()
    return spy / spy.iloc[0]


def compare(results: list, benchmark: pd.Series) -> pd.DataFrame:
    rows = {r.name: r.stats(benchmark) for r in results}
    bench = Result("SPY", benchmark, pd.DataFrame(), pd.Series(dtype=float))
    rows["SPY"] = bench.stats()
    return pd.DataFrame(rows).T
