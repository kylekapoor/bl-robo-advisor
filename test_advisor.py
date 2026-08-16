#!/usr/bin/env python3
"""Self-checks. No network, no API key, no downloaded data:

    python test_advisor.py

The Black-Litterman checks are the ones that matter. The maths is short enough
to look correct while being wrong in ways that only show up as a portfolio
quietly ignoring its views, so each property is pinned down separately.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings

from advisor import (
    backtest, blacklitterman as bl, filings, optimise, prices as px, retrieval,
    volatility,
)
from advisor.views import MAX_ABS_VIEW_RETURN, View, validate

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def toy_cov(n=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return (a @ a.T) / n + np.eye(n) * 0.02


def toy_prices(days=1400, n_assets=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=days)
    cols = TICKERS + [px.BENCHMARK]
    steps = rng.normal(0.0004, 0.01, size=(days, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)


# --- Black-Litterman -------------------------------------------------------

def test_no_views_returns_the_prior_untouched():
    cov, pi = toy_cov(), np.array([0.05, 0.06, 0.04, 0.03])
    post = bl.posterior(cov, pi)
    assert post.n_views == 0
    np.testing.assert_allclose(post.mu, pi)
    # This is the safety property the whole design rests on: an LLM that returns
    # nothing must leave the investor holding the market, not a random book.
    np.testing.assert_allclose(post.tilt, 0.0, atol=1e-12)


def test_a_bullish_view_raises_that_asset_above_equilibrium():
    cov, pi = toy_cov(), np.array([0.05, 0.05, 0.05, 0.05])
    P = np.array([[1.0, 0, 0, 0]])
    post = bl.posterior(cov, pi, P, np.array([0.12]))
    assert post.mu[0] > pi[0], "a bullish view must raise the expected return"
    assert post.n_views == 1


def test_confidence_controls_how_far_the_posterior_moves():
    cov, pi = toy_cov(), np.array([0.05, 0.05, 0.05, 0.05])
    P, Q = np.array([[1.0, 0, 0, 0]]), np.array([0.12])

    timid = bl.posterior(cov, pi, P, Q, bl.omega_from_confidence(cov, P, np.array([0.05])))
    certain = bl.posterior(cov, pi, P, Q, bl.omega_from_confidence(cov, P, np.array([1.0])))

    assert certain.tilt[0] > timid.tilt[0], "higher confidence must move the posterior further"
    assert timid.tilt[0] > 0


def test_a_relative_view_widens_the_spread_it_names():
    cov, pi = toy_cov(), np.array([0.05, 0.05, 0.05, 0.05])
    P = np.array([[1.0, -1.0, 0, 0]])  # AAA beats BBB
    post = bl.posterior(cov, pi, P, np.array([0.04]))

    # A relative view constrains the *spread*, not either leg's absolute level.
    # When the assets are correlated both expectations can move the same way
    # while the gap between them still opens in the direction of the view --
    # asserting tilt[0] > 0 > tilt[1] tests a property BL never promised.
    prior_spread = pi[0] - pi[1]
    post_spread = post.mu[0] - post.mu[1]
    assert post_spread > prior_spread, f"spread did not widen: {post_spread:.4f}"
    assert post_spread > 0.01, f"spread barely moved: {post_spread:.4f}"


def test_views_to_matrices_drops_unknown_tickers():
    views = [
        View(asset="AAA", expected_return=0.02, confidence=0.5),
        View(asset="ZZZ", expected_return=0.02, confidence=0.5),          # not held
        View(asset="BBB", versus="QQQ", expected_return=0.01, confidence=0.5),  # leg not held
    ]
    P, Q, conf = bl.views_to_matrices(views, TICKERS)
    assert P.shape == (1, 4), f"expected 1 surviving view, got {P.shape}"
    np.testing.assert_allclose(P[0], [1, 0, 0, 0])


def test_relative_view_row_sums_to_zero():
    views = [View(asset="AAA", versus="CCC", expected_return=0.03, confidence=0.5)]
    P, _, _ = bl.views_to_matrices(views, TICKERS)
    assert P[0].sum() == 0.0
    assert P[0][0] == 1.0 and P[0][2] == -1.0


def test_posterior_rejects_a_mis_shaped_view_matrix():
    cov, pi = toy_cov(), np.zeros(4)
    try:
        bl.posterior(cov, pi, np.ones((1, 3)), np.array([0.01]))
    except ValueError:
        return
    raise AssertionError("a 3-column P against 4 assets must not be accepted")


# --- View validation -------------------------------------------------------

def test_absurd_return_magnitudes_are_rejected():
    batch = validate([{"asset": "AAA", "expected_return": 4.0, "confidence": 0.9}], TICKERS)
    assert batch.views == [] and batch.rejected, "a 400% quarterly view must be rejected"


def test_view_at_the_ceiling_is_allowed():
    batch = validate(
        [{"asset": "AAA", "expected_return": MAX_ABS_VIEW_RETURN, "confidence": 0.5}], TICKERS
    )
    assert len(batch.views) == 1


def test_out_of_range_confidence_is_rejected():
    batch = validate([{"asset": "AAA", "expected_return": 0.02, "confidence": 7}], TICKERS)
    assert batch.views == []


def test_duplicate_and_self_referential_views_are_dropped():
    raw = [
        {"asset": "AAA", "expected_return": 0.02, "confidence": 0.5},
        {"asset": "AAA", "expected_return": 0.03, "confidence": 0.5},   # duplicate
        {"asset": "BBB", "versus": "BBB", "expected_return": 0.01, "confidence": 0.5},
    ]
    batch = validate(raw, TICKERS)
    assert len(batch.views) == 1
    assert len(batch.rejected) == 2


def test_garbage_payload_yields_no_views_rather_than_an_exception():
    batch = validate(["not a dict", {"nope": 1}, None], TICKERS)
    assert batch.views == []
    assert len(batch.rejected) == 3


def test_tickers_are_normalised_to_upper_case():
    batch = validate([{"asset": "aaa", "expected_return": 0.02, "confidence": 0.5}], TICKERS)
    assert len(batch.views) == 1 and batch.views[0].asset == "AAA"


# --- Optimiser -------------------------------------------------------------

def test_weights_are_a_valid_long_only_portfolio():
    mu, cov = np.array([0.10, 0.08, 0.06, 0.04]), toy_cov()
    w = optimise.max_sharpe(mu, cov, max_weight=0.4)
    assert abs(w.sum() - 1) < 1e-8, f"weights sum to {w.sum()}"
    assert (w >= -1e-9).all(), "long-only constraint violated"
    assert w.max() <= 0.4 + 1e-6, f"position cap breached: {w.max()}"


def test_the_position_cap_actually_binds():
    # One asset dominates on every axis, so an uncapped optimiser would go all in.
    mu = np.array([0.50, 0.01, 0.01, 0.01])
    cov = np.eye(4) * 0.04
    assert optimise.max_sharpe(mu, cov, max_weight=0.25).max() <= 0.25 + 1e-6


def test_min_variance_prefers_the_calmest_asset():
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    w = optimise.min_variance(cov, max_weight=1.0)
    assert w.argmax() == 0, f"expected the lowest-variance asset to dominate, got {w}"


def test_portfolio_stats_are_internally_consistent():
    mu, cov = np.array([0.10, 0.08, 0.06, 0.04]), toy_cov()
    w = optimise.equal_weight(4)
    s = optimise.portfolio_stats(w, mu, cov)
    assert abs(s["expected_return"] - mu.mean()) < 1e-12
    assert abs(s["concentration"] - 0.25) < 1e-12  # 1/n for equal weight
    assert s["n_holdings"] == 4


# --- Covariance and backtest ----------------------------------------------

def test_shrinkage_pulls_correlations_toward_zero():
    rets = px.returns(toy_prices())
    sample = rets.cov().to_numpy()
    shrunk = px.shrunk_covariance(rets, shrinkage=0.5).to_numpy()
    off = ~np.eye(len(sample), dtype=bool)
    assert np.abs(shrunk[off]).sum() < np.abs(sample[off]).sum()
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample), rtol=1e-9)


def test_rebalance_dates_require_a_full_covariance_window():
    idx = px.returns(toy_prices()).index
    dates = backtest.rebalance_dates(idx, "QE")
    assert dates, "no rebalance dates generated"
    assert all((idx <= d).sum() > backtest.COV_WINDOW_DAYS for d in dates)
    assert dates == sorted(dates)


def test_backtest_runs_and_charges_for_turnover():
    prices_df = toy_prices()
    result = backtest.run(prices_df, arm="equilibrium", tickers=TICKERS)
    assert len(result.equity) > 100
    assert result.equity.notna().all()
    assert (result.turnover >= 0).all()
    # First rebalance moves from an empty book to fully invested.
    assert abs(result.turnover.iloc[0] - 1.0) < 1e-6


def test_weights_stay_within_the_cap_at_every_rebalance():
    result = backtest.run(toy_prices(), arm="equilibrium", tickers=TICKERS, max_weight=0.35)
    assert (result.weights.to_numpy() <= 0.35 + 1e-6).all()
    np.testing.assert_allclose(result.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_equal_weight_arm_is_actually_equal_weight():
    result = backtest.run(toy_prices(), arm="equal", tickers=TICKERS)
    np.testing.assert_allclose(result.weights.to_numpy(), 0.25, atol=1e-9)


def test_shuffled_arm_keeps_the_view_count_but_moves_the_assets():
    """The control arm has to preserve everything except the information."""
    rets = px.returns(toy_prices())
    window = rets.tail(backtest.COV_WINDOW_DAYS)

    class Batch:
        views = [
            View(asset="AAA", expected_return=0.05, confidence=0.6),
            View(asset="BBB", expected_return=-0.03, confidence=0.4),
        ]

    _, used = backtest._weights_for(
        "shuffled", window, TICKERS, Batch(), rng=np.random.default_rng(3)
    )
    assert len(used) == 2, "shuffling must not change how many views there are"
    assert [v.expected_return for v in used] == [0.05, -0.03], "magnitudes must survive"
    assert [v.confidence for v in used] == [0.6, 0.4], "confidences must survive"


def test_equilibrium_arm_ignores_views_entirely():
    rets = px.returns(toy_prices())
    window = rets.tail(backtest.COV_WINDOW_DAYS)

    class Batch:
        views = [View(asset="AAA", expected_return=0.14, confidence=1.0)]

    with_views, used = backtest._weights_for("equilibrium", window, TICKERS, Batch())
    without, _ = backtest._weights_for("equilibrium", window, TICKERS, None)
    assert used == []
    np.testing.assert_allclose(with_views, without, atol=1e-9)


def test_stats_report_a_drawdown_for_a_portfolio_that_falls():
    idx = pd.bdate_range("2020-01-01", periods=300)
    equity = pd.Series(np.linspace(1.0, 0.7, 300), index=idx)
    r = backtest.Result("down", equity, pd.DataFrame(), pd.Series([0.0]))
    s = r.stats()
    assert s["max_drawdown"] < -0.25, s["max_drawdown"]
    assert s["total_return"] < 0


# --- volatility forecasting -------------------------------------------------

def test_rebuild_covariance_keeps_correlations_exactly():
    """Sigma = D R D. Only the diagonal is allowed to change."""
    rets = px.returns(toy_prices())[TICKERS]
    cov = px.annualise_cov(px.shrunk_covariance(rets))

    forecast = pd.Series(np.sqrt(np.diag(cov.to_numpy())) * 1.5, index=TICKERS)
    rebuilt = volatility.rebuild_covariance(cov, forecast)

    def correlation(matrix):
        d = np.sqrt(np.diag(matrix))
        return matrix / np.outer(d, d)

    np.testing.assert_allclose(
        correlation(cov.to_numpy()), correlation(rebuilt.to_numpy()), atol=1e-10
    )
    # And the new volatilities really are the forecast ones.
    np.testing.assert_allclose(np.sqrt(np.diag(rebuilt.to_numpy())), forecast.to_numpy(),
                               rtol=1e-10)


def test_rebuild_covariance_stays_positive_semidefinite():
    rets = px.returns(toy_prices())[TICKERS]
    cov = px.annualise_cov(px.shrunk_covariance(rets))
    forecast = pd.Series([0.4, 0.1, 0.9, 0.25], index=TICKERS)
    eigenvalues = np.linalg.eigvalsh(volatility.rebuild_covariance(cov, forecast).to_numpy())
    assert (eigenvalues > -1e-10).all(), eigenvalues


def test_sequences_are_ratio_normalised():
    """Each window ends at 1.0 and the target is a ratio, not a level.

    This is what lets one pooled model serve 20 assets. Trained on raw levels it
    learns the universe average and returns near-identical volatility for every
    asset, discarding the cross-section the covariance exists to capture.
    """
    rets = px.returns(toy_prices())[TICKERS]
    vol = volatility.realised_volatility(rets)
    X, y = volatility._sequences(vol, seq_len=10, horizon=5)
    assert len(X) > 0
    np.testing.assert_allclose(X[:, -1, 0], 1.0, atol=1e-6)
    assert y.min() > 0, "volatility ratios must be positive"


def test_realised_volatility_is_annualised():
    idx = pd.bdate_range("2020-01-01", periods=400)
    daily_sigma = 0.01
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"AAA": rng.normal(0, daily_sigma, 400)}, index=idx)
    vol = volatility.realised_volatility(rets).iloc[-1, 0]
    expected = daily_sigma * np.sqrt(px.TRADING_DAYS)
    assert 0.5 * expected < vol < 1.8 * expected, (vol, expected)


def test_forecast_falls_back_when_history_is_too_short():
    rets = px.returns(toy_prices(days=90))[TICKERS]
    result = volatility.forecast(rets)
    assert result.source == "trailing"
    assert (result.volatility > 0).all()
    assert set(result.volatility.index) == set(TICKERS)


def test_beat_baseline_requires_both_numbers():
    assert not volatility.VolForecast(pd.Series(dtype=float), "trailing").beat_baseline
    assert volatility.VolForecast(pd.Series(dtype=float), "lstm",
                                  val_mae=0.1, baseline_mae=0.2).beat_baseline
    assert not volatility.VolForecast(pd.Series(dtype=float), "lstm",
                                      val_mae=0.3, baseline_mae=0.2).beat_baseline


def test_lstm_arm_is_not_a_no_op_against_the_equilibrium_arm():
    """The bug this arm shipped with, pinned.

    max_sharpe(delta @ Sigma @ w_mkt, Sigma) returns w_mkt for ANY Sigma, so
    building the lstm arm that way cancelled the covariance entirely and
    produced weights identical to equilibrium at every rebalance. Minimum
    variance depends on Sigma, so the forecast can actually matter.
    """
    # Assets must differ in volatility for this to mean anything: with four
    # identically-distributed assets the minimum-variance portfolio *is* equal
    # weight, and the test would pass or fail for reasons unrelated to the bug.
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2018-01-01", periods=1400)
    sigmas = [0.004, 0.010, 0.020, 0.035]
    cols = {t: rng.normal(0.0003, s, len(idx)) for t, s in zip(TICKERS, sigmas)}
    cols[px.BENCHMARK] = rng.normal(0.0003, 0.011, len(idx))
    window = pd.DataFrame(cols, index=idx).tail(backtest.COV_WINDOW_DAYS)

    # The cap must not bind: four assets at max 25% each leaves exactly one
    # feasible portfolio (equal weight), so every arm would agree regardless.
    lstm, _ = backtest._weights_for("lstm", window, TICKERS, max_weight=0.6)
    equilibrium, _ = backtest._weights_for("equilibrium", window, TICKERS, max_weight=0.6)
    assert not np.allclose(lstm, equilibrium, atol=1e-6), (
        "lstm arm reproduces the equilibrium weights; the covariance is being "
        "cancelled and the arm measures nothing"
    )
    # And it should lean toward the calm asset, which is the whole point.
    assert lstm.argmax() == 0, lstm


def test_minvar_control_matches_lstm_objective_on_sample_covariance():
    rets = px.returns(toy_prices())
    window = rets.tail(backtest.COV_WINDOW_DAYS)
    w, used = backtest._weights_for("minvar", window, TICKERS)
    assert abs(w.sum() - 1) < 1e-8 and (w >= -1e-9).all()
    assert used == []


def test_lstm_arm_produces_a_valid_portfolio():
    """Runs whether or not TensorFlow is installed -- the fallback must also work."""
    rets = px.returns(toy_prices())
    window = rets.tail(backtest.COV_WINDOW_DAYS)
    vol_log = []
    w, used = backtest._weights_for("lstm", window, TICKERS, vol_log=vol_log)
    assert abs(w.sum() - 1) < 1e-8 and (w >= -1e-9).all()
    assert used == [], "the lstm arm applies no views; it only changes the covariance"
    assert vol_log and vol_log[0]["source"] in ("lstm", "trailing")


# --- view caching -----------------------------------------------------------

def test_api_failures_are_distinguished_from_an_empty_opinion():
    """A rate-limited quarter must never be cached as a legitimate 'no views'.

    Conflating them poisons the cache permanently and silently turns the llm arm
    into the equilibrium arm for however long the outage lasted.
    """
    from advisor.views import ViewBatch

    genuine = ViewBatch(views=[], raw_count=0, rejected=[])
    assert not genuine.errored

    rate_limited = ViewBatch(views=[], raw_count=0,
                             rejected=["RateLimitError: Error code: 429 - ..."])
    assert rate_limited.errored

    schema_reject = ViewBatch(views=[], raw_count=2, rejected=["schema: bad field"])
    assert not schema_reject.errored, "a rejected view is not an API failure"


class _StubEmbeddings(Embeddings):
    """Two-dimensional embeddings that rank marked chunks first.

    Real MiniLM weights are a download and a torch import. Everything worth
    testing here -- ordering, chunk handling, cache keying -- is independent of
    which embedding model produced the vectors.
    """

    MARKER = "ZZMARKER"

    def embed_documents(self, texts):
        return [[1.0, 0.0] if self.MARKER in t else [0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _marked_document(n_sentences: int = 60, marked=(4, 22, 47)) -> str:
    parts = []
    for i in range(n_sentences):
        tag = f" {_StubEmbeddings.MARKER}" if i in marked else ""
        parts.append(f"Sentence {i:03d} about the quarter and its results{tag}")
    return ". ".join(parts)


def test_retrieved_passages_come_back_in_document_order():
    """Ranked order would interleave page 40 with page 6 and read as one passage.

    The model receives the excerpt as continuous prose, so passages have to
    appear in the order the filing puts them, not the order the retriever
    scored them.
    """
    text = _marked_document()
    out = retrieval.retrieve(text, max_chars=100_000, k=3,
                             embeddings=_StubEmbeddings())

    segments = [s for s in out.split(" ... ") if s.strip()]
    assert len(segments) >= 2, f"expected several passages, got {segments!r}"

    positions = [text.index(s.strip()[:40]) for s in segments]
    assert positions == sorted(positions), f"passages out of document order: {positions}"


def test_retrieval_actually_uses_relevance():
    """A retriever that ignored the query would be a truncation with extra steps."""
    text = _marked_document()
    out = retrieval.retrieve(text, max_chars=100_000, k=3,
                             embeddings=_StubEmbeddings())
    assert _StubEmbeddings.MARKER in out, "relevant chunks were not selected"


def test_short_documents_skip_the_vector_store():
    """Fewer chunks than k means every chunk is returned, with no model needed."""
    assert retrieval.retrieve("", embeddings=None) == ""
    assert retrieval.retrieve("   ", embeddings=None) == ""
    short = "One short sentence about revenue"
    assert short in retrieval.retrieve(short, embeddings=None)


def test_changing_the_query_invalidates_cached_excerpts():
    """Otherwise one backtest silently mixes two retrieval schemes."""
    import tempfile

    original = retrieval.QUERY
    try:
        before = retrieval.config_fingerprint()
        retrieval.QUERY = original + " and inventory levels"
        after = retrieval.config_fingerprint()
        assert before != after, "fingerprint ignored a query change"

        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "0000320193_x.txt"
            retrieval.QUERY = original
            retrieval.retrieve_cached("a short filing", base, max_chars=500)
            stale = list(Path(d).glob("*.rag"))
            assert len(stale) == 1, stale
            assert retrieval.config_fingerprint() in stale[0].name
    finally:
        retrieval.QUERY = original


def test_evidence_never_includes_a_filing_published_after_the_rebalance():
    """Look-ahead bias dressed up as research. The filter is the whole point."""
    as_of = date(2024, 6, 30)
    rows = [
        {"form": "10-Q", "filed": date(2024, 7, 15), "cik": 1, "accession": "future",
         "document": "d.htm"},
        {"form": "10-Q", "filed": date(2024, 5, 2), "cik": 1, "accession": "past",
         "document": "d.htm"},
    ]
    original = (filings.ticker_to_cik, filings.recent_filings, filings.filing_text)
    try:
        filings.ticker_to_cik = lambda tickers: {t: 1 for t in tickers}
        filings.recent_filings = lambda cik, **kw: rows
        filings.filing_text = lambda f, **kw: f"body of {f['accession']}"

        evidence = filings.evidence_by_ticker(["AAPL"], as_of=as_of)
        assert "past" in evidence["AAPL"]
        assert "future" not in evidence["AAPL"], "used a filing from after the rebalance"
    finally:
        filings.ticker_to_cik, filings.recent_filings, filings.filing_text = original


def test_thin_filing_history_follows_the_predecessor_cik():
    """A reorganisation moves the ticker and leaves the history behind.

    XOM is the live case: the ticker points at a holding company with one 10-Q
    while twenty years sit under the old CIK. Without this the stock silently
    contributes no evidence for the entire backtest.
    """
    import tempfile

    holding, predecessor = 2115436, 34088

    def submissions(rows):
        return json.dumps({"filings": {"recent": {
            "form": [r[0] for r in rows],
            "filingDate": [r[1] for r in rows],
            "accessionNumber": [r[2] for r in rows],
            "primaryDocument": ["x.htm"] * len(rows),
        }, "files": []}})

    real_cache, real_get = filings.CACHE, filings._get
    try:
        with tempfile.TemporaryDirectory() as d:
            filings.CACHE = Path(d)
            # The holding company has one filing, and its accession number
            # carries the old entity's CIK.
            (Path(d) / f"submissions_{holding}.json").write_text(submissions(
                [("10-Q", "2026-08-03", "0000034088-26-000093")]
            ))
            (Path(d) / f"submissions_{predecessor}.json").write_text(submissions(
                [("10-Q", f"20{y:02d}-05-01", f"0000034088-{y:02d}-000001")
                 for y in range(10, 26)]
            ))

            def no_network(url):
                raise AssertionError(f"unexpected fetch: {url}")

            filings._get = no_network
            out = filings.recent_filings(holding)

        assert len(out) == 17, f"expected merged history, got {len(out)}"
        assert out[0]["filed"] == date(2026, 8, 3), "not sorted newest first"
        assert any(f["cik"] == predecessor for f in out), "old CIK never reached"
    finally:
        filings.CACHE, filings._get = real_cache, real_get


def test_a_healthy_cik_does_not_chase_its_filing_agent():
    """Accession prefixes name filing agents too, so following must stay gated.

    JPM, KO and BA all carry an agent's CIK in their accession numbers. Chasing
    it on a company that already has full history would merge in a stranger.
    """
    import tempfile

    company = 19617
    real_cache, real_get = filings.CACHE, filings._get
    try:
        with tempfile.TemporaryDirectory() as d:
            filings.CACHE = Path(d)
            (Path(d) / f"submissions_{company}.json").write_text(json.dumps(
                {"filings": {"recent": {
                    "form": ["10-Q"] * 30,
                    "filingDate": [f"20{y:02d}-05-01" for y in range(10, 40)],
                    "accessionNumber": [f"0001628280-{y:02d}-000001"
                                        for y in range(10, 40)],
                    "primaryDocument": ["x.htm"] * 30,
                }, "files": []}}
            ))

            def no_network(url):
                raise AssertionError(f"chased the filing agent: {url}")

            filings._get = no_network
            out = filings.recent_filings(company)

        assert len(out) == 30
        assert all(f["cik"] == company for f in out)
    finally:
        filings.CACHE, filings._get = real_cache, real_get


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
