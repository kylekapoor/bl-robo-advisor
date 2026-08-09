# bl-robo-advisor

A Black-Litterman portfolio optimiser where an LLM supplies the **views** — and
the control arms that establish those views are worth nothing.

The LLM never chooses a weight. It reads SEC filings and emits structured
opinions with confidences; the mathematics decides how much each opinion is
allowed to move the book. That separation is the whole design.

`Python` · `TensorFlow/Keras` · `Llama 3.3` · `Groq` · `yfinance` · `SEC EDGAR` · `NumPy` · `Pandas` · `SciPy` · `Pydantic`

---

## Results

Walk-forward, quarterly rebalance, 2017–2026, 20 US large caps, 10 bps
transaction costs, 25% position cap, 312 LLM views generated across 39 quarters:

| Arm | CAGR | Sharpe | Max DD | Turnover | Info. ratio |
|---|---|---|---|---|---|
| equilibrium (no views) | **21.2%** | **1.163** | −34.7% | 0.026 | 0.87 |
| **llm** | 20.6% | 1.149 | −34.3% | 0.398 | 0.63 |
| **shuffled** *(control)* | 20.7% | 1.163 | −35.5% | 0.535 | 0.65 |
| minvar *(control)* | 13.1% | 0.863 | −33.5% | 0.220 | −0.22 |
| lstm | 12.0% | 0.806 | **−26.0%** | 0.863 | −0.30 |
| SPY | 15.4% | 0.835 | −33.7% | — | — |

**The LLM views carry no information.** The `llm` arm (20.6% CAGR, 1.149 Sharpe)
is indistinguishable from `shuffled` — the same views with the asset each applies
to randomly permuted — and *loses* to holding no views at all. If real views
cannot beat their own shuffle, what is being measured is Black-Litterman's
construction, not the model's reading of a 10-Q.

Worse, the views are expensive: they drive **15× the turnover** (0.398 vs 0.026),
and at 10 bps a side that cost is most of the gap to the baseline. An LLM that
generates plausible-sounding opinions every quarter will churn a portfolio and
charge you for the privilege.

Reporting the `llm` arm against SPY alone would have shown +5.2pp of "alpha" and
been completely misleading. That is what the control arms are for.

**The LSTM did its actual job.** Volatility forecasting is not supposed to raise
returns; it is supposed to make risk estimates better. It cut max drawdown from
−33.5% to −26.0% against its matched `minvar` control on identical objective and
constraints — but generated 4× the turnover doing it, which ate the returns.

One structural note: `equilibrium` and `equal` are the **same portfolio**, not a
coincidence. Reverse optimisation is self-inverting — `max_sharpe(δΣw, Σ)` returns
`w` for any `Σ` — and `market_weights` is equal weight here to avoid look-ahead
bias. So the equilibrium arm is equal weight wearing a hat, and it is the honest
baseline everything else has to beat.

---

## Why views, and not weights

Hand an LLM a portfolio and ask for allocations and you get a plausible-sounding
number with no theory behind it and no way to audit it. Black-Litterman offers a
better-shaped hole to put the model in.

Plain mean-variance optimisation is unusable on real return estimates: sample
means are mostly noise, and the optimiser responds by concentrating the book in
whichever asset got lucky. Black-Litterman starts instead from the returns the
market itself implies — reverse-optimised from equilibrium weights — and moves
away from them only in the directions you hold an opinion about, scaled by how
confident that opinion is.

Drop an LLM into the *views* slot and three useful properties follow:

- **A bad view is survivable.** It tilts the posterior; it does not seize the book.
- **No views is a valid answer.** With none, the posterior equals the prior and
  you hold the market portfolio. So every failure path — API down, rate limited,
  malformed JSON, hallucinated ticker — degrades to "own the market" rather than
  to something wrong.
- **Confidence is a real parameter**, not a decoration. It scales Ω, the view
  uncertainty matrix, and therefore how far the posterior actually moves.

## The views are treated as untrusted input

The model output is parsed, validated, and discarded on any failure:

| Check | Rejects |
|---|---|
| Pydantic schema | missing fields, wrong types, prose instead of JSON |
| magnitude ceiling | any \|return\| > 15% over a quarter — a hallucination, not a forecast |
| confidence range | anything outside [0, 1] |
| universe membership | tickers not held, including invented ones |
| self-reference | "AAPL beats AAPL" |
| duplicates | repeated (asset, versus) pairs |

Survivors are capped at 8 views per rebalance. Everything rejected is counted and
reported, because the rejection rate is itself a measurement of how trustworthy
the generator is.

## Look-ahead bias, and what cannot be fixed

Two distinct problems, and only one of them is solvable.

**Solvable — evidence timing.** At each rebalance the model sees only filings
whose EDGAR filing date is strictly earlier than that date. The filing date is
when the information became public, so this cut is clean.

**Not solvable — the model's own memory.** Llama 3.3 was trained on text written
after the backtest period. It knows how these companies did. Restricting its
inputs does not unlearn that, and no prompt instruction reliably suppresses it.

This is why the run reports six arms rather than one number:

| Arm | What it is |
|---|---|
| `equilibrium` | Black-Litterman with no views. The market portfolio, optimised. |
| `llm` | The same machinery, with the model's views applied. |
| `shuffled` | **The control.** Real views, with the asset each applies to permuted. Same count, same magnitudes, same confidences — the information destroyed, the plumbing identical. |
| `minvar` | Minimum variance on trailing sample covariance. |
| `lstm` | **Matched pair with `minvar`.** Same objective and constraints; only the covariance differs, using forecast volatilities. Any gap between them belongs to the volatility model. |
| `equal` | Naive 1/N. |

`shuffled` is the arm that matters. If `llm` cannot beat its own shuffle, the
views carry nothing and any outperformance came from Black-Litterman's
construction, not the model's reading. Comparing `llm` against SPY alone would
never reveal that.

## Design decisions worth defending

- **Covariance is shrunk toward its diagonal.** The sample covariance of 20
  assets from a few hundred observations is ill-conditioned, and the optimiser
  will exploit noise in its smallest eigenvalues.
- **Equal-weight equilibrium prior, not market cap.** yfinance exposes only
  *current* market cap, and weighting a 2016 portfolio by 2026 caps is exactly
  the look-ahead this project is trying to avoid. Equal weight is blunt but
  honest; a point-in-time cap series would be the real fix.
- **SLSQP, not CVXPY.** Twenty assets, a budget constraint and a box. A convex
  solver would be a dependency and a build step buying nothing.
- **Costs are charged.** 10 bps per unit of turnover, one-way. A strategy that
  only wins before costs has lost.
- **Position cap of 25%** and long-only, so no single view can take the book.

## Usage

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export GROQ_API_KEY=...            # free tier: https://console.groq.com/keys

./.venv/bin/python run.py prices
./.venv/bin/python run.py views --date 2026-06-30    # inspect one rebalance
./.venv/bin/python run.py backtest --start 2015-01-01 \
    --arms equilibrium llm shuffled minvar lstm equal
./.venv/bin/python test_advisor.py                   # 35 checks, no network
```

**Free-tier note:** Groq allows 100,000 tokens per day on `llama-3.3-70b`, and a
39-quarter backtest exhausts that in one afternoon. `views.MODEL_CHAIN` walks
down to `gpt-oss-120b` then `llama-3.1-8b-instant` when a daily budget runs out,
because a weaker model producing schema-validated views beats no views at all --
and every view is bounds-checked before it can touch the portfolio regardless of
which model wrote it.

Views are cached per rebalance date under `data/views/` and committed. They are
the record of what the model actually said on each date — without them the
backtest is not reproducible, and the `llm` and `shuffled` arms must see
identical views for the comparison to mean anything.

Set `OPENAI_BASE_URL=http://localhost:11434/v1` to run the whole thing against
Ollama instead. Same wire format, no code change.

## Things that went wrong

- **8-K filings returned nothing usable.** An 8-K's primary document is a cover
  page that defers the actual news to an exhibit, so the first 800 characters
  were inline-XBRL tags and depositary-share descriptions. The model correctly
  produced zero views from it. Switched to 10-Q/10-K, whose MD&A is inline.
- **Section extraction picked the wrong passage.** These headings appear in the
  table of contents, in cross-references, and in the section itself. Taking the
  last occurrence landed on a cross-reference buried in risk-factor boilerplate
  for 2017-era filings — so several years of the backtest silently got navigation
  text and produced no views. Now every candidate is scored by how much financial
  vocabulary follows it, with markers tried in priority order.
- **EDGAR's `recent` block only reaches back a few years.** Older filings live in
  paginated files under `filings.files`, and ignoring them meant the early
  backtest years had no evidence at all — which does not fail loudly, it just
  quietly turns the `llm` arm into the `equilibrium` arm.
- **A missing client timeout stalled a 40-rebalance run for ten minutes.** The
  OpenAI SDK defaults to 600 s with its own retries on top. Now 45 s, fail fast,
  and a failed call means "no views", which is safe.
- **The view cache poisoned itself.** Rate-limit failures were written to disk as
  legitimate `{"views": []}` results and never retried, which does not fail
  loudly -- it silently turns the `llm` arm into the `equilibrium` arm for
  however long the outage lasted. An API failure and "the model had no opinion"
  are now different things.
- **The `lstm` arm was a no-op.** It applied the forecast covariance through
  `max_sharpe(δΣw, Σ)`, which returns `w` for *any* `Σ` -- the covariance
  cancels algebraically, so the forecast could not possibly matter. Three arms
  came back byte-identical, which is what gave it away. It now uses minimum
  variance, which depends on `Σ` and can actually exploit a better estimate.
