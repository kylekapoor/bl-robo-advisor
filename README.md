# bl-robo-advisor

Can an AI read company financial filings and pick better stocks?

An LLM reads SEC filings and writes down opinions — *"Chevron will outperform
Caterpillar by 3%, I'm 40% confident, because its filing reports X"*. A standard
finance model (Black-Litterman) then decides how much those opinions should
actually move the portfolio. **The LLM never picks a weight.**

The interesting part is the control: the same opinions are re-run with the
company names randomly swapped. If the real opinions can't beat their own
shuffle, they contain nothing.

`Python` · `TensorFlow/Keras` · `Llama 3.3` · `Groq` · `yfinance` · `SEC EDGAR` · `NumPy` · `Pandas` · `SciPy` · `Pydantic`

---

## Results

Quarterly rebalance, 20 US large caps, 10 bps costs, 25% position cap, 244 LLM
views across 39 quarters.

**Held out from 2023 — a window never used to tune anything:**

| Arm | CAGR | Sharpe | Max DD | Info ratio |
|---|---|---|---|---|
| equilibrium (no views) | **25.3%** | **2.010** | −16.2% | +0.21 |
| **llm** | 22.1% | 1.789 | −15.6% | **−0.16** |
| **shuffled** *(control)* | 23.5% | 1.945 | −16.2% | −0.02 |
| minvar | 12.9% | 1.194 | −10.6% | −0.69 |
| lstm | 9.0% | 0.817 | −14.2% | −0.99 |
| SPY | 23.4% | 1.551 | −18.8% | — |

**The LLM's stock picks carry no signal.** On the holdout it loses to its own
shuffle *and* to using no views at all. Over the full period its Sharpe ties the
shuffle exactly, at 1.138.

It is also expensive: the views drive 18× the turnover of the baseline (0.46 vs
0.026), and at 10 bps a side that cost is most of the gap.

Reporting this arm against SPY alone would have shown apparent outperformance and
been meaningless. That is what the control is for.

## The result survived being attacked

A first version produced only directional calls — 83% bullish, 1 relative view in
312 — on 900 characters of filing text. That's a weak test: it measures "an LLM
asked for bullish opinions", not "an LLM comparing peers".

So it was rebuilt. Views are now **relative peer comparisons** (231 of 244, 95%),
each citing a specific figure, on 1,400 characters of Management's Discussion and
Analysis. Relative views are market-neutral by construction — "A beats B" is
simultaneously a negative call on B — which removes the bullish-tilt confound
entirely.

Better views. Same conclusion.

One trap caught along the way: an intermediate run showed the LLM arm *winning*.
It had failed on 36 of 39 quarters from rate limits, so the only quarters with
views were the early ones used for tuning, and the holdout was empty — the arm
was winning by being absent. The holdout existed to catch exactly that.

**The LSTM did its job.** Volatility forecasting is not meant to raise returns; it
is meant to improve risk estimates. Against its matched `minvar` control — same
objective, same constraints, only the covariance differs — it cut max drawdown
from −33.5% to −26.0% over the full period, at 4× the turnover.

## How it works

| Arm | What it is |
|---|---|
| `equilibrium` | Black-Litterman, no views. The market portfolio, optimised. |
| `llm` | Same machinery, with the model's views applied. |
| `shuffled` | **Control.** Real views, asset assignments permuted. Same count, magnitudes, confidences and relative structure — only the information is destroyed. |
| `minvar` | Minimum variance on trailing covariance. |
| `lstm` | **Matched pair with `minvar`.** Only the covariance differs, using forecast volatilities. |
| `equal` | Naive 1/N. |

`equilibrium` and `equal` are the *same portfolio*, not a coincidence: reverse
optimisation is self-inverting — `max_sharpe(δΣw, Σ)` returns `w` for any `Σ` —
and market weights are equal weight here to avoid look-ahead bias.

**Views are untrusted input.** Every one is checked against a Pydantic schema, a
15% magnitude ceiling, a [0,1] confidence range, universe membership,
self-reference and duplicates. Failures are dropped, and dropping everything is a
supported outcome — Black-Litterman with no views returns the market portfolio,
so every failure path degrades to "own the market".

**Point-in-time.** At each rebalance the model sees only filings whose EDGAR
filing date is strictly earlier. What it cannot unlearn is its own training data,
which is why the controls exist.

**Volatility.** A Keras LSTM forecasts each asset's volatility and rebuilds the
covariance as `D·R·D` — historical correlations, forecast variances. It predicts
a *ratio* rather than a level, because a pooled model trained on levels returns
near-identical volatility for every asset and throws away the cross-section.

## Usage

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export GROQ_API_KEY=...        # free tier: console.groq.com/keys

./.venv/bin/python run.py views --date 2026-06-30      # inspect one quarter
./.venv/bin/python run.py backtest --start 2015-01-01 \
    --holdout-start 2023-01-01 --arms equilibrium llm shuffled minvar lstm equal
./.venv/bin/python test_advisor.py                     # 35 checks, no network
```

Groq allows 100,000 tokens/day per model and a 39-quarter run exhausts that, so
`views.MODEL_CHAIN` falls through to the next model when a daily budget runs out.
Cached views are committed — they are the record of what the model said on each
date, and the `llm` and `shuffled` arms must see identical views.

## Bugs worth recording

- **The `lstm` arm was a no-op.** It applied the forecast covariance through
  `max_sharpe(δΣw, Σ)`, which returns `w` for *any* `Σ` — the covariance cancels
  algebraically. Three arms came back byte-identical, which gave it away.
- **The view cache poisoned itself.** Rate-limit failures were stored as
  legitimate `{"views": []}` results and never retried, silently turning the
  `llm` arm into the `equilibrium` arm.
- **8-K filings yielded nothing usable** — the primary document is a cover page
  that defers content to exhibits, so the model received XBRL tags. Switched to
  10-Q/10-K, and section extraction now scores candidate passages for financial
  narrative rather than taking the first or last heading match.
- **A missing client timeout** stalled a 40-quarter run for ten minutes on a
  single wedged request.

## Limits

- The model's training data postdates the backtest. Unfixable; hence the controls.
- Equal-weight equilibrium prior, because yfinance only exposes *current* market
  cap and weighting a 2016 portfolio by 2026 caps is the exact look-ahead this
  project avoids.
- 20 liquid US large caps. Nothing here says anything about small caps or
  illiquid names.
