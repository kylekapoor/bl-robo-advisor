# bl-robo-advisor

[![tests](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml/badge.svg)](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml)

I tested whether an LLM can read company filings and pick better stocks. It
cannot, and proving that took more work than getting a number that looked good.

An LLM reads SEC filings and writes opinions: *"Chevron will outperform
Caterpillar by 3%, confidence 40%, because its filing reports X."* Black-Litterman
then decides how far those opinions move the portfolio. The LLM never picks a
weight.

The control is the part I care about. I re-run the same opinions with the
company names swapped at random. If the real ones cannot beat their own shuffle,
they hold nothing.

`Python` · `TensorFlow/Keras` · `Ollama` · `Llama 3.1` · `yfinance` · `SEC EDGAR` · `NumPy` · `Pandas` · `SciPy` · `Pydantic`

---

## Results

Quarterly rebalance, 20 US large caps, 10 bps costs, 25% position cap, 244 LLM
views over 39 quarters.

Held out from 2023, a window I never used to tune anything:

| Arm | CAGR | Sharpe | Max DD | Info ratio |
|---|---|---|---|---|
| equilibrium (no views) | **25.3%** | **2.010** | −16.2% | +0.21 |
| llm | 22.1% | 1.789 | −15.6% | **−0.16** |
| shuffled *(control)* | 23.5% | 1.945 | −16.2% | −0.02 |
| minvar | 12.9% | 1.194 | −10.6% | −0.69 |
| lstm | 9.0% | 0.817 | −14.2% | −0.99 |
| SPY | 23.4% | 1.551 | −18.8% | n/a |

The LLM's stock picks hold no signal. On the holdout the `llm` arm loses to its
own shuffle and to running with no views at all. Over the full period its Sharpe
matches the shuffle to three decimals, 1.138 against 1.138.

The views also cost money. They drive 18x the baseline turnover, 0.46 against
0.026, and at 10 bps a side that spending accounts for most of the gap.

Had I reported this arm against SPY alone it would have looked like
outperformance. The control is why I know better.

## The result survived being attacked

My first version asked for directional calls and got 83% bullish opinions with
one relative view in 312, built on 900 characters of filing text. That tests an
LLM asked for bullish opinions, not an LLM comparing companies.

So I rebuilt it. Views are now relative comparisons between peers, 231 of 244 of
them, each citing a figure, on 1,400 characters of MD&A. A relative view is
market-neutral by construction, since "A beats B" also says something negative
about B, which drops the bullish tilt. Better views, and the conclusion held.

One trap nearly caught me. An intermediate run showed the `llm` arm winning. It
had failed 36 of 39 quarters on rate limits, leaving views only in the quarters I
had tuned on and none in the holdout, so the arm was winning by not existing.

The LSTM did the job I gave it. Volatility forecasting exists to sharpen risk
estimates, not to raise returns. Against its matched `minvar` control it cut max
drawdown from 33.5% to 26.0%, at 4x the turnover.

## How it works

| Arm | What it is |
|---|---|
| `equilibrium` | Black-Litterman with no views. The market portfolio, optimised. |
| `llm` | The same machinery with the model's views applied. |
| `shuffled` | The control. Real views with asset assignments permuted, keeping count, magnitudes, confidences and relative structure, destroying only the information. |
| `minvar` | Minimum variance on trailing covariance. |
| `lstm` | Matched pair with `minvar`. Only the covariance differs, using forecast volatilities. |
| `equal` | Naive 1/N. |

`equilibrium` and `equal` produce the same portfolio, which is not a
coincidence. Reverse optimisation inverts itself: `max_sharpe(δΣw, Σ)` returns
`w` for any `Σ`, and I use equal weights as the market prior to dodge look-ahead
bias.

I treat views as untrusted input and check each one against a Pydantic schema, a
15% magnitude ceiling, a [0,1] confidence range, universe membership,
self-reference and duplicates. Throwing all of them away is a supported outcome,
since Black-Litterman with no views hands back the market portfolio.

The model only sees filings dated before each rebalance. It cannot unlearn its
own training data, which is why I built the controls.

A Keras LSTM rebuilds the covariance as `D·R·D`, keeping historical correlations
and replacing the variances. It predicts a ratio rather than a level, because a
pooled model trained on levels hands back near-identical volatility for every
asset and throws away the cross-section.

## Usage

```bash
brew install ollama && ollama serve      # or the installer from ollama.com
ollama pull llama3.1:8b

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python run.py views --date 2026-06-30      # inspect one quarter
./.venv/bin/python run.py backtest --start 2015-01-01 \
    --holdout-start 2023-01-01 --arms equilibrium llm shuffled minvar lstm equal
./.venv/bin/python test_advisor.py                     # 35 checks, no network
```

Views come from a local Llama 3.1 through Ollama, so the backtest needs no API
key and no account. This ran on Groq's free tier until Groq decommissioned the
model at the head of my chain with two days' notice, on top of a 100,000 token
daily cap that one 39-quarter run exhausts in an afternoon.

The committed views under `data/views/` are the ones the results above were
built from, generated on the hosted models before the move. They stay committed
because they record what the model said on each date, and because the `llm` and
`shuffled` arms have to see identical views for the comparison to hold. Re-run
locally and you will get different views and different numbers.

## Bugs worth recording

- **The `lstm` arm did nothing.** It ran the forecast covariance through
  `max_sharpe(δΣw, Σ)`, which returns `w` for any `Σ`, so the covariance
  cancelled out. Three arms came back byte-identical, which gave it away.
- **The view cache poisoned itself.** Rate-limit failures got stored as real
  `{"views": []}` results and never retried, which turned the `llm` arm into the
  `equilibrium` arm without any error.
- **8-K filings gave me nothing.** The primary document is a cover page that
  defers to exhibits, so the model received XBRL tags. I switched to 10-Q/10-K.
- **A missing client timeout** stalled a 40-quarter run for ten minutes.
- **The provider deprecated my model.** Two days' notice, mid-project. Local
  inference has no key, no quota and nothing to deprecate.

## Limits

- The model's training data postdates the backtest. I cannot fix that, hence the
  controls.
- The equilibrium prior uses equal weights, because yfinance exposes only current
  market cap and weighting a 2016 portfolio by 2026 caps is the look-ahead this
  project exists to avoid.
- 20 liquid US large caps. Nothing here says anything about small caps or thin
  names.
