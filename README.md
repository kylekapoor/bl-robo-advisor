# bl-robo-advisor

[![tests](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml/badge.svg)](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml)

I tested whether an LLM can read company filings and pick better stocks. After
two rebuilds I still cannot show that it can. This last version came close
enough to fooling me that I had to build a second control to rule it out.

An LLM reads SEC filings and writes opinions: *"Chevron will outperform
Caterpillar by 3%, confidence 40%, because its filing reports X."* Black-Litterman
then decides how far those opinions move the portfolio. The LLM never picks a
weight.

The controls are the part I care about. I re-run the same opinions twice more:
once with the company names swapped at random, and once with a single fixed
opinion set replayed on every date. Beating both is the bar for saying the model
read anything.

`Python` · `LangChain` · `TensorFlow/Keras` · `Ollama` · `Llama 3.1` · `sentence-transformers` · `yfinance` · `SEC EDGAR` · `NumPy` · `Pandas` · `SciPy` · `Pydantic`

---

## Results

Quarterly rebalance, 20 US large caps, 10 bps costs, 25% position cap, 304 views
over 39 quarters.

Held out from 2023, a window I never used to tune anything:

| Arm | CAGR | Sharpe | Max DD | Turnover | Info ratio |
|---|---|---|---|---|---|
| equilibrium (no views) | **25.3%** | 2.010 | −16.2% | 0.00 | **+0.21** |
| llm | 24.7% | **2.046** | −16.2% | 0.37 | +0.10 |
| static *(control)* | 22.0% | 1.849 | −15.3% | 0.05 | −0.18 |
| shuffled *(control)* | 20.9% | 1.736 | −16.0% | 0.55 | −0.27 |
| minvar | 12.9% | 1.194 | −10.6% | 0.15 | −0.69 |
| lstm | 9.0% | 0.817 | −14.2% | 0.85 | −0.99 |
| SPY | 23.4% | 1.551 | −18.8% | n/a | n/a |

The `llm` arm now beats both controls on Sharpe, in the holdout and over the
full period, which is the opposite of what the previous version of this project
found. Before I upgraded retrieval, it lost to its own shuffle.

## The gaps do not clear the noise

I bootstrapped the Sharpe difference on paired daily returns, resampling in
21-day blocks to preserve the autocorrelation:

| Comparison | Full period | Holdout |
|---|---|---|
| llm − shuffled | +0.098 [−0.039, +0.227] | +0.252 [−0.011, +0.504] |
| llm − static | +0.092 [−0.017, +0.190] | +0.159 [−0.054, +0.380] |
| llm − equilibrium | +0.042 [−0.095, +0.173] | +0.035 [−0.238, +0.332] |

Every interval contains zero. All six comparisons lean the same way, and 39
quarters cannot separate any of them from chance. The holdout llm-minus-shuffled
interval misses by 0.011.

The views lose on the two measures that do not reward taking less risk. The
`llm` arm trails `equilibrium` on CAGR and on information ratio in both windows,
at 14x the turnover. It buys lower volatility, which flatters Sharpe.

Better retrieval moved the point estimates from losing to winning. It did not
give me evidence I would trade on.

## Why there are two controls

`shuffled` permutes which companies a view is about. That removes the model's
stock selection, and it also removes any persistent tilt, so `llm` beating
`shuffled` is ambiguous between reading each quarter's filings and restating one
ranking forever.

The views made that worth checking. Across 39 quarters there are 304 of them and
only **58 distinct pairs**. AMZN over BAC appears in 82% of quarters, GOOGL over
GS in 74%, and half of all views belong to a pair seen ten or more times. I chunk
the universe in sector order, so the model compares the same peers every quarter
and repeats itself.

`static` takes the eight most common opinions at their median magnitude and
confidence and applies them unchanged on every date. It keeps the tilt and throws
away everything quarter-specific. I compute the modal set over the whole sample,
so this control runs on hindsight the `llm` arm never had. That makes it a harder
baseline than `llm` deserves.

`llm` beats `static` by +0.092 and +0.159 Sharpe, so the quarter-to-quarter
content is doing something the fixed tilt is not. Both intervals still contain
zero.

## Retrieval

Evidence used to come from a hand-written passage scorer: jump to a heading like
"Management's Discussion and Analysis", then rank candidates by counting
financial vocabulary. It worked until it did not: that heading also appears in
the table of contents and in cross-references, so the scorer sometimes handed
the model navigation text.

Now I chunk the filing and retrieve by embedding similarity, which drops the
heading problem instead of patching it. A contents entry does not resemble a
sentence about margin compression, so it loses on cosine distance with no rule
saying it should.

Choosing the query took longer than choosing the model. My first one listed
topics, including "material business risk", and it retrieved the risk-factor
boilerplate that names those topics without reporting anything. Phrasing it as
movement instead pulled back sentences carrying figures:

> Noninterest revenue $18,852 / $17,638 / 7% ... Lower turnaround expenses
> increased earnings by $10 million

I left no fallback to the old scorer. Retrieval decides what the model sees, so a
silent downgrade would change every number here while the run still looked
healthy.

## How it works

| Arm | What it is |
|---|---|
| `equilibrium` | Black-Litterman with no views. |
| `llm` | The same machinery with the model's views applied. |
| `shuffled` | Real views, asset assignments permuted. Keeps count, magnitudes, confidences and relative structure, destroys only the information. |
| `static` | The modal view set replayed every quarter. Keeps the tilt, destroys everything quarter-specific. |
| `minvar` | Minimum variance on trailing covariance. |
| `lstm` | Matched pair with `minvar`. Only the covariance differs, using forecast volatilities. |
| `equal` | Naive 1/N. |

`equilibrium` and `equal` produce the same portfolio, which is not a
coincidence. Reverse optimisation inverts itself: `max_sharpe(δΣw, Σ)` returns
`w` for any `Σ`, and I use equal weights as the market prior to dodge look-ahead
bias.

I treat views as untrusted input and check each one against a Pydantic schema, a
15% magnitude ceiling, a [0,1] confidence range, universe membership,
self-reference and duplicate comparisons. Throwing all of them away is a
supported outcome, since Black-Litterman with no views hands back the market
portfolio.

The model only sees filings dated before each rebalance. It cannot unlearn its
own training data, which is why I built the controls.

The LSTM did the job I gave it. Volatility forecasting exists to sharpen risk
estimates rather than to raise returns, and against its matched `minvar` control
it cut max drawdown from 33.5% to 26.0%. It rebuilds the covariance as `D·R·D`,
keeping historical correlations and replacing the variances, and it predicts a
ratio rather than a level, because a pooled model trained on levels hands back
near-identical volatility for every asset and throws away the cross-section.

## Usage

```bash
brew install ollama && ollama serve      # or the installer from ollama.com
ollama pull llama3.1:8b

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python run.py views --date 2026-06-30      # inspect one quarter
./.venv/bin/python run.py backtest --start 2015-01-01 --holdout-start 2023-01-01 \
    --arms equilibrium llm static shuffled minvar lstm equal
./.venv/bin/python test_advisor.py                     # 45 checks, no network
```

No API key and no account. Views come from a local Llama 3.1 through Ollama, and
the embedding weights download once from Hugging Face.

The committed views under `data/views/` are the ones the results above were built
from. They stay committed because they record what the model said on each date,
and because `llm`, `shuffled` and `static` have to see identical views for the
comparison to hold. Re-run locally and you will get different views and different
numbers.

## Bugs worth recording

- **A reversed pair counted twice.** The model emits "A beats B by 1%" and "B
  beats A by −1%" for the same pair, because chunking shows it that pair from
  both sides. My dedupe keyed on the ordered tuple, so one opinion entered the
  posterior twice carrying double its stated confidence.
- **XOM contributed nothing for the entire backtest.** The ticker now maps to a
  holding company created in a reorganisation, which has one filing, while twenty
  years of history sit under a CIK that lists no ticker at all. A thin result now
  follows the filer CIK embedded in the accession number back to the original
  entity.
- **The rejection rate read 0% on a batch that discarded 14 of 22 views.** I
  counted views dropped by the eight-view cap as neither kept nor rejected.
- **The `lstm` arm did nothing.** It ran the forecast covariance through
  `max_sharpe(δΣw, Σ)`, which returns `w` for any `Σ`, so the covariance
  cancelled out. Three arms came back byte-identical, which gave it away.
- **The view cache poisoned itself.** It stored rate-limit failures as real
  `{"views": []}` results and never retried them, which turned the `llm` arm into
  the `equilibrium` arm without raising anything.
- **An intermediate run showed the `llm` arm winning.** It had failed 36 of 39
  quarters on rate limits, leaving views only in the quarters I had tuned on and
  none in the holdout, so the arm was winning by not existing.
- **8-K filings gave me nothing.** The primary document is a cover page that
  defers to exhibits, so the model received XBRL tags. I switched to 10-Q/10-K.

## Limits

- 39 quarters cannot separate these arms. I report every number above with its
  interval for that reason.
- The model repeats itself. 304 views cover 58 distinct pairs, so the `llm` arm
  drifts between a handful of fixed opinions instead of doing fresh quarterly
  analysis.
- The model's training data postdates the backtest. I cannot fix that, hence the
  controls.
- The equilibrium prior uses equal weights, because yfinance exposes only current
  market cap and weighting a 2016 portfolio by 2026 caps is the look-ahead this
  project exists to avoid.
- 20 liquid US large caps. Nothing here says anything about small caps or thin
  names.
