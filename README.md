# bl-robo-advisor

[![tests](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml/badge.svg)](https://github.com/kylekapoor/bl-robo-advisor/actions/workflows/tests.yml)

I tested whether an LLM can read company filings and pick better stocks. After
two rebuilds I still cannot show that it can, and this last version came close
enough to fooling me that I had to build a second control to rule it out.

An LLM reads SEC filings and writes opinions: *"Chevron will outperform
Caterpillar by 3%, confidence 40%, because its filing reports X."* Black-Litterman
decides how far those opinions move the portfolio. The LLM never picks a weight.

`Python` · `LangChain` · `TensorFlow/Keras` · `Ollama` · `Llama 3.1` · `sentence-transformers` · `SEC EDGAR` · `SciPy` · `Pydantic`

---

## Results

Quarterly rebalance, 20 US large caps, 10 bps costs, 25% position cap, 304 views
over 39 quarters. Held out from 2023, a window never used to tune anything:

| Arm | CAGR | Sharpe | Max DD | Turnover | Info ratio |
|---|---|---|---|---|---|
| equilibrium (no views) | **25.3%** | 2.010 | −16.2% | 0.00 | **+0.21** |
| llm | 24.7% | **2.046** | −16.2% | 0.37 | +0.10 |
| static *(control)* | 22.0% | 1.849 | −15.3% | 0.05 | −0.18 |
| shuffled *(control)* | 20.9% | 1.736 | −16.0% | 0.55 | −0.27 |
| minvar | 12.9% | 1.194 | −10.6% | 0.15 | −0.69 |
| lstm | 9.0% | 0.817 | −14.2% | 0.85 | −0.99 |
| SPY | 23.4% | 1.551 | −18.8% | n/a | n/a |

The `llm` arm beats both controls on Sharpe, in the holdout and over the full
period. That is the opposite of what the previous version found, where it lost to
its own shuffle.

## The gaps do not clear the noise

I bootstrapped the Sharpe difference on paired daily returns, resampling in
21-day blocks to preserve autocorrelation:

| Comparison | Full period | Holdout |
|---|---|---|
| llm − shuffled | +0.098 [−0.039, +0.227] | +0.252 [−0.011, +0.504] |
| llm − static | +0.092 [−0.017, +0.190] | +0.159 [−0.054, +0.380] |
| llm − equilibrium | +0.042 [−0.095, +0.173] | +0.035 [−0.238, +0.332] |

Every interval contains zero. All six lean the same way, and 39 quarters cannot
separate any of them from chance. The views also trail `equilibrium` on CAGR and
information ratio in both windows, at 14x the turnover. Better retrieval moved
the point estimates from losing to winning and gave me nothing I would trade on.

## Why there are two controls

`shuffled` permutes which companies a view is about, removing stock selection and
any persistent tilt at once. So `llm` beating it is ambiguous between reading each
quarter's filings and restating one ranking forever.

The views made that worth checking: 304 of them cover only **58 distinct pairs**,
AMZN over BAC appears in 82% of quarters, and half belong to a pair seen ten or
more times. I chunk the universe in sector order, so the model compares the same
peers every quarter and repeats itself.

`static` replays the eight most common opinions on every date, keeping the tilt
and discarding everything quarter-specific. I compute it over the whole sample,
so it runs on hindsight the `llm` arm never had, which makes it a harder baseline
than `llm` deserves. `llm` still beats it, so the quarter-to-quarter content does
something the fixed tilt does not. Both intervals contain zero.

## Retrieval

Evidence used to come from a hand-written passage scorer that jumped to headings
like "Management's Discussion and Analysis". That heading also appears in the
table of contents and in cross-references, so it sometimes handed the model
navigation text. Now I chunk each filing and retrieve by embedding similarity,
which drops the problem instead of patching it.

Choosing the query took longer than choosing the model. My first listed topics,
including "material business risk", and retrieved the boilerplate naming those
topics without reporting anything. Phrasing it as movement pulled back sentences
carrying figures.

## Arms

| Arm | What it is |
|---|---|
| `equilibrium` | Black-Litterman with no views. |
| `llm` | The same machinery with the model's views applied. |
| `shuffled` | Real views, asset assignments permuted. Same count, magnitudes and structure, information destroyed. |
| `static` | The modal view set replayed every quarter. |
| `minvar` | Minimum variance on trailing covariance. |
| `lstm` | Matched pair with `minvar`, differing only in forecast volatilities. |
| `equal` | Naive 1/N. |

`equilibrium` and `equal` produce the same portfolio, because reverse
optimisation inverts itself and I use equal weights as the market prior to dodge
look-ahead bias. Views are untrusted input, validated against a Pydantic schema, a
15% magnitude ceiling and duplicate comparisons, and the model only sees filings
dated before each rebalance.

## Bugs worth recording

- **A reversed pair counted twice.** The model emits "A beats B by 1%" and "B
  beats A by −1%" for the same pair, because chunking shows it that pair from both
  sides. My dedupe keyed on the ordered tuple, so one opinion entered the
  posterior twice at double its stated confidence.
- **XOM contributed nothing for the entire backtest.** Its ticker now maps to a
  holding company with one filing while twenty years sit under a CIK listing no
  ticker. A thin result now follows the filer CIK in the accession number back to
  the original entity.
- **The `lstm` arm did nothing.** It ran the forecast covariance through
  `max_sharpe(δΣw, Σ)`, which returns `w` for any `Σ`. Three arms came back
  byte-identical, which gave it away.

## Usage

```bash
ollama pull llama3.1:8b
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python run.py backtest --start 2015-01-01 --holdout-start 2023-01-01 \
    --arms equilibrium llm static shuffled minvar lstm equal
./.venv/bin/python test_advisor.py     # 45 checks, no network
```

Views under `data/views/` are committed, because `llm`, `shuffled` and `static`
must see identical views for the comparison to hold.

## Limits

- 39 quarters cannot separate these arms, hence every number above carries its
  interval.
- The model repeats itself, so the `llm` arm drifts between a handful of fixed
  opinions rather than doing fresh quarterly analysis.
- Its training data postdates the backtest. I cannot fix that, hence the controls.
