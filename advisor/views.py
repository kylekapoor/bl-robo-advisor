"""LLM-generated market views, schema-enforced and bounds-checked.

The model's output is untrusted input. It arrives as free text from a system
that will confidently invent a ticker, quote a 400% return, or answer in prose
when asked for JSON. Everything it produces is validated, clipped, and dropped
on failure -- and dropping every view is a supported outcome, not an error,
because Black-Litterman with no views just returns the market portfolio.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

# A view claiming more than this over one rebalance horizon is not a view, it is
# a hallucination. Real quarterly alpha forecasts live well inside this band.
MAX_ABS_VIEW_RETURN = 0.15
MAX_VIEWS_PER_REBALANCE = 8


class View(BaseModel):
    """One opinion. Absolute if `versus` is None, otherwise relative."""

    asset: str
    versus: str | None = None
    expected_return: float = Field(description="excess return over the horizon, decimal")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("asset", "versus")
    @classmethod
    def _upper(cls, v):
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("expected_return")
    @classmethod
    def _sane_magnitude(cls, v):
        if abs(v) > MAX_ABS_VIEW_RETURN:
            raise ValueError(f"|{v}| exceeds {MAX_ABS_VIEW_RETURN} ceiling")
        return v


# Rejection reasons that mean "the call failed", as opposed to "the model
# considered the evidence and had nothing to say". Conflating the two is how a
# cache poisons itself: a rate-limited quarter gets stored as a legitimate empty
# result and is never retried.
_API_FAILURES = ("RateLimitError", "APIError", "APITimeoutError", "APIConnectionError",
                 "APIStatusError", "InternalServerError", "429")


@dataclass
class ViewBatch:
    views: list
    raw_count: int
    rejected: list
    model: str | None = None

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.raw_count if self.raw_count else 0.0

    @property
    def errored(self) -> bool:
        """True if any part of this batch failed rather than returning nothing."""
        return any(any(marker in reason for marker in _API_FAILURES)
                   for reason in self.rejected)


SYSTEM_PROMPT = """You are a sell-side equity analyst producing structured forecasts.

You will be given recent filing excerpts and headlines for a set of tickers.
Return ONLY a JSON object of the form:

{"views": [{"asset": "AAPL", "versus": "MSFT" or null,
            "expected_return": 0.02, "confidence": 0.4,
            "rationale": "one short sentence"}]}

Rules:
- expected_return is the excess return over the NEXT QUARTER as a decimal.
  Realistic magnitudes are 0.005 to 0.05. Never exceed 0.15.
- confidence is between 0 and 1. Use below 0.3 unless the evidence is specific
  and quantitative.
- Only use tickers from the provided list.
- Set "versus" for a relative call (asset beats versus), null for an absolute one.
- Emit a view for a company when its filing text gives a concrete, checkable
  reason -- a growth rate, a margin move, a guidance change, a named risk.
  Two or three views per group is typical. Skip a company whose text is
  boilerplate, and return an empty list if none of them say anything.
- Every rationale must quote or paraphrase a specific figure or statement from
  the supplied text.
- Base every view on the supplied text. Do not use anything you recall about
  these companies from after the documents provided."""


def client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """OpenAI-compatible client.

    Groq by default. Point OPENAI_BASE_URL at http://localhost:11434/v1 to run
    the whole thing against Ollama instead -- same wire format, no code change.
    """
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
    # The SDK defaults to a 600 s timeout with its own retries on top. A single
    # wedged request then stalls a 40-rebalance backtest for ten minutes with no
    # output and no error -- which is exactly what happened. Fail fast instead;
    # the caller already treats a failed call as "no views", which is safe.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=45.0, max_retries=1)


# Tried in order, best first. Each model carries its own daily token budget on
# Groq's free tier, and the strongest one is also the smallest: 100,000 tokens
# per day for llama-3.3-70b, which a 38-quarter backtest exhausts in a single
# afternoon. Rather than fail the run, walk down the chain -- a weaker model
# producing schema-validated views beats no views at all, and every view is
# bounds-checked before it can touch the portfolio regardless of who wrote it.
MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]
# Bounding the response matters for more than cost: without it some providers
# reserve the model's full output window against the per-minute budget.
MAX_RESPONSE_TOKENS = 900


def _is_daily_limit(exc: Exception) -> bool:
    return "per day" in str(exc) or "TPD" in str(exc)


def _is_rate_limit(exc: Exception) -> bool:
    return "rate_limit" in str(exc) or "429" in str(exc)


def generate(
    evidence: str,
    tickers,
    model: str | None = None,
    temperature: float = 0.2,
    llm: OpenAI | None = None,
    retries: int = 3,
) -> ViewBatch:
    """Ask the model for views and keep only the ones that survive validation.

    Falls through `MODEL_CHAIN` when a model's daily budget is gone. A
    per-minute limit is worth waiting out; a per-day limit is not.
    """
    llm = llm or client()
    tickers = list(tickers)
    chain = [model] if model else list(MODEL_CHAIN)

    payload, used_model, last_error = None, None, ""
    for candidate in chain:
        for attempt in range(retries):
            try:
                response = llm.chat.completions.create(
                    model=candidate,
                    temperature=temperature,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Tickers: {', '.join(tickers)}\n\n{evidence}"},
                    ],
                )
                payload = json.loads(response.choices[0].message.content)
                used_model = candidate
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if _is_daily_limit(exc):
                    break  # no amount of waiting fixes a daily cap
                if _is_rate_limit(exc):
                    time.sleep(min(45, 10 * (attempt + 1)))
                    continue
                break
        if payload is not None:
            break

    if payload is None:
        # Network failure, rate limit, or unparseable output. No views is safe:
        # Black-Litterman with no views returns the market portfolio.
        return ViewBatch(views=[], raw_count=0, rejected=[last_error])

    batch = validate(payload.get("views", []), tickers)
    batch.model = used_model
    return batch


def generate_chunked(
    evidence: dict,
    tickers,
    chunk_size: int = 6,
    model: str | None = None,
    pause: float = 6.0,
    **kwargs,
) -> ViewBatch:
    """Generate views a few tickers at a time to stay inside the token budget.

    The universe is ordered by sector, so a chunk is roughly a peer group --
    which is where relative views ("this bank beats that bank") are worth
    anything anyway. Cross-sector relative views are lost, and that is an
    acceptable trade for staying on a free tier.

    `pause` is sized against the tokens-per-minute ceiling, not latency. Groq's
    free tier allows 12,000 tokens a minute; four chunks fired three seconds
    apart put roughly 8,000 tokens into a twelve-second window, which trips the
    limit, burns the entire retry ladder and stalls a rebalance for a quarter of
    an hour while the quota sits idle. Spreading the same work across ~40
    seconds costs nothing and never retries.
    """
    tickers = [t for t in tickers if t in evidence]
    all_views, raw, rejected, used_model = [], 0, [], None

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        text = "\n\n".join(evidence[t] for t in chunk)
        batch = generate(text, chunk, model=model, **kwargs)
        if batch.model:
            used_model = batch.model
        all_views.extend(batch.views)
        raw += batch.raw_count
        rejected.extend(batch.rejected)
        if i + chunk_size < len(tickers):
            time.sleep(pause)  # stay under tokens-per-minute

    merged = validate([v.model_dump() for v in all_views], tickers)
    return ViewBatch(
        views=merged.views, raw_count=raw, rejected=rejected + merged.rejected,
        model=used_model,
    )


def validate(raw_views, tickers) -> ViewBatch:
    """Schema, universe, and duplicate checks. Returns only survivors."""
    allowed = {t.upper() for t in tickers}
    kept, rejected, seen = [], [], set()

    for raw in raw_views[: MAX_VIEWS_PER_REBALANCE * 3]:
        try:
            view = View.model_validate(raw)
        except (ValidationError, TypeError) as exc:
            rejected.append(f"schema: {str(exc)[:80]}")
            continue

        if view.asset not in allowed:
            rejected.append(f"unknown ticker {view.asset}")
            continue
        if view.versus is not None and view.versus not in allowed:
            rejected.append(f"unknown ticker {view.versus}")
            continue
        if view.versus == view.asset:
            rejected.append(f"{view.asset} vs itself")
            continue

        key = (view.asset, view.versus)
        if key in seen:
            rejected.append(f"duplicate {key}")
            continue

        seen.add(key)
        kept.append(view)

    return ViewBatch(
        views=kept[:MAX_VIEWS_PER_REBALANCE],
        raw_count=len(raw_views),
        rejected=rejected,
    )
