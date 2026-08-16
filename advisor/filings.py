"""SEC EDGAR ingestion.

EDGAR is free and has no API key, but it does have rules: a User-Agent that
identifies you, and no more than ten requests a second. Both are honoured here.

Periodic reports (10-Q, 10-K) are pulled rather than 8-Ks. An 8-K is short and
event-driven, which sounds ideal, but its primary document is usually a cover
page that defers the actual news to an exhibit -- so what you get is XBRL tags
and no content. A 10-Q carries its Management's Discussion and Analysis inline,
which is the closest thing to a quarterly narrative the filing system offers.

Critically, the filing date is the date the information became public, so
filtering on it gives a clean point-in-time cut with no look-ahead.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from html import unescape
from pathlib import Path

import requests

from . import retrieval

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "edgar"

# EDGAR blocks requests that do not identify a contact. Override via env if you
# are running this at any volume.
USER_AGENT = "bl-robo-advisor research contact@example.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
MIN_INTERVAL = 0.12  # seconds between requests; EDGAR's ceiling is 10/s

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

_last_request = 0.0


def _get(url: str) -> requests.Response:
    global _last_request
    wait = MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    response = requests.get(url, headers=HEADERS, timeout=30)
    _last_request = time.time()
    response.raise_for_status()
    return response


def ticker_to_cik(tickers) -> dict:
    """Map tickers to EDGAR CIK numbers, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "company_tickers.json"
    if not path.exists():
        path.write_text(_get(TICKER_MAP_URL).text)

    table = json.loads(path.read_text())
    lookup = {row["ticker"].upper(): int(row["cik_str"]) for row in table.values()}
    return {t: lookup[t.upper()] for t in tickers if t.upper() in lookup}


EXTRA_URL = "https://data.sec.gov/submissions/{name}"


def _rows(block: dict, cik: int, forms, out: list) -> None:
    for form, filed, accession, document in zip(
        block["form"], block["filingDate"],
        block["accessionNumber"], block["primaryDocument"],
    ):
        if forms and form not in forms:
            continue
        out.append({
            "form": form,
            "filed": date.fromisoformat(filed),
            "cik": cik,
            "accession": accession.replace("-", ""),
            "document": document,
        })


# A large cap with fewer periodic reports than this has not been public for
# twenty years -- or, more often, has reorganised and left its history behind
# under a different CIK.
MIN_HISTORY = 20


def recent_filings(cik: int, forms=("10-Q", "10-K"), limit: int = 80,
                   follow_predecessor: bool = True) -> list:
    """Filing metadata for one company, newest first.

    The `recent` block holds only the last ~1000 filings, which for an active
    large-cap reaches back a few years at most. Anything older lives in the
    paginated files listed under `filings.files`, and skipping them silently
    yields zero evidence for early backtest dates -- which does not fail, it
    just quietly turns the LLM arm into the equilibrium arm.

    Reorganisations do the same thing more sharply. `company_tickers.json` maps
    a ticker to whichever entity holds it today, and when a company reincorporates
    under a holding company the new entity takes the ticker while every past
    filing stays with the old CIK. XOM is the live example: the ticker points at
    ExxonMobil Holdings Corp and one 10-Q, while twenty years of Exxon Mobil Corp
    sit under a CIK that now lists no ticker at all.

    The accession number carries the filer's CIK in its first ten digits, so a
    thin result can be repaired by following it. That is safe to attempt blindly:
    filing agents appear in accession prefixes too, but they have no submissions
    file of their own and 404.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"submissions_{cik}.json"
    if not path.exists():
        path.write_text(_get(SUBMISSIONS_URL.format(cik=cik)).text)

    filings_block = json.loads(path.read_text())["filings"]
    out: list = []
    _rows(filings_block["recent"], cik, forms, out)

    for extra in filings_block.get("files", []):
        extra_path = CACHE / extra["name"]
        if not extra_path.exists():
            try:
                extra_path.write_text(_get(EXTRA_URL.format(name=extra["name"])).text)
            except requests.RequestException:
                continue
        _rows(json.loads(extra_path.read_text()), cik, forms, out)

    if follow_predecessor and len(out) < MIN_HISTORY:
        for predecessor in {int(f["accession"][:10]) for f in out} - {cik}:
            try:
                out.extend(recent_filings(predecessor, forms, limit,
                                          follow_predecessor=False))
            except requests.RequestException:
                continue  # a filing agent, not a predecessor

    out.sort(key=lambda f: f["filed"], reverse=True)
    return out[:limit]


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

def filing_text(filing: dict, max_chars: int = 2000) -> str:
    """The passages of a filing most relevant to peer comparison.

    The whole document is cached as text, then retrieval picks from it. Caching
    the raw filing rather than the excerpt means a change to the retrieval
    settings costs embedding time and not 780 more requests to EDGAR.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{filing['cik']}_{filing['accession']}.txt"

    if path.exists():
        text = path.read_text()
    else:
        url = ARCHIVE_URL.format(
            cik=filing["cik"], accession=filing["accession"], document=filing["document"]
        )
        try:
            html = _get(url).text
        except requests.RequestException:
            return ""
        text = _WS.sub(" ", unescape(_TAG.sub(" ", html))).strip()
        path.write_text(text)

    return retrieval.retrieve_cached(text, path, max_chars=max_chars)


def evidence_by_ticker(
    tickers, as_of: date, lookback_days: int = 130, per_ticker: int = 1, max_chars: int = 2000
) -> dict:
    """Dated evidence per ticker, strictly before `as_of`.

    The `as_of` filter is the whole point. Feeding the model a filing published
    after the rebalance date would be look-ahead bias dressed up as research.

    The old limit here was 1,400 characters, sized against a hosted tokens-per-
    minute ceiling rather than against what the model needed. Local inference
    has no such ceiling, so the budget now goes to retrieval: three targeted
    passages instead of one contiguous slice.
    """
    ciks = ticker_to_cik(tickers)
    out = {}
    for ticker, cik in ciks.items():
        try:
            filings = recent_filings(cik)
        except requests.RequestException:
            continue
        window = [
            f for f in filings
            if f["filed"] < as_of and (as_of - f["filed"]).days <= lookback_days
        ][:per_ticker]
        blocks = []
        for f in window:
            text = filing_text(f, max_chars=max_chars)
            if text:
                blocks.append(f"[{ticker} {f['form']} filed {f['filed']}]\n{text}")
        if blocks:
            out[ticker] = "\n".join(blocks)
    return out


