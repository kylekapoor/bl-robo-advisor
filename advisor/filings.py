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


def recent_filings(cik: int, forms=("10-Q", "10-K"), limit: int = 80) -> list:
    """Filing metadata for one company, newest first.

    The `recent` block holds only the last ~1000 filings, which for an active
    large-cap reaches back a few years at most. Anything older lives in the
    paginated files listed under `filings.files`, and skipping them silently
    yields zero evidence for early backtest dates -- which does not fail, it
    just quietly turns the LLM arm into the equilibrium arm.
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

    out.sort(key=lambda f: f["filed"], reverse=True)
    return out[:limit]


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Where the actual content starts, best first. Everything before these markers is
# inline-XBRL context tags and SEC cover-page boilerplate: hundreds of company
# identifiers, note series and depositary-share descriptions carrying no
# information whatsoever. Truncating from the top of the file returns pure noise,
# and a correctly behaving model responds to it by producing no views at all.
_BODY_MARKERS = [
    re.compile(r"Management.{0,3}s Discussion and Analysis", re.I),
    re.compile(r"Results of Operations", re.I),
    re.compile(r"\bItem\s+\d\.\d\d\b"),
    re.compile(r"\bItem\s+2\b", re.I),
]


# Words that show up in actual financial narrative and essentially never in a
# table of contents entry or a cross-reference.
_SIGNAL = re.compile(
    r"increase|decrease|compared to|primarily due|growth|margin|revenue|"
    r"net sales|quarter|billion|million|%",
    re.I,
)


def _score_passage(passage: str) -> int:
    """How much this reads like narrative rather than navigation."""
    return len(_SIGNAL.findall(passage[:3000]))


def _extract_body(text: str) -> str:
    """Jump to the passage that actually contains financial narrative.

    Neither "first match" nor "last match" works. These headings appear in the
    table of contents, in cross-references ("refer to Management's Discussion
    and Analysis of this Form 10-Q"), and in the section itself. Taking the last
    occurrence lands on whichever happens to come last, which for 2017-era
    filings was a cross-reference buried in risk-factor boilerplate -- so the
    model received navigation text, correctly concluded there was nothing to say,
    and returned no views for years of the backtest.

    So: score every candidate by how much financial vocabulary follows it, and
    take the best. Cheap, and robust to filings that arrange themselves oddly.
    """
    fallback, fallback_score = None, -1

    # Markers are tried in priority order and scoring happens *within* a marker,
    # not across them. Ranking globally lets a high-scoring "Item 2. Unregistered
    # Sales of Equity Securities" -- Part II share repurchases, dense with
    # numbers -- outrank the MD&A it is supposed to lose to.
    for marker in _BODY_MARKERS:
        best, best_score = None, -1
        for hit in marker.finditer(text):
            passage = text[hit.start():]
            if len(passage) < 500:
                continue
            score = _score_passage(passage)
            if score > best_score:
                best, best_score = passage, score
        if best is not None and best_score >= 8:
            return best
        if best is not None and best_score > fallback_score:
            fallback, fallback_score = best, best_score

    # Require some real signal; otherwise the whole document is boilerplate and
    # the caller is better off with its start than with a false lead.
    return fallback if fallback is not None and fallback_score >= 5 else text


def filing_text(filing: dict, max_chars: int = 3000) -> str:
    """Plain text of a filing's primary document, cached whole, trimmed on read."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{filing['cik']}_{filing['accession']}.txt"
    if path.exists():
        return _extract_body(path.read_text())[:max_chars]

    url = ARCHIVE_URL.format(
        cik=filing["cik"], accession=filing["accession"], document=filing["document"]
    )
    try:
        html = _get(url).text
    except requests.RequestException:
        return ""

    text = _WS.sub(" ", unescape(_TAG.sub(" ", html))).strip()
    path.write_text(text)
    return _extract_body(text)[:max_chars]


def evidence_by_ticker(
    tickers, as_of: date, lookback_days: int = 130, per_ticker: int = 1, max_chars: int = 900
) -> dict:
    """Dated evidence per ticker, strictly before `as_of`.

    The `as_of` filter is the whole point. Feeding the model a filing published
    after the rebalance date would be look-ahead bias dressed up as research.

    Kept deliberately short. Groq's free tier allows 12,000 tokens a minute, and
    a full 8-K for twenty companies is several times that, so the caller chunks
    this dict rather than sending it all at once.
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


