#!/usr/bin/env python3
"""bl-robo-advisor CLI:  prices -> views -> backtest."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


from advisor import backtest, filings, prices as px, views as views_mod

ROOT = Path(__file__).resolve().parent
VIEW_CACHE = ROOT / "data" / "views"
RESULTS = ROOT / "data" / "results.json"


class CachedViews:
    """Generates views once per rebalance date and caches them on disk.

    Two things need this. The `llm` and `shuffled` arms must see the *same*
    views or the comparison is meaningless, and re-running a backtest should not
    re-bill the API for identical work.
    """

    def __init__(self, model: str, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.stats = {"calls": 0, "cached": 0, "empty": 0, "rejected": 0, "raw": 0}
        VIEW_CACHE.mkdir(parents=True, exist_ok=True)

    def __call__(self, as_of: date, tickers: list):
        path = VIEW_CACHE / f"{as_of}.json"
        if path.exists():
            self.stats["cached"] += 1
            payload = json.loads(path.read_text())
            return views_mod.validate(payload["views"], tickers)

        evidence = filings.evidence_by_ticker(tickers, as_of=as_of)
        if not evidence:
            self.stats["empty"] += 1
            path.write_text(json.dumps({"views": [], "reason": "no filings in window"}))
            return views_mod.ViewBatch(views=[], raw_count=0, rejected=[])

        batch = views_mod.generate_chunked(evidence, tickers, model=self.model)
        self.stats["calls"] += 1
        self.stats["raw"] += batch.raw_count
        self.stats["rejected"] += len(batch.rejected)

        if batch.errored:
            # Do NOT cache. A rate-limited quarter written as {"views": []} looks
            # identical to a quarter the model genuinely had no opinion on, and
            # it is never retried -- which silently turns the llm arm into the
            # equilibrium arm for however long the outage lasted.
            self.stats["failed"] = self.stats.get("failed", 0) + 1
            if self.verbose:
                print(f"  {as_of}  API failure, not cached: "
                      f"{batch.rejected[0][:90]}", flush=True)
            return batch

        path.write_text(json.dumps({"views": [v.model_dump() for v in batch.views],
                                    "model": batch.model}))

        if self.verbose:
            kept = ", ".join(
                f"{v.asset}{'>' + v.versus if v.versus else ''} {v.expected_return:+.1%}"
                f"@{v.confidence:.2f}" for v in batch.views
            ) or "none"
            print(f"  {as_of}  {len(batch.views)}/{batch.raw_count} views kept: {kept}",
                  flush=True)
        return batch


def cmd_prices(args):
    df = px.download(start=args.start, end=args.end)
    print(df.tail(3).to_string())
    print(f"\n{df.shape[0]} days x {df.shape[1]} symbols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")


def cmd_views(args):
    tickers = px.DEFAULT_UNIVERSE
    as_of = date.fromisoformat(args.date)
    evidence = filings.evidence_by_ticker(tickers, as_of=as_of)
    chars = sum(len(v) for v in evidence.values())
    print(f"evidence: {chars:,} chars across {len(evidence)} tickers, "
          f"all filed before {as_of}\n")
    if not evidence:
        print("no filings in the lookback window")
        return

    batch = views_mod.generate_chunked(evidence, tickers, model=args.model)
    print(f"{len(batch.views)} kept of {batch.raw_count} raw "
          f"({batch.rejection_rate:.0%} rejected)\n")
    for v in batch.views:
        target = f"{v.asset} vs {v.versus}" if v.versus else v.asset
        print(f"  {target:<16} {v.expected_return:+.2%}  conf {v.confidence:.2f}  {v.rationale}")
    for r in batch.rejected:
        print(f"  REJECTED: {r}")


class StaticViews:
    """One fixed view set, replayed at every rebalance.

    The `shuffled` control destroys which assets a view is about, so it removes
    the model's stock selection *and* any persistent tilt at the same time. That
    makes `llm` beating `shuffled` ambiguous: it is consistent with reading each
    quarter's filings, and equally consistent with restating one ranking forever.

    The views say the second is worth checking. Across 39 quarters there are
    only 58 distinct pairs, and AMZN over BAC shows up in 82% of them. This arm
    takes the most common opinions, at their median magnitude and confidence,
    and applies them unchanged on every date -- so it holds the tilt and throws
    away everything quarter-specific.

    It is deliberately given hindsight, since the modal view set is computed
    over the whole sample. That makes it a harder baseline than `llm` deserves,
    which is the direction a control should err in.
    """

    def __init__(self, cache_dir: Path, n_views: int = 8):
        counts: Counter = Counter()
        returns, confidences = defaultdict(list), defaultdict(list)

        for path in sorted(cache_dir.glob("*.json")):
            for v in json.loads(path.read_text()).get("views", []):
                key = (v["asset"], v.get("versus"))
                counts[key] += 1
                returns[key].append(v["expected_return"])
                confidences[key].append(v["confidence"])

        self.views = [
            views_mod.View(
                asset=asset, versus=versus,
                expected_return=float(np.median(returns[(asset, versus)])),
                confidence=float(np.median(confidences[(asset, versus)])),
                rationale=f"modal view, seen in {n} of {len(list(cache_dir.glob('*.json')))} quarters",
            )
            for (asset, versus), n in counts.most_common(n_views)
        ]
        self.stats = {"static_views": len(self.views), "distinct_pairs": len(counts)}

    def __call__(self, as_of, tickers):
        return views_mod.ViewBatch(views=self.views, raw_count=len(self.views),
                                   rejected=[])


def cmd_backtest(args):
    prices_df = px.download(start=args.start, end=args.end)
    tickers = [c for c in prices_df.columns if c != px.BENCHMARK]
    print(f"{len(tickers)} assets, {prices_df.index[0].date()} to "
          f"{prices_df.index[-1].date()}\n")

    provider = CachedViews(args.model) if {"llm", "shuffled"} & set(args.arms) else None
    if provider:
        print("generating views (cached per rebalance date):")

    static = StaticViews(VIEW_CACHE) if "static" in args.arms else None
    if static:
        print(f"\nstatic control: {static.stats}")
        for v in static.views:
            print(f"  {v.asset}{'>' + v.versus if v.versus else ''} "
                  f"{v.expected_return:+.1%} @{v.confidence:.2f}")

    results = []
    for arm in args.arms:
        if arm == "static":
            view_provider = static
        elif arm in ("llm", "shuffled"):
            view_provider = provider
        else:
            view_provider = None
        results.append(backtest.run(
            prices_df, arm=arm, tickers=tickers, view_provider=view_provider,
            max_weight=args.max_weight, freq=args.freq,
        ))

    bench = backtest.benchmark_equity(prices_df, results[0].equity)
    table = backtest.compare(results, bench)

    print("\n=== full period ===")
    print(table.to_string(float_format=lambda v: f"{v:8.3f}"))

    holdout_table = None
    if args.holdout_start:
        held = [r.since(args.holdout_start) for r in results]
        held_bench = bench.loc[bench.index >= pd.Timestamp(args.holdout_start)]
        held_bench = held_bench / held_bench.iloc[0]
        holdout_table = backtest.compare(held, held_bench)
        print(f"\n=== held out, from {args.holdout_start} "
              f"(never used to choose anything) ===")
        print(holdout_table.to_string(float_format=lambda v: f"{v:8.3f}"))

    if provider:
        print(f"\nview generation: {provider.stats}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "config": {
            "start": args.start, "end": args.end, "freq": args.freq,
            "max_weight": args.max_weight, "model": args.model,
            "cost_per_turnover": backtest.COST_PER_TURNOVER,
            "universe": tickers,
        },
        "stats": {k: {m: float(x) for m, x in v.items()} for k, v in table.T.items()},
        "holdout_start": args.holdout_start,
        "holdout_stats": (
            {k: {m: float(x) for m, x in v.items()} for k, v in holdout_table.T.items()}
            if holdout_table is not None else None
        ),
        "views": provider.stats if provider else None,
    }, indent=2))
    print(f"\nresults -> {RESULTS}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prices", help="download and cache the price history")
    a.add_argument("--start", default="2015-01-01")
    a.add_argument("--end", default=None)
    a.set_defaults(func=cmd_prices)

    b = sub.add_parser("views", help="generate views for one date and show them")
    b.add_argument("--date", default=str(date.today()))
    b.add_argument("--model", default=None)
    b.set_defaults(func=cmd_views)

    c = sub.add_parser("backtest", help="walk-forward backtest across arms")
    c.add_argument("--start", default="2015-01-01")
    c.add_argument("--end", default=None)
    c.add_argument("--arms", nargs="+",
                   default=["equilibrium", "llm", "shuffled", "equal"])
    c.add_argument("--freq", default="QE")
    c.add_argument("--max-weight", type=float, default=0.25)
    c.add_argument("--model", default=None, help="pin one model; default walks MODEL_CHAIN")
    c.add_argument("--holdout-start", default=None,
                   help="report a second table measured only from this date, "
                        "for evaluation on a window never used for tuning")
    c.set_defaults(func=cmd_backtest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
