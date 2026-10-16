"""Point-in-time backtest: would the goal grades have meant anything?

For each quarterly as-of date and each company, compute metrics and goal
scores using only data available on that date (filed-date filter for XBRL,
prices up to the date), with own-history and peer-universe percentiles. Then
measure what happened next, with a goal-specific outcome:

  value      forward total return minus SPY (1 and 3 years)
  dividend   trailing-12-month dividends 2 years later: cut (>5% lower)? growth?
  growth     realized revenue CAGR over the next 2 years
  stability  realized max drawdown and volatility over the next year
  horizon    stated chance of loss vs what actually happened (Brier score)

Plus the information coefficient (IC): per-date Spearman rank correlation of
score vs forward excess return, averaged across dates with a t-stat. The IC by
horizon feeds the timeline report's "how much do fundamentals matter" line.

Split: dates through 2020 = train (tune here), 2021+ = test (report once).
Caveats are written into every report: survivorship bias (today's large caps),
small universe, no costs or taxes.

  python -m finrag.ratings.backtest --universe config/universe/large_caps.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from finrag.ratings.fundamentals import FactStore
from finrag.ratings.goals import GOAL_LABELS, grade_for, score_goal
from finrag.ratings.horizon import outcomes as horizon_outcomes
from finrag.ratings.macro import TreasuryYield
from finrag.ratings.metrics import Metrics, compute_metrics
from finrag.ratings.prices import PriceSeries

STEP = timedelta(days=91)
TRAIN_END = date(2020, 12, 31)
HORIZONS = (91, 182, 365, 730, 1095)
CAVEATS = [
    "Survivorship bias: the universe is today's large caps, which by definition survived and grew.",
    "Small sample: ~100 companies x ~40 quarters, with overlapping forward windows.",
    "No transaction costs, taxes, or slippage. Past relationships may not persist.",
]


@dataclass
class Row:
    ticker: str
    as_of: date
    scores: dict[str, float | None]
    loss_chance: dict[int, float | None] = field(default_factory=dict)  # stated, per horizon
    outcome: dict[str, float | None] = field(default_factory=dict)


def as_of_dates(start: date, end: date) -> list[date]:
    """Quarterly grid ending at `end` (shared grid: history lookups reuse cached metrics)."""
    out, d = [], end
    while d >= start:
        out.append(d)
        d -= STEP
    return sorted(out)


class MetricCache:
    """metrics(ticker, date), memoized: history percentiles reuse grid dates."""

    def __init__(self, stores: dict[str, FactStore], prices: dict[str, PriceSeries], market: PriceSeries,
                 treasury: TreasuryYield):  # fmt: skip
        self.stores, self.prices, self.market, self.treasury = stores, prices, market, treasury
        self._memo: dict[tuple[str, date], Metrics | None] = {}

    def get(self, ticker: str, d: date) -> Metrics | None:
        key = (ticker, d)
        if key not in self._memo:
            snap = self.stores[ticker].snapshot(d)
            self._memo[key] = (
                None if snap.period_end is None
                else compute_metrics(snap, self.prices[ticker], self.market, self.treasury.on(d))
            )  # fmt: skip
        return self._memo[key]

    def history(self, ticker: str, d: date, years: int = 5) -> list[Metrics]:
        return [m for k in range(1, years * 4 + 1) if (m := self.get(ticker, d - STEP * k))]


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 5:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def forward_outcomes(ticker: str, d: date, prices: PriceSeries, market: PriceSeries,
                     store: FactStore) -> dict[str, float | None]:  # fmt: skip
    """What actually happened after d. Uses future data on purpose (it's the answer key)."""
    out: dict[str, float | None] = {}
    last = prices.dates[-1]
    for h in HORIZONS:
        end = d + timedelta(days=h)
        if end > last:
            out[f"excess_{h}"] = out[f"return_{h}"] = None
            continue
        r, m = prices.total_return(d, end), market.total_return(d, end)
        out[f"return_{h}"] = r
        out[f"excess_{h}"] = None if r is None or m is None else r - m
    # dividends: trailing-12m paid now vs 2 years later
    if d + timedelta(days=730) <= last:
        now, later = (
            prices.trailing_dividends(d),
            prices.trailing_dividends(d + timedelta(days=730)),
        )
        out["dividend_cut"] = float(later < now * 0.95) if now > 0 else None
        out["dividend_growth_2y"] = (later / now - 1) if now > 0 else None
    # growth: realized revenue CAGR over 2 years (as reported by ~a quarter after the window)
    rev_now = store.ttm("revenue", d)
    rev_later = store.ttm("revenue", d + timedelta(days=830), target_end=(rev_now.period_end + timedelta(days=730))
                          if rev_now else None)  # fmt: skip
    if rev_now and rev_later and rev_now.value > 0 and rev_later.value > 0:
        out["revenue_cagr_2y"] = (rev_later.value / rev_now.value) ** 0.5 - 1
    # stability: what the next year looked like
    if d + timedelta(days=365) <= last:
        out["drawdown_1y"] = prices.max_drawdown(d + timedelta(days=365), days=252)
        out["volatility_1y"] = prices.volatility(d + timedelta(days=365), days=252)
    return out


def build_panel(cache: MetricCache, dates: list[date], goals: list[str],
                score_fn: Callable = score_goal) -> list[Row]:  # fmt: skip
    rows = []
    for d in dates:
        current = {t: m for t in cache.stores if (m := cache.get(t, d))}
        universe = list(current.values())
        for t, m in current.items():
            hist = cache.history(t, d)
            scores = {g: score_fn(g, m, hist, universe).score for g in goals}
            stated = {}
            for h in HORIZONS:
                o = horizon_outcomes(cache.prices[t], d, h)
                stated[h] = o.chance_of_loss if o else None
            rows.append(Row(t, d, scores, stated,
                            forward_outcomes(t, d, cache.prices[t], cache.market, cache.stores[t])))  # fmt: skip
    return rows


GOAL_OUTCOME = {"value": "excess_365", "dividend": "dividend_growth_2y", "growth": "revenue_cagr_2y",
                "stability": "drawdown_1y"}  # fmt: skip


def information_coefficient(rows: list[Row], goal: str, outcome: str) -> dict:
    by_date: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        s, o = r.scores.get(goal), r.outcome.get(outcome)
        if s is not None and o is not None:
            by_date[r.as_of].append((s, o))
    ics = [
        ic
        for pairs in by_date.values()
        if (ic := spearman([p[0] for p in pairs], [p[1] for p in pairs])) is not None
    ]
    if len(ics) < 3:
        return {"ic": None, "t": None, "dates": len(ics)}
    mean, sd = statistics.mean(ics), statistics.stdev(ics)
    return {
        "ic": round(mean, 4),
        "t": round(mean / (sd / math.sqrt(len(ics))), 2) if sd else None,
        "dates": len(ics),
    }


def by_grade(rows: list[Row], goal: str, outcome: str) -> dict[str, dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        s, o = r.scores.get(goal), r.outcome.get(outcome)
        if s is not None and o is not None:
            groups[grade_for(s)].append(o)
    return {g: {"n": len(v), "mean": round(statistics.mean(v), 4)} for g, v in groups.items()}


def calibration(rows: list[Row]) -> dict[int, dict]:
    """Stated chance of loss vs realized losses, per horizon (Brier score; lower is better)."""
    out = {}
    for h in HORIZONS:
        pairs = [(r.loss_chance[h], float(r.outcome[f"return_{h}"] < 0))
                 for r in rows if r.loss_chance.get(h) is not None and r.outcome.get(f"return_{h}") is not None]  # fmt: skip
        if pairs:
            brier = statistics.mean((p - y) ** 2 for p, y in pairs)
            out[h] = {"n": len(pairs), "brier": round(brier, 4),
                      "mean_stated": round(statistics.mean(p for p, _ in pairs), 3),
                      "realized": round(statistics.mean(y for _, y in pairs), 3)}  # fmt: skip
    return out


def evaluate(rows: list[Row], goals: list[str]) -> dict:
    report: dict = {"caveats": CAVEATS, "splits": {}}
    for name, subset in (("train", [r for r in rows if r.as_of <= TRAIN_END]),
                         ("test", [r for r in rows if r.as_of > TRAIN_END])):  # fmt: skip
        split: dict = {"rows": len(subset), "goals": {}, "ic_by_horizon": {}}
        for g in goals:
            outcome = GOAL_OUTCOME[g]
            split["goals"][g] = {"outcome": outcome, "ic": information_coefficient(subset, g, outcome),
                                 "by_grade": by_grade(subset, g, outcome)}  # fmt: skip
        if "value" in goals:
            split["ic_by_horizon"] = {
                h: information_coefficient(subset, "value", f"excess_{h}")["ic"] for h in HORIZONS
            }
        split["calibration"] = calibration(subset)
        report["splits"][name] = split
    return report


def random_control(rows: list[Row], goals: list[str], seed: int = 0) -> dict:
    """Same evaluation with random scores: IC should be ~0. If it isn't, the pipeline leaks."""
    rng = random.Random(seed)
    shuffled = [
        Row(r.ticker, r.as_of, {g: rng.uniform(0, 100) for g in goals}, r.loss_chance, r.outcome)
        for r in rows
    ]
    return {g: information_coefficient(shuffled, g, GOAL_OUTCOME[g]) for g in goals}


def main() -> None:
    import yaml

    from finrag.ingest.edgar_client import EdgarClient
    from finrag.ingest.xbrl import load_company_facts
    from finrag.ratings.macro import FredClient
    from finrag.ratings.prices import TiingoClient
    from finrag.settings import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, default=Path("config/universe/large_caps.yaml"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=30))
    parser.add_argument("--goals", nargs="+", default=list(GOAL_LABELS))
    args = parser.parse_args()

    s = get_settings()
    tickers = yaml.safe_load(args.universe.read_text(encoding="utf-8"))["tickers"]
    edgar, tiingo = (
        EdgarClient(s.sec_user_agent),
        TiingoClient(s.tiingo_api_key, s.data_dir / "prices"),
    )
    stores, prices = {}, {}
    for t in tickers:
        try:
            stores[t] = FactStore(
                load_company_facts(edgar, edgar.get_cik(t), s.data_dir / "xbrl", 7 * 86400)
            )
            prices[t] = tiingo.series(t, start=args.start - timedelta(days=6 * 365))
        except (KeyError, RuntimeError) as exc:
            print(f"skip {t}: {exc}")
    market = tiingo.series("SPY", start=args.start - timedelta(days=6 * 365))
    cache = MetricCache(
        stores, prices, market, FredClient(s.fred_api_key, s.data_dir / "macro").treasury_10y()
    )

    rows = build_panel(cache, as_of_dates(args.start, args.end), args.goals)
    report = evaluate(rows, args.goals) | {"random_control": random_control(rows, args.goals),
                                           "universe": len(stores), "as_of_dates": len({r.as_of for r in rows})}  # fmt: skip
    # Under data_dir: the only writable path in the hardened container (read-only root).
    out = s.data_dir / "backtest" / f"backtest-{date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    # The horizon report reads measured IC from here (test split = out-of-sample).
    ic = {
        int(h): v
        for h, v in report["splits"]["test"].get("ic_by_horizon", {}).items()
        if v is not None
    }
    (s.data_dir / "cache").mkdir(parents=True, exist_ok=True)
    (s.data_dir / "cache" / "ic_by_horizon.json").write_text(json.dumps(ic), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"} | {
        "test_ic": {g: v["ic"] for g, v in report["splits"]["test"]["goals"].items()}}, indent=1, default=str))  # fmt: skip
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
