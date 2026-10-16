import random
from datetime import date, timedelta

import pytest

from finrag.ratings.backtest import (
    MetricCache,
    as_of_dates,
    build_panel,
    evaluate,
    random_control,
    spearman,
)
from finrag.ratings.fundamentals import FactStore
from finrag.ratings.macro import TreasuryYield
from finrag.ratings.prices import PriceSeries


def company_facts(growth: float, revenue_bump_after: date | None = None) -> dict:
    """Annual 10-K facts 2010-2025, each filed 60 days after fiscal year end."""
    facts = {"revenue": [], "ni": [], "eps": []}
    rev = 1000.0
    for y in range(2010, 2026):
        rev *= 1 + growth
        filed = date(y + 1, 3, 1)
        value = rev * (3 if revenue_bump_after and filed > revenue_bump_after else 1)
        base = {
            "start": f"{y}-01-01",
            "end": f"{y}-12-31",
            "form": "10-K",
            "filed": filed.isoformat(),
        }
        facts["revenue"].append(base | {"val": value})
        facts["ni"].append(base | {"val": value * 0.1})
        facts["eps"].append(base | {"val": value * 0.1 / 100})
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": facts["revenue"]}},
        "NetIncomeLoss": {"units": {"USD": facts["ni"]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": facts["eps"]}},
    }}}  # fmt: skip


def walk(seed: int, crash_after: date | None = None) -> PriceSeries:
    rng = random.Random(seed)
    days = [date(2008, 1, 1) + timedelta(days=i) for i in range(int(365.25 * 18))]
    vals = [100.0]
    for d in days[1:]:
        step = 1 + 0.0003 + rng.gauss(0, 0.01)
        if crash_after and d > crash_after:
            step *= 0.99
        vals.append(vals[-1] * step)
    return PriceSeries(f"T{seed}", days, vals, vals, [0.0] * len(days))


def universe(bump_after=None, crash_after=None):
    growths = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    stores = {f"T{i}": FactStore(company_facts(g, bump_after)) for i, g in enumerate(growths)}
    prices = {f"T{i}": walk(i, crash_after) for i in range(len(growths))}
    return MetricCache(stores, prices, walk(99), TreasuryYield([(date(2008, 1, 1), 3.0)]))


DATES = as_of_dates(date(2016, 1, 1), date(2022, 12, 31))


def test_spearman():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2], [1, 2]) is None


def test_growth_scores_rank_persistent_growers():
    rows = build_panel(universe(), DATES, ["growth"])
    report = evaluate(rows, ["growth"])
    ic = report["splits"]["train"]["goals"]["growth"]["ic"]
    assert ic["ic"] > 0.8 and ic["t"] > 3
    assert all(abs(v["ic"] or 0) < 0.35 for v in random_control(rows, ["growth"]).values())
    assert "Survivorship bias" in report["caveats"][0]


def test_no_look_ahead_future_data_cannot_change_past_scores():
    cutoff = date(2019, 6, 30)
    past = [d for d in DATES if d <= cutoff]
    before = build_panel(universe(), past, ["growth", "value", "stability"])
    tampered = build_panel(
        universe(bump_after=cutoff, crash_after=cutoff), past, ["growth", "value", "stability"]
    )
    assert [r.scores for r in before] == [r.scores for r in tampered]
    assert [r.loss_chance for r in before] == [r.loss_chance for r in tampered]
