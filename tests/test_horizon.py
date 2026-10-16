import random
from datetime import date, timedelta

import pytest

from finrag.ratings.horizon import expected_dividends, horizon_report, outcomes
from finrag.ratings.prices import PriceSeries


def walk(n=3000, drift=0.0004, vol=0.01, seed=0, ticker="X", divs=None):
    rng = random.Random(seed)
    start = date(2015, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    vals = [100.0]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + drift + rng.gauss(0, vol)))
    d = [0.0] * n
    for i, amt in (divs or {}).items():
        d[i] = amt
    return PriceSeries(ticker, dates, vals, vals, d)


def test_outcomes_shape_and_up_trend_lowers_loss_chance():
    up, flat = walk(drift=0.001), walk(drift=-0.0005)
    o_up, o_flat = outcomes(up, up.dates[-1], 365), outcomes(flat, flat.dates[-1], 365)
    assert o_up.p10 <= o_up.median <= o_up.p90
    assert o_up.chance_of_loss < o_flat.chance_of_loss


def test_too_little_history():
    short = walk(n=200)
    assert outcomes(short, short.dates[-1], 365) is None


def test_expected_dividends_projects_cadence():
    s = walk(divs={2900: 0.5, 2990: 0.5})
    as_of = s.dates[-1]
    # Payments on days 2900 and 2990 recur 365 days later: days 3265 and 3355.
    assert len(expected_dividends(s, as_of, as_of + timedelta(days=300))) == 1  # through day 3299
    proj = expected_dividends(s, as_of, as_of + timedelta(days=400))  # through day 3399
    assert [amt for _, amt in proj] == [0.5, 0.5]
    assert all(as_of < d <= as_of + timedelta(days=400) for d, _ in proj)


def test_fit_leans_on_stability_short_and_goal_long():
    s, m = walk(seed=1), walk(seed=2, ticker="SPY")
    as_of = s.dates[-1]
    short = horizon_report(
        s, m, as_of, as_of + timedelta(days=60), stability_score=20, goal_score=90
    )
    long = horizon_report(
        s, m, as_of, as_of + timedelta(days=5 * 365), stability_score=20, goal_score=90
    )
    assert long.fit_score == 90  # beyond 3 years: all goal
    assert short.fit_score < long.fit_score
    assert short.notes and "noise" in short.notes[0]
    assert short.market is not None


def test_fundamentals_note_requires_measured_ic():
    s = walk()
    as_of = s.dates[-1]
    r = horizon_report(s, None, as_of, as_of + timedelta(days=180), stability_score=50)
    assert r.fundamentals_ic is None and "Not yet measured" in r.fundamentals_note
    r2 = horizon_report(s, None, as_of, as_of + timedelta(days=180), stability_score=50,
                        ic_by_horizon={91: 0.01, 365: 0.06, 1095: 0.09})  # fmt: skip
    assert r2.fundamentals_ic == 0.01 and "weak" in r2.fundamentals_note


def test_sell_date_must_be_in_future():
    s = walk()
    with pytest.raises(ValueError):
        horizon_report(s, None, s.dates[-1], s.dates[-1], stability_score=50)
