"""Timeline check: "I plan to hold this for N months (or until date X)".

Short horizons are dominated by price noise, so instead of a confident
hold/sell call this reports, for the stated holding period:
  * historical outcomes: chance of a loss, and the 10th/50th/90th percentile
    return over every past window of that length, for the stock and for SPY
  * how much fundamentals mattered at this horizon, but only as *measured by
    our backtest* (information coefficient by horizon). Until the backtest has
    run, this says "not yet measured" rather than citing an unverified number.
  * dividends expected in the window (from the recent payment cadence)
  * a horizon fit grade that leans on stability and loss risk at short
    horizons and shifts toward the chosen goal's score as the horizon grows
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from finrag.ratings.goals import anchor_score, grade_for
from finrag.ratings.prices import PriceSeries

SHORT_DAYS, LONG_DAYS = 90, 3 * 365  # where the blend is all-stability vs all-goal


@dataclass
class Outcomes:
    windows: int
    chance_of_loss: float
    p10: float
    median: float
    p90: float


@dataclass
class HorizonReport:
    ticker: str
    as_of: date
    end: date
    days: int
    stock: Outcomes | None
    market: Outcomes | None
    expected_dividends: list[tuple[date, float]] = field(default_factory=list)  # estimated
    fundamentals_ic: float | None = None  # from the backtest at the nearest horizon
    fundamentals_note: str = ""
    fit_score: float | None = None
    fit_grade: str = "Insufficient data"
    notes: list[str] = field(default_factory=list)


def outcomes(prices: PriceSeries, as_of: date, days: int) -> Outcomes | None:
    rets = prices.rolling_returns(as_of, days)
    if len(rets) < 26:  # under ~6 months of weekly windows: too few to describe
        return None
    q = statistics.quantiles(rets, n=10)
    return Outcomes(
        len(rets), sum(r < 0 for r in rets) / len(rets), q[0], statistics.median(rets), q[-1]
    )


def expected_dividends(prices: PriceSeries, as_of: date, end: date) -> list[tuple[date, float]]:
    """Project the last year's payments forward (same amounts, same spacing)."""
    last_year = prices.dividends_between(as_of - timedelta(days=365), as_of)
    if not last_year:
        return []
    projected = []
    for paid_on, amount in last_year:
        nxt = paid_on + timedelta(days=365)
        while nxt <= end:
            if nxt > as_of:
                projected.append((nxt, amount))
            nxt += timedelta(days=365)
    return sorted(projected)


def horizon_report(
    prices: PriceSeries,
    market: PriceSeries | None,
    as_of: date,
    end: date,
    stability_score: float | None,
    goal_score: float | None = None,
    ic_by_horizon: dict[int, float] | None = None,
) -> HorizonReport:
    days = (end - as_of).days
    if days <= 0:
        raise ValueError("The sell date must be after today")
    rep = HorizonReport(prices.ticker, as_of, end, days, outcomes(prices, as_of, days),
                        outcomes(market, as_of, days) if market else None)  # fmt: skip
    rep.expected_dividends = expected_dividends(prices, as_of, end)

    if ic_by_horizon:
        nearest = min(ic_by_horizon, key=lambda h: abs(h - days))
        rep.fundamentals_ic = ic_by_horizon[nearest]
        strength = (
            "weak"
            if abs(rep.fundamentals_ic) < 0.03
            else "modest"
            if abs(rep.fundamentals_ic) < 0.08
            else "meaningful"
        )
        rep.fundamentals_note = (
            f"In our backtest, fundamentals had a {strength} link to returns over ~{nearest} days "
            f"(information coefficient {rep.fundamentals_ic:+.2f})."
        )
    else:
        rep.fundamentals_note = "Not yet measured: run the backtest to see how much fundamentals matter at this horizon."

    # Blend: short horizons -> stability + loss risk; long horizons -> the goal's score.
    w = min(1.0, max(0.0, (days - SHORT_DAYS) / (LONG_DAYS - SHORT_DAYS)))
    near_term = [s for s in (stability_score,
                             anchor_score(rep.stock.chance_of_loss, 0.5, 0.1) if rep.stock else None)
                 if s is not None]  # fmt: skip
    near = sum(near_term) / len(near_term) if near_term else None
    if near is None and goal_score is None:
        return rep
    if goal_score is None or near is None:
        rep.fit_score = goal_score if near is None else near
    else:
        rep.fit_score = (1 - w) * near + w * goal_score
    rep.fit_score = round(rep.fit_score, 1)
    rep.fit_grade = grade_for(rep.fit_score)
    if days < 365:
        rep.notes.append(
            "Over less than a year, price moves are mostly noise; treat this as a risk view, not a forecast."
        )
    return rep
