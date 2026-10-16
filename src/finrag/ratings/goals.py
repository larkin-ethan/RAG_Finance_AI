"""Goal recipes: how well a stock fits a goal (dividend income, growth, value,
stability), as a 0-100 score and a grade.

Each metric's score = 50% absolute level (anchored scale) + 50% relative
standing (percentile vs the company's own last 5 years and, when given, a peer
universe). Relative-only scoring was tried first and failed: a 0.4% yield that's
high *for Apple* scored as a "Strong" dividend fit. Absolute levels matter for
goal fit; relative standing adds "cheap or strong for this company". Metrics
that only make sense on a fixed scale (Piotroski, Altman Z, years without a
dividend cut) are absolute only, as is any metric without enough history.

Guardrails cap a grade whatever the weighted score says (e.g. a dividend
larger than free cash flow can't be a "Strong" dividend fit).

Grade cutoffs are provisional: they get re-fit on the 2015-2020 backtest window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from finrag.ratings.metrics import Metrics, compute_metrics

Goal = Literal["dividend", "growth", "value", "stability"]
GOAL_LABELS = {"dividend": "Dividend income", "growth": "Growth", "value": "Value",
               "stability": "Stability / low risk"}  # fmt: skip
MIN_COVERAGE = 0.6  # share of a recipe's weight that must have data
TOKEN_YIELD = 0.005  # below this, a dividend is token: "Not a dividend stock"
LOW_INCOME_YIELD = 0.015  # below this, the dividend grade is capped at Weak
MIN_HISTORY, MIN_UNIVERSE = 4, 5
GRADES = [(70, "Strong fit"), (50, "Moderate fit"), (30, "Weak fit"), (0, "Poor fit")]
CAP = 49  # a guardrail caps the grade at "Weak fit"


@dataclass(frozen=True)
class Component:
    metric: str
    label: str
    weight: float
    higher_is_better: bool
    anchor: tuple[float, float]  # (worst, best) on an absolute scale
    absolute_only: bool = False


C = Component
RECIPES: dict[str, list[Component]] = {
    "dividend": [
        C("dividend_yield", "Dividend yield", 0.20, True, (0.0, 0.05)),
        C("dividend_yield_spread", "Yield vs 10-year Treasury", 0.10, True, (-0.03, 0.01), True),
        C("payout_ratio", "Payout ratio (of earnings)", 0.15, False, (1.0, 0.4), True),
        C("fcf_payout", "Payout of free cash flow (3 yrs)", 0.15, False, (1.1, 0.5), True),
        C("dividend_cagr_5y", "5-year dividend growth", 0.15, True, (-0.02, 0.10)),
        C("years_without_cut", "Years without a dividend cut", 0.15, True, (0, 15), True),
        C("net_debt_to_ebitda", "Net debt / EBITDA", 0.05, False, (4.0, 0.0)),
        C("interest_coverage", "Interest coverage", 0.05, True, (2.0, 15.0), True),
    ],
    "growth": [
        C("revenue_growth_yoy", "Revenue growth (TTM)", 0.25, True, (-0.05, 0.25)),
        C("eps_growth_yoy", "EPS growth (TTM)", 0.15, True, (-0.10, 0.30)),
        C("revenue_cagr_3y", "3-year revenue CAGR", 0.20, True, (-0.02, 0.20)),
        C("operating_margin_change", "Operating margin trend", 0.15, True, (-0.05, 0.05)),
        C("reinvestment_rate", "Reinvestment (capex + R&D) / revenue", 0.10, True, (0.0, 0.20)),
        C("peg", "P/E relative to growth (PEG)", 0.15, False, (3.0, 0.5)),
    ],
    "value": [
        C("earnings_yield", "Earnings yield", 0.25, True, (0.02, 0.10)),
        C("fcf_yield", "Free-cash-flow yield", 0.25, True, (0.01, 0.08)),
        C("price_to_book", "Price / book", 0.10, False, (8.0, 1.0)),
        C("ev_to_ebit", "EV / EBIT", 0.20, False, (30.0, 8.0)),
        C("earnings_yield_spread", "Earnings yield vs 10-year Treasury", 0.20, True, (-0.03, 0.04), True),
    ],
    "stability": [
        C("earnings_variability", "Earnings variability (5 yrs)", 0.20, False, (0.6, 0.05), True),
        C("altman_z", "Altman Z (bankruptcy risk)", 0.15, True, (1.8, 4.0), True),
        C("interest_coverage", "Interest coverage", 0.10, True, (2.0, 20.0), True),
        C("volatility", "Price volatility (1 yr)", 0.20, False, (0.5, 0.15)),
        C("max_drawdown", "Worst drawdown (3 yrs)", 0.15, True, (-0.6, -0.15)),
        C("beta", "Beta vs S&P 500", 0.10, False, (1.6, 0.6)),
        C("risk_change_share", "New risk-factor text vs last year", 0.10, False, (0.5, 0.0), True),
    ],
}  # fmt: skip


@dataclass
class ComponentScore:
    metric: str
    label: str
    value: float | None
    score: float | None  # 0-100
    weight: float
    basis: str  # "history", "universe", "history+universe", "absolute", "missing"


@dataclass
class GoalResult:
    goal: str
    label: str
    score: float | None
    grade: str
    components: list[ComponentScore] = field(default_factory=list)
    caps: list[str] = field(default_factory=list)  # guardrails that limited the grade
    coverage: float = 0.0


def percentile_score(value: float, refs: list[float], higher_is_better: bool) -> float:
    below = sum(r < value for r in refs) + 0.5 * sum(r == value for r in refs)
    pct = 100 * below / len(refs)
    return pct if higher_is_better else 100 - pct


def anchor_score(value: float, worst: float, best: float) -> float:
    if best == worst:
        return 50.0
    return max(0.0, min(100.0, 100 * (value - worst) / (best - worst)))


def grade_for(score: float) -> str:
    return next(label for cutoff, label in GRADES if score >= cutoff)


def _guardrails(goal: str, m: Metrics) -> list[str]:
    caps = []
    if goal == "dividend":
        if (m.get("dividend_yield") or 0) < LOW_INCOME_YIELD:
            caps.append(f"Yield below {LOW_INCOME_YIELD:.1%}: limited income")
        if (m.get("fcf_payout") or 0) > 1.0:
            caps.append("Dividends exceeded free cash flow over the last 3 years")
        if (m.get("payout_ratio") or 0) > 1.0:
            caps.append("Dividend is larger than earnings per share")
    elif goal == "growth":
        if (m.get("revenue_growth_yoy") or 0) < 0 and (m.get("revenue_cagr_3y") or 0) < 0:
            caps.append("Revenue is shrinking (TTM and 3-year)")
    elif goal == "value":
        if m.get("piotroski") is not None and m["piotroski"] <= 3:
            caps.append(f"Weak fundamentals (Piotroski {m['piotroski']}/9): possible value trap")
        if (m.get("revenue_growth_yoy") or 0) < -0.05:
            caps.append("Revenue down more than 5%: possible value trap")
    elif goal == "stability":
        if m.get("red_flags"):
            caps.append("New red flag in filings: " + ", ".join(m["red_flags"]))
        if m.get("altman_z") is not None and m["altman_z"] < 1.8:
            caps.append("Altman Z in the distress zone (< 1.8)")
    return caps


def score_goal(
    goal: str,
    metrics: Metrics,
    history: list[Metrics] | None = None,
    universe: list[Metrics] | None = None,
) -> GoalResult:
    label = GOAL_LABELS[goal]
    if goal == "dividend" and (metrics.get("dividend_yield") or 0) < TOKEN_YIELD:
        return GoalResult(goal, label, None, "Not a dividend stock")

    parts: list[ComponentScore] = []
    for comp in RECIPES[goal]:
        value = metrics.get(comp.metric)
        if value is None:
            parts.append(
                ComponentScore(comp.metric, comp.label, None, None, comp.weight, "missing")
            )
            continue
        absolute = anchor_score(value, *comp.anchor)
        relative, basis = [], ["absolute"]
        if not comp.absolute_only:
            hist = [h[comp.metric] for h in history or [] if h.get(comp.metric) is not None]
            peers = [u[comp.metric] for u in universe or [] if u.get(comp.metric) is not None]
            if len(hist) >= MIN_HISTORY:
                relative.append(percentile_score(value, hist, comp.higher_is_better))
                basis.append("history")
            if len(peers) >= MIN_UNIVERSE:
                relative.append(percentile_score(value, peers, comp.higher_is_better))
                basis.append("universe")
        score = absolute if not relative else 0.5 * absolute + 0.5 * sum(relative) / len(relative)
        parts.append(
            ComponentScore(comp.metric, comp.label, value, score, comp.weight, "+".join(basis))
        )

    available = [p for p in parts if p.score is not None]
    total = sum(c.weight for c in RECIPES[goal])
    coverage = sum(p.weight for p in available) / total
    if coverage < MIN_COVERAGE:
        return GoalResult(goal, label, None, "Insufficient data", parts, coverage=coverage)
    score = sum(p.score * p.weight for p in available) / sum(p.weight for p in available)
    caps = _guardrails(goal, metrics)
    if caps:
        score = min(score, CAP)
    return GoalResult(
        goal, label, round(score, 1), grade_for(score), parts, caps, round(coverage, 2)
    )


def metric_history(store, prices, market, treasury, as_of: date, years: int = 5) -> list[Metrics]:
    """Metrics at each quarter-end over the past `years`, each computed as of that
    date (point-in-time): the company's own baseline for percentile scoring."""
    out = []
    for k in range(1, years * 4 + 1):
        d = as_of - timedelta(days=91 * k)
        snap = store.snapshot(d)
        if snap.period_end is None:
            break
        out.append(compute_metrics(snap, prices, market, treasury.on(d) if treasury else None))
    return out
