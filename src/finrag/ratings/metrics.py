"""Financial metrics for the goal recipes, computed point-in-time.

`compute_metrics(snapshot, prices, market, treasury_yield)` combines a
fundamentals Snapshot (fundamentals.py) with the price on the same date.
Every metric is `None` when an input is missing or a denominator is zero or
negative where that makes the ratio meaningless (e.g. P/E with a loss).
A missing metric is skipped by the scorer; a wrong one would corrupt it.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta

from finrag.ratings.fundamentals import Snapshot
from finrag.ratings.prices import PriceSeries

Metrics = dict[str, float | None]


def _div(a: float | None, b: float | None, positive_denominator: bool = True) -> float | None:
    if a is None or b is None or b == 0 or (positive_denominator and b < 0):
        return None
    return a / b


def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev <= 0:
        return None
    return cur / prev - 1


def _cagr(series: list[tuple[date, float]], years: int) -> float | None:
    if len(series) < years + 1:
        return None
    start, end = series[-(years + 1)][1], series[-1][1]
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def years_without_cut(dps: list[tuple[date, float]]) -> int | None:
    """Consecutive most-recent fiscal years with dividend per share >= the prior year's
    (1% tolerance for rounding). None if the company pays no dividend."""
    values = [v for _, v in dps if v > 0]
    if not values:
        return None
    streak = 0
    for prev, cur in zip(reversed(values[:-1]), reversed(values[1:])):
        if cur < prev * 0.99:
            break
        streak += 1
    return streak


def fcf_payout_3y(s: Snapshot) -> float | None:
    """Dividends paid / free cash flow over the last 3 fiscal years combined, so a
    single unusual year (e.g. KO's 2024 IRS tax deposit) doesn't dominate."""

    def last3(metric: str) -> dict[date, float]:
        return dict((s.annual.get(metric) or [])[-3:])

    divs, ocf, capex = last3("dividends_paid"), last3("operating_cash_flow"), last3("capex")
    years = sorted(set(divs) & set(ocf) & set(capex))
    if len(years) < 2:
        return None
    fcf = sum(ocf[y] - capex[y] for y in years)
    return sum(divs[y] for y in years) / fcf if fcf > 0 else None


def piotroski(s: Snapshot) -> int | None:
    """Piotroski F-score (0-9) from the last two fiscal years. Higher = stronger."""
    a = s.annual

    def last2(metric: str) -> tuple[float, float] | None:
        series = a.get(metric) or []
        return (series[-2][1], series[-1][1]) if len(series) >= 2 else None

    ni, ocf, assets, rev = (
        last2("net_income"),
        last2("operating_cash_flow"),
        last2("total_assets"),
        last2("revenue"),
    )
    if not (ni and ocf and assets and rev):
        return None
    score = 0
    roa_prev, roa = ni[0] / assets[0], ni[1] / assets[1]
    score += roa > 0
    score += ocf[1] > 0
    score += roa > roa_prev
    score += ocf[1] > ni[1]  # earnings backed by cash (low accruals)
    if ltd := last2("total_debt"):
        score += ltd[1] / assets[1] < ltd[0] / assets[0]
    ca, cl = last2("current_assets"), last2("current_liabilities")
    if ca and cl and cl[0] and cl[1]:
        score += ca[1] / cl[1] > ca[0] / cl[0]
    if sh := last2("diluted_shares"):
        score += sh[1] <= sh[0]  # no dilution
    if gp := last2("gross_profit"):
        score += gp[1] / rev[1] > gp[0] / rev[0]
    score += rev[1] / assets[1] > rev[0] / assets[0]  # asset turnover
    return score


def altman_z(s: Snapshot, market_cap: float | None) -> float | None:
    """Altman Z (public manufacturer form). < 1.8 distress, > 3.0 safe. Not for banks."""
    if s.is_financial:
        return None
    L, T = s.latest, s.ttm
    ta, tl = L.get("total_assets"), L.get("total_liabilities")
    needed = (ta, tl, L.get("current_assets"), L.get("current_liabilities"),
              L.get("retained_earnings"), T.get("ebit"), T.get("revenue"), market_cap)  # fmt: skip
    if any(v is None for v in needed) or not ta or not tl:
        return None
    wc = L["current_assets"] - L["current_liabilities"]
    return (1.2 * wc / ta + 1.4 * L["retained_earnings"] / ta + 3.3 * T["ebit"] / ta
            + 0.6 * market_cap / tl + 1.0 * T["revenue"] / ta)  # fmt: skip


def compute_metrics(
    s: Snapshot,
    prices: PriceSeries | None,
    market: PriceSeries | None = None,
    treasury_yield: float | None = None,
) -> Metrics:
    T, L, P = s.ttm, s.latest, s.ttm_year_ago
    price = prices.price(s.as_of) if prices else None
    shares = s.shares_outstanding or T.get("diluted_shares")
    mcap = price * shares if price and shares else None
    debt = L.get("total_debt")  # None when unknown: never assume zero debt
    cash = L.get("cash") or 0
    fcf = (
        (T["operating_cash_flow"] - T["capex"])
        if "operating_cash_flow" in T and "capex" in T
        else None
    )
    ebitda = (T["ebit"] + T["depreciation"]) if "ebit" in T and "depreciation" in T else None

    dps_ttm = T.get("dividends_per_share")
    if (
        dps_ttm is None and prices
    ):  # XBRL DPS can be missing: fall back to paid dividends from prices
        dps_ttm = prices.trailing_dividends(s.as_of) or None
    dps_annual = s.annual.get("dividends_per_share") or []
    eps = T.get("eps_diluted")

    m: Metrics = {
        "price": price,
        "market_cap": mcap,
        # valuation
        "pe": _div(price, eps),
        "earnings_yield": _div(eps, price),
        "fcf_yield": None if s.is_financial else _div(fcf, mcap),
        "price_to_book": _div(mcap, L.get("equity")),
        "ev_to_ebit": None
        if s.is_financial or debt is None
        else _div(mcap + debt - cash if mcap else None, T.get("ebit")),
        # quality
        "gross_margin": _div(T.get("gross_profit"), T.get("revenue")),
        "operating_margin": _div(T.get("ebit"), T.get("revenue")),
        "net_margin": _div(T.get("net_income"), T.get("revenue")),
        "roe": _div(T.get("net_income"), L.get("equity")),
        "accruals": _div(
            (T.get("net_income") or 0) - (T.get("operating_cash_flow") or 0)
            if "net_income" in T and "operating_cash_flow" in T
            else None,
            L.get("total_assets"),
        ),
        # growth
        "revenue_growth_yoy": _growth(T.get("revenue"), P.get("revenue")),
        "eps_growth_yoy": _growth(eps, P.get("eps_diluted")),
        "revenue_cagr_3y": _cagr(s.annual.get("revenue") or [], 3),
        "reinvestment_rate": _div((T.get("capex") or 0) + (T.get("rnd") or 0), T.get("revenue")),
        # health
        # Unknown debt stays unknown: treating it as $0 made KO look debt-free.
        "net_debt_to_ebitda": None if s.is_financial or debt is None else _div(debt - cash, ebitda),
        "interest_coverage": _div(T.get("ebit"), T.get("interest_expense")),
        "current_ratio": _div(L.get("current_assets"), L.get("current_liabilities")),
        "piotroski": piotroski(s),
        "altman_z": altman_z(s, mcap),
        # dividends
        "dividend_yield": _div(dps_ttm, price) if dps_ttm else 0.0 if price else None,
        "payout_ratio": _div(dps_ttm, eps) if dps_ttm else None,
        "fcf_payout": fcf_payout_3y(s),
        "dividend_cagr_5y": _cagr([(d, v) for d, v in dps_annual if v > 0], 5),
        "years_without_cut": years_without_cut(dps_annual),
        "treasury_10y": treasury_yield,
    }
    m["operating_margin_change"] = (
        m["operating_margin"] - _div(P.get("ebit"), P.get("revenue"))
        if m["operating_margin"] is not None and _div(P.get("ebit"), P.get("revenue")) is not None
        else None
    )
    growth_pct = (m["eps_growth_yoy"] or 0) * 100
    m["peg"] = m["pe"] / growth_pct if m["pe"] and growth_pct > 0 else None
    if treasury_yield is not None:
        m["earnings_yield_spread"] = (
            None if m["earnings_yield"] is None else m["earnings_yield"] - treasury_yield
        )
        m["dividend_yield_spread"] = (
            None if m["dividend_yield"] is None else m["dividend_yield"] - treasury_yield
        )
    ni_series = [v for _, v in (s.annual.get("net_income") or [])][-5:]
    m["earnings_variability"] = (
        statistics.stdev(ni_series) / abs(statistics.mean(ni_series))
        if len(ni_series) >= 3 and statistics.mean(ni_series)
        else None
    )
    if prices:
        m["volatility"] = prices.volatility(s.as_of)
        m["max_drawdown"] = prices.max_drawdown(s.as_of)
        m["beta"] = prices.beta(market, s.as_of) if market else None
        m["return_1y"] = prices.total_return(s.as_of - timedelta(days=365), s.as_of)
    return m
