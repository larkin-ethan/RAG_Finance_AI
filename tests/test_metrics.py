from datetime import date

import pytest

from finrag.ratings.fundamentals import Snapshot
from finrag.ratings.metrics import compute_metrics, fcf_payout_3y, piotroski, years_without_cut
from finrag.ratings.prices import PriceSeries

D = [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31)]


def ann(*values) -> list[tuple[date, float]]:
    return list(zip(D[-len(values) :], values))


def price(p: float, as_of=date(2025, 3, 1)) -> PriceSeries:
    return PriceSeries("X", [as_of], [p], [p], [0.0])


def snap(**kw) -> Snapshot:
    s = Snapshot(as_of=date(2025, 3, 1), period_end=date(2024, 12, 31))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_years_without_cut():
    assert years_without_cut(ann(1.0, 1.1, 1.2, 1.3)) == 3
    assert years_without_cut(ann(1.0, 1.2, 1.1, 1.3)) == 1  # cut in 2023 ends the streak
    assert years_without_cut(ann(0, 0)) is None  # non-payer


def test_fcf_payout_uses_three_years():
    s = snap(
        annual={
            "dividends_paid": ann(8, 8, 8),
            "operating_cash_flow": ann(12, 12, 6),
            "capex": ann(2, 2, 2),
        }
    )
    assert fcf_payout_3y(s) == pytest.approx(24 / 24)  # one bad year doesn't push it to 2.0


def test_piotroski_strong_vs_weak():
    strong = snap(annual={
        "net_income": ann(10, 12), "operating_cash_flow": ann(12, 15), "total_assets": ann(100, 100),
        "revenue": ann(100, 110), "total_debt": ann(30, 25), "current_assets": ann(50, 60),
        "current_liabilities": ann(40, 40), "diluted_shares": ann(10, 9.8), "gross_profit": ann(40, 46),
    })  # fmt: skip
    weak = snap(annual={
        "net_income": ann(10, -5), "operating_cash_flow": ann(12, -8), "total_assets": ann(100, 120),
        "revenue": ann(100, 90), "total_debt": ann(30, 50), "current_assets": ann(50, 40),
        "current_liabilities": ann(40, 45), "diluted_shares": ann(10, 11), "gross_profit": ann(40, 30),
    })  # fmt: skip
    assert piotroski(strong) == 9
    assert piotroski(weak) == 0


def test_unknown_debt_is_not_zero_debt():
    s = snap(ttm={"ebit": 10.0, "depreciation": 2.0, "revenue": 50.0}, latest={"cash": 5.0})
    m = compute_metrics(s, price(10))
    assert m["net_debt_to_ebitda"] is None and m["ev_to_ebit"] is None


def test_valuation_and_dividend_metrics():
    s = snap(
        ttm={"eps_diluted": 5.0, "dividends_per_share": 2.0, "ebit": 100.0, "revenue": 400.0,
             "net_income": 50.0, "operating_cash_flow": 80.0, "capex": 20.0},
        ttm_year_ago={"eps_diluted": 4.0, "revenue": 380.0, "ebit": 90.0},
        latest={"total_debt": 200.0, "cash": 50.0, "equity": 500.0},
        shares_outstanding=10.0,
    )  # fmt: skip
    m = compute_metrics(s, price(100.0), treasury_yield=0.04)
    assert m["pe"] == 20 and m["earnings_yield"] == 0.05
    assert m["dividend_yield"] == 0.02 and m["payout_ratio"] == 0.4
    assert m["dividend_yield_spread"] == pytest.approx(-0.02)
    assert m["fcf_yield"] == pytest.approx(60 / 1000)
    assert m["ev_to_ebit"] == pytest.approx((1000 + 200 - 50) / 100)
    assert m["eps_growth_yoy"] == pytest.approx(0.25) and m["peg"] == pytest.approx(20 / 25)
    assert m["operating_margin_change"] == pytest.approx(100 / 400 - 90 / 380)


def test_loss_makes_pe_undefined_and_non_payer_yield_zero():
    s = snap(ttm={"eps_diluted": -1.0}, shares_outstanding=10.0)
    m = compute_metrics(s, price(50.0))
    assert m["pe"] is None and m["dividend_yield"] == 0.0 and m["payout_ratio"] is None


def test_banks_skip_industrial_metrics():
    s = snap(ttm={"eps_diluted": 5.0, "ebit": 10.0, "operating_cash_flow": 1.0, "capex": 0.5},
             latest={"total_debt": 1.0, "cash": 1.0}, is_financial=True, shares_outstanding=1.0)  # fmt: skip
    m = compute_metrics(s, price(60.0))
    assert m["fcf_yield"] is None and m["ev_to_ebit"] is None and m["altman_z"] is None
    assert m["pe"] == 12
