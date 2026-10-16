import pytest

from finrag.ratings.goals import (
    RECIPES,
    anchor_score,
    grade_for,
    percentile_score,
    score_goal,
)

STRONG_DIVIDEND = {
    "dividend_yield": 0.035, "dividend_yield_spread": -0.005, "payout_ratio": 0.55, "fcf_payout": 0.6,
    "dividend_cagr_5y": 0.06, "years_without_cut": 15, "net_debt_to_ebitda": 1.5, "interest_coverage": 12,
}  # fmt: skip


def test_recipes_weights_sum_to_one():
    for goal, comps in RECIPES.items():
        assert sum(c.weight for c in comps) == pytest.approx(1.0), goal


def test_percentile_and_anchor_scoring():
    assert percentile_score(5, [1, 2, 3, 4], True) == 100
    assert percentile_score(5, [1, 2, 3, 4], False) == 0
    assert percentile_score(2, [1, 2, 3, 4], True) == pytest.approx(37.5)  # ties count half
    assert anchor_score(0.05, 0.0, 0.05) == 100 and anchor_score(-1, 0, 1) == 0
    assert anchor_score(2.0, 4.0, 0.0) == 50  # lower-is-better anchors work too


def test_grade_bands():
    assert [grade_for(s) for s in (85, 70, 69.9, 50, 30, 10)] == [
        "Strong fit", "Strong fit", "Moderate fit", "Moderate fit", "Weak fit", "Poor fit"]  # fmt: skip


def test_non_payer_and_token_dividend_are_not_dividend_stocks():
    assert score_goal("dividend", {"dividend_yield": 0.0}).grade == "Not a dividend stock"
    assert (
        score_goal("dividend", STRONG_DIVIDEND | {"dividend_yield": 0.004}).grade
        == "Not a dividend stock"
    )


def test_low_yield_caps_dividend_grade():
    r = score_goal("dividend", STRONG_DIVIDEND | {"dividend_yield": 0.008})
    assert r.grade == "Weak fit" and "limited income" in r.caps[0]


def test_own_history_alone_cannot_make_a_low_yield_strong():
    # Regression: a yield that's high only relative to the company's own history.
    hist = [{"dividend_yield": y} for y in (0.004, 0.005, 0.0055, 0.006, 0.0065)]
    r = score_goal("dividend", STRONG_DIVIDEND | {"dividend_yield": 0.009}, history=hist)
    assert r.grade != "Strong fit"


def test_strong_dividend_payer():
    r = score_goal("dividend", STRONG_DIVIDEND)
    assert r.grade == "Strong fit" and not r.caps and r.coverage == 1.0


def test_guardrail_caps_unsustainable_dividend():
    r = score_goal("dividend", STRONG_DIVIDEND | {"fcf_payout": 1.3})
    assert r.score <= 49 and r.grade == "Weak fit"
    assert r.caps == ["Dividends exceeded free cash flow over the last 3 years"]


def test_value_trap_guard():
    cheap = {"earnings_yield": 0.12, "fcf_yield": 0.10, "price_to_book": 0.8, "ev_to_ebit": 6,
             "earnings_yield_spread": 0.07, "piotroski": 2}  # fmt: skip
    r = score_goal("value", cheap)
    assert r.score == 49 and "value trap" in r.caps[0]
    assert score_goal("value", cheap | {"piotroski": 7}).grade == "Strong fit"


def test_history_percentile_used_when_available():
    hist = [{"revenue_growth_yoy": g} for g in (0.01, 0.02, 0.03, 0.04, 0.05)]
    r = score_goal("growth", {"revenue_growth_yoy": 0.06, "eps_growth_yoy": 0.1, "revenue_cagr_3y": 0.05,
                              "operating_margin_change": 0.0, "reinvestment_rate": 0.1, "peg": 1.5}, history=hist)  # fmt: skip
    comp = next(c for c in r.components if c.metric == "revenue_growth_yoy")
    # 50% absolute (0.06 on a -5%..25% scale = 36.7) + 50% history percentile (100)
    assert comp.basis == "absolute+history"
    assert comp.score == pytest.approx(0.5 * (100 * 0.11 / 0.30) + 0.5 * 100)


def test_insufficient_data():
    r = score_goal("stability", {"volatility": 0.2})
    assert r.grade == "Insufficient data" and r.score is None


def test_stability_red_flag_cap():
    calm = {"earnings_variability": 0.05, "altman_z": 5, "interest_coverage": 30, "volatility": 0.15,
            "max_drawdown": -0.1, "beta": 0.5, "risk_change_share": 0.0}  # fmt: skip
    assert score_goal("stability", calm).grade == "Strong fit"
    r = score_goal("stability", calm | {"red_flags": ["going concern"]})
    assert r.grade == "Weak fit" and "going concern" in r.caps[0]
