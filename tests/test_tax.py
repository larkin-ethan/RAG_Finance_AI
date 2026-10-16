from datetime import date

import pytest

from finrag.ratings.tax import (
    TaxProfile,
    bracket_tax,
    compare_waiting,
    estimate_sale_tax,
    load_params,
    long_term_date,
    ltcg_tax,
    wash_sale_warning,
)

SINGLE_2026 = load_params(2026)["filing_status"]["single"]


def test_long_term_needs_more_than_one_year():
    assert long_term_date(date(2025, 1, 5)) == date(2026, 1, 6)
    assert long_term_date(date(2024, 2, 29)) == date(2025, 3, 1)  # leap-day purchase


def test_one_year_exactly_is_still_short_term():
    p = TaxProfile("single", 80_000)
    exactly = estimate_sale_tax(p, 10, 100, date(2025, 1, 5), 150, date(2026, 1, 5))
    day_after = estimate_sale_tax(p, 10, 100, date(2025, 1, 5), 150, date(2026, 1, 6))
    assert (exactly.term, day_after.term) == ("short", "long")


def test_ordinary_brackets_2026_single():
    # 12,400 * 10% + (50,400 - 12,400) * 12% = 1,240 + 4,560
    assert bracket_tax(50_400, SINGLE_2026["ordinary"]) == pytest.approx(5_800)
    assert bracket_tax(0, SINGLE_2026["ordinary"]) == 0


def test_short_term_gain_is_incremental_ordinary_tax():
    # $80k income is in the 22% bracket (50,400-105,700): a $10k gain is all at 22%
    est = estimate_sale_tax(
        TaxProfile("single", 80_000), 100, 100, date(2026, 1, 2), 200, date(2026, 6, 1)
    )
    assert est.term == "short" and est.income_tax == pytest.approx(10_000 * 0.22)


def test_long_term_gain_stacks_on_ordinary_income():
    # Ordinary $40k; $20k LT gain: 9,450 at 0% (to 49,450) and 10,550 at 15%
    assert ltcg_tax(40_000, 20_000, SINGLE_2026["ltcg"]) == pytest.approx(10_550 * 0.15)
    assert ltcg_tax(20_000, 10_000, SINGLE_2026["ltcg"]) == 0  # entirely in the 0% band


def test_niit_applies_only_above_threshold():
    rich = estimate_sale_tax(TaxProfile("single", 300_000, magi=320_000), 100, 100, date(2020, 1, 1),
                             300, date(2026, 3, 1))  # fmt: skip
    assert rich.niit == pytest.approx(0.038 * 20_000)  # whole gain is above the $200k threshold
    modest = estimate_sale_tax(
        TaxProfile("single", 60_000), 100, 100, date(2020, 1, 1), 300, date(2026, 3, 1)
    )
    assert modest.niit == 0


def test_loss_has_no_tax_and_explains_offset():
    est = estimate_sale_tax(
        TaxProfile("single", 80_000), 10, 200, date(2025, 1, 1), 150, date(2026, 6, 1)
    )
    assert est.gain == -500 and est.total == 0 and "$3,000" in est.notes[0]


def test_waiting_for_long_term_saves_tax():
    cmp = compare_waiting(
        TaxProfile("single", 80_000), 100, 100, date(2025, 11, 1), 200, today=date(2026, 6, 1)
    )
    assert cmp.later_date == date(2026, 11, 2)
    assert cmp.now.term == "short" and cmp.later.term == "long"
    assert cmp.savings == pytest.approx(10_000 * (0.22 - 0.15))


def test_unsupported_status_rejected():
    with pytest.raises(ValueError):
        estimate_sale_tax(
            TaxProfile("married_separate", 1), 1, 1, date(2025, 1, 1), 2, date(2026, 6, 1)
        )


def test_wash_sale_warning():
    assert wash_sale_warning(date(2026, 6, 1), [date(2026, 5, 20)])
    assert wash_sale_warning(date(2026, 6, 1), [date(2026, 3, 1)]) is None
