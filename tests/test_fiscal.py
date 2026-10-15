from datetime import date

import pytest

from finrag.ingest.fiscal import fiscal_period

# (company, nominal FYE, period end, form, expected (FY, Q)) - checked against
# how each company labels the period in its own filings.
CASES = [
    ("Apple 10-K", "0926", date(2024, 9, 28), "10-K", (2024, None)),
    ("Apple 10-K", "0926", date(2023, 9, 30), "10-K", (2023, None)),
    ("Apple Q1", "0926", date(2024, 12, 28), "10-Q", (2025, 1)),
    ("Apple Q3", "0926", date(2025, 6, 28), "10-Q", (2025, 3)),
    ("NVIDIA 10-K", "0131", date(2025, 1, 26), "10-K", (2025, None)),
    ("NVIDIA Q1", "0131", date(2025, 4, 27), "10-Q", (2026, 1)),
    ("NVIDIA Q2", "0131", date(2025, 7, 27), "10-Q", (2026, 2)),
    ("J&J 10-K FY2022", "0103", date(2023, 1, 1), "10-K", (2022, None)),
    ("J&J 10-K FY2024", "0103", date(2024, 12, 29), "10-K", (2024, None)),
    ("J&J Q1", "0103", date(2024, 3, 31), "10-Q", (2024, 1)),
    ("Costco 10-K", "0830", date(2024, 9, 1), "10-K", (2024, None)),
    ("Costco Q1 (12 wk)", "0830", date(2024, 11, 24), "10-Q", (2025, 1)),
    ("Costco Q2 (12 wk)", "0830", date(2025, 2, 16), "10-Q", (2025, 2)),
    ("Costco Q3 (12 wk)", "0830", date(2025, 5, 11), "10-Q", (2025, 3)),
    ("Microsoft Q2", "0630", date(2024, 12, 31), "10-Q", (2025, 2)),
    ("Coca-Cola 10-K", "1231", date(2024, 12, 31), "10-K", (2024, None)),
    ("Coca-Cola Q1", "1231", date(2024, 3, 29), "10-Q", (2024, 1)),
]


@pytest.mark.parametrize(("name", "fye", "period_end", "form", "expected"), CASES)
def test_fiscal_period(name, fye, period_end, form, expected):
    assert fiscal_period(period_end, fye, form) == expected, name
