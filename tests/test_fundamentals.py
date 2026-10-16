import json
from datetime import date
from pathlib import Path

from finrag.ratings.fundamentals import FactStore

AAPL = FactStore(
    json.loads((Path(__file__).parent / "fixtures" / "aapl_companyfacts_subset.json").read_text())
)
M = 1e6


def test_golden_apple_fy2024_from_the_10k():
    snap = AAPL.snapshot(date(2024, 11, 1))  # 10-K filed this day
    assert snap.period_end == date(2024, 9, 28)
    assert snap.ttm["revenue"] == 391_035 * M
    assert snap.ttm["net_income"] == 93_736 * M
    assert snap.ttm["eps_diluted"] == 6.08
    assert round(snap.ttm["dividends_per_share"], 2) == 0.98
    assert snap.ttm_year_ago["revenue"] == 383_285 * M


def test_ttm_from_year_to_date():
    # Q3 FY2024 10-Q: 9M FY24 + FY23 - 9M FY23 = 296,105 + 383,285 - 293,787
    ttm = AAPL.ttm("revenue", date(2024, 8, 2))
    assert ttm.method == "ytd" and ttm.period_end == date(2024, 6, 29)
    assert ttm.value == 385_603 * M


def test_point_in_time_ignores_facts_filed_later():
    # The day before the Q3 10-Q, only data through Q2 is known.
    before = AAPL.ttm("revenue", date(2024, 8, 1))
    assert before.period_end == date(2024, 3, 30)
    for obs in AAPL.known("revenue", date(2024, 8, 1)):
        assert obs.filed <= date(2024, 8, 1)


def test_restatement_latest_filing_known_on_date_wins():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2023-01-01", "end": "2023-12-31", "val": 100, "form": "10-K", "filed": "2024-02-01"},
        {"start": "2023-01-01", "end": "2023-12-31", "val": 90, "form": "10-K", "filed": "2025-02-01"},
    ]}}}}}  # fmt: skip
    store = FactStore(facts)
    assert store.ttm("revenue", date(2024, 6, 1)).value == 100  # as originally reported
    assert store.ttm("revenue", date(2025, 6, 1)).value == 90  # after the restatement


def test_preferred_concept_wins_and_predecessors_merge():
    old = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2016-01-01",
                                "end": "2016-12-31",
                                "val": 50,
                                "form": "10-K",
                                "filed": "2017-02-01",
                            }
                        ]
                    }
                }
            }
        }
    }
    new = {"facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"start": "2016-01-01", "end": "2016-12-31", "val": 51, "form": "10-K", "filed": "2018-02-01"},
        {"start": "2017-01-01", "end": "2017-12-31", "val": 60, "form": "10-K", "filed": "2018-02-01"}]}}}}}  # fmt: skip
    store = FactStore(old, new)
    series = store.annual_series("revenue", date(2019, 1, 1))
    assert series == [(date(2016, 12, 31), 51), (date(2017, 12, 31), 60)]


def test_bank_detection():
    bank = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [{"start": "2024-01-01", "end": "2024-12-31", "val": 1, "form": "10-K", "filed": "2025-02-01"}]}},
        "Deposits": {"units": {"USD": [{"end": "2024-12-31", "val": 9, "form": "10-K", "filed": "2025-02-01"}]}},
    }}}  # fmt: skip
    assert FactStore(bank).snapshot(date(2025, 3, 1)).is_financial
    assert not AAPL.snapshot(date(2024, 11, 1)).is_financial


def test_nothing_known_before_first_filing():
    assert AAPL.snapshot(date(2000, 1, 1)).period_end is None
