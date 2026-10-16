import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from finrag.ratings.check import StockChecker, to_jsonable
from finrag.ratings.macro import TreasuryYield
from finrag.ratings.positions import Lot
from finrag.ratings.prices import PriceSeries
from finrag.ratings.tax import TaxProfile
from finrag.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_companyfacts_subset.json"
AS_OF = date(2024, 11, 15)


def walk(ticker, seed, start=date(2014, 1, 1), end=AS_OF, p0=50.0):
    rng = random.Random(seed)
    days, vals, d = [], [p0], start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    for _ in days[1:]:
        vals.append(vals[-1] * (1 + 0.0005 + rng.gauss(0, 0.012)))
    divs = [0.25 if i % 91 == 0 and i else 0.0 for i in range(len(days))]
    return PriceSeries(ticker, days, vals, vals, divs)


class FakeEdgar:
    def get_cik(self, ticker):
        return "0000320193"


class FakeTiingo:
    def series(self, ticker):
        return walk(ticker, seed=hash(ticker) % 100)


class FakeFred:
    def treasury_10y(self):
        return TreasuryYield([(date(2014, 1, 1), 2.5), (date(2023, 1, 1), 4.2)])


@pytest.fixture
def checker(tmp_path):
    (tmp_path / "xbrl").mkdir()
    shutil.copy(FIXTURE, tmp_path / "xbrl" / "CIK0000320193.json")
    s = Settings(
        sec_user_agent="t t@example.com", data_dir=tmp_path, watchlist_path=tmp_path / "none.yaml"
    )
    return StockChecker(s, FakeEdgar(), FakeTiingo(), FakeFred())


def test_full_check_with_lot_and_timeline(checker):
    lot = Lot("AAPL", 10, 20.0, date(2024, 6, 3))
    r = checker.check("aapl", ["dividend", "value"], horizon_end=AS_OF + timedelta(days=365),
                      lots=[lot, Lot("MSFT", 1, 1, date(2020, 1, 1))],
                      tax=TaxProfile("single", 90_000), as_of=AS_OF)  # fmt: skip
    assert r.ticker == "AAPL" and r.period_end == date(2024, 9, 28)
    assert [g.goal for g in r.goals] == ["dividend", "value"]
    assert r.horizon and r.horizon.stock is not None and r.horizon.fit_grade
    assert len(r.lots) == 1  # only this ticker's lots
    v = r.lots[0]
    assert v.long_term_date == date(2025, 6, 4)
    assert v.tax_now is not None and v.savings_by_waiting is not None and v.savings_by_waiting >= 0
    assert r.explanation.source == "template"  # no LLM configured
    assert "Risk-factor changes not included" in " ".join(r.notes)
    assert "not investment or tax advice" in r.disclaimer
    to_jsonable(r)  # API-serializable


def test_unknown_goal_rejected(checker):
    with pytest.raises(ValueError, match="unknown goals"):
        checker.check("AAPL", ["moonshot"], as_of=AS_OF)


def test_point_in_time_as_of_respected(checker):
    r = checker.check("AAPL", ["growth"], as_of=date(2024, 8, 1))  # before the Q3 10-Q
    assert r.period_end == date(2024, 3, 30)
