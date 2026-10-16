from datetime import date, timedelta

import httpx
import pytest

from finrag.ratings.macro import FredClient
from finrag.ratings.prices import PriceSeries, TiingoClient


def series(values: list[float], start=date(2020, 1, 1), dividends: dict[int, float] | None = None,
           ticker="TEST") -> PriceSeries:  # fmt: skip
    days = [start + timedelta(days=i) for i in range(len(values))]
    divs = [(dividends or {}).get(i, 0.0) for i in range(len(values))]
    return PriceSeries(ticker, days, list(values), list(values), divs)


def test_price_is_point_in_time():
    s = series([10, 11, 12])
    assert s.price(date(2020, 1, 2)) == 11
    assert s.price(date(2019, 12, 31)) is None
    assert s.price(date(2030, 1, 1)) == 12  # last known close


def test_max_drawdown_and_total_return():
    s = series([100, 120, 60, 90])
    assert s.max_drawdown(date(2020, 1, 4)) == pytest.approx(-0.5)
    assert s.total_return(date(2020, 1, 1), date(2020, 1, 4)) == pytest.approx(-0.1)


def test_volatility_zero_for_steady_growth_and_beta_one_vs_itself():
    steady = series([100 * 1.001**i for i in range(300)])
    assert steady.volatility(date(2020, 12, 31)) == pytest.approx(0, abs=1e-9)
    import random

    rng = random.Random(1)
    vals = [100.0]
    for _ in range(400):
        vals.append(vals[-1] * (1 + rng.gauss(0, 0.01)))
    market = series(vals, ticker="SPY")
    assert market.beta(market, date(2021, 3, 1)) == pytest.approx(1.0)


def test_dividends_and_rolling_returns():
    s = series([100] * 800, dividends={100: 0.5, 190: 0.5, 280: 0.5, 370: 0.5, 460: 0.6})
    as_of = s.dates[460]
    # 365-day window ending on day 460 starts at day 95: payments on days 100..460
    assert s.trailing_dividends(as_of) == pytest.approx(0.5 * 4 + 0.6)
    rr = s.rolling_returns(s.dates[-1], horizon_days=180)
    assert rr and all(r == 0 for r in rr)  # flat prices: every window returns 0


def test_tiingo_key_in_header_not_url_and_cached(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[
            {"date": "2024-01-02T00:00:00.000Z", "close": 185.6, "adjClose": 183.9, "divCash": 0.0},
            {"date": "2024-01-03T00:00:00.000Z", "close": 184.2, "adjClose": 182.5, "divCash": 0.24},
        ])  # fmt: skip

    client = TiingoClient("tk-secret", tmp_path, transport=httpx.MockTransport(handler))
    s = client.series("aapl")
    assert s.ticker == "AAPL" and s.dividends == [0.0, 0.24]
    assert seen[0].headers["Authorization"] == "Token tk-secret"
    assert "tk-secret" not in str(seen[0].url)
    client.series("AAPL")
    assert len(seen) == 1  # second call served from cache


def test_fred_skips_missing_values_and_hides_key_on_error(tmp_path):
    def ok(request):
        return httpx.Response(200, json={"observations": [
            {"date": "2024-01-02", "value": "3.95"}, {"date": "2024-01-15", "value": "."},
            {"date": "2024-01-16", "value": "4.07"}]})  # fmt: skip

    ty = FredClient("fk", tmp_path, transport=httpx.MockTransport(ok)).treasury_10y()
    assert ty.on(date(2024, 1, 15)) == pytest.approx(0.0395)  # holiday "." skipped -> prior value
    assert ty.on(date(2024, 2, 1)) == pytest.approx(0.0407)

    def fail(request):
        return httpx.Response(500)

    with pytest.raises(RuntimeError) as exc:
        FredClient(
            "fred-secret", tmp_path / "x", transport=httpx.MockTransport(fail)
        ).treasury_10y()
    assert "fred-secret" not in str(exc.value)
