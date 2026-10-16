"""Daily prices (Tiingo) and the price-based measures the goals use.

Tiingo's `adjClose` is adjusted for splits AND dividends, so returns computed
from it are total returns. Every query takes an `as_of` date and only looks
at data on or before it (point-in-time, like the fundamentals).

Free tier, personal-use license: fine for the private local deployment only.
The API key goes in an Authorization header, never in the URL (keeps it out of
proxy and server logs).
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
TRADING_DAYS = 252


@dataclass
class PriceSeries:
    ticker: str
    dates: list[date]
    close: list[float]  # as traded (for yields, market cap)
    adj_close: list[float]  # split + dividend adjusted (for returns)
    dividends: list[float]  # cash dividend per share on each ex-date (0 otherwise)

    @classmethod
    def from_tiingo(cls, ticker: str, rows: list[dict]) -> PriceSeries:
        rows = sorted(rows, key=lambda r: r["date"])
        return cls(
            ticker=ticker.upper(),
            dates=[date.fromisoformat(r["date"][:10]) for r in rows],
            close=[float(r["close"]) for r in rows],
            adj_close=[float(r["adjClose"]) for r in rows],
            dividends=[float(r.get("divCash") or 0.0) for r in rows],
        )

    def _idx(self, as_of: date) -> int:
        """Index of the last trading day on or before as_of (-1 if none)."""
        return bisect.bisect_right(self.dates, as_of) - 1

    def price(self, as_of: date) -> float | None:
        i = self._idx(as_of)
        return self.close[i] if i >= 0 else None

    def total_return(self, start: date, end: date) -> float | None:
        i, j = self._idx(start), self._idx(end)
        if i < 0 or j <= i:
            return None
        return self.adj_close[j] / self.adj_close[i] - 1

    def _daily_returns(self, as_of: date, days: int) -> list[float]:
        j = self._idx(as_of)
        i = max(0, j - days)
        return [math.log(self.adj_close[k] / self.adj_close[k - 1]) for k in range(i + 1, j + 1)]

    def volatility(self, as_of: date, days: int = TRADING_DAYS) -> float | None:
        r = self._daily_returns(as_of, days)
        return statistics.stdev(r) * math.sqrt(TRADING_DAYS) if len(r) > 20 else None

    def max_drawdown(self, as_of: date, days: int = 3 * TRADING_DAYS) -> float | None:
        """Worst peak-to-trough fall in the window, as a negative fraction."""
        j = self._idx(as_of)
        if j < 1:
            return None
        peak, worst = 0.0, 0.0
        for v in self.adj_close[max(0, j - days) : j + 1]:
            peak = max(peak, v)
            worst = min(worst, v / peak - 1)
        return worst

    def beta(self, market: PriceSeries, as_of: date, days: int = 2 * TRADING_DAYS) -> float | None:
        mine = dict(zip(self.dates, self.adj_close))
        pairs = []
        j = market._idx(as_of)
        for k in range(max(1, j - days), j + 1):
            d0, d1 = market.dates[k - 1], market.dates[k]
            if d0 in mine and d1 in mine:
                pairs.append(
                    (mine[d1] / mine[d0] - 1, market.adj_close[k] / market.adj_close[k - 1] - 1)
                )
        if len(pairs) < 60:
            return None
        xs, ys = [m for _, m in pairs], [s for s, _ in pairs]
        var = statistics.variance(xs)
        return statistics.covariance(xs, ys) / var if var else None

    def dividends_between(self, start: date, end: date) -> list[tuple[date, float]]:
        return [(d, v) for d, v in zip(self.dates, self.dividends) if start < d <= end and v > 0]

    def trailing_dividends(self, as_of: date, days: int = 365) -> float:
        return sum(v for _, v in self.dividends_between(as_of - timedelta(days=days), as_of))

    def rolling_returns(self, as_of: date, horizon_days: int, years: int = 20) -> list[float]:
        """Every past `horizon_days`-calendar-day total return that ended on or before
        as_of (sampled weekly). The horizon report's historical distribution."""
        out = []
        start_limit = as_of - timedelta(days=365 * years)
        end = as_of
        while True:
            start = end - timedelta(days=horizon_days)
            if start < max(start_limit, self.dates[0]):
                break
            if (r := self.total_return(start, end)) is not None:
                out.append(r)
            end -= timedelta(days=7)
        return out


class TiingoClient:
    def __init__(self, api_key: str, cache_dir: Path, transport: httpx.BaseTransport | None = None,
                 max_age_s: float = 20 * 3600):  # fmt: skip
        if not api_key:
            raise ValueError("TIINGO_API_KEY is not set (free signup at tiingo.com)")
        self.cache_dir, self.max_age_s = cache_dir, max_age_s
        self._http = httpx.Client(headers={"Authorization": f"Token {api_key}"}, timeout=60.0,
                                  transport=transport)  # fmt: skip

    def series(self, ticker: str, start: date = date(2000, 1, 1)) -> PriceSeries:
        ticker = ticker.upper()
        path = self.cache_dir / f"{ticker}.json"
        if path.exists() and time.time() - path.stat().st_mtime < self.max_age_s:
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = self._http.get(TIINGO_URL.format(ticker=ticker),
                               params={"startDate": start.isoformat(), "format": "json"})  # fmt: skip
            if r.is_error:
                raise RuntimeError(f"Tiingo {r.status_code} for {ticker}: {r.text[:200]}")
            rows = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows), encoding="utf-8")
        return PriceSeries.from_tiingo(ticker, rows)
