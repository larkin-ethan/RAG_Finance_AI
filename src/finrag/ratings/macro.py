"""10-year US Treasury yield (FRED series DGS10): the market benchmark for
earnings and dividend yields ("is a 3% dividend attractive when bonds pay 4.5%?").

FRED only accepts its key as a URL query parameter, so this client never logs
request URLs. Cached on disk; looked up point-in-time like everything else.
"""

from __future__ import annotations

import bisect
import json
import time
from datetime import date
from pathlib import Path

import httpx

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class TreasuryYield:
    def __init__(self, observations: list[tuple[date, float]]):
        self.dates = [d for d, _ in observations]
        self.values = [v for _, v in observations]

    def on(self, as_of: date) -> float | None:
        """Yield as a fraction (4.25% -> 0.0425) on the last day <= as_of."""
        i = bisect.bisect_right(self.dates, as_of) - 1
        return self.values[i] / 100 if i >= 0 else None


class FredClient:
    def __init__(self, api_key: str, cache_dir: Path, transport: httpx.BaseTransport | None = None,
                 max_age_s: float = 20 * 3600):  # fmt: skip
        if not api_key:
            raise ValueError("FRED_API_KEY is not set (free key at fred.stlouisfed.org)")
        self._key, self.cache_dir, self.max_age_s = api_key, cache_dir, max_age_s
        self._http = httpx.Client(timeout=60.0, transport=transport)

    def treasury_10y(self) -> TreasuryYield:
        path = self.cache_dir / "DGS10.json"
        if path.exists() and time.time() - path.stat().st_mtime < self.max_age_s:
            obs = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = self._http.get(FRED_URL, params={"series_id": "DGS10", "api_key": self._key,
                                                 "file_type": "json"})  # fmt: skip
            if r.is_error:  # deliberately not echoing the URL (contains the key)
                raise RuntimeError(f"FRED request failed with HTTP {r.status_code}")
            obs = [o for o in r.json()["observations"] if o["value"] not in (".", "")]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obs), encoding="utf-8")
        return TreasuryYield([(date.fromisoformat(o["date"]), float(o["value"])) for o in obs])
