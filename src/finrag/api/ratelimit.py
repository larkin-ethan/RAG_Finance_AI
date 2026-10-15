"""In-memory abuse protection for the public demo.

  * Per-client sliding window: at most N questions per 60 seconds per IP.
  * Global daily cap: protects the free-tier Gemini quota for everyone.

In-memory means per process. The Cloud Run deploy uses max-instances=1, so
the limits are exact. With more instances you'd move this to Redis or the DB.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


class RateLimiter:
    def __init__(self, per_minute: int, daily_cap: int, clock=time.monotonic):
        self.per_minute, self.daily_cap, self._clock = per_minute, daily_cap, clock
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._day, self._count = "", 0
        self._lock = threading.Lock()

    def check(self, client: str, today: str | None = None) -> str | None:
        """Record one request. Returns None if allowed, else a reason string."""
        today = today or datetime.now(UTC).date().isoformat()
        now = self._clock()
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.daily_cap:
                return "The demo's daily question limit is reached. Please try again tomorrow."
            window = self._windows[client]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return f"Too many questions. Limit is {self.per_minute} per minute."
            window.append(now)
            self._count += 1
            return None


def client_ip(headers, peer: str | None, trust_proxy: bool) -> str:
    """Behind Cloud Run the peer is Google's proxy and the real client is in
    X-Forwarded-For. Clients can forge leading entries, so take the rightmost
    (appended by the proxy). Only trusted when we know we're behind one."""
    if trust_proxy and (forwarded := headers.get("x-forwarded-for")):
        return forwarded.split(",")[-1].strip()
    return peer or "unknown"
