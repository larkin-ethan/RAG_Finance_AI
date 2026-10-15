"""Lifetime token budget for paid APIs, persisted to disk.

Before each call we check estimated tokens against the cap and refuse calls
that would cross it. A bug like an accidental re-embed loop then fails loudly
instead of quietly running past the free tier. Actual billed tokens (from the
API response) are what get recorded.

Each model's cap is the smaller of the configured budget and 90% of that
model's free allowance (Voyage pricing page). Models with no free tokens get a
cap of 0, so using one by accident is refused rather than billed. Free tokens
are per account, so this ledger only protects spend made from this machine's
cache directory. Keep an eye on the Voyage dashboard too.
"""

from __future__ import annotations

import json
from pathlib import Path

# Lifetime free tokens per account (https://docs.voyageai.com/docs/pricing).
FREE_TOKENS = {
    "voyage-4": 200_000_000,
    "voyage-4-lite": 200_000_000,
    "voyage-4-large": 200_000_000,
    "voyage-context-4": 200_000_000,
    "voyage-finance-2": 50_000_000,
    "rerank-2.5": 200_000_000,
    "rerank-2.5-lite": 200_000_000,
}
FREE_SAFETY = 0.9


def free_cap(model: str) -> int:
    return int(FREE_TOKENS.get(model, 0) * FREE_SAFETY)


class BudgetExceeded(RuntimeError):
    pass


class TokenBudget:
    def __init__(self, path: Path, limit: int, model: str):
        self.path, self.model = path, model
        self.limit = min(limit, free_cap(model))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @property
    def spent(self) -> int:
        return self._ledger.get(self.model, 0)

    def check(self, estimated_tokens: int) -> None:
        if self.spent + estimated_tokens > self.limit:
            raise BudgetExceeded(
                f"{self.model}: spent {self.spent:,} + ~{estimated_tokens:,} would exceed "
                f"the {self.limit:,} token cap (min of EMBED_BUDGET_TOKENS and 90% of "
                f"the model's free tokens, {FREE_TOKENS.get(self.model, 0):,}). Ledger: {self.path}"
            )

    def record(self, tokens: int) -> None:
        self._ledger[self.model] = self.spent + tokens
        self.path.write_text(json.dumps(self._ledger, indent=2), encoding="utf-8")
