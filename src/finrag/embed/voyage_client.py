"""Thin client for Voyage's embeddings API (https://docs.voyageai.com).

  POST https://api.voyageai.com/v1/embeddings
  {"input": [...], "model": "voyage-4", "input_type": "document"|"query",
   "output_dimension": 1024}
  -> {"data": [{"embedding": [...], "index": i}], "usage": {"total_tokens": n}}

input_type matters: Voyage prepends a different retrieval prompt for queries
vs documents, so always embed chunks as "document" and questions as "query".
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from finrag.chunking.base import count_tokens
from finrag.ingest.edgar_client import RateLimiter

EMBED_URL = "https://api.voyageai.com/v1/embeddings"
MAX_TEXTS_PER_REQUEST = 1000
MAX_TOKENS_PER_REQUEST = {"voyage-4": 320_000, "voyage-4-lite": 1_000_000}
DEFAULT_MAX_TOKENS = 120_000  # voyage-4-large, voyage-finance-2
# Our token count is approximate, so batches target well under the API cap.
BATCH_SAFETY = 0.5


class VoyageAPIError(RuntimeError):
    """The API rejected a request; the message includes Voyage's explanation."""


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: int  # billed tokens, as reported by the API


class VoyageClient:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4",
        dim: int | None = 1024,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 6,
        max_batch_tokens: int | None = None,
        requests_per_minute: float | None = None,
    ):
        """`max_batch_tokens` / `requests_per_minute` support low-limit accounts.
        Without a payment method, Voyage allows 3 RPM and 10K TPM. A request
        larger than the TPM cap can never succeed, so batches must be smaller."""
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is not set (see .env.example)")
        self.model, self.dim, self.max_attempts = model, dim, max_attempts
        self.max_batch_tokens = max_batch_tokens
        self._limiter = RateLimiter(requests_per_minute / 60) if requests_per_minute else None
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=120.0, transport=transport
        )

    def cache_tag(self, input_type: str) -> str:
        return input_type

    def batches(self, texts: list[str]) -> Iterator[list[str]]:
        """Split texts into requests that respect count and token limits."""
        limit = MAX_TOKENS_PER_REQUEST.get(self.model, DEFAULT_MAX_TOKENS) * BATCH_SAFETY
        if self.max_batch_tokens:
            limit = min(limit, self.max_batch_tokens)
        batch, size = [], 0
        for text in texts:
            n = count_tokens(text)
            if batch and (size + n > limit or len(batch) == MAX_TEXTS_PER_REQUEST):
                yield batch
                batch, size = [], 0
            batch.append(text)
            size += n
        if batch:
            yield batch

    def _post(self, payload: dict) -> dict:
        """POST with retries. 429 (rate limit) and 5xx back off exponentially,
        capped at 60s. Accounts without a payment method have very low limits."""
        for attempt in range(self.max_attempts):
            if self._limiter:
                self._limiter.wait()
            response = self._http.post(EMBED_URL, json=payload)
            temporary = response.status_code == 429 or response.status_code >= 500
            if temporary and attempt < self.max_attempts - 1:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            if response.is_error:
                raise VoyageAPIError(
                    f"Voyage API {response.status_code} after {attempt + 1} attempt(s): "
                    f"{response.text[:600]}"
                )
            return response.json()
        raise AssertionError("unreachable")

    def embed_batch(self, texts: list[str], input_type: str) -> EmbedResult:
        """One API call. Use `batches()` to split large inputs first."""
        payload = {"input": texts, "model": self.model, "input_type": input_type}
        if self.dim:
            payload["output_dimension"] = self.dim
        data = self._post(payload)
        rows = sorted(data["data"], key=lambda d: d["index"])
        return EmbedResult([r["embedding"] for r in rows], data["usage"]["total_tokens"])
