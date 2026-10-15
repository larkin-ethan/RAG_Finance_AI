import json

import httpx
import pytest

from finrag.embed import voyage_client
from finrag.embed.budget import BudgetExceeded, TokenBudget, free_cap
from finrag.embed.cache import EmbeddingCache
from finrag.embed.embedder import Embedder
from finrag.embed.voyage_client import VoyageAPIError, VoyageClient


class FakeVoyage:
    """Returns a deterministic 3-dim vector per text; bills 1 token per char."""

    def __init__(self, fail_first: int = 0):
        self.calls: list[dict] = []
        self.fail_first = fail_first

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(429)
        body = json.loads(request.content)
        self.calls.append(body | {"auth": request.headers["Authorization"]})
        data = [
            {"object": "embedding", "embedding": [float(len(t)), float(i), 1.0], "index": i}
            for i, t in enumerate(body["input"])
        ]
        data.reverse()  # API order isn't guaranteed; client must sort by index
        tokens = sum(len(t) for t in body["input"])
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": tokens}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(voyage_client.time, "sleep", lambda s: None)


def client(fake: FakeVoyage, model: str = "voyage-4") -> VoyageClient:
    return VoyageClient("key-123", model, 1024, transport=httpx.MockTransport(fake.handler))


def embedder(tmp_path, fake: FakeVoyage, limit: int = 10_000) -> Embedder:
    return Embedder(
        client(fake),
        EmbeddingCache(tmp_path / "emb.sqlite"),
        TokenBudget(tmp_path / "ledger.json", limit, "voyage-4"),
    )


def test_requires_api_key():
    with pytest.raises(ValueError):
        VoyageClient("")


def test_embed_batch_payload_and_order():
    fake = FakeVoyage()
    result = client(fake).embed_batch(["a", "bbb"], "document")
    assert result.vectors == [[1.0, 0.0, 1.0], [3.0, 1.0, 1.0]]  # sorted by index
    assert result.tokens == 4
    call = fake.calls[0]
    assert call["auth"] == "Bearer key-123"
    assert (call["model"], call["input_type"], call["output_dimension"]) == (
        "voyage-4",
        "document",
        1024,
    )


def test_retries_rate_limit():
    fake = FakeVoyage(fail_first=2)
    assert client(fake).embed_batch(["x"], "query").tokens == 1
    assert len(fake.calls) == 1


def test_batches_respect_token_limit():
    big = "word " * 50_000  # ~50k tokens each
    batches = list(client(FakeVoyage(), model="voyage-finance-2").batches([big] * 4))
    # voyage-finance-2 caps requests at 120k tokens; we target half of that.
    assert [len(b) for b in batches] == [1, 1, 1, 1]
    assert len(list(client(FakeVoyage()).batches(["hi"] * 2500))) == 3  # 1000 texts max


def test_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    cache.put_many({"k1": [0.5, -1.25]})
    assert cache.get_many(["k1", "missing"]) == {"k1": [0.5, -1.25]}
    assert len(cache) == 1


def test_second_run_is_free(tmp_path):
    fake = FakeVoyage()
    e = embedder(tmp_path, fake)
    first = e.embed(["alpha", "beta", "alpha"])
    assert len(fake.calls) == 1 and fake.calls[0]["input"] == ["alpha", "beta"]  # deduped

    again = embedder(tmp_path, fake)  # new process, same disk cache
    assert again.embed(["beta", "alpha"]) == [first[1], first[0]]
    assert len(fake.calls) == 1  # no new API call
    assert again.stats.cached == 2


def test_query_and_document_cached_separately(tmp_path):
    fake = FakeVoyage()
    e = embedder(tmp_path, fake)
    e.embed(["revenue"], "document")
    e.embed_query("revenue")
    assert [c["input_type"] for c in fake.calls] == ["document", "query"]


def test_budget_blocks_before_calling_api(tmp_path):
    fake = FakeVoyage()
    e = embedder(tmp_path, fake, limit=3)
    with pytest.raises(BudgetExceeded):
        e.embed(["one two three four five"])
    assert fake.calls == []


def test_budget_persists_billed_tokens(tmp_path):
    e = embedder(tmp_path, FakeVoyage())
    e.embed(["abcd"])
    assert TokenBudget(tmp_path / "ledger.json", 10_000, "voyage-4").spent == 4


def test_api_error_message_surfaced():
    fake = FakeVoyage(fail_first=99)
    with pytest.raises(VoyageAPIError, match="429 after 6 attempt"):
        client(fake).embed_batch(["x"], "document")


def test_low_limit_mode_caps_batch_tokens():
    c = VoyageClient(
        "k",
        "voyage-4",
        1024,
        transport=httpx.MockTransport(FakeVoyage().handler),
        max_batch_tokens=8000,
        requests_per_minute=3,
    )
    batches = list(c.batches(["word " * 3000] * 5))  # ~3k tokens each
    assert [len(b) for b in batches] == [2, 2, 1]  # <= 8k tokens per request


def test_cap_never_exceeds_model_free_allowance(tmp_path):
    big = 10**12
    assert TokenBudget(tmp_path / "l.json", big, "voyage-4").limit == 180_000_000
    assert TokenBudget(tmp_path / "l.json", big, "voyage-finance-2").limit == 45_000_000
    assert TokenBudget(tmp_path / "l.json", 1_000, "voyage-4").limit == 1_000


def test_model_without_free_tokens_is_refused(tmp_path):
    assert free_cap("voyage-3.5") == 0
    budget = TokenBudget(tmp_path / "l.json", 10**9, "voyage-3.5")
    with pytest.raises(BudgetExceeded):
        budget.check(1)
