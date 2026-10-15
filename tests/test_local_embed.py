import numpy as np
import pytest

from finrag.embed.cache import EmbeddingCache
from finrag.embed.embedder import Embedder
from finrag.embed.local_client import QUERY_INSTRUCTION, LocalClient


class FakeEncoder:
    """Stands in for SentenceTransformer: records calls, returns unit vectors."""

    def __init__(self, nan: bool = False):
        self.calls: list[dict] = []
        self.nan = nan

    def encode(self, texts, prompt=None, **kwargs):
        self.calls.append({"texts": list(texts), "prompt": prompt, **kwargs})
        out = np.zeros((len(texts), 4), dtype=np.float32)
        out[:, 0] = 1.0
        if self.nan:
            out[0, 1] = np.nan
        return out


def test_documents_have_no_prompt_queries_get_instruction():
    fake = FakeEncoder()
    client = LocalClient(encoder=fake)
    client.embed_batch(["Revenue grew."], "document")
    client.embed_batch(["What was revenue?"], "query")
    assert fake.calls[0]["prompt"] is None
    assert fake.calls[1]["prompt"] == QUERY_INSTRUCTION
    assert all(c["normalize_embeddings"] for c in fake.calls)


def test_nan_vectors_rejected():
    with pytest.raises(RuntimeError, match="NaN"):
        LocalClient(encoder=FakeEncoder(nan=True)).embed_batch(["x"], "document")


def test_query_cache_tag_tracks_instruction():
    client = LocalClient(encoder=FakeEncoder())
    assert client.cache_tag("document") == "document"
    assert client.cache_tag("query").startswith("query:")


def test_local_embedder_caches_and_has_no_budget(tmp_path):
    fake = FakeEncoder()
    e = Embedder(LocalClient(encoder=fake), EmbeddingCache(tmp_path / "c.sqlite"), budget=None)
    first = e.embed(["a", "b"])
    assert e.embed(["b", "a"]) == [first[1], first[0]]
    assert len(fake.calls) == 1
