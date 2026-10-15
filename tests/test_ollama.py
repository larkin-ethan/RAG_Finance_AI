import json

import httpx
import pytest

from finrag.embed.cache import EmbeddingCache
from finrag.embed.embedder import Embedder
from finrag.embed.local_client import QUERY_INSTRUCTION
from finrag.embed.ollama_client import OllamaEmbedClient
from finrag.generate import llm as llm_module
from finrag.generate.answer import answer_question
from finrag.generate.llm import LLMError, OllamaClient, make_llm, to_json_schema
from finrag.generate.prompts import ANSWER_SCHEMA
from finrag.retrieve.vector import Hit
from finrag.settings import Settings


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)


class FakeOllama:
    def __init__(self, chat_reply: dict | None = None, dim: int = 4):
        self.requests: list[dict] = []
        self.chat_reply, self.dim = chat_reply or {}, dim

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, **body})
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[3.0, 4.0] + [0.0] * (self.dim - 2)
                                                            for _ in body["input"]]})  # fmt: skip
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": json.dumps(self.chat_reply)},
            "prompt_eval_count": 1200, "eval_count": 80,
        })  # fmt: skip


def test_schema_conversion_to_standard_json_schema():
    converted = to_json_schema(ANSWER_SCHEMA)
    assert converted["type"] == "object"
    assert (
        converted["properties"]["citations"]["items"]["properties"]["source"]["type"] == "integer"
    )
    assert "propertyOrdering" not in json.dumps(converted)


def test_ollama_chat_request_sets_context_and_schema():
    fake = FakeOllama({"answerable": False, "answer": "n/a", "citations": []})
    client = OllamaClient(
        "http://ollama:11434", "qwen3:14b", transport=httpx.MockTransport(fake.handler)
    )
    resp = client.generate("sys", "question?", ANSWER_SCHEMA)
    req = fake.requests[0]
    assert req["path"] == "/api/chat" and req["stream"] is False and req["think"] is False
    assert req["options"]["num_ctx"] >= 8192  # never silently truncate the evidence
    assert req["format"]["type"] == "object"
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert resp.json()["answerable"] is False
    assert (resp.input_tokens, resp.output_tokens) == (1200, 80)


def test_ollama_unreachable_raises_llm_error():
    def down(request):
        raise httpx.ConnectError("refused")

    client = OllamaClient("http://ollama:11434", "m", transport=httpx.MockTransport(down))
    with pytest.raises(LLMError, match="Can't reach Ollama"):
        client.generate("s", "p")


def test_answer_pipeline_works_with_ollama():
    table = "| | 2024 |\n|---|---|\n| Total net sales | $391,035 |"
    fake = FakeOllama({
        "answerable": True, "answer": "Total net sales were $391,035 million [1].",
        "citations": [{"source": 1, "quote": "Total net sales | $391,035"}],
    })  # fmt: skip
    client = OllamaClient("http://o", "m", transport=httpx.MockTransport(fake.handler))
    hit = Hit(
        "c", 0.9, "AAPL", "Apple Inc.", "10-K", 2024, None, "2024-09-28", "7", "MD&A", "u", table
    )
    ans = answer_question("Apple revenue FY2024?", lambda q: [hit], client)
    assert ans.grounded and ans.citations[0].verified


def test_ollama_embeddings_query_instruction_normalized_and_labeled(tmp_path):
    fake = FakeOllama(dim=4)
    client = OllamaEmbedClient("http://o", "qwen3-embedding:0.6b", dim=4,
                               transport=httpx.MockTransport(fake.handler))  # fmt: skip
    e = Embedder(client, EmbeddingCache(tmp_path / "c.sqlite"), budget=None)
    doc = e.embed(["revenue"])[0]
    e.embed_query("revenue")
    assert doc == pytest.approx([0.6, 0.8, 0.0, 0.0])  # L2-normalized
    assert fake.requests[0]["input"] == ["revenue"]
    assert fake.requests[1]["input"] == [QUERY_INSTRUCTION + "revenue"]
    assert fake.requests[0]["dimensions"] == 4
    assert client.model == "ollama/qwen3-embedding:0.6b"  # distinct label: never mixed


def test_wrong_dimension_rejected():
    fake = FakeOllama(dim=8)
    client = OllamaEmbedClient("http://o", dim=4, transport=httpx.MockTransport(fake.handler))
    with pytest.raises(RuntimeError, match="unexpected embeddings shape"):
        client.embed_batch(["x"], "document")


def test_make_llm_picks_provider_and_distinct_judge(tmp_path):
    s = Settings(sec_user_agent="t t@example.com", llm_provider="ollama", data_dir=tmp_path)
    answer, judge = make_llm(s, "answer"), make_llm(s, "judge")
    assert isinstance(answer.inner, OllamaClient)
    assert answer.model != judge.model
