import json

import pytest
from fastapi.testclient import TestClient

from finrag.api.main import app
from finrag.api.services import Services
from finrag.generate.llm import LLMResponse
from finrag.retrieve.vector import Hit
from finrag.settings import Settings

TABLE = "| | 2024 |\n|---|---|\n| Total net sales | $391,035 |"


class ScriptedLLM:
    model = "scripted"

    def generate(self, system, prompt, schema=None, temperature=0.0):
        payload = {
            "answerable": True,
            "answer": "Total net sales were $391,035 million [1].",
            "citations": [{"source": 1, "quote": "Total net sales | $391,035"}],
        }
        return LLMResponse(json.dumps(payload), self.model, 800, 40, 0.3)


class FakeServices(Services):
    """Real request handling; fake retrieval and LLM, no DB or model needed."""

    def __init__(self):
        super().__init__(Settings(sec_user_agent="t t@example.com"))
        self.seen_filters = None

    @property
    def llm(self):
        return ScriptedLLM()

    def retrieve(self, question, filters, k):
        self.seen_filters = (filters, k)
        return [Hit("c1", 0.9, "AAPL", "Apple Inc.", "10-K", 2024, None, "2024-09-28", "7",
                    "MD&A", "https://sec.gov/x", TABLE)]  # fmt: skip

    def db_ok(self):
        return True

    def save_feedback(self, fb, client_ip):
        self.feedback = (fb, client_ip)
        return 7

    def list_feedback(self, limit=100):
        return [{"id": 7, "question": "q"}]


@pytest.fixture
def client():
    fake = FakeServices()
    app.state.services = fake
    yield TestClient(app), fake
    del app.state.services


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "db": True}


def test_ask_returns_grounded_cited_answer(client):
    c, fake = client
    r = c.post("/ask", json={"question": "Apple FY2024 revenue?", "tickers": ["aapl"], "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["citations"][0]["verified"] is True
    assert body["citations"][0]["hit"]["url"] == "https://sec.gov/x"
    filters, k = fake.seen_filters
    assert filters.tickers == ["aapl"] and k == 4  # normalized later in SQL


def test_ask_validates_input(client):
    c, _ = client
    assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    assert c.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_ingest_rejects_bad_ticker(client):
    c, _ = client
    assert c.post("/ingest", json={"ticker": "AAPL; DROP TABLE"}).status_code == 422


def test_unknown_job_404(client):
    c, _ = client
    assert c.get("/ingest/nope").status_code == 404


def test_feedback_submit_and_validation(client):
    c, fake = client
    body = {"rating": -1, "question": "Apple revenue?", "issue_type": "made_up_number",
            "details": "wrong number"}  # fmt: skip
    assert c.post("/feedback", json=body).json() == {"id": 7}
    assert fake.feedback[0].issue_type == "made_up_number"
    assert c.post("/feedback", json=body | {"issue_type": "nonsense"}).status_code == 422
    assert c.post("/feedback", json=body | {"rating": 5}).status_code == 422


def test_reading_feedback_requires_admin_token(client):
    c, fake = client
    assert c.get("/feedback").status_code == 403  # no token configured at all
    fake.settings.admin_token = "s3cret"
    assert c.get("/feedback", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert c.get("/feedback", headers={"X-Admin-Token": "s3cret"}).json() == [
        {"id": 7, "question": "q"}
    ]


def test_config_has_no_secrets(client):
    c, fake = client
    fake.settings.gemini_api_key = "AIza-secret"
    body = c.get("/config").json()
    assert body["llm_model"] and "AIza-secret" not in str(body)


def test_evals_lists_saved_runs(client, tmp_path, monkeypatch):
    import json as _json

    from finrag.api import main

    c, _ = client
    (tmp_path / "base-dev.json").write_text(_json.dumps({"summary": {"config": {"name": "base"}}}))
    (tmp_path / "broken.json").write_text("{not json")
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    runs = c.get("/evals").json()
    assert [r["file"] for r in runs] == ["base-dev.json"]


# ---------- public demo protections ----------

from finrag.api.ratelimit import RateLimiter, client_ip


def test_rate_limiter_window_and_daily_cap():
    now = [0.0]
    rl = RateLimiter(per_minute=2, daily_cap=3, clock=lambda: now[0])
    assert rl.check("a", "d1") is None and rl.check("a", "d1") is None
    assert "per minute" in rl.check("a", "d1")
    now[0] = 61.0  # window slides
    assert rl.check("a", "d1") is None
    assert "daily" in rl.check("b", "d1")  # 3 questions used today, globally
    assert rl.check("b", "d2") is None  # new day resets


def test_client_ip_uses_rightmost_forwarded_entry_only_behind_proxy():
    headers = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}  # first entry forged by client
    assert client_ip(headers, "10.0.0.1", trust_proxy=True) == "1.2.3.4"
    assert client_ip(headers, "10.0.0.1", trust_proxy=False) == "10.0.0.1"


def test_public_demo_limits_ask_and_locks_ingest(client):
    c, fake = client
    fake.settings.public_demo, fake.settings.ask_per_minute = True, 1
    app.state.limiter = RateLimiter(1, 100)
    assert c.post("/ask", json={"question": "Apple revenue?"}).status_code == 200
    r = c.post("/ask", json={"question": "Apple revenue?"})
    assert r.status_code == 429 and "per minute" in r.json()["detail"]
    assert c.post("/ingest", json={"ticker": "TSLA"}).status_code == 403
    del app.state.limiter


def test_database_down_returns_503_without_leaking_details(client):
    import psycopg

    c, fake = client

    def boom():
        raise psycopg.OperationalError('connection to server at "10.1.2.3", port 5432 failed')

    fake.companies = boom
    r = c.get("/companies")
    assert r.status_code == 503 and "10.1.2.3" not in r.text and "5432" not in r.text
