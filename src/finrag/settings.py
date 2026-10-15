"""Central configuration, loaded from environment variables / .env."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sec_user_agent: str
    voyage_api_key: str = ""
    gemini_api_key: str = ""
    database_url: str = "postgresql://finrag:finrag@localhost:5432/finrag"
    watchlist_path: Path = Path("config/demo.yaml")
    data_dir: Path = Path("data")

    # Embeddings. "local" runs Qwen3-Embedding on this machine: free, no card,
    # no rate limits. "voyage" uses the Voyage API (see the budget guard).
    embed_provider: Literal["local", "voyage", "ollama"] = "local"
    local_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Voyage: voyage-4 has 200M free tokens; one full pass over the demo corpus
    # is ~8M. The budget guard refuses calls past this lifetime cap.
    voyage_model: str = "voyage-4"
    embed_dim: int = 1024
    embed_budget_tokens: int = 150_000_000
    # Low-limit mode, for Voyage accounts without a payment method (3 RPM, 10K
    # TPM): set VOYAGE_MAX_BATCH_TOKENS=8000 and VOYAGE_RPM=3 in .env. Unset = fast.
    voyage_max_batch_tokens: int | None = None
    voyage_rpm: float | None = None

    # LLM (Gemini API, free tier). Flash answers questions; Flash-Lite handles
    # cheap tasks (eval judge, query parsing). The judge is deliberately a
    # different model from the generator, to reduce self-preference bias.
    # Which LLM answers questions: "gemini" (public demo) or "ollama" (local,
    # private; nothing leaves the machine). See generate/llm.py.
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_judge_model: str = "gemma3:12b"  # a different model family from the answerer
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    gemini_model: str = "gemini-3.8-flash"
    gemini_judge_model: str = "gemini-3.5-flash-lite"
    # List prices (USD per 1M tokens) for the cost column in eval results. We
    # run on the free tier, but report what it would cost. Fill in from the
    # Gemini pricing page; left unset, cost shows as n/a.
    gemini_price_input_per_mtok: float | None = None
    gemini_price_output_per_mtok: float | None = None

    # API: reading feedback requires this token (header X-Admin-Token). Unset =
    # feedback can be submitted but not read through the API.
    admin_token: str = ""
    # Public demo mode: rate limits on /ask, and /ingest requires the admin token.
    public_demo: bool = False
    ask_per_minute: int = 6
    daily_question_cap: int = 300
    trust_proxy: bool = False  # True on Cloud Run (read client IP from X-Forwarded-For)
    # Private local deployment (security). All empty = disabled (public demo / dev).
    api_token: str = ""  # if set, every API call except /health needs "Authorization: Bearer"
    app_password_hash: str = ""  # argon2id hash; if set, the UI requires login
    field_encryption_keys: str = ""  # Fernet keys, newest first (see security/crypto.py)
    session_idle_minutes: int = 30
    # Locally: on API start, check EDGAR for new filings if the last check is >24h old.
    auto_update_on_start: bool = False


class Watchlist(BaseModel):
    tickers: list[str]
    forms: list[str] = ["10-K", "10-Q"]
    since: date | None = None
    # Predecessor CIKs per ticker, for companies that reorganized or merged:
    # {"XOM": ["34088"]}. See EdgarClient.list_filings.
    extra_ciks: dict[str, list[str]] = {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_watchlist(path: Path) -> Watchlist:
    return Watchlist.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
