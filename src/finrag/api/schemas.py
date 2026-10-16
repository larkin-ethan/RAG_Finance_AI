"""Request/response models for the API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

IssueType = Literal[
    "incorrect_answer",
    "wrong_or_missing_source",
    "made_up_number",
    "should_have_said_idk",
    "unhelpful",
    "other",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    tickers: list[str] = []
    fiscal_years: list[int] = []
    forms: list[str] = []
    k: int = Field(6, ge=1, le=20)


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z.\-]+$")
    since: date | None = None


class FeedbackRequest(BaseModel):
    rating: Literal[-1, 1]
    question: str = Field(min_length=1, max_length=2000)
    answer: str | None = Field(None, max_length=8000)
    issue_type: IssueType | None = None
    details: str | None = Field(None, max_length=4000)
    citations: list[dict] = Field(default_factory=list, max_length=20)
    model: str | None = Field(None, max_length=100)


# ---------- stock checks & portfolio (private deployment only) ----------


class LotIn(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    shares: float = Field(gt=0, lt=1e9)
    cost_per_share: float = Field(ge=0, lt=1e7)
    purchase_date: date
    account: str = Field("", max_length=60)


class CsvImportIn(BaseModel):
    csv_text: str = Field(max_length=1_000_000)
    mapping: dict[str, str]
    save_mapping_as: str | None = Field(None, max_length=40, pattern=r"^[\w .-]+$")
    dry_run: bool = True


class TaxProfileIn(BaseModel):
    filing_status: Literal["single", "married_joint", "head_of_household"]
    taxable_income: float = Field(ge=0, lt=1e9)
    magi: float | None = Field(None, ge=0, lt=1e9)
