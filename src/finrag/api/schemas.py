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
