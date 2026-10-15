import json

import httpx
import pytest

from finrag.generate import llm as llm_module
from finrag.generate.answer import IDK, answer_question
from finrag.generate.citations import extract_numbers, quote_in_text, unsupported_numbers, verify
from finrag.generate.llm import CachedLLM, GeminiClient, LLMError, LLMResponse
from finrag.retrieve.vector import Hit

APPLE_TABLE = (
    "The following table shows net sales by category for 2024, 2023 and 2022 "
    "(dollars in millions):\n\n"
    "|  | 2024 | Change | 2023 |\n|---|---|---|---|\n"
    "| iPhone | $201,183 | —% | $200,583 |\n"
    "| Services (1) | 96,169 | 13% | 85,200 |\n"
    "| Total net sales | $391,035 | 2% | $383,285 |"
)
RISK = "The Company’s operations are subject to risks — including supply chain disruption."


def hit(text: str, ticker: str = "AAPL") -> Hit:
    return Hit("c1", 0.9, ticker, "Apple Inc.", "10-K", 2024, None, "2024-09-28", "7",
               "MD&A", "https://sec.gov/x", text)  # fmt: skip


# ---------- quote check ----------


def test_quote_normalizes_whitespace_quotes_dashes_case():
    assert quote_in_text("the company's operations are subject to risks - including", RISK)


def test_quote_across_table_pipes():
    assert quote_in_text("Total net sales $391,035 2% $383,285", APPLE_TABLE)


def test_quote_with_ellipsis_must_be_in_order():
    assert quote_in_text("Company’s operations ... supply chain disruption", RISK)
    assert not quote_in_text("supply chain disruption ... Company’s operations", RISK)


def test_fabricated_quote_rejected():
    assert not quote_in_text("Total net sales were $400 billion", APPLE_TABLE)
    assert not quote_in_text("...", APPLE_TABLE)


# ---------- number check ----------


def test_extract_skips_labels_years_dates_and_small_ints():
    nums = extract_numbers(
        "In FY2024 Q3 (Item 7A, Note 14) on September 28, 2024, 3 segments grew 13% to $96,169 [1]."
    )
    assert [n.raw for n in nums] == ["13%", "$96,169"]


def test_scaled_answer_matches_table_in_millions():
    assert unsupported_numbers("Total net sales were $391.0 billion [1].", [APPLE_TABLE]) == []
    assert unsupported_numbers("Total net sales were $391,035 million.", [APPLE_TABLE]) == []
    assert unsupported_numbers("Total net sales were about $391 billion.", [APPLE_TABLE]) == []


def test_wrong_number_flagged():
    assert unsupported_numbers("Total net sales were $395.0 billion.", [APPLE_TABLE]) == [
        "$395.0 billion"
    ]


def test_percent_only_matches_percent():
    assert unsupported_numbers("Services grew 13%.", [APPLE_TABLE]) == []
    assert unsupported_numbers("Services grew 14%.", [APPLE_TABLE]) == ["14%"]


# ---------- verify ----------


def test_verify_grounded_answer():
    v = verify(
        "Apple's total net sales were $391,035 million in fiscal 2024 [1].",
        [{"source": 1, "quote": "Total net sales | $391,035"}],
        [APPLE_TABLE],
    )
    assert v.grounded and v.verified_sources == {1}


def test_verify_flags_bad_source_and_uncited_marker():
    v = verify(
        "Sales were $391,035 million [1][2].",
        [{"source": 5, "quote": "Total net sales"}],
        [APPLE_TABLE],
    )
    assert not v.grounded
    assert v.citations[0].valid_source is False
    assert v.uncited_markers == [1, 2]


# ---------- Gemini client ----------


class FakeGemini:
    def __init__(self, reply: dict | None = None, status: int = 200):
        self.requests: list[httpx.Request] = []
        self.reply, self.status = reply, status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="quota exceeded")
        return httpx.Response(200, json=self.reply)


def gemini_reply(payload: dict, thought: str | None = None) -> dict:
    parts = ([{"text": thought, "thought": True}] if thought else []) + [
        {"text": json.dumps(payload)}
    ]
    return {
        "candidates": [{"content": {"parts": parts}}],
        "usageMetadata": {
            "promptTokenCount": 900,
            "candidatesTokenCount": 60,
            "thoughtsTokenCount": 40,
        },
    }


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)


def test_gemini_request_shape_and_thought_parts_skipped():
    fake = FakeGemini(gemini_reply({"ok": True}, thought="Let me think..."))
    client = GeminiClient("secret", "gemini-test", transport=httpx.MockTransport(fake.handler))
    response = client.generate("sys", "question?", schema={"type": "OBJECT"})

    req = fake.requests[0]
    assert req.headers["x-goog-api-key"] == "secret"
    assert "secret" not in str(req.url)  # key never in the URL
    assert str(req.url).endswith("/models/gemini-test:generateContent")
    body = json.loads(req.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert response.json() == {"ok": True}
    assert (response.input_tokens, response.output_tokens) == (900, 100)


def test_gemini_error_surfaces_message():
    fake = FakeGemini(status=400)
    client = GeminiClient("k", "m", transport=httpx.MockTransport(fake.handler))
    with pytest.raises(LLMError, match="400: quota exceeded"):
        client.generate("s", "p")


def test_json_fence_tolerated():
    assert LLMResponse('```json\n{"a": 1}\n```', "m").json() == {"a": 1}


def test_cached_llm_second_call_free(tmp_path):
    fake = FakeGemini(gemini_reply({"x": 1}))
    inner = GeminiClient("k", "m", transport=httpx.MockTransport(fake.handler))
    cached = CachedLLM(inner, tmp_path / "llm.sqlite")
    first = cached.generate("s", "p", {"type": "OBJECT"})
    second = CachedLLM(inner, tmp_path / "llm.sqlite").generate("s", "p", {"type": "OBJECT"})
    assert len(fake.requests) == 1
    assert second.cached and second.text == first.text
    cached.generate("s", "different prompt")
    assert len(fake.requests) == 2


# ---------- answer_question ----------


class ScriptedLLM:
    model = "scripted"

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload, self.error, self.prompts = payload, error, []

    def generate(self, system, prompt, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return LLMResponse(json.dumps(self.payload), self.model, 1000, 50, 0.4)


def test_grounded_answer_end_to_end():
    llm = ScriptedLLM({
        "answerable": True,
        "answer": "Apple's total net sales were $391,035 million in fiscal 2024 [1].",
        "citations": [{"source": 1, "quote": "Total net sales | $391,035"}],
    })  # fmt: skip
    ans = answer_question("Apple FY2024 revenue?", lambda q: [hit(APPLE_TABLE)], llm)
    assert ans.answerable and ans.grounded
    assert ans.citations[0].verified and ans.citations[0].hit.ticker == "AAPL"
    assert "[1] Apple Inc. (AAPL) | Form 10-K | FY2024" in llm.prompts[0]


def test_strict_mode_turns_unverifiable_answer_into_idk():
    llm = ScriptedLLM({
        "answerable": True,
        "answer": "Revenue was $500 billion [1].",
        "citations": [{"source": 1, "quote": "Revenue was $500 billion"}],
    })  # fmt: skip
    ans = answer_question("q", lambda q: [hit(APPLE_TABLE)], llm)
    assert ans.text == IDK and ans.abstain_reason == "no citation could be verified"


def test_partially_grounded_answer_is_flagged_not_hidden():
    llm = ScriptedLLM({
        "answerable": True,
        "answer": "Total net sales were $391,035 million, up 2.02% [1].",
        "citations": [{"source": 1, "quote": "Total net sales | $391,035"}],
    })  # fmt: skip
    ans = answer_question("q", lambda q: [hit(APPLE_TABLE)], llm)
    assert ans.answerable and not ans.grounded
    assert ans.unsupported_numbers == ["2.02%"]  # computed, not in the source


def test_not_answerable_and_llm_errors_become_idk():
    llm = ScriptedLLM({"answerable": False, "answer": "Not in sources", "citations": []})
    assert answer_question("q", lambda q: [hit(RISK)], llm).text == IDK
    broken = ScriptedLLM(error=LLMError("boom"))
    assert answer_question("q", lambda q: [hit(RISK)], broken).abstain_reason.startswith("LLM")
    assert answer_question("q", lambda q: [], llm).abstain_reason == "no passages retrieved"


def test_eps_needs_exact_match():
    source = "Diluted earnings per share | $6.08 | $6.13"
    assert unsupported_numbers("Diluted EPS was $6.08.", [source]) == []
    assert unsupported_numbers("Diluted EPS was $6.11.", [source]) == ["$6.11"]
