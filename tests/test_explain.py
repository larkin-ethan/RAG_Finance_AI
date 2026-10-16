import json

from finrag.generate.llm import LLMResponse
from finrag.ratings.explain import explain, fact_lines, template
from finrag.ratings.goals import score_goal

GOOD = {"dividend_yield": 0.035, "dividend_yield_spread": -0.005, "payout_ratio": 0.55, "fcf_payout": 0.6,
        "dividend_cagr_5y": 0.06, "years_without_cut": 15, "net_debt_to_ebitda": 1.5, "interest_coverage": 12}  # fmt: skip
RISK = "Our suppliers in Asia may fail to deliver components, causing shortages and production delays for us."


class Scripted:
    model = "scripted"

    def __init__(self, payload):
        self.payload, self.prompts = payload, []

    def generate(self, system, prompt, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse(json.dumps(self.payload), self.model)


def results():
    return [score_goal("dividend", GOOD)]


def test_facts_and_template_are_deterministic():
    lines = fact_lines("KO", results())
    assert lines[0].startswith("KO Dividend income: Strong fit")
    assert any("Dividend yield: 3.5%" in line for line in lines)
    assert template("KO", results()).startswith("Dividend income: Strong fit")


def test_verified_llm_rationale_is_used():
    llm = Scripted({"summary": "KO is a Strong fit for dividend income: its dividend yield is 3.5% and it has "
                               "15 years without a dividend cut. A new risk notes supply problems [R1].",
                    "citations": [{"source": 1, "quote": "may fail to deliver components"}]})  # fmt: skip
    e = explain("KO", results(), [RISK], llm)
    assert e.source == "llm" and e.risk_quotes == [(1, "may fail to deliver components")]
    assert "[R1] <<<" in llm.prompts[0]


def test_made_up_number_falls_back_to_template():
    llm = Scripted({"summary": "KO yields 6.2% with a strong record.", "citations": []})
    e = explain("KO", results(), [], llm)
    assert e.source == "template" and "unsupported numbers" in e.rejected_reason


def test_fake_quote_and_advice_wording_rejected():
    fake = Scripted(
        {
            "summary": "KO is a Strong fit [R1].",
            "citations": [{"source": 1, "quote": "bankruptcy is near"}],
        }
    )
    assert explain("KO", results(), [RISK], fake).rejected_reason == "unverified quote [R1]"
    advice = Scripted({"summary": "Given this, you should buy more KO.", "citations": []})
    assert explain("KO", results(), [], advice).source == "template"


def test_no_llm_uses_template():
    assert explain("KO", results(), [], None).source == "template"
