import re
from collections import Counter

from finrag.ratings.risk_diff import diff_risk_factors, negative_share, new_red_flags, paragraphs

WORDS = [
    "the",
    "company",
    "faces",
    "competition",
    "from",
    "many",
    "rivals",
    "in",
    "every",
    "market",
    "where",
    "it",
    "sells",
    "products",
    "and",
    "services",
]


def bow_embed(texts):
    """Fake embedder: bag-of-words vectors over a shared vocabulary."""
    vocab = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
    out = []
    for t in texts:
        c = Counter(re.findall(r"[a-z]+", t.lower()))
        out.append([float(c[w]) for w in vocab])
    return out


def para(*extra: str) -> str:
    return " ".join(WORDS + list(extra) + WORDS)


def test_paragraphs_skip_short_lines_and_tables():
    text = f"Item 1A. Risk Factors\n\n{para()}\n\n| a | b |\n|---|---|\n\nshort line"
    assert paragraphs(text) == [para()]


def test_reworded_risk_matches_new_risk_does_not():
    supply = (
        "our suppliers in asia may fail to deliver components causing shortages and delays " * 3
    )
    prior = f"{para('pricing')}\n\n{para('regulation')}"
    current = f"{para('pricing', 'pressure')}\n\n{para('regulation')}\n\n{supply}"
    d = diff_risk_factors(current, prior, bow_embed)
    assert d.new_paragraphs == [supply.strip()]  # the reworded pricing risk still matches
    assert 0.2 < d.new_share < 0.5 and d.removed_share == 0


def test_red_flags_only_when_new():
    prior = "We are subject to financial covenants under our credit agreement."
    current = prior + " We were not in compliance with certain financial covenants as of year end."
    assert new_red_flags(current, prior) == ["covenant breach"]
    assert new_red_flags(current, current) == []  # already disclosed last year: not new
    gc = "There is substantial doubt about our ability to continue as a going concern."
    assert new_red_flags(gc, "") == ["going concern"]


def test_tone_change():
    assert negative_share("revenue grew strongly") == 0
    d = diff_risk_factors(para(), para(), bow_embed, current_mdna="sales declined amid weak demand",
                          prior_mdna="sales grew amid strong demand")  # fmt: skip
    assert d.tone_change > 0
