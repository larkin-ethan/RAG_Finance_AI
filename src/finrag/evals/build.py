"""Generate the XBRL-grounded part of the eval set.

  python -m finrag.evals.build   ->  evals/questions/generated.jsonl

For each watchlist company: take XBRL facts (revenue, net income, EPS, ...),
match each to the filing whose period it reports, and locate the number in
that filing's chunks. Only facts whose number is found become questions (the
miss rate is printed: it measures parser coverage). Then build question types:

  numeric       one fact, one filing (annual or quarterly)
  multi_filing  year-over-year change: two facts from two different 10-Ks
  comparison    same metric and fiscal year for two companies
  unanswerable  out-of-range years, companies we don't have, info no filing holds

Generated questions start with reviewed=false. Spot-check them before trusting
the numbers (see evals/README.md).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from finrag.chunking.base import Filing
from finrag.evals.dataset import EvalQuestion, Gold, assign_stratified_split, save_questions
from finrag.generate.citations import normalize
from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.fiscal import fiscal_period
from finrag.ingest.xbrl import Fact, extract_facts, load_company_facts
from finrag.settings import get_settings, load_watchlist

SEED = 42
TARGETS = {"annual": 30, "quarterly": 12, "multi_filing": 10, "comparison": 6}
SHORT_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson", "XOM": "ExxonMobil", "KO": "Coca-Cola", "COST": "Costco",
}  # fmt: skip


@dataclass
class Located:
    """An XBRL fact matched to its filing and found in that filing's chunks."""

    fact: Fact
    filing: Filing
    evidence: str
    fiscal_year: int
    fiscal_quarter: int | None


def short_name(filing: Filing) -> str:
    return SHORT_NAMES.get(filing.ticker, filing.company.title())


def evidence_candidates(fact: Fact) -> list[str]:
    """How the number is likely printed in the filing's tables."""
    v = abs(fact.value)
    if fact.metric.unit == "USD/shares":
        return [f"{v:.2f}"]
    out = []
    for scale in (1e6, 1e3):  # "(in millions)" is most common, then thousands
        scaled = v / scale
        if abs(scaled - round(scaled)) < 1e-6:
            out.append(f"{round(scaled):,}")
    return out


def money(fact: Fact) -> str:
    v, sign = abs(fact.value), "-" if fact.value < 0 else ""
    if fact.metric.unit == "USD/shares":
        return f"{sign}${v:.2f}"
    if v >= 1e9:
        return f"{sign}${v / 1e6:,.0f} million (about ${v / 1e9:,.1f} billion)"
    return f"{sign}${v / 1e6:,.0f} million"


def period_ok(fact: Fact, filing: Filing) -> bool:
    if fact.end != filing.period_end:
        return False
    if fact.metric.instant:
        return True
    days = fact.days or 0
    return 330 <= days <= 380 if filing.form == "10-K" else 80 <= days <= 100


def locate(facts: list[Fact], filings: dict[str, Filing], chunks: dict[str, list[str]]):
    located, missed = [], 0
    for fact in facts:
        filing = filings.get(fact.accession_number)
        if filing is None or not period_ok(fact, filing):
            continue
        texts = chunks.get(fact.accession_number, [])
        evidence = next(
            (e for e in evidence_candidates(fact) if any(normalize(e) in t for t in texts)), None
        )
        if evidence is None:
            missed += 1
            continue
        fy, fq = fiscal_period(fact.end, filing.fiscal_year_end, filing.form)
        located.append(Located(fact, filing, evidence, fy, fq))
    return located, missed


def gold(loc: Located) -> Gold:
    f = loc.fact
    return Gold(
        accession_number=f.accession_number, ticker=f.ticker, evidence=loc.evidence,
        value=f.value, unit=f.metric.unit, fiscal_year=loc.fiscal_year,
        fiscal_quarter=loc.fiscal_quarter, concept=f.concept,
    )  # fmt: skip


def numeric_question(loc: Located, rng: random.Random) -> EvalQuestion:
    f, name, fy = loc.fact, short_name(loc.filing), loc.fiscal_year
    label = f.metric.label
    if loc.fiscal_quarter and f.metric.instant:
        when = f"the quarter ended {f.end:%B} {f.end.day}, {f.end.year}"
        question = f"What were {name}'s {label} as of the end of {when}?"
    elif loc.fiscal_quarter:
        when = f"the quarter ended {f.end:%B} {f.end.day}, {f.end.year}"
        question = rng.choice([
            f"What was {name}'s {label} for {when}?",
            f"How much {label} did {name} report for {when}?",
        ])  # fmt: skip
    elif f.metric.instant:
        when = f"the end of fiscal {fy}"
        question = f"What were {name}'s {label} at {when}?"
    else:
        when = f"fiscal {fy}"
        question = rng.choice([
            f"What was {name}'s {label} for fiscal year {fy}?",
            f"How much {label} did {name} report in fiscal {fy}?",
            f"According to its 10-K, what was {name}'s {label} in fiscal {fy}?",
        ])  # fmt: skip
    q = f"q{loc.fiscal_quarter}" if loc.fiscal_quarter else ""
    return EvalQuestion(
        id=f"num-{f.ticker.lower()}-{f.metric.key}-fy{fy}{q}",
        type="numeric",
        question=question,
        answerable=True,
        reference_answer=f"{name}'s {label} for {when} was {money(f)}.",
        gold=[gold(loc)],
        tickers=[f.ticker],
        fiscal_years=[fy],
        source="xbrl",
    )


def yoy_question(prev: Located, cur: Located) -> EvalQuestion:
    name, label = short_name(cur.filing), cur.fact.metric.label
    a, b = prev.fact.value, cur.fact.value
    change = (b - a) / abs(a) * 100 if a else 0.0
    direction = "an increase" if change >= 0 else "a decrease"
    instant = cur.fact.metric.instant  # balance-sheet items are "at the end of" a year

    def when(x: Located) -> str:
        return f"the end of fiscal {x.fiscal_year}" if instant else f"fiscal {x.fiscal_year}"

    def at(x: Located) -> str:
        return f"at the end of fiscal {x.fiscal_year}" if instant else f"in fiscal {x.fiscal_year}"

    verb = "were" if instant else "was"
    return EvalQuestion(
        id=f"yoy-{cur.fact.ticker.lower()}-{cur.fact.metric.key}-fy{cur.fiscal_year}",
        type="multi_filing",
        question=(f"How did {name}'s {label} change from {when(prev)} to {when(cur)}?"),
        answerable=True,
        reference_answer=(
            f"{name}'s {label} {verb} {money(prev.fact)} {at(prev)} and {money(cur.fact)} "
            f"{at(cur)}, {direction} of about {abs(change):.1f}%."
        ),
        gold=[gold(prev), gold(cur)],
        tickers=[cur.fact.ticker],
        fiscal_years=[prev.fiscal_year, cur.fiscal_year],
        source="xbrl",
    )


def comparison_question(x: Located, y: Located) -> EvalQuestion:
    nx, ny, label, fy = (
        short_name(x.filing),
        short_name(y.filing),
        x.fact.metric.label,
        x.fiscal_year,
    )
    hi, lo = (x, y) if x.fact.value >= y.fact.value else (y, x)
    return EvalQuestion(
        id=f"cmp-{x.fact.ticker.lower()}-{y.fact.ticker.lower()}-{x.fact.metric.key}-fy{fy}",
        type="comparison",
        question=f"Which company reported higher {label} in fiscal {fy}: {nx} or {ny}?",
        answerable=True,
        reference_answer=(
            f"{short_name(hi.filing)} reported higher {label} in fiscal {fy} "
            f"({money(hi.fact)}) than {short_name(lo.filing)} ({money(lo.fact)})."
        ),
        gold=[gold(x), gold(y)],
        tickers=[x.fact.ticker, y.fact.ticker],
        fiscal_years=[fy],
        source="xbrl",
    )


def unanswerable_questions(tickers: list[str]) -> list[EvalQuestion]:
    """Realistic questions the corpus cannot answer. The system should abstain."""
    names = [SHORT_NAMES.get(t, t) for t in tickers]
    rows = [
        # Years outside the corpus (we hold ~2023 onward).
        *[(f"oor-{n}", f"What was {n}'s total revenue in fiscal 2012?") for n in names[:4]],
        *[(f"future-{n}", f"What will {n}'s net income be in fiscal 2030?") for n in names[4:7]],
        # Companies that aren't loaded.
        ("absent-tesla", "What was Tesla's total revenue for fiscal 2024?"),
        ("absent-amazon", "How much did Amazon spend on research and development in fiscal 2024?"),
        ("absent-pfizer", "What was Pfizer's diluted EPS for fiscal 2024?"),
        ("absent-walmart", "What were Walmart's total assets at the end of fiscal 2025?"),
        # Information filings don't contain.
        (f"price-{names[0]}", f"What is {names[0]}'s current stock price?"),
        (f"forecast-{names[1]}", f"What will {names[1]}'s stock price be at the end of next year?"),
        (f"personal-{names[2]}", f"What is the name of {names[2]}'s CEO's dog?"),
        (f"opinion-{names[3]}", f"Is {names[3]} a good stock to buy right now?"),
        (f"rumor-{names[5]}", f"Which company is {names[5]} secretly planning to acquire next year?"),
        (f"weather-{names[6]}", f"What was the weather at {names[6]}'s headquarters on its last filing date?"),
    ]  # fmt: skip
    return [
        EvalQuestion(
            id=f"unans-{key.lower().replace(' ', '-').replace('&', 'and')}",
            type="unanswerable",
            question=text,
            answerable=False,
            reference_answer="I don't know. The filings don't contain this information.",
            source="template",
        )
        for key, text in rows
    ]


def main() -> None:
    settings = get_settings()
    rng = random.Random(SEED)
    watchlist = load_watchlist(settings.watchlist_path)
    client = EdgarClient(settings.sec_user_agent)

    filings = {
        f.accession_number: f
        for f in (Filing.load(p) for p in (settings.data_dir / "parsed").glob("*/*.json"))
    }
    chunks: dict[str, list[str]] = defaultdict(list)
    with (settings.data_dir / "chunks" / "section-512.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            chunks[c["accession_number"]].append(normalize(c["text"]))

    located: list[Located] = []
    for ticker in watchlist.tickers:
        ciks = [
            client.get_cik(ticker),
            *(f"{int(c):010d}" for c in watchlist.extra_ciks.get(ticker, [])),
        ]
        facts = [
            fact
            for cik in ciks
            for fact in extract_facts(
                ticker, load_company_facts(client, cik, settings.data_dir / "xbrl")
            )
        ]
        found, missed = locate(facts, filings, chunks)
        located += found
        total = len(found) + missed
        print(f"{ticker:5} {len(found):4} facts located in chunks, {missed:3} not found"
              f" ({missed / total:.0%} miss)" if total else f"{ticker:5} no matching facts")  # fmt: skip

    annual = [x for x in located if x.fiscal_quarter is None]
    quarterly = [x for x in located if x.fiscal_quarter]

    def sample(items: list, n: int) -> list:
        """Spread picks evenly over (company, metric): round-robin over groups."""
        groups: dict[tuple, list] = defaultdict(list)
        for item in rng.sample(items, len(items)):
            first = item[0] if isinstance(item, tuple) else item
            groups[(first.fact.ticker, first.fact.metric.key)].append(item)
        picked, queues = [], rng.sample(list(groups.values()), len(groups))
        while len(picked) < n and any(queues):
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop())
        return picked

    by_key = {(x.fact.ticker, x.fact.metric.key, x.fiscal_year): x for x in annual}
    yoy_pairs = [
        (by_key[(t, m, fy - 1)], x)
        for (t, m, fy), x in by_key.items()
        if (t, m, fy - 1) in by_key
        and by_key[(t, m, fy - 1)].fact.accession_number != x.fact.accession_number
    ]
    cmp_pairs = [
        (x, y)
        for x in annual for y in annual
        if x.fact.ticker < y.fact.ticker and x.fact.metric.key == y.fact.metric.key
        and x.fiscal_year == y.fiscal_year and x.fact.metric.key in ("revenue", "net_income")
    ]  # fmt: skip

    questions = (
        [numeric_question(x, rng) for x in sample(annual, TARGETS["annual"])]
        + [numeric_question(x, rng) for x in sample(quarterly, TARGETS["quarterly"])]
        + [yoy_question(a, b) for a, b in sample(yoy_pairs, TARGETS["multi_filing"])]
        + [comparison_question(a, b) for a, b in sample(cmp_pairs, TARGETS["comparison"])]
        + unanswerable_questions(watchlist.tickers)
    )
    assign_stratified_split(questions)
    out = settings.data_dir.parent / "evals" / "questions" / "generated.jsonl"
    save_questions(questions, out)
    counts = defaultdict(int)
    for q in questions:
        counts[(q.type, q.split)] += 1
    print(f"\nWrote {len(questions)} questions to {out}")
    for t in ("numeric", "multi_filing", "comparison", "unanswerable"):
        print(f"  {t:13} dev {counts[(t, 'dev')]:3}  test {counts[(t, 'test')]:3}")


if __name__ == "__main__":
    main()
