from finrag.retrieve.filters import Filters
from finrag.retrieve.hybrid import rrf
from finrag.retrieve.keyword import build_keyword_sql
from finrag.retrieve.query_parse import QueryParser
from finrag.retrieve.rerank import Reranker
from finrag.retrieve.retriever import RetrievalConfig, Retriever
from finrag.retrieve.vector import Hit

PARSER = QueryParser.for_tickers(["AAPL", "MSFT", "NVDA", "JNJ", "KO", "COST"])


def h(cid: str, text: str = "", ticker: str = "AAPL") -> Hit:
    return Hit(cid, 0.0, ticker, "Co", "10-K", 2024, None, "2024-09-28", "7", "MD&A", "u", text)


# ---------- RRF ----------


def test_rrf_rewards_agreement_between_lists():
    vector = [h("a"), h("b"), h("c")]
    keyword = [h("c"), h("d"), h("b")]
    fused = [x.chunk_id for x in rrf([vector, keyword])]
    assert fused[:2] == ["c", "b"]  # in both lists beats #1 in only one
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_limit():
    assert len(rrf([[h(str(i)) for i in range(50)]], limit=10)) == 10


# ---------- query parsing ----------


def test_parse_companies_and_fiscal_years():
    f = PARSER.parse("How did Apple's net income change from fiscal 2023 to FY2024?")
    assert f.tickers == ["AAPL"] and f.fiscal_years == [2023, 2024]


def test_parse_comparison_and_aliases():
    f = PARSER.parse("Which reported higher revenue in fiscal year 2025: J&J or Coca-Cola?")
    assert sorted(f.tickers) == ["JNJ", "KO"] and f.fiscal_years == [2025]


def test_calendar_dates_never_become_fiscal_years():
    # NVIDIA's quarter ended July 27, 2025 is fiscal 2026.
    f = PARSER.parse("What was NVIDIA's revenue for the quarter ended July 27, 2025?")
    assert f.tickers == ["NVDA"] and f.fiscal_years is None


def test_lowercase_cost_is_not_costco_but_name_is():
    assert PARSER.tickers("What drove the cost of revenue at Microsoft?") == ["MSFT"]
    assert PARSER.tickers("How does Costco's membership model work?") == ["COST"]
    assert PARSER.tickers("Compare COST and AAPL margins") == ["AAPL", "COST"]


def test_forms():
    assert PARSER.forms("In its latest 10-Q, what did Apple say?") == ["10-Q"]
    assert PARSER.forms("According to the annual report") == ["10-K"]


def test_unknown_company_yields_no_ticker_filter():
    assert PARSER.parse("What was Tesla's revenue in fiscal 2024?").tickers is None


# ---------- keyword SQL ----------


def test_keyword_query_ors_terms_and_keeps_filters():
    sql, params = build_keyword_sql(Filters(tickers=["AAPL"]))
    assert "replace(plainto_tsquery('english', %(q)s)::text, '&', '|')" in sql
    assert "c.ticker = ANY(%(tickers)s)" in sql and params == {"tickers": ["AAPL"]}


# ---------- Retriever ----------


class FakeScorer:
    def predict(self, pairs, batch_size=16):
        return [float(len(text)) for _, text in pairs]  # longer passage = more relevant


def test_retriever_hybrid_auto_filters_and_rerank():
    calls = []

    def vector_fn(q, n, f):
        calls.append(("vector", n, f.tickers))
        return [h("a", "x"), h("b", "xxxx")]

    def keyword_fn(q, n, f):
        calls.append(("keyword", n, f.tickers))
        return [h("b", "xxxx"), h("c", "xxxxxxxx")]

    r = Retriever(
        RetrievalConfig(method="hybrid", filters="auto", rerank=True, candidates=20),
        vector_fn, keyword_fn, PARSER, Reranker(scorer=FakeScorer()),
    )  # fmt: skip
    hits = r("Apple revenue in fiscal 2024?", k=2)
    assert calls == [("vector", 20, ["AAPL"]), ("keyword", 20, ["AAPL"])]
    assert [x.chunk_id for x in hits] == ["c", "b"]  # reranker reorders the fused pool
    assert r.last_filters.fiscal_years == [2024]


def test_plain_vector_asks_for_exactly_k():
    seen = []
    r = Retriever(RetrievalConfig(), vector_fn=lambda q, n, f: seen.append(n) or [h("a")])
    r("anything", k=6)
    assert seen == [6]
