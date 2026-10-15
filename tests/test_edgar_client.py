"""Tests for the EDGAR client. No network: a fake transport serves canned JSON."""

from datetime import date

import httpx
import pytest

from finrag.ingest import edgar_client
from finrag.ingest.edgar_client import EdgarClient, FilingRef

UA = "Test User test@example.com"

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}

SUBMISSIONS = {
    "name": "Apple Inc.",
    "fiscalYearEnd": "0928",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000081",
                "0000320193-24-000070",
                "0000320193-22-000108",
                "0000320193-24-000099",
            ],
            "form": ["10-K", "10-Q", "8-K", "10-K", "10-K/A"],
            "filingDate": ["2024-11-01", "2024-08-02", "2024-07-01", "2022-10-28", "2024-09-01"],
            "reportDate": ["2024-09-28", "", "2024-07-01", "2022-09-24", "2024-09-28"],
            "primaryDocument": [
                "aapl-20240928.htm",
                "aapl-20240629.htm",
                "x8k.htm",
                "aapl-20220924.htm",
                "amend.htm",
            ],
        }
    },
}


OLDER_PAGE = {
    "accessionNumber": ["0000320193-21-000105", "0000320193-19-000119"],
    "form": ["10-K", "10-K"],
    "filingDate": ["2021-10-29", "2019-10-31"],
    "reportDate": ["2021-09-25", "2019-09-28"],
    "primaryDocument": ["aapl-20210925.htm", "a10-k20199282019.htm"],
}
SUBMISSIONS_PAGED = {
    "filings": {
        "recent": SUBMISSIONS["filings"]["recent"],
        "files": [
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingFrom": "2021-01-01",
                "filingTo": "2022-06-30",
            },
            {
                "name": "CIK0000320193-submissions-002.json",
                "filingFrom": "2015-01-01",
                "filingTo": "2020-12-31",
            },
        ],
    }
}


class FakeSEC:
    """Records requests and serves canned responses. Set `fail_first` to simulate 503s."""

    def __init__(self, fail_first: int = 0, fail_status: int = 503, paged: bool = False):
        self.requests: list[httpx.Request] = []
        self.paged = paged
        self.fail_first = fail_first
        self.fail_status = fail_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(self.fail_status)
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "submissions/CIK0000320193.json" in url:
            return httpx.Response(200, json=SUBMISSIONS_PAGED if self.paged else SUBMISSIONS)
        if "submissions/CIK0000000042.json" in url:  # predecessor entity
            return httpx.Response(200, json={"filings": {"recent": OLDER_PAGE}})
        if "submissions/CIK0000320193-submissions-" in url:
            return httpx.Response(200, json=OLDER_PAGE)
        if "/Archives/edgar/data/" in url:
            return httpx.Response(200, content=b"<html>filing</html>")
        return httpx.Response(404)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make backoff/rate-limit sleeps instant so tests run fast."""
    monkeypatch.setattr(edgar_client.time, "sleep", lambda s: None)


def make_client(fake: FakeSEC) -> EdgarClient:
    return EdgarClient(UA, transport=httpx.MockTransport(fake.handler))


def test_requires_contact_email():
    with pytest.raises(ValueError):
        EdgarClient("no email here")


def test_user_agent_sent():
    fake = FakeSEC()
    make_client(fake).get_cik("AAPL")
    assert fake.requests[0].headers["User-Agent"] == UA


def test_get_cik_padded_case_insensitive_and_cached():
    fake = FakeSEC()
    client = make_client(fake)
    assert client.get_cik("aapl") == "0000320193"
    assert client.get_cik("MSFT") == "0000789019"
    assert len(fake.requests) == 1  # ticker map downloaded only once


def test_get_cik_unknown():
    with pytest.raises(KeyError):
        make_client(FakeSEC()).get_cik("ZZZZ")


def test_list_filings_filters_and_sorts():
    refs = make_client(FakeSEC()).list_filings("AAPL")
    assert [r.accession_number for r in refs] == [
        "0000320193-24-000123",
        "0000320193-24-000081",
        "0000320193-22-000108",
    ]  # 8-K and 10-K/A excluded; newest first
    assert refs[0].report_date == date(2024, 9, 28)
    assert refs[1].report_date is None  # blank reportDate
    assert refs[0].cik == "0000320193"
    assert refs[0].ticker == "AAPL"


def test_list_filings_since_and_forms():
    refs = make_client(FakeSEC()).list_filings("AAPL", forms=("10-K",), since=date(2023, 1, 1))
    assert [r.accession_number for r in refs] == ["0000320193-24-000123"]


def test_filing_url():
    ref = FilingRef(
        "AAPL",
        "0000320193",
        "10-K",
        "0000320193-24-000123",
        date(2024, 11, 1),
        None,
        "aapl-20240928.htm",
    )
    assert ref.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )


def test_download_writes_file_and_is_idempotent(tmp_path):
    fake = FakeSEC()
    client = make_client(fake)
    ref = client.list_filings("AAPL")[0]
    n_before = len(fake.requests)

    path = client.download(ref, tmp_path)
    assert path == tmp_path / "AAPL" / "0000320193-24-000123_aapl-20240928.htm"
    assert path.read_bytes() == b"<html>filing</html>"
    assert len(fake.requests) == n_before + 1

    assert client.download(ref, tmp_path) == path
    assert len(fake.requests) == n_before + 1  # no second request


def test_retries_on_503_then_succeeds():
    fake = FakeSEC(fail_first=2)
    assert make_client(fake).get_cik("AAPL") == "0000320193"
    assert len(fake.requests) == 3


def test_gives_up_after_three_attempts():
    fake = FakeSEC(fail_first=10)
    with pytest.raises(httpx.HTTPStatusError):
        make_client(fake).get_cik("AAPL")
    assert len(fake.requests) == 3


def test_no_retry_on_404():
    fake = FakeSEC(fail_first=1, fail_status=404)
    with pytest.raises(httpx.HTTPStatusError):
        make_client(fake).get_cik("AAPL")
    assert len(fake.requests) == 1


def test_list_filings_follows_older_pages_within_range():
    fake = FakeSEC(paged=True)
    refs = make_client(fake).list_filings("AAPL", forms=("10-K",), since=date(2021, 1, 1))
    assert [r.accession_number for r in refs] == [
        "0000320193-24-000123",
        "0000320193-22-000108",
        "0000320193-21-000105",  # only on an older page
    ]
    fetched = [str(r.url) for r in fake.requests]
    assert any("submissions-001" in u for u in fetched)
    assert not any("submissions-002" in u for u in fetched)  # ends before `since`


def test_list_filings_includes_predecessor_ciks():
    refs = make_client(FakeSEC()).list_filings("AAPL", forms=("10-K",), extra_ciks=("42",))
    by_accession = {r.accession_number: r for r in refs}
    assert "0000320193-21-000105" in by_accession  # from the predecessor
    assert by_accession["0000320193-21-000105"].cik == "0000000042"  # archive path uses its CIK
    assert by_accession["0000320193-24-000123"].cik == "0000320193"


def test_company_info():
    info = make_client(FakeSEC()).company_info("AAPL")
    assert (info.name, info.fiscal_year_end) == ("Apple Inc.", "0928")
