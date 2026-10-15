"""Minimal client for SEC EDGAR's free JSON APIs.

EDGAR rules we respect (https://www.sec.gov/os/accessing-edgar-data):
  * Every request needs a User-Agent like "Name email@example.com".
    Without it SEC returns 403.
  * Max 10 requests/second. Going over gets your IP blocked for ~10 minutes.

Endpoints:
  * Ticker -> CIK map:  https://www.sec.gov/files/company_tickers.json
      {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
  * Filing history:     https://data.sec.gov/submissions/CIK##########.json
      (CIK zero-padded to 10 digits). The useful part is
      data["filings"]["recent"], a dict of *parallel arrays*:
      {"accessionNumber": [...], "form": [...], "filingDate": [...],
       "reportDate": [...], "primaryDocument": [...]}
      Index i across all arrays describes one filing.
      "recent" holds only the latest ~1000 filings. Heavy filers (banks issuing
      structured notes, companies with many Form 4s) burn through that in
      months, so older history is paged out to data["filings"]["files"]:
      [{"name": "CIK...-submissions-001.json", "filingFrom": ..., "filingTo": ...}]
      Each page has the same parallel arrays at its top level.
  * The document itself:
      https://www.sec.gov/Archives/edgar/data/{cik as int}/{accession without dashes}/{primaryDocument}
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class CompanyInfo:
    name: str  # e.g. "Apple Inc."
    fiscal_year_end: str  # "MMDD", e.g. "0928" (nominal; 52/53-week years drift a few days)


@dataclass(frozen=True)
class FilingRef:
    """Pointer to one filing. Everything we need to download and label it."""

    ticker: str
    cik: str  # 10-digit, zero-padded, e.g. "0000320193"
    form: str  # "10-K" or "10-Q"
    accession_number: str  # with dashes, e.g. "0000320193-24-000123"
    filing_date: date
    report_date: date | None  # period end; may be blank in EDGAR
    primary_document: str  # e.g. "aapl-20240928.htm"

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik_int=int(self.cik),
            accession_nodash=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )


class RateLimiter:
    """Allows at most `max_per_second` calls to `wait()` per second (thread-safe).

    Provided for you. Call `self._limiter.wait()` before every HTTP request.
    """

    def __init__(self, max_per_second: float):
        self._min_interval = 1.0 / max_per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            sleep_for = self._last + self._min_interval - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        max_requests_per_second: float = 8.0,  # stay safely under SEC's 10/s
        transport: httpx.BaseTransport | None = None,  # tests inject a fake here
    ):
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent containing a contact email")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = RateLimiter(max_requests_per_second)
        self._ticker_to_cik: dict[str, str] | None = None  # lazy cache

    def _get(self, url: str) -> httpx.Response:
        """GET with rate limiting. Retries 429/5xx up to 3 attempts with
        exponential backoff (1s, 2s); any other error status raises immediately."""
        for attempt in range(3):
            self._limiter.wait()
            response = self._http.get(url)
            is_temporary = response.status_code == 429 or response.status_code >= 500
            if is_temporary and attempt < 2:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()  # no-op on 2xx; raises on the final failure
            return response
        raise AssertionError("unreachable")

    def get_cik(self, ticker: str) -> str:
        """Return the 10-digit zero-padded CIK for a ticker (case-insensitive).
        The ticker map is downloaded once and cached on the client."""
        if self._ticker_to_cik is None:
            data = self._get(TICKERS_URL).json()
            self._ticker_to_cik = {
                row["ticker"].upper(): f"{row['cik_str']:010d}" for row in data.values()
            }
        try:
            return self._ticker_to_cik[ticker.upper()]
        except KeyError:
            raise KeyError(f"Unknown ticker: {ticker}") from None

    def company_info(self, ticker: str) -> CompanyInfo:
        """Name and fiscal year-end from the ticker's current submissions JSON."""
        data = self._get(SUBMISSIONS_URL.format(cik=self.get_cik(ticker))).json()
        return CompanyInfo(name=data["name"], fiscal_year_end=data.get("fiscalYearEnd") or "1231")

    def list_filings(
        self,
        ticker: str,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        since: date | None = None,
        extra_ciks: tuple[str, ...] = (),
    ) -> list[FilingRef]:
        """List filings of the given forms (exact match, so amendments like
        "10-K/A" are excluded) filed on/after `since`, newest first.

        `extra_ciks`: predecessor entities whose filings belong to this ticker.
        After a holding-company reorganization or merger, the ticker maps to a
        new CIK whose history starts at the reorg date (e.g. XOM in 2026).
        """
        refs: dict[str, FilingRef] = {}  # keyed by accession: pages/CIKs may overlap
        for cik in (self.get_cik(ticker), *(f"{int(c):010d}" for c in extra_ciks)):
            for arrays in self._submission_pages(cik, since):
                for ref in self._rows(ticker.upper(), cik, arrays):
                    if ref.form in forms and (since is None or ref.filing_date >= since):
                        refs.setdefault(ref.accession_number, ref)
        return sorted(refs.values(), key=lambda r: r.filing_date, reverse=True)

    def _submission_pages(self, cik: str, since: date | None) -> list[dict]:
        """The "recent" arrays plus any older pages overlapping `since`."""
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        pages = [data["filings"]["recent"]]
        for page in data["filings"].get("files", []):
            if since is None or date.fromisoformat(page["filingTo"]) >= since:
                pages.append(self._get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json())
        return pages

    @staticmethod
    def _rows(ticker: str, cik: str, arrays: dict) -> list[FilingRef]:
        """Turn one page of parallel arrays into FilingRefs."""
        rows = zip(
            arrays["accessionNumber"],
            arrays["form"],
            arrays["filingDate"],
            arrays["reportDate"],
            arrays["primaryDocument"],
        )
        return [
            FilingRef(
                ticker=ticker,
                cik=cik,
                form=form,
                accession_number=accession,
                filing_date=date.fromisoformat(filed),
                report_date=date.fromisoformat(reported) if reported else None,
                primary_document=document,
            )
            for accession, form, filed, reported, document in rows
        ]

    def download(self, ref: FilingRef, dest_dir: Path) -> Path:
        """Save the primary document to dest_dir/ticker/{accession}_{document}.
        Skips the request if the file already exists (cheap re-runs)."""
        path = dest_dir / ref.ticker / f"{ref.accession_number}_{ref.primary_document}"
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._get(ref.url).content)
        return path
