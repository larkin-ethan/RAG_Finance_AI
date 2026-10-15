"""CLI: download all watchlist filings to data/raw/.

Each filing is saved as {accession}_{document}.htm plus a sidecar
{accession}.meta.json holding its FilingRef (form, dates, ...), which the
parser needs later and which can't be recovered from the HTML alone.

Usage: python -m finrag.ingest.fetch  (or `make fetch`)
"""

import json
from dataclasses import asdict
from pathlib import Path

from finrag.ingest.edgar_client import CompanyInfo, EdgarClient, FilingRef
from finrag.settings import get_settings, load_watchlist


def meta_path(html_path: Path, ref: FilingRef) -> Path:
    return html_path.parent / f"{ref.accession_number}.meta.json"


def save_filing(client: EdgarClient, ref: FilingRef, info: CompanyInfo, raw_dir: Path) -> Path:
    """Download one filing (skipped if present) and write its sidecar metadata.
    Returns the metadata path, which is what the parser reads."""
    path = client.download(ref, raw_dir)
    meta = asdict(ref) | {
        "html_file": path.name,
        "url": ref.url,
        "company": info.name,
        "fiscal_year_end": info.fiscal_year_end,
    }
    out = meta_path(path, ref)
    out.write_text(json.dumps(meta, default=str, indent=2), encoding="utf-8")
    return out


def main() -> None:
    settings = get_settings()
    watchlist = load_watchlist(settings.watchlist_path)
    client = EdgarClient(settings.sec_user_agent)
    raw_dir = settings.data_dir / "raw"

    for ticker in watchlist.tickers:
        info = client.company_info(ticker)
        extra = tuple(watchlist.extra_ciks.get(ticker, []))
        refs = client.list_filings(ticker, tuple(watchlist.forms), watchlist.since, extra)
        print(f"{ticker}: {len(refs)} filings")
        for ref in refs:
            save_filing(client, ref, info, raw_dir)
            print(f"  {ref.form} {ref.filing_date} -> {ref.accession_number}")


if __name__ == "__main__":
    main()
