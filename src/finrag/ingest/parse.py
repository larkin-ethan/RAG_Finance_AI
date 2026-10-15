"""CLI: parse every downloaded filing into sections.

Reads  data/raw/{TICKER}/{accession}.meta.json (+ the HTML next to it)
Writes data/parsed/{TICKER}/{accession}.json = filing metadata + sections

Prints the Items found per filing so parsing gaps are easy to spot,
e.g. a 10-K with no "7" (MD&A) means heading detection missed it.

Usage: python -m finrag.ingest.parse  (or `make parse`)
"""

import json
from dataclasses import asdict
from pathlib import Path

from finrag.ingest.parse_html import parse_blocks
from finrag.ingest.sections import PREAMBLE, split_sections
from finrag.settings import get_settings

# Items we expect in nearly every filing; missing ones are flagged.
EXPECTED = {"10-K": {"1", "1A", "7", "8"}, "10-Q": {"I-1", "I-2"}}


def parse_filing(meta_file: Path, out_dir: Path) -> dict:
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    html = (meta_file.parent / meta["html_file"]).read_bytes()
    sections = split_sections(parse_blocks(html), meta["form"])

    out = out_dir / meta["ticker"] / f"{meta['accession_number']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(meta | {"sections": [asdict(s) for s in sections]}, indent=1), encoding="utf-8"
    )
    return meta | {"keys": [s.key for s in sections if s.key != PREAMBLE]}


def main() -> None:
    settings = get_settings()
    raw_dir, out_dir = settings.data_dir / "raw", settings.data_dir / "parsed"
    meta_files = sorted(raw_dir.glob("*/*.meta.json"))
    if not meta_files:
        print(f"No filings in {raw_dir}. Run `make fetch` first.")
        return

    flagged = 0
    for meta_file in meta_files:
        info = parse_filing(meta_file, out_dir)
        missing = EXPECTED.get(info["form"], set()) - set(info["keys"])
        flag = f"  !! missing {sorted(missing)}" if missing else ""
        flagged += bool(missing)
        print(
            f"{info['ticker']:6} {info['form']:5} {info['filing_date']}  {' '.join(info['keys'])}{flag}"
        )
    print(f"\nParsed {len(meta_files)} filings, {flagged} flagged.")


if __name__ == "__main__":
    main()
