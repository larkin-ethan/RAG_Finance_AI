"""FinRAG: cited Q&A over SEC filings."""

import sys

# On Windows, output redirected to a file or pipe defaults to the legacy code
# page (cp1252), and printing symbols like ✓ or → raises UnicodeEncodeError.
# Force UTF-8 everywhere; "replace" guarantees printing never crashes a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
