"""Map a filing's period-end date to the company's fiscal year and quarter.

Why this exists: "Apple FY2025 Q1" ended Dec 28, 2024; NVIDIA's fiscal 2026
began in Feb 2025; J&J's fiscal 2022 ended Jan 1, 2023. Metadata filters like
"fiscal_year = 2024" only work if we label periods the way companies do.

Inputs: EDGAR's nominal fiscal year-end "MMDD" (e.g. Apple "0926"). Many
companies use 52/53-week years, so the actual end drifts up to a week around
the nominal date.

Rules:
  * The fiscal year containing period end p ends at the first nominal year-end
    on or after (p - 7 days). The slack absorbs 52/53-week drift.
  * That year is labeled by its end date's calendar year, except years ending
    in the first week of January, which are labeled by the prior year
    (J&J's "fiscal 2022" ended Jan 1, 2023). NVIDIA's late-January year-end is
    labeled forward ("fiscal 2025" ended Jan 26, 2025), as the rule implies.
  * Quarter = whole quarters elapsed since the previous fiscal year-end.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

DRIFT = timedelta(days=7)


def _nominal_end(year: int, mmdd: str) -> date:
    month, day = int(mmdd[:2]), int(mmdd[2:])
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def fiscal_year_end_for(period_end: date, mmdd: str) -> date:
    """Nominal end date of the fiscal year containing `period_end`."""
    for year in range(period_end.year - 1, period_end.year + 3):
        end = _nominal_end(year, mmdd)
        if end >= period_end - DRIFT:
            return end
    raise ValueError(f"no fiscal year end found for {period_end} / {mmdd}")


def fiscal_year_label(year_end: date) -> int:
    return year_end.year - 1 if (year_end.month == 1 and year_end.day <= 7) else year_end.year


def fiscal_period(period_end: date, mmdd: str, form: str) -> tuple[int, int | None]:
    """Return (fiscal_year, fiscal_quarter). Quarter is None for a 10-K."""
    year_end = fiscal_year_end_for(period_end, mmdd)
    fy = fiscal_year_label(year_end)
    if form.upper().startswith("10-K"):
        return fy, None
    prev_end = _nominal_end(year_end.year - 1, mmdd)
    quarter = round((period_end - prev_end).days / 91.3)
    return fy, min(max(quarter, 1), 3)
