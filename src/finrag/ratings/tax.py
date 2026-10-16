"""US federal tax ESTIMATES for selling a lot. Not tax advice.

  * Holding period: long-term only if held MORE than one year. Bought Jan 5,
    2025 -> sold Jan 5, 2026 is still short-term; Jan 6, 2026 is long-term.
  * Short-term gains: ordinary rates, computed incrementally on top of your
    other taxable income (tax with the gain minus tax without it).
  * Long-term gains: 0% / 15% / 20% bands, stacked on top of ordinary taxable
    income (as on the IRS Qualified Dividends and Capital Gain worksheet).
  * NIIT: 3.8% on the part of the gain that pushes MAGI over the threshold
    (assumes no other investment income; noted in the output).
Ignores: state tax, AMT, loss carryovers, other gains or losses this year.
Bracket data: config/tax/us_federal_<year>.yaml (versioned, sourced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

TAX_DIR = Path(__file__).resolve().parents[3] / "config" / "tax"
NIIT_RATE = 0.038
SUPPORTED = ("single", "married_joint", "head_of_household")


@dataclass
class TaxProfile:
    filing_status: str  # one of SUPPORTED
    taxable_income: float  # this year's estimated taxable income BEFORE the sale
    magi: float | None = None  # modified AGI before the sale (defaults to taxable + std deduction)


@dataclass
class TaxEstimate:
    gain: float
    term: str  # "short" | "long"
    days_held: int
    long_term_date: date  # first sale date that qualifies as long-term
    income_tax: float
    niit: float
    year: int
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.income_tax + self.niit

    @property
    def effective_rate(self) -> float | None:
        return self.total / self.gain if self.gain > 0 else None


@lru_cache
def load_params(year: int) -> dict:
    path = TAX_DIR / f"us_federal_{year}.yaml"
    if not path.exists():
        available = sorted(int(p.stem.rsplit("_", 1)[1]) for p in TAX_DIR.glob("us_federal_*.yaml"))
        if not available:
            raise FileNotFoundError(f"No tax tables in {TAX_DIR}")
        path = TAX_DIR / f"us_federal_{available[-1]}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def long_term_date(purchase: date) -> date:
    """First date a sale counts as long-term (held more than one year)."""
    try:
        anniversary = purchase.replace(year=purchase.year + 1)
    except ValueError:  # bought Feb 29
        anniversary = date(purchase.year + 1, 2, 28)
    return anniversary + timedelta(days=1)


def bracket_tax(income: float, brackets: list[list]) -> float:
    tax, lower = 0.0, 0.0
    for upper, rate in brackets:
        top = income if upper is None else min(income, upper)
        if top > lower:
            tax += (top - lower) * rate
        if upper is None or income <= upper:
            break
        lower = upper
    return tax


def ltcg_tax(ordinary_income: float, gain: float, bands: list[list]) -> float:
    """Long-term gain stacked on top of ordinary taxable income."""
    tax, start, end = 0.0, ordinary_income, ordinary_income + gain
    lower = 0.0
    for upper, rate in bands:
        band_top = float("inf") if upper is None else upper
        overlap = max(0.0, min(end, band_top) - max(start, lower))
        tax += overlap * rate
        lower = band_top
    return tax


def estimate_sale_tax(
    profile: TaxProfile, shares: float, cost_per_share: float, purchase: date, sale_price: float,
    sale_date: date,
) -> TaxEstimate:  # fmt: skip
    if profile.filing_status not in SUPPORTED:
        raise ValueError(
            f"Filing status '{profile.filing_status}' isn't supported (supported: {SUPPORTED})"
        )
    params = load_params(sale_date.year)
    p = params["filing_status"][profile.filing_status]
    gain = (sale_price - cost_per_share) * shares
    lt_date = long_term_date(purchase)
    term = "long" if sale_date >= lt_date else "short"
    est = TaxEstimate(gain, term, (sale_date - purchase).days, lt_date, 0.0, 0.0, params["year"])
    if params["year"] != sale_date.year:
        est.notes.append(f"No {sale_date.year} tax tables yet; used {params['year']} brackets.")
    if gain <= 0:
        est.notes.append("A loss: no tax on this sale. Net capital losses can offset gains and up to "
                         "$3,000 of ordinary income per year.")  # fmt: skip
        return est
    base = max(0.0, profile.taxable_income)
    if term == "short":
        est.income_tax = bracket_tax(base + gain, p["ordinary"]) - bracket_tax(base, p["ordinary"])
    else:
        est.income_tax = ltcg_tax(base, gain, p["ltcg"])
    magi = profile.magi if profile.magi is not None else base + p["standard_deduction"]
    est.niit = NIIT_RATE * max(0.0, min(gain, magi + gain - p["niit_threshold"]))
    if est.niit:
        est.notes.append("NIIT estimate assumes no other investment income this year.")
    return est


@dataclass
class WaitComparison:
    now: TaxEstimate
    later: TaxEstimate
    later_date: date

    @property
    def savings(self) -> float:
        return self.now.total - self.later.total


def compare_waiting(profile: TaxProfile, shares: float, cost: float, purchase: date, price: float,
                    today: date, planned: date | None = None) -> WaitComparison | None:  # fmt: skip
    """Tax if sold today vs at the planned date (or the long-term date if sooner
    and still ahead), at TODAY's price. Prices will move; this isolates the tax effect."""
    lt = long_term_date(purchase)
    later = planned or lt
    if later <= today:
        return None
    return WaitComparison(
        estimate_sale_tax(profile, shares, cost, purchase, price, today),
        estimate_sale_tax(profile, shares, cost, purchase, price, later),
        later,
    )


def wash_sale_warning(sale_date: date, purchase_dates: list[date]) -> str | None:
    """Selling at a loss within 30 days of buying the same stock can disallow the loss."""
    near = [d for d in purchase_dates if abs((d - sale_date).days) <= 30]
    if near:
        return ("Wash-sale rule: you bought this stock within 30 days of the sale date. A loss on this "
                "sale may be disallowed (it's added to the new shares' cost basis instead).")  # fmt: skip
    return None
