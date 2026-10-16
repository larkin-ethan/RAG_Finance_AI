"""Point-in-time fundamentals from SEC XBRL company facts.

`FactStore.snapshot(as_of)` answers: what did the filings say, as known on
date D? Only facts *filed* on or before D are used (no look-ahead bias),
and when a period is reported more than once (each 10-K repeats prior years,
sometimes restated) the latest version filed by D wins.

Trailing twelve months (TTM) for flow items like revenue:
  * latest period is a fiscal year  -> that year's value
  * latest period is year-to-date (Q1, H1, 9M from a 10-Q)
        -> YTD + prior fiscal year - prior-year YTD
     e.g. Apple Q3 FY2024: 296,105 + 383,285 - 293,787 = $385,603M
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

# metric -> (XBRL concepts in priority order, unit, instant?)
# Companies switch tags over time (e.g. Revenues -> RevenueFromContract... in 2018),
# so series are merged across concepts; a higher-priority concept wins per period.
RATING_METRICS: dict[str, tuple[tuple[str, ...], str, bool]] = {
    "revenue": (("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                 "RevenuesNetOfInterestExpense", "SalesRevenueNet"), "USD", False),
    "gross_profit": (("GrossProfit",), "USD", False),
    "operating_income": (("OperatingIncomeLoss",), "USD", False),
    # Pre-tax income: EBIT fallback for filers that don't tag operating income (XOM, JNJ).
    "pretax_income": (("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"), "USD", False),
    "net_income": (("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"), "USD", False),
    "eps_diluted": (("EarningsPerShareDiluted",), "USD/shares", False),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), "USD", False),
    "capex": (("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"), "USD", False),
    "rnd": (("ResearchAndDevelopmentExpense",
             "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"), "USD", False),
    "interest_expense": (("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"), "USD", False),
    "depreciation": (("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                      "DepreciationAmortizationAndAccretionNet"), "USD", False),
    "dividends_per_share": (("CommonStockDividendsPerShareDeclared",
                             "CommonStockDividendsPerShareCashPaid"), "USD/shares", False),
    "dividends_paid": (("PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfOrdinaryDividends",
                        "DividendsCommonStockCash"), "USD", False),
    # XOM has no dilutive securities and tags only the basic count.
    "diluted_shares": (("WeightedAverageNumberOfDilutedSharesOutstanding",
                        "WeightedAverageNumberOfSharesOutstandingBasic"), "shares", False),
    "total_assets": (("Assets",), "USD", True),
    "total_liabilities": (("Liabilities",), "USD", True),
    # JNJ tags equity only including noncontrolling interest.
    "equity": (("StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "USD", True),
    "equity_incl_nci": (("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                         "StockholdersEquity"), "USD", True),
    "current_assets": (("AssetsCurrent",), "USD", True),
    "current_liabilities": (("LiabilitiesCurrent",), "USD", True),
    "cash": (("CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"), "USD", True),
    # Debt is tagged very differently across companies (KO: one all-in total;
    # AAPL/MSFT/JNJ: LongTermDebt incl. current maturities; XOM: noncurrent + current).
    # total_debt is derived from these in snapshot(); see _total_debt().
    "debt_total": (("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
                    "DebtLongtermAndShorttermCombinedAmount", "LongTermDebt"), "USD", True),
    "debt_noncurrent": (("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"), "USD", True),
    "debt_current": (("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"), "USD", True),
    "short_borrowings": (("CommercialPaper", "ShortTermBorrowings"), "USD", True),
    "retained_earnings": (("RetainedEarningsAccumulatedDeficit",), "USD", True),
    "deposits": (("Deposits",), "USD", True),  # banks only: used to detect financials
}  # fmt: skip
SHARES_OUTSTANDING = ("dei", "EntityCommonStockSharesOutstanding", "shares")

ANNUAL_DAYS = (350, 380)
YTD_DAYS = (80, 290)  # Q1 (~91), H1 (~182), 9M (~273)
SLACK = timedelta(days=10)  # 52/53-week calendars shift period ends by a few days


@dataclass(frozen=True)
class Obs:
    start: date | None
    end: date
    value: float
    filed: date
    form: str
    accession: str
    priority: int  # index of the concept in RATING_METRICS (lower = preferred)
    concept: str = ""

    @property
    def days(self) -> int | None:
        return (self.end - self.start).days if self.start else None


@dataclass(frozen=True)
class TTM:
    value: float
    period_end: date
    method: str  # "annual" or "ytd"


@dataclass
class Snapshot:
    """Everything known about a company on `as_of`."""

    as_of: date
    period_end: date | None  # end of the latest reported period
    ttm: dict[str, float] = field(default_factory=dict)  # flow metrics, trailing 12 months
    ttm_year_ago: dict[str, float] = field(default_factory=dict)  # same, one year earlier
    latest: dict[str, float] = field(default_factory=dict)  # balance sheet, latest
    year_ago: dict[str, float] = field(default_factory=dict)  # balance sheet, one year earlier
    annual: dict[str, list[tuple[date, float]]] = field(
        default_factory=dict
    )  # fiscal years, oldest first
    shares_outstanding: float | None = None
    is_financial: bool = False


def _total_debt(v: dict[str, float]) -> float | None:
    """Long-term debt incl. current maturities, plus short-term borrowings."""
    if "debt_total" in v:
        long_term = v["debt_total"]
    elif "debt_noncurrent" in v or "debt_current" in v:
        long_term = v.get("debt_noncurrent", 0.0) + v.get("debt_current", 0.0)
    else:
        long_term = None
    short = v.get("short_borrowings")
    if long_term is None and short is None:
        return None
    return (long_term or 0.0) + (short or 0.0)


def _derive_balance_sheet(v: dict[str, float]) -> None:
    """Fill derived balance-sheet values in place (when the parts exist)."""
    if (debt := _total_debt(v)) is not None:
        v["total_debt"] = debt
    if "total_liabilities" not in v and "total_assets" in v and "equity_incl_nci" in v:
        v["total_liabilities"] = v["total_assets"] - v["equity_incl_nci"]  # e.g. KO tags no total


def _derive_ebit(v: dict[str, float]) -> None:
    """EBIT = operating income, else pre-tax income + interest expense."""
    if "operating_income" in v:
        v["ebit"] = v["operating_income"]
    elif "pretax_income" in v:
        v["ebit"] = v["pretax_income"] + v.get("interest_expense", 0.0)


def _parse_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


class FactStore:
    """Indexed XBRL facts for one company (merge predecessor CIKs by passing several)."""

    def __init__(self, *company_facts: dict):
        self._obs: dict[str, list[Obs]] = defaultdict(list)
        for facts in company_facts:
            gaap = facts.get("facts", {}).get("us-gaap", {})
            for metric, (concepts, unit, _) in RATING_METRICS.items():
                for priority, concept in enumerate(concepts):
                    for f in gaap.get(concept, {}).get("units", {}).get(unit, []):
                        if f.get("form") in ("10-K", "10-Q") and "filed" in f:
                            self._obs[metric].append(self._make(f, priority, concept))
            ns, concept, unit = SHARES_OUTSTANDING
            for f in (
                facts.get("facts", {}).get(ns, {}).get(concept, {}).get("units", {}).get(unit, [])
            ):
                if "filed" in f:
                    self._obs["shares_outstanding"].append(self._make(f, 0, concept))

    @staticmethod
    def _make(f: dict, priority: int, concept: str) -> Obs:
        return Obs(_parse_date(f.get("start")), date.fromisoformat(f["end"]), float(f["val"]),
                   date.fromisoformat(f["filed"]), f.get("form", ""), f.get("accn", ""), priority,
                   concept)  # fmt: skip

    def known(self, metric: str, as_of: date) -> list[Obs]:
        """One observation per period, as known on as_of: best concept, then latest filing."""
        best: dict[tuple, Obs] = {}
        for o in self._obs.get(metric, []):
            if o.filed > as_of:
                continue  # filed after as_of: invisible (no look-ahead)
            key = (o.start, o.end)
            cur = best.get(key)
            if cur is None or (o.priority, -o.filed.toordinal()) < (
                cur.priority,
                -cur.filed.toordinal(),
            ):
                best[key] = o
        result = sorted(best.values(), key=lambda o: (o.end, o.days or 0))
        assert all(o.filed <= as_of for o in result)  # point-in-time guard
        return result

    def ttm(self, metric: str, as_of: date, target_end: date | None = None) -> TTM | None:
        obs = [o for o in self.known(metric, as_of) if o.start]
        annual = [o for o in obs if ANNUAL_DAYS[0] <= o.days <= ANNUAL_DAYS[1]]
        ytd = [o for o in obs if YTD_DAYS[0] <= o.days <= YTD_DAYS[1]]
        ends = sorted({o.end for o in annual + ytd})
        if not ends:
            return None
        if target_end is None:
            end = ends[-1]
        else:
            near = [e for e in ends if abs(e - target_end) <= SLACK]
            if not near:
                return None
            end = near[0]

        for o in annual:
            if o.end == end:
                return TTM(o.value, end, "annual")
        current = [o for o in ytd if o.end == end]
        if not current:
            return None
        cur = max(
            current, key=lambda o: o.days
        )  # YTD from fiscal-year start, not the 3-month quarter
        prev_fy_end = cur.start - timedelta(days=1)
        prev_fy = next((o for o in annual if abs(o.end - prev_fy_end) <= SLACK), None)
        prior = next(
            (o for o in ytd
             if abs(o.end - (end - timedelta(days=364))) <= SLACK and abs(o.days - cur.days) <= 10),
            None,
        )  # fmt: skip
        if prev_fy is None or prior is None:
            return None
        return TTM(cur.value + prev_fy.value - prior.value, end, "ytd")

    def instant(self, metric: str, as_of: date, target_end: date | None = None) -> Obs | None:
        obs = [o for o in self.known(metric, as_of) if o.start is None]
        if target_end is not None:
            obs = [o for o in obs if abs(o.end - target_end) <= SLACK]
        return obs[-1] if obs else None

    def annual_series(self, metric: str, as_of: date) -> list[tuple[date, float]]:
        """Fiscal-year values (flow) or fiscal-year-end values (instant), oldest first."""
        instant = RATING_METRICS.get(metric, ((), "", False))[2]
        fy_ends = [o.end for o in self.known("revenue", as_of) + self.known("net_income", as_of)
                   if o.start and ANNUAL_DAYS[0] <= o.days <= ANNUAL_DAYS[1]]  # fmt: skip
        out: dict[date, float] = {}
        for o in self.known(metric, as_of):
            if (
                instant
                and o.start is None
                and any(abs(o.end - e) <= SLACK for e in fy_ends)
                or not instant
                and o.start
                and ANNUAL_DAYS[0] <= o.days <= ANNUAL_DAYS[1]
            ):
                out[o.end] = o.value
        return sorted(out.items())

    def _derived_annual(self, metric: str, as_of: date) -> list[tuple[date, float]]:
        parts = ("debt_total", "debt_noncurrent", "debt_current", "short_borrowings", "total_assets",
                 "equity_incl_nci", "total_liabilities")  # fmt: skip
        by_end: dict[date, dict[str, float]] = defaultdict(dict)
        for part in parts:
            for end, value in self.annual_series(part, as_of):
                by_end[end][part] = value
        out = []
        for end, values in sorted(by_end.items()):
            _derive_balance_sheet(values)
            if metric in values:
                out.append((end, values[metric]))
        return out

    def concepts_used(self, as_of: date) -> dict[str, str]:
        """metric -> XBRL concept that supplied its latest value (for audits)."""
        used = {}
        for metric in [*RATING_METRICS, "shares_outstanding"]:
            if obs := self.known(metric, as_of):
                used[metric] = obs[-1].concept
        return used

    def snapshot(self, as_of: date) -> Snapshot:
        rev = self.ttm("revenue", as_of) or self.ttm("net_income", as_of)
        snap = Snapshot(as_of=as_of, period_end=rev.period_end if rev else None)
        if snap.period_end is None:
            return snap
        year_ago = snap.period_end - timedelta(days=364)
        for metric, (_, _, instant) in RATING_METRICS.items():
            if instant:
                if o := self.instant(metric, as_of, snap.period_end):
                    snap.latest[metric] = o.value
                if o := self.instant(metric, as_of, year_ago):
                    snap.year_ago[metric] = o.value
            else:
                if t := self.ttm(metric, as_of, snap.period_end):
                    snap.ttm[metric] = t.value
                if t := self.ttm(metric, as_of, year_ago):
                    snap.ttm_year_ago[metric] = t.value
            snap.annual[metric] = self.annual_series(metric, as_of)
        if shares := [o for o in self.known("shares_outstanding", as_of)]:
            snap.shares_outstanding = shares[-1].value
        for bucket in (snap.latest, snap.year_ago):
            _derive_balance_sheet(bucket)
        for flows in (snap.ttm, snap.ttm_year_ago):
            _derive_ebit(flows)
        for metric in ("total_debt", "total_liabilities"):
            snap.annual[metric] = self._derived_annual(metric, as_of)
        # Banks report deposits and have no current-asset subtotal: score them separately.
        snap.is_financial = "deposits" in snap.latest and "current_assets" not in snap.latest
        return snap
