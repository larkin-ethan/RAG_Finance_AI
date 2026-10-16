"""Run a full stock check for one ticker: the thing the API and UI call.

  fundamentals (point-in-time XBRL) + prices + 5-year history -> metrics
  -> goal grades (+ risk-factor changes for stability) -> timeline report
  -> your lots' gain/loss and tax view -> verified explanation

Private deployment only (settings.ratings_enabled). Your lots and tax profile
are used here for arithmetic, but never sent to the LLM.

  python -m finrag.ratings.check AAPL --goals dividend value --horizon-days 730
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.xbrl import load_company_facts
from finrag.ratings.explain import Explanation, explain
from finrag.ratings.fundamentals import FactStore
from finrag.ratings.goals import GOAL_LABELS, GoalResult, metric_history, score_goal
from finrag.ratings.horizon import HorizonReport, horizon_report
from finrag.ratings.macro import FredClient, TreasuryYield
from finrag.ratings.metrics import Metrics, compute_metrics
from finrag.ratings.positions import Lot
from finrag.ratings.prices import TiingoClient
from finrag.ratings.risk_diff import RiskDiff, diff_risk_factors
from finrag.ratings.tax import TaxProfile, compare_waiting, estimate_sale_tax, wash_sale_warning

DISCLAIMER = ("Research tool, not investment or tax advice. Grades describe how well a stock's "
              "reported data fits a goal; they don't predict returns.")  # fmt: skip
FACTS_MAX_AGE_S = 20 * 3600


@dataclass
class LotView:
    lot: Lot
    market_value: float | None
    gain: float | None
    days_held: int
    long_term_date: date
    tax_now: float | None = None
    tax_later: float | None = None
    later_date: date | None = None
    savings_by_waiting: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    ticker: str
    as_of: date
    period_end: date | None
    price: float | None
    goals: list[GoalResult]
    metrics: Metrics
    risk: RiskDiff | None
    horizon: HorizonReport | None
    lots: list[LotView]
    explanation: Explanation | None
    notes: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def latest_two_10k_sections(parsed_dir: Path, ticker: str, as_of: date) -> tuple[dict, dict] | None:
    """{section_key: text} for the latest and previous 10-K filed on or before as_of."""
    filings = []
    for f in (parsed_dir / ticker).glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["form"] == "10-K" and date.fromisoformat(d["filing_date"]) <= as_of:
            filings.append(d)
    filings.sort(key=lambda d: d["filing_date"])
    if len(filings) < 2:
        return None

    def sections(d):
        return {s["key"]: "\n\n".join(b["text"] for b in s["blocks"]) for s in d["sections"]}

    return sections(filings[-1]), sections(filings[-2])


class StockChecker:
    def __init__(self, settings, edgar: EdgarClient | None = None, tiingo: TiingoClient | None = None,
                 fred: FredClient | None = None, embed: Callable[[list[str]], list[list[float]]] | None = None,
                 llm=None, ic_by_horizon: dict[int, float] | None = None):  # fmt: skip
        self.s = settings
        data = settings.data_dir
        self.edgar = edgar or EdgarClient(settings.sec_user_agent)
        self.tiingo = tiingo or TiingoClient(settings.tiingo_api_key, data / "prices")
        self.fred = fred or FredClient(settings.fred_api_key, data / "macro")
        self.embed, self.llm = embed, llm
        # Measured by the backtest (out-of-sample split); absent until it has run.
        ic_file = data / "cache" / "ic_by_horizon.json"
        if ic_by_horizon is None and ic_file.exists():
            ic_by_horizon = {
                int(k): v for k, v in json.loads(ic_file.read_text(encoding="utf-8")).items()
            }
        self.ic = ic_by_horizon
        self._treasury: TreasuryYield | None = None

    def facts(self, ticker: str) -> FactStore:
        from finrag.settings import load_watchlist

        extra = []
        try:
            extra = load_watchlist(self.s.watchlist_path).extra_ciks.get(ticker, [])
        except (OSError, ValueError):
            pass
        ciks = [self.edgar.get_cik(ticker), *(f"{int(c):010d}" for c in extra)]
        return FactStore(*(load_company_facts(self.edgar, c, self.s.data_dir / "xbrl", FACTS_MAX_AGE_S)
                           for c in ciks))  # fmt: skip

    def treasury(self) -> TreasuryYield:
        if self._treasury is None:
            self._treasury = self.fred.treasury_10y()
        return self._treasury

    def check(self, ticker: str, goals: list[str], horizon_end: date | None = None,
              lots: list[Lot] | None = None, tax: TaxProfile | None = None,
              as_of: date | None = None) -> CheckResult:  # fmt: skip
        ticker, as_of = ticker.upper(), as_of or date.today()
        unknown = [g for g in goals if g not in GOAL_LABELS]
        if unknown:
            raise ValueError(f"unknown goals: {unknown}")
        notes: list[str] = []
        store = self.facts(ticker)
        prices, market = self.tiingo.series(ticker), self.tiingo.series("SPY")
        ty = self.treasury()
        snap = store.snapshot(as_of)
        if snap.period_end is None:
            raise ValueError(f"No financial data found for {ticker}")
        metrics = compute_metrics(snap, prices, market, ty.on(as_of))
        history = metric_history(store, prices, market, ty, as_of)
        if snap.is_financial:
            notes.append(
                "Bank/financial company: industrial metrics (EV/EBIT, free cash flow, Altman Z) are skipped."
            )
        if metrics.get("years_without_cut") is not None:
            notes.append(
                "Years without a dividend cut counts only years in SEC XBRL data (since about 2009)."
            )

        risk = None
        sections = latest_two_10k_sections(self.s.data_dir / "parsed", ticker, as_of)
        if sections and self.embed:
            cur, prev = sections
            risk = diff_risk_factors(
                cur.get("1A", ""),
                prev.get("1A", ""),
                self.embed,
                cur.get("7", ""),
                prev.get("7", ""),
            )
            metrics["risk_change_share"] = risk.new_share
            metrics["red_flags"] = risk.red_flags
        else:
            notes.append("Risk-factor changes not included: load this company's filings first.")

        need = list(dict.fromkeys([*goals, "stability"]))
        scored = {g: score_goal(g, metrics, history) for g in need}
        results = [scored[g] for g in goals]

        report = None
        if horizon_end:
            main_goal = next((scored[g].score for g in goals if scored[g].score is not None), None)
            report = horizon_report(
                prices, market, as_of, horizon_end, scored["stability"].score, main_goal, self.ic
            )

        views = [self._lot_view(lot, prices.price(as_of), as_of, horizon_end, tax, lots or [])
                 for lot in lots or [] if lot.ticker == ticker]  # fmt: skip
        expl = explain(ticker, results, risk.new_paragraphs if risk else [], self.llm)
        return CheckResult(ticker, as_of, snap.period_end, metrics.get("price"), results, metrics, risk,
                           report, views, expl, notes)  # fmt: skip

    @staticmethod
    def _lot_view(lot: Lot, price: float | None, today: date, planned: date | None,
                  tax: TaxProfile | None, all_lots: list[Lot]) -> LotView:  # fmt: skip
        from finrag.ratings.tax import long_term_date

        v = LotView(lot, price * lot.shares if price else None,
                    (price - lot.cost_per_share) * lot.shares if price else None,
                    (today - lot.purchase_date).days, long_term_date(lot.purchase_date))  # fmt: skip
        if price and tax:
            now = estimate_sale_tax(
                tax, lot.shares, lot.cost_per_share, lot.purchase_date, price, today
            )
            v.tax_now = round(now.total, 2)
            v.notes += now.notes
            if cmp := compare_waiting(
                tax, lot.shares, lot.cost_per_share, lot.purchase_date, price, today, planned
            ):
                v.tax_later, v.later_date = round(cmp.later.total, 2), cmp.later_date
                v.savings_by_waiting = round(cmp.savings, 2)
        others = [x.purchase_date for x in all_lots if x.ticker == lot.ticker and x != lot]
        if v.gain is not None and v.gain < 0 and (w := wash_sale_warning(today, others)):
            v.notes.append(w)
        return v


def to_jsonable(result: CheckResult) -> dict:
    def conv(o):
        if isinstance(o, date):
            return o.isoformat()
        raise TypeError(type(o))

    return json.loads(json.dumps(asdict(result), default=conv))


def main() -> None:
    from finrag.settings import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--goals", nargs="+", default=["dividend", "growth", "value", "stability"])
    parser.add_argument("--horizon-days", type=int)
    args = parser.parse_args()
    checker = StockChecker(get_settings())
    end = date.today() + timedelta(days=args.horizon_days) if args.horizon_days else None
    r = checker.check(args.ticker, args.goals, end)
    print(f"{r.ticker} as of {r.as_of} (financials through {r.period_end}, price {r.price})")
    for g in r.goals:
        print(f"  {g.label:22} {g.grade}" + (f" ({g.score})" if g.score is not None else "") +
              (f"  capped: {'; '.join(g.caps)}" if g.caps else ""))  # fmt: skip
    if r.horizon:
        h = r.horizon
        print(
            f"  Horizon {h.days} days: {h.fit_grade}"
            + (f", chance of loss {h.stock.chance_of_loss:.0%}" if h.stock else "")
        )
    print(f"\n{r.explanation.text}\n\n{r.disclaimer}")


if __name__ == "__main__":
    main()
