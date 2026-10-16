"""Stock-check and portfolio endpoints. Private deployment only: every route
returns 404 unless RATINGS_ENABLED=true and PUBLIC_DEMO is off, so the public
demo can never expose them (even by misconfiguration of a single route)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from finrag.api.schemas import CsvImportIn, LotIn, TaxProfileIn
from finrag.ratings.positions import parse_csv, validate_lot
from finrag.ratings.tax import TaxProfile


def _services(request: Request):
    from finrag.api.main import services

    return services(request)


def require_ratings(request: Request) -> None:
    s = _services(request).settings
    if not s.ratings_enabled or s.public_demo:
        raise HTTPException(404, "Not found")


router = APIRouter(dependencies=[Depends(require_ratings)])
ALL_GOALS = ["dividend", "growth", "value", "stability"]


def _tax_profile(svc) -> TaxProfile | None:
    p = svc.positions.get_tax_profile()
    return TaxProfile(**p) if p else None


@router.get("/check/{ticker}")
def check(
    ticker: str,
    request: Request,
    goals: Annotated[list[str], Query()] = ALL_GOALS,
    horizon_days: Annotated[int | None, Query(ge=1, le=3650)] = None,
    sell_date: date | None = None,
) -> dict:
    from finrag.ratings.check import to_jsonable

    svc = _services(request)
    end = sell_date or (date.today() + timedelta(days=horizon_days) if horizon_days else None)
    try:
        result = svc.checker.check(ticker, goals, end, svc.positions.list(), _tax_profile(svc))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return to_jsonable(result)


@router.get("/positions")
def list_positions(request: Request) -> list[dict]:
    return [
        lot.__dict__ | {"purchase_date": lot.purchase_date.isoformat()}
        for lot in _services(request).positions.list()
    ]


@router.post("/positions")
def add_position(body: LotIn, request: Request) -> dict:
    try:
        lot = validate_lot(
            body.ticker, body.shares, body.cost_per_share, body.purchase_date, body.account
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"id": _services(request).positions.add(lot)}


@router.put("/positions/{pid}")
def update_position(pid: int, body: LotIn, request: Request) -> dict:
    try:
        lot = validate_lot(
            body.ticker, body.shares, body.cost_per_share, body.purchase_date, body.account
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _services(request).positions.update(pid, lot)
    return {"id": pid}


@router.delete("/positions/{pid}")
def delete_position(pid: int, request: Request) -> dict:
    _services(request).positions.delete(pid)
    return {"deleted": pid}


@router.post("/positions/import")
def import_positions(body: CsvImportIn, request: Request) -> dict:
    """dry_run=true: validate and preview only. dry_run=false: import, but only if
    every row is valid (all-or-nothing, so a half-imported file can't happen)."""
    svc = _services(request)
    result = parse_csv(body.csv_text.encode("utf-8"), body.mapping)
    preview = [
        lot.__dict__ | {"purchase_date": lot.purchase_date.isoformat()} for lot in result.lots
    ]
    if body.dry_run or result.errors:
        return {"imported": 0, "preview": preview, "errors": result.errors}
    n = svc.positions.add_many(result.lots)
    if body.save_mapping_as:
        svc.save_csv_mapping(body.save_mapping_as, body.mapping)
    return {"imported": n, "preview": preview, "errors": []}


@router.get("/csv-mappings")
def csv_mappings(request: Request) -> dict:
    return _services(request).csv_mappings()


@router.get("/tax-profile")
def get_tax_profile(request: Request) -> dict:
    return _services(request).positions.get_tax_profile() or {}


@router.put("/tax-profile")
def put_tax_profile(body: TaxProfileIn, request: Request) -> dict:
    _services(request).positions.set_tax_profile(json.loads(body.model_dump_json()))
    return {"saved": True}


@router.get("/portfolio/report")
def portfolio_report(request: Request) -> list[dict]:
    """One summary per held ticker: goal grades plus each lot's gain and tax view."""
    from finrag.ratings.check import to_jsonable

    svc = _services(request)
    lots, tax = svc.positions.list(), _tax_profile(svc)
    out = []
    for ticker in sorted({lot.ticker for lot in lots}):
        try:
            r = to_jsonable(
                svc.checker.check(
                    ticker, ["dividend", "growth", "value", "stability"], None, lots, tax
                )
            )
            out.append(
                {
                    "ticker": ticker,
                    "goals": r["goals"],
                    "lots": r["lots"],
                    "price": r["price"],
                    "notes": r["notes"],
                }
            )
        except (ValueError, RuntimeError) as exc:
            out.append({"ticker": ticker, "error": str(exc)})
    return out
