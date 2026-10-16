"""Your positions (tax lots): validation, broker CSV import, encrypted storage.

Storage: each lot is one encrypted JSON blob (`positions.lot_enc`), so the
database never holds a readable ticker, share count, cost, or date. Changes
are recorded in `audit_log` without values.

CSV import: brokers export different columns, so the user maps them once
(ticker / shares / cost / date [/ account]); mappings are saved by name.
Every row is validated before anything is stored; one bad row rejects nothing
silently: errors are returned per row.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime

from finrag.security.crypto import FieldCipher

MAX_CSV_BYTES = 1_000_000
MAX_ROWS = 2_000
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y")
REQUIRED_FIELDS = ("ticker", "shares", "cost_per_share", "purchase_date")


@dataclass(frozen=True)
class Lot:
    ticker: str
    shares: float
    cost_per_share: float
    purchase_date: date
    account: str = ""
    id: int | None = None

    def to_json(self) -> str:
        d = asdict(self) | {"purchase_date": self.purchase_date.isoformat()}
        d.pop("id")
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str, id: int | None = None) -> Lot:
        d = json.loads(s)
        return cls(d["ticker"], d["shares"], d["cost_per_share"], date.fromisoformat(d["purchase_date"]),
                   d.get("account", ""), id)  # fmt: skip


def _number(raw: str) -> float:
    cleaned = raw.strip().replace("$", "").replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    return float(cleaned)


def _date(raw: str) -> date:
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date '{raw}' (use YYYY-MM-DD or MM/DD/YYYY)")


def validate_lot(ticker: str, shares: float, cost: float, purchase: date, account: str = "",
                 today: date | None = None) -> Lot:  # fmt: skip
    ticker = ticker.strip().upper()
    if not TICKER_RE.match(ticker):
        raise ValueError(f"invalid ticker '{ticker}'")
    if not 0 < shares < 1e9:
        raise ValueError("shares must be positive")
    if not 0 <= cost < 1e7:
        raise ValueError("cost per share must be between 0 and 10,000,000")
    if purchase > (today or date.today()):
        raise ValueError("purchase date is in the future")
    if purchase < date(1950, 1, 1):
        raise ValueError("purchase date is before 1950")
    return Lot(ticker, float(shares), float(cost), purchase, account.strip()[:60])


@dataclass
class ImportResult:
    lots: list[Lot]
    errors: list[str]  # "row 7: invalid ticker 'XX!'"


def parse_csv(data: bytes, mapping: dict[str, str], today: date | None = None) -> ImportResult:
    """mapping: field -> CSV column header, e.g. {"ticker": "Symbol", "shares": "Quantity",
    "cost_per_share": "Cost Basis Per Share", "purchase_date": "Date Acquired"}.
    Optionally "account". Use "total_cost" instead of "cost_per_share" if the
    broker exports the lot's total cost."""
    if len(data) > MAX_CSV_BYTES:
        return ImportResult([], [f"file is larger than {MAX_CSV_BYTES // 1000} KB"])
    missing = [f for f in REQUIRED_FIELDS if f not in mapping
               and not (f == "cost_per_share" and "total_cost" in mapping)]  # fmt: skip
    if missing:
        return ImportResult([], [f"map these columns first: {', '.join(missing)}"])
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ImportResult([], ["file isn't UTF-8 text (export as CSV, not Excel)"])
    reader = csv.DictReader(io.StringIO(text))
    absent = [c for c in mapping.values() if c not in (reader.fieldnames or [])]
    if absent:
        return ImportResult([], [f"columns not found in file: {', '.join(absent)}"])

    lots, errors = [], []
    for n, row in enumerate(reader, start=2):  # row 1 is the header
        if n - 1 > MAX_ROWS:
            errors.append(f"stopped after {MAX_ROWS} rows")
            break
        if not any((v or "").strip() for v in row.values()):
            continue  # blank line
        try:
            shares = _number(row[mapping["shares"]])
            if "cost_per_share" in mapping:
                cost = _number(row[mapping["cost_per_share"]])
            else:
                cost = _number(row[mapping["total_cost"]]) / shares
            lots.append(validate_lot(row[mapping["ticker"]], shares, cost,
                                     _date(row[mapping["purchase_date"]]),
                                     row.get(mapping.get("account", ""), "") or "", today))  # fmt: skip
        except (ValueError, ZeroDivisionError, KeyError) as exc:
            errors.append(f"row {n}: {exc}")
    return ImportResult(lots, errors)


class PositionsRepo:
    """CRUD over encrypted lots. `conn` is a psycopg connection."""

    def __init__(self, conn, cipher: FieldCipher):
        self.conn, self.cipher = conn, cipher

    def _audit(self, action: str, position_id: int | None) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (action, position_id) VALUES (%s, %s)", (action, position_id)
        )

    def add(self, lot: Lot) -> int:
        pid = self.conn.execute(
            "INSERT INTO positions (lot_enc) VALUES (%s) RETURNING id",
            (self.cipher.encrypt(lot.to_json()),),
        ).fetchone()[0]
        self._audit("add", pid)
        return pid

    def add_many(self, lots: list[Lot]) -> int:
        with self.conn.transaction():
            for lot in lots:
                self.add(lot)
        return len(lots)

    def list(self) -> list[Lot]:
        rows = self.conn.execute("SELECT id, lot_enc FROM positions ORDER BY id").fetchall()
        return [Lot.from_json(self.cipher.decrypt(enc), pid) for pid, enc in rows]

    def update(self, pid: int, lot: Lot) -> None:
        self.conn.execute("UPDATE positions SET lot_enc = %s, updated_at = now() WHERE id = %s",
                          (self.cipher.encrypt(lot.to_json()), pid))  # fmt: skip
        self._audit("update", pid)

    def delete(self, pid: int) -> None:
        self.conn.execute("DELETE FROM positions WHERE id = %s", (pid,))
        self._audit("delete", pid)

    def get_tax_profile(self) -> dict | None:
        row = self.conn.execute("SELECT profile_enc FROM tax_profile WHERE id = 1").fetchone()
        return json.loads(self.cipher.decrypt(row[0])) if row else None

    def set_tax_profile(self, profile: dict) -> None:
        self.conn.execute(
            "INSERT INTO tax_profile (id, profile_enc) VALUES (1, %s) ON CONFLICT (id) "
            "DO UPDATE SET profile_enc = EXCLUDED.profile_enc, updated_at = now()",
            (self.cipher.encrypt(json.dumps(profile)),),
        )
        self._audit("tax_profile", None)
