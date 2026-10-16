from datetime import date

import pytest
from fastapi.testclient import TestClient

from finrag.api.main import app
from finrag.api.services import Services
from finrag.ratings.check import CheckResult
from finrag.ratings.goals import GoalResult
from finrag.settings import Settings

CSV = "Symbol,Quantity,Cost,Acquired\nAAPL,10,150,2021-03-15\nKO,5,55,2022-01-10\n"
MAPPING = {
    "ticker": "Symbol",
    "shares": "Quantity",
    "cost_per_share": "Cost",
    "purchase_date": "Acquired",
}


class FakeRepo:
    def __init__(self):
        self.lots, self.profile, self.next_id = {}, None, 1

    def add(self, lot):
        pid, self.next_id = self.next_id, self.next_id + 1
        self.lots[pid] = lot
        return pid

    def add_many(self, lots):
        return sum(1 for lot in lots if self.add(lot))

    def list(self):
        from dataclasses import replace

        return [replace(lot, id=pid) for pid, lot in self.lots.items()]

    def update(self, pid, lot):
        self.lots[pid] = lot

    def delete(self, pid):
        self.lots.pop(pid, None)

    def get_tax_profile(self):
        return self.profile

    def set_tax_profile(self, p):
        self.profile = p


class FakeChecker:
    def __init__(self):
        self.calls = []

    def check(self, ticker, goals, end, lots, tax):
        self.calls.append((ticker, goals, end, len(lots), tax))
        return CheckResult(ticker.upper(), date(2026, 10, 16), date(2026, 6, 30), 100.0,
                           [GoalResult(g, g, 60.0, "Moderate fit") for g in goals], {}, None, None, [], None)  # fmt: skip


class FakeServices(Services):
    def __init__(self, **settings):
        super().__init__(Settings(sec_user_agent="t t@example.com", **settings))
        self.repo, self.fake_checker, self.mappings = FakeRepo(), FakeChecker(), {}

    @property
    def positions(self):
        return self.repo

    @property
    def checker(self):
        return self.fake_checker

    def save_csv_mapping(self, name, mapping):
        self.mappings[name] = mapping

    def csv_mappings(self):
        return self.mappings


@pytest.fixture
def client():
    def make(**settings):
        app.state.services = FakeServices(**settings)
        return TestClient(app), app.state.services

    yield make
    del app.state.services


@pytest.mark.parametrize("settings", [{}, {"ratings_enabled": True, "public_demo": True}])
def test_routes_hidden_unless_enabled_and_private(client, settings):
    c, _ = client(**settings)
    for path in ("/check/AAPL", "/positions", "/tax-profile", "/portfolio/report", "/csv-mappings"):
        assert c.get(path).status_code == 404, path
    assert (
        c.post("/positions/import", json={"csv_text": CSV, "mapping": MAPPING}).status_code == 404
    )
    assert c.get("/config").json()["ratings_enabled"] is False


def test_lot_crud_and_validation(client):
    c, _svc = client(ratings_enabled=True)
    lot = {"ticker": "aapl", "shares": 10, "cost_per_share": 150, "purchase_date": "2021-03-15"}
    pid = c.post("/positions", json=lot).json()["id"]
    assert c.get("/positions").json()[0]["ticker"] == "AAPL"
    c.put(f"/positions/{pid}", json=lot | {"shares": 12})
    assert c.get("/positions").json()[0]["shares"] == 12
    assert c.post("/positions", json=lot | {"purchase_date": "2099-01-01"}).status_code == 422
    assert c.post("/positions", json=lot | {"shares": -1}).status_code == 422
    c.delete(f"/positions/{pid}")
    assert c.get("/positions").json() == []


def test_csv_import_dry_run_then_all_or_nothing(client):
    c, svc = client(ratings_enabled=True)
    preview = c.post("/positions/import", json={"csv_text": CSV, "mapping": MAPPING}).json()
    assert preview["imported"] == 0 and len(preview["preview"]) == 2 and svc.repo.lots == {}
    bad = CSV + "BAD!,1,1,2021-01-01\n"
    r = c.post(
        "/positions/import", json={"csv_text": bad, "mapping": MAPPING, "dry_run": False}
    ).json()
    assert r["imported"] == 0 and r["errors"] and svc.repo.lots == {}  # nothing half-imported
    ok = c.post("/positions/import", json={"csv_text": CSV, "mapping": MAPPING, "dry_run": False,
                                           "save_mapping_as": "My broker"}).json()  # fmt: skip
    assert ok["imported"] == 2 and svc.mappings["My broker"] == MAPPING


def test_tax_profile_and_check_uses_lots_and_profile(client):
    c, svc = client(ratings_enabled=True)
    assert (
        c.put("/tax-profile", json={"filing_status": "single", "taxable_income": 90000}).status_code
        == 200
    )
    assert (
        c.put("/tax-profile", json={"filing_status": "pirate", "taxable_income": 1}).status_code
        == 422
    )
    c.post(
        "/positions",
        json={"ticker": "KO", "shares": 5, "cost_per_share": 55, "purchase_date": "2022-01-10"},
    )
    body = c.get("/check/ko", params={"goals": ["dividend"], "horizon_days": 365}).json()
    assert body["ticker"] == "KO" and body["goals"][0]["grade"] == "Moderate fit"
    _ticker, goals, end, n_lots, tax = svc.fake_checker.calls[-1]
    assert (
        goals == ["dividend"] and end is not None and n_lots == 1 and tax.filing_status == "single"
    )
    report = c.get("/portfolio/report").json()
    assert [r["ticker"] for r in report] == ["KO"]
