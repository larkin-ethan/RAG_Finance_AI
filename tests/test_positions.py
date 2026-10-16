from datetime import date

import pytest
from cryptography.fernet import Fernet

from finrag.ratings.positions import Lot, parse_csv, validate_lot

FIDELITY_LIKE = {"ticker": "Symbol", "shares": "Quantity", "cost_per_share": "Cost Basis Per Share",
                 "purchase_date": "Date Acquired", "account": "Account Name"}  # fmt: skip
TODAY = date(2026, 10, 16)


def test_valid_csv_with_broker_formats():
    data = (
        b"Account Name,Symbol,Quantity,Cost Basis Per Share,Date Acquired\n"
        b'Roth IRA,aapl,10,"$1,150.25",03/15/2021\n'
        b"Brokerage,KO,25.5,58.10,2023-07-01\n"
        b"\n"
    )
    r = parse_csv(data, FIDELITY_LIKE, today=TODAY)
    assert r.errors == []
    assert r.lots[0] == Lot("AAPL", 10.0, 1150.25, date(2021, 3, 15), "Roth IRA")
    assert r.lots[1].shares == 25.5


def test_bad_rows_reported_individually():
    data = (
        b"Account Name,Symbol,Quantity,Cost Basis Per Share,Date Acquired\n"
        b"A,AAPL,10,100,2021-01-01\n"
        b"A,BAD!,10,100,2021-01-01\n"
        b"A,MSFT,-5,100,2021-01-01\n"
        b"A,KO,1,50,2030-01-01\n"
        b"A,JNJ,1,50,not-a-date\n"
    )
    r = parse_csv(data, FIDELITY_LIKE, today=TODAY)
    assert [lot.ticker for lot in r.lots] == ["AAPL"]
    assert [e.split(":")[0] for e in r.errors] == ["row 3", "row 4", "row 5", "row 6"]
    assert "future" in r.errors[2]


def test_total_cost_column_divided_by_shares():
    mapping = FIDELITY_LIKE | {"total_cost": "Cost"}
    mapping.pop("cost_per_share")
    data = b"Account Name,Symbol,Quantity,Cost,Date Acquired\nA,NVDA,4,500,2024-01-02\n"
    assert parse_csv(data, mapping, today=TODAY).lots[0].cost_per_share == 125


def test_mapping_and_file_problems():
    assert "map these columns" in parse_csv(b"x\n1\n", {"ticker": "x"}).errors[0]
    assert "not found" in parse_csv(b"a,b\n", FIDELITY_LIKE).errors[0]
    assert "UTF-8" in parse_csv("Symbol\nÄ".encode("utf-16"), FIDELITY_LIKE).errors[0]
    assert "larger" in parse_csv(b"x" * 1_000_001, FIDELITY_LIKE).errors[0]


def test_validate_lot_rules():
    with pytest.raises(ValueError):
        validate_lot("AAPL", 0, 10, date(2020, 1, 1))
    with pytest.raises(ValueError):
        validate_lot("AAPL; DROP TABLE", 1, 10, date(2020, 1, 1))
    assert validate_lot(" brk.b ", 1, 10, date(2020, 1, 1)).ticker == "BRK.B"


def test_lot_json_roundtrip_and_ciphertext_hides_everything():
    from finrag.security.crypto import FieldCipher

    lot = Lot("AAPL", 10, 150.0, date(2021, 3, 15), "Roth IRA")
    cipher = FieldCipher(Fernet.generate_key().decode())
    token = cipher.encrypt(lot.to_json())
    for secret in ("AAPL", "150", "2021", "Roth"):
        assert secret not in token
    assert Lot.from_json(cipher.decrypt(token)) == lot
