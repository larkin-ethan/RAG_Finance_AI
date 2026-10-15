import pytest
from fastapi.testclient import TestClient

from finrag.security.auth import IdleSession, LoginThrottle, hash_password, verify_password
from finrag.security.crypto import FieldCipher


def test_argon2id_hash_and_verify():
    h = hash_password("correct horse battery")
    assert h.startswith("$argon2id$")
    assert verify_password(h, "correct horse battery")
    assert not verify_password(h, "wrong password!!")
    assert not verify_password("", "anything")  # unconfigured = locked
    assert not verify_password("not-a-hash", "x")


def test_short_password_rejected():
    with pytest.raises(ValueError):
        hash_password("short")


def test_throttle_escalates_and_resets():
    now = [0.0]
    t = LoginThrottle(free_attempts=2, base_lock_s=60, clock=lambda: now[0])
    t.record(False)
    assert t.seconds_locked() == 0
    t.record(False)
    assert t.seconds_locked() == 60
    t.record(False)
    assert t.seconds_locked() == 120  # doubles
    t.record(True)
    assert t.seconds_locked() == 0


def test_idle_session_expires():
    now = [0.0]
    s = IdleSession(idle_minutes=30, clock=lambda: now[0])
    now[0] = 29 * 60
    assert not s.expired()
    s.touch()
    now[0] += 31 * 60
    assert s.expired()


def test_field_encryption_roundtrip_rotation_and_tamper():
    from cryptography.fernet import Fernet

    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    token = FieldCipher(old).encrypt(123.45)
    assert "123.45" not in token
    both = FieldCipher(f"{new},{old}")
    assert both.decrypt(token) == "123.45"  # old data still readable after adding a key
    rotated = both.rotate(token)
    assert FieldCipher(new).decrypt(rotated) == "123.45"  # now under the new key only
    with pytest.raises(ValueError, match="tampered"):
        FieldCipher(new).decrypt(token[:-4] + "AAAA")
    with pytest.raises(ValueError):
        FieldCipher("")


def test_api_requires_service_token(monkeypatch):
    from finrag.api import main
    from finrag.settings import Settings

    monkeypatch.setattr(main, "get_settings", lambda: Settings(sec_user_agent="t t@e.com",
                                                               api_token="svc-token"))  # fmt: skip
    client = TestClient(main.app)
    assert client.get("/config").status_code == 401
    assert client.get("/config", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health").status_code == 200  # probes stay open
    assert client.get("/config", headers={"Authorization": "Bearer svc-token"}).status_code == 200
