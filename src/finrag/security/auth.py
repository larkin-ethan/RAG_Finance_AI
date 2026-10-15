"""Login for the private local deployment (single user).

  * Passwords: argon2id (memory-hard; the OWASP-recommended choice). Only the
    hash is stored (Kubernetes Secret APP_PASSWORD_HASH), never the password.
  * Throttling: after 5 failures, logins lock for 60s, doubling each further
    failure (capped at 1 hour). Global, not per session, so opening new tabs
    doesn't reset it.
  * Idle timeout: a session expires after N minutes without activity.

Generate a hash:  python -m finrag.security.auth   (prompts; nothing is echoed)
"""

from __future__ import annotations

import getpass
import threading
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()  # argon2id with library defaults (RFC 9106 recommended params)
MIN_PASSWORD_LENGTH = 12


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Use at least {MIN_PASSWORD_LENGTH} characters")
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    if not stored_hash:
        return False  # no password configured: nobody gets in
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


class LoginThrottle:
    def __init__(self, free_attempts: int = 5, base_lock_s: float = 60, max_lock_s: float = 3600,
                 clock=time.monotonic):  # fmt: skip
        self.free_attempts, self.base_lock_s, self.max_lock_s = (
            free_attempts,
            base_lock_s,
            max_lock_s,
        )
        self._clock, self._failures, self._locked_until = clock, 0, 0.0
        self._lock = threading.Lock()

    def seconds_locked(self) -> float:
        with self._lock:
            return max(0.0, self._locked_until - self._clock())

    def record(self, success: bool) -> None:
        with self._lock:
            if success:
                self._failures, self._locked_until = 0, 0.0
                return
            self._failures += 1
            extra = self._failures - self.free_attempts
            if extra >= 0:
                lock = min(self.base_lock_s * 2**extra, self.max_lock_s)
                self._locked_until = self._clock() + lock


class IdleSession:
    """Tracks activity; expired() once idle longer than the timeout."""

    def __init__(self, idle_minutes: float, clock=time.monotonic):
        self.idle_s, self._clock = idle_minutes * 60, clock
        self.last_seen = clock()

    def touch(self) -> None:
        self.last_seen = self._clock()

    def expired(self) -> bool:
        return self._clock() - self.last_seen > self.idle_s


def main() -> None:
    first = getpass.getpass("New app password (min 12 chars): ")
    if first != getpass.getpass("Repeat: "):
        raise SystemExit("Passwords don't match")
    print(hash_password(first))


if __name__ == "__main__":
    main()
