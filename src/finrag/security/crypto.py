"""Application-level field encryption for sensitive data (positions, tax profile).

Defense in depth on top of disk encryption: a database dump or a stray backup
doesn't reveal holdings without the key. Fernet = AES-128-CBC + HMAC-SHA256
with authenticated, timestamped tokens (`cryptography` library).

Keys come from FIELD_ENCRYPTION_KEYS (comma-separated, newest first). Encrypt
always uses the first key; decrypt tries all of them (MultiFernet), so a key
can be rotated: add a new key in front, run `rotate()` over stored values,
then drop the old key.

Generate a key:  python -m finrag.security.crypto
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken, MultiFernet


class FieldCipher:
    def __init__(self, keys: str):
        parts = [k.strip() for k in keys.split(",") if k.strip()]
        if not parts:
            raise ValueError(
                "FIELD_ENCRYPTION_KEYS is empty; generate one with python -m finrag.security.crypto"
            )
        self._f = MultiFernet([Fernet(k.encode()) for k in parts])

    def encrypt(self, value: str | float) -> str:
        return self._f.encrypt(str(value).encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._f.decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Can't decrypt: wrong key or tampered data") from exc

    def rotate(self, token: str) -> str:
        """Re-encrypt under the newest key."""
        return self._f.rotate(token.encode()).decode()


def main() -> None:
    print(Fernet.generate_key().decode())


if __name__ == "__main__":
    main()
