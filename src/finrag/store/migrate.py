"""Apply the schema as the owner role, then grant the app role data access only.

  MIGRATE_DATABASE_URL=postgresql://finrag_owner:...@postgres/finrag \
  APP_DB_ROLE=finrag_app python -m finrag.store.migrate

Least privilege: the app role can read and write rows but cannot create, alter,
or drop tables. A compromised API process can't change the schema or the grants.
"""

from __future__ import annotations

import os

import psycopg
from psycopg import sql

from finrag.store.db import SCHEMA


def grants(app_role: str) -> list[sql.Composed]:
    role = sql.Identifier(app_role)
    return [
        sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role),
        sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(
            role
        ),
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(role),
        sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(role),
    ]


def main() -> None:
    url = os.environ["MIGRATE_DATABASE_URL"]
    app_role = os.environ.get("APP_DB_ROLE", "finrag_app")
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
        for stmt in grants(app_role):
            conn.execute(stmt)
    print(f"Schema applied; {app_role} granted DML only.")


if __name__ == "__main__":
    main()
