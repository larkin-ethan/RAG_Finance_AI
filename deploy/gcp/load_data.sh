#!/usr/bin/env bash
# Load the demo corpus into Cloud SQL from your machine, through the Cloud SQL
# Auth Proxy. Embeddings come from your local cache (data/cache/embeddings.sqlite),
# so this computes nothing new. It only uploads.
#
# Needs: cloud-sql-proxy installed
#   https://cloud.google.com/sql/docs/postgres/sql-proxy#install
# and the chunk files in data/chunks (make chunk).
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
CONN="$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')"
PORT=5433

cloud-sql-proxy "$CONN" --port "$PORT" &
PROXY=$!
trap 'kill $PROXY' EXIT
sleep 5

# Same credentials as the Cloud Run secret, but via the local proxy port.
CLOUD_URL="$(gcloud secrets versions access latest --secret finrag-database-url)"
export DATABASE_URL="$(printf '%s' "$CLOUD_URL" | sed -E "s#@/([^?]+)\?host=.*#@127.0.0.1:${PORT}/\1#")"

cd ../..
uv run python -m finrag.store.load --strategy section-512
echo "Loaded. Verify iterative scans are available (pgvector >= 0.8):"
uv run python -c "from finrag.store.db import connect; import os; c = connect(os.environ['DATABASE_URL']); print(c.execute(\"SELECT extversion FROM pg_extension WHERE extname='vector'\").fetchone())"
