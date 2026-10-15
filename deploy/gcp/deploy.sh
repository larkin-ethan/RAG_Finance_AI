#!/usr/bin/env bash
# Build the image with Cloud Build (no local Docker needed) and deploy:
#   finrag-api  FastAPI, public-demo mode (rate limits, ingest locked), 1 instance max
#   finrag-ui   Streamlit, talks to the API's URL
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
ROOT="$(cd ../.. && pwd)"
TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/finrag:${TAG}"
CONN="$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')"

gcloud builds submit "$ROOT" --tag "$IMAGE" --region "$REGION"
echo "$IMAGE" > .image  # job.sh reuses it

SECRETS="DATABASE_URL=finrag-database-url:latest,GEMINI_API_KEY=finrag-gemini-key:latest"
SECRETS+=",SEC_USER_AGENT=finrag-sec-user-agent:latest,ADMIN_TOKEN=finrag-admin-token:latest"

# max-instances=1 keeps the in-memory rate limits exact and the bill bounded.
# 4 GiB: the Qwen embedding model (fp32 on CPU) needs ~2.5 GB.
gcloud run deploy "$API_SERVICE" --image "$IMAGE" --region "$REGION" \
  --service-account "$SA" --add-cloudsql-instances "$CONN" \
  --set-secrets "$SECRETS" \
  --set-env-vars "PUBLIC_DEMO=true,TRUST_PROXY=true,WATCHLIST_PATH=config/demo.yaml,EMBED_PROVIDER=local" \
  --memory 4Gi --cpu 2 --min-instances 0 --max-instances 1 --timeout 300 \
  --allow-unauthenticated

API_URL="$(gcloud run services describe "$API_SERVICE" --region "$REGION" --format='value(status.url)')"

# Session affinity keeps a browser's Streamlit websocket on one instance.
gcloud run deploy "$UI_SERVICE" --image "$IMAGE" --region "$REGION" \
  --service-account "$SA" \
  --command uv --args "run,--no-sync,streamlit,run,ui/Home.py,--server.port,8080,--server.address,0.0.0.0" \
  --set-env-vars "FINRAG_API_URL=${API_URL}" \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 2 --session-affinity \
  --allow-unauthenticated

UI_URL="$(gcloud run services describe "$UI_SERVICE" --region "$REGION" --format='value(status.url)')"
echo "API: $API_URL/docs"
echo "UI:  $UI_URL   <- your live demo link"
