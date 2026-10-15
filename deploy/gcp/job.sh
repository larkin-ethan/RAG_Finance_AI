#!/usr/bin/env bash
# Daily new-filing check: a Cloud Run Job (same image) triggered by Cloud Scheduler.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
IMAGE="$(cat .image 2>/dev/null)" || { echo "Run deploy.sh first" >&2; exit 1; }
CONN="$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')"

gcloud run jobs deploy "$JOB" --image "$IMAGE" --region "$REGION" \
  --service-account "$SA" --set-cloudsql-instances "$CONN" \
  --set-secrets "DATABASE_URL=finrag-database-url:latest,SEC_USER_AGENT=finrag-sec-user-agent:latest" \
  --set-env-vars "WATCHLIST_PATH=config/demo.yaml,EMBED_PROVIDER=local" \
  --command uv --args "run,--no-sync,python,-m,finrag.jobs.update_filings" \
  --memory 4Gi --cpu 2 --task-timeout 3600 --max-retries 1

# Let the service account trigger the job, then schedule it.
gcloud run jobs add-iam-policy-binding "$JOB" --region "$REGION" \
  --member "serviceAccount:$SA" --role roles/run.invoker >/dev/null
URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB}:run"
if gcloud scheduler jobs describe "${JOB}-daily" --location "$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "${JOB}-daily" --location "$REGION" --schedule "$SCHEDULE"
else
  gcloud scheduler jobs create http "${JOB}-daily" --location "$REGION" \
    --schedule "$SCHEDULE" --time-zone "Etc/UTC" --uri "$URI" --http-method POST \
    --oauth-service-account-email "$SA"
fi
echo "Scheduled. Run now with: gcloud run jobs execute $JOB --region $REGION"
