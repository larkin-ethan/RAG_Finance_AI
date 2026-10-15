#!/usr/bin/env bash
# Delete everything the other scripts created. Cloud SQL bills hourly even
# when idle, so run this when you're done trying it out.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
read -r -p "Delete ALL FinRAG resources in project $PROJECT_ID (incl. the database)? [y/N] " ok
[[ "$ok" == "y" ]] || exit 0

gcloud scheduler jobs delete "${JOB}-daily" --location "$REGION" --quiet || true
gcloud run jobs delete "$JOB" --region "$REGION" --quiet || true
gcloud run services delete "$UI_SERVICE" --region "$REGION" --quiet || true
gcloud run services delete "$API_SERVICE" --region "$REGION" --quiet || true
gcloud sql instances delete "$SQL_INSTANCE" --quiet || true
for s in finrag-database-url finrag-gemini-key finrag-sec-user-agent finrag-admin-token; do
  gcloud secrets delete "$s" --quiet || true
done
gcloud artifacts repositories delete "$REPO" --location "$REGION" --quiet || true
gcloud iam service-accounts delete "$SA" --quiet || true
rm -f .image
echo "Torn down. Check Billing > Reports over the next day to confirm charges stop."
