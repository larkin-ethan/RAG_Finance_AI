#!/usr/bin/env bash
# One-time setup: APIs, Artifact Registry, Cloud SQL (Postgres 16 + pgvector),
# secrets, and a least-privilege service account.
#
# Needs: gcloud CLI logged in (gcloud auth login), a project on the free trial,
# and in your environment: GEMINI_API_KEY, SEC_USER_AGENT.
# Optional: BILLING_ACCOUNT (to create a $25 budget alert; strongly recommended).
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
: "${GEMINI_API_KEY:?export GEMINI_API_KEY first}"
: "${SEC_USER_AGENT:?export SEC_USER_AGENT first (\"Name email@example.com\")}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com sqladmin.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com \
  cloudscheduler.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

# Cloud SQL: smallest shared-core tier (~$8-10/mo, covered by trial credit).
# --edition=ENTERPRISE matters: new Postgres instances default to Enterprise Plus,
# which has no shared-core tiers. No HA, 10 GB SSD.
if ! gcloud sql instances describe "$SQL_INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances create "$SQL_INSTANCE" \
    --database-version=POSTGRES_16 --edition=ENTERPRISE --tier=db-f1-micro \
    --region="$REGION" --storage-type=SSD --storage-size=10 --availability-type=zonal
fi
gcloud sql databases describe "$DB_NAME" --instance "$SQL_INSTANCE" >/dev/null 2>&1 ||
  gcloud sql databases create "$DB_NAME" --instance "$SQL_INSTANCE"

DB_PASS="$(openssl rand -base64 30 | tr -dc 'A-Za-z0-9' | head -c 32)"
if gcloud sql users list --instance "$SQL_INSTANCE" --format='value(name)' | grep -qx "$DB_USER"; then
  gcloud sql users set-password "$DB_USER" --instance "$SQL_INSTANCE" --password "$DB_PASS"
else
  gcloud sql users create "$DB_USER" --instance "$SQL_INSTANCE" --password "$DB_PASS"
fi
CONN="$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')"

put_secret() {  # name, value (read from stdin so it never appears in process args)
  if gcloud secrets describe "$1" >/dev/null 2>&1; then
    printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=-
  else
    printf '%s' "$2" | gcloud secrets create "$1" --data-file=- --replication-policy=automatic
  fi
}
# Cloud Run connects to Cloud SQL over a Unix socket: no public DB exposure.
put_secret finrag-database-url "postgresql://${DB_USER}:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${CONN}"
put_secret finrag-gemini-key "$GEMINI_API_KEY"
put_secret finrag-sec-user-agent "$SEC_USER_AGENT"
put_secret finrag-admin-token "$(openssl rand -hex 24)"
unset DB_PASS

gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create "$SA_NAME" --display-name "FinRAG runtime"
for role in roles/cloudsql.client roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SA" --role "$role" --condition=None >/dev/null
done

if [[ -n "${BILLING_ACCOUNT:-}" ]]; then
  gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" \
    --display-name "finrag-budget" --budget-amount 25USD \
    --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0
else
  echo "!! No BILLING_ACCOUNT set: create a budget alert in the console (Billing > Budgets)."
fi
echo "Setup done. Cloud SQL connection: $CONN"
echo "Admin token: gcloud secrets versions access latest --secret finrag-admin-token"
