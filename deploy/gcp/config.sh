# Shared settings for the deploy scripts. Edit PROJECT_ID before running anything.
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
REPO="finrag"                       # Artifact Registry repository
SQL_INSTANCE="finrag-db"
DB_NAME="finrag"
DB_USER="finrag"
SA_NAME="finrag-run"                # service account the services/job run as
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
API_SERVICE="finrag-api"
UI_SERVICE="finrag-ui"
JOB="finrag-update"
SCHEDULE="0 12 * * *"               # daily, 12:00 UTC (after most filings land)

if [[ "$PROJECT_ID" == "your-gcp-project-id" ]]; then
  echo "Set PROJECT_ID (export PROJECT_ID=... or edit deploy/gcp/config.sh)" >&2
  exit 1
fi
