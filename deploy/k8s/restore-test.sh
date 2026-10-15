#!/usr/bin/env bash
# Prove backups are restorable: run a fresh backup, decrypt it with your age key
# (on this machine; the key never enters the cluster), restore into a throwaway
# database, and compare row counts. Decrypted data only ever flows through a pipe.
set -euo pipefail
NS=finrag
KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"
JOB="backup-test-$(date +%s)"

kubectl -n $NS create job "$JOB" --from=cronjob/backup >/dev/null
kubectl -n $NS wait --for=condition=complete "job/$JOB" --timeout=600s >/dev/null
FILE="$(kubectl -n $NS logs "job/$JOB" -c encrypt | sed -n 's/^backup written: //p')"
echo "Fresh backup: $FILE"

# Read the encrypted file through a short-lived pod that mounts the backups volume.
READER="backup-reader-$RANDOM"
kubectl -n $NS run "$READER" --restart=Never --image=finrag-local:"$(git rev-parse --short HEAD)" \
  --overrides='{"spec":{"automountServiceAccountToken":false,"securityContext":{"runAsNonRoot":true,"runAsUser":10001,"fsGroup":10001,"seccompProfile":{"type":"RuntimeDefault"}},
  "containers":[{"name":"r","image":"finrag-local:'"$(git rev-parse --short HEAD)"'","imagePullPolicy":"Never","command":["sleep","300"],
  "securityContext":{"allowPrivilegeEscalation":false,"readOnlyRootFilesystem":true,"capabilities":{"drop":["ALL"]}},
  "volumeMounts":[{"name":"b","mountPath":"/backups","readOnly":true}]}],
  "volumes":[{"name":"b","persistentVolumeClaim":{"claimName":"backups"}}]}}' >/dev/null
kubectl -n $NS wait --for=condition=ready "pod/$READER" --timeout=120s >/dev/null
trap 'kubectl -n $NS delete pod "$READER" --now >/dev/null 2>&1; kubectl -n $NS exec postgres-0 -- psql -U postgres -qc "DROP DATABASE IF EXISTS finrag_restore_test" >/dev/null 2>&1' EXIT

kubectl -n $NS exec postgres-0 -- psql -U postgres -qc "CREATE DATABASE finrag_restore_test"
kubectl -n $NS exec "$READER" -- cat "$FILE" \
  | age --decrypt --identity "$KEY_FILE" \
  | kubectl -n $NS exec -i postgres-0 -- pg_restore -U postgres --no-owner --dbname finrag_restore_test

count() { kubectl -n $NS exec postgres-0 -- psql -U postgres -d "$1" -tAc "select (select count(*) from filings)||'/'||(select count(*) from chunks)"; }
LIVE="$(count finrag)"; RESTORED="$(count finrag_restore_test)"
echo "filings/chunks live=$LIVE restored=$RESTORED"
[[ "$LIVE" == "$RESTORED" ]]
