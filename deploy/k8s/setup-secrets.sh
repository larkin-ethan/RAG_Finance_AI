#!/usr/bin/env bash
# One-time (or rotation) secret generation for the private k8s deployment.
# Run in WSL2. Produces overlays/local/secrets.enc.yaml, encrypted with SOPS+age.
# Plaintext only ever exists in a tmpfs scratch dir that's shredded on exit.
#
# Needs: age, age-keygen, sops, openssl, kubectl, uv; config/holdings.yaml present.
set -euo pipefail
umask 077
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

for tool in age age-keygen sops openssl kubectl uv; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
[[ -f "$ROOT/config/holdings.yaml" ]] || { echo "create config/holdings.yaml first (see holdings.example.yaml)" >&2; exit 1; }

OUT=overlays/local/secrets.enc.yaml
if [[ -f "$OUT" ]]; then
  read -r -p "$OUT exists. Regenerating rotates ALL passwords and keys (existing DB logins and encrypted fields break unless you migrate). Continue? [y/N] " ok
  [[ "$ok" == "y" ]] || exit 0
fi

# --- age key (decrypts secrets and backups; NEVER enters the cluster or git) ---
KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"
if [[ ! -f "$KEY_FILE" ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  age-keygen -o "$KEY_FILE" 2>/dev/null
  echo "!! Created $KEY_FILE. Back it up offline (e.g. password manager). Without it,"
  echo "!! your secrets and every backup are unrecoverable."
fi
AGE_PUB="$(age-keygen -y "$KEY_FILE")"
sed -i "s/AGE_PUBLIC_KEY/$AGE_PUB/" .sops.yaml

# --- scratch space in RAM ---
SCRATCH="$(mktemp -d "${XDG_RUNTIME_DIR:-/dev/shm}/finrag.XXXXXX")"
trap 'find "$SCRATCH" -type f -exec shred -u {} + 2>/dev/null; rm -rf "$SCRATCH"' EXIT

# --- local CA, name-constrained: even if its key leaked it could only sign
#     finrag.localhost and the in-cluster postgres names, never e.g. google.com ---
CA_DIR="$HOME/.finrag/ca"
mkdir -p "$CA_DIR"
if [[ ! -f "$CA_DIR/ca.key" ]]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 825 \
    -keyout "$CA_DIR/ca.key" -out "$CA_DIR/ca.crt" -subj "/CN=FinRAG Local CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -addext "nameConstraints=critical,permitted;DNS:finrag.localhost,permitted;DNS:postgres,permitted;DNS:postgres.finrag.svc,permitted;DNS:postgres.finrag.svc.cluster.local"
  echo "Created local CA at $CA_DIR. Trust it on Windows (current user only, no admin):"
  echo "  PowerShell: Import-Certificate -FilePath \\\\wsl\$\\<distro>$CA_DIR/ca.crt -CertStoreLocation Cert:\\CurrentUser\\Root"
fi

leaf() {  # name, SAN list -> $SCRATCH/name.{key,crt}
  openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "$SCRATCH/$1.key" -out "$SCRATCH/$1.csr" -subj "/CN=$1" 2>/dev/null
  openssl x509 -req -in "$SCRATCH/$1.csr" -CA "$CA_DIR/ca.crt" -CAkey "$CA_DIR/ca.key" \
    -CAcreateserial -days 397 -out "$SCRATCH/$1.crt" -extfile <(printf '%s\n' \
    "subjectAltName=$2" "extendedKeyUsage=serverAuth" "keyUsage=critical,digitalSignature" \
    "basicConstraints=critical,CA:FALSE") 2>/dev/null
}
leaf ui "DNS:finrag.localhost"
leaf postgres "DNS:postgres,DNS:postgres.finrag.svc,DNS:postgres.finrag.svc.cluster.local"

# --- generated secrets ---
rand() { openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | head -c "$1"; }
PG_SUPER="$(rand 40)"; PG_OWNER="$(rand 40)"; PG_APP="$(rand 40)"
API_TOKEN="$(openssl rand -hex 32)"
FERNET="$(cd "$ROOT" && uv run --no-sync python -m finrag.security.crypto)"
echo "Choose the app login password (min 12 chars):"
PW_HASH="$(cd "$ROOT" && uv run --no-sync python -m finrag.security.auth)"
read -r -p 'SEC User-Agent ("Your Name you@example.com"): ' SEC_UA
read -r -s -p "Tiingo API key (Enter to skip for now): " TIINGO; echo
read -r -s -p "FRED API key (Enter to skip for now): " FRED; echo

PGQ="sslmode=verify-full&sslrootcert=/tls-ca/ca.crt"
K="kubectl create --dry-run=client -o yaml -n finrag"
{
  $K secret generic finrag-secrets \
    --from-literal=POSTGRES_PASSWORD="$PG_SUPER" \
    --from-literal=FINRAG_OWNER_PASSWORD="$PG_OWNER" \
    --from-literal=FINRAG_APP_PASSWORD="$PG_APP" \
    --from-literal=DATABASE_URL="postgresql://finrag_app:${PG_APP}@postgres:5432/finrag?${PGQ}" \
    --from-literal=MIGRATE_DATABASE_URL="postgresql://finrag_owner:${PG_OWNER}@postgres:5432/finrag?${PGQ}" \
    --from-literal=API_TOKEN="$API_TOKEN" \
    --from-literal=APP_PASSWORD_HASH="$PW_HASH" \
    --from-literal=FIELD_ENCRYPTION_KEYS="$FERNET" \
    --from-literal=SEC_USER_AGENT="$SEC_UA" \
    --from-literal=TIINGO_API_KEY="$TIINGO" \
    --from-literal=FRED_API_KEY="$FRED" \
    --from-literal=AGE_RECIPIENT="$AGE_PUB"
  echo "---"
  $K secret tls finrag-tls --cert="$SCRATCH/ui.crt" --key="$SCRATCH/ui.key"
  echo "---"
  $K secret tls postgres-tls --cert="$SCRATCH/postgres.crt" --key="$SCRATCH/postgres.key"
  echo "---"
  $K secret generic postgres-ca --from-file=ca.crt="$CA_DIR/ca.crt"
  echo "---"
  $K secret generic finrag-holdings --from-file=holdings.yaml="$ROOT/config/holdings.yaml"
} > "$SCRATCH/secrets.enc.yaml"

sops --config .sops.yaml --encrypt "$SCRATCH/secrets.enc.yaml" > "$OUT"
grep -q "ENC\[" "$OUT" && echo "Wrote $OUT (encrypted). Safe to commit."
