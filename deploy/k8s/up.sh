#!/usr/bin/env bash
# Create (if needed) and deploy the private cluster. Idempotent: re-run to update.
# Run in WSL2 with Docker Desktop (WSL2 backend), k3d, kubectl, sops installed.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
for tool in docker k3d kubectl sops; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
[[ -f overlays/local/secrets.enc.yaml ]] || { echo "run ./setup-secrets.sh first" >&2; exit 1; }
mkdir -p .build

PY_IMAGE="python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce"

# 1. Where pods reach Ollama on the Windows host, and is it there?
HOST_IP="${OLLAMA_HOST_IP:-$(docker run --rm "$PY_IMAGE" python -c \
  "import socket; print(socket.gethostbyname('host.docker.internal'))")}"
if ! docker run --rm "$PY_IMAGE" python -c \
  "import urllib.request; urllib.request.urlopen('http://$HOST_IP:11434/api/version', timeout=5)" 2>/dev/null; then
  cat >&2 <<MSG
!! Ollama isn't reachable from containers at $HOST_IP:11434.
   On Windows, Ollama listens on 127.0.0.1 by default. Either:
   (a) set the Windows env var OLLAMA_HOST=$HOST_IP and restart Ollama, or
   (b) set OLLAMA_HOST=0.0.0.0 AND add a firewall rule allowing TCP 11434 only from
       the Docker/WSL subnet (see docs/SECURITY.md). Never open it to your LAN.
   Then re-run. (Override detection with OLLAMA_HOST_IP=...)
MSG
  exit 1
fi
cat > overlays/local/ollama.patch.yaml <<YAML
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: clients-to-ollama, namespace: finrag}
spec:
  egress:
    - to: [{ipBlock: {cidr: ${HOST_IP}/32}}]
      ports: [{protocol: TCP, port: 11434}]
---
apiVersion: v1
kind: ConfigMap
metadata: {name: finrag-config, namespace: finrag}
data:
  OLLAMA_URL: http://${HOST_IP}:11434
YAML

# 2. Cluster (loopback-only API + ingress; volumes on the host disk)
export FINRAG_STORAGE="${FINRAG_STORAGE:-$HOME/.finrag/storage}"
mkdir -p "$FINRAG_STORAGE" && chmod 700 "$FINRAG_STORAGE"
if ! k3d cluster list finrag >/dev/null 2>&1; then
  sed "s#\${FINRAG_STORAGE}#$FINRAG_STORAGE#" ../k3d/cluster.yaml > .build/cluster.yaml
  k3d cluster create --config .build/cluster.yaml
fi
kubectl config use-context k3d-finrag >/dev/null

# 3. Image: build locally, import into the cluster (never pushed to a registry)
TAG="$(git -C "$ROOT" rev-parse --short HEAD)"
docker build -f "$ROOT/Dockerfile.local" -t "finrag-local:$TAG" "$ROOT"
k3d image import "finrag-local:$TAG" -c finrag

# 4. Namespace, secrets (decrypted in a pipe, never written to disk), app
kubectl apply -f base/namespace.yaml
sops --decrypt overlays/local/secrets.enc.yaml | kubectl apply -f -
kubectl -n finrag delete job migrate --ignore-not-found   # Jobs are immutable; re-run fresh
kubectl kustomize overlays/local | sed "s#finrag-local:IMAGE_TAG#finrag-local:$TAG#g" | kubectl apply -f -

# 5. Wait: database, schema migration, then the app
kubectl -n finrag rollout status statefulset/postgres --timeout=300s
kubectl -n finrag wait --for=condition=complete job/migrate --timeout=300s
for d in egress-proxy api ui; do kubectl -n finrag rollout status "deployment/$d" --timeout=300s; done

echo
echo "FinRAG is up: https://finrag.localhost:8443  (this computer only)"
echo "Next: ./verify.sh   (security checks)"
