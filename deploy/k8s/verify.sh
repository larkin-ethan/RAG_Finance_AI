#!/usr/bin/env bash
# Security verification for the private cluster. Every check prints PASS/FAIL;
# exits non-zero if any check fails. Run after up.sh (in WSL2).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
NS=finrag
TAG="$(git -C "$ROOT" rev-parse --short HEAD)"
IMG="finrag-local:$TAG"
TRIVY="aquasec/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
FAILED=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILED=1; }
check() { if eval "$2" >/dev/null 2>&1; then pass "$1"; else fail "$1"; fi; }

echo "== Pod hardening"
kubectl -n $NS get pods -o json | python3 -c '
import json, sys
bad = []
for p in json.load(sys.stdin)["items"]:
    spec, name = p["spec"], p["metadata"]["name"]
    if spec.get("automountServiceAccountToken", True): bad.append(f"{name}: SA token mounted")
    if not spec.get("securityContext", {}).get("runAsNonRoot"): bad.append(f"{name}: runAsNonRoot")
    for c in spec.get("initContainers", []) + spec["containers"]:
        sc = c.get("securityContext", {})
        if not sc.get("readOnlyRootFilesystem"): bad.append(f"{name}/{c[\"name\"]}: writable rootfs")
        if sc.get("allowPrivilegeEscalation", True): bad.append(f"{name}/{c[\"name\"]}: privilege escalation")
        if "ALL" not in sc.get("capabilities", {}).get("drop", []): bad.append(f"{name}/{c[\"name\"]}: capabilities")
print("\n".join(bad)); sys.exit(1 if bad else 0)' && pass "all pods non-root, read-only rootfs, no escalation, caps dropped" \
  || fail "pod hardening (see above)"
check "Pod Security 'restricted' rejects a privileged pod" \
  "! kubectl -n $NS run pss-probe --image=$IMG --privileged --restart=Never --dry-run=server"

echo "== Network policies (probes run from short-lived restricted pods)"
probe() {  # name, labels(json), python-statement -> exit 0 if the statement succeeds
  local ov
  ov=$(python3 -c "import json,sys; print(json.dumps({'metadata':{'labels':json.loads(sys.argv[1])},'spec':{
    'automountServiceAccountToken':False,'securityContext':{'runAsNonRoot':True,'runAsUser':10001,'seccompProfile':{'type':'RuntimeDefault'}},
    'containers':[{'name':'probe','image':sys.argv[2],'imagePullPolicy':'Never','command':['python','-c',sys.argv[3]],
    'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}}}]}}))" "$2" "$IMG" "$3")
  kubectl -n $NS run "np-$1-$RANDOM" --image="$IMG" --restart=Never --rm -i --quiet \
    --overrides="$ov" --timeout=60s >/dev/null 2>&1
}
tcp() { echo "import socket; socket.create_connection(('$1', $2), 4).close()"; }
connect_via_proxy() { echo "import socket; s=socket.create_connection(('egress-proxy',3128),5); s.sendall(b'CONNECT $1:443 HTTP/1.1\r\n\r\n'); r=s.recv(64); assert r.startswith(b'HTTP/1.1 200'), r"; }

probe none '{}' "$(tcp postgres 5432)" && fail "unlabeled pod -> postgres blocked" || pass "unlabeled pod -> postgres blocked"
probe none '{}' "$(tcp api 8000)" && fail "unlabeled pod -> api blocked" || pass "unlabeled pod -> api blocked"
probe none '{}' "$(tcp 1.1.1.1 443)" && fail "unlabeled pod -> internet blocked" || pass "unlabeled pod -> internet blocked"
probe ui '{"app":"ui"}' "$(tcp api 8000)" && pass "ui -> api allowed" || fail "ui -> api allowed"
probe ui '{"app":"ui"}' "$(tcp postgres 5432)" && fail "ui -> postgres blocked" || pass "ui -> postgres blocked"
probe pc '{"postgres-client":"true"}' "$(tcp postgres 5432)" && pass "db client -> postgres allowed" || fail "db client -> postgres allowed"
probe ep '{"egress-proxy-client":"true"}' "$(connect_via_proxy www.sec.gov)" && pass "proxy client -> www.sec.gov via proxy allowed" || fail "proxy -> www.sec.gov allowed"
probe ep '{"egress-proxy-client":"true"}' "$(connect_via_proxy example.com)" && fail "proxy -> example.com denied" || pass "proxy -> example.com denied"
probe ep '{"egress-proxy-client":"true"}' "$(tcp 1.1.1.1 443)" && fail "proxy client -> direct internet blocked" || pass "proxy client -> direct internet blocked"

echo "== Database"
check "postgres rejects non-TLS connections" \
  "! kubectl -n $NS exec postgres-0 -- psql 'host=127.0.0.1 dbname=finrag user=finrag_app sslmode=disable' -c 'select 1'"
check "app role cannot create tables (no DDL)" \
  "kubectl -n $NS exec postgres-0 -- psql -U postgres -d finrag -tAc \"select not has_schema_privilege('finrag_app','public','CREATE')\" | grep -qx t"

echo "== Exposure (listening sockets should be 127.0.0.1 only)"
echo "   On Windows PowerShell run:  Get-NetTCPConnection -State Listen -LocalPort 8443,6445 | ft LocalAddress,LocalPort"
echo "   Expected LocalAddress 127.0.0.1 only. Also try https://<this-PC-LAN-IP>:8443 from your phone: it must fail."

echo "== Secrets in git"
check "secrets.enc.yaml is SOPS-encrypted" "grep -q 'ENC\\[' overlays/local/secrets.enc.yaml && grep -q '^sops:' overlays/local/secrets.enc.yaml"
check "no plaintext keys or DB passwords tracked in git" \
  "! git -C '$ROOT' grep -nIE 'BEGIN [A-Z ]*PRIVATE KEY|AGE-SECRET-KEY[-]|postgresql://[^:/]+:[^@\\$]{12,}@' -- ':!*.enc.yaml'"

echo "== Vulnerability scan (Trivy)"
check "image has no HIGH/CRITICAL vulnerabilities with a fix available" \
  "docker run --rm -v /var/run/docker.sock:/var/run/docker.sock $TRIVY image --quiet --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1 $IMG"
check "k8s manifests pass Trivy misconfiguration scan (HIGH/CRITICAL)" \
  "docker run --rm -v '$ROOT/deploy/k8s':/k8s:ro $TRIVY config --quiet --severity HIGH,CRITICAL --exit-code 1 /k8s/base"

echo "== Backups"
if kubectl -n $NS get cronjob backup >/dev/null 2>&1; then
  ./restore-test.sh && pass "backup restore test" || fail "backup restore test"
fi

echo
[[ $FAILED == 0 ]] && echo "All security checks passed." || { echo "Some checks FAILED."; exit 1; }
