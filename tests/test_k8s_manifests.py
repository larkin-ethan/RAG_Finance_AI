"""Policy-as-tests for the private k8s manifests: no cluster needed (runs in CI).
A future edit that weakens pod security fails the build here first."""

from pathlib import Path

import pytest
import yaml

BASE = Path(__file__).resolve().parents[1] / "deploy" / "k8s" / "base"
DOCS = [d for f in sorted(BASE.glob("*.yaml")) for d in yaml.safe_load_all(f.read_text()) if d]
WORKLOADS = [d for d in DOCS if d["kind"] in ("Deployment", "StatefulSet", "Job", "CronJob")]


def pod_spec(doc: dict) -> dict:
    if doc["kind"] == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return doc["spec"]["template"]["spec"]


def pod_labels(doc: dict) -> dict:
    if doc["kind"] == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["metadata"]["labels"]
    return doc["spec"]["template"]["metadata"]["labels"]


def containers(doc: dict) -> list[dict]:
    spec = pod_spec(doc)
    return spec.get("initContainers", []) + spec["containers"]


def test_kustomization_lists_every_manifest():
    listed = set(yaml.safe_load((BASE / "kustomization.yaml").read_text())["resources"])
    present = {f.name for f in BASE.glob("*.yaml")} - {"kustomization.yaml"}
    assert listed == present


def test_namespace_enforces_restricted_pod_security():
    ns = next(d for d in DOCS if d["kind"] == "Namespace")
    assert ns["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"


@pytest.mark.parametrize("doc", WORKLOADS, ids=lambda d: f"{d['kind']}/{d['metadata']['name']}")
def test_pod_security(doc):
    spec = pod_spec(doc)
    assert spec.get("automountServiceAccountToken") is False
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True and psc["runAsUser"] != 0
    assert psc["seccompProfile"]["type"] == "RuntimeDefault"
    assert not spec.get("hostNetwork") and not spec.get("hostPID") and not spec.get("hostIPC")
    for vol in spec.get("volumes", []):
        assert "hostPath" not in vol, "hostPath volumes are forbidden"
    for c in containers(doc):
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False, c["name"]
        assert sc["readOnlyRootFilesystem"] is True, c["name"]
        assert sc["capabilities"]["drop"] == ["ALL"], c["name"]
        assert not sc.get("privileged"), c["name"]
        assert c["resources"]["limits"]["memory"] and c["resources"]["limits"]["cpu"], c["name"]


@pytest.mark.parametrize("doc", WORKLOADS, ids=lambda d: f"{d['kind']}/{d['metadata']['name']}")
def test_images_pinned_or_local(doc):
    for c in containers(doc):
        image = c["image"]
        if image.startswith("finrag-local:"):
            assert c["imagePullPolicy"] == "Never"  # built locally, never pulled
        else:
            assert "@sha256:" in image, f"{image} must be pinned by digest"


def test_no_nodeport_or_loadbalancer_services():
    for d in DOCS:
        if d["kind"] == "Service":
            assert d["spec"].get("type", "ClusterIP") == "ClusterIP", d["metadata"]["name"]


def test_default_deny_and_every_workload_selected_by_a_policy():
    policies = [d for d in DOCS if d["kind"] == "NetworkPolicy"]
    deny = next(p for p in policies if p["metadata"]["name"] == "default-deny-all")
    assert deny["spec"]["podSelector"] == {}
    assert set(deny["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    assert "ingress" not in deny["spec"] and "egress" not in deny["spec"]  # allows nothing

    specific = [p for p in policies if p["spec"]["podSelector"].get("matchLabels")]
    for doc in WORKLOADS:
        labels = pod_labels(doc)
        selected = [
            p["metadata"]["name"]
            for p in specific
            if p["spec"]["podSelector"]["matchLabels"].items() <= labels.items()
        ]
        assert selected, f"{doc['metadata']['name']} has no allow policy (it could do nothing)"


def test_only_ui_is_exposed_and_tls_is_on():
    ingresses = [d for d in DOCS if d["kind"] == "Ingress"]
    assert len(ingresses) == 1
    ing = ingresses[0]
    backends = {
        p["backend"]["service"]["name"] for r in ing["spec"]["rules"] for p in r["http"]["paths"]
    }
    assert backends == {"ui"}  # the API is never reachable from the browser
    assert ing["spec"]["tls"][0]["secretName"] == "finrag-tls"
    assert (
        ing["metadata"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"]
        == "websecure"
    )


def test_api_ingress_only_from_ui():
    pol = next(
        d
        for d in DOCS
        if d["kind"] == "NetworkPolicy" and d["metadata"]["name"] == "api-from-ui-only"
    )
    sources = [f["podSelector"]["matchLabels"] for r in pol["spec"]["ingress"] for f in r["from"]]
    assert sources == [{"app": "ui"}]


def test_egress_proxy_only_https_and_no_private_ranges():
    pol = next(
        d
        for d in DOCS
        if d["kind"] == "NetworkPolicy" and d["metadata"]["name"] == "egress-proxy-policy"
    )
    rule = pol["spec"]["egress"][0]
    assert [p["port"] for p in rule["ports"]] == [443]
    excluded = set(rule["to"][0]["ipBlock"]["except"])
    assert {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"} <= excluded


def test_secrets_come_from_secret_refs_never_literals():
    sensitive = ("PASSWORD", "TOKEN", "KEY", "DATABASE_URL", "HASH")
    for doc in WORKLOADS:
        for c in containers(doc):
            for env in c.get("env", []):
                if any(s in env["name"] for s in sensitive) and env["name"] != "AGE_RECIPIENT":
                    assert "valueFrom" in env, f"{c['name']}: {env['name']} must come from a Secret"


def test_postgres_requires_tls():
    cm = next(
        d for d in DOCS if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "postgres-config"
    )
    hba = cm["data"]["pg_hba.conf"]
    assert "hostnossl all  all  0.0.0.0/0      reject" in hba
    sts = next(d for d in DOCS if d["kind"] == "StatefulSet")
    assert "ssl=on" in sts["spec"]["template"]["spec"]["containers"][0]["args"]


def test_db_clients_have_db_label_and_app_never_migrates():
    for doc in WORKLOADS:
        for c in containers(doc):
            env = {e["name"]: e for e in c.get("env", [])}
            if "DATABASE_URL" in env:
                assert env["DB_AUTO_MIGRATE"]["value"] == "false"
                assert pod_labels(doc).get("postgres-client") == "true"
