# Security: private local deployment

Scope: the **private** FinRAG instance in a local k3d cluster on the Windows PC
(holdings, positions, tax profile, stock checks). The public Google Cloud demo holds only
public SEC filings and is out of scope here.

## Assets

| Asset | Where | Sensitivity |
|---|---|---|
| Positions, cost basis, tax profile | Postgres (field-encrypted) | **High**: personal financial data |
| Holdings watchlist | Secret `finrag-holdings` | High: reveals your portfolio |
| App password | Only its argon2id hash (Secret) | High |
| age private key | `~/.config/sops/age/keys.txt` (WSL), plus your offline backup | **Critical**: decrypts secrets and backups |
| Local CA key | `~/.finrag/ca/ca.key` | Medium (name-constrained, see below) |
| Questions, answers, LLM prompts | Postgres caches, Ollama | Medium |
| Public filings, embeddings | Postgres | Low |

## Threats and mitigations

| # | Threat | Mitigations |
|---|---|---|
| T1 | **Someone on your network** reaches the app or the Kubernetes API | Ingress and API server bound to **127.0.0.1** only. No port 80, no NodePorts or LoadBalancer Services (ResourceQuota = 0). `verify.sh` checks this, plus the manual Windows socket check. |
| T2 | **Private data leaves the machine** (cloud LLM, telemetry) | LLM and embeddings run on **Ollama on the host GPU**. Pods have **no direct internet access** (default-deny NetworkPolicy). Outbound HTTPS goes only through the allowlist proxy to `www.sec.gov`, `data.sec.gov`, `api.tiingo.com` and `api.stlouisfed.org`. Streamlit usage stats are off. Position and lot data is never put in prompts. |
| T3 | **A compromised container** (dependency CVE, RCE) moves sideways | Restricted Pod Security Standard is enforced: non-root, read-only root filesystem, all capabilities dropped, no privilege escalation, seccomp, no service-account token. Per-flow NetworkPolicies (e.g. the UI can't reach Postgres, nothing but the UI reaches the API). The app DB role has **no DDL**. The proxy refuses private IPs (no SSRF to the host or cloud metadata). |
| T4 | **Stolen database files or backups** | Positions and tax fields are **Fernet-encrypted** in the app. Backups are **age-encrypted** with a key that never enters the cluster. Volumes live on the host disk under **BitLocker** (you turn this on). Secrets are encrypted at rest in k3s (`--secrets-encryption`). |
| T5 | **Secrets leak through git** | Only **SOPS + age encrypted** `secrets.enc.yaml` is committed. Plaintext exists only in RAM during `setup-secrets.sh`. CI and `verify.sh` grep for private keys and DB passwords, and CI fails on a hit. |
| T6 | **Password guessing** at the login page | argon2id hashing. After 5 failures, a 60 s lockout that doubles each time (max 1 h), shared across tabs. 30-minute idle timeout. No password stored, only the hash. |
| T7 | **Malicious web content** (another site attacking the app in your browser) | The API isn't browser-reachable (internal Service + service token). Streamlit XSRF protection is on. Security headers: HSTS, frame-deny (no clickjacking), nosniff, no-referrer, restrictive Permissions-Policy. TLS only, from a **name-constrained** local CA. |
| T8 | **Prompt injection via filings** ("ignore your instructions…") | Filing text is delimited and marked as data in the prompt. Grades are computed deterministically and the LLM can't change them. Every number and quote in LLM output is verified against the sources, and unverifiable output becomes "I don't know" or a template. |
| T9 | **Supply chain** (malicious or vulnerable packages and images) | `uv.lock` pins every dependency with hashes. Images are pinned by digest. CI runs `pip-audit` and a Trivy manifest scan, and `verify.sh` runs a Trivy image scan. Dependabot opens weekly update PRs. The app image is built locally and never pulled from a registry. |
| T10 | **Data loss** | Nightly encrypted backups, 14 days kept. `restore-test.sh` proves a backup restores, with row counts compared. Data volumes survive cluster deletion. |

### The name-constrained CA
`setup-secrets.sh` creates a local CA that may **only** sign `finrag.localhost` and the
in-cluster Postgres names. Trusting it in Windows (current-user store) therefore can't be
abused to impersonate other sites, even if its key leaked. That's the main risk of
local CAs, and this design removes it.

### Ollama exposure
Ollama runs on the Windows host. Pods reach it at the address `up.sh` detects
(`host.docker.internal`). A NetworkPolicy allows only `api` and jobs to reach that one IP
on port 11434.
- **Default:** Ollama listens on `127.0.0.1`. If pods can't reach it, `up.sh` tells you.
- **Preferred fix:** set `OLLAMA_HOST` to the Docker host IP that `up.sh` prints.
- **Only if needed:** listen on `0.0.0.0` **and** add a Windows Firewall rule limiting TCP
  11434 to the Docker/WSL subnet. **Never** allow it from your LAN or the internet:
  Ollama has no authentication.

## Residual risks (accepted, by design)

- **A compromised Windows account defeats all of this.** It can read the age key, the
  kubeconfig, and decrypted data in memory. Keep Windows updated, use a strong login,
  and turn on BitLocker.
- **Single-user design:** no roles or multi-user audit beyond the position-change log.
- **In-memory throttling** resets when the UI pod restarts. That's acceptable on loopback only.
- **Free data sources** (SEC, Tiingo, FRED) are trusted for correctness. The system
  cross-checks numbers but can't prove upstream data is right.
- **This is a research tool, not investment or tax advice.** Tax figures are federal estimates.

## Your checklist

- [ ] BitLocker on for the system drive (Settings → Privacy & security → Device encryption)
- [ ] age key backed up offline, and the backup tested (`restore-test.sh`)
- [ ] Local CA trusted in the **current-user** store only
- [ ] `./verify.sh` passes after every `up.sh`
- [ ] Ollama not reachable from another device (try `http://<PC-LAN-IP>:11434` from your phone; it must fail)
- [ ] Review Dependabot PRs weekly, and re-run `setup-secrets.sh` yearly before the leaf certificates expire (397 days)
