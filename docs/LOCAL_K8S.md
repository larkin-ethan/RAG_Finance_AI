# Private local deployment (k3d + Ollama)

Your holdings, positions, and stock checks run here, on this PC only. See
[SECURITY.md](SECURITY.md) for the threat model. Run Phase A ([WINDOWS_RUNBOOK.md](WINDOWS_RUNBOOK.md))
first if you haven't.

## 1. One-time installs

**Windows:**
1. **Ollama:** https://ollama.com/download. Then, in PowerShell:
   ```powershell
   ollama pull qwen3:14b
   ollama pull gemma3:12b
   ollama pull qwen3-embedding:0.6b
   ```
2. **Docker Desktop** with the WSL2 backend (already done in Phase A). In Settings →
   Resources → WSL integration, enable your Ubuntu distro.
3. **BitLocker / Device encryption:** turn it on (Settings → Privacy & security).

**WSL2 Ubuntu** (`wsl` in PowerShell, then):
```bash
sudo apt-get update && sudo apt-get install -y age openssl git
# k3d, kubectl, sops, uv: official installers
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
  && sudo install -m 0755 kubectl /usr/local/bin/ && rm kubectl
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**sops:** download the latest `sops-*.linux.amd64` from
https://github.com/getsops/sops/releases, verify its checksum, and install it to `/usr/local/bin/sops`.

Clone the repo **inside WSL** (e.g. `~/RAG_Finance_AI`), then:
```bash
cd ~/RAG_Finance_AI && uv sync
cp config/holdings.example.yaml config/holdings.yaml && nano config/holdings.yaml   # your tickers
```

## 2. Secrets (once)

```bash
deploy/k8s/setup-secrets.sh
```
This asks for your app password, SEC User-Agent, and (optional) Tiingo/FRED keys. It creates
your **age key** (back it up offline now) and a name-constrained local CA, then writes
`deploy/k8s/overlays/local/secrets.enc.yaml` (encrypted, safe to commit).

**Trust the CA in Windows**, current user only, no admin. In PowerShell:
```powershell
Import-Certificate -FilePath "\\wsl$\Ubuntu\home\<you>\.finrag\ca\ca.crt" -CertStoreLocation Cert:\CurrentUser\Root
```

## 3. Bring it up

```bash
deploy/k8s/up.sh
```
✅ Prints `FinRAG is up: https://finrag.localhost:8443`.

If it stops with **"Ollama isn't reachable"**, follow its message (see SECURITY.md →
Ollama exposure) and re-run.

## 4. First ingest of your holdings

```bash
kubectl -n finrag create job first-ingest --from=cronjob/update-filings
kubectl -n finrag logs -f job/first-ingest
```
This fetches your holdings' filings through the egress proxy, then parses, chunks, and
embeds them with Ollama on the GPU. After that, the CronJob runs daily at 07:00.

## 5. Verify security

```bash
deploy/k8s/verify.sh
```
Every line should read `PASS`. Also do the two manual checks it prints (Windows listening
sockets, and that your phone **can't** open `https://<PC-LAN-IP>:8443`).

## 6. Stock checks and your portfolio

Needs free API keys for **Tiingo** (prices) and **FRED** (10-year Treasury yield),
entered in `setup-secrets.sh`. If you skipped them, re-run it (read its warning first).

- **Stock Check** page: pick a ticker, goals (dividend income, growth, value, stability)
  and an optional timeline. You get fit grades, the factors behind them, historical outcomes
  for your holding period, new risk-factor text, and your lots' tax view.
- **Portfolio** page: add lots by hand or import your broker's CSV (map the columns once;
  every row is validated; nothing imports unless all rows are valid). Enter your
  filing status and estimated income for federal tax estimates.

Everything here is **research, not investment or tax advice**. Grades describe how
well reported data fits a goal; they don't predict returns.

**Backtest** (measures whether the grades meant anything historically, and feeds the
timeline report's "how much do fundamentals matter" line):
```bash
kubectl -n finrag exec deploy/api -- python -m finrag.ratings.backtest
```
It fetches ~100 companies' XBRL data and prices once (cached), so the first run takes a while.

## Day-to-day

| Task | Command |
|---|---|
| Open the app | https://finrag.localhost:8443 |
| Update after code changes | `git pull && deploy/k8s/up.sh` |
| Logs | `kubectl -n finrag logs deploy/api` (or `ui`, `egress-proxy`) |
| Manual backup | `kubectl -n finrag create job backup-now --from=cronjob/backup` |
| Test a restore | `deploy/k8s/restore-test.sh` |
| Stop (keeps data) | `deploy/k8s/down.sh` |
| Rotate secrets | `deploy/k8s/setup-secrets.sh` (read its warning first) |
