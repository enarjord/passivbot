# Passivbot Risk Management Extension

The risk management extension bundles portfolio monitoring, alerting, and a
full web dashboard that plugs into an existing Passivbot deployment without
modifying the core trading bot.  It consumes snapshots created either manually
or from live exchange data and surfaces consolidated exposure metrics, account
health, and automated notifications.

## Features at a glance

- **Terminal dashboard** – render JSON snapshots or live data in the console
  with exposure summaries, alert status, and per-account health checks.
- **Realtime data fetcher** – aggregate balances, positions, and orders across
  multiple ccxt-supported venues while honouring custom endpoint overrides and
  kill-switch commands.
- **Web dashboard** – FastAPI application with authenticated access, TLS
  support, report downloads, optional Grafana embeds, and kill-switch controls.
- **Alerting helpers** – configurable thresholds, SMTP email delivery, and
  human-readable notification channels shown alongside the dashboards.

## Prerequisites

- Python **3.9 or newer** available on the host system.
- `bash` compatible shell (on Windows use WSL or Git Bash).
- Access to the Passivbot repository (the extension expects to live inside the
  checkout).
- Exchange API credentials with reading permissions for balances and
  positions.  (Trading permissions are only required when you enable the kill
  switch.)

> ℹ️  The extension is intentionally isolated from the trading bot.  It runs in
> its own virtual environment and imports Passivbot only when you explicitly
> opt in via the installer flags described below.

## 1. Install the standalone environment

All tooling is packaged under `risk_management/`.  Bootstrap the virtual
environment once and reuse it whenever you work on the dashboard or run the
fetcher.

```bash
cd risk_management
./scripts/install_passivbot.sh --upgrade-packaging
source .venv_passivbot_risk/bin/activate
```

The helper script performs the following actions:

| Capability | Description |
| --- | --- |
| Virtualenv | Creates (or reuses) `.venv_passivbot_risk` under the module root. |
| Packaging tools | Upgrades `pip`, `setuptools`, and `wheel` when `--upgrade-packaging` is supplied. |
| Passivbot import path | Drops a `.pth` file that points to `../src`, making the Passivbot source tree importable without installing the package.  Override the location with `--link-passivbot /custom/path`. |
| Editable install | Add `--install-passivbot` to install Passivbot into the environment (useful for publishing or integration testing).  Extra arguments after `--` are forwarded to `pip install`. |

After activation, verify the installation by rendering the sample dashboard:

```bash
(risk) python -m risk_management.dashboard
```

You should see two example accounts, simulated positions, and a list of alert
thresholds.  Exit the environment later with `deactivate`.

### Manual installation (optional)

If you prefer to manage dependencies yourself:

```bash
cd risk_management
python -m venv .venv_passivbot_risk
source .venv_passivbot_risk/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r ../requirements.txt
python -m pip install -e ..  # optional: expose Passivbot modules
```

## 2. Configure realtime access

Copy the template configuration and fill in your details:

```bash
cp risk_management/realtime_config.example.json risk_management/realtime_config.local.json
```

Update the new file with the following information.

### Accounts

- `accounts` – list of exchanges you want to monitor.  Each entry accepts:
  - `name`: human-friendly label shown in dashboards.
  - `exchange`: ccxt identifier (for example `binanceusdm`, `okx`, `bybit`).
  - `api_key_id`: key within `api-keys.json`.  When present the loader pulls
    credentials automatically and merges any overrides from the `credentials`
    block.
  - `settle_currency`: wallet currency used to compute exposures (default
    `USDT`).
  - `symbols`: optional allowlist of markets to fetch.
  - `params.balance` / `params.positions`: forwarded verbatim to ccxt
    (useful for venues that require `{"type": "swap"}`).
  - `enabled`: set to `false` to temporarily disable an account without
    removing the block.
  - `debug_api_payloads`: enable verbose logging for a single account.
  - `credentials.enableRateLimit`: defaults to `true`.  The loader preserves
    ccxt's built-in throttling so API calls respect exchange rate limits unless
    you explicitly opt out.

### Credentials discovery

Place your trading keys in `api-keys.json`.  The loader searches for the file in
this order:

1. Next to the realtime configuration file.
2. Repository root (matching Passivbot's default location).

Override the lookup with `"custom_endpoints": {"path": "../configs/custom_endpoints.json", "autodiscover": false}` to mirror the trading bot's proxy settings.  Pass `--custom-endpoints` on the CLI to force a particular behaviour (`auto`, `none`, or a file path).

### Alerting and notifications

- `alert_thresholds` – wallet-wide and per-position percentages that trigger
  alerts in both the terminal and web dashboards.
- `notification_channels` – free-form strings describing where alerts should be
  sent.  Entries prefixed with `email:` are used for SMTP delivery when email
  settings are provided; other values are displayed for situational awareness.
- `email` – optional SMTP configuration.  Provide at least `host` and, if
  required by your server, the `port`, `username`, `password`, and TLS options
  (`use_tls`/`use_ssl`).  When missing, email alerts are skipped silently.

### Authentication and TLS

- `auth` – required for the web dashboard.  Supply a strong `secret_key`, a map
  of `users` to bcrypt hashes, and (optionally) a custom
  `session_cookie_name` or `https_only` flag.  Generate hashes with:
  ```bash
  python risk_management/scripts/hash_password.py
  ```
- `reports_dir` – directory where CSV exports generated by the web UI are
  stored.  Defaults to `<config-directory>/reports`.
- `grafana` – optional embed configuration with `dashboards` (title, url,
  description, height) and `base_url` for relative links.

### Debugging helpers

Set `"debug_api_payloads": true` globally or per account to dump the raw JSON
returned by ccxt.  When disabled the loader still provisions Passivbot's
handlers at INFO level so warnings and errors surface consistently.  Enabling
the flag raises verbosity to the familiar trading/backtesting format so payloads
respect `TRACE`/`DEBUG` levels and include timestamps.  Use this sparingly;
responses include large payloads and secret values are not redacted
automatically.

## 3. Run the terminal dashboard

The CLI consumes either a static snapshot or the realtime configuration created
above.

```bash
# Render a static JSON snapshot repeatedly (update the file in another terminal)
python -m risk_management.dashboard \
  --config risk_management/dashboard_config.json \
  --interval 5 --iterations 0

# Fetch data from the exchanges defined in realtime_config.local.json
python -m risk_management.dashboard \
  --realtime-config risk_management/realtime_config.local.json \
  --interval 30 --iterations 0
```

Use `Ctrl+C` to stop continuous runs.  When `--interval` is omitted the command
renders exactly once.

## 4. Launch the web dashboard

Serve an authenticated dashboard backed by the realtime fetcher:

```bash
python -m risk_management.web_server \
  --config risk_management/realtime_config.local.json \
  --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and sign in using the credentials from the
configuration file.  The page displays:

- Portfolio and per-account exposure metrics.
- Live funding and volatility snapshots.
- Outstanding alerts and notification targets.
- Kill-switch buttons (global, per account, or per position).
- CSV report downloads (stored under `reports_dir`).
- Optional Grafana panels embedded beneath the summary cards.

### Enabling TLS

Provide certificate files directly:

```bash
python -m risk_management.web_server \
  --config risk_management/realtime_config.local.json \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

or let the helper obtain certificates through Let's Encrypt:

```bash
python -m risk_management.web_server \
  --config risk_management/realtime_config.local.json \
  --letsencrypt-domain dashboard.example.com \
  --letsencrypt-email sre@example.com \
  --letsencrypt-http-port 80
```

Use `--letsencrypt-staging` and `--letsencrypt-dry-run` while testing to avoid
production rate limits.  When `auth.https_only` is true the server enforces
HTTPS and reminds you to supply certificates.

## 5. Exporting snapshots and reports

The web UI exposes a **Generate report** action for each account.  Reports are
stored as timestamped CSV files inside `reports_dir`.  You can also build a
presentable JSON snapshot programmatically via
`risk_management.snapshot_utils.build_presentable_snapshot()` for downstream
systems.

## 6. Email alerts

When the realtime fetcher detects alert conditions it sends an email to all
`notification_channels` entries prefixed with `email:`.  Ensure the SMTP server
allows the chosen sender address and credentials.  Errors are logged without
interrupting the polling loop.

## 7. Verifying the installation

Run the automated test suite to make sure all core behaviours work as expected:

```bash
pytest \
  tests/test_risk_management_account_clients.py \
  tests/test_risk_management_realtime.py \
  tests/test_risk_management_web.py \
  tests/risk_management
```

The tests mock external services and can be executed without live exchange
access.

## 8. Troubleshooting

- **`ModuleNotFoundError: custom_endpoint_overrides`** – run commands from the
  repository root or activate the virtual environment created by the installer
  so Passivbot modules are importable.
- **`Authentication failed` messages** – double-check API keys, required
  passphrases, and whether the credentials have the proper permissions.  The
  realtime fetcher caches the last error per account and resumes automatically
  when the issue is resolved.
- **Email alerts not delivered** – confirm the `email` block contains the
  correct server details and that `notification_channels` lists at least one
  `email:` recipient.
- **TLS errors** – ensure both `--ssl-certfile` and `--ssl-keyfile` are
  provided.  For automatic provisioning, verify that `certbot` is installed and
  reachable through the executable path supplied via `--letsencrypt-executable`.

With the configuration complete you can run the dashboard continuously,
monitor exposure across all connected accounts, and trigger protective actions
without touching the trading bot itself.
