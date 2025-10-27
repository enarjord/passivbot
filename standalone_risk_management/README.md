# Standalone Risk Management Toolkit

This repository packages Passivbot's risk-management utilities as an
independent, ccxt-powered toolkit.  It provides realtime account polling,
alerting helpers, and both command-line and FastAPI dashboards without
depending on the Passivbot source tree.

## Features

- **Realtime polling** – aggregate balances, open positions, and orders from any
  ccxt-supported exchange.  Custom endpoint overrides and rate-limit controls
  are applied automatically.
- **Kill switch** – cancel open orders and close positions from the dashboard or
  command line.
- **Dashboards** – run `python -m risk_management.dashboard` for a terminal
  overview or `python -m risk_management.web_server` for an authenticated web
  UI built with FastAPI.
- **Notifications** – configurable alert thresholds, optional email delivery,
  Telegram integrations, and daily snapshot summaries.
- **Reporting** – export CSV reports and archive snapshots for historical
  analysis.

## Project layout

```
standalone_risk_management/
├── README.md
├── LICENSE
├── pyproject.toml
├── custom_endpoint_overrides.py
└── risk_management/
    ├── __init__.py
    ├── account_clients.py
    ├── ccxt_helpers.py
    ├── configuration.py
    ├── dashboard.py
    ├── dashboard_config.json
    ├── email_notifications.py
    ├── history.py
    ├── realtime.py
    ├── realtime_config.example.json
    ├── realtime_config.json
    ├── reporting.py
    ├── scripts/
    │   └── hash_password.py
    ├── snapshot_utils.py
    ├── templates/
    │   └── ...
    ├── web.py
    └── web_server.py
```

The package keeps the original `risk_management` module name to minimise update
friction.  Your Passivbot deployment can remove the in-repo copy and depend on
this project instead.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .[all]
```

- The base install provides the realtime fetcher and CLI dashboard.
- `.[dashboard]` adds FastAPI, uvicorn, and template dependencies.
- `.[telegram]` enables Telegram notifications via `httpx`.
- `.[tls]` installs Certbot helpers used by the optional HTTPS automation.
- `.[all]` pulls every optional extra.

## Getting started

1. Copy `risk_management/realtime_config.example.json` to a new file and fill in
   your exchange credentials, alert thresholds, and notification targets.
2. Place your API keys in `api-keys.json` next to the configuration file.  The
   loader merges these credentials with any inline overrides.
3. Launch the terminal dashboard:

   ```bash
   python -m risk_management.dashboard \
       --realtime-config risk_management/realtime_config.json \
       --interval 30
   ```

4. (Optional) Serve the web dashboard:

   ```bash
   python -m risk_management.web_server \
       --config risk_management/realtime_config.json \
       --host 0.0.0.0 --port 8000
   ```

   The configuration file controls authentication, TLS certificates, and where
   CSV reports are saved.

## Packaging the repository

Create a distributable archive at any time with Python's built-in `zipfile`
module:

```bash
python -m zipfile -c standalone-risk.zip standalone_risk_management
```

Upload the generated `standalone-risk.zip` file to a fresh Git repository (or
share it directly) to continue development independently of Passivbot.

## License

The toolkit inherits Passivbot's [Unlicense](LICENSE), leaving you free to use
or modify it for any purpose.
