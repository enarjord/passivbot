# Passivbot Risk Management Extension

This directory contains a stand-alone risk management, portfolio monitoring,
and alerting system designed to work *with* Passivbot without modifying the
core trading bot.  The extension will grow iteratively.  In this iteration we
ship a self-contained terminal dashboard that consumes a JSON snapshot and
highlights portfolio exposure alongside simulated alert messages.  Everything
can run without touching a live Passivbot environment so you can experiment
freely.

## Quick start

1. (Optional) Bootstrap the isolated virtual environment so dependencies stay
   separate from your trading installation:

   ```bash
   cd risk_management
   ./scripts/install_passivbot.sh --upgrade-packaging
   source .venv_passivbot_risk/bin/activate
   ```

2. Render the dashboard using the included sample snapshot:

   ```bash
   python -m risk_management.dashboard
   ```

   The command prints a summary of two example accounts, their positions, and
   any alerts triggered by the configured thresholds.  Edit
   `risk_management/dashboard_config.json` to plug in your own numbers or point
   the command to a custom snapshot via `--config /path/to/file.json`.

3. To mimic continuous monitoring, add `--interval 5 --iterations 0` and update
   the JSON file in another terminal.  The CLI will re-read the file on the
   chosen cadence and immediately reflect the changes.

## Realtime monitoring

Provide exchange credentials via `risk_management/realtime_config.json` (see
`realtime_config.example.json` for a complete template) and point the CLI at the
file to fetch balances and positions directly from the exchanges:

```bash
python -m risk_management.dashboard --realtime-config risk_management/realtime_config.json --interval 30 --iterations 0
```

The command connects to each configured account, aggregates the portfolio
metrics, and continuously renders the dashboard.  Any fetch issues are surfaced
inline under the affected account.

## Web dashboard

Launch the FastAPI web server to obtain an authenticated dashboard with live
updates:

```bash
python -m risk_management.web_server --config risk_management/realtime_config.json --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000` to sign in and view the interactive
dashboard.  The page automatically polls for fresh data and updates account
cards, alerts, and notification channels without a full refresh.

### Authentication

The web UI requires bcrypt hashed passwords.  Use the helper script to generate
hashes:

```bash
python risk_management/scripts/hash_password.py
```

Paste the resulting hash into the `auth.users` section of your realtime
configuration file.

The previous quick start guide that focused solely on creating the virtual
environment is kept below for reference.

## Installation Overview

The risk management service is developed as a separate Python package that
imports Passivbot as a library.  To keep concerns separated and avoid mutating
existing Passivbot installations, we maintain an isolated virtual environment
under `risk_management/.venv_passivbot_risk` and link it directly to the
repository's source tree.

Run the helper script to bootstrap the environment:

```bash
./scripts/install_passivbot.sh
```

The script prepares the virtual environment without touching your existing
Passivbot installation.  By default it does **not** install Passivbot or link to
its source tree, keeping the workspace fully isolated for the upcoming risk
management utilities.

If you want code inside the virtual environment to import Passivbot directly
from a local checkout, provide the path to Passivbot's `src/` directory via
`--link-passivbot`:

```bash
./scripts/install_passivbot.sh --link-passivbot /path/to/passivbot/src
```

This optional flag drops a `.pth` file into the environment's `site-packages`
directory so modules under the supplied path become importable.  Skipping the
flag leaves the environment unaware of Passivbot entirely, which can be useful
if you plan to interact with Passivbot over APIs or other integration points
instead of importing its Python modules.
existing Passivbot installation requirements, we maintain an isolated virtual
environment under `risk_management/.venv_passivbot_risk`.

Run the helper script to bootstrap the environment and install Passivbot in
editable mode:



```bash
./scripts/install_passivbot.sh
```


The script prepares the virtual environment and writes a `.pth` file so that
`risk_management` code can import Passivbot modules directly from `../src`
without reinstalling Passivbot.  This lets you keep running Passivbot from your
existing environment while prototyping new risk tooling separately.

If you want the helper to refresh `pip`, `setuptools`, and `wheel` inside the
virtual environment, add `--upgrade-packaging` to the command.  Otherwise those
tools are left untouched to avoid unnecessary downloads.

The script upgrades core packaging tools inside the virtual environment and
writes a `.pth` file so that `risk_management` code can import Passivbot
modules directly from `../src` without a redundant pip installation.  This lets
you keep running Passivbot from your existing environment while prototyping new
risk tooling separately.

If you *do* want Passivbot installed into the risk-management environment (for
example, to publish the package to an index or test installation flows), pass
`--install-passivbot`.  Any arguments after `--` are forwarded to `pip
install`:

```bash
./scripts/install_passivbot.sh --install-passivbot -- --no-build-isolation
```

After bootstrapping the virtual environment you can activate it with `source
.venv_passivbot_risk/bin/activate` and proceed with future iterations—portfolio
analytics, monitoring, and alerting—while keeping the main Passivbot setup
untouched.


If you need to adjust the build invocation (for example, to pass additional
flags to `pip install`), append them to the script call and they will be
forwarded to the editable install step:

```bash
./scripts/install_passivbot.sh --no-build-isolation
```

After installation the virtual environment will be ready for future
iterations—where portfolio analytics, monitoring, and alerting features will be
added—to import Passivbot modules and configurations.




## What the installer does

* Creates (or reuses) the virtual environment at
  `risk_management/.venv_passivbot_risk`.

* Optionally writes a `.pth` file into the environment's `site-packages`
  directory when `--link-passivbot` is supplied so the referenced Passivbot
  source tree becomes importable without additional installation steps.
* Optionally upgrades `pip`, `setuptools`, and `wheel` when
  `--upgrade-packaging` is provided.

* Drops a `.pth` file into the environment's `site-packages` directory so the
  Passivbot source tree at `../src` is importable without additional
  installation steps.
* Optionally upgrades `pip`, `setuptools`, and `wheel` when
  `--upgrade-packaging` is provided.

* Upgrades `pip`, `setuptools`, and `wheel` to recent versions inside that
  environment.
* Drops a `.pth` file into the environment's `site-packages` directory so the
  Passivbot source tree at `../src` is importable without additional
  installation steps.
* Optionally installs Passivbot into the environment when
  `--install-passivbot` is requested, defaulting to a `pip install -e .
  --use-pep517` invocation that still supports forwarding custom flags.

* Upgrades `pip`, `setuptools`, and `wheel` to recent versions.

* Installs Passivbot's build prerequisite `setuptools-rust` that is
  required during editable installations of the core project.
* Installs Passivbot from the repository root in editable mode with PEP 517
  builds enabled by default, ensuring nested requirement files are resolved
  correctly. Any extra flags passed to the script are forwarded to the `pip`
  command so you can tailor the build locally.


* Installs Passivbot's build prerequisite `setuptools-rust` that is
  required during editable installations of the core project.

* Installs Passivbot from the repository root in editable mode so that local
  changes to Passivbot are instantly available to the risk management package.


## Requirements

* Python 3.9+ available on the host system.
* `bash` compatible shell (for Windows users, WSL or Git Bash is recommended).

Future iterations will introduce the risk management package itself, portfolio
metrics calculations, monitoring pipelines, and alert integrations while
respecting the isolation between Passivbot and the new tooling.
