# Passivbot Risk Management Extension

This directory contains a stand-alone risk management, portfolio monitoring,
and alerting system designed to work *with* Passivbot without modifying the
core trading bot.  The extension will grow iteratively.  In this iteration we
focus on providing a reproducible way to prepare an isolated virtual
environment that can import Passivbot's source tree without altering the
existing installation you may already be using for live trading.

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

If you want the helper to refresh `pip`, `setuptools`, and `wheel` inside the
virtual environment, add `--upgrade-packaging` to the command.  Otherwise those
tools are left untouched to avoid unnecessary downloads.

After bootstrapping the virtual environment you can activate it with `source
.venv_passivbot_risk/bin/activate` and proceed with future iterations—portfolio
analytics, monitoring, and alerting—while keeping the main Passivbot setup
untouched.

## What the installer does

* Creates (or reuses) the virtual environment at
  `risk_management/.venv_passivbot_risk`.
* Optionally writes a `.pth` file into the environment's `site-packages`
  directory when `--link-passivbot` is supplied so the referenced Passivbot
  source tree becomes importable without additional installation steps.
* Optionally upgrades `pip`, `setuptools`, and `wheel` when
  `--upgrade-packaging` is provided.

## Requirements

* Python 3.9+ available on the host system.
* `bash` compatible shell (for Windows users, WSL or Git Bash is recommended).

Future iterations will introduce the risk management package itself, portfolio
metrics calculations, monitoring pipelines, and alert integrations while
respecting the isolation between Passivbot and the new tooling.
