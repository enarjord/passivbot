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

## What the installer does

* Creates (or reuses) the virtual environment at
  `risk_management/.venv_passivbot_risk`.
* Upgrades `pip`, `setuptools`, and `wheel` to recent versions inside that
  environment.
* Drops a `.pth` file into the environment's `site-packages` directory so the
  Passivbot source tree at `../src` is importable without additional
  installation steps.
* Optionally installs Passivbot into the environment when
  `--install-passivbot` is requested, defaulting to a `pip install -e .
  --use-pep517` invocation that still supports forwarding custom flags.

## Requirements

* Python 3.9+ available on the host system.
* `bash` compatible shell (for Windows users, WSL or Git Bash is recommended).

Future iterations will introduce the risk management package itself, portfolio
metrics calculations, monitoring pipelines, and alert integrations while
respecting the isolation between Passivbot and the new tooling.
