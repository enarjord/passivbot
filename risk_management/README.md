# Passivbot Risk Management Extension

This directory contains a stand-alone risk management, portfolio monitoring,
and alerting system designed to work *with* Passivbot without modifying the
core trading bot.  The extension will grow iteratively.  In this iteration we
focus on providing a reproducible way to install Passivbot into a dedicated
virtual environment that the risk management service will rely on.

## Installation Overview

The risk management service is developed as a separate Python package that
imports Passivbot as a library.  To keep concerns separated and avoid mutating
existing Passivbot installation requirements, we maintain an isolated virtual
environment under `risk_management/.venv_passivbot_risk`.

Run the helper script to bootstrap the environment and install Passivbot in
editable mode:

```bash
./scripts/install_passivbot.sh
```


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
