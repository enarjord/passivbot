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
