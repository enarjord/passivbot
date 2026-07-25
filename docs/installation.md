# Installing Passivbot

This guide collects all steps (and common pitfalls) for setting up Passivbot on a fresh machine. For quick instructions see the README; this file adds the missing detail when things go wrong.

## 1. Prerequisites

- **Python 3.12 or 3.14** – Earlier versions and Python 3.13 are not supported. On Debian/Ubuntu
  systems whose default interpreter is a supported version, install it and its build headers with
  `sudo apt install python3 python3-venv python3-dev`. Otherwise install a supported newer
  interpreter and note its executable name, such as `python3.14`.
- **Rust toolchain** – Passivbot’s hot paths live in Rust. Install via [rustup](https://rustup.rs/) if `rustc --version` is not available.
- **C build tools** – Ubuntu/Debian: `sudo apt install build-essential`. macOS: Xcode command-line tools (`xcode-select --install`).
- **Virtual environment** – Strongly recommended so dependencies do not leak into the system interpreter.

## 2. Clone & create venv

```bash
# Clone the repo
 git clone https://github.com/enarjord/passivbot.git
 cd passivbot

# Select the supported interpreter you installed. Use python3 only if it is 3.12 or 3.14.
 PYTHON_BIN=python3.14
 "$PYTHON_BIN" --version

# Create + activate the venv with that exact interpreter (inside repo root)
 "$PYTHON_BIN" -m venv venv
 source venv/bin/activate  # Windows: venv\Scripts\activate
 python --version
```

Replace `python3.14` with `python3.12` as needed. Do not assume the system `python3` changed when
a versioned interpreter was installed alongside it.

## 3. Install Passivbot

Choose the install profile that matches the machine:

- **Live-only VPS**: `python3 -m pip install -e .`
- **Backtesting / optimization / research**: `python3 -m pip install -e ".[full]"`
- **Contributing / docs / linting**: `python3 -m pip install -e ".[dev]"`

Typical live-only install:

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

## 4. Build the Rust extension

Passivbot exposes the Rust core through `passivbot_rust.*.so`. `python3 -m pip install -e .` builds it as part
of installation, but you can still rebuild manually when iterating on Rust code:

```bash
source venv/bin/activate
maturin develop --release
```

The Rust crate's default features include `abi3-py312`, so one extension build
targets the stable Python 3.12 ABI and remains loadable by newer supported
interpreters, including Python 3.14. Do not use `--no-default-features` to test
the supported installation path: that disables the ABI3 compatibility feature.

Common errors:

- `error: linker cc not found` → install build tools: `sudo apt install build-essential`. On macOS ensure Xcode CLT is installed.
- `No such command 'maturin'` → re-run `pip install -r requirements-rust.txt`.
- `failed to run custom build command … cc not found` on WSL/Ubuntu → install the compiler and
  Python headers (`sudo apt install build-essential python3-dev`).
- `failed to parse manifest ... feature edition2024 is required` or `cargo metadata ... failed` during `python3 -m pip install -e ".[full]"` → your Rust/Cargo is too old for the transitive crates Cargo resolved. Update Rust with `rustup update stable`, confirm `cargo --version` / `rustc --version`, then retry the install. If you installed Rust from distro packages, prefer the `rustup` toolchain instead.

## 5. Verify the install

```bash
pytest -q
passivbot -h
```

For backtesting and optimization environments, also verify:

```bash
passivbot backtest -h
passivbot optimize -h
```

If pytest reports missing `passivbot_rust`, double-check that the venv is active and `maturin develop --release` completed successfully.

## 6. Keeping it up to date

When pulling new commits:

```bash
source venv/bin/activate
git pull
python3 -m pip install -e .            # live-only refresh
# or: python3 -m pip install -e ".[full]"  # full research/runtime refresh
# or: python3 -m pip install -e ".[dev]"   # contributor refresh
maturin develop --release              # only when passivbot-rust changed
```

If you see linker errors after an OS update (e.g. new glibc), rebuild the extension with `maturin develop --release`.

## 7. Special environments

- **Docker / hosted containers** – Use `Dockerfile_live` for the canonical live image. It installs the compiled extension and starts through the `passivbot live` wrapper contract documented in [container_deployment.md](container_deployment.md). Railway and similar hosts should reuse that same image/env contract instead of maintaining platform-specific configs.
- **Windows** – WSL2 (Ubuntu) is the recommended route; native Windows lacks some dependency support.
- **ARM (Raspberry Pi / AWS Graviton)** – Works, but builds are slower; make sure your Rust toolchain targets the correct architecture.

## 8. Troubleshooting checklist

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: passivbot_rust…` | Activate venv or rerun `maturin develop --release`. |
| `passivbot optimize requires the full Passivbot install` | Install the full profile: `python3 -m pip install -e ".[full]"`. |
| `venv/bin/passivbot ...` works but `passivbot ...` behaves differently | Run `command -v passivbot`, then refresh shell command lookup with `hash -r` and, if your shell supports it, `rehash`. |
| `linker cc not found` / `cannot find crt1.o` | Install build-essential + `python3-dev`. |
| `rustup: command not found` | Install Rust via https://rustup.rs/. |
| `feature edition2024 is required` during the Rust build | Update Rust with `rustup update stable`, then retry `python3 -m pip install -e ".[full]"`. |
| `python3 -m pip install … failed due to SSL` | Update `certifi` or set `PIP_CERT` if corporate proxies intercept TLS. |
| `maturin develop` can’t find Python | Ensure you run it inside the venv (`which python` should point to `venv/bin/python`). |
| `TypeError: unsupported operand type(s) for |: ...` | You are running an unsupported Python version; install Python 3.12 or 3.14 and recreate the venv with that interpreter. |

For more detail, see [docs/troubleshooting.md](troubleshooting.md).

Still stuck? Open an issue with the full error log and details about your OS/architecture.
