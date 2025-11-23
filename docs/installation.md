# Installing Passivbot

This guide collects all steps (and common pitfalls) for setting up Passivbot on a fresh machine. For quick instructions see the README; this file adds the missing detail when things go wrong.

## 1. Prerequisites

- **Python 3.12** – Earlier versions are no longer supported. On Debian/Ubuntu use `sudo apt install python3.12 python3.12-venv python3.12-dev`.
- **Rust toolchain** – Passivbot’s hot paths live in Rust. Install via [rustup](https://rustup.rs/) if `rustc --version` is not available.
- **C build tools** – Ubuntu/Debian: `sudo apt install build-essential`. macOS: Xcode command-line tools (`xcode-select --install`).
- **Virtual environment** – Strongly recommended so dependencies do not leak into the system interpreter.

## 2. Clone & create venv

```bash
# Clone the repo
 git clone https://github.com/enarjord/passivbot.git
 cd passivbot

# Create + activate Python 3.12 venv (inside repo root)
 python3.12 -m venv venv
 source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-rust.txt   # only needed when building the Rust extension yourself
```

## 4. Build the Rust extension

Passivbot exposes the Rust core through `passivbot_rust.*.so`. Build it once per environment:

```bash
source venv/bin/activate
maturin develop --release
```

Common errors:

- `error: linker cc not found` → install build tools: `sudo apt install build-essential`. On macOS ensure Xcode CLT is installed.
- `No such command 'maturin'` → re-run `pip install -r requirements-rust.txt`.
- `failed to run custom build command … cc not found` on WSL/Ubuntu → install `python3-dev` (`sudo apt install python3.12-dev`).

## 5. Verify the install

```bash
pytest -q
python3 src/passivbot.py -h
```

If pytest reports missing `passivbot_rust`, double-check that the venv is active and `maturin develop --release` completed successfully.

## 6. Keeping it up to date

When pulling new commits:

```bash
source venv/bin/activate
git pull
pip install -r requirements.txt       # only when dependencies change
maturin develop --release              # only when passivbot-rust changed
```

If you see linker errors after an OS update (e.g. new glibc), rebuild the extension with `maturin develop --release`.

## 7. Special environments

- **Docker** – Use `Dockerfile`/`Dockerfile_live` in the repo root. The Docker images already include the compiled extension.
- **Windows** – WSL2 (Ubuntu) is the recommended route; native Windows lacks some dependency support.
- **ARM (Raspberry Pi / AWS Graviton)** – Works, but builds are slower; make sure your Rust toolchain targets the correct architecture.

## 8. Troubleshooting checklist

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: passivbot_rust…` | Activate venv or rerun `maturin develop --release`. |
| `linker cc not found` / `cannot find crt1.o` | Install build-essential + `python3-dev`. |
| `rustup: command not found` | Install Rust via https://rustup.rs/. |
| `pip install … failed due to SSL` | Update `certifi` or set `PIP_CERT` if corporate proxies intercept TLS. |
| `maturin develop` can’t find Python | Ensure you run it inside the venv (`which python` should point to `venv/bin/python`). |

Still stuck? Open an issue with the full error log and details about your OS/architecture.
