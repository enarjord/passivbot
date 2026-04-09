# Troubleshooting

This page collects common setup and CLI issues that are easy to fix once you know what to check.

## `passivbot` runs the wrong command after reinstalling or switching environments

Symptom:

- `venv/bin/passivbot ...` behaves differently from `passivbot ...`
- `passivbot -h` still shows old help output after reinstalling
- `passivbot` appears to ignore changes from the current checkout

Cause:

- Your shell is resolving a different `passivbot` executable than the one inside the active virtualenv.
- This is common with `pyenv`, multiple virtualenvs, or shells that cache command paths.
- Newer Passivbot builds detect this and either re-exec into the active environment's `passivbot`
  script or fail loudly with an explicit mismatch message.

Check which command is being used:

```bash
type -a passivbot
command -v passivbot
python -c "import sys; print(sys.executable)"
```

Expected result:

- `command -v passivbot` should point to the active venv, for example:

```bash
/path/to/passivbot/venv/bin/passivbot
```

The Python executable should also point to the same virtualenv, for example:

```bash
/path/to/passivbot/venv/bin/python
```

If either command points somewhere else, refresh shell command lookup:

```bash
hash -r
```

If your shell also supports `rehash` (`zsh`, `tcsh`), run that too:

```bash
rehash
```

If that is not enough, reactivate the venv:

```bash
deactivate
source venv/bin/activate
hash -r
```

For shells with `rehash`, run it after reactivating:

```bash
rehash
```

If you use `pyenv`, also refresh its shims:

```bash
pyenv which passivbot
pyenv rehash
```

If you still need to confirm the current checkout is correct, compare:

```bash
venv/bin/passivbot optimize -h | head -40
passivbot optimize -h | head -40
```

If the first command is correct and the second is not, the problem is shell path resolution, not Passivbot itself.

## `passivbot optimize requires the full Passivbot install`

Install the full profile:

```bash
python3 -m pip install -e ".[full]"
```

## `passivbot_rust` missing or stale

Activate the venv and rebuild the Rust extension:

```bash
source venv/bin/activate
maturin develop --release
```

If you recently pulled new commits, refresh the install too:

```bash
python3 -m pip install -e .
# or: python3 -m pip install -e ".[full]"
# or: python3 -m pip install -e ".[dev]"
```

## `python3 -m pip install -e ".[full]"` fails with `feature edition2024 is required`

This is a Rust toolchain issue, not a Python dependency issue.

If the Rust build logs include errors like:

- `feature edition2024 is required`
- `cargo metadata ... failed`
- `failed to parse manifest` inside `~/.cargo/registry/...`

then your local Cargo is too old for the crates it resolved during the build.

Check your toolchain:

```bash
cargo --version
rustc --version
```

Update to the current stable Rust toolchain and retry:

```bash
rustup update stable
python3 -m pip install -e ".[full]"
```

If Rust was installed from system packages instead of `rustup`, prefer switching to the `rustup`-managed toolchain.

## `maturin develop` says it could not find a virtualenv or conda environment

This usually means the build is running outside the project venv.

Activate the venv first:

```bash
source venv/bin/activate
which python
python -m pip --version
```

Expected result:

- `which python` should point to `.../venv/bin/python`
- `python -m pip --version` should reference the same venv

Then rerun:

```bash
maturin develop --release
```

If you are still seeing the wrong interpreter, recreate the venv from the desired Python 3.12 install and reinstall Passivbot inside that venv.
