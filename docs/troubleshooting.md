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

Activate the intended venv and reinstall the profile you use:

```bash
source venv/bin/activate
python3 -m pip install -e .
# or: python3 -m pip install -e ".[full]"
# or: python3 -m pip install -e ".[dev]"
```

If an import still loads a stale extension, follow the
[locked rebuild and verification procedure](ai/runbooks/rust_extension.md#rebuild-and-verify)
from the checkout root. It selects the crate, removes shadowing local copies, and verifies the
loaded artifact against the current Rust sources.

## Rust build fails with `feature edition2024 is required`

From the checkout root, check the active Cargo and Rust versions against `rust-toolchain.toml`.
The repository supplies a Cargo lockfile and a pinned toolchain.

```bash
command -v cargo
rustup show active-toolchain
cargo --version
rustc --version
```

If the shell resolves a distro Cargo or another override, switch to the Rustup-managed pinned
toolchain and retry the same install profile.

## `maturin develop` says it could not find a virtualenv or conda environment

Activate the environment first and verify which interpreter it selects.

```bash
source venv/bin/activate
python -m pip --version
python -c "import sys; print(sys.executable)"
```

Both paths should belong to the intended venv. Then follow the
[locked rebuild and verification procedure](ai/runbooks/rust_extension.md#rebuild-and-verify).
If necessary, recreate the venv with Python 3.12 or 3.14 and reinstall.
