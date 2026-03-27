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

Check which command is being used:

```bash
type -a passivbot
command -v passivbot
```

Expected result:

- `command -v passivbot` should point to the active venv, for example:

```bash
/path/to/passivbot/venv/bin/passivbot
```

If it points somewhere else, refresh shell command lookup:

```bash
rehash
hash -r
```

If that is not enough, reactivate the venv:

```bash
deactivate
source venv/bin/activate
rehash
hash -r
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
pip install -e ".[full]"
```

## `passivbot_rust` missing or stale

Activate the venv and rebuild the Rust extension:

```bash
source venv/bin/activate
maturin develop --release
```

If you recently pulled new commits, refresh the install too:

```bash
pip install -e .
# or: pip install -e ".[full]"
# or: pip install -e ".[dev]"
```
