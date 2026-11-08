# Rust/PyO3 Extension Runbook

Use this whenever Rust changes must be exercised through Python.

## Source And Artifact Contract

The runtime should load the editable-install artifact from the active environment, not a stale
`src/passivbot_rust*.so` shadow copy. ABI3 builds may appear as `passivbot_rust.abi3.so` or inside
a package layout rather than with the interpreter-specific suffix.

`src/rust_utils.py` owns artifact discovery, source fingerprints, shadow-copy pruning, rebuild
locking, stamping, and runtime verification. Prefer repository helpers over copying extensions by
hand.

## Rebuild And Verify

With the repository environment active:

```bash
PYTHONPATH=src python - <<'PY'
from rust_utils import check_and_maybe_compile
check_and_maybe_compile(force=True)

import passivbot_rust  # import only after the rebuild/stamp check
from rust_utils import verify_loaded_runtime_extension
print(verify_loaded_runtime_extension())
PY
```

The reported runtime path must be the intended environment artifact. A source-fingerprint mismatch
means the process must restart after a successful rebuild. `check_and_maybe_compile()` uses the
repository's build lock, prunes shadow copies, and stamps the rebuilt artifact. A bare manual
`maturin` build may leave an old source stamp beside a new binary, so do not suppress verification
to make that artifact load.

The Python wheel build embeds the checkout fingerprint in the Rust extension and
packages the expected value. An installed runtime verifies those values agree and
checks both files against the Passivbot installation's wheel `RECORD`. It reads its
provenance from the installation. A wheel with missing or mismatched provenance
must be replaced by a clean build and installation.

The application tracks `passivbot-rust/Cargo.lock` and pins its Rust toolchain in
`rust-toolchain.toml`. Repository builds, package builds and CI use `--locked`: a missing or
outdated dependency lock fails instead of resolving different dependencies on another machine.
Update the lock deliberately with Cargo when changing dependencies, and commit it with the change.
The Rust source fingerprint includes the lockfile; a dependency change therefore requires a fresh
build and cannot reuse fitness from an incompatible evaluator. Platform-specific native artifact
hashes can differ even when the declared source and dependency lock match.

## Rust Tests

Default builds enable PyO3's `extension-module` feature. Plain Rust test binaries may fail to link
under that feature, so use:

```bash
cd passivbot-rust && cargo test --locked --no-default-features && cd ..
cd passivbot-rust && cargo check --locked --tests && cd ..
```

The first runs Rust tests without extension linkage; the second preserves default-feature compile
coverage.

## Python Test Modes

Some tests intentionally use a stub `passivbot_rust`; behavioral parity tests require the real
extension. Use the `require_real_passivbot_rust_module` fixture when the compiled implementation is
part of the claim. A passing stub-compatible test does not validate changed Rust behavior.

## Troubleshooting

1. Inspect the loaded module path before debugging behavior.
2. Check for shadowing local extension copies and multiple environment artifacts.
3. Restart Python after rebuilding; an imported native module cannot be hot-replaced safely.
4. If source and artifact still disagree, clean Rust output and rerun the repository helper:

   ```bash
   cd passivbot-rust && cargo clean && cd ..
   PYTHONPATH=src python -c "from rust_utils import check_and_maybe_compile; check_and_maybe_compile(force=True)"
   ```

5. Re-run `verify_loaded_runtime_extension()` and the real-extension test slice.

Do not assume that a fast `maturin` completion is wrong; trust the resolved artifact, source stamp,
and behavior verification.
