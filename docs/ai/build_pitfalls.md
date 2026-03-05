# Rust/PyO3 Build Pitfalls

Read this when Rust changes do not appear in Python runtime/tests.

## 1) Stale Extension Copy (src vs venv)

Problem: test/runtime import path may load a different compiled module copy than expected.

Guideline:

1. Rebuild extension with `maturin develop --release`.
2. Confirm which `passivbot_rust` module path Python resolves at runtime.
3. If behavior is stale, clean rebuild and re-verify import path.

## 2) `cargo test --lib` Linker Noise On PyO3

Problem: PyO3 projects often fail linking for `cargo test --lib` without Python runtime symbols.

Use instead:

```bash
cd passivbot-rust && cargo check --tests && cd ..
```

## 3) `conftest.py` Stub vs Real Module Ordering

Problem: some tests can run with stubbed `passivbot_rust`; others require real extension.

Guideline:

1. If test needs real Rust module, use `require_real_passivbot_rust_module` fixture.
2. If test is stub-compatible, ensure that is intentional and explicit.

## 4) Fast `maturin` Completion Can Be Misleading

Problem: very fast completion can indicate no meaningful rebuild.

Guideline:

```bash
cd passivbot-rust && cargo clean && maturin develop --release && cd ..
```

Use clean rebuild when behavior and binary timestamp do not match expectations.
