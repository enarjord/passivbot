# V8 Strategy Runtime Merge Readiness Plan

This plan covers the next highest-value pass on `v8`.

The branch already contains the high-leverage v8 strategy/config/optimizer contract consolidation:
Rust-owned strategy metadata, Python adapters in `src/config/strategy_spec.py`, centralized path
resolution in `src/config/param_paths.py`, and targeted parity tests. The next pass should make the
branch clean, verifiable, and reviewable for merge. It should not start a broad split of
`src/passivbot.py`, `passivbot-rust/src/backtest.rs`, or `passivbot-rust/src/python.rs`.

## Goal

Make `v8` merge-ready by removing known hygiene failures and validating
that the current Rust/Python/config/optimizer contract works from a fresh checkout state.

Success means:

- `git diff --check origin/master...HEAD` passes.
- The installed `passivbot_rust` extension is rebuilt and stamped for this checkout.
- Rust build/test gates pass for the changed runtime surface.
- Targeted Python tests pass for config, optimizer, strategy metadata, warmup, and suite paths.
- Smoke backtests prove the canonical v8 configs still run through the current CLI.
- Any remaining merge risk is explicitly listed with evidence rather than hidden behind passing
  tests.

## Non-Goals

- Do not change order behavior, risk behavior, or backtest outputs unless validation proves a
  real blocker.
- Do not add v7 compatibility shims. V8 remains a clean break.
- Do not perform a large runtime-file split in this pass.
- Do not rewrite docs or examples beyond what is required for stale or incorrect v8 merge-readiness
  content.
- Do not treat broad `pytest` success as sufficient if the Rust extension freshness check or
  backtest smoke path has not been exercised.

## Historical Known Evidence

- Previous reviewed branch tip: `15ac8094 Fix v8 full suite test expectations`.
- At that review point, `HEAD` matched `origin/v8` after fetch.
- `git diff --check origin/master...HEAD` passed at that reviewed tip; the older
  `src/config/param_paths.py` blank-EOF hygiene blocker was resolved there.
- The installed `passivbot_rust` extension must still be rebuilt for this checkout before Python
  tests that read Rust-owned strategy metadata are meaningful.
- Latest review smoke after rebuilding the Rust extension verified:
  - `passivbot_rust.get_strategy_kinds()` returned `("trailing_martingale", "ema_anchor")`.
  - `tests/test_config_pipeline.py`, `tests/test_config_utils_helpers.py`, and
    `tests/test_passivbot_version.py` passed.
- This section is historical evidence, not a current merge-readiness certification. Before merging,
  refresh `origin/v8` and re-run the primary contract suite below because the branch may have
  advanced since both the `15ac8094` smoke and the earlier `229 passed, 14 warnings`
  contract-test snapshot.

## Pass Sequence

### 1. Refresh And Establish The Review Baseline

Purpose: avoid reviewing stale local state.

Commands:

```bash
git fetch origin v8
git status --short --branch
git rev-parse HEAD
git rev-parse origin/v8
git log --oneline -n 10
```

Acceptance:

- Local branch is still `v8`.
- `HEAD` matches `origin/v8`, or any local-only commits are intentional
  and listed in the final report.
- Untracked files are classified as unrelated, planned, or blockers.

Stop conditions:

- If the remote advanced, inspect the new delta before applying this plan.
- If unrelated user changes touch files needed by this pass, preserve them and work around them.

### 2. Fix Mechanical Hygiene

Purpose: remove the known confirmed merge blocker before spending time on heavier validation.

Tasks:

1. Confirm the previously reported blank trailing EOF line in `src/config/param_paths.py` remains fixed.
2. Run the whitespace/check gate.
3. Inspect the diff to confirm no unrelated formatting churn.

Commands:

```bash
git diff --check origin/master...HEAD
git diff -- src/config/param_paths.py
```

Acceptance:

- `git diff --check origin/master...HEAD` exits cleanly.
- No code edit is needed if the EOF hygiene issue remains fixed; otherwise fix only mechanical hygiene failures.

Stop conditions:

- If `git diff --check` reports more files, fix only mechanical issues and list each touched file.

### 3. Rebuild And Verify The Rust Extension Freshness

Purpose: make Python tests meaningful. This branch exposes Rust strategy metadata to Python, so a
stale extension can make collection fail or hide real failures.

Preferred command:

```bash
./venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import rust_utils; rust_utils.check_and_maybe_compile(force=True)"
```

Verification commands:

```bash
./venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
import passivbot_rust as pbr
from rust_utils import verify_loaded_runtime_extension
print(pbr.__file__)
print(tuple(pbr.get_strategy_kinds()))
print(verify_loaded_runtime_extension())
PY
```

Acceptance:

- `passivbot_rust.get_strategy_kinds()` exists and returns the expected strategy kinds.
- `verify_loaded_runtime_extension()` does not report a source fingerprint mismatch.
- The loaded module path is the expected editable package for this checkout.

Stop conditions:

- If the rebuild appears too fast or the loaded module is still stale, use a clean rebuild per
  `docs/ai/build_pitfalls.md`:

```bash
cd passivbot-rust && cargo clean && maturin develop --release && cd ..
```

### 4. Run Rust Gates

Purpose: catch Rust compile/test regressions before Python integration tests.

Commands:

```bash
cd passivbot-rust && cargo check --tests && cd ..
cd passivbot-rust && cargo test --no-default-features && cd ..
```

Acceptance:

- Both commands pass.
- Any failures are triaged as actual branch blockers or known environment issues with exact error
  text.

Stop conditions:

- If `cargo test --no-default-features` fails due to PyO3 linkage, do not paper over it. Confirm it
  is the known linkage class before deciding whether `cargo check --tests` plus Python integration
  tests are sufficient for this branch.

### 5. Run Targeted Python Contract Tests

Purpose: validate the actual surfaces touched by the strategy runtime branch.

Primary targeted suite:

```bash
./venv/bin/python -m pytest \
  tests/test_config_pipeline.py \
  tests/optimization/test_config_adapter.py \
  tests/optimization/test_optimize.py \
  tests/test_suite_runner.py \
  tests/optimization/test_optimizer_warmup.py \
  tests/test_shared_bot.py
```

Add these if files changed during the pass or if the prior run is stale:

```bash
./venv/bin/python -m pytest \
  tests/test_config_utils_helpers.py \
  tests/test_format_config.py \
  tests/test_orchestrator_json_api.py \
  tests/test_orchestrator_integration.py \
  tests/test_backtest_analysis.py
```

Acceptance:

- The primary targeted suite passes.
- Any added test slice for touched files passes.
- Warnings are summarized only if they are new, related, or indicate a future failure.

Stop conditions:

- If tests fail during collection with Rust metadata errors, return to the freshness step.
- If tests fail after a fresh extension, fix the smallest branch-local cause and add or update the
  targeted regression if coverage is missing.

### 6. Run CLI And Backtest Smoke Checks

Purpose: prove that the branch works through the user-facing CLI, not only helper tests.

Fast config-flow smoke:

```bash
./venv/bin/passivbot backtest configs/refactor_test_v8.json
```

Strategy example smoke:

```bash
./venv/bin/passivbot backtest configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2022-01-07
```

Optional short review smoke if runtime/order logic changed during the pass:

```bash
./venv/bin/passivbot backtest configs/examples/default_trailing_martingale_long_npos4.json -sd 2025-01-01 -ed 2025-01-08
```

Acceptance:

- Each selected smoke completes without Rust freshness, config normalization, optimizer-bound, or
  strategy metadata errors.
- Generated artifacts are ignored unless the user explicitly wants them committed.
- If a smoke output differs from an accepted baseline, identify the first behavioral divergence
  before proceeding.

Stop conditions:

- If backtest data/cache availability blocks these commands, record the exact missing dependency
  and run the shortest available equivalent that exercises `prepare_config`, runtime compilation,
  and Rust backtest entrypoints.

### 7. Final Review Sweep

Purpose: catch merge blockers that targeted tests do not prove.

Commands:

```bash
git status --short --branch
git diff --stat origin/master...HEAD
git diff --check origin/master...HEAD
rg -n "except Exception|return_exceptions=True|\\.get\\([^\\n]*,\\s*(0|0\\.0|None|False|\\{\\}|\\[\\])\\)" src tests
```

Manual checks:

- Confirm `CHANGELOG.md` describes the user-visible v8 branch delta from `master`, not the internal
  sequence of branch iterations.
- Confirm docs and examples use canonical v8 selectors and paths:
  - `trailing_martingale`
  - `ema_anchor`
  - `bot.<side>.strategy.<kind>`
  - dotted optimizer selectors such as `long.strategy`, not old flattened selector docs.
- Confirm no new fallback behavior was introduced without explicit visibility and tests.
- Confirm `position_side` / `side` terminology remains correct in any touched review notes or code.
- Confirm no generated artifacts, logs, monitor output, caches, or local configs are accidentally
  staged.

Acceptance:

- No hygiene failures.
- No unexplained risky silent-handling patterns in touched areas.
- Review report lists exact commands and results.
- Remaining unrelated untracked files are explicitly excluded from the pass.

## Triage Rules

Use this order when deciding what to fix:

1. Confirmed failing gate on current branch.
2. Stale Rust extension or broken source stamp.
3. Regression in the v8 strategy/config/optimizer contract.
4. Backtest smoke failure or first behavioral divergence.
5. Stale docs/examples that would mislead users after merge.
6. Large structural cleanup.

Do not start item 6 in this pass unless items 1-5 are complete and the user explicitly asks to keep
working beyond merge readiness.

## Deliverables

At the end of the pass, provide:

- Files changed.
- Hygiene result.
- Rust rebuild/freshness result.
- Rust gate results.
- Python targeted test results.
- Backtest smoke results.
- Remaining merge risks, if any.
- Clear merge-readiness judgment.

## Definition Of Done

The pass is complete when:

- The previously reported `param_paths.py` EOF issue remains fixed.
- All selected commands above are run or explicitly marked blocked with exact blocker evidence.
- No validation step relies on a stale Rust extension.
- The final report can say either "merge-ready" or "not merge-ready because ..." with concrete
  file, command, or artifact evidence.
