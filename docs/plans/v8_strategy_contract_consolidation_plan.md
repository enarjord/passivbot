# V8 Strategy Contract Consolidation Plan

This plan covers the highest-value follow-up work on `v8`: make the v8
strategy/config/optimizer contract single-sourced, remove duplicated path-resolution logic, and add
tests that prove the contract survives the full config and optimizer pipeline.

It is an implementation plan, not a historical note. The target is behavior-preserving cleanup with
stronger invariants.

## Goal

Reduce the current drift risk around v8 strategy metadata and grouped bot config aliases.

The branch now has several places describing the same concepts:

- Python strategy params and defaults in `src/config/strategy.py`
- Python optimizer bounds in `src/config/optimize_bounds.py`
- Rust strategy specs in `passivbot-rust/src/strategies/spec.rs`
- optimizer key/path resolution in `src/optimization/config_adapter.py`
- override and fine-tune selector rewrite logic in `src/optimize.py`
- runtime compilation and grouped/flat shared-bot aliasing in `src/config/runtime_compile.py` and
  `src/config/shared_bot.py`
- suite scenario override resolution in `src/suite_runner.py`

The implementation should make one path responsible for each contract and make tests fail loudly if
the Rust, config, and optimizer views diverge.

## Non-Goals

- Do not change order behavior, backtest results, or live trading decisions.
- Do not add v7 compatibility shims. V8 is a clean break.
- Do not refactor the Rust simulation hot loop as part of this work.
- Do not split `src/passivbot.py`, `passivbot-rust/src/backtest.rs`, or
  `passivbot-rust/src/python.rs` in the same patch unless the split is strictly needed for the
  contract consolidation.

## 1. Single-Source Strategy Metadata

### Current Problem

Rust already exposes a strategy spec through `passivbot_rust.get_strategy_spec(strategy_kind)`, and
the optimizer adapter consumes it. But Python still has independent constants for:

- supported strategy kinds
- strategy parameter keys
- strategy defaults
- strategy optimize bounds

That means adding or renaming one strategy field requires editing multiple files correctly. The
recent grouped/flat alias bug was not caused directly by strategy metadata duplication, but it is
the same class of problem: several pipeline stages each knew part of the config contract and one
stage reintroduced stale alias state.

### Target Contract

Rust owns the strategy parameter contract:

- strategy kind names
- parameter names
- canonical config paths
- default values
- optimizer keys
- optimizer default bounds

Python may cache or adapt that metadata, but it should not maintain a second authoritative table for
strategy fields.

Shared bot groups are separate from strategy metadata. They remain Python-owned unless/until Rust
needs a spec for them:

- `risk`
- `forager`
- `hsl`
- `unstuck`

### Implementation Steps

1. Add a Python strategy metadata adapter module.
   - Suggested path: `src/config/strategy_spec.py`
   - Responsibilities:
     - call `passivbot_rust.get_strategy_spec(strategy_kind)`
     - call a Rust-owned strategy-kind discovery API if one exists, or keep only a tiny Python
       bootstrap list until that API is added
     - normalize supported strategy kind names
     - expose `get_strategy_param_keys(kind)`
     - expose `get_strategy_defaults(kind)` in the nested Python config shape
     - expose `get_strategy_optimize_bounds(kind)` in the nested `optimize.bounds` shape
     - expose `strategy_optimize_key_path_map(kind)` for optimizer keys

2. Move strategy defaults and strategy bounds consumers onto the adapter.
   - `src/config/strategy.py`
     - keep config-shape helpers such as `sync_canonical_strategy_config()`,
       `build_runtime_strategy_side()`, and `merge_runtime_bot_side()`
     - remove or demote hardcoded `STRATEGY_DEFAULTS_BY_KIND` and
       `STRATEGY_PARAM_KEYS_BY_KIND`
   - `src/config/optimize_bounds.py`
     - keep shared bot bounds there
     - source strategy bounds from the adapter instead of hardcoded
       `STRATEGY_OPTIMIZE_BOUNDS_DEFAULTS`

3. Add a Rust extension freshness guard for this path.
   - If `passivbot_rust.get_strategy_spec()` is unavailable or stale, the metadata request should
     fail loudly with an actionable rebuild message.
   - The failure should occur when strategy metadata is requested, not at arbitrary module import
     time.
   - Do not silently fall back to duplicated Python constants. That would preserve the drift risk.

4. Decide whether Python should import Rust metadata during module import.
   - Prefer lazy calls/cached functions over import-time calls.
   - Config modules are imported by many tools; lazy lookup gives clearer error locations and avoids
     surprising import failures before config work begins.

5. Avoid import cycles.
   - `strategy_spec.py` should be low-level and lazy.
   - If `src/config/strategy.py` imports the adapter, the adapter must not import
     `src/config/strategy.py` back.
   - Put shared bootstrap constants or normalization helpers in one direction only.

### Acceptance Criteria

- No independent Python table remains for strategy defaults or strategy optimize bounds.
- Supported strategy kind discovery is Rust-owned, or the remaining Python list is explicitly
  documented as a temporary bootstrap list rather than a second metadata source of truth.
- `get_template_config()` still produces canonical grouped v8 config.
- `get_optimize_bounds_defaults()` still produces the same nested bounds shape.
- Existing configs and shipped examples load without output changes, aside from transform-log text
  if line wording changes.
- Rust/Python metadata mismatch cannot pass silently.

## 2. Centralized Optimizer Key And Config Path Resolution

### Current Problem

Several places resolve or rewrite the same parameter paths:

- `_apply_config_overrides()` rewrites `bot.<side>.<flat_key>` to grouped shared-bot paths or active
  strategy paths.
- `optimization.config_adapter.resolve_optimization_bound_path()` maps optimizer keys to config
  paths.
- `apply_fine_tune_bounds()` normalizes dotted selectors and matches them against bound paths.
- optimizer warmup code has its own override path helper.
- suite runner scenario overrides have their own override path helper.
- `shared_bot.py` owns flat/grouped shared-bot alias maps.

This fragmentation makes regressions likely. A new key can work in one path but fail in another,
or a grouped write can be overwritten by a stale flat runtime alias during re-flattening.

### Target Contract

There should be one resolver for optimizer-facing keys and dotted selectors:

```text
optimizer key      -> canonical config path
optimizer flat key -> canonical config path
dotted selector    -> set of optimizer keys / canonical config paths
runtime flat alias -> generated only at runtime boundary
```

Here, "optimizer flat key" means existing optimizer-facing keys such as `long_n_positions` and
`long_entry_threshold_base_pct`. It does not permit v7 strategy compatibility aliases or removed
strategy fields. V8 remains a clean break.

The resolver must understand both domains:

- shared bot grouped paths, via `src/config/shared_bot.py`
- active strategy paths, via the Rust strategy spec adapter

### Implementation Steps

1. Create a central resolver module.
   - Suggested path: `src/config/param_paths.py`
   - Responsibilities:
     - `resolve_optimizer_key_path(config, key) -> tuple[str, ...] | None`
     - `resolve_dotted_config_path(config, selector_or_path) -> tuple[str, ...] | None`
     - `resolve_bound_selectors(config, selectors, flat_bounds) -> dict[str, tuple[str, ...]]`
     - `canonical_path_for_bot_side_flat_key(config, pside, flat_key)`
     - shared helper for `long.*` / `short.*` aliases and one-segment wildcard selectors

2. Keep shared-bot alias ownership in `shared_bot.py`.
   - `param_paths.py` should call `canonical_shared_bot_path_for_flat_key()` and
     `resolve_shared_bot_path()`.
   - Do not duplicate `BOT_GROUP_FIELD_MAP`.

3. Keep strategy ownership in the strategy spec adapter.
   - `param_paths.py` should call the adapter for strategy optimizer key/path maps.
   - Do not inspect strategy field names manually outside the adapter.

4. Replace callers incrementally.
   - `src/optimization/config_adapter.py`
     - replace `_strategy_path_map()` and `resolve_optimization_bound_path()` internals with the
       central resolver
   - `src/optimize.py::_apply_config_overrides()`
     - use the central resolver to resolve override destinations
     - after writing grouped/canonical values, refresh runtime flat aliases from grouped values
   - `src/optimize.py::apply_fine_tune_bounds()`
     - move selector normalization/matching into the resolver
   - `src/optimization/warmup.py`
     - replace local path helper with the central resolver
   - `src/suite_runner.py`
     - replace scenario override path helper with the central resolver

5. Make unsupported paths loud where they affect optimization.
   - For user-provided optimize bounds, unknown keys should raise.
   - For fine-tune selectors, keep the existing warning behavior for no matches unless the user has
     explicitly requested stricter behavior.
   - For internal override paths, prefer raising with the unresolved path and source context.

### Acceptance Criteria

- Only one module knows how to map optimizer keys to canonical config paths.
- `BOT_GROUP_FIELD_MAP` is not duplicated outside `shared_bot.py`.
- Strategy field names and paths come from Rust strategy metadata.
- A grouped override cannot be overwritten by an existing stale flat alias during optimizer config
  assembly.
- Suite scenario overrides and optimizer overrides resolve through the same canonical path logic.
- Dotted selectors still support:
  - `long.*`
  - `short.*`
  - `long.strategy`
  - `long.strategy.close`
  - `*.strategy.close`
  - full `bot.long.strategy.<kind>...` paths

## 3. Parity And Round-Trip Tests

### Current Problem

There are good targeted tests already, but the important invariant is broader than any one helper:

```text
Rust strategy spec
  -> Python template defaults
  -> normalize_config
  -> compile_runtime_config
  -> optimizer bounds/key paths
  -> individual_to_config / override application
  -> canonical grouped config + correct runtime aliases
```

The tests should prove this pipeline, not only individual helpers.

### Test Plan

Add focused tests before or alongside refactoring so behavior is pinned.

Suggested files:

- `tests/optimization/test_config_adapter.py`
- `tests/optimization/test_optimize.py`
- `tests/test_config_pipeline.py`
- `tests/test_shared_bot.py`

### Required Test Cases

1. Rust strategy spec matches Python template strategy defaults.
   - For each supported strategy kind:
     - read `passivbot_rust.get_strategy_spec(kind)`
     - prepare a template config for that kind
     - assert `bot.long.strategy.<kind>` and `bot.short.strategy.<kind>` contain exactly the spec
       defaults, in nested config shape

2. Rust strategy spec matches generated Python strategy optimize bounds.
   - For each supported strategy kind:
     - read `spec["optimize_bounds"]`
     - read `get_optimize_bounds_defaults()`
     - flatten only the active strategy subtree
     - assert same keys and values

3. Optimizer key path map is exhaustive for strategy keys.
   - For each `spec["parameters"]` item:
     - `optimize_key` resolves to `bot.<side>.strategy.<kind>.<path>`
     - the resolved path exists in a prepared config
     - the path points to a scalar numeric value

4. Shared bot grouped override survives runtime alias refresh.
   - Start with a config containing both grouped and stale flat aliases.
   - Apply an override like:
     - `bot.long.hsl_no_restart_drawdown_threshold = 1.0`
     - `bot.long.risk.entry_cooldown_minutes = 2.5`
   - Assert grouped canonical values and flat runtime aliases both equal the new value.
   - Repeat through suite scenario override application, not only optimizer override application.

5. Strategy override survives optimizer round trip.
   - Use one trailing martingale nested key and one ema_anchor flat optimize key.
   - Build an individual from config, mutate one value, run `individual_to_config()`.
   - Assert the value lands under the active strategy subtree and no inactive strategy subtree is
     reintroduced.

6. Dotted fine-tune selectors resolve only intended bounds.
   - `long.strategy.close` matches close params for the active long strategy only.
   - `*.strategy.close` matches close params on both sides.
   - `long.strategy` matches all long active-strategy params.
   - A substring-like selector must not match accidentally.

7. Stale extension / missing spec fails loudly.
   - Monkeypatch the adapter to simulate missing `get_strategy_spec`.
   - Config preparation or optimizer shape construction should raise an actionable error.
   - Do not fall back to embedded Python strategy defaults.

### Verification Commands

Minimum targeted checks:

```bash
./venv/bin/python -m pytest \
  tests/test_shared_bot.py \
  tests/test_config_pipeline.py \
  tests/test_config_utils_helpers.py \
  tests/optimization/test_config_adapter.py \
  tests/optimization/test_optimize.py \
  tests/optimization/test_optimizer_warmup.py -q
```

Rust boundary check:

```bash
cd passivbot-rust && cargo check --tests
```

If Rust metadata code changes:

```bash
cd passivbot-rust && maturin develop --release && cd ..
./venv/bin/python -m pytest tests/optimization/test_config_adapter.py tests/test_config_pipeline.py -q
```

Backtest smoke, only if runtime compilation or Rust strategy parsing changes:

```bash
./venv/bin/passivbot backtest configs/refactor_test_v8.json
```

## Implementation Order

### Phase 0: Pin Current Behavior

- Add parity tests that compare Rust strategy spec defaults/bounds against current Python output.
- Add round-trip tests around grouped shared-bot overrides and active strategy overrides.
- Run the targeted test command.

This phase should not change production code except test-only helpers if needed.

### Phase 1: Introduce Strategy Spec Adapter

- Add `src/config/strategy_spec.py`.
- Implement cached lazy reads from `passivbot_rust.get_strategy_spec`.
- Add a Rust-owned strategy-kind discovery API if practical; otherwise leave a clearly documented
  temporary Python bootstrap list for supported kinds only.
- Convert Rust flat `spec["defaults"]` into nested Python strategy config shape.
- Convert Rust `spec["optimize_bounds"]` into nested `optimize.bounds.<side>.strategy.<kind>`
  shape.
- Keep `src/config/strategy.py` public helper names stable for callers.
- Keep imports acyclic: the adapter may be used by `strategy.py`, but it must not depend on
  `strategy.py`.
- Re-run Phase 0 tests.

### Phase 2: Remove Duplicate Strategy Tables

- Remove Python hardcoded strategy defaults and strategy bounds.
- Update `get_all_strategy_defaults()`, `get_strategy_defaults()`, and
  `get_optimize_bounds_defaults()` to use the adapter.
- Keep supported strategy kind normalization explicit, but source the valid set from Rust metadata
  where practical.
- Re-run Phase 0 tests.

### Phase 3: Centralize Path Resolution

- Add `src/config/param_paths.py`.
- Port strategy optimizer key mapping from `optimization.config_adapter`.
- Port dotted selector normalization from `optimize.apply_fine_tune_bounds()`.
- Route shared-bot flat aliases through `shared_bot.py`.
- Replace callers one at a time with tests after each caller:
  - `optimization.config_adapter`
  - `optimization.warmup`
  - `optimize._apply_config_overrides`
  - `optimize.apply_fine_tune_bounds`
  - `suite_runner` scenario overrides

### Phase 4: Tighten Failure Modes

- Make unknown optimizer bound keys raise from one place.
- Add actionable error text for stale/missing Rust strategy metadata.
- Ensure stale/missing metadata failures are lazy and contextual, not import-time failures for
  unrelated tooling.
- Keep no-match fine-tune selectors as warnings unless stricter CLI behavior is explicitly desired.
- Re-run targeted tests and `cargo check --tests`.

### Phase 5: Cleanup And Documentation

- Remove dead helper functions and duplicated constants.
- Update `docs/ai/features/strategy_runtime.md` only if the user/agent-facing contract changed.
- No `CHANGELOG.md` entry is needed if this remains behavior-preserving internal cleanup.
- If behavior or user-visible error messages change materially, add an Unreleased changelog note.

## Risks

- Importing Rust metadata too early can make unrelated config imports fail before the user sees a
  useful command-context error. Prefer lazy cached access.
- Introducing `strategy_spec.py` can create import cycles if normalization helpers are split poorly.
  Keep the dependency direction explicit before moving callers.
- Rebuilding nested bounds from flat spec data can accidentally change ordering. Tests should assert
  key order only where order is a real output contract.
- Runtime flat aliases are still needed at engine boundaries. The cleanup should eliminate stale
  alias authority, not remove runtime aliases entirely.
- If raw flat precedence remains necessary anywhere, that caller must name the exception explicitly
  and have a regression test. Otherwise grouped canonical values win.
- `passivbot-rust/src/python.rs` currently contains both strategy spec API and much broader PyO3
  bindings. Splitting it is valuable later, but doing it during this work increases review noise.

## Definition Of Done

- Strategy defaults and strategy optimize bounds have one source of truth.
- All optimizer key/path resolution flows through one resolver.
- Grouped shared-bot canonical values win over stale flat aliases unless a caller explicitly asks
  for raw flat override precedence.
- Any raw flat override precedence exception is named in code and covered by a targeted regression
  test.
- Suite scenario overrides use the same resolver as optimizer overrides.
- Tests cover Rust spec parity, config hydration, runtime compilation, optimizer path resolution,
  fine-tune selectors, and optimizer round trips.
- Targeted Python tests and `cargo check --tests` pass.
