# Backtester Performance Optimization Checklist

This is the implementation checklist for speeding up the refactored Rust backtester on
`refactor/rust-strategy-runtime-plan`.

It is intentionally performance-focused. Unless an explicit correctness bug is found, every pass
must preserve backtest outputs exactly.

## Goals

- Recover performance lost during the strategy-runtime refactor.
- Keep backtest and live behavior aligned through the shared Rust orchestrator.
- Optimize the refactored architecture rather than reintroducing legacy coupling.

## Non-Negotiables

1. Correctness first.
   - Speed work must not silently change orders, fills, or metrics.
2. Measure before optimizing.
   - Every optimization pass needs a before/after timing comparison.
3. Preserve shared logic.
   - Rust orchestrator remains the source of truth for behavior.
4. Prefer structural optimizations.
   - Remove allocation, cloning, hashing, span-scanning, and repeated rebuild work before changing
     algorithms.

## Baseline Commands

Primary parity/performance command:

```bash
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/refactor_test.json
```

Secondary performance command for the new strategy seam:

```bash
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02
```

Profiler command:

```bash
PASSIVBOT_ORCH_PROFILE=1 \
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/refactor_test.json
```

Profiler artifact:

- `measurements/orch_profile_orchestrator.json`

## Measurement Protocol

For every optimization slice:

1. Record wall-clock runtime for:
   - HLCV preparation
   - backtest loop
   - analysis
2. Record orchestrator profile breakdown when relevant:
   - `clear_orders_ns`
   - `peek_hints_ns`
   - `input_update_ns`
   - `compute_ns`
   - `distribute_ns`
   - `sort_bundles_ns`
3. Re-run parity checks:
   - `configs/refactor_test.json`
   - `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
4. Compare:
   - key metrics
   - fills count
   - if needed, `fills.csv` and `analysis.json`

## Hotspot Hypotheses

These are the most likely regression sources in the refactored backtester:

1. Per-candle orchestrator input update work in `get_orchestrator_input_cached()`.
2. Repeated span-scanning and vector mutation for EMA bundle updates.
3. Dense backtest state still stored behind `HashMap<usize, ...>` access patterns.
4. Temporary vector churn inside orchestrator compute/workspace flow.
5. Materializing a fully nested orchestrator input shape for every candle even though backtest
   already owns typed dense state.

## Pass Breakdown

### Pass 0: Freeze Performance Baseline

Commit intent:

- document current runtime and profiler numbers before optimization

Tasks:

1. Run `configs/refactor_test.json` and record timings.
2. Run `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02` and record timings.
3. Run with `PASSIVBOT_ORCH_PROFILE=1` and save the current profile artifact.
4. Add a short benchmark section to this doc with the current measured results.

Verification:

- no code changes

### Pass 1: Precompute EMA Slot Indices

Commit intent:

- stop scanning EMA span vectors on every candle update

Tasks:

1. Precompute per-symbol/per-side indices for:
   - `m1.close`
   - `m1.volume`
   - `m1.log_range` for forager
   - `m1.log_range` for `ema_anchor` fast volatility
   - `h1.log_range` for slow strategy volatility
2. Replace span-matching update loops in `get_orchestrator_input_cached()` with direct index writes.
3. Keep exact payload shape unchanged.

Expected gain:

- lower `input_update_ns`

Verification:

- targeted Rust tests
- parity backtests
- profiler comparison

### Pass 2: Slim Cached Orchestrator Input Hot Path

Commit intent:

- reduce per-candle rebuilding/mutation overhead in the cached input path

Tasks:

1. Identify fields that are fully static for the run.
2. Identify fields that change only on fills or rare events.
3. Restrict per-candle updates to hot fields only:
   - prices
   - tradability
   - positions
   - runtime budget
   - trailing values
   - EMA values
   - next candle
4. Avoid redundant cloning of strategy/bot/runtime config fields.

Expected gain:

- lower `input_update_ns`
- lower total backtest runtime

Verification:

- parity backtests
- profiler comparison

### Pass 3: Convert Dense Runtime State Away From HashMaps

Commit intent:

- remove hash lookup overhead from dense symbol-indexed loops

Tasks:

1. Audit hot-loop structures currently accessed as `HashMap<usize, ...>`.
2. Convert dense always-indexed structures to `Vec`-backed storage where safe:
   - positions
   - trailing state
   - open orders if profitable
3. Keep sparse maps only where sparsity materially matters.

Expected gain:

- lower hot-loop overhead in backtest and orchestrator staging

Verification:

- targeted Rust tests
- parity backtests

### Pass 4: Reduce Orchestrator Workspace Allocation And Data Motion

Commit intent:

- make shared orchestrator compute cheaper without changing behavior

Tasks:

1. Audit repeated `Vec` growth/clear/append/drain patterns in `OrchestratorWorkspace`.
2. Reuse scratch buffers consistently.
3. Reserve capacities where order counts are predictable.
4. Avoid unnecessary movement between per-symbol buffers and aggregate buffers.

Expected gain:

- lower `compute_ns`
- lower `distribute_ns`

Verification:

- orchestrator tests
- parity backtests
- profiler comparison

### Pass 5: Add A Typed Backtest Fast Path

Commit intent:

- keep shared orchestrator logic while avoiding repeated construction of the full nested input form

Tasks:

1. Introduce a typed internal backtest adapter path feeding the shared orchestrator engine.
2. Reuse dense typed state directly instead of repeatedly materializing full nested structs where
   possible.
3. Keep external live/orchestrator JSON boundaries unchanged.

Expected gain:

- larger reduction in `input_update_ns`
- possible reduction in `compute_ns`

Verification:

- full parity protocol
- profiler comparison

### Pass 6: Strategy Hot-Path Tightening

Commit intent:

- remove leftover per-strategy waste after shared-path optimization

Tasks:

1. Audit `trailing_grid` repeated calculations that can be hoisted once per side.
2. Ensure `ema_anchor` does no extra volatility work when weights are zero.
3. Precompute side-level booleans where useful.

Expected gain:

- smaller but real `compute_ns` improvement

Verification:

- strategy regression tests
- parity backtests

### Pass 7: Python-Side Startup And Preparation Cleanup

Commit intent:

- trim any measurable Python overhead after Rust hotspots are addressed

Tasks:

1. Measure startup/config preparation cost separately from backtest loop cost.
2. Remove repeated work in Python if it shows up materially in timing.
3. Keep this strictly secondary to Rust engine optimization.

Verification:

- no behavior regressions
- timing comparison

## Reporting Format

For each optimization commit, record:

- command used
- before runtime
- after runtime
- profiler deltas if applicable
- parity result
- any residual risk

## Current Status

- Pass 0 baseline recorded on `refactor/rust-strategy-runtime-plan`
- Pass 1 partial (EMA slot indices complete)
- Pass 2 not started
- Pass 3 partial (dense backtest state complete)
- Pass 4 partial (compute-side derived cache complete; two regressing workspace experiments were
  tested and reverted)
- Pass 5 partial (backtest now injects pre-parsed strategy params directly into the cached
  orchestrator input)
- Pass 6 not started
- Pass 7 not started

## Pass 0 Baseline

Recorded on 2026-04-06 using cached HLCV data.

### `configs/refactor_test.json`

Command:

```bash
PASSIVBOT_ORCH_PROFILE=1 \
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/refactor_test.json
```

Artifact:

- `backtests/combined/2026-04-06T14_49_51`

Observed timings:

- HLCV cache load: `0.5167s`
- Backtest loop: `8.1972s`
- Analysis: `0.2162s`

Profile notes:

- Initial profiled baseline after fixing profiler output persistence:
  - artifact: `backtests/combined/2026-04-06T14_52_53`
  - HLCV cache load: `0.4040s`
  - Backtest loop: `5.6999s`
  - Analysis: `0.1501s`
  - `measurements/orch_profile_orchestrator.json`
  - `total_ns = 4_982_309_858`
  - `input_update_ns = 78_591_118`
  - `compute_ns = 4_782_081_039`
  - `distribute_ns = 38_372_487`

Notes:

- Output metrics still match the frozen refactor baseline.
- The orchestrator profiler persistence issue is fixed in this branch by creating the parent
  `measurements/` directory before writing the JSON file.

### `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`

Command:

```bash
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02
```

Artifact:

- `backtests/combined/2026-04-06T14_49_56`

Observed timings:

- HLCV cache load: `0.4076s`
- Backtest loop: `12.6168s`
- Analysis: `0.4121s`

Notes:

- This is the more expensive current benchmark and should remain part of every performance pass.
- Current verified metrics remain:
  - `gain_usd = 1.1068665907303028`
  - `sharpe_ratio_strategy_pnl_rebased = 0.03834167190620828`
  - `drawdown_worst_hsl = 0.030361878962243738`

## Pass 1 Progress

Completed so far:

- precomputed backtest EMA bundle slot indices so cached orchestrator input updates stop scanning
  EMA span vectors on every candle

Latest profiled `configs/refactor_test.json` run after the slot-index change:

- artifact: `backtests/combined/2026-04-06T15_04_24`
- HLCV cache load: `0.3832s`
- Backtest loop: `5.3829s`
- Analysis: `0.1496s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 4_672_699_380`
- `input_update_ns = 74_880_980`
- `compute_ns = 4_475_817_874`
- `distribute_ns = 37_373_764`

Current takeaway:

- the slot-index optimization produced a modest but real backtest-loop improvement
- `compute_ns` still dominates runtime by a wide margin
- next work should focus on reusing compute-side derived data rather than spending more time on
  cached input micro-optimizations

## Pass 4 Progress

Completed so far:

- cached per-symbol runtime budgets once per orchestrator call
- cached parsed strategy params and derived EMA/volatility inputs within the orchestrator workspace
- reused those caches across forager selection, one-way gating, order generation, unstuck, and
  diagnostics

Best profiled `configs/refactor_test.json` run after the compute-side cache:

- artifact: `backtests/combined/2026-04-06T15_12_09`
- HLCV cache load: `0.3792s`
- Backtest loop: `3.8375s`
- Analysis: `0.1762s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 3_108_624_204`
- `input_update_ns = 75_794_456`
- `compute_ns = 2_908_802_686`
- `distribute_ns = 39_621_318`

Observed effect versus the profiled baseline at `2026-04-06T14_52_53`:

- backtest loop improved from `5.6999s` to `3.8375s`
- `compute_ns` improved from `4_782_081_039` to `2_908_802_686`

Follow-up experiment:

- skipping backtest-only diagnostics work inside the orchestrator produced mixed/noisy results
  (`3.9572s` to `3.9973s` loop time on later runs), so it should be treated as neutral until a
  more isolated benchmark proves otherwise
- flattening `per_long` / `per_short` workspace storage from `Vec<Option<...>>` to dense vectors
  regressed or came out roughly flat on repeated runs, so it was reverted
- reusing strategy raw-order scratch vectors in the orchestrator also regressed clean single-run
  measurements, so it was reverted

Latest clean profiled `configs/refactor_test.json` run after keeping only the proven wins:

- artifact: `backtests/combined/2026-04-06T16_09_25`
- HLCV cache load: `0.3492s`
- Backtest loop: `3.4822s`
- Analysis: `0.1306s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 2_852_255_712`
- `clear_orders_ns = 5_646_566`
- `input_update_ns = 41_453_145`
- `compute_ns = 2_743_021_234`
- `distribute_ns = 9_605_834`

Observed effect versus the previous kept best run at `2026-04-06T15_49_49`:

- backtest loop improved from `3.6757s` to `3.4822s`
- `compute_ns` improved from `2_870_941_928` to `2_743_021_234`
- `input_update_ns` improved from `43_213_954` to `41_453_145`
- `distribute_ns` improved from `10_124_388` to `9_605_834`

Secondary benchmark on the current kept code:

- `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
  - artifact: `backtests/combined/2026-04-06T16_09_42`
  - HLCV cache load: `0.2691s`
  - Backtest loop: `5.8347s`
  - Analysis: `0.4074s`
  - metrics unchanged from the verified `ema_anchor` baseline

Additional follow-up kept on top of the proven wins:

- made hot-path numeric structs and strategy parameter structs `Copy`
- removed cached derived-data clones inside the orchestrator for:
  - runtime budgets
  - strategy params
  - EMA bands
  - state/order-book assembly

Latest clean profiled `configs/refactor_test.json` run after the clone-removal pass:

- artifact: `backtests/combined/2026-04-06T16_34_16`
- HLCV cache load: `0.3391s`
- Backtest loop: `3.4629s`
- Analysis: `0.1242s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 2_823_663_924`
- `clear_orders_ns = 5_586_729`
- `input_update_ns = 41_918_856`
- `compute_ns = 2_704_966_091`
- `distribute_ns = 9_550_963`

Observed effect versus the previous kept best run at `2026-04-06T16_09_25`:

- backtest loop improved from `3.4822s` to `3.4629s`
- `compute_ns` improved from `2_743_021_234` to `2_704_966_091`
- `distribute_ns` improved slightly from `9_605_834` to `9_550_963`
- `input_update_ns` was effectively flat (`41_453_145` to `41_918_856`)

Secondary benchmark on the clone-removal pass:

- `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
  - artifact: `backtests/combined/2026-04-06T16_33_57`
  - HLCV cache load: `0.2888s`
  - Backtest loop: `5.9783s`
  - Analysis: `0.4925s`
  - metrics unchanged from the verified `ema_anchor` baseline

Additional experiment that was tested and reverted:

- replacing trailing-grid `BotParams` runtime cloning with a compact runtime-params struct
  regressed clean single-run timings:
  - `configs/refactor_test.json`
    - backtest loop regressed from `3.4629s` to `3.8014s`
    - `compute_ns` regressed from `2_704_966_091` to `2_932_749_705`
  - `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
    - backtest loop regressed from `5.9783s` to `6.0321s`
  - the slice was reverted and should not be revisited without finer-grained evidence

## Pass 3 Progress

Completed:

- replaced dense backtest-side `HashMap` / `BTreeMap` state with fixed-length `Vec` storage for:
  - positions
  - trailing prices
  - open orders
- changed open-order clearing from map clears to per-bundle vector clears, preserving capacities
- replaced `did_fill_long` / `did_fill_short` `HashSet`s with dense `Vec<bool>` flags

Latest profiled `configs/refactor_test.json` run on the final dense-state code:

- artifact: `backtests/combined/2026-04-06T15_49_49`
- HLCV cache load: `0.3993s`
- Backtest loop: `3.6757s`
- Analysis: `0.1579s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 2_984_269_356`
- `clear_orders_ns = 5_723_353`

## Pass 5 Progress

Completed so far:

- added an internal typed strategy-param fast path on `SymbolSideInput`
- backtest now injects pre-parsed `StrategyParams` directly into the cached orchestrator input
- backtest stopped cloning raw `serde_json::Value` strategy payloads into every symbol/side/candle
- orchestrator still preserves the old raw-JSON parse path as fallback for non-backtest callers

Latest clean profiled `configs/refactor_test.json` run after the typed-param fast path:

- artifact: `backtests/combined/2026-04-06T20_18_30`
- HLCV cache load: `0.4395s`
- Backtest loop: `1.3508s`
- Analysis: `0.2229s`
- `measurements/orch_profile_orchestrator.json`
- `total_ns = 632_264_486`
- `clear_orders_ns = 5_853_642`
- `input_update_ns = 55_950_527`
- `compute_ns = 503_782_582`
- `distribute_ns = 10_324_786`

Observed effect versus the previous kept best run at `2026-04-06T19_07_14`:

- backtest loop improved from `3.4427s` to `1.3508s`
- `compute_ns` improved from `2_664_757_951` to `503_782_582`
- `total_ns` improved from `2_774_347_621` to `632_264_486`
- `input_update_ns` regressed from `41_694_799` to `55_950_527`, but that cost is dwarfed by the
  compute-side win

Secondary benchmark on the typed-param fast path:

- `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
  - artifact: `backtests/combined/2026-04-06T20_18_49`
  - HLCV cache load: `0.3339s`
  - Backtest loop: `3.2079s`
  - Analysis: `0.4758s`
  - metrics unchanged from the verified `ema_anchor` baseline
- `input_update_ns = 43_213_954`
- `compute_ns = 2_870_941_928`
- `distribute_ns = 10_124_388`

Observed effect versus the pre-dense-state compute-cache run at `2026-04-06T15_12_09`:

- backtest loop improved from `3.8375s` to `3.6757s`
- `clear_orders_ns` improved from `30_336_319` to `5_723_353`
- `input_update_ns` improved from `75_794_456` to `43_213_954`
- `distribute_ns` improved from `39_621_318` to `10_124_388`
- `compute_ns` improved slightly from `2_908_802_686` to `2_870_941_928`

Secondary benchmark:

- `configs/examples/ema_anchor.json -sd 2022-01-01 -ed 2026-04-02`
  - artifact: `backtests/combined/2026-04-06T15_49_52`
  - HLCV cache load: `0.3211s`
  - Backtest loop: `6.2732s`
  - Analysis: `0.5132s`
  - metrics unchanged from the verified `ema_anchor` baseline
