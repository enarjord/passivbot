# Rust Strategy Runtime Refactor

## Status

This plan supersedes the earlier master-based draft and tracks the current
`refactor/rust-strategy-runtime-plan` branch state.

It uses the early EMA-anchor experiment only as the motivating example, not as the desired target
architecture.

## Purpose

Refactor the config schema, Rust runtime, backtester, and live bot so Passivbot can support
multiple strategies without repeating the same mistakes for every experiment:

- growing shared `BotParams` for strategy-only ideas
- embedding strategy math directly in `orchestrator.rs`
- coupling backtest sizing semantics to mutable shared runtime fields
- hardcoding optimizer/config plumbing per strategy

The immediate benchmark remains `ema_anchor`, but the design target is broader:

- strategy order generation should be swappable
- forager / coin selection should remain reusable across strategies
- TWEL/WEL, realized-loss gates, unstuck, and related enforcers should remain centralized
- live and backtest must continue to share the same Rust decision path

## Current Branch Baseline

The refactor must start from what already exists on this branch, not from the older master
snapshot that predated the HSL/config pipeline work.

Important current realities:

1. The config pipeline has already been split into staged modules under `src/config/`.
   - `schema.py` is now the canonical defaults source
   - `normalize.py`, `project.py`, and `runtime_compile.py` already exist
   - `tests/test_config_pipeline.py` and related tests already cover that pipeline

2. HSL is already merged into the canonical per-side bot config.
   - `bot.{long,short}.hsl`
   - `live.hsl_signal_mode`

3. Live and backtest already share the Rust orchestrator path, but the runtime is still
   monolithic.
   - `passivbot-rust/src/orchestrator.rs`
   - `passivbot-rust/src/backtest.rs`
   - `passivbot-rust/src/python.rs`

4. The current Rust runtime still assumes one dominant strategy shape.
   - strategy math is inline in `orchestrator.rs`
   - `BotParams` mixes strategy fields with forager/risk/unstuck/HSL fields
   - backtest mutates effective WEL inside runtime state

5. Optimizer/config plumbing still carries legacy flat-key assumptions at the adapter boundary.
   - see `src/optimization/config_adapter.py`

6. Live HSL control is still an adjacent controller concern, not a solved shared-engine concern.
   - the live loop and HSL runtime still interact in Python
   - the strategy refactor should not block on a simultaneous HSL architecture rewrite

## Phase-0 Branch Notes

These branch notes are the implementation baseline to preserve while extracting the strategy seam.

1. Config ownership already lives in the staged `src/config/` pipeline.
   - `schema.py` owns defaults
   - `normalize.py` owns migration into canonical shape
   - `project.py` owns target projection
   - `runtime_compile.py` owns runtime-only aliases
   - `validate.py` owns canonical validation

2. Live HSL ownership is still partially Python-side on this branch.
   - backtest HSL is already expressed through Rust runtime inputs
   - live still has Python-side mode control and adjacent controller behavior
   - phase 1 of the strategy refactor must preserve the current `TradingMode` contract rather than
     rewrite live HSL architecture

3. Backtest dynamic WEL currently works by mutating runtime bot params.
   - `src/backtest.py` injects `wallet_exposure_limit = -1.0` as the dynamic-WEL sentinel when no
     coin override is set
   - `passivbot-rust/src/backtest.rs` derives effective WEL from tradability and writes it back
     into per-coin `bot_params`
   - cached orchestrator input then copies that mutable WEL into symbol bot params before order
     generation
   - this behavior must be frozen first, then replaced with explicit runtime budget state in phase 2

## What The `ema_anchor` Experiment Actually Proved

The experiment showed more than "we need another strategy module."

It exposed three structural problems:

1. Strategy math needs an isolated runtime seam.
   - the EMA-anchor prototype had to be bolted into the existing orchestrator instead of plugged
     in

2. Immutable strategy config is currently mixed with mutable runtime budget state.
   - backtest currently changes effective `wallet_exposure_limit`
   - that leaks shared portfolio policy into strategy sizing semantics

3. Config and optimizer plumbing still assume strategy params live inside `bot`.
   - that is manageable for one strategy
   - it becomes a tax on every new experiment

The second point is the most important architectural lesson from the experiment.

## Goals

1. A new strategy should be implementable mostly by:
   - adding one Rust strategy module
   - registering one strategy spec
   - adding tests

2. Shared orchestration should remain centralized:
   - coin filtering / forager selection
   - one-way blocking
   - TWEL/WEL gates
   - realized-loss gates
   - unstuck and enforcers
   - order sorting / trimming
   - fill simulation

3. Strategy params must stop expanding shared `BotParams`.

4. The config system should build on the existing staged pipeline, not replace it.

5. Live and backtest must continue to use the same Rust strategy runtime.

6. Optimizer bounds and validation should be strategy-spec-driven rather than hardcoded around
   `bot.*` scalar key lists.

## Non-Goals

1. Runtime-loaded external strategy plugins.
   - strategies remain statically compiled in

2. A full live HSL architecture rewrite in the same refactor.
   - the strategy refactor must integrate with current HSL/live mode control first

3. Rewriting the fill simulator.

4. Generalizing portfolio policy beyond concrete needs discovered by real strategies.

5. Replacing the current config pipeline with another config-loader redesign.

## Recommended Target Shape

The detailed branch checklist lives in
[`docs/plans/rust_strategy_runtime_refactor_execution_plan.md`](./rust_strategy_runtime_refactor_execution_plan.md).

The schema guidance below reflects the final agreed direction for this branch and supersedes the
older top-level `strategy.long` / `strategy.short` draft.

### 1. Keep The Existing Config Pipeline, Extend It

Do not re-open the config-loader redesign.

The current `src/config/` pipeline is already the correct place to integrate strategy support:

- `schema.py`: canonical defaults
- `normalize.py`: canonical normalization and migration
- `project.py`: target projection
- `runtime_compile.py`: runtime-only aliases / compilation
- `validate.py`: canonical validation

The strategy refactor should add to that pipeline, not route around it.

### 2. Keep A Fixed Per-Side Schema Under `bot`

Keep shared orchestration and risk policy in `bot`, but group it by subsystem rather than as one
flat side config.

Keep strategy params in fixed namespaces under `bot.<side>.strategy.<strategy_kind>`.

The full reference schema may contain all supported strategy subtrees, but user-facing configs may
omit inactive strategy subtrees.

Recommended canonical shape:

```json
{
  "live": {
    "strategy_kind": "ema_anchor"
  },
  "bot": {
    "long": {
      "risk": {
        "n_positions": 3,
        "total_wallet_exposure_limit": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_threshold": 1.0,
        "we_excess_allowance_pct": 0.0
      },
      "forager": {
        "volume_ema_span": 360.0,
        "volatility_ema_span": 60.0,
        "volume_drop_pct": 0.5,
        "score_weights": {
          "volume": 0.0,
          "ema_readiness": 0.0,
          "volatility": 1.0
        }
      },
      "hsl": {
        "enabled": false,
        "red_threshold": 0.2,
        "ema_span_minutes": 60
      },
      "unstuck": {
        "close_pct": 0.05,
        "ema_dist": -0.2,
        "loss_allowance_pct": 0.01,
        "threshold": 0.4
      },
      "strategy": {
        "ema_anchor": {
          "base_qty_pct": 0.01,
          "ema_span_0": 200.0,
          "ema_span_1": 800.0,
          "offset": 0.002,
          "offset_psize_weight": 0.1
        },
        "trailing_grid": {}
      },
    },
    "short": {
      "risk": {
        "n_positions": 3,
        "total_wallet_exposure_limit": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_threshold": 1.0,
        "we_excess_allowance_pct": 0.0
      },
      "forager": {
        "volume_ema_span": 360.0,
        "volatility_ema_span": 60.0,
        "volume_drop_pct": 0.5,
        "score_weights": {
          "volume": 0.0,
          "ema_readiness": 0.0,
          "volatility": 1.0
        }
      },
      "hsl": {
        "enabled": false,
        "red_threshold": 0.2,
        "ema_span_minutes": 60
      },
      "unstuck": {
        "close_pct": 0.05,
        "ema_dist": -0.2,
        "loss_allowance_pct": 0.01,
        "threshold": 0.4
      },
      "strategy": {
        "ema_anchor": {
          "base_qty_pct": 0.01,
          "ema_span_0": 200.0,
          "ema_span_1": 800.0,
          "offset": 0.002,
          "offset_psize_weight": 0.1
        },
        "trailing_grid": {}
      },
    }
  }
}
```

Notes:

- `live.strategy_kind` is the dispatch key
- `bot.<side>` is grouped by subsystem for human readability
- `bot.<side>.strategy` has fixed strategy namespaces
- runtime uses only the subtree selected by `live.strategy_kind`
- user-facing configs and artifact configs should normally show only the active strategy subtree

### 3. Use Typed Strategy Params Internally

Do not use `HashMap<String, f64>` as the main runtime representation inside Rust.

That is acceptable only as a temporary boundary format.

Recommended rule:

1. Python may pass a generic JSON-like strategy section.
2. Rust should parse that once into typed per-strategy structs.
3. Strategy modules should operate on typed params only.

Recommended shape:

```rust
pub enum StrategyParams {
    TrailingGrid(TrailingGridParamsPair),
    EmaAnchor(EmaAnchorParamsPair),
}
```

or equivalently:

```rust
pub enum StrategyInstance {
    TrailingGrid(TrailingGridRuntime),
    EmaAnchor(EmaAnchorRuntime),
}
```

Why:

- strategy code stays type-safe
- validation errors are explicit
- per-tick lookups stay cheap
- adding parameters does not force shared struct growth

### 4. Separate Immutable Config From Runtime-Derived Budget State

This is the most important correction to the earlier draft.

Today, backtest effectively mutates `wallet_exposure_limit` as runtime state. That makes strategy
logic inherit shared allocator behavior implicitly.

Instead, split:

1. immutable configured policy
2. runtime-derived allocation state

Recommended concept split:

```rust
pub struct SharedBotParams {
    pub n_positions: usize,
    pub total_wallet_exposure_limit: f64,
    pub risk_twel_enforcer_threshold: f64,
    pub risk_wel_enforcer_threshold: f64,
    pub risk_we_excess_allowance_pct: f64,
    pub forager_volume_ema_span: f64,
    pub forager_volatility_ema_span: f64,
    pub forager_volume_drop_pct: f64,
    pub forager_score_weights: ForagerScoreWeights,
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub unstuck_threshold: f64,
    pub hsl: SharedHslConfig,
}

pub struct RuntimeBudgetState {
    pub configured_wallet_exposure_limit: f64,
    pub effective_wallet_exposure_limit: f64,
    pub effective_n_positions: usize,
}
```

The strategy contract should receive runtime budget state explicitly rather than silently reading a
mutated config field.

This is what prevents `ema_anchor` from accidentally inheriting trailing-grid sizing semantics.

### 5. Make Strategy Order Generation A Separable Layer

The orchestrator should become a coordinator over:

1. selection / activation
2. strategy proposal generation
3. shared gating / enforcement
4. post-processing / sorting

Recommended Rust layout:

- `passivbot-rust/src/engine/mod.rs`
- `passivbot-rust/src/engine/selection.rs`
- `passivbot-rust/src/engine/one_way.rs`
- `passivbot-rust/src/engine/risk.rs`
- `passivbot-rust/src/engine/postprocess.rs`
- `passivbot-rust/src/strategies/mod.rs`
- `passivbot-rust/src/strategies/spec.rs`
- `passivbot-rust/src/strategies/registry.rs`
- `passivbot-rust/src/strategies/trailing_grid.rs`
- `passivbot-rust/src/strategies/ema_anchor.rs`

The earlier draft's basic direction was correct here.

### 6. Keep HSL As A Neighboring Controller Concern In Phase 1

The shared engine boundary on this branch is not identical to the final idealized architecture.

On the current branch:

- live HSL still influences per-side mode control from Python
- backtest HSL is already integrated on the Rust side

Therefore phase 1 should treat HSL like this:

1. strategy refactor preserves the existing `TradingMode` contract into Rust
2. shared orchestrator continues to honor `Normal`, `GracefulStop`, `TpOnly`, `Panic`, `Manual`
3. live HSL internals are not rewritten as part of the first strategy refactor

That keeps scope under control without weakening the long-term design.

### 7. Use A Small, Concrete Feature Layer First

The earlier draft was right that indicator derivation should not keep growing inline in
`orchestrator.rs`.

However, the first implementation should avoid an overly generic feature framework.

Recommended first step:

- introduce a typed `StrategyInputs` / `StrategyFeatures` layer
- support only the currently needed derived inputs
- add new providers only when a real strategy needs them

Examples:

```rust
pub struct StrategyFeatures {
    pub ema_bands: Option<EMABands>,
    pub entry_volatility_logrange_ema_1h: Option<f64>,
    pub forager_volume_score: Option<f64>,
    pub forager_volatility_score: Option<f64>,
}
```

Then later, if more strategies justify it, this can grow into a more formal request/provider
registry.

This is a better flexibility tradeoff than building a large abstraction before the second and
third real strategies exist.

## Strategy Spec

Every strategy should provide a spec that is visible to Python.

Recommended responsibilities of the spec:

- parameter names
- defaults
- bounds metadata
- side mirroring behavior
- required features

Recommended sketch:

```rust
pub struct StrategySpec {
    pub kind: StrategyKind,
    pub params: &'static [ParamSpec],
    pub required_features: &'static [FeatureRequirement],
    pub legacy_bot_aliases: &'static [LegacyBotAlias],
}
```

This spec should become the source for:

- config validation
- runtime normalization
- optimizer bounds discovery
- help/docs generation where useful

## Compatibility And Migration Rules

### Branch Compatibility Policy

1. Backward compatibility is only for official `master` release transitions.
2. Do not add compatibility aliases, duplicate schema support, or branch-local shims between
   iterations of this dev branch unless explicitly requested.

### Recommended Default Strategy Naming

- canonical strategy name: `trailing_grid`
- canonical experiment strategy name: `ema_anchor`
- old branch-local names should be removed, not aliased

### Migration Behavior

Use the existing config normalization pipeline for migration:

- `normalize.py` maps canonical config into the grouped bot-side schema
- `validate.py` validates the canonical active strategy subtree
- `project.py` preserves only relevant sections for each target
- `runtime_compile.py` injects runtime-only aliases if still needed

Do not push these concerns into ad hoc code inside `backtest.py`, `passivbot.py`, or
`config_utils.py`.

## Python Boundary Changes

### Required

1. Config pipeline support for:
   - `live.strategy_kind`
   - `bot.long.strategy.<kind>`
   - `bot.short.strategy.<kind>`

2. Optimizer plumbing that can discover strategy-owned bounds from a Rust-exposed spec.

3. Generic payload building so live and backtest pass:
   - shared bot params
   - strategy kind
   - strategy params
   - runtime mode inputs

### Not Required In Phase 1

1. Per-strategy Python math.
2. A new Python-side strategy execution layer.
3. Another config subsystem rewrite.

## Backtester Changes

The backtester should stop representing strategy semantics by mutating shared bot params.

Required changes:

1. Keep immutable shared config separate from runtime budget state.
2. Pass strategy params separately into the Rust runtime.
3. Make dynamic tradability / effective WEL an engine allocation input, not a silent strategy
   config rewrite.
4. Preserve live/backtest parity by keeping strategy dispatch inside Rust.

For `ema_anchor`, this means:

- fixed clip sizing based on configured strategy semantics can remain fixed
- shared TWEL/WEL/risk gates still apply
- no custom fill engine is needed

## Recommended Implementation Phases

### Phase 0: Lock The Branch Baseline

Goal: codify the current branch assumptions before refactoring.

Tasks:

1. Add a short design note documenting:
   - current config pipeline ownership in `src/config/`
   - current HSL/live-mode ownership split
   - current mutable-WEL behavior in backtest

2. Add regression tests that freeze current adaptive-grid behavior.

### Phase 1: Extract A Strategy Seam In Rust

Goal: isolate strategy order generation without changing adaptive-grid behavior.

Tasks:

1. Add `StrategyKind`.
2. Add `strategies/` modules.
3. Move current adaptive-grid order generation behind that seam.
4. Keep shared selection, one-way logic, gates, unstuck, and sorting in the engine path.

Success criteria:

- adaptive-grid behavior is unchanged
- `orchestrator.rs` no longer owns inline strategy math

### Phase 2: Separate Immutable Shared Config From Runtime Budget State

Goal: stop mutating strategy semantics through shared runtime fields.

Tasks:

1. Introduce `SharedBotParams`.
2. Introduce runtime budget/allocation state.
3. Update backtest to compute effective WEL / effective slot counts as engine state.
4. Make strategy inputs read explicit runtime budget state.

Success criteria:

- backtest dynamic-WEL behavior is explicit engine state
- strategies no longer depend on mutated config fields

### Phase 3: Add Canonical Grouped Strategy Config Support

Goal: stop growing `bot.*` for strategy-only parameters.

Tasks:

1. Extend `src/config/schema.py` with grouped per-side strategy namespaces.
2. Add normalization and validation for `bot.<side>.strategy.<kind>`.
3. Keep saved configs and artifacts active-only by default.
4. Add tests in the current config pipeline suite.

Success criteria:

- canonical strategy params live under `bot.<side>.strategy.<kind>`
- user-facing configs do not regrow inactive strategy trees
- transform log reflects meaningful canonicalization steps

### Phase 4: Rust-Driven Strategy Specs For Python

Goal: remove hardcoded optimizer/config knowledge of strategy params.

Tasks:

1. Expose strategy specs from Rust via PyO3.
2. Use them in config validation and optimizer bounds handling.
3. Replace hardcoded `bot.*` scalar assumptions in `src/optimization/config_adapter.py`.

Success criteria:

- optimizer can discover strategy params generically
- adding a strategy no longer requires hardcoded Python bound mappings

### Phase 5: Add `ema_anchor` Properly

Goal: port the experiment onto the new seam cleanly.

Tasks:

1. Implement typed `EmaAnchorParams`.
2. Express `ema_anchor` through canonical bot-side strategy config.
3. Add backtest/live parity tests.
4. Add explicit tests for fixed strategy sizing vs shared risk gating.

Success criteria:

- no custom orchestrator hack
- no short-side "read long params ad hoc" shortcut
- no dependence on mutated WEL fields for strategy semantics

### Phase 6: Expand Feature Provider Only As Needed

Goal: keep extensibility high without speculative abstraction.

Only do this when a concrete next strategy requires more derived inputs than the initial
`StrategyFeatures` layer handles cleanly.

## Acceptance Criteria

The refactor is successful when all of the following are true:

1. A new strategy using existing shared engine behavior can be added without editing shared
   selection/risk policy code.

2. Strategy params do not require expanding shared `BotParams`.

3. Backtest and live still use the same Rust strategy runtime.

4. The config pipeline supports canonical strategy config through `src/config/`, not through
   one-off adapters.

5. Optimizer bounds can be discovered from strategy metadata rather than hardcoded bot-key lists.

6. Dynamic allocation state is explicit runtime state rather than silent mutation of strategy
   config.

7. `ema_anchor` runs as a normal strategy implementation, not as an orchestrator hack.

## Summary Recommendation

The core direction of the earlier draft was right:

- separate shared engine logic from strategy logic
- keep Rust as the source of truth for live/backtest behavior
- keep forager and risk systems reusable across strategies

But the branch-aligned version should be stricter in three places:

1. Build on the existing config pipeline instead of redesigning it again.
2. Use typed strategy params internally, not generic maps deep in Rust.
3. Separate immutable strategy config from runtime allocation state before building a large feature
   abstraction.

That is the highest-value path for future strategy experimentation on this codebase.
