# Rust Strategy Runtime Refactor

## Purpose

Define a durable Rust-side structure that makes it cheap to add, backtest, optimize, and ship new
strategies without repeatedly expanding shared bot structs, duplicating orchestration logic, or
adding custom Python plumbing for every experiment.

The immediate case study is `simple_ema_mm`, but the design target is broader:

- new strategies should be able to use existing indicators and shared risk logic with minimal code
- strategy-specific params should not require shared `BotParams` growth
- live and backtest should continue to use the same Rust decision path
- adding a new indicator should be a one-time feature-provider change, not a full plumbing rewrite

## Current Pain Points

The current Rust shape is good for one dominant strategy, but too coupled for multi-strategy work.

Observed pain points:

1. Strategy logic is embedded inside shared orchestrator flow.
   - `StrategyKind` already exists in [passivbot-rust/src/types.rs](/tmp/passivbot-master-spec/passivbot-rust/src/types.rs)
   - strategy branching currently happens directly inside [passivbot-rust/src/orchestrator.rs](/tmp/passivbot-master-spec/passivbot-rust/src/orchestrator.rs)

2. Strategy-specific param adaptation is ad hoc.
   - `simple_ema_mm` currently needs a custom adapter in orchestrator to reinterpret existing config

3. Shared and strategy-specific params are mixed together.
   - every new experiment pressures `BotParams` and Python config plumbing

4. Indicator derivation is hardcoded in the orchestrator.
   - EMA bands and volatility EMA derivation are pulled inline during order generation

5. Backtest/live parity is good, but extensibility is poor.
   - the decision path is unified
   - the seams for plugging in new strategy behavior are not

6. Shared portfolio logic and strategy sizing semantics can conflict.
   - `simple_ema_mm` exposed this with configured `total_wallet_exposure_limit / n_positions`
     semantics versus dynamic backtest WEL behavior

## Goals

1. New strategies should be implementable mostly by adding one Rust module and one strategy spec.
2. Shared risk, one-way blocking, fill simulation, and diagnostics must remain centralized.
3. Strategy-specific params must stop bloating shared core structs.
4. Python should not need custom per-strategy payload plumbing beyond:
   - strategy kind
   - strategy params blob
   - features requested by the strategy
5. Live and backtest must use the same Rust strategy runtime.
6. New indicators should be add-on feature-provider work, not orchestrator surgery.

## Non-Goals

1. Runtime-loaded external plugins.
   - keep strategy registration static and compiled in
2. Rewriting the backtest fill simulator.
3. Replacing existing risk/TWEL/WEL/unstuck systems in phase 1.
4. Generalizing everything up front to support arbitrary portfolio policies.
   - support shared policy first
   - allow custom portfolio policy later only if needed

## Recommended Target Shape

Split Rust into three layers:

1. Shared engine layer
2. Strategy runtime layer
3. Feature provider layer

### 1. Shared Engine Layer

This layer remains the Rust source of truth for behavior shared across strategies.

Responsibilities:

- active symbol selection / forager policy
- one-way blocking
- side enablement and trading-mode interpretation
- TWEL/WEL enforcement
- realized-loss gate
- panic / auto-reduce / unstuck actions
- min-cost / dust guards
- order trimming, sorting, diagnostics
- backtest fill simulation

This layer should not contain strategy-specific pricing or sizing formulas.

Suggested new files:

- `passivbot-rust/src/engine/mod.rs`
- `passivbot-rust/src/engine/selection.rs`
- `passivbot-rust/src/engine/one_way.rs`
- `passivbot-rust/src/engine/risk.rs`
- `passivbot-rust/src/engine/order_postprocess.rs`

The existing logic in `orchestrator.rs` should be gradually moved into these modules, then
`orchestrator.rs` becomes a coordinator rather than the place where everything lives.

### 2. Strategy Runtime Layer

Each strategy gets an isolated module implementing one shared contract.

Recommended interface:

```rust
pub trait StrategyRuntime {
    fn kind(&self) -> StrategyKind;
    fn spec(&self) -> &'static StrategySpec;
    fn generate_orders(&self, ctx: &StrategyContext) -> Result<StrategyProposal, StrategyError>;
}
```

Use static dispatch through a registry/enum, not dynamic plugin loading.

Recommended files:

- `passivbot-rust/src/strategies/mod.rs`
- `passivbot-rust/src/strategies/registry.rs`
- `passivbot-rust/src/strategies/spec.rs`
- `passivbot-rust/src/strategies/adaptive_grid.rs`
- `passivbot-rust/src/strategies/simple_ema_mm.rs`

### 3. Feature Provider Layer

Strategies should declare what they need; the engine should provide it.

Recommended files:

- `passivbot-rust/src/features/mod.rs`
- `passivbot-rust/src/features/requests.rs`
- `passivbot-rust/src/features/provider.rs`
- `passivbot-rust/src/features/ema.rs`
- `passivbot-rust/src/features/volatility.rs`
- `passivbot-rust/src/features/account.rs`

This layer owns feature derivation from raw inputs already available to the orchestrator/backtester.

## Core Rust Types

### StrategyKind

Keep `StrategyKind` in `types.rs`, but make it only a dispatch key, not the thing that forces
branching all over orchestrator internals.

### SharedBotParams

Split the current monolithic `BotParams` into:

1. shared engine params
2. strategy params

Recommended shape:

```rust
pub struct SharedBotParams {
    pub total_wallet_exposure_limit: f64,
    pub n_positions: usize,
    pub wallet_exposure_limit: f64,
    pub risk_twel_enforcer_threshold: f64,
    pub risk_wel_enforcer_threshold: f64,
    pub risk_we_excess_allowance_pct: f64,
    pub filter_volume_ema_span: f64,
    pub filter_volume_drop_pct: f64,
    pub filter_volatility_ema_span: f64,
    pub filter_volatility_drop_pct: f64,
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub unstuck_threshold: f64,
    pub close_grid_qty_pct: f64,
    pub close_grid_markup_start: f64,
    pub close_grid_markup_end: f64,
    pub close_trailing_*: ...,
}
```

Only keep fields here if they are truly shared engine/risk/close policy concepts.

### StrategyParams

Do not create a new typed Rust struct for every experiment at the Python boundary.

Recommended phase-1 shape:

```rust
pub type StrategyParams = std::collections::HashMap<String, f64>;
```

Recommended phase-2 shape:

```rust
pub enum ParamValue {
    F64(f64),
    Bool(bool),
    String(String),
}

pub type StrategyParams = HashMap<String, ParamValue>;
```

Why this is the right tradeoff:

- adding a new strategy param does not require editing `BotParams`
- Python config/optimizer plumbing stays generic
- strategy modules can parse only what they need
- shared risk code stays typed

### StrategySpec

Every strategy must provide a spec.

Recommended:

```rust
pub struct StrategySpec {
    pub kind: StrategyKind,
    pub params: &'static [ParamSpec],
    pub feature_requests: &'static [FeatureRequestTemplate],
    pub side_mode: StrategySideMode,
    pub portfolio_policy: PortfolioPolicyKind,
}

pub struct ParamSpec {
    pub name: &'static str,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub side_behavior: ParamSideBehavior,
}
```

This spec becomes the single source of truth for:

- config defaults
- config validation
- optimizer bounds generation
- UI/help text
- mirrored long/short behavior

### StrategyContext

Strategies should not receive raw `SymbolInput` or pull from orchestrator internals directly.

Recommended:

```rust
pub struct StrategyContext<'a> {
    pub symbol_idx: usize,
    pub pside: PositionSide,
    pub shared: &'a SharedBotParams,
    pub strategy_params: &'a StrategyParams,
    pub exchange: &'a ExchangeParams,
    pub order_book: &'a OrderBook,
    pub position: &'a Position,
    pub mode: TradingMode,
    pub balance: f64,
    pub balance_raw: f64,
    pub features: StrategyFeatureView<'a>,
    pub diagnostics: StrategyDiagInput<'a>,
}
```

### StrategyProposal

Recommended:

```rust
pub struct StrategyProposal {
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
    pub notes: StrategyNotes,
}
```

The shared engine consumes this and applies final gating/post-processing.

## Feature System

### Why It Matters

This is the main mechanism that makes new strategies cheap.

Without it:

- every strategy change adds more inline derivation logic to orchestrator
- backtest/live need manual per-strategy data plumbed in Python

With it:

- the strategy declares the features it needs
- the engine provides them
- new indicators become one new provider implementation and one request enum

### Feature Requests

Recommended request enums:

```rust
pub enum FeatureRequest {
    EmaClose { tf: Timeframe, span: f64 },
    EmaBand { tf: Timeframe, spans: Vec<f64> },
    LogRangeEma { tf: Timeframe, span: f64 },
    TrailingBundle,
    Balance,
    BalanceRaw,
    EffectiveMinCost,
    RealizedLossState,
    WalletExposure,
    PositionAge,
}
```

### Feature Provider

The provider should expose a typed view:

```rust
pub struct StrategyFeatureView<'a> { ... }

impl StrategyFeatureView<'_> {
    pub fn ema_close(&self, tf: Timeframe, span: f64) -> Option<f64>;
    pub fn ema_band(&self, tf: Timeframe, spans: &[f64]) -> Option<EMABands>;
    pub fn log_range_ema(&self, tf: Timeframe, span: f64) -> Option<f64>;
    pub fn trailing(&self) -> Option<&TrailingPriceBundle>;
}
```

### Python Boundary For Features

Python should not hardcode strategy-specific indicator gathering.

Instead:

1. Python provides raw/current indicator inputs already available today
2. Rust feature provider derives strategy views from them
3. When a truly new indicator is required, add a new provider capability and include it in the
   payload-builder contract

For live:

- Python asks Rust for required feature requests for the active strategy kind
- Python gathers only the needed base data
- Python builds one generic payload

For backtest:

- backtest already has the full stepwise state
- the feature provider runs inside Rust per timestep

## Portfolio Policy

Do not over-generalize this in phase 1.

Recommended:

```rust
pub enum PortfolioPolicyKind {
    SharedPerSide,
    Custom,
}
```

Phase-1 support:

- all strategies, including `simple_ema_mm`, use `SharedPerSide`

Phase-2 support:

- if a future strategy genuinely needs global long+short slot budgeting or gross-exposure caps,
  allow a custom strategy portfolio policy

This avoids blocking the refactor on speculative complexity.

## Config Design

### Recommended New Shape

Shared engine/risk params remain in `bot`.

Strategy-specific params move to a new config section.

Recommended:

```json
{
  "live": {
    "strategy_kind": "simple_ema_mm"
  },
  "bot": {
    "long": {
      "total_wallet_exposure_limit": 1.0,
      "n_positions": 3,
      "risk_twel_enforcer_threshold": 1.0
    },
    "short": {
      "total_wallet_exposure_limit": 1.0,
      "n_positions": 3,
      "risk_twel_enforcer_threshold": 1.0
    }
  },
  "strategy": {
    "long": {
      "base_qty_pct": 0.01,
      "ema_span_0": 200.0,
      "ema_span_1": 800.0,
      "offset": 0.002,
      "offset_psize_weight": 0.1
    },
    "short": {
      "mirror_long": true
    }
  }
}
```

### Compatibility Rule

During migration:

- continue accepting current `bot.*` strategy fields for the adaptive-grid strategy
- allow `simple_ema_mm` to read old mapped config on the research branch
- add a normalized internal config representation so Rust receives:
  - `shared_bot_params`
  - `strategy_params`
  - `strategy_kind`

## Python Changes

The Python side should remain an orchestrator/data-loader, not a strategy implementation layer.

### Required Python Changes

1. Config normalization
   - add `strategy` section support
   - normalize legacy config into the new internal representation

2. Strategy spec exposure
   - Python should be able to ask Rust for supported params and defaults

3. Optimizer adapter changes
   - optimizer bounds should be generated from strategy spec, not hardcoded bot-key lists

4. Payload builder changes
   - payload should include strategy params map
   - payload should remain generic across strategies

### Not Required

- Python should not contain per-strategy pricing/sizing math
- Python should not duplicate strategy indicator semantics if Rust can derive them from raw inputs

## Backtester Changes

### Keep

- fill simulation
- equity tracking
- fee handling
- metrics
- suite logic

### Change

The backtester should stop assuming the strategy is represented by the current `BotParams`.

Instead:

- keep shared risk/portfolio state as typed params
- pass strategy params separately into the orchestrator runtime
- let the strategy runtime define sizing semantics explicitly

This is the key fix for cases like `simple_ema_mm`, where configured sizing semantics should not
implicitly inherit dynamic-WEL behavior unless the strategy explicitly opts into that.

## Migration Plan

### Phase 1: Extract Strategy Runtime Seam

Goal: isolate strategy order generation without changing external behavior for the default strategy.

Tasks:

1. Create `strategies/` module with:
   - `adaptive_grid.rs`
   - `simple_ema_mm.rs`
   - `registry.rs`
   - `spec.rs`

2. Define:
   - `StrategyContext`
   - `StrategyProposal`
   - `StrategySpec`

3. Move adaptive-grid entry/close generation behind the strategy contract.

4. Keep shared orchestrator policy intact.

Success criteria:

- no behavior change for adaptive-grid
- `simple_ema_mm` still runs
- orchestrator no longer contains inline strategy math branches

### Phase 2: Introduce Feature Provider

Goal: stop hardcoding feature derivation in the orchestrator.

Tasks:

1. Create `features/` module
2. Move EMA-band derivation there
3. Move log-range EMA derivation there
4. Provide typed feature access from `StrategyContext`
5. Make each strategy declare required features

Success criteria:

- orchestrator builds feature views generically
- adding a strategy using only existing features requires no orchestrator edits

### Phase 3: Split Shared Params From Strategy Params

Goal: stop growing `BotParams` for every new experiment.

Tasks:

1. Introduce `SharedBotParams`
2. Introduce `StrategyParams`
3. Add config normalization from current schema to the new internal shape
4. Keep backwards compatibility for adaptive-grid

Success criteria:

- new strategy params do not require new fields in `BotParams`
- Python optimizer/config validation works from strategy spec

### Phase 4: Rust-Driven Strategy Specs For Python

Goal: eliminate repeated per-strategy Python plumbing.

Tasks:

1. Expose strategy specs from Rust via PyO3
2. Update config validation to use those specs
3. Update optimizer bounds handling to use those specs
4. Update docs/help generation if desired

Success criteria:

- adding a strategy requires minimal Python changes
- optimizer can discover params generically

### Phase 5: Optional Custom Portfolio Policies

Only do this if a real strategy needs it.

Examples:

- global long+short slot budget
- gross-exposure budgeting across both sides
- strategy-specific open-position allocator

Until there is a concrete need, keep all strategies on shared per-side policy.

## `simple_ema_mm` Case Study

This strategy is the right benchmark for the refactor because it needs:

- existing EMA-based features
- existing balance/position state
- custom pricing and sizing semantics
- no custom fill engine
- shared one-way/risk/backtest machinery

If the new runtime makes `simple_ema_mm` clean, it will make many future experiments cheap.

### What `simple_ema_mm` Should Look Like After Refactor

Strategy module:

- reads `base_qty_pct`, `ema_span_0`, `ema_span_1`, `offset`, `offset_psize_weight`
- requests EMA-band feature
- emits one entry clip and one close clip
- relies on shared engine for:
  - one-way blocking
  - TWEL/WEL/loss gates
  - min-cost handling
  - fill simulation

No custom orchestrator adaptation function should be needed.

## Acceptance Criteria

The refactor is successful when all of the following are true:

1. A new strategy using only existing indicators can be added by:
   - adding one Rust strategy module
   - registering one strategy spec
   - adding tests
   - without modifying orchestrator policy code

2. A new strategy param does not require expanding shared `BotParams`.

3. Backtest and live still use the same Rust strategy runtime.

4. Optimizer bounds can be generated from strategy metadata instead of hardcoded key lists.

5. Feature derivation for existing indicators is centralized and reusable.

6. Shared risk/portfolio behavior remains centralized.

## Recommended First Implementation Scope

Do not try to land the whole end-state in one PR.

Best first milestone:

1. Extract adaptive-grid and `simple_ema_mm` into `strategies/`
2. Define `StrategyContext`, `StrategyProposal`, `StrategySpec`
3. Keep current shared `BotParams` for the moment
4. Add a thin `strategy_params` blob only for non-adaptive-grid strategies

That captures most of the extensibility win without forcing the full config/optimizer redesign in
the first step.

## Summary Recommendation

The best practical shape is:

- Rust shared engine for portfolio/risk/execution
- Rust strategy modules with a common runtime contract
- Rust feature provider for indicators/state views
- Python as generic payload builder and executor
- strategy-specific params separated from shared bot/risk params

This keeps Passivbot’s strongest property intact, Rust as source of truth for behavior, while
making future strategy experiments materially cheaper to implement and ship.
