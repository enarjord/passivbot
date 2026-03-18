# Bug Report & Implementation Plan: EMA Timing Parity Between Backtest and Live

## Bug Summary

The HSL (equity hard stop loss) drawdown EMA produces different values on
different exchanges running identical configs.  Observed live:

- **Hyperliquid**: `drawdown_ema ≈ 0.002` with `drawdown_raw ≈ 0.04`
- **Bybit**: `drawdown_ema ≈ 0.003` with `drawdown_raw ≈ 0.05`

Both instances use the same config, have been running for ~2.5 days, and show
similar raw drawdown.  The EMA should be nearly identical.

## Root Cause

### The immediate bug

`_equity_hard_stop_sample_minutes()` in `passivbot.py:946` computes:

```python
sample_minutes = float(self.live_value("execution_delay_seconds")) / 60.0
```

This is a **fixed config value** (e.g., 5 seconds → 0.083 minutes), but the
actual time between EMA updates varies because the main execution loop
(`passivbot.py:3052-3057`) adds:

1. `execution_delay_seconds` sleep
2. Loop processing time (varies by exchange latency)
3. Up to 30 additional seconds of sleep, interruptible by websocket events

The EMA formula uses `alpha = 2 / (ema_span_minutes / sample_minutes + 1)`.
When `sample_minutes` is wrong, `alpha` is wrong, and the EMA drifts at a rate
proportional to how often the loop actually runs.

Exchanges with more websocket activity (order fills, balance updates) break the
30-second sleep more frequently → more loop iterations → more EMA
accumulations per unit time → higher EMA.

From the logs:
- Bybit: 77 fills, 281 orders → faster loop → higher EMA
- Hyperliquid: 54 fills, 238 orders → slower loop → lower EMA

### The deeper problem: three parity violations

| Scenario | `sample_minutes` used | Actual sample interval | Parity? |
|---|---|---|---|
| Backtest (1m candles) | `1.0` (candle interval) | Exactly 1 minute | Correct |
| Backtest (5m candles) | `5.0` (candle interval) | Exactly 5 minutes | Correct for itself, but uses different alpha than 1m |
| Live (history init) | `1.0` (hardcoded) | 1-minute history rows | Correct, matches 1m backtest |
| Live (ongoing) | `execution_delay_seconds / 60` | Variable (5-35+ seconds) | **Broken** |

Additionally, `timestamp_ms` is passed to `apply_sample()` in the PyO3 binding
but **explicitly discarded** (`let _ = timestamp_ms;` at `python.rs:200`).

### Why "just use actual elapsed time" is insufficient

Using raw elapsed time as `sample_minutes` would fix the cross-exchange
divergence but still not match backtest behavior, because:

1. The EMA alpha formula `2 / (span/sample_minutes + 1)` applied once per call
   is not perfectly equivalent to applying `N` one-minute steps.  For span=60:
   one step with `sample_minutes=5` gives decay `11/13 = 0.846`, while five
   steps with `sample_minutes=1` gives decay `(59/61)^5 = 0.848`.
2. Intra-minute calls would still jitter the EMA with sub-minute alpha values
   that no backtest ever produces.

## Design: Minute-Quantized EMA

### Core principle

All minute-span EMAs update **only on whole-minute boundaries**.  The Rust
function owns the timing:

1. Receives `timestamp_ms` from the caller
2. Quantizes to `current_minute = timestamp_ms / 60_000`
3. If `elapsed_minutes == 0` since last update: returns cached result, no
   mutation
4. If `elapsed_minutes >= 1`: applies closed-form multi-step EMA update, caches
   result

### The closed-form update

For `N` elapsed minutes with constant `drawdown_raw`:

```
alpha = 2.0 / (ema_span_minutes + 1.0)
decay = (1.0 - alpha).powf(N as f64)
drawdown_ema_new = drawdown_raw + (drawdown_ema_old - drawdown_raw) * decay
```

This is mathematically equivalent to applying `N` individual one-minute EMA
steps with the same input value.  It runs in O(1) regardless of gap size.

### Why closed form and not a loop

When the live bot misses minutes (network lag, slow loop, etc.), we don't have
intermediate equity values.  We only have equity at call time.  The best
assumption is that `drawdown_raw` was constant during the gap — exactly what
the closed form computes.  A loop of N steps with the same input gives an
identical result but costs O(N).

### Parity table after fix

| Scenario | alpha | Steps per call | Effective decay |
|---|---|---|---|
| Backtest 1m candles | `2/(span+1)` | 1 | `(1-alpha)^1` |
| Backtest 5m candles | `2/(span+1)` | 5 | `(1-alpha)^5` |
| Live (any exchange) | `2/(span+1)` | elapsed minutes | `(1-alpha)^N` |
| History init (1m rows) | `2/(span+1)` | 1 per row | `(1-alpha)^1` |

All paths use the same alpha (computed for 1-minute resolution) and the same
closed-form step logic.

### Behavioral change for non-1m backtests

Current 5m backtest uses `alpha_5 = 2/(span/5 + 1)` applied once.  Proposed
uses `alpha_1 = 2/(span + 1)` applied 5 times via closed form.  For span=60
the decay changes from 0.846 to 0.848 — a negligible difference, but now
**every path is mathematically identical**.

## Implementation Plan

### Step 1: Modify `HardStopState` (equity_hard_stop_loss.rs)

Add fields for minute-quantized operation:

```rust
pub struct HardStopState {
    pub peak_strategy_equity: f64,
    pub drawdown_ema: f64,
    pub tier: HardStopTier,
    pub red_latched: bool,
    pub initialized: bool,
    pub last_minute: Option<u64>,           // NEW
    pub cached_step: Option<HardStopStep>,  // NEW
}
```

`last_minute` is `None` until the first sample.  `cached_step` holds the
result returned on intra-minute re-calls.

### Step 2: Change `step_with_peak_strategy_equity` signature

Replace `sample_minutes: f64` with `timestamp_ms: u64`:

```rust
pub fn step_with_peak_strategy_equity(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    peak_strategy_equity: f64,
    timestamp_ms: u64,
) -> Result<HardStopStep, String>
```

Inside the function:

```rust
let current_minute = timestamp_ms / 60_000;

if !state.initialized {
    state.initialized = true;
    state.peak_strategy_equity = peak_strategy_equity;
    state.drawdown_ema = 0.0;
    state.last_minute = Some(current_minute);
    state.tier = if state.red_latched { Red } else { Green };
    let step = HardStopStep { drawdown_raw: 0.0, drawdown_score: 0.0, ... };
    state.cached_step = Some(step);
    return Ok(step);
}

let last_minute = state.last_minute.expect("initialized but no last_minute");
if current_minute < last_minute {
    return Err(format!(
        "timestamp must be non-decreasing: minute {} < last {}",
        current_minute, last_minute
    ));
}
let elapsed_minutes = current_minute - last_minute;

if elapsed_minutes == 0 {
    // Intra-minute re-call: return cached result, no state mutation
    return Ok(state.cached_step.expect("initialized but no cached step"));
}

// --- Minute boundary crossed: update EMA ---
state.peak_strategy_equity = peak_strategy_equity;
let drawdown_raw = (1.0 - equity / state.peak_strategy_equity.max(f64::EPSILON)).max(0.0);
let alpha = 2.0 / (config.ema_span_minutes + 1.0);
let decay = (1.0 - alpha).powi(elapsed_minutes as i32);
state.drawdown_ema = drawdown_raw + (state.drawdown_ema - drawdown_raw) * decay;
let drawdown_score = drawdown_raw.min(state.drawdown_ema);

// ... tier classification (unchanged) ...

state.last_minute = Some(current_minute);
let step = HardStopStep { drawdown_raw, drawdown_score, tier, changed, alpha, elapsed_minutes };
state.cached_step = Some(step);
Ok(step)
```

### Step 3: Remove `span_samples()` function

No longer needed.  Alpha is always `2 / (ema_span_minutes + 1)`.  Remove the
function and all references.

### Step 4: Update `step()` convenience wrapper

Same signature change: `sample_minutes` → `timestamp_ms`.  Pass through.

### Step 5: Update `HardStopStep` struct

Replace `span_samples` and `alpha` fields:

```rust
pub struct HardStopStep {
    pub drawdown_raw: f64,
    pub drawdown_score: f64,
    pub tier: HardStopTier,
    pub changed: bool,
    pub alpha: f64,              // always 2/(ema_span_minutes+1)
    pub elapsed_minutes: u64,    // NEW: replaces span_samples
}
```

### Step 6: Update PyO3 binding `apply_sample()` (python.rs)

- Remove `sample_minutes` parameter
- Stop discarding `timestamp_ms` — pass it to `step_with_peak_strategy_equity()`
- Replace `span_samples` in output dict with `elapsed_minutes`

### Step 7: Update Python callers (passivbot.py)

- **Delete** `_equity_hard_stop_sample_minutes()` (lines 945–951)
- Remove `sample_minutes` parameter from `_equity_hard_stop_apply_sample()` and
  its three call sites:
  - `_equity_hard_stop_initialize_from_history()` line 1425 (history replay)
  - `_equity_hard_stop_initialize_from_history()` line 1497 (final live sample)
  - `_equity_hard_stop_check()` line 1608 (ongoing live)
- Remove `sample_minutes` and `span_samples` from metrics dict
- Add `elapsed_minutes` to metrics dict (for observability)

### Step 8: Update backtester call site (backtest.rs)

Line 2501–2509: replace `sample_minutes` with `timestamp_ms`:

```rust
// REMOVE: let sample_minutes = self.backtest_params.candle_interval_minutes.max(1) as f64;
let step = ehsl::step_with_peak_strategy_equity(
    state,
    hs_cfg,
    strategy_equity,
    peak_strategy_equity,
    timestamp_ms,  // already available at line 2459
)
```

For a 5m backtest, consecutive timestamps are 300,000ms apart.
`elapsed_minutes = 5`, and the closed form applies 5 one-minute steps.

### Step 9: Update standalone `equity_hard_stop_step_py` (python.rs)

This stateless one-shot function (line 460) currently takes `sample_minutes`.
Replace with `timestamp_ms` + `last_timestamp_ms` parameters so callers can
specify the time delta.  Or convert to a thin wrapper that creates a temporary
`HardStopState` with `last_minute` pre-seeded.

### Step 10: Update Rust tests (equity_hard_stop_loss.rs)

All tests pass `sample_minutes` to `step()`.  Change to pass `timestamp_ms`
values:

- Tests that used `sample_minutes = 1.0` → timestamps 60,000ms apart
- Tests that used `sample_minutes = 5.0` → timestamps 300,000ms apart
- Add new tests:
  - **Intra-minute returns cached result**: call twice with timestamps in the
    same minute, verify identical output and no state mutation
  - **Gap filling**: call at minute 0, then minute 5 → verify EMA equals
    `raw + (0 - raw) * (1-alpha)^5`
  - **Timestamp monotonicity**: verify error on decreasing timestamp
  - **Large gap convergence**: call at minute 0 then minute 100000 → verify
    EMA ≈ drawdown_raw (fully converged)

### Step 11: Update Python tests (test_unstucking_safeguards.py etc.)

Remove `sample_minutes` from `apply_sample()` calls.  Ensure timestamps are
properly spaced.

## Edge Cases & Dangers

### Intra-minute re-calls (elapsed = 0)

**Behavior**: Return cached result.  No state mutation.

**Danger**: If equity changes significantly within a minute, the EMA won't
reflect it until the next minute boundary.

**Mitigation**: Acceptable.  The backtest also only updates per-candle (minimum
1 minute).  This is the definition of parity.

**Important**: Peak strategy equity must NOT be updated on intra-minute calls.
If we updated the peak but not the EMA, the drawdown_raw would change while
the cached step reflects the old drawdown_raw.  The entire state snapshot must
be frozen intra-minute.

### Large gaps (bot was offline for hours/days)

**Behavior**: `(1-alpha)^N → 0` for large N, so
`drawdown_ema → drawdown_raw`.  The EMA fully converges to the current
drawdown in one call.

**Danger**: If the bot was offline during a drawdown and comes back after
recovery, `drawdown_raw` at restart might be 0, and the EMA would also go to
~0.  This is correct — the EMA should reflect current state, not stale
history.

**Danger**: Integer overflow in `elapsed_minutes` if timestamps are wildly
wrong.  Use `u64` throughout (max ~34 billion years in minutes).

**Danger**: `powi(N)` with very large N.  Rust's `f64::powi` handles large
exponents correctly (returns 0.0 for `(0.99)^huge`).  But use `.powf()` with
`elapsed as f64` to avoid `i32` overflow on `powi` — `i32::MAX` is ~2 billion
which is ~4000 years in minutes, safe in practice but `.powf()` is strictly
safer.

### Timestamps going backwards

**Behavior**: Hard error.  `"timestamp must be non-decreasing"`.

**Danger**: Exchange clock drift, bad test data, or feed replays with
out-of-order data.

**Mitigation**: This is correct.  EMA state is path-dependent; out-of-order
timestamps make it undefined.  Fail loudly per AGENTS.md §3.

### First sample initialization

**Behavior**: Set `drawdown_ema = 0.0`, record `last_minute`, return
`drawdown_raw = 0.0`.  No EMA step on the first call.

**Danger**: If the bot initializes during a drawdown, the first sample
produces `drawdown_raw = 0` (because `peak_strategy_equity = equity` on
init).  The EMA starts at 0.  Subsequent samples compute the real drawdown.

**Mitigation**: This matches current behavior.  History initialization replays
the full equity curve before live trading starts, so the EMA is warmed up.

### History replay → live transition

**Behavior**: History rows are 1-minute-spaced (timestamps in ms, 60,000
apart).  The Rust function sees `elapsed_minutes = 1` for each row.  The final
live sample has whatever elapsed time since the last history row.

**Danger**: If the last history row is from 30 minutes ago, the live sample
sees `elapsed_minutes = 30` and applies a 30-minute closed-form step.  This
is correct — the EMA should catch up to current state.

### ema_span_minutes = 0 or negative

**Behavior**: `config.validate()` rejects it.  Hard error.

**Mitigation**: Already handled.  No change needed.  This follows AGENTS.md §3
(fail loudly) and §6 (config correctness is centralized — invalid config
should never reach here).

### ema_span_minutes very small (e.g., 0.001)

**Behavior**: `alpha = 2 / (0.001 + 1) ≈ 2.0`.  Clamped to `alpha.min(1.0)`
(alpha > 1 is invalid for EMA).

**Mitigation**: Add a clamp: `let alpha = (2.0 / (config.ema_span_minutes +
1.0)).min(1.0)`.  This already exists in the current code's validation path.

### Backtest with sub-minute candle interval

**Behavior**: Not currently supported (backtester uses
`candle_interval_minutes.max(1)`).  If it were: timestamps would be < 60,000ms
apart → `elapsed_minutes = 0` → cached result → EMA never updates.

**Mitigation**: Sub-minute backtests are not a current feature.  If added
later, the EMA quantization unit would need to be configurable (or sub-minute
candles would simply not affect the EMA, which is arguably correct for a
minutes-scale risk metric).

## Error Handling Contract

Per AGENTS.md §3 and `docs/ai/error_contract.md`:

### Must fail loudly (hard error)

- `ema_span_minutes` missing, zero, negative, NaN, or Inf
- `timestamp_ms` decreasing
- `equity` non-finite or <= 0
- `peak_strategy_equity` non-finite or < equity
- `HardStopState` invariant violations (`initialized` but `last_minute` is
  `None`, etc.)

### Must NOT use fallback defaults

- Do not default `ema_span_minutes` to a "safe" value if missing from config
- Do not silently skip an EMA update on bad input — return `Err`
- Do not clamp `elapsed_minutes` to 1 if it's 0 — return the cached result
- Do not silently ignore timestamp regression — return `Err`

### Observable diagnostics

The returned `HardStopStep` (and Python metrics dict) should include:

- `alpha`: the EMA alpha used (always `2/(span+1)`)
- `elapsed_minutes`: how many minute-steps were applied this call
- `drawdown_ema`: the updated EMA value
- `drawdown_raw`: the raw drawdown at this timestamp

This allows operators to verify that the EMA is updating at the expected rate
and with the expected alpha.

## Future: Standardizing All EMAs

The drawdown EMA is the first candidate for minute-quantized, Rust-owned EMA
timing.  The same pattern applies to all other EMAs in the system:

| EMA | Current span unit | Natural resolution | Current owner |
|---|---|---|---|
| HSL drawdown | minutes | 1 minute | Rust (ehsl) |
| Price (ema_span_0, ema_span_1) | minutes | 1 minute | Rust (backtest), Python (live via candlestick_manager) |
| Volume filter | minutes | 1 minute | Rust (backtest), Python (live via candlestick_manager) |
| Volatility filter | minutes | 1 minute | Rust (backtest), Python (live via candlestick_manager) |
| Entry volatility | hours | 1 hour | Rust (backtest), Python (live) |

### The template

For each EMA type:

1. **Rust function** takes `timestamp_ms`, quantizes to the natural resolution
   (1 minute or 1 hour), applies closed-form multi-step update
2. **Python** passes `timestamp_ms` and consumes the result — no alpha
   computation, no timing logic
3. **Backtest** calls the same Rust function with candle timestamps
4. **Tests** verify cross-resolution parity: 1m backtest, 5m backtest, and
   simulated live calls with irregular timestamps all produce identical EMA
   values for the same equity curve

### Migration priority

1. **HSL drawdown EMA** (this plan) — risk-critical, parity bug confirmed live
2. **Price EMAs** — trading-critical, used for entry/exit thresholds
3. **Volume/volatility filter EMAs** — forager-critical, used for coin
   selection
4. **Entry volatility EMAs** — hourly resolution, lower urgency

Each migration follows the same pattern: move timing ownership from Python to
Rust, quantize to the natural resolution, use the closed-form update for gap
filling, fail loudly on bad inputs.
