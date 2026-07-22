# Strategy Runtime Contracts

## Canonical And Legacy Strategy Schemas

The canonical strategy kind is `trailing_martingale`. The released v7 `trailing_grid` schema is not
aliased to it. Do not add compatibility migrations, duplicate strategy names, or silent shims for
removed fields unless the user explicitly asks for released-version compatibility.

`trailing_grid_v7` is the explicit compatibility exception. It is a normal Rust strategy kind,
deprecated from introduction, and exists only for configs converted by
`passivbot tool migrate-config-v7`. Keep its v7-only fields under
`bot.<side>.strategy.trailing_grid_v7`; do not add them to `trailing_martingale` or shared
`BotParams` unless a shared runtime function truly requires it.

The legacy pre-V8 `pb_multi` root shape (`TWE_long`, `TWE_short`,
`universal_live_config`, and related top-level fields) is not a supported V8 config input or
migration contract. Residual flavor detection, formatter code, and narrow unit fixtures are stale
internal compatibility surfaces: ignore them when adding current config fields, and do not infer
production or live support from their presence. The only supported pre-V8 strategy migration path
is the explicit `passivbot tool migrate-config-v7` workflow for normalized V7 trailing-grid input.

With `entry_cooldown_minutes = 0.0`, `trailing_grid_v7` preserves v7's simultaneous grid-entry
ladder even when a later trailing leg uses retracement. Its recursive generator stops expansion
before stacking retracement-dependent trailing orders. Positive entry cooldowns still stage at
most one position-adding order and apply their configured post-fill delay.

Removed v7 trailing-grid concepts:

- `entry_trailing_grid_ratio`
- `close_trailing_grid_ratio`
- `close_grid_markup_start`
- `close_grid_markup_end`
- linear `markup_start` to `markup_end` TP grids
- separate entry grid spacing vs trailing threshold knobs

Canonical config path:

```text
bot.<side>.strategy.trailing_martingale
```

Deprecated v7 compatibility config path:

```text
bot.<side>.strategy.trailing_grid_v7
```

Optimizer selector contract:

- `optimize.fixed_params` and `--fine_tune_params` use dotted config-path selectors.
- `long.*` / `short.*` selectors are aliases for `bot.long.*` / `bot.short.*`.
- Selectors match path segments by prefix or suffix, not partial substring. Use `long.strategy`
  to match the whole active strategy subtree, `long.strategy.close` to match only
  `bot.long.strategy.<active_strategy>.close.*`, or a leaf selector such as
  `we_excess_allowance_pct` to match every bound ending with that parameter name.
- `*` is allowed as a one-segment wildcard, for example `*.strategy.close`.
- Do not use flattened underscore selector names such as `long_entry_*` in current user-facing docs
  or agent instructions.
- When `--fine_tune_params` is combined with `--start`, the starting configs are anchor configs:
  non-tuned optimizer-bound bot params are fixed from the selected anchor, while the fine-tune
  selectors remain tunable. Base-config policy fields, including boolean toggles such as
  `bot.<side>.hsl.enabled`, still win over anchors. Anchor and seed values outside
  `optimize.bounds` are clamped into bounds with aggregated source/key logging. Without
  `--fine_tune_params`, `--start` remains seed-only.

Timeframe-specific EMA spans use explicit horizon suffixes in canonical config names. Use `_1m`
for 1-minute candle inputs and `_1h` for 1-hour candle inputs, for example
`volatility_ema_span_1m`, `volatility_ema_span_1h`, `forager_volume_ema_span_1m`, and
`forager_volatility_ema_span_1m`. Generic strategy EMA spans such as `ema_span_0` and `ema_span_1`
remain unsuffixed because their timeframe comes from the strategy's base candle stream.

## Trailing Martingale Semantics

Entries and closes use threshold/retracement fields.

- `retracement_base_pct <= 0.0`: trailing disabled, use passive recursive limit-order behavior.
- `retracement_base_pct > 0.0`: threshold is the required excursion, retracement is the confirmation move.
- Trailing extrema reset after any fill for the same coin+pside.
- Passivbot tracks trailing state itself from candle inputs; exchange-native trailing order types are not part of the core contract.

Entry thresholds and retracements are multiplicative distances. Positive volatility or wallet
exposure weights widen them.

`bot.<side>.strategy.trailing_martingale.entry.ema_gate_mode` controls which entry orders are
capped/floored by the EMA entry band:

- `disabled`: no emitted entry order is capped/floored by EMA. Flat initial entries rest at current
  best bid/ask; re-entries use normal threshold/retracement logic.
- `all`: initial entries, partial initial entries, and re-entries are capped/floored by EMA.
- `initial`: normal initial and partial initial entries are capped/floored by EMA. Re-entries are
  not EMA capped/floored. This is the default and preserves the previous canonical behavior.
- `reentry`: flat initial entries are not capped/floored by EMA; re-entries are capped/floored by
  EMA.

The value is fixed config, not an optimizer parameter. In one-way mode, if both sides are flat and
both long and short are otherwise eligible, the long-vs-short tie-break still uses the EMA entry
band distance even when `ema_gate_mode = "disabled"`. Candles and EMA bands are therefore required
for that tie-break, and missing EMA inputs must fail loudly.

Close thresholds are additive so they can intentionally cross through break-even or negative markup
as wallet exposure rises. Close retracement is volatility-weighted and intentionally has no
wallet-exposure modifier.

Auto-unstuck has its own EMA trigger toggle:
`bot.<side>.unstuck.ema_gating_enabled`. It defaults to `true`. When false, auto-unstuck skips the
EMA trigger/readiness check but still requires `unstuck.enabled`, loss allowance, exposure
threshold, close sizing, and valid market/exchange inputs. The toggle does not add independent
unstuck EMA spans.

## Live/Backtest Market Slippage Boundary

`backtest.market_order_slippage_pct` is a backtest simulation knob only. Live orchestrator input
must not read or forward it; live callers intentionally omit `market_order_slippage_pct` and rely
on Rust's `0.0` serde default for live loss projections. Live market-order execution uses the
current bid/ask snapshot plus exchange-side behavior, controlled by `live.market_orders_allowed`
and `live.market_order_near_touch_threshold`.

## Close Recursion Contract

Close orders are computed recursively when the close threshold depends on wallet exposure:

1. Compute the next close from current position exposure.
2. Size it up to `close.qty_pct`.
3. Simulate that close as filled.
4. Recompute wallet-exposure ratio.
5. Repeat until the position is exhausted or the ladder is complete.

If `close.retracement_base_pct <= 0.0` and `close.threshold_we_weight == 0.0`, recursive closes all
have the same price. Rust intentionally emits one full-position close in that case; `close.qty_pct`
is effectively moot because multiple same-price slices would be redundant.

## Close Reducer Compatibility

For each coin and position side, Rust selects at most one protective reducer per ideal-order batch.
Active panic, TWEL/WEL exposure-repair, and auto-unstuck intents are consolidated by keeping the
largest requested absolute reduction, not by summing their quantities. This prevents a small
auto-reduce from suppressing a materially larger unstuck close. A full-position HSL panic is
therefore largest and remains exclusive; equal-size ties keep panic first and otherwise prefer the
closest-to-fill candidate.

A selected non-panic reducer may coexist with ordinary grid, trailing, or EMA-anchor closes. Its
quantity is reserved first; if aggregate close quantity would exceed the position, ordinary closes
are trimmed furthest-from-fill first against the remaining quantity. Aggregate reduce-only quantity
must never exceed the position after quantity-step and effective-minimum handling. The cumulative
realized-loss gate applies after this allocation and evaluates the selected reducer before ordinary
closes. Live reconciliation reapplies the same aggregate cap against current exchange position
size, still trimming ordinary closes before the reducer if the position shrank after planning.
This contract is shared by every strategy kind, including `trailing_grid_v7`.

## Source Of Truth

Rust owns strategy dispatch and order behavior. If Python docs, adapters, or tests imply old
`trailing_grid` behavior, update those surfaces to match Rust rather than adding Python-side
behavior patches.

## Validation

- Rust unit tests cover strategy dispatch, entries, closes, recursion, risk, and unstuck behavior.
- Python tests use the verified real extension when asserting Rust output.
- Live/backtest parity tests compare optimized behavior with a simple reference contract.
- Config migration/schema tests reject removed fields outside `trailing_grid_v7`.
- Behaviorally relevant changes include a bounded real backtest smoke.

## Key Code And Tests

- `passivbot-rust/src/orchestrator.rs`
- `passivbot-rust/src/entries.rs`
- `passivbot-rust/src/closes.rs`
- `tests/test_orchestrator_json_api.py`
- `tests/test_orchestrator_integration.py`
- `tests/test_auto_unstuck_allowance.py`
