# Strategy Runtime Contracts

## V8 Clean Break

The canonical v8 strategy kind is `trailing_martingale`. The v7 `trailing_grid` schema is not
aliased to it. Do not add compatibility migrations, duplicate strategy names, or silent shims for
removed fields unless the user explicitly asks for released-version compatibility.

Removed v7 trailing-grid concepts:

- `entry_trailing_grid_ratio`
- `close_trailing_grid_ratio`
- `close_grid_markup_start`
- `close_grid_markup_end`
- linear `markup_start` to `markup_end` TP grids
- separate entry grid spacing vs trailing threshold knobs

Canonical v8 config path:

```text
bot.<side>.strategy.trailing_martingale
```

Optimizer selector contract:

- `optimize.fixed_params` and `--fine_tune_params` use dotted config-path selectors.
- `long.*` / `short.*` selectors are aliases for `bot.long.*` / `bot.short.*`.
- Selectors match path segments by prefix, not substring. Use `long.strategy` to match the
  whole active strategy subtree, or `long.strategy.close` to match only
  `bot.long.strategy.<active_strategy>.close.*`.
- `*` is allowed as a one-segment wildcard, for example `*.strategy.close`.
- Do not use flattened underscore selector names such as `long_entry_*` in v8 user-facing docs
  or agent instructions.

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

Close thresholds are additive so they can intentionally cross through break-even or negative markup
as wallet exposure rises. Close retracement is volatility-weighted and intentionally has no
wallet-exposure modifier.

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

## Source Of Truth

Rust owns strategy dispatch and order behavior. If Python docs, adapters, or tests imply old
`trailing_grid` behavior, update those surfaces to match Rust rather than adding Python-side
behavior patches.
