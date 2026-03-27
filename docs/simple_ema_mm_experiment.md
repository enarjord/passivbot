# Simple EMA MM Experiment

This document defines the first-pass `simple_ema_mm` research strategy added for quick
backtesting and optimizer experiments.

## Goal

Provide a minimal, testable strategy variant that:

- reuses the existing Rust orchestrator/backtester
- works with backtests, optimization, fills review, and plotting
- avoids adding a separate simulation engine or a large config surface

## Enable

Set:

```json
{
  "live": {
    "strategy_kind": "simple_ema_mm"
  }
}
```

Default remains:

```json
{
  "live": {
    "strategy_kind": "adaptive_grid"
  }
}
```

## Strategy Definition

The strategy uses three close-price EMAs:

- `ema_span_0`
- `ema_span_1`
- `sqrt(ema_span_0 * ema_span_1)`

For each step:

- `lower_ema_band = min(ema0, ema1, ema2)`
- `upper_ema_band = max(ema0, ema1, ema2)`
- `pside_bias = psize * mid / balance`
- `mid = (bid + ask) / 2`

Quotes:

- `bid_price = lower_ema_band * (1 - offset - pside_bias * offset_psize_weight)`
- `ask_price = upper_ema_band * (1 + offset - pside_bias * offset_psize_weight)`

Sizing:

- `base_qty = balance * wallet_exposure_limit * base_qty_pct`
- no double-down factor in this first pass
- one entry and one close per active pside at most

## Current Param Mapping

For this experiment, the strategy shape params are read from `config.bot.long` for both long and
short order generation:

- `base_qty_pct -> entry_initial_qty_pct`
- `ema_span_0 -> ema_span_0`
- `ema_span_1 -> ema_span_1`
- `offset -> entry_initial_ema_dist`
- `offset_psize_weight -> entry_grid_spacing_we_weight`

## Important Assumptions

- One-way mode is handled by the existing orchestrator blocking logic. No new net-position model was added.
- Portfolio/risk semantics remain standard Passivbot semantics for this first pass.
- Per-side exposure and slot limits still come from the existing side-specific runtime budget logic.
- Only the strategy shape params are mirrored from `bot.long` into short-side order generation.
- Existing short-side shape params are ignored by `simple_ema_mm`.
- Existing foraging/coin-selection machinery remains orthogonal and reusable.

## Order Behavior

- Long side:
  - flat: one bid entry
  - in position: one bid entry plus one ask close
- Short side:
  - flat: one ask entry
  - in position: one bid close plus one ask entry

The implementation reuses existing order types where practical:

- entries: `entry_initial_normal_*` or `entry_grid_normal_*`
- closes: `close_grid_*`

## Deliberate Non-Goals For V1

- no new portfolio model
- no global gross-exposure cap for the strategy
- no strategy-specific optimizer schema
- no new fill simulator
- no generalized multi-strategy orchestrator refactor yet

## Practical Notes

- For optimization runs, tune the long-side shape params; short-side shape params are ignored by
  this experiment.
- Backtest plots, fills review, and analysis remain the standard Passivbot outputs.
