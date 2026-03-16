# Equity Hard Stop Loss Reference

This file tracks implementation notes, parity surfaces, and remaining edge cases for HSL.

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Risk Management](risk_management.md)
3. [Configuration](configuration.md)

## Current Scope

1. Runtime HSL behavior is side-specific by `pside`.
2. Config lives under:
   - `bot.long.hsl_*`
   - `bot.short.hsl_*`
3. Live and backtest use the same reconstructed strategy-drawdown concept for each `pside`.
4. RED can halt permanently or restart after cooldown, per `pside`.
5. Live startup behavior is reconstructed from exchange-derived history rather than depending on a local latch file.
6. Backtests retain:
   - global account-level HSL metrics under `*_hsl`
   - side-specific HSL metrics under `*_hsl_long` / `*_hsl_short`

## Restart / Statelessness Edge Cases

These should stay under explicit review as HSL evolves:

1. Restart during active RED before flat confirmation on one `pside`
2. Restart after flat confirmation but before cooldown expiry on one `pside`
3. Restart after cooldown expiry, where that `pside` should begin a fresh post-restart regime
4. Restart after a terminal RED stop, which must still block trading on that `pside`
5. Manual trading or balance changes during downtime, which should be reflected purely through fetched exchange state/history
6. Missing or incomplete exchange history rows, which must fail clearly rather than silently changing restart behavior
7. Exchange time skew around cooldown boundaries
8. Restart while panic-close orders are still live on the exchange
9. Restart while a `pside` is flat but stale non-panic close orders remain open
10. Restart after partial manual cleanup, where the account state no longer matches what the bot originally intended

## Backtest / Live Parity Review Items

These are the main parity surfaces that should be reviewed together:

1. Per-`pside` strategy drawdown reconstruction
   - `strategy_pnl`
   - `peak_strategy_equity`
   - `drawdown_raw`
   - `drawdown_ema`
   - `drawdown_score`
2. ORANGE behavior
   - `graceful_stop`
   - `tp_only_with_active_entry_cancellation`
3. RED behavior
   - panic close order type
   - flat confirmation
   - cooldown restart
   - terminal latch
4. Order execution intent
   - Rust emits `limit` vs `market`
   - live routes it
   - backtest simulates it
5. Fee and liquidity semantics
   - maker vs taker fee application
   - `fills.csv` liquidity tagging
6. Liquidation / early termination behavior
   - `backtest.liquidation_threshold`
   - `backtest_completion_ratio`

## Parity Audit Status

### Implemented / Aligned

1. Side-specific runtime HSL
   - Live and backtest both apply HSL independently to long and short `psides`
2. Strategy drawdown concept
   - Live and backtest both use:
     - `strategy_pnl`
     - `peak_strategy_equity`
     - `drawdown_raw`
     - `drawdown_ema`
     - `drawdown_score`
3. ORANGE `graceful_stop`
   - Implemented in both live and backtest
4. ORANGE `tp_only_with_active_entry_cancellation`
   - Live behavior implemented directly
   - Backtest approximates this through `TpOnly`, which is acceptable because the backtest order book is rebuilt every step
5. RED panic order type
   - `hsl_panic_close_order_type` is respected in both live and backtest
6. Market vs limit execution intent
   - Rust emits execution intent
   - Live routes that intent
   - Backtest simulates that intent with slippage and taker fees for market execution
7. Terminal no-restart policy
   - Live and backtest both evaluate `hsl_no_restart_drawdown_threshold` from persistent cross-restart HSL drawdown

### Confirmed Gaps / Risks

1. Flat-confirmation parity
   - Live RED supervisor finalization still depends on live exchange state and open-order cleanup details
   - Backtest finalization remains an approximation of that process
2. Stateless restart coverage is still incomplete
   - Startup reconstruction from exchange-derived history is implemented in live
   - But the edge-case matrix still needs broader regression coverage
3. Global HSL metrics are aggregate diagnostics, not a runtime controller
   - Runtime decisions are made per `pside`
   - Global `*_hsl` metrics are retained for risk inspection and optimizer use

### Missing or Weak Test Coverage

1. Direct live/backtest sample-parity regression for per-`pside` strategy drawdown reconstruction
2. Restart during active RED before flat confirmation
3. Restart while panic-close orders are still open
4. Manual trading during downtime

## Optimizer Work

Recommended HSL-focused optimizer study:

1. Treat `hsl_no_restart_drawdown_threshold` as an operator/runtime control, not a default optimization variable.
2. Use fixed optimize-time overrides:
   - `optimize.fixed_runtime_overrides["bot.long.hsl_no_restart_drawdown_threshold"] = 1.0`
   - `optimize.fixed_runtime_overrides["bot.short.hsl_no_restart_drawdown_threshold"] = 1.0`
3. Tune:
   - `long_hsl_red_threshold`
   - `long_hsl_ema_span_minutes`
   - `long_hsl_cooldown_minutes_after_red`
   - `short_hsl_red_threshold`
   - `short_hsl_ema_span_minutes`
   - `short_hsl_cooldown_minutes_after_red`
4. Constrain:
   - `drawdown_worst_hsl`
   - `drawdown_worst_mean_1pct_hsl`
   - `peak_recovery_hours_hsl`
   - `backtest_completion_ratio`

## Candidate Starting Defaults To Validate

These are candidate regions to test, not final shipped defaults:

1. `hsl_enabled = false`
2. `hsl_red_threshold = 0.22` to `0.25`
3. `hsl_ema_span_minutes = 60`
4. `hsl_cooldown_minutes_after_red = 720` to `1440`
5. `hsl_no_restart_drawdown_threshold = 0.54` to `0.60`
6. `hsl_orange_tier_mode = tp_only_with_active_entry_cancellation`
7. `hsl_panic_close_order_type = limit`

## Remaining Cleanup / Hardening

1. Add stronger direct live/backtest sample-parity regression coverage
2. Enrich user docs with:
   - execution-intent table
   - HSL lifecycle table
   - optimizer recipe
3. Decide final shipped example/default HSL profile
4. Add live/manual validation on a tiny account before merge
