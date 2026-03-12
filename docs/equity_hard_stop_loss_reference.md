# Equity Hard Stop Loss Reference

This file tracks implementation notes, open edge cases, and remaining work for the account-level Equity Hard Stop Loss (HSL).

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Risk Management](risk_management.md)
3. [Configuration](configuration.md)

## Current Scope

1. HSL is account-level.
2. Config lives under `bot.common.equity_hard_stop_loss`.
3. Live and backtest use the same reconstructed strategy-drawdown concept.
4. RED can halt permanently or restart after cooldown.
5. Live startup behavior is reconstructed from exchange-derived history rather than depending on a local latch file.

## Restart / Statelessness Edge Cases

These should stay under explicit review as HSL evolves:

1. Restart during active RED before flat confirmation
2. Restart after flat confirmation but before cooldown expiry
3. Restart after cooldown expiry, where HSL should begin a fresh post-restart regime
4. Restart after a terminal RED stop, which must still block trading
5. Manual trading or balance changes during downtime, which should be reflected purely through fetched exchange state/history
6. Missing or incomplete exchange history rows, which must fail clearly rather than silently changing restart behavior
7. Exchange time skew around cooldown boundaries
8. Restart while panic-close orders are still live on the exchange
9. Restart while positions are flat but stale non-panic close orders remain open
10. Restart after partial manual cleanup, where the account state no longer matches what the bot originally intended

## Backtest / Live Parity Review Items

These are the main parity surfaces that should be reviewed together:

1. Strategy drawdown reconstruction
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

1. Strategy drawdown concept
   - Live and backtest both use the same HSL vocabulary and the same conceptual metric family:
     - `strategy_pnl`
     - `peak_strategy_equity`
     - `drawdown_raw`
     - `drawdown_ema`
     - `drawdown_score`
2. ORANGE `graceful_stop`
   - Implemented in both live and backtest
3. ORANGE `tp_only_with_active_entry_cancellation`
   - Live behavior implemented directly
   - Backtest approximates this through `TpOnly`, which is acceptable because the backtest order book is rebuilt every step
4. RED panic order type
   - `panic_close_order_type` is respected in both live and backtest
5. Market vs limit execution intent
   - Rust now emits execution intent
   - Live routes that intent
   - Backtest simulates that intent with slippage and taker fees for market execution
6. HSL restart baseline reset
   - Live and backtest both reset HSL peak state after cooldown restart

7. Terminal no-restart policy differs today
   - Live still decides terminal latch from the finalized RED-stop drawdown snapshot
   - Backtest decides it from persistent cross-restart HSL drawdown
   - This difference should stay visible in parity reviews

### Confirmed Gaps / Risks

1. Flat-confirmation parity
   - Live RED supervisor blocks finalization on:
     - open positions
     - entry orders
     - non-panic close orders
   - Backtest currently blocks finalization on any open order
   - This may make backtest stricter than live if panic close orders remain open
2. Stateless restart coverage is still incomplete
   - Startup reconstruction from exchange-derived history is now implemented in live
   - But the edge-case matrix still needs broader regression coverage
3. HSL metrics export path is internally redundant
   - Rust still fills HSL metrics into both USD and BTC analysis structs
   - Python later deduplicates these shared account metrics
   - Functionally correct, but unnecessary duplication remains in the export path

### Missing or Weak Test Coverage

1. Direct live/backtest sample-parity regression for strategy drawdown reconstruction
   - no test currently feeds the same synthetic history through both paths and compares tier / drawdown evolution
2. Restart during active RED before flat confirmation
   - needs explicit live startup regression coverage
3. Restart while panic-close orders are still open
   - needs explicit live startup regression coverage
4. Manual trading during downtime
   - needs explicit reconstruction tests to verify behavior comes only from fetched exchange state/history

## Optimizer Work

Recommended HSL-focused optimizer study:

1. Treat `no_restart_drawdown_threshold` as an operator/runtime control, not a default optimization variable.
2. Use fixed optimize-time override:
   - `optimize.fixed_runtime_overrides["bot.common.equity_hard_stop_loss.no_restart_drawdown_threshold"] = 1.0`
3. Tune:
   - `common_equity_hard_stop_loss_red_threshold`
   - `common_equity_hard_stop_loss_ema_span_minutes`
   - `common_equity_hard_stop_loss_cooldown_minutes_after_red`
4. Constrain:
   - `drawdown_worst_hsl`
   - `drawdown_worst_mean_1pct_hsl`
   - `peak_recovery_hours_hsl`
   - `backtest_completion_ratio`

## Candidate Starting Defaults To Validate

These are candidate regions to test, not final shipped defaults:

1. `enabled = false`
2. `red_threshold = 0.22` to `0.25`
3. `ema_span_minutes = 60`
4. `cooldown_minutes_after_red = 720` to `1440`
5. `no_restart_drawdown_threshold = 0.54` to `0.60`
6. `orange_tier_mode = tp_only_with_active_entry_cancellation`
7. `panic_close_order_type = limit`

## Remaining Cleanup / Hardening

1. Complete a full live/backtest parity checklist review
2. Harden `src/rust_utils.py` logging and tests around stale-extension detection
3. Enrich user docs with:
   - execution-intent table
   - HSL lifecycle table
   - optimizer recipe
4. Decide final shipped example/default HSL profile
5. Add live/manual validation on a tiny account before merge

## Explicitly Deferred

1. Per-position-side HSL
2. Further config migration into `bot.common`
3. Merge review until the hardening pass is complete
