# Equity Hard Stop Loss Reference

This file tracks implementation notes, parity surfaces, and remaining edge cases for HSL.

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Risk Management](risk_management.md)
3. [Configuration](configuration.md)
4. [HSL Cooldown Contracts](equity_hard_stop_loss_cooldown_contracts.md)

## Current Scope

1. Runtime HSL behavior is side-specific by `pside`, with optional `coin` mode creating per-`coin+pside` controllers.
2. Config lives under:
   - `bot.long.hsl.*`
   - `bot.short.hsl.*`
3. `live.hsl_signal_mode` selects whether those `pside` runtimes use:
   - one shared `unified` signal
   - side-local `pside` signals
   - coin-local `coin` signals (default)
4. Live and backtest use the same reconstructed strategy-drawdown concept for each `pside`; `coin` mode instead uses realized PnL cumsum drawdown plus current UPnL divided by the configured slot budget.
5. RED can halt permanently or restart after cooldown, per `pside`.
6. Live startup behavior is reconstructed from exchange-derived history rather than depending on a local latch file.
7. Backtests export:
   - global account-level strategy-equity metrics under `*_strategy_eq`
   - side-specific strategy-equity metrics under `*_strategy_eq_long` / `*_strategy_eq_short`

## Startup And Config-Change Contract

HSL is stateless trading behavior. Live startup may use local caches or
checkpoints to speed reconstruction, but the authoritative inputs are exchange
state, fill history, candle history, config, and current time. Cache contents
must never be required to decide whether HSL is RED, in cooldown, or terminal.

Important consequences:

1. Enabling HSL on an account with existing positions can trigger immediate
   panic closes if reconstructed current-episode drawdown is RED.
2. `live.pnls_max_lookback_days` is shared by HSL, realized-loss gating,
   auto-unstuck allowance, plotting, and backtests. It is a memory horizon, not
   a process-local lifetime.
3. Changing HSL thresholds, signal mode, `n_positions`, TWEL, excess allowance,
   or lookback length can retroactively change reconstructed RED/cooldown/
   no-restart outcomes. Treat those edits as risk-policy migrations.
4. Live `coin` mode uses configured `n_positions` for slot-budget sensitivity.
   It must not silently switch to dynamic live coin eligibility; backtests may
   use dynamic historical effective n-positions only behind an explicit option.
5. Missing fill or candle coverage must be visible and fail/defer rather than
   creating neutral drawdown values.
6. Performance caches are allowed only when they can be invalidated safely and
   extended from exchange-derived history without changing trading decisions.

## Restart / Statelessness Edge Cases

These should stay under explicit review as HSL evolves:

1. Restart during active RED before all positions on one `pside` are fully closed
2. Restart after all positions on one `pside` are fully closed but before cooldown expiry
3. Restart after cooldown expiry, where that `pside` should begin a fresh post-restart regime
4. Restart after a terminal RED stop, which must still block trading on that `pside`
5. Manual trading or balance changes during downtime, which should be reflected purely through fetched exchange state/history
6. Missing or incomplete exchange history rows, which must fail clearly rather than silently changing restart behavior
7. Exchange time skew around cooldown boundaries
8. Restart while panic-close orders are still live on the exchange
9. Restart while a `pside` has no open positions but stale non-panic close orders remain open
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
   - confirmation that all positions on the triggered `pside`, or the triggered `coin+pside` in `coin` mode, are fully closed
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
8. Rust-owned HSL primitives
   - Live orchestration and exact backtests share the Rust drawdown-step, coin-signal, and RED-episode-finalization contracts
   - Apple MPS screening remains approximate and is always revalidated by exact Rust
9. Apple MPS controller
   - One shared Rust-owned Metal HSL controller is composed into both EMA Anchor and Trailing Martingale directional kernels
   - Unified, pside, and coin signal modes use explicit encoded identities instead of a coin/non-coin boolean
   - Deterministic M3 conformance coverage compares Metal drawdown, EMA, tier, active-RED, latch, and RED-finalization traces against the exact Rust runtime for all three signal modes and restart policies
   - One-sided and dual-side single-coin runs support unified, pside, and coin signals; unified controllers share account PnL and require both sides to be flat, while pside and coin retain directional realized-PnL and flatness state
   - One-sided multi-coin runs support unified and pside signals through one shared-balance portfolio controller that tracks all positions and blocking orders on the enabled side
   - One-sided multi-coin coin signals use one independent controller per coin, each with coin-local realized net PnL, drawdown EMA, warning/RED episode, panic close, halt, and restart state; the coin drawdown budget uses the dynamic effective position-slot count
   - Coin-mode lifecycle and panic-loss metrics aggregate across coin episodes, while warning-tier time samples the worst active coin tier once per minute, matching exact Rust reporting
   - Multi-coin limit and market panic closes flatten every open coin on the enabled side and export the same lifecycle and panic-loss metric surface as single-coin HSL; market execution uses each coin's taker fee and directionally quantized configured slippage
   - One-sided multi-coin coin mode resolves all ten canonical per-coin HSL settings independently, including enablement, thresholds, restart/tier behavior, and limit/market panic execution; compatible suites may supply scenario-local overrides
   - Dual-side multi-coin EMA Anchor uses a fused shared-account strategy kernel for unified, pside, and coin signals, including shared lifecycle/panic-loss metrics and per-coin HSL overrides in coin mode
   - Dual-side multi-coin Trailing Martingale uses a fused shared-account strategy kernel for unified, pside, and coin signals, including shared lifecycle/panic-loss metrics and per-coin HSL overrides in coin mode
   - Apple MPS exports the worst EMA-smoothed strategy-equity drawdown for long and short HSL controllers; the account metric is the same `max(long, short)` reduction as exact Rust
   - Apple MPS exports the raw worst strategy-equity drawdown for each long and short HSL controller through opt-in bounded peak-and-maximum state; exact Rust validation remains authoritative for proxy-fill drift
   - Apple MPS exports the raw mean-worst-1% daily strategy-equity drawdown for each long and short HSL controller through opt-in daily-worst state and a bounded logarithmic tail histogram; exact Rust validation and rolling drift gates remain authoritative
   - Apple MPS exports each HSL controller's longest strict strategy-equity time-to-exceed interval as `peak_recovery_{hours,days}_strategy_eq_{long,short}`, including an unrecovered tail through the backtest end

### Confirmed Gaps / Risks

1. All-positions-closed confirmation parity
   - Live RED supervision necessarily depends on exchange position state, fill evidence, and open-order cleanup
   - Exact backtests model deterministic fills and therefore cannot prove connector-specific exchange races
2. Global strategy-equity metrics are aggregate diagnostics, not a runtime controller
   - Runtime decisions are made per `pside`
   - Global `*_strategy_eq` metrics are canonical for risk inspection and optimizer use
   - Deprecated `*_hsl` metric names remain accepted as aliases for older configs/results
3. Remaining GPU HSL metric gaps
   - The per-side HSL controllers expose their longest strict strategy-equity recovery interval,
     but not a full per-side recovery distribution
   - Global strategy-equity recovery distributions are available for single-coin EMA Anchor and
     Trailing Martingale through opt-in hourly proxy samples; multi-coin portfolio kernels do not
     yet emit equivalent samples and fail closed for those metrics

### Missing or Weak Test Coverage

1. End-to-end replay of one identical fill/candle history through live reconstruction and exact backtest orchestration; shared Rust primitive tests cover the calculations but not the complete orchestration trace
2. Connector-level restart races while protective panic-close orders are live on an exchange
3. Manual or external trading during downtime across the full exchange-adapter matrix
4. Apple MPS parity for per-side HSL-controller recovery distributions and global multi-coin
   strategy-equity recovery distributions

## Optimizer Work

Recommended HSL-focused optimizer study:

1. Treat `hsl_no_restart_drawdown_threshold` as an operator/runtime control, not a default optimization variable.
2. Use fixed optimize-time overrides:
   - `optimize.fixed_runtime_overrides["bot.long.hsl.no_restart_drawdown_threshold"] = 1.0`
   - `optimize.fixed_runtime_overrides["bot.short.hsl.no_restart_drawdown_threshold"] = 1.0`
3. Tune:
   - `long_hsl_red_threshold`
   - `long_hsl_ema_span_minutes`
   - `long_hsl_cooldown_minutes_after_red`
   - `short_hsl_red_threshold`
   - `short_hsl_ema_span_minutes`
   - `short_hsl_cooldown_minutes_after_red`
4. Constrain:
   - `drawdown_worst_strategy_eq`
   - `drawdown_worst_mean_1pct_strategy_eq`
   - `strategy_eq_recovery_days_max`
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

1. Add an end-to-end live-replay/exact-backtest orchestration parity fixture
2. Enrich user docs with:
   - execution-intent table
   - HSL lifecycle table
   - optimizer recipe
3. Decide final shipped example/default HSL profile
4. Add explicitly authorized live/manual validation on a tiny account before declaring exchange-level operational proof
5. Extend the shared Metal controller only through fail-closed, exact-Rust-validated topology slices
