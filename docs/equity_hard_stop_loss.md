# Equity Hard Stop Loss

Passivbot includes a side-specific Equity Hard Stop Loss (HSL) that acts as a circuit breaker when strategy drawdown becomes too severe.

HSL is configured separately for each `pside`:

1. `bot.long.hsl_*`
2. `bot.short.hsl_*`

Signal construction is selected globally with `live.hsl_signal_mode`:

1. `pside`
   - long HSL uses long realized/unrealized strategy PnL
   - short HSL uses short realized/unrealized strategy PnL
2. `unified`
   - long and short keep separate HSL controllers
   - both controllers are fed from the same combined account-level strategy signal

### Choosing a Signal Mode

`pside` is the better default in most cases:

1. long HSL reacts to long deterioration
2. short HSL reacts to short deterioration
3. one profitable `pside` cannot hide a weak one
4. side-specific `*_strategy_eq_long` and `*_strategy_eq_short` metrics are easier to interpret

Use `pside` when:

1. long and short are tuned differently
2. one `pside` should be allowed to halt while the other continues
3. you want clearer side-local diagnostics and optimization feedback

`unified` is better when you want whole-account awareness:

1. long and short still keep separate thresholds, cooldowns, and halts
2. both controllers see the same combined account-level strategy signal
3. one profitable `pside` can offset stress on the other in the HSL trigger signal

Use `unified` when:

1. the strategy is intended to behave as one combined book
2. long and short naturally hedge or subsidize each other
3. you want account-level stress on one side to influence the HSL trigger signal on the other side

This is separate from auto-unstuck and the realized-loss gate:

1. Auto-unstuck gradually trims stuck positions while continuing to trade.
2. The realized-loss gate blocks loss-realizing closes below a configured balance floor.
3. HSL is the last-resort supervisory stop. It can switch one `pside` into reduced-risk modes and, at RED, force panic exits and halt only that `pside`.

See also:

1. [Risk Management](risk_management.md)
2. [Configuration](configuration.md)
3. [HSL Reference](equity_hard_stop_loss_reference.md)
4. [HSL Cooldown Contracts](equity_hard_stop_loss_cooldown_contracts.md)

## How It Works

HSL uses a collateral-FX-robust drawdown metric based on reconstructed strategy PnL rather than raw exchange equity peaks.

High level for each `pside` controller:

1. Reconstruct the HSL signal according to `live.hsl_signal_mode`
2. Track a rolling rebased `peak_strategy_equity_pside`
3. Compute raw drawdown and an EMA-smoothed drawdown
4. Use `drawdown_score = min(drawdown_raw, drawdown_ema)` as the trigger metric

This avoids false triggers caused only by collateral price moves when strategy PnL itself has not deteriorated.

## Tiers

Each `pside` has four tiers:

1. `green`: normal trading
2. `yellow`: warning tier
3. `orange`: reduced-risk mode
4. `red`: hard stop

Tier thresholds are derived from that `pside`'s `hsl_red_threshold`:

1. yellow threshold = `hsl_tier_ratios.yellow * hsl_red_threshold`
2. orange threshold = `hsl_tier_ratios.orange * hsl_red_threshold`
3. red threshold = `hsl_red_threshold`

### ORANGE behavior

`hsl_orange_tier_mode` controls what happens in ORANGE for that `pside`:

1. `graceful_stop`
2. `tp_only_with_active_entry_cancellation`

### RED behavior

At RED for one `pside`:

1. that `pside` is forced into panic mode
2. positions on that `pside` are closed using `hsl_panic_close_order_type`
3. the bot waits until all positions on that `pside` are fully closed
4. that `pside` halts
5. optional cooldown-based restart may occur for that `pside`

The opposite `pside` can continue running if its own HSL remains green/orange/yellow.

In both live and backtests, `hsl_no_restart_drawdown_threshold` is evaluated against persistent cross-restart HSL drawdown for that `pside`, not just the local RED-halt snapshot. Values below `hsl_red_threshold` are treated as `hsl_red_threshold`.

### Restart Replay Contract

The intended HSL contract after a valid RED panic is:

1. once a `pside` panic-close has finalized and all positions on that `pside` are fully closed, that RED stop is considered complete
2. that `pside`'s HSL equity tracker is then reset from after that panic
3. any later cooldown, restart, and future RED decisions for that `pside` are measured from the post-panic state, not from pre-panic peaks

This contract applies both to live runtime and to restart-time history replay.

In practical terms, restart replay must treat a historical RED panic close that fully closed all
positions on that `pside` as a completed RED stop even if:

1. the panic-close was split across multiple fills
2. re-entry happened later in the same minute
3. the old pre-panic drawdown would otherwise still be above the RED threshold

Without this reset, a restarted bot could incorrectly inherit stale pre-panic peaks and repanic immediately after cooldown or after restart. `hsl_no_restart_drawdown_threshold` is the intended protection against repeated unfavorable restart cycles, not stale pre-panic equity tracking.

### Live Cooldown Intervention Policy

Live trading has one extra case that backtests do not: a human can open a position on a `pside`
that is currently halted in RED cooldown.

`live.hsl_position_during_cooldown_policy` controls what happens next:

1. `panic`
   - panic-close that position immediately
   - once all positions on that `pside` are fully closed, restart the cooldown timer from that new panic-close
   - this is the safest default
2. `normal`
   - treat the position as an explicit operator override
   - while that `pside` still has no open positions, the bot remains halted and will not open fresh initials on its own
   - clear the halt for that `pside`
   - reset HSL drawdown tracking and rolling-peak state from the current live state
3. `manual`
   - keep the original cooldown deadline
   - leave the position in `manual` mode and do not let the bot resume normal trading on that `pside`
4. `tp_only`
   - keep the original cooldown deadline
   - block new entries on that `pside`
   - allow Passivbot to manage closes only
5. `graceful_stop`
   - keep the original cooldown deadline
   - manage the existing position with `graceful_stop` semantics while still blocking fresh initials

This policy is live-only. Backtests do not model human intervention during cooldown.

## Parameters

Each `pside` has the same HSL parameter set:

1. `hsl_enabled`
   - enables HSL on that `pside`
2. `hsl_red_threshold`
   - RED trigger drawdown score
3. `hsl_ema_span_minutes`
   - EMA span used for smoothed drawdown
   - in backtests, if this is smaller than `backtest.candle_interval_minutes`, HSL falls back to a one-sample EMA, which disables smoothing
4. `hsl_cooldown_minutes_after_red`
   - wait time before auto-restart after RED
   - `0` means no auto-restart
5. `hsl_no_restart_drawdown_threshold`
   - terminal no-restart threshold for that `pside`
   - evaluated against persistent cross-restart HSL drawdown
   - values below `hsl_red_threshold` are clamped to `hsl_red_threshold`
6. `hsl_tier_ratios.yellow`
   - yellow threshold multiplier
7. `hsl_tier_ratios.orange`
   - orange threshold multiplier
8. `hsl_orange_tier_mode`
   - ORANGE behavior selector
9. `hsl_panic_close_order_type`
   - `market` or `limit`

Live-only HSL parameter:

10. `live.hsl_position_during_cooldown_policy`
    - controls how the live bot responds if an open position appears on a halted `pside` during RED cooldown
    - supported values are listed in the section above

## Backtest Behavior

Backtests honor the same side-specific HSL config surface as live.

Important backtest details:

1. `hsl_panic_close_order_type = "limit"`
   - panic exits use the normal backtest crossed-limit execution model
2. `hsl_panic_close_order_type = "market"`
   - panic exits use simulated taker execution on the next bar
   - slippage is controlled by `backtest.market_order_slippage_pct`
3. Backtests export both:
   - operational HSL telemetry under `hard_stop_*`
   - collateral-agnostic strategy-equity risk metrics under `*_strategy_eq`

Main optimizer-facing strategy-equity risk metrics:

1. `drawdown_worst_strategy_eq`
2. `drawdown_worst_ema_strategy_eq`
3. `drawdown_worst_mean_1pct_strategy_eq`
4. `drawdown_worst_mean_1pct_ema_strategy_eq`
5. `peak_recovery_days_strategy_eq`

For the shared EMA-smoothed metrics, long and short each use their own configured
`hsl_ema_span_minutes`. The shared values are reported conservatively as `max(long, short)`
rather than trying to invent one combined EMA span.

Useful side-specific strategy-equity metrics:

1. `drawdown_worst_strategy_eq_long`
2. `drawdown_worst_strategy_eq_short`
3. `drawdown_worst_ema_strategy_eq_long`
4. `drawdown_worst_ema_strategy_eq_short`
5. `drawdown_worst_mean_1pct_strategy_eq_long`
6. `drawdown_worst_mean_1pct_strategy_eq_short`
7. `drawdown_worst_mean_1pct_ema_strategy_eq_long`
8. `drawdown_worst_mean_1pct_ema_strategy_eq_short`
9. `peak_recovery_days_strategy_eq_long`
10. `peak_recovery_days_strategy_eq_short`
11. `hard_stop_triggers_long`
12. `hard_stop_triggers_short`
13. `hard_stop_restarts_long`
14. `hard_stop_restarts_short`

Useful global HSL backtest metrics include:

1. `hard_stop_triggers`
2. `hard_stop_restarts`
3. `hard_stop_triggers_per_year`
4. `hard_stop_restarts_per_year`
5. `hard_stop_restarts_per_year_long`
6. `hard_stop_restarts_per_year_short`
5. `hard_stop_time_in_yellow_pct`
6. `hard_stop_time_in_orange_pct`
7. `hard_stop_time_in_red_pct`
8. `hard_stop_duration_minutes_mean`
9. `hard_stop_duration_minutes_max`
10. `hard_stop_trigger_drawdown_mean`
11. `hard_stop_panic_close_loss_sum`
12. `hard_stop_panic_close_loss_max`
13. `hard_stop_flatten_time_minutes_mean`
14. `hard_stop_post_restart_retrigger_pct`
15. `hard_stop_halt_to_restart_equity_loss_pct`

## Interpreting HSL Metrics

The shared HSL metrics are account-level summaries. They are not split into `_usd` and `_btc`, and
the tier-time metrics are based on the worst active tier across long and short at each sampled
moment.

### Tier-time metrics

1. `hard_stop_time_in_yellow_pct`
   - Fraction of sampled runtime where the account-level HSL tier was YELLOW.
   - Useful as an early-warning “stress frequency” metric.
   - High values mean the bot spends a lot of time near the danger zone even if RED is rare.
2. `hard_stop_time_in_orange_pct`
   - Fraction of sampled runtime where the worst active HSL tier was ORANGE.
   - High values mean the bot often spends time in reduced-risk behavior such as graceful stop or
     TP-only-with-entry-cancellation.
3. `hard_stop_time_in_red_pct`
   - Fraction of sampled runtime where the worst active HSL tier was RED.
   - This is usually the clearest “how often was the system in emergency mode” metric.
   - In practice, you normally want this close to zero. Even small values can be meaningful because
     RED includes forced position closing and halted time.

### Trigger and halt metrics

1. `hard_stop_trigger_drawdown_mean`
   - Mean drawdown score at the moment RED triggers fired.
   - Use this to see whether triggers happen barely above threshold or only after deeper damage.
2. `hard_stop_duration_minutes_mean`
   - Average elapsed minutes from RED halt start until that halt fully ends.
   - Includes position closing and, when enabled, cooldown waiting before restart.
3. `hard_stop_duration_minutes_max`
   - Longest single halt duration observed during the run.
   - Useful for spotting rare but operationally painful stalls.
4. `hard_stop_flatten_time_minutes_mean`
   - Average time from RED trigger until all positions on the triggered `pside` are fully closed.
   - This isolates execution/exit latency from the later cooldown portion of the halt.

### Restart quality metrics

1. `hard_stop_post_restart_retrigger_pct`
   - Fraction of cooldown restarts that later retriggered RED.
   - High values usually mean cooldowns are too short, thresholds are too permissive, or the
     strategy tends to restart into the same adverse regime.
2. `hard_stop_halt_to_restart_equity_loss_pct`
   - Panic-close loss accumulated across HSL events, normalized by starting balance.
   - This estimates how expensive HSL-triggered emergency exits were over the whole run.

As a rule of thumb:

1. `hard_stop_time_in_red_pct` and `hard_stop_post_restart_retrigger_pct` are often the most
   actionable first-pass stability metrics.
2. `hard_stop_trigger_drawdown_mean` helps tune thresholds.
3. `hard_stop_flatten_time_minutes_mean` helps compare `market` vs `limit` panic-close behavior.

## Optimizer Support

Some HSL parameters can be optimized through `optimize.bounds` using side-specific prefixes:

1. `long_hsl_red_threshold`
2. `long_hsl_ema_span_minutes`
3. `long_hsl_cooldown_minutes_after_red`
4. `short_hsl_red_threshold`
5. `short_hsl_ema_span_minutes`
6. `short_hsl_cooldown_minutes_after_red`

`long_hsl_no_restart_drawdown_threshold` and `short_hsl_no_restart_drawdown_threshold` are intentionally not part of the default optimize bounds.

Optimizer runs instead disable terminal no-restart by default through:

1. `optimize.fixed_runtime_overrides["bot.long.hsl_no_restart_drawdown_threshold"] = 1.0`
2. `optimize.fixed_runtime_overrides["bot.short.hsl_no_restart_drawdown_threshold"] = 1.0`

The optimizer should constrain risk through canonical `*_strategy_eq` metrics rather than by terminating candidates early with terminal no-restart.

## Notes

1. Runtime HSL behavior is side-specific by `pside`.
2. Global `*_strategy_eq` metrics are the canonical optimizer and whole-account risk-inspection metrics; deprecated `*_hsl` metric names remain accepted as aliases for older configs/results.
3. HSL is intended as a supervisory backstop, not as a replacement for sane wallet-exposure settings.

## Stateless Restart Behavior

Live HSL startup behavior is reconstructed from exchange-derived account history plus current exchange state. Restart behavior must not depend on any local latch file being present on disk.

For the ongoing edge-case checklist and parity notes, see:

1. [HSL Reference](equity_hard_stop_loss_reference.md)
