# Equity Hard Stop Loss

Passivbot includes an account-level Equity Hard Stop Loss (HSL) that acts as a circuit breaker when strategy drawdown becomes too severe.

The HSL is configured at:

`bot.common.equity_hard_stop_loss`

This is separate from auto-unstuck and the realized-loss gate:

1. Auto-unstuck gradually trims stuck positions while continuing to trade.
2. The realized-loss gate blocks loss-realizing closes below a configured balance floor.
3. HSL is the last-resort account-level stop. It can switch the bot into reduced-risk modes and, at RED, force panic exits and halt trading.

See also:

1. [Risk Management](/Users/eiriknarjord/repos/passivbot-3/docs/risk_management.md)
2. [Configuration](/Users/eiriknarjord/repos/passivbot-3/docs/configuration.md)
3. [HSL Reference](/Users/eiriknarjord/repos/passivbot-3/docs/equity_hard_stop_loss_reference.md)

## How It Works

HSL uses a collateral-FX-robust drawdown metric based on reconstructed strategy PnL rather than raw exchange equity peaks.

High level:

1. Reconstruct `strategy_pnl = realized_pnl + unrealized_pnl`
2. Rebase that against the current balance to produce `strategy_equity`
3. Track a rolling `peak_strategy_equity`
4. Compute raw drawdown and an EMA-smoothed drawdown
5. Use `drawdown_score = min(drawdown_raw, drawdown_ema)` as the trigger metric

This avoids false triggers caused only by collateral price moves when strategy PnL itself has not deteriorated.

## Tiers

HSL has four tiers:

1. `green`: normal trading
2. `yellow`: warning tier
3. `orange`: reduced-risk mode
4. `red`: hard stop

Tier thresholds are derived from `red_threshold`:

1. yellow threshold = `tier_ratios.yellow * red_threshold`
2. orange threshold = `tier_ratios.orange * red_threshold`
3. red threshold = `red_threshold`

### ORANGE behavior

`orange_tier_mode` controls what happens in ORANGE:

1. `graceful_stop`
2. `tp_only_with_active_entry_cancellation`

### RED behavior

At RED:

1. both long and short sides are forced into panic mode
2. the bot closes positions using `panic_close_order_type`
3. the bot waits until the account is flat
4. trading halts
5. during auto-restart cooldown, any new positions stay blocked
6. optional cooldown-based restart may occur

If the trigger drawdown is at or above `no_restart_drawdown_threshold`, the halt becomes terminal and auto-restart is disabled. Values below `red_threshold` are treated as `red_threshold`.

### Cooldown intervention policy

`live.hsl_position_during_cooldown_policy` controls how live runtime handles positions that still exist or reappear while RED cooldown is active:

1. `panic`
   - default
   - forces panic mode again and restarts the cooldown after the account is flat
2. `normal`
   - clears the halt immediately and resumes normal trading
3. `manual`
   - keeps the halt active and leaves the position unmanaged by PB mode forcing
4. `tp_only`
   - keeps the halt active and forces `tp_only` on the open position
5. `graceful_stop`
   - keeps the halt active and forces `graceful_stop` on the open position

## Parameters

All parameters live under `bot.common.equity_hard_stop_loss`:

1. `enabled`
   - enables HSL
2. `red_threshold`
   - RED trigger drawdown score
3. `ema_span_minutes`
   - EMA span used for smoothed drawdown
   - in backtests this must be `>= backtest.candle_interval_minutes`
4. `cooldown_minutes_after_red`
   - wait time before auto-restart after RED
   - `0` means no auto-restart
5. `no_restart_drawdown_threshold`
   - if trigger drawdown is at or above this level, the halt will not auto-restart
   - values below `red_threshold` are clamped to `red_threshold`
6. `tier_ratios.yellow`
   - yellow threshold multiplier
7. `tier_ratios.orange`
   - orange threshold multiplier
8. `orange_tier_mode`
   - ORANGE behavior selector
9. `panic_close_order_type`
   - `market` or `limit`

Live-only parameter under `live`:

1. `hsl_position_during_cooldown_policy`
   - `normal`, `panic`, `manual`, `tp_only`, or `graceful_stop`
   - default is `panic`

## Backtest Behavior

Backtests honor the same HSL config as live.

Important backtest details:

1. `panic_close_order_type = "limit"`
   - panic exits use the normal backtest crossed-limit execution model
2. `panic_close_order_type = "market"`
   - panic exits use simulated taker execution on the next bar
   - slippage is controlled by `backtest.panic_market_slippage_pct`
3. HSL metrics are exported as shared account metrics, not split into `_usd` and `_btc`

Useful HSL backtest metrics include:

1. `hard_stop_triggers`
2. `hard_stop_restarts`
3. `hard_stop_time_in_yellow_pct`
4. `hard_stop_time_in_orange_pct`
5. `hard_stop_time_in_red_pct`
6. `hard_stop_duration_minutes_mean`
7. `hard_stop_duration_minutes_max`
8. `hard_stop_trigger_drawdown_mean`
9. `hard_stop_panic_close_loss_sum`
10. `hard_stop_panic_close_loss_max`
11. `hard_stop_flatten_time_minutes_mean`
12. `hard_stop_post_restart_retrigger_pct`
13. `hard_stop_halt_to_restart_equity_loss_pct`

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
     RED includes forced flattening and halted time.

### Trigger and halt metrics

1. `hard_stop_trigger_drawdown_mean`
   - Mean drawdown score at the moment RED triggers fired.
   - Use this to see whether triggers happen barely above threshold or only after deeper damage.
2. `hard_stop_duration_minutes_mean`
   - Average elapsed minutes from RED halt start until that halt fully ends.
   - Includes flattening and, when enabled, cooldown waiting before restart.
3. `hard_stop_duration_minutes_max`
   - Longest single halt duration observed during the run.
   - Useful for spotting rare but operationally painful stalls.
4. `hard_stop_flatten_time_minutes_mean`
   - Average time from RED trigger to fully flat position state.
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

HSL parameters can be optimized through `optimize.bounds` using the `common_` prefix:

1. `common_equity_hard_stop_loss_red_threshold`
2. `common_equity_hard_stop_loss_ema_span_minutes`
3. `common_equity_hard_stop_loss_cooldown_minutes_after_red`
4. `common_equity_hard_stop_loss_no_restart_drawdown_threshold`

The HSL backtest metrics listed above can also be used in:

1. `optimize.scoring`
2. `optimize.limits`

## Notes

1. HSL is currently account-level, not per-position-side.
2. HSL is intended as a supervisory backstop, not as a replacement for sane wallet-exposure settings.

## Stateless Restart Behavior

Live HSL startup behavior is reconstructed from exchange-derived account history plus current exchange state. Restart behavior must not depend on any local latch file being present on disk.

For the ongoing edge-case checklist and remaining implementation work, see:

1. [HSL Reference](/Users/eiriknarjord/repos/passivbot-3/docs/equity_hard_stop_loss_reference.md)
