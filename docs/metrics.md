# Metrics reference

This page documents the main backtest metrics exposed by `passivbot-rust`. Values may appear with
`_usd`/`_btc` suffixes. `_usd` metrics are computed from USD-denominated balance/equity, while
`_btc` metrics are computed from BTC-denominated balance/equity even when the backtest never holds
BTC collateral. Metrics without a suffix are currency-agnostic (e.g., position counts) or already
expressed as percentages/ratios.

## Core growth metrics

- `gain`: Terminal equity divided by starting equity, where terminal equity is the mean of the
  last up to three daily equity values.
- `adg`: Average daily gain derived from that smoothed terminal equity (`gain.powf(1 / n_days) - 1`).
- `adg_w`: Mean of `adg` computed on the trailing 10% slices (full run, last half, last third, …).
- `adg_rolling_hmean_strategy_eq`: Path-sensitive growth on collateral-agnostic strategy equity.
  For each automatic horizon `h`, it computes every complete rolling growth factor
  `G(t,h) = equity(t) / equity(t-h)`, then calculates
  `R(h) = harmonic_mean(G(t,h))^(1/h) - 1`. The final metric is the geometric mean of
  `1 + R(h)`, minus one, so each horizon contributes equally in daily log-growth space. The automatic
  horizons retain roughly 48, 16, and 8 non-overlapping-window equivalents and are capped at 30,
  90, and 180 days; histories of about four years or more therefore use exactly 30/90/180 days.
  The harmonic mean makes persistently weak windows matter more than isolated windfalls. Keep
  terminal `adg_strategy_eq` as a separate objective because overlapping rolling windows give the
  ends of the backtest less coverage than its middle.
- `adg_time_integrated_strategy_eq`: Dailyized area under log strategy equity relative to its
  starting value: `exp(2 * trapezoid_sum(log(equity/start)) / days^2) - 1`. It equals ordinary ADG
  on a perfectly exponential curve, rewards gains that arrive earlier, and penalizes equity that
  spends much of the run below its start. Use it together with terminal ADG: an early windfall
  followed by deterioration can still have positive area.
- `positive_gain_participation_strategy_eq`: Effective participation of positive daily log gains,
  normalized to `[0, 1]`: `(sum(p)^2) / (N * sum(p^2))`, where
  `p = max(log(equity(t) / equity(t-1)), 0)`. If positive gain is spread equally across `k` of `N`
  daily intervals, the score is `k / N`; concentration in a few unusually large positive days
  lowers it further. The metric deliberately ignores negative returns, leaving their magnitude and
  duration to Sortino and drawdown objectives, and should be paired with a gain objective so tiny
  frequent gains cannot win on participation alone.

For example, over 240 daily intervals, equal positive gains on all 240 days score `1.0`; equal
positive gains on 120, 24, or one day score `0.5`, `0.1`, or about `0.0042`. Unequal gains reduce
the score further. Two curves can therefore finish at the same equity while participation strongly
prefers the one whose gains were broadly shared. Rolling-harmonic ADG asks a different question:
whether growth remains sound across many possible 30/90/180-day start and end points.

- `adg_pnl`: Collateral-agnostic daily PnL ratio. For each day, sum all `pnl` and divide by that
  day’s last recorded `usd_total_balance`, then average those daily ratios across the run.
- `adg_pnl_w`: Weighted version of `adg_pnl` using the same 10-slice trailing averaging as `adg_w`.
- `mdg`: Median of daily percentage equity changes.
- `mdg_w`: Weighted version of `mdg` across the trailing slices.
- `mdg_pnl`: Median of the collateral-agnostic daily PnL ratios (same daily ratios as `adg_pnl`,
  aggregated via median instead of mean).
- `mdg_pnl_w`: Weighted version of `mdg_pnl` across the trailing slices.
- `sharpe_ratio_pnl`: Sharpe on the collateral-agnostic daily PnL ratios (`adg_pnl` divided by the
  standard deviation of those daily ratios).
- `sortino_ratio_pnl`: Sortino on the same daily PnL ratios (`adg_pnl` divided by downside
  deviation of negative daily ratios).

Note: Sharpe/Sortino on equity (`sharpe_ratio`, `sortino_ratio`) use daily equity returns
(mark-to-market, including unrealized swings), so their variance usually reflects BTC collateral and
intra-day volatility. The PnL variants (`*_pnl`) use realized PnL ratios divided by end-of-day
balance, which often yields lower variance (and fewer negative days), so the ratios can be higher
and more stable across collateral caps.

## Risk/return ratios
- `sharpe_ratio`: `adg` divided by the standard deviation of daily min-equity returns.
- `sortino_ratio`: `adg` divided by downside deviation (only negative daily min-equity returns).
- `omega_ratio`: Sum of positive daily returns divided by the absolute sum of negative daily returns.
- `sterling_ratio`: `adg` divided by the average of the worst 1% drawdowns.
- `calmar_ratio`: `adg` divided by the worst drawdown observed over the full equity curve.
- `paper_loss_ratio`: `adg` divided by the worst absolute negative equity-vs-balance gap.
- `paper_loss_mean_ratio`: `adg` divided by the mean absolute negative equity-vs-balance gap.
- `exposure_ratio`: `adg` divided by the maximum absolute recorded wallet exposure.
- `exposure_mean_ratio`: `adg` divided by the mean absolute recorded wallet exposure.

Weighted `_w` variants use the same trailing-slice averaging as the rest of the `_w` metrics.

## Drawdown and tail metrics
- `drawdown_worst`: Maximum absolute drawdown over the equity curve.
- `drawdown_worst_mean_1pct`: Mean of the worst 1% daily worst drawdowns, where drawdown is computed from the full-resolution equity curve before reducing each day to its worst underwater point.
- `expected_shortfall_1pct`: Average loss of the worst 1% daily min-equity returns.

## HSL metrics
- `hard_stop_triggers`: Absolute count of RED trigger events during the run.
- `hard_stop_restarts`: Absolute count of cooldown restarts after RED halts.
- `hard_stop_total_loss_pct`: Total panic-close loss as a fraction of starting balance.
- `hard_stop_triggers_per_year`: `hard_stop_triggers / n_days * 365.25`.
- `hard_stop_restarts_per_year`: `hard_stop_restarts / n_days * 365.25`.

## Exposure, volume, and timing
- `total_wallet_exposure_max/mean/median`: Stats over recorded wallet exposure values.
- `volume_pct_per_day_avg`: Average daily traded notional as a percentage of balance at fill time.
- `positions_held_per_day`: Average number of positions opened per day.
- `position_held_hours_mean/median/max`: Holding-time stats for closed (or still-open) positions.
- `position_held_days_mean/median/max`: Same holding-time stats converted to days.
- `position_unchanged_hours_max`: Longest span with no fills on an open position.
- `position_unchanged_days_max`: Same unchanged-position span converted to days.
- `fills_gap_time_weighted_mean_hours`: Time-weighted mean portfolio no-fill gap. Unique fill
  timestamps split the full analysis window, including its leading and trailing boundaries, into
  gaps `g`; the metric is `sum(g^2) / sum(g)`. A randomly selected moment is therefore weighted by
  the length of the gap containing it, so long droughts contribute more strongly than clustered
  fills. A zero-fill run equals the full analysis duration.
- `peak_recovery_hours_equity`: Longest time to make a new high on the equity curve.
- `peak_recovery_days_equity`: Same equity recovery duration converted to days.
- `peak_recovery_hours_pnl`: Same calculation on cumulative realized PnL.
- `peak_recovery_days_pnl`: Same realized-PnL recovery duration converted to days.
- `strategy_eq_recovery_days_*`: Distribution of per-sample strategy-equity recovery durations. For each strategy-equity sample, recovery is the time until a later sample strictly exceeds that equity; samples not exceeded by the end use the open tail to the final timestamp. Available summaries are `mean`, `median`, `p95`, `p99`, `mean_worst_5pct`, `mean_worst_1pct`, and `max`.
- `peak_recovery_days_strategy_eq`: Legacy alias for `strategy_eq_recovery_days_max`.

## Trade-level metrics
- `win_rate`: Fraction of completed trades with positive net realized PnL.
- `win_rate_w`: Mean `win_rate` across the same trailing-slice weighted analysis used for other
  `_w` metrics.
- `trade_loss_max`: Worst completed-trade loss as a fraction of the account balance at trade open.
- `trade_loss_mean`: Mean losing-trade loss fraction in that same unit.
- `trade_loss_median`: Median losing-trade loss fraction in that same unit.

A completed trade is one full position lifecycle from open to flat for a single `coin + side`.
Realized PnL is accumulated from `fill.pnl` over that lifecycle. Positions that remain open at the
end of the backtest are excluded from these trade-level metrics.
