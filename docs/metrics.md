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
- `peak_recovery_hours_equity`: Longest time to make a new high on the equity curve.
- `peak_recovery_days_equity`: Same equity recovery duration converted to days.
- `peak_recovery_hours_pnl`: Same calculation on cumulative realized PnL.
- `peak_recovery_days_pnl`: Same realized-PnL recovery duration converted to days.

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
