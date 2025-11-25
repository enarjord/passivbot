# Metrics reference

This page documents the main backtest metrics exposed by `passivbot-rust`. Values may appear with
`_usd`/`_btc` suffixes, depending on the denomination used for a run. Metrics without a suffix are
currency-agnostic (e.g., position counts) or already expressed as percentages/ratios.

## Core growth metrics
- `gain`: Terminal equity divided by starting equity (after EMA smoothing of daily equity).
- `adg`: Average daily gain derived from smoothed daily equity (`gain.powf(1 / n_days) - 1`).
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

## Drawdown and tail metrics
- `drawdown_worst`: Maximum absolute drawdown over the equity curve.
- `drawdown_worst_mean_1pct`: Mean of the worst 1% daily drawdowns.
- `expected_shortfall_1pct`: Average loss of the worst 1% daily min-equity returns.

## Exposure, volume, and timing
- `total_wallet_exposure_max/mean/median`: Stats over recorded wallet exposure values.
- `volume_pct_per_day_avg`: Average daily traded notional as a percentage of balance at fill time.
- `positions_held_per_day`: Average number of positions opened per day.
- `position_held_hours_mean/median/max`: Holding-time stats for closed (or still-open) positions.
- `position_unchanged_hours_max`: Longest span with no fills on an open position.
- `peak_recovery_hours_equity`: Longest time to make a new high on the equity curve.
- `peak_recovery_hours_pnl`: Same calculation on cumulative realized PnL.
