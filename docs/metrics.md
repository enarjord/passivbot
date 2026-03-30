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

## Strategy-PnL rebased metrics
- `gain_strategy_pnl_rebased`: Growth on the synthetic collateral-agnostic equity curve
  `starting_balance + strategy_pnl`, where `strategy_pnl = realized_pnl + unrealized_pnl`.
- `adg_strategy_pnl_rebased`: Geometric daily growth on that rebased curve, using the same terminal
  smoothing as `adg`.
- `adg_strategy_pnl_rebased_w`: Recency-weighted version of `adg_strategy_pnl_rebased`.
- `mdg_strategy_pnl_rebased`: Median daily percentage change of the rebased curve.
- `mdg_strategy_pnl_rebased_w`: Recency-weighted version of `mdg_strategy_pnl_rebased`.
- `sharpe_ratio_strategy_pnl_rebased`, `sortino_ratio_strategy_pnl_rebased`,
  `omega_ratio_strategy_pnl_rebased`, `calmar_ratio_strategy_pnl_rebased`,
  `sterling_ratio_strategy_pnl_rebased`: Ratio family computed from the same rebased curve.
- Weighted `_w` variants are available for the main ratio metrics above.
- `expected_shortfall_1pct_strategy_pnl_rebased`: Tail-loss statistic on the rebased daily series.

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

## HSL risk metrics
- `drawdown_worst_hsl`: Worst account-level HSL drawdown sample.
- `drawdown_worst_ema_hsl`: Worst EMA-smoothed HSL drawdown. Since long and short may use
  different `hsl_ema_span_minutes`, the shared metric is `max(drawdown_worst_ema_hsl_long,
  drawdown_worst_ema_hsl_short)`.
- `drawdown_worst_mean_1pct_hsl`: Mean of the worst 1% HSL drawdown samples.
- `drawdown_worst_mean_1pct_ema_hsl`: Mean of the worst 1% EMA-smoothed HSL drawdown samples,
  using the same conservative shared `max(long, short)` rule.
- `peak_recovery_hours_hsl`: Longest time spent below the all-time rebased HSL peak before
  recovery. This uses the all-time rebased peak rather than the rolling trigger window.
- `hard_stop_triggers`: Absolute count of RED trigger events during the run.
- `hard_stop_restarts`: Absolute count of cooldown restarts after RED halts.
- `hard_stop_triggers_per_year`: `hard_stop_triggers / n_days * 365.25`.
- `hard_stop_restarts_per_year`: `hard_stop_restarts / n_days * 365.25`.
- `hard_stop_restarts_per_year_long`: `hard_stop_restarts_long / n_days * 365.25`.
- `hard_stop_restarts_per_year_short`: `hard_stop_restarts_short / n_days * 365.25`.

## Exposure, volume, and timing
- `total_wallet_exposure_max/mean/median`: Stats over recorded wallet exposure values.
- `volume_pct_per_day_avg`: Average daily traded notional as a percentage of balance at fill time.
- `positions_held_per_day`: Average number of positions opened per day.
- `position_held_hours_mean/median/max`: Holding-time stats for closed (or still-open) positions.
- `position_unchanged_hours_max`: Longest span with no fills on an open position.
- `peak_recovery_hours_equity`: Longest time to make a new high on the equity curve.
- `peak_recovery_hours_pnl`: Same calculation on cumulative realized PnL.

## Visible metrics in standalone backtests

`backtest.visible_metrics` controls which metrics are printed to the terminal after a standalone
backtest:

- `null`: show the metrics implied by `optimize.scoring` and `optimize.limits`
- `[]`: show all metrics
- `["metric_a", "metric_b"]`: show optimize-derived metrics plus the explicitly listed ones

This only affects CLI visibility. The full metric set is still computed and persisted.
