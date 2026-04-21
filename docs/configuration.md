# Passivbot Parameters Explanation

This document explains the canonical config schema used by Passivbot.

- The source of truth for defaults is `src/config/schema.py`.
- The example config `configs/examples/default_trailing_grid_long_npos10.json` mirrors those hardcoded defaults exactly.
- If you omit `config_path`, Passivbot loads those in-code defaults.

For the recommended user workflow, examples, and best practices, see [Config Workflow](config_workflow.md).

## Backtest Settings

- **base_dir**: Location to save backtest results.
- **compress_cache**: Set to `true` to save disk space. Set to `false` for faster loading.
- **end_date**: End date of backtest, e.g., `2024-06-23`. Set to `'now'` to use today's date as the end date.
- **exchanges**: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing. Supported exchanges include `binance`, `bybit`, `gateio`, and `bitget`. The current default profile uses `['binance', 'bybit']`.
  **GateIO note:** If you already have `caches/ohlcv/gateio` data on disk, delete it before a fresh run so Passivbot rebuilds the cache with base-volume-normalized data.
- **coin_sources**: Optional mapping of `coin -> exchange` used to override the automatic exchange selection when multiple exchanges are configured. Scenarios may add more overrides; conflicting assignments raise an error.
- **market_settings_sources**: Optional mapping of `coin -> exchange` used specifically for exchange metadata such as `price_step`, `qty_step`, fees, and min-size rules. This is separate from `coin_sources`: you may source candles from one exchange while borrowing market settings from another.
- **ohlcv_source_dir**: Optional path to a pre-populated OHLCV directory to use before hitting exchange archives. Expected structure: `<dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npz` or `.npy`. Coin keys are normalized to base coins, but CCXT-style symbol folder names are accepted (e.g., `ETH_USDC:USDC`).
- **volume_normalization**: When `true` (default), normalize volume data across exchanges to make combined datasets comparable.
- **start_date**: Start date of backtest.
- **starting_balance**: Starting balance in USD at the beginning of the backtest.
- **filter_by_min_effective_cost**: When `true`, skip coins whose projected initial entry
  (balance × wallet_exposure_limit × entry_initial_qty_pct, including WE excess allowance)
  would fall below the exchange’s effective minimum cost.
- **dynamic_wel_by_tradability**: Backtest-only WEL denominator mode.  
  - `true` (default): `wallet_exposure_limit = total_wallet_exposure_limit / min(n_positions, n_tradable_max)` where `n_tradable_max` is the highest number of coins that have had real candles at any timestep so far (non-shrinking).  
  - `false`: fixed denominator, same as live: `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
- **candle_interval_minutes**: Aggregates raw 1m OHLCVs into coarser candles before the backtest loop runs. `1` keeps native 1m behavior; values above `1` speed up backtests and optimizer runs at the cost of losing intra-interval fill ordering.
- **gap_tolerance_ohlcvs_minutes**: Maximum tolerated hole size in prepared OHLCV data before the dataset is considered broken for that coin/exchange. Larger values accept sparser historical data; smaller values fail sooner on archive gaps.
- **liquidation_threshold**: Early-stop backtest equity-floor guard. The run terminates once total equity falls to or below `starting_balance * liquidation_threshold`, and `backtest_completion_ratio` will fall below `1.0`. Example: with `starting_balance = 1000` and `liquidation_threshold = 0.05`, the backtest stops at equity `<= 50`. This is not a “5% drawdown” threshold; if the run never rises above the start, it corresponds to roughly a `0.95` worst drawdown. Must satisfy `0.0 <= liquidation_threshold < 1.0`.
- **maker_fee_override**: Optional maker fee override (part-per-one; use `0.0002` for 0.02%). Leave `null` to use the exchange-derived maker fees.
- **taker_fee_override**: Optional taker fee override (part-per-one; use `0.00055` for 0.055%). Leave `null` to use the exchange-derived taker fees.
- **market_order_slippage_pct**: Backtest-only slippage applied whenever the backtester simulates market-order execution. This applies both to HSL panic closes when `bot.{long,short}.hsl_panic_close_order_type` is `"market"` and to normal orchestrator orders promoted to market execution by `live.market_orders_allowed`. A sell fills at `close * (1 - slippage_pct)` rounded down to `price_step`; a buy fills at `close * (1 + slippage_pct)` rounded up. The fill is guaranteed once the market-execution path is chosen, and the resulting fill also uses taker fees. Default `0.0005` (5 bps).
- **visible_metrics**: Controls which metrics are printed to the terminal after a standalone backtest. `null` shows the metrics implied by `optimize.scoring` and `optimize.limits`, `[]` shows all metrics, and an explicit list adds extra named metrics to the default view. This affects CLI visibility only; the full metric set is still computed and persisted.
- **config_version**: Top-level schema version string for the config file. Canonical `v7.9` configs use `v7.9.0`. Older configs without this field are treated as legacy and migrated during load.
- **balance_sample_divider**: Minutes per bucket when sampling balances/equity for
  `balance_and_equity.csv` and related plots. `1` keeps full per-minute resolution; higher values
  thin out the series (e.g., `15` stores one point every 15 minutes) to reduce file sizes.
- **btc_collateral_cap**: Target (and ceiling) share of account equity to hold in BTC collateral. `0` keeps the account fully in USD; `1.0` targets fully-BTC collateral; values `>1` allow leveraged BTC collateral, accepting negative USD balances.
- **btc_collateral_ltv_cap**: Optional loan-to-value ceiling (`USD debt ÷ equity`) enforced when topping up BTC. Leave `null` (default) to allow unlimited debt, or set to a float (e.g., `0.6`) to stop buying BTC once leverage exceeds that threshold.
### Suite Scenarios

Suite configuration uses a flattened structure directly under `backtest`:

- **backtest.suite_enabled**: Master switch for suite runs (`--suite [y/n]` overrides it at runtime). Default `false` in the canonical schema and example config.
- **backtest.scenarios**: List of scenario dicts. Supported per-scenario keys:
  - `label`: Directory name under `backtests/suite_runs/<timestamp>/`.
  - `start_date`, `end_date`: Override the global date window.
  - `coins`, `ignored_coins`: Restrict or skip symbols.
  - `exchanges`: Limit which exchanges can contribute data to this scenario.
  - `coin_sources`: Scenario-specific overrides for `coin_sources`.
  - `overrides`: Arbitrary config path overrides (e.g., `{"bot.long.total_wallet_exposure_limit": 2}`).
- **backtest.aggregate**: Dict of metric-specific aggregation modes (default `mean`). Keys fall back to the `default` entry if unspecified.

See [Suite Examples](suite_examples.md) for practical examples and suggested usage.

Example per-metric aggregation:

```json
"backtest": {
  "aggregate": {
    "default": "mean",
    "mdg_usd": "median",
    "sharpe_ratio": "std",
    "drawdown_worst_usd": "max"
  }
}
```

## Logging

- **level**: Controls global verbosity for Passivbot and tooling.
  - Accepted values: `0` (warnings), `1` (info), `2` (debug), `3` (trace).
  - The CLI flag `--debug-level`/`--log-level` on `passivbot live` and `passivbot backtest` overrides the configured value for a single run.
  - Components such as the CandlestickManager inherit this level, so EMA warm-up and candle maintenance logs follow the same verbosity.
- **persist_to_file**: When `true`, `passivbot live` also writes the console log stream to a timestamped file on disk and refreshes `logs/{user}.log` as a stable alias to the current run. The canonical default is `true`, so live runs write to `logs/` unless you disable it explicitly. In this first integrated version, backtest/optimize still use console logging unless you wrap them externally.
- **dir**: Directory used for persisted live log files and the stable current-run alias when `persist_to_file` is enabled. Default `logs`.
- **rotation**: Enables rotating live log files instead of appending to one file per process. Default `false`.
- **max_bytes_mb**: Maximum size in megabytes for each live log file before rotation. Used only when `rotation = true`. Default `10`.
- **backup_count**: Number of rotated backup files to keep when rotation is enabled. Default `5`.
- **memory_snapshot_interval_minutes**: Interval between `_log_memory_snapshot` telemetry entries (RSS, cache footprint, asyncio task counts). Default `30`; lower values surface leaks sooner, higher values reduce noise.
- **volume_refresh_info_threshold_seconds**: Minimum duration a bulk volume-EMA refresh must take before it is promoted to an INFO log. Runs that finish faster emit only DEBUG output (when debug logging is enabled). Set `0` to log every refresh at INFO.

## Monitor

The monitor publisher writes a read-only dashboard data root to disk when enabled.

- **enabled**: Master switch for monitor publication. Default `true`.
- **root_dir**: Base directory for monitor output. Per-bot data is written under `root_dir/{exchange}/{user}`.
- **snapshot_interval_seconds**: Best-effort minimum interval between `state.latest.json` writes.
- **checkpoint_interval_minutes**: Interval between compressed checkpoint snapshots. Set `0` to disable checkpoints.
- **event_rotation_mb**: Rotate `events/current.ndjson` after it exceeds this size.
- **event_rotation_minutes**: Rotate `events/current.ndjson` after this elapsed time even if size threshold is not reached.
- **retain_days**: Age-based retention for rotated event/history/checkpoint files.
- **max_total_bytes**: Global byte cap for the monitor root. Old rotated event/history/checkpoint files are pruned first.
- **retain_price_ticks**, **retain_candles**, **retain_fills**: Enable or disable the current history streams for price ticks, completed candles, and normalized fills.
- **compress_rotated_segments**: If `true`, gzip rotated event segments and checkpoints.
- **price_tick_min_interval_ms**: Minimum per-symbol interval for emitting `history/price_ticks.current.ndjson` entries.
- **emit_completed_candles**: Enable or disable completed 1m/1h candle history publication.
- **include_raw_fill_payloads**: If `true`, include exchange/raw fill payloads alongside the normalized fill history payload.

See [monitor.md](monitor.md) for current output files and event kinds.

## Bot Settings

### Side-Specific HSL Parameters

HSL now lives directly under each `pside`:

1. `bot.long.hsl_*`
2. `bot.short.hsl_*`
3. `live.hsl_signal_mode`

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Risk Management](risk_management.md)

### Equity Hard Stop Loss (`bot.{long,short}.hsl_*`)

Side-specific drawdown circuit breaker.

Each `pside` has the same parameter set:

- **hsl_enabled**:
  - Enables or disables HSL on that `pside`.
- **hsl_red_threshold**:
  - RED trigger threshold for the HSL drawdown score.
- **hsl_ema_span_minutes**:
  - EMA span used for smoothed drawdown.
  - In backtests, if this is smaller than `backtest.candle_interval_minutes`, smoothing is effectively disabled and HSL uses raw drawdown for the EMA leg.
- **hsl_cooldown_minutes_after_red**:
  - Minutes to wait before auto-restart after a RED halt on that `pside`.
  - `0.0` means halt without auto-restart.
  - Restart-time HSL replay treats a historical panic-flatten on that `pside` as a completed RED stop and resets tracking from after that panic before evaluating later cooldown/restart behavior.
- **hsl_no_restart_drawdown_threshold**:
  - Terminal no-restart threshold for that `pside`.
  - Evaluated from persistent cross-restart HSL drawdown.
  - Values below `hsl_red_threshold` are clamped up to `hsl_red_threshold`.
  - Must satisfy: `hsl_red_threshold <= hsl_no_restart_drawdown_threshold <= 1.0`.
- **hsl_tier_ratios.yellow / hsl_tier_ratios.orange**:
  - Multipliers used to derive YELLOW and ORANGE thresholds from `hsl_red_threshold`.
  - Must satisfy: `0 < yellow < orange < 1`.
- **hsl_orange_tier_mode**:
  - Allowed values:
    - `graceful_stop`
    - `tp_only_with_active_entry_cancellation`
  - Determines how the bot behaves in ORANGE on that `pside`.
- **hsl_panic_close_order_type**:
  - Allowed values:
    - `market`
    - `limit`
  - Determines how RED panic exits are executed or simulated for that `pside`.

Behavior summary:

1. YELLOW: warning tier for that `pside`
2. ORANGE: reduced-risk mode for that `pside`
3. RED: panic close, flat confirmation, halt, optional cooldown restart for that `pside`

Signal mode:

1. `live.hsl_signal_mode = "pside"`
   - each `pside` controller uses its own realized/unrealized strategy PnL
2. `live.hsl_signal_mode = "unified"`
   - long and short keep separate HSL controllers
   - both are fed from the same unified account-level strategy signal

Backtest-specific note:

1. If `hsl_panic_close_order_type = "market"`, the backtester uses `backtest.market_order_slippage_pct` for simulated taker execution and charges taker fees (exchange-derived by default, or `backtest.taker_fee_override` when set).

Key HSL analysis metrics:

1. Global account metrics:
   - `drawdown_worst_strategy_eq`
   - `drawdown_worst_mean_1pct_strategy_eq`
   - `peak_recovery_hours_strategy_eq`
   - `hard_stop_triggers`
   - `hard_stop_restarts`
2. Side-specific metrics:
   - `drawdown_worst_strategy_eq_long`
   - `drawdown_worst_strategy_eq_short`
   - `drawdown_worst_mean_1pct_strategy_eq_long`
   - `drawdown_worst_mean_1pct_strategy_eq_short`
   - `peak_recovery_hours_strategy_eq_long`
   - `peak_recovery_hours_strategy_eq_short`
   - `hard_stop_triggers_long`
   - `hard_stop_triggers_short`
   - `hard_stop_restarts_long`
   - `hard_stop_restarts_short`

### General Parameters for Long and Short

- **ema_span_0**, **ema_span_1**:
  - Spans are given in minutes.
  - Formula: `next_EMA = prev_EMA * (1 - alpha) + new_val * alpha`, where `alpha = 2 / (span + 1)`.
  - An additional EMA span is calculated as `(ema_span_0 * ema_span_1)**0.5`.
  - The three EMAs form an upper and lower EMA band:
    - `ema_band_lower = min(emas)`
    - `ema_band_upper = max(emas)`
  - These bands are used for initial entries and auto unstuck closes.
- **n_positions**: Maximum number of positions to open. Set to `0` to disable long/short.
- **total_wallet_exposure_limit**: Maximum exposure allowed.
  - Example: `total_wallet_exposure_limit = 0.75` means 75% of (unleveraged) wallet balance is used.
  - Example: `total_wallet_exposure_limit = 1.6` means 160% of (unleveraged) wallet balance is used.
  - Live denominator is fixed: `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
  - Backtest denominator is controlled by `backtest.dynamic_wel_by_tradability`.
  - See more: `docs/risk_management.md`.

### Grid Entry Parameters

Passivbot can be configured to create a grid of entry orders, with prices and quantities determined by the following parameters:

- **entry_grid_double_down_factor**:
  - Quantity of the next grid entry is position size times the double down factor.
  - Example: If position size is `1.4` and `double_down_factor` is `0.9`, then the next entry quantity is `1.4 * 0.9 = 1.26`.
  - Also applies to trailing entries.
- **entry_grid_inflation_enabled**:
  - When `true`, grid-mode re-entries may inflate the current order near the effective wallet exposure cap if the next grid step would otherwise become tiny.
  - When `false`, grid re-entries are only normal or cropped so the bot observes effective WEL without pulling future size forward.
  - `false` is the canonical setting going forward. The current default remains `true` for backwards compatibility, and config parsing warns that the inflated path is scheduled for deprecation.
- **entry_grid_spacing_pct**, **entry_grid_spacing_we_weight**:
  - Grid re-entry prices are determined as follows:
    - `next_reentry_price_long = pos_price * (1 - entry_grid_spacing_pct * multiplier)`
    - `next_reentry_price_short = pos_price * (1 + entry_grid_spacing_pct * multiplier)`
  - `multiplier = 1 + (wallet_exposure / wallet_exposure_limit) * entry_grid_spacing_we_weight + log_component`
  - Setting `entry_grid_spacing_we_weight` > 0 widens spacing as the position approaches the wallet exposure limit; negative values tighten spacing when exposure is small.
- **entry_grid_spacing_volatility_weight**, **entry_volatility_ema_span_hours**:
  - The `log_component` in the multiplier above is derived from the EMA of the per-candle log range `ln(high/low)`.
  - `entry_grid_spacing_volatility_weight` controls how strongly the recent log range widens or narrows spacing. A value of `0` disables the log-based adjustment.
  - `entry_volatility_ema_span_hours` sets the EMA span (in hours) used when smoothing the volatility (log-range) signal before applying the weight. The same volatility EMA also powers the multipliers for `entry_trailing_threshold_volatility_weight` and `entry_trailing_retracement_volatility_weight`.
- **entry_initial_ema_dist**:
  - Offset from lower/upper EMA band.
  - Long initial entry/short unstuck close prices are lower EMA band minus offset.
  - Short initial entry/long unstuck close prices are upper EMA band plus offset.
  - See `ema_span_0`/`ema_span_1`.
- **entry_initial_qty_pct**:
  - `initial_entry_cost = balance * wallet_exposure_limit * entry_initial_qty_pct`
- **entry_trailing_double_down_factor**:
  - Multiplier controlling how aggressively trailing re-entries ramp up. As with the grid equivalent, any positive value increases the size of successive fills (higher values grow them faster).
- **entry_trailing_threshold_pct**, **entry_trailing_retracement_pct**:
  - Same semantics as the trailing-close parameters below, but applied to trailing entries. The bot waits for a favorable move (`threshold_pct`) and subsequent pullback (`retracement_pct`) before firing a trailing re-entry.
- **entry_trailing_threshold_we_weight**, **entry_trailing_retracement_we_weight**:
  - Extra scaling based on wallet exposure. As exposure approaches the per-symbol limit, positive weights widen the trailing bands to slow additional entries. Set to `0.0` to disable the adjustment.
- **entry_trailing_threshold_volatility_weight**, **entry_trailing_retracement_volatility_weight**:
  - Adds sensitivity to recent volatility using the shared `entry_volatility_ema_span_hours` EMA. Positive weights increase the thresholds in choppy markets; `0.0` removes the volatility modulation.

### Trailing Parameters

The same logic applies to both trailing entries and trailing closes.

- **trailing_grid_ratio**:
  - Sets trailing and grid allocations.
  - If `trailing_grid_ratio = 0.0`, grid orders only.
  - If `trailing_grid_ratio = 1.0` or `trailing_grid_ratio = -1.0`, trailing orders only.
  - If `trailing_grid_ratio > 0.0`, trailing orders first, then grid orders.
  - If `trailing_grid_ratio < 0.0`, grid orders first, then trailing orders.
    - Example: `trailing_grid_ratio = 0.3`: Trailing orders until position is 30% full, then grid orders for the rest.
    - Example: `trailing_grid_ratio = -0.9`: Grid orders until position is `(1 - 0.9) = 10%` full, then trailing orders for the rest.
    - Example: `trailing_grid_ratio = -0.12`: Grid orders until position is `(1 - 0.12) = 88%` full, then trailing orders for the rest.
- **trailing_retracement_pct**, **trailing_threshold_pct**:
  - Two conditions trigger a trailing order: 1) threshold and 2) retracement.
  - If `trailing_threshold_pct <= 0.0`, the threshold condition is always triggered.
  - For long positions:
    - `if highest price since position change > position price * (1 + trailing_threshold_pct)`, the first condition is met.
    - `if lowest price since highest price < highest price since position change * (1 - trailing_retracement_pct)`, the second condition is met. Place order.
  - Passivbot tracks its own trailing prices and does not use special trailing order types from the exchange.
  - Trailing price tracker resets when the position changes (add to or partially close).
  - Trailing price tracking is based on 1m OHLCVs, updated on each new whole minute.

### Grid Close Parameters

- **close_grid_markup_start**, **close_grid_markup_end**, **close_grid_qty_pct**:
  - Take Profit (TP) prices are linearly spaced between:
    - `pos_price * (1 + markup_start)` to `pos_price * (1 + markup_end)` for **long**.
    - `pos_price * (1 - markup_start)` to `pos_price * (1 - markup_end)` for **short**.
  - The TP direction depends on the relative values of `markup_start` and `markup_end`:
    - If `markup_start > markup_end`: TP grid is built **backwards** (starting at higher price and descending for long / ascending for short).
    - If `markup_start < markup_end`: TP grid is built **forwards** (starting at lower price and ascending for long / descending for short).
  - Example (**long**, backwards TP): If `pos_price = 100`, `markup_start = 0.01`, `markup_end = 0.005`, and `close_grid_qty_pct = 0.2`, TP prices are: `[101.0, 100.9, 100.8, 100.7, 100.6]`.
  - Example (**long**, forwards TP): If `markup_start = 0.005`, `markup_end = 0.01`, TP prices are: `[100.5, 100.6, 100.7, 100.8, 100.9]`.
  - Example (**short**, forwards TP): If `pos_price = 100`, `markup_start = 0.005`, `markup_end = 0.01`, TP prices are: `[99.5, 99.4, 99.3, 99.2, 99.1]`.
  - Example (**short**, backwards TP): If `markup_start = 0.01`, `markup_end = 0.005`, TP prices are: `[99.0, 99.1, 99.2, 99.3, 99.4]`.
  - Quantity per order is `full pos size * close_grid_qty_pct`.
  - Note: Full position size refers to the maxed-out size. If the actual position is smaller, fewer than `1 / close_grid_qty_pct` orders may be created.
  - The TP grid is filled in order from `markup_start` to `markup_end`, allocating each slice up to the respective quantity:
    - First TP up to `close_grid_qty_pct * full_pos_size`.
    - Second TP from `close_grid_qty_pct` to `2 * close_grid_qty_pct`, etc.
  - Example: If `full_pos_size = 100` and `long_pos_size = 55`, and prices are built backwards, then TP orders might be `[15@100.8, 20@100.9, 20@101.0]`.
  - If position exceeds full position size, excess size is added to the TP order closest to `markup_start`.
    - Example: If `long_pos_size = 130` and grid is forwards, TP orders are `[50@100.5, 20@100.6, 20@100.7, 20@100.8, 20@100.9]`.

### Trailing Close Parameters

- **close_trailing_grid_ratio**: See Trailing Parameters above.
- **close_trailing_qty_pct**: Close quantity is `full pos size * close_trailing_qty_pct`.
- **close_trailing_retracement_pct**: See Trailing Parameters above.
- **close_trailing_threshold_pct**: See Trailing Parameters above.

### Unstuck Parameters

If a position is stuck, the bot uses profits from other positions to realize losses for the stuck position. If multiple positions are stuck, the position with the lowest price action distance is selected for unstucking.

- **unstuck_close_pct**:
  - Percentage of `full pos size * wallet_exposure_limit` to close for each unstucking order.
- **unstuck_ema_dist**:
  - Distance from EMA band to place unstucking order:
    - `long_unstuck_close_price = upper_EMA_band * (1 + unstuck_ema_dist)`
    - `short_unstuck_close_price = lower_EMA_band * (1 - unstuck_ema_dist)`
- **unstuck_loss_allowance_pct**:
  - Weighted percentage below past peak balance to allow losses.
  - `loss_allowance = past_peak_balance * (1 - unstuck_loss_allowance_pct * total_wallet_exposure_limit)`
  - Example: If past peak balance was `$10,000`, `unstuck_loss_allowance_pct = 0.02`, and `total_wallet_exposure_limit = 1.5`, the bot stops taking losses when balance reaches `$10,000 * (1 - 0.02 * 1.5) = $9,700`.
- **unstuck_threshold**:
  - If a position is larger than the threshold, consider it stuck and activate unstucking.
  - `if wallet_exposure / wallet_exposure_limit > unstuck_threshold: unstucking enabled`
  - Example: If a position size is `$500` and max allowed position size is `$1000`, the position is 50% full. If `unstuck_threshold = 0.45`, unstuck the position until its size is `$450`.

### Filter Parameters

Forager coin selection now uses a two-stage model: coarse volume pruning, then weighted ranking across volume, EMA readiness, and volatility.

- **forager_volume_drop_pct**: Coarse low-volume prune. Drops the lowest relative-volume fraction before final ranking, while still retaining enough candidates to fill the configured slots.
  - Example: `forager_volume_drop_pct = 0.1` drops the bottom 10% by relative volume. Set to `0` to skip the prune stage.
- **forager_volatility_ema_span / forager_volume_ema_span**: Number of minutes to look into the past to compute the 1m volatility (log-range) and quote-volume EMAs used by forager mode.
  - Log range is computed from 1m OHLCVs as `mean(ln(high / low))`.
  - These spans control the raw inputs to forager ranking; they are separate from `entry_volatility_ema_span_hours`, which is used for entry logic.
- **forager_score_weights**: Final weighted forager ranking weights.
  - Required keys: `volume`, `ema_readiness`, `volatility`.
  - Default: `{"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}`.
  - Positive weights are relative and normalized to unit sum before use.
  - If all three are `0.0`, Passivbot normalizes them to EMA-readiness-only ranking.
  - `ema_readiness` ranks by distance to the actual offset initial-entry threshold, not raw EMA bands.

See [docs/forager.md](forager.md) for a full description of motivation, ranking rules, caveats, and usage examples.

## Coin Overrides
- **coin_overrides**:
  - Specify full or partial configs for individual coins, overriding values from master config.
  - Format: {"COIN1": overrides1, "COIN2": overrides2}
  - Whole configs may be loaded with parameter "override_config_path". May either be full path to config, or filename for alternate config file from the same directory as master config file.
  - Specific override parameters take precedence over override parameters loaded from external config.
  - Only a subset of config parameters are eligible for overriding master config:
    - config.bot.long/short:
      ```
      [
        close_grid_markup_end, close_grid_markup_start, close_grid_qty_pct, close_trailing_grid_ratio, close_trailing_qty_pct,
    close_trailing_retracement_pct, close_trailing_threshold_pct, ema_span_0, ema_span_1,
        entry_grid_double_down_factor, entry_grid_spacing_pct, entry_grid_spacing_we_weight,
        entry_grid_spacing_volatility_weight, entry_volatility_ema_span_hours, entry_initial_ema_dist,
        entry_initial_qty_pct, entry_trailing_double_down_factor, entry_trailing_grid_ratio, entry_trailing_retracement_pct,
        entry_trailing_threshold_pct, unstuck_close_pct, unstuck_ema_dist, unstuck_threshold, wallet_exposure_limit
      ]
      ```
    -config.live:
    ```
    [forced_mode_long, forced_mode_short, leverage]
    ```
  - Examples:
    - `{"COIN1": {"override_config_path": "path/to/override_config.json"}}` -- Will attempt to load "path/to/override_config.json" and apply all eligible parameters from there for COIN1
    - `{"COIN2": {"override_config_path": "path/to/other_override_config.json", {"bot": {"long": {"close_grid_markup_start": 0.005}}}}}` -- Will attempt to load `"path/to/other_override_config.json"` first, and apply `{"bot": {"long": {"close_grid_markup_start": 0.005}}}` after.
    - `{"COIN3": {"bot": {"short": {"entry_initial_qty_pct": 0.01}}, "live": {"forced_mode_long": "panic"}}}` -- Will apply given overrides for COIN3.
- **forced_modes**:
  - Choices: `[n (normal), m (manual), gs (graceful_stop), t (tp_only), p (panic)]`.
    - **Normal mode**: Passivbot manages the position as normal.
    - **Manual mode**: Passivbot ignores the position.
    - **Graceful stop**: If there is a position, Passivbot manages it; otherwise, no new positions are opened.
    - **Take Profit Only mode**: Passivbot only manages closing orders.
    - **Panic mode**: Passivbot closes the position immediately.

## Live Trading Settings

- **approved_coins**:
  - List of coins approved for trading.
    - Backtester and optimizer use `live.approved_coins` minus `live.ignored_coins`.
  - May be given as a path to an external file, read continuously by Passivbot.
  - May be split into long and short:
    - Example: `{"long": ["COIN1", "COIN2"], "short": ["COIN2", "COIN3"]}`
    - Example: `{"long": ["COIN1", "COIN2"], "short": "all"}`
  - Explicit empty values disable trading for the affected side:
    - `approved_coins = []`, `{}`, `""`, or `null` disables both sides
    - `approved_coins = {"long": ["BTC"], "short": []}` keeps long curated and disables short
  - The explicit value `"all"` means all eligible coins for the affected side:
    - `approved_coins = "all"` enables all eligible coins for both sides
    - `approved_coins = {"long": "all", "short": ["BTC", "ETH"]}` enables all eligible longs and curated shorts
  - Older configs using `live.empty_means_all_approved=true` still migrate for now:
    - A globally empty `approved_coins` input is converted to `approved_coins = "all"`
    - The parser logs that `live.empty_means_all_approved` is deprecated
- **auto_gs**: Automatically enable graceful stop for positions on disapproved coins.
  - Graceful stop: The bot continues trading as normal but does not open a new position after the current position is fully closed.
  - If `auto_gs=false`, positions on disapproved coins are put on manual mode.
- **enable_archive_candle_fetch**: Enables the archive-candle fallback path in live mode. Keep `false` unless you specifically want the live bot to supplement its local candle state from exchange archive endpoints.
- **execution_delay_seconds**: Wait `x` seconds after executing to exchange.
- **hedge_mode**: Requests simultaneous long and short positions on the same coin when the exchange supports it. Effective behavior is `config.live.hedge_mode AND exchange_capability`; on one-way-only venues the live bot will still run one-way even if this is `true`.
- **hsl_position_during_cooldown_policy**: Live-only policy for a position that appears on a halted `pside` during HSL RED cooldown.
  - `panic`: panic-close it again and restart the cooldown from that new flatten.
  - `normal`: treat it as an explicit operator override once a real non-flat position appears during cooldown; while flat the bot still blocks fresh initials on that `pside`, and only after the position appears does it clear the halt and restart HSL drawdown tracking from the current state.
  - `manual`: leave that position in `manual` mode while keeping the original cooldown running and blocking fresh initials.
  - `tp_only`: keep the original cooldown running, block new entries, and allow only close management on that `pside`.
  - `graceful_stop`: keep the original cooldown running and manage any existing position with `graceful_stop` semantics while still blocking fresh initials.
- **hsl_signal_mode**: Selects whether HSL drawdown is tracked per-side (`"pside"`) or from one combined account-level strategy signal (`"unified"`). See [Equity Hard Stop Loss](equity_hard_stop_loss.md).
- **max_memory_candles_per_symbol**: Maximum number of 1m candles retained in RAM per symbol. Older entries are trimmed once this cap is exceeded. Default is `200_000`.
- **max_disk_candles_per_symbol_per_tf**: Maximum number of candles persisted on disk per symbol and timeframe. Oldest shards are pruned once the limit is hit (default `2_000_000`).
- **candle_lock_timeout_seconds**: Seconds to wait when another process holds the CandlestickManager per-symbol candle fetch lock (default `10`). Increase when running many bots sharing the same cache directory to avoid spurious timeouts during slow API calls.
- **inactive_coin_candle_ttl_minutes**: How long 1m candles for inactive symbols may stay in RAM before the live bot refreshes them. Lower values keep inactive symbols fresher at the cost of more network/disk churn.
- **filter_by_min_effective_cost**: If `true`, disallows coins where `balance * WE_limit * entry_initial_qty_pct < min_effective_cost`.
  - Example: If the exchange's effective minimum cost for a coin is `$5`, but the bot wants to make an order of `$2`, disallow that coin.
- **forced_mode_long**, **forced_mode_short**: Force all coins long/short to a given mode.
  - Choices: `[m (manual), gs (graceful_stop), p (panic), t (take_profit_only)]`.
- **ignored_coins**:
  - List of coins the bot will not make positions on. If there are positions on that coin, enable graceful stop or manual mode.
  - May be given as a path to an external file, read continuously by Passivbot.
  - May be split into long and short:
    - Example: `{"long": ["COIN1", "COIN2"], "short": ["COIN2", "COIN3"]}`
- **leverage**: Leverage set on the exchange. Default is `10`.
- **margin_mode_preference**: Preferred live margin mode when a symbol supports both cross and isolated.
  - `auto` / `auto_cross`: prefer cross when both modes are available.
  - `auto_isolated`: prefer isolated when both modes are available.
  - `cross`: require cross for new entries; isolated-only symbols are skipped for new entries but existing positions/orders remain manageable.
  - `isolated`: require isolated for new entries; cross-only symbols are skipped for new entries but existing positions/orders remain manageable.
  - If the exchange reports an already-open live position or open orders on a symbol, the live bot preserves that symbol's actual live margin mode for state management instead of forcing the configured preference mid-position.
  - Hyperliquid HIP-3 exception: isolated HIP-3 live trading is currently unsupported. Cross-capable HIP-3 markets are forced to cross for new entries, isolated-only HIP-3 markets are skipped for new entries, and existing isolated HIP-3 live state causes startup to fail loudly.
- **market_orders_allowed**: If `true`, allows Passivbot to place market orders when the order price is very close to the current market price. If `false`, only places limit orders. The current default profile uses `false`.
- **market_order_near_touch_threshold**: Unified threshold used by Rust order orchestration when `market_orders_allowed` is enabled. If an order price is within this fractional distance of the current market price, Rust emits it as a market order. Crossing orders also become market orders (`bid >= market` for buys, `ask <= market` for sells). This execution intent is now shared by both live and backtest. Default is `0.001`.
  - Decision rules:
    - non-panic buy with `price >= market_price` => `market`
    - non-panic sell with `price <= market_price` => `market`
    - otherwise, if `abs(order_price_diff) <= market_order_near_touch_threshold` => `market`
    - otherwise => `limit`
    - panic closes are still controlled separately by `bot.{long,short}.hsl_panic_close_order_type`
  - Ownership is `config.live`. Backtests always inherit `live.market_orders_allowed` and `live.market_order_near_touch_threshold`; `config.backtest` does not accept overrides for either field.
- **order_match_tolerance_pct**: Percentage tolerance (in %) used to match near-identical cancel/create pairs and avoid order churn. When a newly proposed order is within this tolerance of an existing open order, Passivbot may keep the existing order instead of cancelling/replacing it.
- **max_n_cancellations_per_batch**: Cancels `n` open orders per execution.
- **max_n_creations_per_batch**: Creates `n` new orders per execution.
- **max_n_restarts_per_day**: If the bot crashes, restart up to `n` times per day before stopping completely.
- **max_ohlcv_fetches_per_minute**: Live OHLCV/network budget for candle-backed indicators such as forager ranking and warm-up maintenance. Set lower to reduce REST pressure; set to `0` to disallow new fetches and rely only on what is already cached.
- **minimum_coin_age_days**: Disallows coins younger than a given number of days.
- **balance_override**: Optional numeric override for wallet balance used by the live bot (useful for dry-runs and debugging). When set, the bot will not fetch balance from the exchange. Can also be useful when using BTC collateral and you want to keep an effectively “fixed USD balance” for sizing, instead of having the USD-denominated balance fluctuate with the BTC/USD price.
- **balance_hysteresis_snap_pct**: Hysteresis snap percentage applied to balance updates to reduce noise. Set `0.0` to disable hysteresis.
- **recv_window_ms**: Millisecond tolerance for authenticated REST calls (default `5000`). Increase if your exchange intermittently rejects requests with `invalid request ... recv_window` errors due to clock drift.
- Candlestick management is handled by the CandlestickManager with on-disk caching and TTL-based refresh. Legacy settings `ohlcvs_1m_rolling_window_days` and `ohlcvs_1m_update_after_minutes` are no longer used.
- **pnls_max_lookback_days**: How far into the past to fetch PnL history. This also feeds the rolling realized-PnL window used by both live risk logic and backtests. Ownership is `config.live`; `config.backtest` does not accept an override.
  - `0`: minimal lookback window at the consumer's native sampling resolution (resets as often as that path can meaningfully observe).
  - `> 0`: rolling window of that many days.
  - `"all"`: full available history.
  - Live and backtest use the same contract for realized-PnL risk windows: filter realized fill events to the active lookback window, then recompute cumulative PnL, current value, and peak from only that filtered sequence.
- **price_distance_threshold**: Minimum distance to current price action required for EMA-based limit orders.
- **risk_wel_enforcer_threshold**: Per-symbol multiplier that triggers the WEL enforcer. When a position’s exposure exceeds `wallet_exposure_limit * (1 + risk_we_excess_allowance_pct) * risk_wel_enforcer_threshold` the bot emits a reduce-only order to bring it back under control. Set <1.0 for continual trimming, `1.0` for a hard cap, or ≤0 to disable.
- **risk_twel_enforcer_threshold**: Fraction of the configured `total_wallet_exposure_limit` that triggers the TWEL enforcer. When aggregate exposure exceeds this threshold the bot queues reduction orders instead of new entries. Set >1.0 to allow a grace margin, `1.0` for strict enforcement, or ≤0 to disable.
- **risk_we_excess_allowance_pct**: Per-symbol allowance above the configured wallet exposure limit that the enforcer tolerates before trimming. Useful for smoothing reductions; leave at `0.0` for a hard cap.
- **max_realized_loss_pct**: Global realized-loss gate for close orders, anchored to peak realized balance from fill history. For each close order, if projected realized PnL would push balance below `peak_balance * (1 - max_realized_loss_pct)`, the order is blocked. Applies to all close order types (including WEL/TWEL auto-reduce and unstuck) except panic closes.
  - Default: `1.0` (disabled).
  - `<= 0.0`: block all lossy closes.
  - `>= 1.0`: disable the gate.
  - Example: with peak balance `$10,000` and `max_realized_loss_pct = 0.05`, lossy closes are blocked once projected balance would fall below `$9,500`.
- **max_warmup_minutes**: Hard ceiling applied to the historical warm-up window for both backtests and live warm-ups. Use `0` to disable the cap; otherwise values above `0` clamp the per-symbol warmup calculated from EMA spans.
- **warmup_ratio**: Multiplier applied to the longest EMA or log-range span (in minutes) across long/short settings to decide how much 1m history to prefetch before trading. A value of `0.2`, for example, warmups ~20% of the deepest lookback, capped by `max_warmup_minutes`.
- **warmup_jitter_seconds**: Random startup delay spread applied before warm-up work begins. This helps multiple bots sharing one machine or cache avoid stampeding the same files and APIs at the same second.
- **warmup_concurrency**: Concurrency cap for live warm-up tasks. `0` lets Passivbot auto-select; positive values bound how many symbols are warmed in parallel.
- **max_concurrent_api_requests**: Optional global live REST concurrency cap. Leave `null` to use exchange/default behavior; set an integer to throttle authenticated and public request fan-out more aggressively.
- **warmup_minutes**: Not a config key. This is a derived per-coin warm-up window computed internally from `warmup_ratio`, indicator spans, and `max_warmup_minutes`.
- **time_in_force**: Default is Good-Till-Cancelled.
- **user**: Fetch API key/secret from `api-keys.json`.

## Optimization Settings

### Bounds

When optimizing, parameter values are constrained within the lower and upper bounds. Bounds support an optional third element specifying a discrete step size for grid-based optimization.

**Bounds Formats:**

- `[low, high]` - Continuous optimization between `low` and `high` (current behavior, unchanged)
- `[low, high, step]` - Discrete optimization with values constrained to the grid: `low`, `low + step`, `low + 2*step`, ..., `high`
- `[low, high, 0]` or `[low, high, null]` - Treated as continuous (equivalent to `[low, high]`)
- Single value (e.g., `0.5`) - Fixed parameter (not optimized)

**Step Behavior:**

When a step is defined, the optimizer only explores values on the discrete grid. The genetic algorithm performs crossover and mutation in *index space* (i.e., the indices of valid grid values) to ensure offspring values always land on the grid.

For example, with bounds `[0.01, 0.10, 0.02]`:
- Valid values are: 0.01, 0.03, 0.05, 0.07, 0.09
- The optimizer will never produce values like 0.02 or 0.04

**When to Use Stepped Bounds:**

- **Integer parameters**: Use step `1` for parameters that should be integers (e.g., `n_positions`)
- **Coarse search**: Use larger steps to reduce search space and speed up optimization
- **Known granularity**: When you know the parameter only makes sense at certain intervals

**Example Configuration:**

```json
"optimize": {
    "bounds": {
        "long_n_positions": [1, 20, 1],
        "long_total_wallet_exposure_limit": [0.1, 2.0, 0.1],
        "long_entry_grid_spacing_pct": [0.005, 0.05, 0.005],
        "long_ema_span_0": [100, 10000],
        "long_ema_span_1": [200, 20000]
    }
}
```

In this example:
- `n_positions`: Integers from 1 to 20
- `total_wallet_exposure_limit`: Values 0.1, 0.2, 0.3, ..., 2.0
- `entry_grid_spacing_pct`: Values 0.005, 0.01, 0.015, ..., 0.05
- `ema_span_0` and `ema_span_1`: Continuous optimization (no step defined)

HSL bounds now use side-specific prefixes:

1. `long_hsl_red_threshold`
2. `long_hsl_ema_span_minutes`
3. `long_hsl_cooldown_minutes_after_red`
4. `short_hsl_red_threshold`
5. `short_hsl_ema_span_minutes`
6. `short_hsl_cooldown_minutes_after_red`

`long_hsl_no_restart_drawdown_threshold` and `short_hsl_no_restart_drawdown_threshold` are intentionally not part of the default optimize bounds. The runtime parameters still live under `bot.{long,short}.hsl_*`, but optimizer runs disable terminal no-restart by default via:

1. `optimize.fixed_runtime_overrides["bot.long.hsl_no_restart_drawdown_threshold"] = 1.0`
2. `optimize.fixed_runtime_overrides["bot.short.hsl_no_restart_drawdown_threshold"] = 1.0`

Risk should be constrained through canonical `*_strategy_eq` metrics instead. Deprecated `*_hsl` metric names remain accepted as aliases for older configs/results.

**Validation:**

- Step must be positive; negative or zero steps are treated as continuous
- Step must not exceed the range (`high - low`); if it does, a warning is logged and the parameter is treated as continuous

### Other Optimization Parameters

- **compress_results_file**: If `true`, compresses optimize output results file to save space.
- **enable_overrides**: List of constraint overrides applied during optimization to enforce specific parameter relationships. The optimizer evaluator checks these conditions and apply the overrides before running each backtest (defaults to none):
  - **"lossless_close_trailing"**: Ensures trailing stops are profitable by enforcing `close_trailing_threshold_pct` > `close_trailing_retracement_pct`. This prevents the retracement from triggering before reaching the minimum profit threshold.
  - **"forward_tp_grid"**: Creates an ascending take-profit grid where `close_grid_markup_start` < `close_grid_markup_end`
  - **"backward_tp_grid"**: Creates a descending take-profit grid where `close_grid_markup_start` > `close_grid_markup_end`.
- **crossover_probability**: Probability of performing crossover between two individuals in the genetic algorithm. Determines how often parents exchange genetic information to create offspring.
- **crossover_eta**: Crowding factor (η) for simulated-binary crossover. Lower values (<20) allow offspring to move farther away from their parents; higher values keep them closer. Default is `20.0`.
- **fixed_params**: List of `optimize.bounds` keys to freeze at the current config value for the whole run. This is the config-file equivalent of fine-tuning only a subset of parameters.
- **fixed_runtime_overrides**: Runtime-only overrides applied during optimize evaluations without mutating the stored config. Use this for optimizer-specific safety knobs such as disabling terminal HSL no-restart while still keeping the live/backtest config unchanged on disk.
- **iters**: Number of backtests per optimize session.
- **mutation_probability**: Probability of mutating an individual in the genetic algorithm. Determines how often random changes are introduced to maintain diversity.
- **mutation_eta**: Crowding factor (η) for polynomial mutation. Smaller values (<20) produce heavier-tailed steps that explore more aggressively, while larger values confine mutations near the current value. Default is `20.0`.
- **mutation_indpb**: Probability that each attribute mutates when a mutation is triggered. Set to `0` (default) to auto-scale to `1 / number_of_parameters`, or supply an explicit probability between `0` and `1`.
- **n_cpus**: Number of CPU cores utilized in parallel.
- **offspring_multiplier**: Multiplier applied to `population_size` to determine how many offspring (`λ`) are produced each generation in the μ+λ evolution strategy. Values >1.0 increase exploration by sampling more children per generation. Default is `1.0`.
- **pareto_max_size**: Maximum number of Pareto-optimal configs kept on disk under `optimize_results/.../pareto/`. Members are pruned by crowding (least diverse removed first, while per-objective extremes are preserved), not by age. Default is `1000`.
- **population_size**: Size of population for genetic optimization algorithm.
- **backend**: Optimizer backend. Default is `pymoo`. With the default `optimize.pymoo.algorithm: "auto"`, Passivbot uses `nsga2` for `3` or fewer objectives and `nsga3` for `4+`.
- **round_to_n_significant_digits**: Quantization precision used when hashing configs, deduplicating candidates, and writing optimizer artifacts. Lower values collapse near-identical candidates more aggressively; higher values preserve more distinct variants.
- **scoring**:
  - The optimizer minimizes the configured objective list and keeps the Pareto front.
  - The current default profile uses:
    - `adg_strategy_eq`
    - `adg_strategy_eq_w`
    - `mdg_strategy_eq`
    - `mdg_strategy_eq_w`
    - `peak_recovery_hours_strategy_eq`
    - `position_held_hours_max`
    - `drawdown_worst_strategy_eq`
    - `drawdown_worst_mean_1pct_strategy_eq`
  - With the default `pymoo` backend, Passivbot uses `nsga2` for `3` or fewer objectives and `nsga3` for `4+` objectives unless explicitly overridden.
  - Full list of options: `[adg, adg_w, calmar_ratio, calmar_ratio_w, drawdown_worst, drawdown_worst_mean_1pct, equity_balance_diff_neg_max, equity_balance_diff_neg_mean, equity_balance_diff_pos_max, equity_balance_diff_pos_mean, expected_shortfall_1pct, gain, hard_stop_duration_minutes_max, hard_stop_duration_minutes_mean, hard_stop_flatten_time_minutes_mean, hard_stop_halt_to_restart_equity_loss_pct, hard_stop_panic_close_loss_max, hard_stop_panic_close_loss_sum, hard_stop_post_restart_retrigger_pct, hard_stop_time_in_orange_pct, hard_stop_time_in_red_pct, hard_stop_time_in_yellow_pct, hard_stop_trigger_drawdown_mean, loss_profit_ratio, loss_profit_ratio_w, mdg, mdg_w, omega_ratio, omega_ratio_w, peak_recovery_hours_equity, peak_recovery_hours_pnl, position_held_hours_max, position_held_hours_mean, position_held_hours_median, position_unchanged_hours_max, positions_held_per_day, sharpe_ratio, sharpe_ratio_w, sortino_ratio, sortino_ratio_w, sterling_ratio, sterling_ratio_w]`
  - Suffix `_w` indicates mean across 10 temporal subsets (whole, last_half, last_third, ..., last_tenth) to weigh recent data more heavily.
  - Examples: `["mdg", "sharpe_ratio", "loss_profit_ratio"]`, `["adg", "sortino_ratio", "drawdown_worst"]`, `["sortino_ratio", "omega_ratio", "adg_w", "position_unchanged_hours_max"]`, `["adg_pnl_w", "hard_stop_time_in_red_pct", "hard_stop_panic_close_loss_sum"]`
    - Note: metrics may be suffixed with `_usd` or `_btc` to select denomination. If `config.backtest.btc_collateral_cap` is `0`, BTC values still represent the USD equity translated into BTC terms.
- **write_all_results**: Controls whether every evaluated candidate is appended to `all_results.bin`. Keep `true` for full replay/analysis history; set `false` to reduce disk writes and store only the maintained Pareto/state artifacts.

### Optimizer Suites

The optimizer reuses the backtest suite configuration when `--suite [y/n]` is enabled.

- **backtest.suite_enabled**: Can be toggled for optimizer runs via `--suite [y/n]` on `passivbot optimize`.
- **backtest.aggregate**: Per-metric aggregation rules applied to scenario results before feeding into `optimize.scoring` and `optimize.limits`.
- **backtest.scenarios**: Scenario dictionaries. Each one may override `coins`, `ignored_coins`, `start_date`, `end_date`, `exchanges`, `coin_sources`, and `overrides` (arbitrary config path overrides).

Use `--suite-config path/to/file.json` to layer additional scenario definitions at runtime.

### Optimization Limits

The optimizer penalizes backtests whose metric values exceed or fall short of specified thresholds. Penalties are added to the fitness score to discourage undesirable configurations but do not disqualify the config.

Any metric listed above (and its `btc_` prefixed counterpart when `backtest.use_btc_collateral=True`) can be used when defining limits. This includes the shared HSL metrics such as `hard_stop_time_in_red_pct`, `hard_stop_post_restart_retrigger_pct`, and `hard_stop_halt_to_restart_equity_loss_pct`, plus `backtest_completion_ratio` for rejecting truncated runs. HSL metrics are account-level shared metrics and therefore remain single-valued rather than being split into `_usd` and `_btc`. Each limit entry is a dictionary with:

- `metric`: canonical metric name (`drawdown_worst_btc`, `loss_profit_ratio`, `peak_recovery_hours_pnl`, etc.).
- `penalize_if`: one of `<`, `<=`, `>`, `>=`, `==`, `outside_range`, or `inside_range` (aliases like `less_than`, `greater_than`, `auto`, etc. are also accepted). Use `outside_range` to keep a metric within `[low, high]`, and `inside_range` to forbid a specific band.
- `value`: numeric threshold for `<`/`>` modes.
- `range`: two-value list `[low, high]` for the range modes.
- Optional `enabled`: set to `false` to disable a default limit without deleting it. This prevents config normalization from re-adding that metric's default limit later.
- Optional `stat`: when you want to compare against a specific statistic (`min`, `max`, `mean`, `std`). The default is `_max` for `>` checks, `_min` for `<` checks, and `_mean` for range checks.

#### Format

Define limits in `optimize.limits` as a list:

```json
"limits": [
  {"metric": "drawdown_worst_btc", "penalize_if": ">", "value": 0.3},
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg_btc", "penalize_if": "<", "value": 0.0005, "stat": "mean"},
  {"metric": "hard_stop_time_in_red_pct", "penalize_if": ">", "value": 0.02},
  {"metric": "backtest_completion_ratio", "penalize_if": "<", "value": 1.0}
]
```

To intentionally opt out of a default limit, keep the metric name but disable it:

```json
{"metric": "backtest_completion_ratio", "enabled": false}
```

For CLI overrides you can replace the full list with a JSON/HJSON payload:

```
passivbot optimize --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]'
```

For repeatable one-off entries, use `--limit`. Symbolic scalar operators in `--limit` are
written as keep conditions, matching `pareto_store.py` filtering:

```bash
passivbot optimize \
  --clear-limits \
  --limit 'drawdown_worst <= 0.35' \
  --limit 'backtest_completion_ratio>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'adg > 0.0008 stat=mean'
```

CLI replacement rules:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` appends one parsed limit entry and may be repeated.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

## Configuration Internals

Passivbot stores a few metadata keys alongside the normalized config:

- `_raw` retains the exact user input as it appeared on disk before formatting/normalization. It is
  meant for inspection and diffing—callers should treat it as read-only.
- `_coins_sources` records where approved/ignored coin lists originated (inline strings, external
  files, CLI overrides). Future overrides update both the normalized lists and their
  `_coins_sources` entries so live reloads honour the latest intent.
- `_transform_log` captures a chronological record of high-level configuration mutations (load,
  formatting, CLI overrides, etc.). Each entry stores a `step`, optional `details`, and a timestamp,
  making it easier to audit how the runtime config diverged from `_raw`.

Additional reserved keys may appear in future releases; all keys beginning with an underscore are
ignored by persistence helpers to keep user configs tidy.
