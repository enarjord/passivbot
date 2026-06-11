# Passivbot Parameters Explanation

This document explains the canonical config schema used by Passivbot.

- The source of truth for defaults is `src/config/schema.py`.
- The example config `configs/examples/default_trailing_martingale_long_npos4.json` mirrors those hardcoded defaults exactly.
- If you omit `config_path`, Passivbot loads those in-code defaults.

For the recommended user workflow, examples, and best practices, see [Config Workflow](config_workflow.md).

## Backtest Settings

- **base_dir**: Location to save backtest results.
- **compress_cache**: Set to `true` to save disk space. Set to `false` for faster loading.
- **end_date**: End date of backtest, e.g., `2024-06-23`. Set to `'now'` to use today's date as the end date.
- **exchanges**: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing. The most exercised choices are `binance` and `bybit`, which are also the current default profile. `bitget` and `gateio` are supported, with the GateIO history caveat below. `kucoin` has archive-fetch support but should be treated as experimental for backtesting until broader real-data smoke coverage is collected. Use short exchange names in configs; Passivbot converts to CCXT-specific IDs such as `binanceusdm` and `kucoinfutures` only when it talks to CCXT.
  **GateIO note:** If you already have `caches/ohlcv/gateio` data on disk, delete it before a fresh run so Passivbot rebuilds the cache with base-volume-normalized data.
  GateIO's public 1m OHLCV endpoint only serves a recent window of roughly 10,000 candles; use `backtest.ohlcv_source_dir` or another candle source for older GateIO backtests.
- **coin_sources**: Optional mapping of `coin -> exchange` used to override the automatic exchange selection when multiple exchanges are configured. Scenarios may add more overrides; conflicting assignments raise an error.
- **market_settings_sources**: Optional mapping of `coin -> exchange` used specifically for exchange metadata such as `price_step`, `qty_step`, fees, and min-size rules. This is separate from `coin_sources`: you may source candles from one exchange while borrowing market settings from another.
- **ohlcv_source_dir**: Optional path to a pre-populated legacy OHLCV directory to import before hitting exchange archives. Expected structure: `<dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npz` or `.npy`. Coin keys are normalized to base coins, but CCXT-style symbol folder names are accepted (e.g., `ETH_USDC:USDC`).
- **hlcvs_data_dir**: Optional path to a prepared final HLCV dataset under
  `caches/hlcvs_data/`. The dataset must have a valid manifest whose hashes
  verify `hlcvs`, `timestamps`, `btc_usd_prices`, `coins`, and
  `market_specific_settings`.
- **hlcvs_data_override_mode**: How `hlcvs_data_dir` is matched to the current
  config. `intersection` (default) keeps the config's requested coins/date
  window clipped to the verified dataset. `dataset` adopts the dataset's
  effective coins and timestamp window for exact artifact replay.
- **volume_normalization**: When `true` (default), normalize volume data across exchanges to make combined datasets comparable.
- **start_date**: Start date of backtest.
- **starting_balance**: Starting balance in USD at the beginning of the backtest.
- **filter_by_min_effective_cost**: When `true`, skip coins whose projected initial entry
  (balance × wallet_exposure_limit × the active strategy initial sizing fraction, including WE
  excess allowance)
  would fall below the exchange’s effective minimum cost.
- **dynamic_wel_by_tradability**: Backtest-only WEL denominator mode.  
  - `true` (default): `wallet_exposure_limit = total_wallet_exposure_limit / min(n_positions, n_tradable_max)` where `n_tradable_max` is the highest number of coins that have had real candles at any timestep so far (non-shrinking).  
  - `false`: fixed denominator, same as live: `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
- **candle_interval_minutes**: Aggregates raw 1m OHLCVs into coarser candles before the backtest loop runs. `1` keeps native 1m behavior; values above `1` speed up backtests and optimizer runs at the cost of losing intra-interval fill ordering.
- **gap_tolerance_ohlcvs_minutes**: Maximum internal hole size that can be filled in prepared OHLCV data. Larger or persistent gaps are repaired from local v2 data, legacy shards, and targeted remote fetches; if a large internal gap remains, it is excluded from the returned tradable window rather than made tradable with synthetic candles. Verified exchange-side late starts and early ends do not by themselves abort a run, but local corruption, malformed candles, missing BTC benchmark data, or no tradable candles still fail loudly.
- **liquidation_threshold**: Early-stop backtest equity-floor guard. The run terminates once total equity falls to or below `starting_balance * liquidation_threshold`, and `backtest_completion_ratio` will fall below `1.0`. Example: with `starting_balance = 1000` and `liquidation_threshold = 0.05`, the backtest stops at equity `<= 50`. This is not a “5% drawdown” threshold; if the run never rises above the start, it corresponds to roughly a `0.95` worst drawdown. Must satisfy `0.0 <= liquidation_threshold < 1.0`.
- **maker_fee_override**: Optional maker fee override (part-per-one; use `0.0002` for 0.02%). Leave `null` to use the exchange-derived maker fees.
- **taker_fee_override**: Optional taker fee override (part-per-one; use `0.00055` for 0.055%). Leave `null` to use the exchange-derived taker fees.
- **market_order_slippage_pct**: Backtest-only slippage applied whenever the backtester simulates market-order execution. This applies both to HSL panic closes when `bot.{long,short}.hsl.panic_close_order_type` is `"market"` and to normal orchestrator orders promoted to market execution by `live.market_orders_allowed`. A sell fills at `close * (1 - slippage_pct)` rounded down to `price_step`; a buy fills at `close * (1 + slippage_pct)` rounded up. The fill is guaranteed once the market-execution path is chosen, and the resulting fill also uses taker fees. Default `0.0005` (5 bps). This field is not a live slippage cap; live market orders use the exchange adapter's order semantics and any exchange/CCXT slippage controls.
- **visible_metrics**: Controls which metrics are printed to the terminal after a standalone backtest. `null` shows the metrics implied by `optimize.scoring` and `optimize.limits`, `[]` shows all metrics, and an explicit list adds extra named metrics to the default view. This affects CLI visibility only; the full metric set is still computed and persisted.
  Fill-activity metrics use the `fills_*` prefix, including fill counts, per-day entry/close and long/short rates, no-fill gap durations, per-position-slot activity, active fill day counts/ratio, analysis duration, active symbol count, and top-symbol fill share.
- **config_version**: Top-level schema version string for the config file. Canonical V8 configs must use `v8.0.0`. V8 is a breaking config schema and does not automatically convert v7 or pre-v8 configs; start from a V8 example config and port settings manually.
- **balance_sample_divider**: Minutes per bucket when sampling balances/equity for
  `balance_and_equity.csv.gz` and related plots. `1` keeps full per-minute resolution; higher values
  thin out the series (e.g., `15` stores one point every 15 minutes) to reduce file sizes. The CSV
  includes account balance/equity in USD and BTC plus collateral-agnostic `strategy_equity`.
- **btc_collateral_cap**: Target (and ceiling) share of account equity to hold in BTC collateral. `0` keeps the account fully in USD; `1.0` targets fully-BTC collateral; values `>1` allow leveraged BTC collateral, accepting negative USD balances. Backtests initialize the BTC collateral position at the first active trading step, not during EMA warmup.
- **btc_collateral_ltv_cap**: Optional loan-to-value ceiling (`USD debt ÷ equity`) enforced when topping up BTC. Leave `null` (default) to allow unlimited debt, or set to a float (e.g., `0.6`) to stop buying BTC once leverage exceeds that threshold.
### Suite Scenarios

Suite configuration uses a flattened structure directly under `backtest`:

- **backtest.suite_enabled**: Master switch for suite runs (`--suite [y/n]` overrides it at runtime). Default `false` in the canonical schema and example config.
- **backtest.scenarios**: List of scenario dicts. Supported per-scenario keys:
  - `label`: Directory name under `backtests/suite_runs/<timestamp>/`.
  - `start_date`, `end_date`: Override the global date window.
  - `coins`, `ignored_coins`: Restrict or skip symbols.
  - `exchanges`: Exchanges that can contribute data to this scenario. Scenario-only exchanges are added to the suite preparation set before the run starts.
  - `coin_sources`: Scenario-specific overrides for `coin_sources`.
  - `overrides`: Arbitrary config path overrides (e.g., `{"bot.long.risk.total_wallet_exposure_limit": 2}`).
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
  - The CLI flag `--log-level` on `passivbot live` and `passivbot backtest` overrides the configured value for a single run. It accepts `warning`, `info`, `debug`, `trace`, or `0-3`.
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

HSL now lives under the grouped side config for each `pside`:

1. `bot.long.hsl.*`
2. `bot.short.hsl.*`
3. `live.hsl_signal_mode`

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Risk Management](risk_management.md)

### Equity Hard Stop Loss (`bot.{long,short}.hsl.*`)

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
  - Restart-time HSL replay treats a historical RED panic close that fully closed all positions on that `pside` as a completed RED stop and resets tracking from after that panic before evaluating later cooldown/restart behavior.
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
3. RED: panic close, wait until all positions on that `pside` are fully closed, halt, optional cooldown restart for that `pside`

Signal mode:

1. `live.hsl_signal_mode = "unified"` (default)
   - long and short keep separate HSL controllers
   - both are fed from the same combined account-level strategy signal
2. `live.hsl_signal_mode = "pside"`
   - each `pside` controller uses its own realized/unrealized strategy PnL
3. `live.hsl_signal_mode = "coin"`
   - each `coin+pside` controller uses realized PnL drawdown inside `live.pnls_max_lookback_days` plus current UPnL
   - RED panic-closes only the affected `coin+pside`
   - live denominator is `balance * total_wallet_exposure_limit / config.n_positions`; runtime effective position count and WE-excess allowance are intentionally not included
   - backtests use configured `n_positions` when `backtest.dynamic_wel_by_tradability=false`, and the effective tradability-aware denominator when it is `true`

Backtest-specific note:

1. If `hsl_panic_close_order_type = "market"`, the backtester uses `backtest.market_order_slippage_pct` for simulated taker execution and charges taker fees (exchange-derived by default, or `backtest.taker_fee_override` when set).

Key HSL analysis metrics:

1. Global account metrics:
   - `drawdown_worst_strategy_eq`
   - `drawdown_worst_mean_1pct_strategy_eq`
   - `strategy_eq_recovery_days_max`
   - `strategy_eq_recovery_days_mean_worst_1pct`
   - `hard_stop_triggers`
   - `hard_stop_restarts`
2. Side-specific metrics:
   - `drawdown_worst_strategy_eq_long`
   - `drawdown_worst_strategy_eq_short`
   - `drawdown_worst_mean_1pct_strategy_eq_long`
   - `drawdown_worst_mean_1pct_strategy_eq_short`
   - `peak_recovery_days_strategy_eq_long`
   - `peak_recovery_days_strategy_eq_short`
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

### Trailing Martingale Entries

The canonical V8 strategy kind is `trailing_martingale`. It replaces the v7 `trailing_grid` schema with threshold/retracement fields. If retracement is disabled, threshold behaves like recursive DCA/grid spacing. If retracement is enabled, threshold is the required excursion and retracement is the confirmation move.

- **strategy.trailing_martingale.entry.double_down_factor**:
  - Quantity of the next re-entry is position size times the double down factor.
  - Example: If position size is `1.4` and `double_down_factor` is `0.9`, then the next entry quantity is `1.4 * 0.9 = 1.26`.
- **strategy.trailing_martingale.entry.threshold_base_pct**:
  - Re-entry threshold from the current position price.
  - If entry retracement is disabled:
    - `next_reentry_price_long = pos_price * (1 - effective_threshold)`
    - `next_reentry_price_short = pos_price * (1 + effective_threshold)`
  - If entry retracement is enabled, price must first move at least this far in the favorable direction before the retracement condition may place an order.
- **strategy.trailing_martingale.entry.threshold_we_weight**, **threshold_volatility_1h_weight**, **threshold_volatility_1m_weight**:
  - Entry threshold is widened multiplicatively:
    - `effective_threshold = threshold_base_pct * max(1, 1 + we_ratio * threshold_we_weight + volatility_1h * threshold_volatility_1h_weight + volatility_1m * threshold_volatility_1m_weight)`
  - `we_ratio = wallet_exposure / effective_wallet_exposure_limit`.
  - Positive weights make entries less eager as exposure or volatility rises.
- **strategy.trailing_martingale.entry.retracement_base_pct**:
  - If `<= 0.0`, trailing confirmation is disabled and entries are passive recursive limit orders.
  - If `> 0.0`, the bot waits for a favorable excursion past the threshold and then for a pullback by the effective retracement before placing the entry.
- **strategy.trailing_martingale.entry.retracement_we_weight**, **retracement_volatility_1h_weight**, **retracement_volatility_1m_weight**:
  - Entry retracement is widened multiplicatively with the same `max(1, 1 + ...)` style as entry threshold.
- **strategy.trailing_martingale.volatility_ema_span_1h**, **volatility_ema_span_1m**:
  - Volatility is the EMA of per-candle log range `ln(high / low)` on 1h and 1m candles.
  - A volatility weight of `0` disables that horizon for the affected threshold or retracement.
- **strategy.trailing_martingale.entry.initial_ema_dist**:
  - Offset from lower/upper EMA band.
  - Long initial entry/short unstuck close prices are lower EMA band minus offset.
  - Short initial entry/long unstuck close prices are upper EMA band plus offset.
  - See `ema_span_0`/`ema_span_1`.
- **strategy.trailing_martingale.entry.initial_qty_pct**:
  - `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`
  - This is the initial sizing fraction used by min-effective-cost checks when
    `live.strategy_kind = "trailing_martingale"`.

### Trailing Martingale Closes

Close threshold/retracement mirrors entries, but close threshold is additive so it can intentionally move through break-even or negative markup as wallet exposure rises.

- **strategy.trailing_martingale.close.qty_pct**:
  - Recursive close sizing fraction.
  - When close retracement is disabled and the close threshold does not depend on wallet exposure, recursive closes collapse to the same price. Rust intentionally emits one full-position close in that case because multiple same-price slices are redundant.
  - When close threshold depends on wallet exposure, `qty_pct` controls the recursive ladder: each synthetic close fill lowers `we_ratio`, then the next close is recomputed.
- **strategy.trailing_martingale.close.threshold_base_pct**:
  - Base close distance from position price.
  - With retracement disabled, this acts as the resting close markup:
    - `close_price_long = pos_price * (1 + effective_threshold)`
    - `close_price_short = pos_price * (1 - effective_threshold)`
  - With retracement enabled, this is the first trailing condition: market price must first reach the threshold before retracement can trigger the close.
- **strategy.trailing_martingale.close.threshold_we_weight**:
  - Close threshold is adjusted additively by exposure:
    - `effective_threshold = threshold_base_pct + we_ratio * threshold_we_weight + volatility_1h * threshold_volatility_1h_weight + volatility_1m * threshold_volatility_1m_weight`
  - Negative values make closes more eager as position exposure grows, including break-even or negative-markup closes.
  - Positive values make closes less eager as exposure grows.
- **strategy.trailing_martingale.close.threshold_volatility_1h_weight**, **threshold_volatility_1m_weight**:
  - Positive values shift close thresholds farther from position price during volatile periods.
- **strategy.trailing_martingale.close.retracement_base_pct**:
  - If `<= 0.0`, close trailing is disabled and closes are resting limit orders.
  - If `> 0.0`, the bot waits for price to reach the close threshold and then retrace by the effective retracement.
- **strategy.trailing_martingale.close.retracement_volatility_1h_weight**, **retracement_volatility_1m_weight**:
  - Close retracement is widened multiplicatively by volatility only. There is intentionally no close retracement wallet-exposure weight.

The V8 `trailing_martingale` schema is a clean break from the v7 `trailing_grid` schema. Removed concepts include `entry_trailing_grid_ratio`, `close_trailing_grid_ratio`, `close_grid_markup_start`, and `close_grid_markup_end`. The old `markup_start`/`markup_end` linear TP grid is not part of V8 behavior; recursive closes are now driven by `close.threshold_*`, `close.retracement_*`, and `close.qty_pct`.

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
- **forager_volatility_ema_span_1m / forager_volume_ema_span_1m**: Number of minutes to look into the past to compute the 1m volatility (log-range) and quote-volume EMAs used by forager mode.
  - Log range is computed from 1m OHLCVs as `mean(ln(high / low))`.
  - These spans control the raw inputs to forager ranking; they are separate from strategy volatility spans such as `volatility_ema_span_1h` and `offset_volatility_ema_span_1h`.
- **forager_score_weights**: Final weighted forager ranking weights.
  - Required keys: `volume`, `ema_readiness`, `volatility`.
  - Default: `{"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}`.
  - Positive weights are relative and normalized to unit sum before use.
  - If all three are `0.0`, Passivbot normalizes them to EMA-readiness-only ranking.
  - `ema_readiness` ranks by distance to the actual offset initial-entry threshold, not raw EMA bands.

See [docs/forager.md](forager.md) for a full description of motivation, ranking rules, caveats, and usage examples.

## Coin Overrides
- **coin_overrides**:
  - Specify full or partial configs for individual coins, overriding values from the master config.
  - Format: `{"COIN1": overrides1, "COIN2": overrides2}`.
  - Whole configs may be loaded with `override_config_path`. This may be a full path or a filename for an alternate config file in the same directory as the master config file.
  - Specific override parameters take precedence over override parameters loaded from an external config.
  - Only a subset of config parameters are eligible for overriding master config:
    - `config.bot.long/short.strategy.trailing_martingale`:
      ```
      [
        ema_span_0, ema_span_1, volatility_ema_span_1h, volatility_ema_span_1m,
        entry.double_down_factor, entry.initial_ema_dist, entry.initial_qty_pct,
        entry.threshold_base_pct, entry.threshold_we_weight,
        entry.threshold_volatility_1h_weight, entry.threshold_volatility_1m_weight,
        entry.retracement_base_pct, entry.retracement_we_weight,
        entry.retracement_volatility_1h_weight, entry.retracement_volatility_1m_weight,
        close.qty_pct, close.threshold_base_pct, close.threshold_we_weight,
        close.threshold_volatility_1h_weight, close.threshold_volatility_1m_weight,
        close.retracement_base_pct,
        close.retracement_volatility_1h_weight, close.retracement_volatility_1m_weight
      ]
      ```
    - `config.bot.long/short` shared groups:
      ```
      [
        risk.*, forager.*, hsl.*, unstuck.*, wallet_exposure_limit
      ]
      ```
    - `config.live`:
    ```
    [forced_mode_long, forced_mode_short, leverage]
    ```
  - Examples:
    - `{"COIN1": {"override_config_path": "path/to/override_config.json"}}` -- Will attempt to load "path/to/override_config.json" and apply all eligible parameters from there for COIN1
    - `{"COIN2": {"override_config_path": "path/to/other_override_config.json", "bot": {"long": {"strategy": {"trailing_martingale": {"close": {"threshold_base_pct": 0.005}}}}}}}` -- Will attempt to load `"path/to/other_override_config.json"` first, and apply the given close threshold override after.
    - `{"COIN3": {"bot": {"short": {"strategy": {"trailing_martingale": {"entry": {"initial_qty_pct": 0.01}}}}}, "live": {"forced_mode_long": "panic"}}}` -- Will apply given overrides for COIN3.
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
  - `panic`: panic-close it again and restart the cooldown after all positions on that `pside` are fully closed.
  - `normal`: treat it as an explicit operator override once a real open position appears during cooldown; while there are no open positions the bot still blocks fresh initials on that `pside`, and only after the position appears does it clear the halt and restart HSL drawdown tracking from the current state.
  - `manual`: leave that position in `manual` mode while keeping the original cooldown running and blocking fresh initials.
  - `tp_only`: keep the original cooldown running, block new entries, and allow only close management on that `pside`.
  - `graceful_stop`: keep the original cooldown running and manage any existing position with `graceful_stop` semantics while still blocking fresh initials.
- **hsl_signal_mode**: Selects whether HSL drawdown is tracked from one combined account-level strategy signal (`"unified"`, default), independently per side (`"pside"`), or per `coin+pside` slot (`"coin"`). See [Equity Hard Stop Loss](equity_hard_stop_loss.md).
- **max_memory_candles_per_symbol**: Maximum number of 1m candles retained in RAM per symbol. Older entries are trimmed once this cap is exceeded. Default is `200_000`.
- **max_disk_candles_per_symbol_per_tf**: Maximum number of candles persisted on disk per symbol and timeframe. Oldest shards are pruned once the limit is hit (default `2_000_000`).
- **candle_lock_timeout_seconds**: Seconds to wait when another process holds the CandlestickManager per-symbol candle fetch lock (default `10`). Increase when running many bots sharing the same cache directory to avoid spurious timeouts during slow API calls.
- **inactive_coin_candle_ttl_minutes**: How long 1m candles for inactive symbols may stay in RAM before the live bot refreshes them. Lower values keep inactive symbols fresher at the cost of more network/disk churn.
- **filter_by_min_effective_cost**: If `true`, disallows coins where
  `balance * allowed_wallet_exposure_limit * strategy_initial_sizing_fraction < min_effective_cost`.
  For `trailing_martingale`, the sizing fraction is
  `bot.<side>.strategy.trailing_martingale.entry.initial_qty_pct`; for `ema_anchor`, it is
  `bot.<side>.strategy.ema_anchor.base_qty_pct`.
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
- **market_orders_allowed**: If `true`, allows Passivbot to place market orders when the order price is very close to the current market price. If `false`, only places limit orders. The current default profile uses `false`. Live market orders execute as taker orders and may fill materially away from the reference price during fast moves or thin liquidity; exchange adapter and CCXT slippage controls apply.
- **market_order_near_touch_threshold**: Unified threshold used by Rust order orchestration when `market_orders_allowed` is enabled. If an order price is within this fractional distance of the current market price, Rust emits it as a market order. Crossing orders also become market orders (`bid >= market` for buys, `ask <= market` for sells). This execution intent is now shared by both live and backtest. Default is `0.001`. This threshold decides when an order may become market execution; it is not a live slippage cap.
  - Decision rules:
    - non-panic buy with `price >= market_price` => `market`
    - non-panic sell with `price <= market_price` => `market`
    - otherwise, if `abs(order_price_diff) <= market_order_near_touch_threshold` => `market`
    - otherwise => `limit`
    - panic closes are still controlled separately by `bot.{long,short}.hsl.panic_close_order_type`
  - Ownership is `config.live`. Backtests always inherit `live.market_orders_allowed` and `live.market_order_near_touch_threshold`; `config.backtest` does not accept overrides for either field.
- **market_snapshot_ticker_strategy**: Selects the live market-snapshot ticker path. `auto` lets Passivbot choose the exchange-appropriate default, `bulk` requests one broad ticker snapshot when the exchange supports it, and `symbols` fetches tickers symbol by symbol.
- **custom_endpoints_path**: Optional live-config path to a custom endpoint override JSON file. Set to `null` to use the default auto-discovery behavior, set to a path such as `configs/custom_endpoints.json` to force that file, or set to `"none"` to disable endpoint overrides even if the default file exists. See [Running live](live.md#custom-exchange-endpoints).
- **forager_score_hysteresis_pct**: Fractional normalized-score tolerance for forager incumbent selection. Default is `0.02`, meaning an already-selected flat forager coin is kept if a challenger beats it by no more than 2.0 percentage points of final forager score. Applies to live, backtest, and optimizer.
- **initial_entry_exec_max_market_dist_pct**: Live executor-side distance gate for `entry_initial_*` order creation. Default is `0.005`, meaning initial entries farther than 0.5% from current market price are logged but not posted until price comes closer. Set `0.0` to disable. Existing matching initial entries are kept by `order_match_tolerance_pct`; if they drift beyond tolerance while still outside this gate, they may be cancelled without immediate re-creation.
- **order_match_tolerance_pct**: Fractional relative tolerance used to match near-identical cancel/create pairs and avoid order churn. Default is `0.0002`, meaning 0.02% relative price/quantity tolerance. When a newly proposed order is within this tolerance of an existing open order, Passivbot may keep the existing order instead of cancelling/replacing it.
- **max_n_cancellations_per_batch**: Cancels `n` open orders per execution. Must be greater than `max_n_creations_per_batch` so the bot can make room before posting replacement orders.
- **max_n_creations_per_batch**: Creates `n` new orders per execution. Must be lower than `max_n_cancellations_per_batch`.
- **max_n_restarts_per_day**: If the bot crashes, restart up to `n` times per day before stopping completely.
- **max_active_candle_tail_gap_minutes**: Maximum open-ended 1m candle tail gap tolerated for active symbols before staged live planning blocks that symbol's trading-critical candle surface. Default is `10`. Within this bound, Passivbot projects provisional no-trade EMA inputs for close, quote-volume, and log-range without persisting synthetic candles or normal EMA cache entries. Real candles returned later always replace prior projections on the next read; bounded historical gaps still need real candles before and after before synthetic no-trade candles are replayed.
- **max_ohlcv_fetches_per_minute**: Live OHLCV/network budget for candle-backed indicators such as forager ranking and warm-up maintenance. Default is `24`. Set lower to reduce REST pressure; set to `0` to disallow new fetches and rely only on what is already cached.
- **max_forager_candle_staleness_minutes**: Optional cap on acceptable completed-candle staleness for broad forager-candidate refresh budgeting. `null` lets Passivbot derive the target from `max_ohlcv_fetches_per_minute` and candidate count.
- **max_forager_candle_refresh_seconds**: Wall-time cap for one best-effort broad forager-candidate candle refresh task. Default is `45`. When the cap is reached, Passivbot pauses the broad refresh and retries remaining stale candidates later; active position/open-order candle refreshes are not capped by this setting.
- **defer_broad_candle_warmup**: If `true`, startup warms only trading-critical symbols first and catches up broad approved-coin candles in the background. Set `false` to block startup until broad candle warmup completes.
- **minimum_coin_age_days**: Disallows coins younger than a given number of days.
- **balance_override**: Optional numeric override for wallet balance used by the live bot (useful for dry-runs and debugging). When set, the bot will not fetch balance from the exchange. Can also be useful when using BTC collateral and you want to keep an effectively “fixed USD balance” for sizing, instead of having the USD-denominated balance fluctuate with the BTC/USD price.
- **balance_hysteresis_snap_pct**: Hysteresis snap percentage applied to balance updates to reduce noise. Set `0.0` to disable hysteresis.
- **recv_window_ms**: Millisecond tolerance for authenticated REST calls (default `10000`). Increase if your exchange intermittently rejects requests with `invalid request ... recv_window` errors due to clock drift.
- Candlestick management is handled by the CandlestickManager with on-disk caching and TTL-based refresh. Legacy settings `ohlcvs_1m_rolling_window_days` and `ohlcvs_1m_update_after_minutes` are no longer used.
- **fills_recent_overlap_minutes**: Time overlap used for routine incremental live fill refreshes once the fill cache is warm. Default is `10.0`, which keeps ordinary minute-by-minute fill checks narrow to reduce private REST load.
- **fills_confirmation_overlap_minutes**: Time overlap used when fills are explicitly required to confirm an account-critical state change, such as a suspected fill or pending staged confirmation. Default is `60.0`, preserving the wider safety window for confirmation refreshes.
- **pnls_max_lookback_days**: How far into the past to fetch PnL history. This also feeds the rolling realized-PnL window used by both live risk logic and backtests. Ownership is `config.live`; `config.backtest` does not accept an override.
  - `0`: minimal lookback window at the consumer's native sampling resolution (resets as often as that path can meaningfully observe).
  - `> 0`: rolling window of that many days.
  - `"all"`: full available history.
  - Live and backtest use the same contract for realized-PnL risk windows: filter realized fill events to the active lookback window, then recompute cumulative PnL, current value, and peak from only that filtered sequence.
- **position_exposure_enforcer_threshold**: Per-position multiplier that triggers the position exposure enforcer. When a bot-managed position’s exposure exceeds `wallet_exposure_limit * (1 + effective_we_excess_allowance_pct) * position_exposure_enforcer_threshold`, the bot emits a reduce-only order to bring it back under control. Set <1.0 for continual trimming or `1.0` for a hard cap; use `position_exposure_enforcer_enabled = false` to disable.
- **total_exposure_enforcer_threshold**: Fraction of the configured `total_wallet_exposure_limit` that triggers the total exposure enforcer. When bot-scope aggregate exposure exceeds this threshold the bot queues reduction orders instead of new entries. Set >1.0 to allow a grace margin or `1.0` for strict enforcement; use `total_exposure_enforcer_enabled = false` to disable.
- **risk_we_excess_allowance_pct**: Per-symbol allowance above the configured wallet exposure limit that per-position logic tolerates before trimming. The effective allowance is capped at `max(0, total_wallet_exposure_limit / wallet_exposure_limit - 1)`, so it cannot expand a single symbol above the side's configured total exposure limit. Useful for smoothing reductions; leave at `0.0` for a hard cap.
- **max_realized_loss_pct**: Global realized-loss gate for close orders, anchored to peak realized balance from fill history. For each close order, if projected realized PnL would push balance below `peak_balance * (1 - max_realized_loss_pct)`, the order is blocked. Applies to all close order types (including WEL/TWEL auto-reduce and unstuck) except panic closes.
  - Default: `1.0` (disabled).
  - `<= 0.0`: block all lossy closes.
  - `>= 1.0`: disable the gate.
  - Example: with peak balance `$10,000` and `max_realized_loss_pct = 0.05`, lossy closes are blocked once projected balance would fall below `$9,500`.
- **fee_pct_fallback**: Fallback fee percentage used when a live fill has no usable quote-currency fee after reported-fee parsing and non-quote fee conversion. Default `0.0002` (0.02%). Set `0.0` to make the fallback fee exactly zero, so net realized PnL equals gross realized PnL for fallback-priced fills.
- **fee_pct_sanity_abs_max**: Absolute fee/notional sanity threshold for live fill accounting. Default `0.001` (0.1%). Reported or converted fees outside this range are replaced by `fee_pct_fallback`.
- **fee_conversion_max_age_ms**: Maximum timestamp distance allowed when using ticker data to convert a non-quote fee token to quote currency. Default `86400000` (24 hours).
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

`long_hsl_no_restart_drawdown_threshold` and `short_hsl_no_restart_drawdown_threshold` are intentionally not part of the default optimize bounds. The runtime parameters still live under `bot.{long,short}.hsl.*`, but optimizer runs disable terminal no-restart by default via:

1. `optimize.fixed_runtime_overrides["bot.long.hsl.no_restart_drawdown_threshold"] = 1.0`
2. `optimize.fixed_runtime_overrides["bot.short.hsl.no_restart_drawdown_threshold"] = 1.0`

Risk should be constrained through canonical `*_strategy_eq` metrics instead. Deprecated `*_hsl` metric names remain accepted as aliases for older configs/results.

**Validation:**

- Step must be positive; negative or zero steps are treated as continuous
- Step must not exceed the range (`high - low`); if it does, a warning is logged and the parameter is treated as continuous

### Other Optimization Parameters

- **compress_results_file**: If `true`, compresses optimize output results file to save space.
- **enable_overrides**: List of constraint overrides applied during optimization to enforce specific parameter relationships. The optimizer evaluator checks these conditions and apply the overrides before running each backtest (defaults to none):
  - **"lossless_close_trailing"**: Ensures trailing closes are profitable by enforcing `strategy.trailing_martingale.close.threshold_base_pct` > `strategy.trailing_martingale.close.retracement_base_pct`. This prevents retracement from triggering before reaching the minimum profit threshold.
  - **"forward_tp_grid"**, **"backward_tp_grid"**: Legacy v7 override names. They are not part of the V8 `trailing_martingale` close contract because `close_grid_markup_start` / `close_grid_markup_end` no longer define a linear TP grid.
  - **"mirror_short_from_long"**: Mirrors `bot.short` from `bot.long` for the shared side groups (`risk`, `forager`, `hsl`, `unstuck`) plus the active strategy selected by `live.strategy_kind`. This is useful when optimizing a mirrored long/short strategy while only searching the long-side parameter space.
- **crossover_probability**: Probability of performing crossover between two individuals in the genetic algorithm. Determines how often parents exchange genetic information to create offspring.
- **crossover_eta**: Crowding factor (η) for simulated-binary crossover. Lower values (<20) allow offspring to move farther away from their parents; higher values keep them closer. Default is `20.0`.
- **fixed_params**: List of dotted config-path selectors to freeze at the current config value for the whole run. Selectors match full path segments by prefix or suffix, not partial substrings. The leading `bot.` may be omitted for side-local paths, so `long.strategy` freezes `bot.long.strategy.<active_strategy>.*` and leaves `bot.long.risk`, `bot.long.forager`, and `bot.long.unstuck` tunable. A leaf selector such as `we_excess_allowance_pct` matches every optimizer bound whose config path ends with that parameter name. A `*` path segment is a one-segment wildcard. `--fine_tune_params` uses the same selector contract for the inverse operation.
- **fixed_runtime_overrides**: Runtime-only overrides applied during optimize evaluations without mutating the stored config. Use this for optimizer-specific safety knobs such as disabling terminal HSL no-restart while still keeping the live/backtest config unchanged on disk.
- **iters**: Number of backtests per optimize session.
- **mutation_probability**: Probability of mutating an individual in the genetic algorithm. Determines how often random changes are introduced to maintain diversity.
- **mutation_eta**: Crowding factor (η) for polynomial mutation. Smaller values (<20) produce heavier-tailed steps that explore more aggressively, while larger values confine mutations near the current value. Default is `20.0`.
- **mutation_indpb**: Probability that each attribute mutates when a mutation is triggered. Set to `0` (default) to auto-scale to `1 / number_of_parameters`, or supply an explicit probability between `0` and `1`.
- **n_cpus**: Number of CPU cores utilized in parallel.
- **offspring_multiplier**: Multiplier applied to `population_size` to determine how many offspring (`λ`) are produced each generation in the μ+λ evolution strategy. Values >1.0 increase exploration by sampling more children per generation. Default is `1.0`.
- **pareto_max_size**: Maximum number of Pareto-optimal configs kept on disk under `optimize_results/.../pareto/`. Members are pruned by crowding (least diverse removed first, while per-objective extremes are preserved), not by age. Default is `1000`.
- **population_size**: Size of population for genetic optimization algorithm. With the default `pymoo` backend, `null` means auto: NSGA-II resolves to `250`, while NSGA-III resolves to a default population budget of `500` and chooses the finest auto reference-direction grid that fits inside that budget. Set an explicit integer to change the NSGA-III per-generation evaluation budget and auto reference-direction coarseness.
- **backend**: Optimizer backend. Default is `pymoo`. With the default `optimize.pymoo.algorithm: "auto"`, Passivbot uses `nsga2` for `3` or fewer objectives and `nsga3` for `4+`.
- **round_to_n_significant_digits**: Quantization precision used when hashing configs, deduplicating candidates, and writing optimizer artifacts. Lower values collapse near-identical candidates more aggressively; higher values preserve more distinct variants.
- **scoring**:
  - The optimizer minimizes the configured objective list and keeps the Pareto front.
  - The current default profile uses:
    - `adg_strategy_eq`
    - `adg_strategy_eq_w`
    - `mdg_strategy_eq`
    - `mdg_strategy_eq_w`
    - `strategy_eq_recovery_days_max`
    - `position_held_days_max`
    - `drawdown_worst_strategy_eq`
    - `drawdown_worst_mean_1pct_strategy_eq`
  - With the default `pymoo` backend, Passivbot uses `nsga2` for `3` or fewer objectives and `nsga3` for `4+` objectives unless explicitly overridden.
  - Full list of options: `[adg, adg_w, calmar_ratio, calmar_ratio_w, drawdown_worst, drawdown_worst_mean_1pct, equity_balance_diff_neg_max, equity_balance_diff_neg_mean, equity_balance_diff_pos_max, equity_balance_diff_pos_mean, expected_shortfall_1pct, gain, hard_stop_duration_minutes_max, hard_stop_duration_minutes_mean, hard_stop_flatten_time_minutes_mean, hard_stop_halt_to_restart_equity_loss_pct, hard_stop_panic_close_loss_drawdown_pct_max, hard_stop_panic_close_loss_drawdown_pct_mean, hard_stop_panic_close_loss_drawdown_pct_min, hard_stop_panic_close_loss_max, hard_stop_panic_close_loss_sum, hard_stop_post_restart_retrigger_pct, hard_stop_time_in_orange_pct, hard_stop_time_in_red_pct, hard_stop_time_in_yellow_pct, hard_stop_trigger_drawdown_mean, high_exposure_days_max_long, high_exposure_days_max_short, high_exposure_hours_max_long, high_exposure_hours_max_short, loss_profit_ratio, loss_profit_ratio_w, mdg, mdg_w, omega_ratio, omega_ratio_w, peak_recovery_days_equity, peak_recovery_days_pnl, peak_recovery_days_strategy_eq, peak_recovery_hours_equity, peak_recovery_hours_pnl, peak_recovery_hours_strategy_eq, position_held_days_max, position_held_days_mean, position_held_days_median, position_held_hours_max, position_held_hours_mean, position_held_hours_median, position_unchanged_days_max, position_unchanged_hours_max, positions_held_per_day, sharpe_ratio, sharpe_ratio_w, sortino_ratio, sortino_ratio_w, strategy_eq_recovery_days_max, strategy_eq_recovery_days_mean, strategy_eq_recovery_days_mean_worst_1pct, strategy_eq_recovery_days_mean_worst_5pct, strategy_eq_recovery_days_median, strategy_eq_recovery_days_p95, strategy_eq_recovery_days_p99, sterling_ratio, sterling_ratio_w]`
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

Any metric listed above can be used when defining limits. Currency-specific metrics use `_usd` and `_btc` suffixes where both denominations are available; BTC-denominated metrics are available even when `backtest.btc_collateral_cap = 0`. This includes the shared HSL metrics such as `hard_stop_time_in_red_pct`, `hard_stop_post_restart_retrigger_pct`, and `hard_stop_halt_to_restart_equity_loss_pct`, plus `backtest_completion_ratio` for rejecting truncated runs. HSL metrics are account-level shared metrics and therefore remain single-valued rather than being split into `_usd` and `_btc`. Each limit entry is a dictionary with:

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
