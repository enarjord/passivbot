# Passivbot Parameters Explanation

For editor completion and early structure/type checks, see the generated
[v9 JSON Schema](../schemas/config-v9.schema.json) and the
[editor setup guide](config_editor.md). Runtime validators still enforce context-dependent
trading, dataset and optimizer rules.

This document explains the canonical config schema used by Passivbot.

- The source of truth for defaults is `src/config/schema.py`.
- The example config `configs/examples/default_trailing_martingale_long.json` provides the maintained default strategy profile.
- If you omit `config_path`, Passivbot loads those in-code defaults.

For the recommended user workflow, examples, and best practices, see [Config Workflow](config_workflow.md).

## Config version

`config_version` is a top-level schema field, not a backtest setting or package version.
Current configs use the final `v9.0.0` layout. Released v8 inputs require
[explicit migration](v8_to_v9_migration.md), semantic choices and fresh backtests.
Normal loading rejects unknown/retired fields and applies documented defaults. Pre-v8,
`pb_multi` and unreleased intermediate layouts are unsupported. See [release status](releases.md)
for the distinction between schema, package version and tags.

## Backtest Settings

- **base_dir**: Location to save backtest results.
- **compress_cache**: Set to `true` to save disk space. Set to `false` for faster loading.
- **ew_half_life_days**: `null` by default (disabled), or a finite number at least `1`, including fractional days such as `180.5`. Required when scoring or limits select an `_ew` metric. Applies globally across exchanges and scenarios; scenario overrides are rejected. CLI: `--backtest.ew_half_life_days 180.5` (or `null` to disable). See [exponential metric definitions](metrics.md#exponential-recency-weighting).
- **end_date**: End date of backtest, e.g., `2024-06-23`. Set to `'now'` to use today's date as the end date.
- **exchanges**: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing. The most exercised choices are `binance` and `bybit`, which are also the current default profile. `bitget` and `gateio` are supported, with the GateIO history caveat below. `kucoin` has archive-fetch support but should be treated as experimental for backtesting until broader real-data smoke coverage is collected. Use short exchange names in configs; Passivbot converts to CCXT-specific IDs such as `binanceusdm` and `kucoinfutures` only when it talks to CCXT.
  Binance backtests fill missing v2 1m data from Binance Vision monthly archives first, eligible
  daily archives second, and CCXT last. Monthly archives are used only for closed, published months
  with more than seven days of missing candles; daily archives are used for older days with more
  than 1,000 missing candles. Recent or small gaps and gaps inside archive data fall back to CCXT.
  Archive downloads run concurrently within each tier and are verified against Binance's SHA-256
  sidecars before being written to the v2 store.
  **GateIO note:** If you already have `caches/ohlcv/gateio` data on disk, delete it before a fresh run so Passivbot rebuilds the cache with base-volume-normalized data.
  GateIO's public 1m OHLCV endpoint only serves a recent window of roughly 10,000 candles; use `backtest.ohlcv_source_dir` or another candle source for older GateIO backtests.
- **coin_sources**: Optional mapping of `coin -> exchange` used to override the automatic exchange selection when multiple exchanges are configured. Scenarios may add more overrides; conflicting assignments raise an error.
- **market_settings_sources**: Optional mapping of `coin -> exchange` used specifically for exchange metadata such as `price_step`, `qty_step`, fees, and min-size rules. This is separate from `coin_sources`: you may source candles from one exchange while borrowing market settings from another.
- **ohlcv_source_dir**: Optional path to a pre-populated OHLCV directory to read directly before using Passivbot's v2 raw OHLCV store or exchange archives. Expected structure: `<dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npz` or `.npy`. Coin keys are normalized to base coins, but CCXT-style symbol folder names are accepted (e.g., `ETH_USDC:USDC`). Passivbot still writes the final prepared HLCV dataset cache for the run, but it does not mirror explicit source-dir raw candles into `caches/ohlcvs/`.
- **hlcvs_data_dir**: Optional path to a prepared final HLCV dataset under
  `caches/hlcvs_data/`. The dataset must have a valid manifest whose hashes
  verify `hlcvs`, `timestamps`, `btc_usd_prices`, `coins`, and
  `market_specific_settings`.
- **hlcvs_data_override_mode**: How `hlcvs_data_dir` is matched to the current
  config. `intersection` (default) keeps the config's requested coins/date
  window clipped to the verified dataset. `dataset` adopts the dataset's
  effective coins and timestamp window for exact artifact replay.
- **volume_normalization**: When `true` (default), normalize volume data across exchanges using
  the median of per-coin median daily log-volume ratios over each coin/exchange pair's latest
  complete-overlap UTC days. Daily estimates require at least 95% common minute coverage, and
  underdetermined exchange links fail loudly instead of silently using an unstable factor. Forced
  source exchanges contribute normalization-only overlap candidates, bounded to the normalization
  lookback, without entering unrelated source selection. `coin_sources` overrides outside the
  effective coin universe do not trigger candle preparation. Set to `false` to preserve each selected
  exchange's native volume. The first selected exchange in `backtest.exchanges` is the stable
  normalization reference. Equivalent full-range candle sources are also selected by that
  configured order; total volume is never a source-selection tie-breaker. Cache manifests and
  backtest `dataset.json` artifacts record the chosen sources, reasons, estimation window,
  contributors, dispersion, exclusions, reference exchange, and full-precision scale factors.
- **start_date**: Start date of backtest.
- **starting_balance**: Starting balance in USD at the beginning of the backtest.
- **filter_by_min_effective_cost**: `null` (default) inherits `bot.global.universe.filter_by_min_effective_cost`; explicit `true` or `false` overrides that policy for simulation. When enabled, skip coins whose projected initial entry
  (balance × wallet_exposure_limit × the active strategy initial sizing fraction, including WE
  excess allowance)
  would fall below the exchange’s effective minimum cost.
- **dynamic_wel_by_tradability**: Backtest-only WEL denominator mode.
  - `true` (default): `wallet_exposure_limit = total_wallet_exposure_limit / min(max_positions, n_tradable_max)` where `n_tradable_max` is the highest number of coins that have had real candles at any timestep so far (non-shrinking).
  - `false`: fixed denominator, same as live: `wallet_exposure_limit = total_wallet_exposure_limit / max_positions`.
- **candle_interval_minutes**: Aggregates raw 1m OHLCVs into coarser candles before the backtest loop runs. `1` keeps native 1m behavior; values above `1` speed up backtests and optimizer runs at the cost of losing intra-interval fill ordering.
- **gap_tolerance_ohlcvs_minutes**: Maximum internal hole size that can be filled in prepared OHLCV data. Larger or persistent gaps are repaired from local v2 data, legacy shards, and targeted remote fetches; if a large internal gap remains, it is excluded from the returned tradable window rather than made tradable with synthetic candles. Verified exchange-side late starts and early ends do not by themselves abort a run, but local corruption, malformed candles, missing BTC benchmark data, or no tradable candles still fail loudly.
- **equity_floor_ratio**: Early-stop backtest equity-floor guard. The run terminates once total equity falls to or below `starting_balance * equity_floor_ratio`, and `backtest_completion_ratio` will fall below `1.0`. Example: with `starting_balance = 1000` and `equity_floor_ratio = 0.05`, the backtest stops at equity `<= 50`. This is not a “5% drawdown” threshold; if the run never rises above the start, it corresponds to roughly a `0.95` worst drawdown. Must satisfy `0.0 <= equity_floor_ratio < 1.0`.
- **maker_fee_override**: Optional maker fee override (part-per-one; use `0.0002` for 0.02%). Leave `null` to use exchange-derived per-coin maker fees. CLI: `--maker-fee-override`.
- **taker_fee_override**: Optional taker fee override (part-per-one; use `0.00055` for 0.055%). Leave `null` to use exchange-derived per-coin taker fees. CLI: `--taker-fee-override`.
- **limit_order_fill_buffer_fraction**: Backtest-only fill buffer, expressed as a fraction of the limit price (`0.0001` = 0.01%). A buy fills only when `low < limit_price * (1 - buffer)`; a sell fills only when `high > limit_price * (1 + buffer)`. Equality does not fill. Successful fills retain the original limit price and maker fee. Applies to all limit entries and closes, including protective limit closes, but not market execution. Default `0.0` preserves the original strict-crossing behavior. Must be finite and in `[0, 1)`. CPU and GPU optimization support this fixed simulation setting. GPU screening precomputes buffered fill boundaries without altering raw candles or adding per-candidate kernel work. Higher values may skip losing entries as well as profitable exits, so they do not guarantee worse performance.
- **market_order_slippage_fraction**: Backtest-only slippage applied whenever the backtester simulates market-order execution. This applies both to HSL panic closes when `bot.{long,short}.hsl.panic_close_order_type` is `"market"` and to normal orchestrator orders promoted to market execution by `bot.global.execution.market_orders_allowed`. A sell fills at `close * (1 - slippage_pct)` rounded down to `price_step`; a buy fills at `close * (1 + slippage_pct)` rounded up. The fill is guaranteed once the market-execution path is chosen, and the resulting fill also uses taker fees. Default `0.0005` (5 bps). This field is not a live slippage cap; live market orders use the exchange adapter's order semantics and any exchange/CCXT slippage controls.
- **visible_metrics**: Controls which metrics are printed to the terminal after a standalone backtest. `null` shows the metrics implied by `optimize.scoring` and `optimize.limits`, `[]` shows all metrics, and an explicit list adds extra named metrics to the default view. This affects CLI visibility only; the full metric set is still computed and persisted.
  Fill-activity metrics use the `fills_*` prefix, including fill counts, per-day entry/close and long/short rates, no-fill gap durations, per-position-slot activity, active fill day counts/ratio, analysis duration, active symbol count, and top-symbol fill share.
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
- **backtest.reducer**: Dict of metric-specific reduction modes (default `mean`). Keys fall back to the `default` entry if unspecified.

`reducer` is the canonical field name for suite reduction in `backtest`, `optimize.scoring`, and
`optimize.limits`. Final v9 configs use `reducer` and `metric` explicitly; normal loading rejects
retired structural aliases such as `aggregate`, `stat`, `scenario_stat`, and limit `field`.
The explicit released-v8 converter normalizes supported historical spellings and reports the
changes; conflicting values are rejected.

See [Suite Examples](suite_examples.md) for practical examples and suggested usage.

Example per-metric aggregation:

```json
"backtest": {
  "reducer": {
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
- **persist_to_file**: When `true`, `passivbot live` also writes the console log stream to a timestamped file on disk and refreshes `logs/{user}.log` as a stable alias to the current run. On Windows without symlink privileges, this alias is a text pointer containing the absolute path to the current run log; Passivbot's monitor tooling follows it automatically. The canonical default is `true`, so live runs write to `logs/` unless you disable it explicitly. In this first integrated version, backtest/optimize still use console logging unless you wrap them externally.
- **dir**: Directory used for persisted live log files and the stable current-run alias when `persist_to_file` is enabled. Default `logs`.
- **rotation**: Enables rotating live log files instead of appending to one file per process. Default `true`.
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
- **price_tick_min_interval_milliseconds**: Minimum per-symbol interval for emitting `history/price_ticks.current.ndjson` entries.
- **emit_completed_candles**: Enable or disable completed 1m/1h candle history publication.
- **include_raw_fill_payloads**: If `true`, include exchange/raw fill payloads alongside the normalized fill history payload.

See [monitor.md](monitor.md) for current output files and event kinds.

## Bot Settings

### Equity Hard Stop Loss

HSL settings live under `bot.long.hsl` and `bot.short.hsl` for coin/position_side modes,
and explicit `bot.global.hsl` for portfolio mode. Supported leaves are `enabled`,
`drawdown_ratio_threshold`, `drawdown_ratio_ema_span_minutes`, `memory_ema_span_minutes`,
`cooldown_after_red_minutes`, and `panic_close_order_type`. Coin scope also supports
`scale_budget_with_excess_allowance`.
HSL has one implementation; obsolete engine selectors are not runtime options. See the
[configuration migration rules](#hsl-configuration) before using older settings.

`bot.global.hsl.scope` selects `coin` (default), `position_side`, or `portfolio` signal construction.

For parameter definitions, RED handling, cooldown/restart rules, per-coin overrides, and
backtest behavior, see [Equity Hard Stop Loss](equity_hard_stop_loss.md) and the
[HSL reference](equity_hard_stop_loss_reference.md). The [metrics reference](metrics.md)
describes HSL analysis outputs. These focused guides are the maintained reference for HSL
behavior.

### General Parameters for Long and Short

- **strategy.trailing_martingale.entry.ema_span_0_minutes**, **entry.ema_span_1_minutes**:
  - Spans are given in minutes.
  - Formula: `next_EMA = prev_EMA * (1 - alpha) + new_val * alpha`. For a base candle interval
    of `c` minutes, `alpha = min(1, 2 / (span_minutes / c + 1))`; live price candles use `c = 1`.
  - An additional EMA span is calculated as `(ema_span_0_minutes * ema_span_1_minutes)**0.5`.
  - The three EMAs form an upper and lower EMA band:
    - `ema_band_lower = min(emas)`
    - `ema_band_upper = max(emas)`
  - These bands govern entry gating, forager entry readiness, and one-way entry arbitration.
  - Auto-unstuck uses its own `unstuck.ema_span_0_minutes` and `unstuck.ema_span_1_minutes`; ordinary trailing-martingale closes do not use the entry price band.
  - Final v9 loading requires these entry paths. The explicit released-v8 converter moves old
    strategy-root spans into `entry` and rejects conflicting authored values.
- **max_positions**: Maximum number of positions to open. Set to `0` to disable long/short.
- **total_wallet_exposure_limit**: Maximum exposure allowed.
  - Example: `total_wallet_exposure_limit = 0.75` means 75% of (unleveraged) wallet balance is used.
  - Example: `total_wallet_exposure_limit = 1.6` means 160% of (unleveraged) wallet balance is used.
  - Live denominator is fixed: `wallet_exposure_limit = total_wallet_exposure_limit / max_positions`.
  - Backtest denominator is controlled by `backtest.dynamic_wel_by_tradability`.
  - See more: `docs/risk_management.md`.

### Trailing Martingale Entries

The retained strategy kinds are `trailing_martingale` and `ema_anchor`.
Trailing Martingale uses threshold/retracement fields: with retracement disabled the threshold
acts as recursive DCA spacing; with retracement enabled it is the required excursion before
confirmation. The v7 strategy and its dedicated migration command have been removed.
Released v8 inputs require the [explicit v9 converter](v8_to_v9_migration.md).

- **strategy.trailing_martingale.entry.double_down_factor**:
  - Quantity of the next re-entry is position size times the double down factor.
  - Example: If position size is `1.4` and `double_down_factor` is `0.9`, then the next entry quantity is `1.4 * 0.9 = 1.26`.
- **strategy.trailing_martingale.entry.threshold_base_fraction**:
  - Re-entry threshold from the current position price.
  - If entry retracement is disabled:
    - `next_reentry_price_long = pos_price * (1 - effective_threshold)`
    - `next_reentry_price_short = pos_price * (1 + effective_threshold)`
  - If entry retracement is enabled, price must first move at least this far in the favorable direction before the retracement condition may place an order.
- **strategy.trailing_martingale.entry.threshold_we_weight**, **threshold_volatility_1h_weight**, **threshold_volatility_1m_weight**:
  - Entry threshold is widened multiplicatively:
    - `effective_threshold = threshold_base_fraction * max(1, 1 + we_ratio * threshold_we_weight + volatility_1h * threshold_volatility_1h_weight + volatility_1m * threshold_volatility_1m_weight)`
  - `we_ratio = wallet_exposure / effective_wallet_exposure_limit`.
  - Positive weights make entries less eager as exposure or volatility rises.
- **strategy.trailing_martingale.entry.retracement_base_fraction**:
  - If `<= 0.0`, trailing confirmation is disabled and entries are passive recursive limit orders.
  - If `> 0.0`, the bot waits for a favorable excursion past the threshold and then for a pullback by the effective retracement before placing the entry.
- **strategy.trailing_martingale.entry.retracement_we_weight**, **retracement_volatility_1h_weight**, **retracement_volatility_1m_weight**:
  - Entry retracement is widened multiplicatively with the same `max(1, 1 + ...)` style as entry threshold.
- **strategy.trailing_martingale.volatility_ema_span_1h_candles**, **volatility_ema_span_minutes**:
  - Volatility is the EMA of per-candle log range `ln(high / low)` on 1h and 1m candles.
  - The hourly horizon is a floating count of calendar-hour candle samples. The minute horizon
    is converted to base-candle periods when backtests aggregate their input candles.
  - A volatility weight of `0` disables that horizon for the affected threshold or retracement.
- **strategy.trailing_martingale.entry.initial_ema_distance_fraction**:
  - Offset from lower/upper EMA band.
  - Long initial entry/short unstuck close prices are lower EMA band minus offset.
  - Short initial entry/long unstuck close prices are upper EMA band plus offset.
  - See `ema_span_0_minutes`/`ema_span_1_minutes`.
- **strategy.trailing_martingale.entry.ema_gate_mode**:
  - Fixed enum, not optimized. Allowed values are `disabled`, `all`, `initial`, and `reentry`.
  - `initial` is the default: normal initial and partial initial entries are EMA-gated; re-entries are not.
  - `disabled` places initial/partial-initial entries at best bid/ask and leaves re-entries to normal threshold/retracement logic.
  - `all` gates initial, partial-initial, and re-entry orders. `reentry` gates re-entries only.
  - In one-way mode, when both sides are flat and otherwise eligible, EMA bands are still required for the long-vs-short tie-break even if entry EMA gating is disabled.
- **strategy.trailing_martingale.entry.initial_qty_fraction**:
  - `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_fraction`
  - This is the initial sizing fraction used by min-effective-cost checks when
    `bot.global.strategy_kind = "trailing_martingale"`.

### Trailing Martingale Closes

Close threshold/retracement mirrors entries, but close threshold is additive so it can intentionally move through break-even or negative markup as wallet exposure rises.

- **strategy.trailing_martingale.close.qty_fraction**:
  - Fraction of the budget-derived full position at the position entry price, plus any over-budget remainder; rounding, minimums and the actual position cap still apply.
  - When close retracement is disabled and the close threshold does not depend on wallet exposure, recursive closes collapse to the same price. Rust intentionally emits one full-position close in that case because multiple same-price slices are redundant.
  - When close threshold depends on wallet exposure, `qty_fraction` controls the recursive ladder: each synthetic close fill lowers `we_ratio`, then the next close is recomputed.
- **strategy.trailing_martingale.close.threshold_base_fraction**:
  - Base close distance from position price.
  - With retracement disabled, this acts as the resting close markup:
    - `close_price_long = pos_price * (1 + effective_threshold)`
    - `close_price_short = pos_price * (1 - effective_threshold)`
  - With retracement enabled, this is the first trailing condition: market price must first reach the threshold before retracement can trigger the close.
- **strategy.trailing_martingale.close.threshold_we_weight**:
  - Close threshold is adjusted additively by exposure:
    - `effective_threshold = threshold_base_fraction + we_ratio * threshold_we_weight + volatility_1h * threshold_volatility_1h_weight + volatility_1m * threshold_volatility_1m_weight`
  - Negative values make closes more eager as position exposure grows, including break-even or negative-markup closes.
  - Positive values make closes less eager as exposure grows.
- **strategy.trailing_martingale.close.threshold_volatility_1h_weight**, **threshold_volatility_1m_weight**:
  - Positive values shift close thresholds farther from position price during volatile periods.
- **strategy.trailing_martingale.close.retracement_base_fraction**:
  - If `<= 0.0`, close trailing is disabled and closes are resting limit orders.
  - If `> 0.0`, the bot waits for price to reach the close threshold and then retrace by the effective retracement.
- **strategy.trailing_martingale.close.retracement_volatility_1h_weight**, **retracement_volatility_1m_weight**:
  - Close retracement is widened multiplicatively by volatility only. There is intentionally no close retracement wallet-exposure weight.

The V8 `trailing_martingale` schema is a clean break from the v7 `trailing_grid` schema. Removed concepts include `entry_trailing_grid_ratio`, `close_trailing_grid_ratio`, `close_grid_markup_start`, and `close_grid_markup_end`. The old `markup_start`/`markup_end` linear TP grid is not part of `trailing_martingale` behavior; recursive closes are now driven by `close.threshold_*`, `close.retracement_*`, and `close.qty_fraction`. The v7 compatibility strategy is removed.

### Unstuck Parameters

If a position is stuck, the bot uses profits from other positions to realize losses for the stuck position. If multiple positions are stuck, the position with the lowest price action distance is selected for unstucking.

These settings live under `bot.<side>.unstuck`. Effective WEL includes the resolved excess
allowance: `effective_wel = wallet_exposure_limit * (1 + effective_we_excess_allowance_pct)`.

- **unstuck.close_fraction**:
  - Fraction of the effective exposure budget used to size each close, not a fraction of the current position.
  - Before quantity rounding and other constraints, `close_qty_abs = balance * effective_wel * close_fraction / (close_price * c_mult)`.
  - Rust applies quantity-step rounding, exchange minimums, remaining-position sizing and loss-allowance scaling. There is no additional cap to stop precisely at `unstuck.exposure_trigger_ratio`.
- **unstuck.ema_distance_fraction**:
  - Offset for the EMA eligibility trigger, not the submitted order price:
    - Long: current price must reach `upper_EMA_band * (1 + ema_distance_fraction)`, rounded up to the price tick.
    - Short: current price must reach `lower_EMA_band * (1 - ema_distance_fraction)`, rounded down to the price tick.
  - Once eligible, the close uses current price rounded up for longs or down for shorts.
- **unstuck.ema_gating_enabled**:
  - Fixed boolean toggle for the auto-unstuck EMA trigger. Default is `true`.
  - When `false`, auto-unstuck skips the EMA trigger/readiness check but still requires loss allowance, exposure threshold, close sizing, and valid market/exchange inputs.
- **unstuck.loss_allowance_fraction**:
  - Sets a realized-loss budget weighted by the side's `total_wallet_exposure_limit`.
  - `balance_peak = balance + (realized_pnl_cumsum_max - realized_pnl_cumsum_last)`.
  - `loss_floor = balance_peak * (1 - loss_allowance_fraction * total_wallet_exposure_limit)`; remaining allowance is `max(0, balance - loss_floor)`.
  - Example: A reconstructed peak balance of `$10,000`, `loss_allowance_fraction = 0.02`, and `total_wallet_exposure_limit = 1.5` give a `$9,700` floor. At a current balance of `$9,800`, the remaining allowance is `$100`.
  - This is a pacing budget: exchange minimum sizing may exceed it. See the [loss-allowance contract](risk_management.md#auto-unstuck-loss-allowance-contract).
  - Per-coin overrides may set `bot.<side>.unstuck.loss_allowance_fraction`; the selected coin+side then uses that fraction in the same account-wide formula.
- **unstuck.exposure_trigger_ratio**:
  - Exposure eligibility trigger: `wallet_exposure / effective_wel > threshold`. Equality does not qualify.
  - Wallet exposure uses the position's average entry price: `abs(position_size) * position_price * c_mult / balance`.
  - This is not a target remaining exposure or a floor on the close. An eligible position can finish below the threshold after a close sized by `close_fraction`.
  - For an illustrative long at 100% of effective WEL, `threshold = 0.90` and `close_fraction = 0.12` would leave about 88% if close price equals entry price and balance is unchanged, before rounding and other constraints. The order is not reduced to leave exactly 90%.
  - When close price differs from entry price, the exposure reduction also differs from `close_fraction`; see the [sizing formulas](config.bot.md#auto-unstucking).

One non-panic protective reducer (TWEL/WEL auto-reduce or auto-unstuck) may coexist with ordinary
grid, trailing, or EMA-anchor closes for the same position. Passivbot reserves the reducer quantity
first and caps ordinary closes to the remaining position quantity. Panic close remains exclusive,
and competing protective reducers do not stack: Passivbot keeps the largest loss-admissible
absolute reduction after final position/minimum sizing rather than summing active safety intents,
trying the next-largest intent when the realized-loss gate blocks a larger non-panic close.
Final reducer quantities compete largest-first across the shared batch loss allowance.

### Filter Parameters

Forager ranks eligible candidates using volume, EMA readiness, volatility and unilateralness.
The old volume-drop filter is removed; volume continues to contribute through its configured weight.

- **forager.volatility_ema_span_minutes / forager.volume_ema_span_minutes**: Number of minutes to look into the past to compute the 1m volatility (log-range) and quote-volume EMAs used by forager mode.
  - Log range is computed from 1m OHLCVs as `mean(ln(high / low))`.
  - These spans control the raw inputs to forager ranking; they are separate from strategy volatility spans such as `volatility_ema_span_1h_candles` and `offset_volatility_ema_span_1h_candles`.
- **forager_score_weights**: Final weighted forager ranking weights.
  - Required keys: `volume`, `ema_readiness`, `volatility`, `unilateralness`.
  - The unilateralness weight defaults to zero; see the canonical template for other weights.
  - Positive weights are relative and normalized to unit sum before use.
  - If all weights are `0.0`, Passivbot normalizes them to EMA-readiness-only ranking.
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
        entry.ema_span_0_minutes, entry.ema_span_1_minutes, volatility_ema_span_1h_candles, volatility_ema_span_minutes,
        entry.double_down_factor, entry.initial_ema_distance_fraction, entry.initial_qty_fraction,
        entry.threshold_base_fraction, entry.threshold_we_weight,
        entry.threshold_volatility_1h_weight, entry.threshold_volatility_1m_weight,
        entry.retracement_base_fraction, entry.retracement_we_weight,
        entry.retracement_volatility_1h_weight, entry.retracement_volatility_1m_weight,
        close.qty_fraction, close.threshold_base_fraction, close.threshold_we_weight,
        close.threshold_volatility_1h_weight, close.threshold_volatility_1m_weight,
        close.retracement_base_fraction,
        close.retracement_volatility_1h_weight, close.retracement_volatility_1m_weight
      ]
      ```
    - `config.bot.long/short` shared fields:
      ```
      [
        risk.position_exposure_enforcer.enabled,
        risk.position_exposure_enforcer.threshold_ratio,
        entry_cooldown.base_duration_minutes,
        risk.we_excess_allowance_fraction,
        hsl.cooldown_after_red_minutes,
        hsl.drawdown_ratio_ema_span_minutes,
        hsl.enabled,
        hsl.panic_close_order_type,
        hsl.drawdown_ratio_threshold,
        unstuck.close_fraction,
        unstuck.ema_distance_fraction,
        unstuck.ema_gating_enabled,
        unstuck.enabled,
        unstuck.loss_allowance_fraction,
        unstuck.exposure_trigger_ratio,
        wallet_exposure_limit
      ]
      ```
      The `hsl.*` entries are eligible only when the main config's global
      `bot.global.hsl.scope` is `"coin"`. The signal mode is not itself overridable.
    - `config.live`:
    ```
    [live.modes.long, live.modes.short, live.account.leverage]
    ```
  - Examples:
    - `{"COIN1": {"override_config_path": "path/to/override_config.json"}}` -- Loads the required file and applies all explicitly present eligible parameters from it for COIN1
    - `{"COIN2": {"override_config_path": "path/to/other_override_config.json", "bot": {"long": {"strategy": {"trailing_martingale": {"close": {"threshold_base_fraction": 0.005}}}}}}}` -- Will attempt to load `"path/to/other_override_config.json"` first, and apply the given close threshold override after.
    - `{"COIN3": {"bot": {"short": {"strategy": {"trailing_martingale": {"entry": {"initial_qty_fraction": 0.01}}}}}, "live": {"modes": {"long": "panic"}}}}` -- Will apply given overrides for COIN3.
- **forced_modes**:
  - Choices: `[n (normal), m (manual), gs (graceful_stop), t (tp_only), p (panic)]`.
    - **Normal mode**: Passivbot manages the position as normal.
    - **Manual mode**: Passivbot ignores the position.
    - **Graceful stop**: If there is a position, Passivbot manages it; otherwise, no new positions are opened.
    - **Take Profit Only mode**: Passivbot only manages closing orders.
    - **Panic mode**: Passivbot closes the position immediately.

## Shared Bot Policy

Behavior shared by live trading, backtests and optimization belongs in `bot`. Global policy
lives in `bot.global`; directional policy lives in `bot.long` and `bot.short`. `live` contains
operational execution, exchange I/O and startup settings.

### Universe and selection

- **bot.<side>.universe.approved_coins**: A coin list, `"all"` for every eligible coin, or a path
  to an external coin-list file. An empty list disables entries for that side. Long and short
  have separate source leaves; live refresh preserves each side's file reference.
- **bot.<side>.universe.ignored_coins**: A coin list or external coin-list file. The side's
  tradable universe is its approved coins minus ignored coins, subject to age/market checks.
- **bot.<side>.forager.always_include**: Ordered, deduplicated coin/market identifiers that
  remain selected by Forager. They must still pass approval, ignored-coin, minimum-age,
  market-availability, risk, HSL and operational mode checks. This list does not add coins to
  the approved universe and does not override a live forced mode.

```json
"bot": {
    "long": {"universe": {"approved_coins": ["BTC", "ETH"], "ignored_coins": []}},
    "short": {"universe": {"approved_coins": "all", "ignored_coins": ["BTC"]}}
}
```

CLI `--symbols`/`-s` and `--ignored-coins`/`-ic` deliberately set both side sources.
Use `--bot.long.universe.approved_coins BTC,ETH` or the corresponding short/ignored path to
change one side. A per-side option takes precedence over a both-side convenience in the same
command. Retired `--live...` policy paths are rejected.

- **bot.global.universe.minimum_coin_age_days**: Disallows coins younger than a given number of days.

- **bot.global.forager.score_hysteresis_margin**: Absolute normalized-score margin for forager incumbent selection. Default is `0.02`, meaning an already-selected flat forager coin is kept if a challenger beats it by no more than 2.0 percentage points of final forager score. Applies to live, backtest, and optimizer.

### Strategy, execution and warmup

- **bot.global.strategy_kind**: Selects the active strategy subtree on both sides.

- **bot.global.hedge_mode**: Requests simultaneous long and short positions on the same coin when the exchange supports it. Effective behavior is `config.bot.global.hedge_mode AND exchange_capability`; on one-way-only venues the live bot will still run one-way even if this is `true`.

- **bot.global.universe.filter_by_min_effective_cost**: If `true`, disallows coins where
  `balance * allowed_wallet_exposure_limit * strategy_initial_sizing_fraction < min_effective_cost`.
  For `trailing_martingale`, the sizing fraction is
  `bot.<side>.strategy.trailing_martingale.entry.initial_qty_fraction`; for `ema_anchor`, it is
  `bot.<side>.strategy.ema_anchor.base_qty_fraction`.
  - Example: If the exchange's effective minimum cost for a coin is `$5`, but the bot wants to make an order of `$2`, disallow that coin.

- **bot.global.execution.market_orders_allowed**: If `true`, allows Passivbot to place market orders when the order price is very close to the current market price. If `false`, only places limit orders. The current default profile uses `false`. Live market orders execute as taker orders and may fill materially away from the reference price during fast moves or thin liquidity; exchange adapter and CCXT slippage controls apply.

- **bot.global.execution.market_order_near_touch_fraction**: Unified threshold used by Rust order orchestration when `market_orders_allowed` is enabled. If an order price is within this fractional distance of the current market price, Rust emits it as a market order. Crossing orders also become market orders (`bid >= market` for buys, `ask <= market` for sells). This execution intent is now shared by both live and backtest. Default is `0.001`. This threshold decides when an order may become market execution; it is not a live slippage cap.
  - Decision rules:
    - non-panic buy with `price >= market_price` => `market`
    - non-panic sell with `price <= market_price` => `market`
    - otherwise, if `abs(order_price_diff) <= market_order_near_touch_fraction` => `market`
    - otherwise => `limit`
    - panic closes are still controlled separately by `bot.{long,short}.hsl.panic_close_order_type`
  - Ownership is `config.bot.global.execution`. Backtests always inherit `bot.global.execution.market_orders_allowed` and `bot.global.execution.market_order_near_touch_fraction`; `config.backtest` does not accept overrides for either field.

- **bot.global.warmup.max_minutes**: Ceiling on the existing historical warm-up calculation for backtests and live warm-ups. Use `null` for no cap, or a positive finite number of minutes. Zero is rejected. Existing rounding remains: live acquisition truncates the cap to whole minutes, so a cap below one minute becomes internally uncapped; backtest calculation applies the fractional cap before rounding its final window upward.

- **bot.global.warmup.span_multiplier**: Controls the span-based warmup terms. Live history acquisition uses the deepest required indicator span times `1 + span_multiplier`; backtest warmup and trade activation use the span times `span_multiplier`. For example, `0.2` adds a 20% acquisition buffer in live mode (120% of the span), while the activation/backtest term is 20% of the span. These terms remain subject to `bot.global.warmup.max_minutes` and other history requirements.

## Live Trading Settings

Operational controls use eight groups. Shared strategy and selection policy lives under
[`bot`](#shared-bot-policy).

| Group | Purpose |
|---|---|
| `live.account` | Account selection, balance presentation, leverage and margin mode |
| `live.api` | Endpoint, request and ticker acquisition settings |
| `live.candles` | Candle acquisition, retention and freshness |
| `live.startup` | Startup scheduling, phase budgets and risk-history acquisition |
| `live.fills` | Fill-history overlap and fee conversion |
| `live.execution` | Order submission, matching and replacement controls |
| `live.recovery` | Process recovery and temporary symbol suspension |
| `live.modes` | Root long/short mode and disapproved-position policy |

- **live.modes.disapproved_position**: Policy for existing positions on disapproved coins.
  `graceful_stop` keeps managing the current position and prevents reopening after flat;
  `manual` delegates management to the operator. Default is `graceful_stop`.
- **live.candles.enable_archive_candle_fetch**: Enables the archive-candle fallback path in live mode. Keep `false` unless you specifically want the live bot to supplement its local candle state from exchange archive endpoints.
- **live.candles.enable_forager_ws_candles**: Persists proven-final public 1m WebSocket candles for flat forager candidates when the exchange supports CCXT Pro `watchOHLCV`. The first nonempty snapshot of each watcher session only primes provenance. Changed values may correct an existing canonical timestamp, but extending canonical history requires fresh-successor proof, and persistence is read-verified before cache/EMA exposure. REST remains the complete fallback for startup basis, gaps, reconnect recovery, prolonged silence, and forced periodic integrity overlaps. WebSocket silence never proves a no-trade candle, and unstable streams cool down while REST maintenance continues. The lightweight reconciler stays ready for later runtime transitions into forager mode. Default is `true`; set `false` to use REST-only candidate maintenance. A zero `max_ohlcv_fetches_per_minute` disables this network path too.
- **live.execution.execution_delay_seconds**: Wait `x` seconds after executing to exchange.
- **live.recovery.exchange_symbol_unavailable_cooldown_hours**: RAM-only, per-symbol entry cooldown after a connector proves from an exact exchange-specific error code that API trading is temporarily unavailable for that symbol. Default is `6.0` hours; `0` disables the cooldown; the maximum is `876600` hours (100 years). Flat affected symbols are excluded from new planning, while held symbols retain close and panic management with normal entries suppressed. Expiry retries the symbol automatically, a repeated qualifying response starts a fresh cooldown, and restarting the bot deliberately clears all cooldowns so the symbol is tried immediately. WEEX currently classifies only structured error code `-1058`; other exchanges require their own exact, fixture-tested classifier before using this policy.
- **live.candles.max_memory_candles_per_symbol**: Maximum number of 1m candles retained in RAM per symbol. Older entries are trimmed once this cap is exceeded. Default is `200_000`.
- **live.candles.max_disk_candles_per_symbol_per_tf**: Maximum number of candles persisted on disk per symbol and timeframe. Oldest shards are pruned once the limit is hit (default `2_000_000`).
- **live.candles.candle_lock_timeout_seconds**: Seconds to wait when another process holds the CandlestickManager per-symbol candle fetch lock (default `10`). Increase when running many bots sharing the same cache directory to avoid spurious timeouts during slow API calls.
- **live.candles.inactive_coin_candle_ttl_minutes**: How long 1m candles for inactive symbols may stay in RAM before the live bot refreshes them. Lower values keep inactive symbols fresher at the cost of more network/disk churn.
- **live.modes.long**, **live.modes.short**: Force all coins long/short to a given mode, or use `null` (default) to leave the mode unforced. A per-coin `null` clears the root forced mode; omission inherits it. Empty strings are rejected. CLI `--live.modes.long null` clears the setting explicitly.
  - Choices: `[m (manual), gs (graceful_stop), p (panic), t (take_profit_only)]`.
- **live.account.leverage**: Leverage set on the exchange. Default is `10`.
- **live.account.margin_mode_preference**: Preferred live margin mode when a symbol supports both cross and isolated.
  - `auto` / `auto_cross`: prefer cross when both modes are available.
  - `auto_isolated`: prefer isolated when both modes are available.
  - `cross`: require cross for new entries; isolated-only symbols are skipped for new entries but existing positions/orders remain manageable.
  - `isolated`: require isolated for new entries; cross-only symbols are skipped for new entries but existing positions/orders remain manageable.
  - If the exchange reports an already-open live position or open orders on a symbol, the live bot preserves that symbol's actual live margin mode for state management instead of forcing the configured preference mid-position.
  - Hyperliquid HIP-3 exception: isolated HIP-3 live trading is currently unsupported. Cross-capable HIP-3 markets are forced to cross for new entries, isolated-only HIP-3 markets are skipped for new entries, and existing isolated HIP-3 live state causes startup to fail loudly.
- **live.api.market_snapshot_ticker_strategy**: Selects the live market-snapshot ticker path. `auto` lets Passivbot choose the exchange-appropriate default, `bulk` requests one broad ticker snapshot when the exchange supports it, and `symbols` fetches tickers symbol by symbol.
- **live.api.custom_endpoints_path**: Optional live-config path to a custom endpoint override JSON file. Set to `null` to use the default auto-discovery behavior, set to a path such as `configs/custom_endpoints.json` to force that file, or set to `"none"` to disable endpoint overrides even if the default file exists. See [Running live](live.md#custom-exchange-endpoints).
- **live.candles.forager_ws_candle_rest_audit_minutes**: Maximum time a healthy WebSocket-advanced flat-candidate 1m tail may defer its next REST integrity audit. Default is `30`; accepted values are greater than `0` and at most `60`. Missing basis, internal gaps, WebSocket silence, reconnect gaps, and active position/open-order surfaces continue to use the existing REST readiness and refresh rules.
- **live.execution.order_match_tolerance_fraction**: Fractional relative tolerance used to match near-identical cancel/create pairs and avoid order churn. The accepted range is `0.0` through `0.01` inclusive (0% through 1%). Default is `0.0002`, meaning 0.02% relative price/quantity tolerance. When a newly proposed order is within this tolerance of an existing open order, Passivbot may keep the existing order instead of cancelling/replacing it.
- **live.execution.order_replacement_churn_gate_activation_count**: Account-wide connector-bound create-attempt count in the rolling window below which far churn-evidenced ordinary orders may still be placed. Default `10`; `0` disables only this economy deferral. Exact reconciliation and cancel-confirm-refresh-replan safety remain active. Market, risk-critical, and near-market creates never wait for economy capacity but still count.
- **live.execution.order_replacement_churn_gate_window_minutes**: Window for per-symbol Rust-ideal history and account-wide create-attempt accounting. Default `10.0` minutes. State is RAM-only and resets on bot restart.
- **live.execution.order_replacement_churn_gate_stability_minutes**: Minimum observed duration needed to prove sustained monotonic price or quantity drift, or repeated exclusive reappearance of the same semantic order cohort. Exclusive switching requires at least three appearances of the current cohort while every recurring cohort retains its ladder cardinality; its duration is fixed from the preceding appearance to the current run's start, while first appearance and first reappearance remain fail-open. A newest contiguous tight-match duration of the same length clears older evidence and its completed stable prefix becomes the symbol's observable switching-history boundary. Default `2.0` minutes.
- **live.execution.order_replacement_churn_gate_market_distance_fraction**: Final side-aware market-distance threshold inside which a current limit order is always exempt from churn-allowance waiting. Default `0.005` (0.5%). This never preserves a stale actual order or bypasses cancel-first sequencing.
  - Churn history uses `order_match_tolerance_fraction`; there is no wider churn-only tolerance.
    Normal loading rejects the retired `order_replacement_churn_gate_tracking_tolerance_pct`
    and `initial_entry_exec_max_market_dist_pct` fields.
  - The explicit released-v8 converter removes the retired tracking tolerance. A positive old
    distance becomes `order_replacement_churn_gate_market_distance_fraction`; a `null` or non-positive
    value becomes `order_replacement_churn_gate_activation_count = 0`. Conflicting authored
    values fail. The pre-v8 `pb_multi` root shape is unsupported.
- **live.execution.max_n_cancellations_per_batch**: Cancels `n` open orders per execution. Must be greater than `max_n_creations_per_batch` so the bot can make room before posting replacement orders.
- **live.execution.max_n_creations_per_batch**: Creates `n` new orders per execution. Must be lower than `max_n_cancellations_per_batch`.
- **live.recovery.max_n_restarts_per_day**: If the bot crashes, restart up to `n` times per day before stopping completely.


- **live.candles.max_active_candle_tail_gap_minutes**: Maximum open-ended 1m candle tail gap tolerated for active symbols before staged live planning blocks that symbol's trading-critical candle surface. Default is `10`. Within this bound, Passivbot projects provisional no-trade EMA inputs for close, quote-volume, and log-range without persisting synthetic candles or normal EMA cache entries. Real candles returned later always replace prior projections on the next read; bounded historical gaps still need real candles before and after before synthetic no-trade candles are replayed.
- **live.candles.max_ohlcv_fetches_per_minute**: Live OHLCV/network budget for candle-backed indicators such as forager ranking and warm-up maintenance. Default is `24`. Set lower to reduce REST pressure; set to `0` to disallow new fetches and rely only on what is already cached.
- **live.candles.max_forager_candle_staleness_minutes**: Optional cap on acceptable completed-candle staleness for broad forager-candidate ranking and refresh budgeting. `null` lets Passivbot derive the target from `max_ohlcv_fetches_per_minute` and candidate count, with `max_active_candle_tail_gap_minutes` as the minimum grace period so the refresh budget cannot make flat candidates nontradable earlier than active symbols. Setting an explicit positive value is an operator override and may shorten that grace period.
  - Within the cap, stale flat candidates remain rankable: close EMA readiness may use bounded flat-close projection, while quote-volume and log-range EMA inputs should carry forward the latest known values rather than append synthetic zero-volume or zero-range tails.
  - Beyond the cap, or without any prior feature basis, a candidate is unavailable for new forager entries until refreshed.
- **live.candles.max_forager_candle_refresh_seconds**: Wall-time cap for one best-effort broad forager-candidate candle refresh task. Default is `45`. When the cap is reached, Passivbot pauses the broad refresh and retries remaining stale candidates later; active position/open-order candle refreshes are not capped by this setting.
- **live.startup.defer_broad_candle_warmup**: If `true`, startup warms only trading-critical symbols first and catches up broad approved-coin candles in the background. Set `false` to block startup until broad candle warmup completes.
- **live.account.balance_override**: Optional numeric override for wallet balance used by the live bot (useful for dry-runs and debugging). When set, the bot will not fetch balance from the exchange. Can also be useful when using BTC collateral and you want to keep an effectively “fixed USD balance” for sizing, instead of having the USD-denominated balance fluctuate with the BTC/USD price.
- **live.account.balance_hysteresis_snap_fraction**: Relative change from the absolute previous hysteresis balance required to snap to a new balance. Set `0.0` to disable hysteresis. Snapped balance is used for sizing and entry-gating surfaces where hysteresis is intentional; raw balance remains the basis for exact exposure-repair surfaces such as WEL/TWEL auto-reduce. Values above roughly `0.05` can make near-boundary behavior less intuitive and are surfaced by `passivbot tool live-config-preflight`.
- **live.api.recv_window_milliseconds**: Millisecond tolerance for authenticated REST calls (default `10000`). Increase if your exchange intermittently rejects requests with `invalid request ... recv_window` errors due to clock drift.
- Candlestick management is handled by the CandlestickManager with on-disk caching and TTL-based refresh. Legacy settings `ohlcvs_1m_rolling_window_days` and `ohlcvs_1m_update_after_minutes` are no longer used.
- **live.fills.fills_recent_overlap_minutes**: Time overlap used for routine incremental live fill refreshes once the fill cache is warm. Default is `10.0`, which keeps ordinary minute-by-minute fill checks narrow to reduce private REST load.
- **live.fills.fills_confirmation_overlap_minutes**: Time overlap used when fills are explicitly required to confirm an account-critical state change, such as a suspected fill or pending staged confirmation. Default is `60.0`, preserving the wider safety window for confirmation refreshes.
- Risk memory controls are consumer-owned: `bot.<side>.hsl.memory_ema_span_minutes`
  (or `bot.global.hsl.memory_ema_span_minutes` in portfolio scope),
  `bot.global.unstuck.memory_ema_span_minutes`, and `bot.global.loss_gate.memory_ema_span_minutes`.
  HSL ratio smoothing separately uses `drawdown_ratio_ema_span_minutes`. All are EMA spans,
  not half-lives or rolling cutoffs; fractional minutes are retained. Migration does not infer
  equivalence from an old rolling duration.
- **live.startup.risk_bootstrap_max_lookback_minutes**: Live-only reconstruction cap, default/maximum `129600.0` minutes (90 days). Live targets five slow spans and retains the required active cooldown horizon, capped here, with unknown/truncated history reported. Enabled cooldowns exceeding the cap fail live capability validation. This is acquisition policy, not exponential forgetting. Ordinary CPU/GPU backtesting advances forward without historical rescans.
- **position_exposure_enforcer.threshold_ratio**: Per-position multiplier that triggers the position exposure enforcer. When a bot-managed position’s exposure exceeds `wallet_exposure_limit * (1 + effective_we_excess_allowance_pct) * position_exposure_enforcer.threshold_ratio`, the bot emits a reduce-only order to bring it back under control. Set <1.0 for continual trimming or `1.0` for a hard cap; use `position_exposure_enforcer.enabled = false` to disable.
- **entry_cooldown.base_duration_minutes**: Time-based cooldown after the last position-increasing fill for that coin+position_side. Full simultaneous entry ladders are emitted only when the effective cooldown is zero and entry retracement is disabled. Any positive value enforces a cooldown of that many minutes and limits staged position-adding entries to one order, including fractional values such as `0.05` for roughly three seconds. Backtests evaluate entries on one-minute steps, so any positive sub-minute cooldown prevents a same-minute replacement/add and effectively waits until the next backtest decision minute; live trading enforces the actual millisecond duration between intra-minute checks.
  This field may be overridden per coin+side as
  `coin_overrides.<coin>.bot.<side>.entry_cooldown.base_duration_minutes`.
  Normal loading requires this canonical path and rejects `risk.entry_cooldown_minutes`.
  The explicit released-v8 converter moves the old leaf and rejects conflicting authored values.
  Supported CLI spellings write the canonical destination. Optimizer bounds use
  `optimize.bounds.<side>.entry_cooldown.base_duration_minutes`; the internal optimizer key
  `long/short_risk_entry_cooldown_minutes` remains stable. In `fixed_params` and fine-tune
  matching, documented `risk` group selectors still include the original cooldown leaf;
  `entry_cooldown` selects the cooldown block. Authored fixed/runtime and scenario patches use
  literal full canonical paths.
  Optional additive exposure and adverse-RMS weights, a floor and a ceiling live in the same
  `entry_cooldown` block. See [adaptive entry cooldown](adaptive_entry_cooldown.md) for formulas,
  zero-base behavior, per-coin overrides, shared EMA span and GPU screening support.
- The total-exposure entry cap always applies. Bot-generated entries are blocked or cropped
  before projected snapped-balance TWE, including existing same-side positions, exceeds
  `min(total_wallet_exposure_limit, total_wallet_exposure_limit * total_exposure_enforcer.threshold_ratio)`.
- **total_exposure_enforcer.enabled**: Enables TWEL auto-reduce repair for already-over-target same-side exchange exposure. Manual and panic exposure counts toward the trigger, but only managed positions can receive TWEL auto-reduce orders. Disabling repair leaves the unconditional entry cap active.
- **total_exposure_enforcer.policy**: TWEL auto-reduce candidate policy. `reduce_overweight` trims managed positions whose WE is above `total_wallet_exposure_limit * total_exposure_enforcer.threshold_ratio / effective_n_positions`, where live `effective_n_positions` follows the current dynamic tradable-slot count and falls back to the current held-position count when no symbols are entry-eligible. `reduce_portfolio` can trim any managed open position on that side. Both policies prefer profitable/breakeven reductions before shallow adverse-loss reductions and stop once projected TWE reaches the repair target. Default: `reduce_overweight`.
- **total_exposure_enforcer.threshold_ratio**: Fraction of the configured `total_wallet_exposure_limit` used by the unconditional entry cap and TWEL auto-reduce. Values below `1.0` cap entries below raw TWEL; values above `1.0` keep the entry cap at raw TWEL while delaying auto-reduce until the thresholded repair target is exceeded.
- **risk.we_excess_allowance_fraction**: Per-symbol allowance above the configured wallet exposure limit that per-position logic tolerates before trimming. The effective allowance is capped at `max(0, total_wallet_exposure_limit / wallet_exposure_limit - 1)`, so it cannot expand a single symbol above the side's configured total exposure limit. Useful for smoothing reductions; leave at `0.0` for a hard cap.
- **bot.global.loss_gate.max_drawdown_ratio**: Global realized-loss gate for close orders. With raw balance B, independent gate net-realized drawdown D, and fraction r, its remaining allowance is `max(0, r * (B + D) - D)` in the [shared Rust arithmetic](v9_risk_model.md). TWEL does not multiply r. A projected loss exceeding the remaining allowance is blocked. Applies to all close order types (including WEL/TWEL auto-reduce and unstuck) except panic closes.
  - Default: `1.0` (disabled).
  - `0.0`: block all lossy closes; negative or nonfinite values are invalid.
  - `>= 1.0`: disable the gate.
  - Example: with raw balance `$10,000`, remembered realized loss D=`$200` and r=`0.05`, remaining allowance is `$310`. Ordered net fills and elapsed-minute decay change D; deposits and UPNL do not.
- **live.fills.fallback_fee_fraction**: Fallback fee fraction of absolute quote notional used when a live fill has no usable quote-currency fee after reported-fee parsing and non-quote fee conversion. Default `0.0002` (0.02%). Set `0.0` to make the fallback fee exactly zero, so net realized PnL equals gross realized PnL for fallback-priced fills.
- **live.fills.max_abs_fee_fraction**: Absolute fee/notional sanity threshold for live fill accounting. Default `0.001` (0.1%). Reported or converted fees outside this range are replaced by `fallback_fee_fraction`.
- **live.fills.fee_conversion_max_age_milliseconds**: Maximum timestamp distance allowed when using ticker data to convert a non-quote fee token to quote currency. Default `86400000` (24 hours).
- **live.startup.warmup_jitter_seconds**: Random startup delay spread applied before warm-up work begins. This helps multiple bots sharing one machine or cache avoid stampeding the same files and APIs at the same second.
- **live.startup.warmup_concurrency**: Concurrency cap for live warm-up tasks. `"auto"` (default) retains the context/exchange policy; a positive integer sets an explicit cap. Zero, null and booleans are rejected.
- **live.api.max_concurrent_api_requests**: Optional global live REST concurrency cap. Leave `null` to use exchange/default behavior; set an integer to throttle authenticated and public request fan-out more aggressively.
- **warmup_minutes**: Not a config key. This is a derived per-coin warm-up window computed internally from `bot.global.warmup.span_multiplier`, indicator spans, and `bot.global.warmup.max_minutes`.
- **live.execution.time_in_force**: Default is Good-Till-Cancelled.
- **live.account.user**: Fetch API key/secret from `api-keys.json`.

## Optimization Settings

### Bounds

When optimizing, parameter values are constrained within the lower and upper bounds. Bounds support an optional third element specifying a discrete step size for grid-based optimization.

Bounds mirror the corresponding bot path exactly: `bot.long.forager.score_weights.volume`
has its bound at `optimize.bounds.long.forager.score_weights.volume`. Both trees use the
scopes `global`, `long`, and `short`; flattened optimizer identifiers are internal adapters.

Only explicitly listed v9 bounds authorize independent search dimensions. An omitted parameter,
side, `bounds`, or `optimize` section fixes every omitted numeric optimizer gene at its configured
value after canonical normalization. Resolved exports show these as `[value, value]`; a null cooldown ceiling remains outside
the numeric genome unless a finite bound is explicitly provided. Sorting, cleaning, and reloading
must not add recommended search ranges.

The complete schema template and the no-config optimizer command are explicit starting profiles:
their visible bounds contain recommended ranges. To replace a config's search space with that
profile deliberately, write a separate file:

```bash
passivbot tool optimizer-bounds input.json search.json --preset recommended
```

This offline authoring command replaces `optimize.bounds`, keeps other configuration policy,
and refuses to overwrite an existing output. Review its visible ranges before optimizing.
Released-v8 migration materializes its formerly inherited ranges before adopting v9 omission
semantics; previously optional adaptive bounds still freeze at configured values.

Explicit `optimize.enable_overrides` relationships still apply after candidate genes are assigned.
Their targets are derived effective values: `mirror_short_from_long` can change omitted short
parameters, `couple_unstuck_ema_spans` derives unstuck spans from strategy spans, and
`lossless_close_trailing` derives the close threshold from its retracement constraint. A fixed
range or fixed-parameter selector on such a target fixes its independent gene, not the derived
result. Review these authored relationships together with bounds and runtime/scenario pins.

Forager score weights form one normalized group. A varying weight must not renormalize an omitted,
constant, runtime-pinned, scenario-pinned, or `fixed_params` positive weight. Search all participating positive
weights explicitly or freeze the whole `forager.score_weights` group. Omitted zero weights may
remain zero, but a reachable all-zero vector also requires authority for the `ema_readiness`
normalization fallback, or a positive lower bound. These checks also apply to fine-tune selectors
and anchors, and validate active scenario pins before market or dataset preparation. A complete
scenario weight-object pin freezes the whole group; partial object replacements must use leaf
overrides or supply every required weight. Migration rejects an incompatible legacy group rather than silently widening its
optional bounds, including in best-effort mode.

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

- **Integer parameters**: Use step `1` for parameters that should be integers (e.g., `max_positions`)
- **Coarse search**: Use larger steps to reduce search space and speed up optimization
- **Known granularity**: When you know the parameter only makes sense at certain intervals

**Example Configuration:**

```json
"optimize": {
    "bounds": {
        "long": {
            "risk": {
                "max_positions": [1, 20, 1],
                "total_wallet_exposure_limit": [0.1, 2.0, 0.1]
            },
            "forager": {"score_weights": {
                "volume": [0.0, 1.0, 0.01],
                "volatility": [0.0, 1.0, 0.01],
                "ema_readiness": [0.0, 1.0, 0.01]
            }},
            "strategy": {
                "trailing_martingale": {
                    "entry": {"ema_span_0_minutes": [100, 10000], "ema_span_1_minutes": [200, 20000]}
                }
            }
        }
    }
}
```

In this example:
- `max_positions`: Integers from 1 to 20
- `total_wallet_exposure_limit`: Values 0.1, 0.2, 0.3, ..., 2.0
- `forager.score_weights.volume`: Values 0.0, 0.01, 0.02, ..., 1.0
- `ema_span_0_minutes` and `ema_span_1_minutes`: Continuous optimization (no step defined)

HSL bounds use `optimize.bounds.<side>.hsl` in coin and position_side modes, with
`drawdown_ratio_threshold`, `drawdown_ratio_ema_span_minutes`, `memory_ema_span_minutes`, and
`cooldown_after_red_minutes` leaves. Portfolio HSL uses `optimize.bounds.global.hsl`.
Independent slow spans use their canonical consumer-owned bound blocks. There is no
restart-policy field. Use independent ordinary account-equity definitions for cross-version
economic comparison; HSL signal metrics are model-dependent and all fitness is recomputed
after migration.

**Validation:**

- Step must be positive; negative or zero steps are treated as continuous
- Step must not exceed the range (`high - low`); if it does, a warning is logged and the parameter is treated as continuous

### Other Optimization Parameters

- **compress_results_file**: If `true`, compresses optimize output results file to save space.
- **enable_overrides**: List of constraint overrides applied during optimization to enforce specific parameter relationships. The optimizer evaluator checks these conditions and apply the overrides before running each backtest (defaults to none):
  - **"lossless_close_trailing"**: Ensures trailing closes are profitable by enforcing `strategy.trailing_martingale.close.threshold_base_fraction` > `strategy.trailing_martingale.close.retracement_base_fraction`. This prevents retracement from triggering before reaching the minimum profit threshold.
  - **"forward_tp_grid"**, **"backward_tp_grid"**: Legacy v7 override names. They are not part of the V8 `trailing_martingale` close contract because `close_grid_markup_start` / `close_grid_markup_end` no longer define a linear TP grid. These retired overrides are rejected.
  - **"mirror_short_from_long"**: Mirrors `bot.short` from `bot.long` for the shared side groups (`risk`, `forager`, `hsl`, `unstuck`) plus the active strategy selected by `bot.global.strategy_kind`. This is useful when optimizing a mirrored long/short strategy while only searching the long-side parameter space.
- **fixed_params**: List of dotted config-path selectors to freeze at the current config value for the whole run. Selectors match full path segments by prefix or suffix, not partial substrings. The leading `bot.` may be omitted for side-local paths, so `long.strategy` freezes `bot.long.strategy.<active_strategy>.*` and leaves `bot.long.risk`, `bot.long.forager`, and `bot.long.unstuck` tunable. A leaf selector such as `we_excess_allowance_fraction` matches every optimizer bound whose config path ends with that parameter name. A `*` path segment is a one-segment wildcard. `--fine_tune_params` uses the same selector contract for the inverse operation.
- **fixed_runtime_overrides**: Runtime-only overrides applied during optimize evaluations without mutating the stored config. Use canonical config paths for evaluation-only settings while retaining the stored proposal.
- **iters**: Number of backtests per optimize session.
- **n_cpus**: Number of CPU cores utilized in parallel.
- **pareto_max_size**: Maximum number of Pareto-optimal configs kept on disk under `optimize_results/.../pareto/`. Members are pruned by crowding (least diverse removed first, while per-objective extremes are preserved), not by age. Default is `1000`.
- **population_size**: Size of population for genetic optimization algorithm. With the default `pymoo` backend, `null` means auto: NSGA-II resolves to `250`, while NSGA-III resolves to a default population budget of `500` and chooses the finest auto reference-direction grid that fits inside that budget. Set an explicit integer to change the NSGA-III per-generation evaluation budget and auto reference-direction coarseness.
- **backend**: `pymoo` (default) or `gpu`; both use pymoo NSGA-II/NSGA-III search.
  `optimize.pymoo.algorithm="auto"` selects NSGA-II for at most three objectives and NSGA-III
  for four or more. Operator controls live under `optimize.pymoo.shared`. GPU execution uses
  supported shared-account native replay and fails explicitly for unsupported capabilities.
  See [optimizing](optimizing.md) for device scope and checkpoint compatibility.
- **round_to_n_significant_digits**: Quantization precision used when hashing configs, deduplicating candidates, and writing optimizer artifacts. Lower values collapse near-identical candidates more aggressively; higher values preserve more distinct variants.
- **scoring**:
  - The optimizer minimizes the configured objective list and keeps the Pareto front.
  - Each object-form scoring entry accepts `metric`, `goal`, and optional suite-only `scenario`
    and `reducer` selectors. An omitted `scenario` inherits `optimize.objective_scenario`; a
    named value selects that scenario; explicit `null` selects suite aggregation. `reducer`
    overrides the reducer for that objective and supports `mean`, `min`, `max`, `std`, and
    `median`. It is invalid when the objective resolves to a named scenario.
  - The current default profile uses:
    - `adg_strategy_eq`
    - `mdg_strategy_eq`
    - `sortino_ratio_strategy_eq`
    - `volume_pct_per_day_avg`
    - `sharpe_ratio_strategy_eq`
    - `strategy_eq_recovery_days_max`
    - `position_held_days_max`
    - `drawdown_worst_strategy_eq`
    - `loss_profit_ratio`
    - `strategy_eq_underwater_pct_mean`
  - With the default `pymoo` backend, Passivbot uses `nsga2` for `3` or fewer objectives and `nsga3` for `4+` objectives unless explicitly overridden.
  - Built-in shorthand metrics and their default goals are registered in
    `src/config/scoring.py`; see `docs/optimizing.md` for metric descriptions.
  - Legacy `_w` formulas and dependent aliases are removed. Select ordinary metrics or explicit
    `_ew` formulas deliberately; migration requires a replacement and formula acknowledgment.
  - Authored scoring is a list of objects, for example
    `[{"metric":"adg_strategy_eq","goal":"max"},{"metric":"drawdown_worst_strategy_eq","goal":"min"}]`.
    Currency metrics may use `_usd` or `_btc` to select denomination.
- **write_all_results**: Controls whether every evaluated candidate is appended to `all_results.bin`. Keep `true` for full replay/analysis history; set `false` to reduce disk writes and store only the maintained Pareto/state artifacts.

### Optimizer Suites

The optimizer reuses the backtest suite configuration when `--suite [y/n]` is enabled.

- **backtest.suite_enabled**: Can be toggled for optimizer runs via `--suite [y/n]` on `passivbot optimize`.
- **backtest.reducer**: Per-metric aggregation rules applied to scenario results before feeding into `optimize.scoring` and `optimize.limits`.
- **backtest.scenarios**: Scenario dictionaries. Each one may override `coins`, `ignored_coins`, `start_date`, `end_date`, `exchanges`, `coin_sources`, and `overrides` (arbitrary config path overrides).
  `overrides` accepts either dotted config paths or nested config fragments; nested fragments are
  flattened to leaf paths, while a top-level `coin_overrides` mapping remains an atomic scenario
  replacement so scenarios may introduce coin-specific entries absent from the base config.
- **optimize.objective_scenario**: Default scoring scenario. Set it to a unique scenario label to
  score objectives from that scenario by default, or to `null` to use suite aggregation by
  default. Individual `optimize.scoring` entries may override the default with a named `scenario`
  or explicit `scenario: null`, and suite-reduced entries may set their own `reducer`.
  Individual `optimize.limits` entries independently select either a named scenario or a suite
  reduced statistic.

Use `--suite-config path/to/file.json` to layer additional scenario definitions at runtime.

### Optimization Limits

The optimizer penalizes backtests whose metric values exceed or fall short of specified thresholds. Penalties are added to the fitness score to discourage undesirable configurations but do not disqualify the config.

Any metric listed above can be used when defining limits. Currency-specific metrics use `_usd` and `_btc` suffixes where both denominations are available; BTC-denominated metrics are available even when `backtest.btc_collateral_cap = 0`. This includes the shared HSL metrics such as `hard_stop_time_in_red_pct`, `hard_stop_post_restart_retrigger_pct`, and `hard_stop_halt_to_restart_equity_loss_pct`, plus `backtest_completion_ratio` for rejecting truncated runs. HSL metrics are account-level shared metrics and therefore remain single-valued rather than being split into `_usd` and `_btc`. Each limit entry is a dictionary with:

- `metric`: canonical metric name (`drawdown_worst_btc`, `loss_profit_ratio`, `peak_recovery_hours_pnl`, etc.).
- `penalize_if`: one of `<`, `<=`, `>`, `>=`, `==`, `outside_range`, or `inside_range` (aliases like `less_than`, `greater_than`, `auto`, etc. are also accepted). Use `outside_range` to keep a metric within `[low, high]`, and `inside_range` to forbid a specific band.
- `value`: numeric threshold for `<`/`>` modes.
- `range`: two-value list `[low, high]` for the range modes.
- Optional `enabled`: set to `false` to disable a default limit without deleting it. This prevents config normalization from re-adding that metric's default limit later.
- Optional `scenario`: a named suite scenario to evaluate for this limit. Omitted or explicit
  `null` uses suite aggregation. A named scenario uses that scenario's metric value and cannot be
  combined with `reducer`; unknown labels and scenario limits outside suite optimization are rejected.
- Optional `reducer`: for suite-aggregate limits, the statistic to compare against (`min`, `max`,
  `mean`, `std`, or `median`). If omitted, Passivbot uses the metric's `backtest.reducer` rule,
  then `backtest.reducer.default`, then `mean`.

#### Format

Define limits in `optimize.limits` as a list:

```json
"limits": [
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  },
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg_btc", "penalize_if": "<", "value": 0.0005, "reducer": "mean"},
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
  --limit 'drawdown_worst_strategy_eq <= 0.5 scenario=base' \
  --limit 'backtest_completion_ratio>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'adg > 0.0008 reducer=mean'
```

CLI replacement rules:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` appends one parsed limit entry and may be repeated.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

## Configuration Internals

The [generated field reference](ai/generated/config_fields.md) lists declared serialization
types and defaults. Numeric export follows each field's type rather than the spelling of its
template default; fractional values are preserved. Field validators continue to define accepted
inputs and constraints.

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

## Retired excess allowance selector

Excess allowance is always bounded. Final v9 loading rejects the retired
`we_excess_allowance_mode` field and its flat risk spelling. Remove this selector explicitly
before converting a released-v8 config. An input using `"legacy_raw"` also requires a policy
review: bounded sizing can reduce exposure headroom, and raw sizing cannot be preserved
automatically. Re-backtest the resulting policy before use.
The same check includes coin overrides, external override files, optimizer fixed
overrides and suite scenarios.

## HSL configuration

V9 requires `config_version="v9.0.0"` and the explicit
[v8-to-v9 migration](v8_to_v9_migration.md) before using older configurations. Relabeling a
v8 file does not convert its policy. Main configs reject retired fields in root, CLI,
coin/scenario overrides and optimizer policy. There is no engine selector or restart policy.

Use `bot.long.hsl` and `bot.short.hsl` in `coin`/`position_side` modes. `portfolio` requires an
explicit `bot.global.hsl` block with `enabled`, `drawdown_ratio_threshold`, `drawdown_ratio_ema_span_minutes`,
`memory_ema_span_minutes`, `panic_close_order_type` and `cooldown_after_red_minutes`. Side policy is never silently
promoted to portfolio authority. `scale_budget_with_excess_allowance` defaults to `false`
and may vary per coin in coin scope.

Each HSL policy's slow `memory_ema_span_minutes` drives decaying currency drawdown. Its
`drawdown_ratio_ema_span_minutes` independently smooths its sampled ratio, starting from zero at a known
new episode. Both spans preserve fractional minutes and require a finite supported f32
recurrence. The [risk reference](v9_risk_model.md) specifies sampling, arithmetic,
bootstrap and malformed-input behavior; the [HSL guide](equity_hard_stop_loss.md) explains
scopes and finite cooldown.

A flatten starts finite cooldown when the latest pre-final-fill assessment **or** the
post-final-accounting assessment is RED. Once qualified, later GREEN/budget changes do not
cancel or renew the integer deadline. New actual exposure starts a new episode. Zero
cooldown allows immediate restart subject to ordinary gates. There is no `never` option;
permanent shutdown belongs to the operator or an external supervisor.

Portfolio optimizer bounds use `optimize.bounds.global.hsl`; local HSL bounds use their side
blocks. Global unstuck and loss-gate span bounds use `optimize.bounds.global.unstuck` and
`optimize.bounds.global.loss_gate`. Omitted slow-span bounds freeze configured values;
conversion never widens new dimensions implicitly. Per-coin override support does not create
per-coin optimizer genes. Schema, policies, inputs and implementation participate in saved-fitness
identity; incompatible checkpoints fail and migrated numeric seeds require fresh evaluation.

Backtests retain HSL summaries and transitions. Set `backtest.hsl_detailed_report=true`
for optional per-minute risk samples and plots; optimizer replay omits additional diagnostic
arrays. Independent ordinary account-equity metrics remain required even when HSL
reporting is disabled. See [metrics](metrics.md) and [qualification](hsl_testing.md).

Rollback requires version-matched source/runtime and the original config backup. Offline
validation and migration authorize no bot startup, deployment or account action.
