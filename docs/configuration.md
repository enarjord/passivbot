# Passivbot Parameters Explanation

This document provides an overview of the parameters found in `config/template.json`.

## Backtest Settings

- **base_dir**: Location to save backtest results.
- **compress_cache**: Set to `true` to save disk space. Set to `false` for faster loading.
- **end_date**: End date of backtest, e.g., `2024-06-23`. Set to `'now'` to use today's date as the end date.
- **exchanges**: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing. The template ships with `['binance', 'bybit']`; additional exchanges can be wired up manually if you maintain your own archives.
- **combine_ohlcvs**: When `true`, build a single “combined” dataset by taking the best-quality feed for each coin across all configured exchanges. When `false`, the backtester/optimizer runs each exchange independently.
- **coin_sources**: Optional mapping of `coin -> exchange` used to override the automatic selection performed when `combine_ohlcvs` is `true`. Scenarios may add more overrides; conflicting assignments raise an error.
- **start_date**: Start date of backtest.
- **starting_balance**: Starting balance in USD at the beginning of the backtest.
- **filter_by_min_effective_cost**: When `true`, skip coins whose projected initial entry
  (balance × wallet_exposure_limit × entry_initial_qty_pct, including WE excess allowance)
  would fall below the exchange’s effective minimum cost.
- **balance_sample_divider**: Minutes per bucket when sampling balances/equity for
  `balance_and_equity.csv` and related plots. `1` keeps full per-minute resolution; higher values
  thin out the series (e.g., `15` stores one point every 15 minutes) to reduce file sizes.
- **btc_collateral_cap**: Target (and ceiling) share of account equity to hold in BTC collateral. `0` keeps the account fully in USD; `1.0` mirrors the legacy 100% BTC mode; values `>1` allow leveraged BTC collateral, accepting negative USD balances.
- **btc_collateral_ltv_cap**: Optional loan-to-value ceiling (`USD debt ÷ equity`) enforced when topping up BTC. Leave `null` (default) to allow unlimited debt, or set to a float (e.g., `0.6`) to stop buying BTC once leverage exceeds that threshold.
### Suite Scenarios

- **backtest.suite.enabled**: Master switch for suite runs (`--suite [y/n]` overrides it at runtime).
- **backtest.suite.include_base_scenario** / **base_label**: Optionally prepend a scenario that mirrors the base config.
- **backtest.suite.aggregate**: Dict of metric-specific aggregation modes (default `mean`). Keys fall back to the `default` entry if unspecified.
- **backtest.suite.scenarios**: List of scenario dicts. Supported per-scenario keys:
  - `label`: Directory name under `backtests/suite_runs/<timestamp>/`.
  - `start_date`, `end_date`: Override the global date window.
  - `coins`, `ignored_coins`: Restrict or skip symbols.
  - `exchanges`: Limit which exchanges can contribute data to this scenario.
  - `coin_sources`: Scenario-specific overrides for `coin_sources`.

Refer to `configs/examples/suite_example.json` for a practical template.

## Logging

- **level**: Controls global verbosity for Passivbot and tooling.
  - Accepted values: `0` (warnings), `1` (info), `2` (debug), `3` (trace).
  - The CLI flag `--debug-level`/`--log-level` on `src/passivbot.py` and `src/backtest.py` overrides the configured value for a single run.
  - Components such as the CandlestickManager inherit this level, so EMA warm-up and candle maintenance logs follow the same verbosity.
- **memory_snapshot_interval_minutes**: Interval between `_log_memory_snapshot` telemetry entries (RSS, cache footprint, asyncio task counts). Default `30`; lower values surface leaks sooner, higher values reduce noise.
- **volume_refresh_info_threshold_seconds**: Minimum duration a bulk volume-EMA refresh must take before it is promoted to an INFO log. Runs that finish faster emit only DEBUG output (when debug logging is enabled). Set `0` to keep the previous always-INFO behaviour.

## Bot Settings

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
  - Each position is given an equal share: `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
  - See more: `docs/risk_management.md`.
- **enforce_exposure_limit**: If `true`, enforces exposure limits for each position.
  - Example: If a position's exposure exceeds 1% of the limit, reduce the position at market price to the exposure limit.
  - Useful for risk management, e.g., if the user withdraws balance or changes settings.

### Grid Entry Parameters

Passivbot can be configured to create a grid of entry orders, with prices and quantities determined by the following parameters:

- **entry_grid_double_down_factor**:
  - Quantity of the next grid entry is position size times the double down factor.
  - Example: If position size is `1.4` and `double_down_factor` is `0.9`, then the next entry quantity is `1.4 * 0.9 = 1.26`.
  - Also applies to trailing entries.
- **entry_grid_spacing_pct**, **entry_grid_spacing_we_weight** *(formerly `entry_grid_spacing_weight`)*:
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

Coins selected for trading are filtered by volume and log range. First, filter coins by volume, dropping a percentage of the lowest volume coins. Then, sort eligible coins by log range and select the most volatile coins for trading.

- **filter_volume_drop_pct**: Volume filter. Disapproves the lowest relative volume coins.
  - Example: `filter_volume_drop_pct = 0.1` drops the 10% lowest volume coins. Set to `0` to allow all.
- **filter_volatility_ema_span / filter_volume_ema_span**: Number of minutes to look into the past to compute the volatility (log-range) and volume EMAs used for dynamic coin selection in forager mode.
- **filter_volatility_drop_pct**: Volatility clip. Drops the highest-volatility fraction after volume filtering. Example: `0.2` drops the top 20% most volatile coins, forcing the selector to choose among the calmer 80%.
  - Log range is computed from 1m OHLCVs as `mean(ln(high / low))`.
  - In forager mode, the bot selects coins with the highest log-range values for opening positions.

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
  - List of coins approved for trading. If empty, see `live.empty_means_all_approved`.
    - Backtester and optimizer use `live.approved_coins` minus `live.ignored_coins`.
  - May be given as a path to an external file, read continuously by Passivbot.
  - May be split into long and short:
    - Example: `{"long": ["COIN1", "COIN2"], "short": ["COIN2", "COIN3"]}`
- **auto_gs**: Automatically enable graceful stop for positions on disapproved coins.
  - Graceful stop: The bot continues trading as normal but does not open a new position after the current position is fully closed.
  - If `auto_gs=false`, positions on disapproved coins are put on manual mode.
- **empty_means_all_approved**:
  - If `true`, `approved_coins=[]` means all coins are approved.
  - If `false`, `approved_coins=[]` means no coins are approved.
- **execution_delay_seconds**: Wait `x` seconds after executing to exchange.
- **max_memory_candles_per_symbol**: Maximum number of 1m candles retained in RAM per symbol. Older entries are trimmed once this cap is exceeded. Default (`20_000`) balances memory footprint with trailing-history visibility.
- **max_disk_candles_per_symbol_per_tf**: Maximum number of candles persisted on disk per symbol and timeframe. Oldest shards are pruned once the limit is hit (default `2_000_000`).
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
- **market_orders_allowed**: If `true`, allows Passivbot to place market orders when the order price is very close to the current market price. If `false`, only places limit orders. Default is `true`.
- **max_n_cancellations_per_batch**: Cancels `n` open orders per execution.
- **max_n_creations_per_batch**: Creates `n` new orders per execution.
- **max_n_restarts_per_day**: If the bot crashes, restart up to `n` times per day before stopping completely.
- **minimum_coin_age_days**: Disallows coins younger than a given number of days.
- **recv_window_ms**: Millisecond tolerance for authenticated REST calls (default `5000`). Increase if your exchange intermittently rejects requests with `invalid request ... recv_window` errors due to clock drift.
- Candlestick management is handled by the CandlestickManager with on-disk caching and TTL-based refresh. Legacy settings `ohlcvs_1m_rolling_window_days` and `ohlcvs_1m_update_after_minutes` are no longer used. Use `inactive_coin_candle_ttl_minutes` to control how long 1m candles for inactive symbols are kept in RAM before being refreshed.
- **pnls_max_lookback_days**: How far into the past to fetch PnL history.
- **price_distance_threshold**: Minimum distance to current price action required for EMA-based limit orders.
- **risk_wel_enforcer_threshold**: Per-symbol multiplier that triggers the WEL enforcer. When a position’s exposure exceeds `wallet_exposure_limit * (1 + risk_we_excess_allowance_pct) * risk_wel_enforcer_threshold` the bot emits a reduce-only order to bring it back under control. Set <1.0 for continual trimming, `1.0` for a hard cap, or ≤0 to disable.
- **risk_twel_enforcer_threshold**: Fraction of the configured `total_wallet_exposure_limit` that triggers the TWEL enforcer. When aggregate exposure exceeds this threshold the bot queues reduction orders instead of new entries. Set >1.0 to allow a grace margin, `1.0` for strict enforcement, or ≤0 to disable.
- **risk_we_excess_allowance_pct**: Per-symbol allowance above the configured wallet exposure limit that the enforcer tolerates before trimming. Useful for smoothing reductions; leave at `0.0` for a hard cap.
- **max_warmup_minutes**: Hard ceiling applied to the historical warm-up window for both backtests and live warm-ups. Use `0` to disable the cap; otherwise values above `0` clamp the per-symbol warmup calculated from EMA spans.
- **warmup_ratio**: Multiplier applied to the longest EMA or log-range span (in minutes) across long/short settings to decide how much 1m history to prefetch before trading. A value of `0.2`, for example, warmups ~20% of the deepest lookback, capped by `max_warmup_minutes`.
- **warmup_minutes**: Per-coin warm-up window (in minutes) derived from `warmup_ratio`, indicator spans, and the optional `max_warmup_minutes` ceiling. This value is used by the backtester and CandlestickManager to skip the earliest candles until indicators are fully primed; adjust `warmup_ratio` or the spans themselves to change it.
- **time_in_force**: Default is Good-Till-Cancelled.
- **user**: Fetch API key/secret from `api-keys.json`.

## Optimization Settings

### Bounds

When optimizing, parameter values are within the lower and upper bounds.

### Other Optimization Parameters

- **compress_results_file**: If `true`, compresses optimize output results file to save space.
- **enable_overrides**: List of constraint overrides applied during optimization to enforce specific parameter relationships. The optimizer evaluator checks these conditions and apply the overrides before running each backtest (defaults to none):
  - **"lossless_close_trailing"**: Ensures trailing stops are profitable by enforcing `close_trailing_threshold_pct` > `close_trailing_retracement_pct`. This prevents the retracement from triggering before reaching the minimum profit threshold.
  - **"forward_tp_grid"**: Creates an ascending take-profit grid where `close_grid_markup_start` < `close_grid_markup_end`
  - **"backward_tp_grid"**: Creates a descending take-profit grid where `close_grid_markup_start` > `close_grid_markup_end`.
- **crossover_probability**: Probability of performing crossover between two individuals in the genetic algorithm. Determines how often parents exchange genetic information to create offspring.
- **crossover_eta**: Crowding factor (η) for simulated-binary crossover. Lower values (<20) allow offspring to move farther away from their parents; higher values keep them closer. Default is `20.0`.
- **iters**: Number of backtests per optimize session.
- **mutation_probability**: Probability of mutating an individual in the genetic algorithm. Determines how often random changes are introduced to maintain diversity.
- **mutation_eta**: Crowding factor (η) for polynomial mutation. Smaller values (<20) produce heavier-tailed steps that explore more aggressively, while larger values confine mutations near the current value. Default is `20.0`.
- **mutation_indpb**: Probability that each attribute mutates when a mutation is triggered. Set to `0` (default) to auto-scale to `1 / number_of_parameters`, or supply an explicit probability between `0` and `1`.
- **n_cpus**: Number of CPU cores utilized in parallel.
- **offspring_multiplier**: Multiplier applied to `population_size` to determine how many offspring (`λ`) are produced each generation in the μ+λ evolution strategy. Values >1.0 increase exploration by sampling more children per generation. Default is `1.0`.
- **population_size**: Size of population for genetic optimization algorithm.
- **scoring**:
  - The optimizer uses two objectives and finds the Pareto front.
  - Chooses the optimal candidate based on the lowest Euclidean distance to the ideal point.
  - Default values are median daily gain and Sharpe ratio.
  - Uses the NSGA-II algorithm (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.
  - The fitness function minimizes both objectives (converted to negative values internally).
  - Full list of options: `[adg, adg_w, calmar_ratio, calmar_ratio_w, drawdown_worst, drawdown_worst_mean_1pct, equity_balance_diff_neg_max, equity_balance_diff_neg_mean, equity_balance_diff_pos_max, equity_balance_diff_pos_mean, expected_shortfall_1pct, gain, loss_profit_ratio, loss_profit_ratio_w, mdg, mdg_w, omega_ratio, omega_ratio_w, peak_recovery_hours_equity, peak_recovery_hours_pnl, position_held_hours_max, position_held_hours_mean, position_held_hours_median, position_unchanged_hours_max, positions_held_per_day, sharpe_ratio, sharpe_ratio_w, sortino_ratio, sortino_ratio_w, sterling_ratio, sterling_ratio_w]`
  - Suffix `_w` indicates mean across 10 temporal subsets (whole, last_half, last_third, ..., last_tenth) to weigh recent data more heavily.
  - Examples: `["mdg", "sharpe_ratio", "loss_profit_ratio"]`, `["adg", "sortino_ratio", "drawdown_worst"]`, `["sortino_ratio", "omega_ratio", "adg_w", "position_unchanged_hours_max"]`
    - Note: metrics may be suffixed with `_usd` or `_btc` to select denomination. If `config.backtest.btc_collateral_cap` is `0`, BTC values still represent the USD equity translated into BTC terms.

### Optimizer Suites

- **optimize.suite.enabled**: Evaluate every candidate across the configured scenarios. Override via `--suite [y/n]` on `src/optimize.py`.
- **optimize.suite.include_base_scenario** / **base_label**: Same semantics as the backtest suite.
- **optimize.suite.aggregate**: Per-metric aggregation rules applied to the scenario results before scoring.
- **optimize.suite.scenarios**: Scenario dictionaries (same keys as `backtest.suite.scenarios`). Each one may override `coins`, `ignored_coins`, `start_date`, `end_date`, `exchanges`, and `coin_sources`.

The optimizer automatically uploads all scenario slices into shared memory so the extra evaluations add minimal overhead.

### Optimization Limits

The optimizer penalizes backtests whose metric values exceed or fall short of specified thresholds. Penalties are added to the fitness score to discourage undesirable configurations but do not disqualify the config.

Any metric listed above (and its `btc_` prefixed counterpart when `backtest.use_btc_collateral=True`) can be used when defining limits. Each limit entry is a dictionary with:

- `metric`: canonical metric name (`drawdown_worst_btc`, `loss_profit_ratio`, `peak_recovery_hours_pnl`, etc.).
- `penalize_if`: one of `<`, `>`, `outside_range`, or `inside_range` (aliases like `less_than`, `greater_than`, `auto`, etc. are also accepted). Use `outside_range` to keep a metric within `[low, high]`, and `inside_range` to forbid a specific band.
- `value`: numeric threshold for `<`/`>` modes.
- `range`: two-value list `[low, high]` for the range modes.
- Optional `stat`: when you want to compare against a specific statistic (`min`, `max`, `mean`, `std`). Defaults mirror the legacy behaviour (`>` checks use `_max`, `<` checks use `_min`, range checks use `_mean`).

#### Format

Define limits in `optimize.limits` as a list:

```json
"limits": [
  {"metric": "drawdown_worst_btc", "penalize_if": ">", "value": 0.3},
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg_btc", "penalize_if": "<", "value": 0.0005, "stat": "mean"}
]
```

For quick CLI overrides you can pass the JSON/HJSON string directly:

```
python3 src/optimize.py --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]'
```

The legacy syntax (`--penalize_if_greater_than_*`) is still accepted for backwards compatibility; it is normalized into the list form at runtime.

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
