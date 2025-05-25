# Passivbot Parameters Explanation

This document provides an overview of the parameters found in `config/template.json`.

## Backtest Settings

- **base_dir**: Location to save backtest results.
- **compress_cache**: Set to `true` to save disk space. Set to `false` for faster loading.
- **end_date**: End date of backtest, e.g., `2024-06-23`. Set to `'now'` to use today's date as the end date.
- **exchanges**: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing. Options: `[binance, bybit, gateio, bitget]`.
- **start_date**: Start date of backtest.
- **starting_balance**: Starting balance in USD at the beginning of the backtest.
- **use_btc_collateral**: `true`/`false`. Set to `true` to backtest with BTC as collateral, simulating starting with 100% BTC and buying BTC with all USD profits, but not selling BTC when taking losses (instead go into USD debt).
  - Example: Given BTC/USD price of `$100,000`, if BTC balance is `1.0` and backtester makes `$10` profit, BTC balance becomes `1.0001` and USD balance is `0`. If backtester loses `$20`, BTC balance remains `1.0001` and USD balance becomes `-20`. If backtester then makes `$15` profit, USD debt is paid off first: BTC balance remains `1.0001`, USD balance becomes `-5`. If the backtester then makes `$10` profit: BTC balance becomes `1.00015`, USD balance is `0`.

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
- **entry_grid_spacing_pct**, **entry_grid_spacing_weight**:
  - Grid re-entry prices are determined as follows:
    - `next_reentry_price_long = pos_price * (1 - entry_grid_spacing_pct * modifier)`
    - `next_reentry_price_short = pos_price * (1 + entry_grid_spacing_pct * modifier)`
  - Where `modifier = (1 + ratio * entry_grid_spacing_weight)` and `ratio = wallet_exposure / wallet_exposure_limit`.
- **entry_initial_ema_dist**:
  - Offset from lower/upper EMA band.
  - Long initial entry/short unstuck close prices are lower EMA band minus offset.
  - Short initial entry/long unstuck close prices are upper EMA band plus offset.
  - See `ema_span_0`/`ema_span_1`.
- **entry_initial_qty_pct**:
  - `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`

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

Coins selected for trading are filtered by volume and noisiness. First, filter coins by volume, dropping a percentage of the lowest volume coins. Then, sort eligible coins by noisiness and select the top noisiest coins for trading.

- **filter_volume_drop_pct**: Volume filter. Disapproves the lowest relative volume coins.
  - Example: `filter_volume_drop_pct = 0.1` drops the 10% lowest volume coins. Set to `0` to allow all.
- **filter_noisiness_rolling_window/filter_volume_rolling_window**: Number of minutes to look into the past to compute volume and noisiness, used for dynamic coin selection in forager mode.
  - Noisiness is normalized relative range of 1m OHLCVs: `mean((high - low) / close)`.
  - In forager mode, the bot selects coins with the highest noisiness for opening positions.

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
- **coin_flags**:
  - Specify flags for individual coins, overriding values from bot config.
  - Example: `coin_flags: {"ETH": "-sm n -lm gs", "XRP": "-lm p -lc path/to/other_config.json"}` forces short mode to normal and long mode to graceful stop for ETH; sets long mode to panic and uses another config for XRP.
  - Flags:
    - `-lm` or `-sm`: Long or short mode. Choices: `[n (normal), m (manual), gs (graceful_stop), p (panic), t (take_profit_only)]`.
      - **Normal mode**: Passivbot manages the position as normal.
      - **Manual mode**: Passivbot ignores the position.
      - **Graceful stop**: If there is a position, Passivbot manages it; otherwise, no new positions are opened.
      - **Take profit only mode**: Passivbot only manages closing orders.
      - **Panic mode**: Passivbot closes the position immediately.
    - `-lw` or `-sw`: Long or short wallet exposure limit.
    - `-lev`: Leverage.
    - `-lc`: Path to live config. Loads all bot parameters from another config except `[n_positions, total_wallet_exposure_limit, unstuck_loss_allowance_pct, unstuck_close_pct, filter_noisiness_rolling_window, filter_volume_rolling_window, filter_volume_drop_pct]`.
- **empty_means_all_approved**:
  - If `true`, `approved_coins=[]` means all coins are approved.
  - If `false`, `approved_coins=[]` means no coins are approved.
- **execution_delay_seconds**: Wait `x` seconds after executing to exchange.
- **filter_by_min_effective_cost**: If `true`, disallows coins where `balance * WE_limit * initial_qty_pct < min_effective_cost`.
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
- **mimic_backtest_1m_delay**: If `true`, the bot will only update and evaluate open orders once per full minute, synchronized to the clock (e.g., 12:01:00, 12:02:00, etc.). This mimics the backtester's timestep logic and avoids intraminute updates. Useful for achieving higher fidelity between backtest and live performance.
- **minimum_coin_age_days**: Disallows coins younger than a given number of days.
- **ohlcvs_1m_rolling_window_days**: How many days of OHLCVs the bot keeps in memory. Reduce if RAM consumption is an issue.
- **ohlcvs_1m_update_after_minutes**: How many minutes old OHLCVs for a coin may be before fetching fresh ones from the exchange. Increase if rate limiting is an issue.
- **pnls_max_lookback_days**: How far into the past to fetch PnL history.
- **price_distance_threshold**: Minimum distance to current price action required for EMA-based limit orders.
- **time_in_force**: Default is Good-Till-Cancelled.
- **user**: Fetch API key/secret from `api-keys.json`.

## Optimization Settings

### Bounds

When optimizing, parameter values are within the lower and upper bounds.

### Other Optimization Parameters

- **compress_results_file**: If `true`, compresses optimize output results file to save space.
- **enable_overrides**: List of custom optimizer overrides to enable. Use `optimizer_overrides.py` for overrides. Defaults to none.
- **crossover_probability**: Probability of performing crossover between two individuals in the genetic algorithm. Determines how often parents exchange genetic information to create offspring.
- **iters**: Number of backtests per optimize session.
- **mutation_probability**: Probability of mutating an individual in the genetic algorithm. Determines how often random changes are introduced to maintain diversity.
- **n_cpus**: Number of CPU cores utilized in parallel.
- **population_size**: Size of population for genetic optimization algorithm.
- **scoring**:
  - The optimizer uses two objectives and finds the Pareto front.
  - Chooses the optimal candidate based on the lowest Euclidean distance to the ideal point.
  - Default values are median daily gain and Sharpe ratio.
  - Uses the NSGA-II algorithm (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.
  - The fitness function minimizes both objectives (converted to negative values internally).
  - Full list of options: `[adg, adg_w, calmar_ratio, calmar_ratio_w, drawdown_worst, drawdown_worst_mean_1pct, equity_balance_diff_neg_max, equity_balance_diff_neg_mean, equity_balance_diff_pos_max, equity_balance_diff_pos_mean, expected_shortfall_1pct, gain, loss_profit_ratio, loss_profit_ratio_w, mdg, mdg_w, omega_ratio, omega_ratio_w, position_held_hours_max, position_held_hours_mean, position_held_hours_median, position_unchanged_hours_max, positions_held_per_day, sharpe_ratio, sharpe_ratio_w, sortino_ratio, sortino_ratio_w, sterling_ratio, sterling_ratio_w]`
  - Suffix `_w` indicates mean across 10 temporal subsets (whole, last_half, last_third, ..., last_tenth) to weigh recent data more heavily.
  - Examples: `["mdg", "sharpe_ratio", "loss_profit_ratio"]`, `["adg", "sortino_ratio", "drawdown_worst"]`, `["sortino_ratio", "omega_ratio", "adg_w", "position_unchanged_hours_max"]`
    - Note: if config.backtest.use_btc_collateral=True, add prefix "btc_" to use btc denominated metrics, e.g. btc_adg or btc_drawdown_worst.

### Optimization Limits

The optimizer penalizes backtests whose metric values exceed or fall short of specified thresholds. Penalties are added to the fitness score to discourage undesirable configurations but do not disqualify the config.

#### Format

Limits can be set in the config file under `optimize.limits` or passed via CLI using `--limits`.

##### Config Example

```json
"limits": {
  "penalize_if_greater_than_drawdown_worst": 0.3,
  "penalize_if_lower_than_adg": 0.001
}
```