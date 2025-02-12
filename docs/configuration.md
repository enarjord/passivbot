# Passivbot Parameters Explanation

Here follows an overview of the parameters found in `config/template.json`.

## Backtest Settings

- `base_dir`: Location to save backtest results.
- `compress_cache`: set to true to save disk space. Set to false to load faster.
- `end_date`: End date of backtest, e.g., 2024-06-23. Set to 'now' to use today's date as end date.
- `exchanges`: Exchanges from which to fetch 1m OHLCV data for backtesting and optimizing.
- `start_date`: Start date of backtest.
- `starting_balance`: Starting balance in USD at the beginning of backtest.
- `symbols`: Coins which were backtested for each exchange. Note: coins for backtesting are live.approved_coins minus live.ignored_coins.

## Bot Settings

### General Parameters for Long and Short

- `ema_span_0`, `ema_span_1`: 
  - Spans are given in minutes.
  - `next_EMA = prev_EMA * (1 - alpha) + new_val * alpha`
  - Where `alpha = 2 / (span + 1)`.
  - One more EMA span is added in between `ema_span_0` and `ema_span_1`:
  - `EMA_spans = [ema_span_0, (ema_span_0 * ema_span_1)**0.5, ema_span_1]`
  - These three EMAs are used to make an upper and a lower EMA band:
    - `ema_band_lower = min(emas)`
    - `ema_band_upper = max(emas)`
  - These are used for initial entries and auto unstuck closes.
- `n_positions`: Maximum number of positions to open. Set to zero to disable long/short.
- `total_wallet_exposure_limit`: Maximum exposure allowed.
  - For example, `total_wallet_exposure_limit = 0.75` means 75% of (unleveraged) wallet balance is used.
  - For example, `total_wallet_exposure_limit = 1.6` means 160% of (unleveraged) wallet balance is used.
  - Each position is given equal share of total exposure limit, i.e., `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
  - See more: `docs/risk_management.md`.
- `enforce_exposure_limit`: If true, will enforce exposure limits for each position.
  - E.g. if for any reason a position's exposure exceeds 1% of the limit, reduce the position at market price to exposure limit.
  - Useful for risk management if, for example, user withdraws balance or changes settings.

### Grid Entry Parameters

Passivbot may be configured to make a grid of entry orders, the prices and quantities of which are determined by the following parameters:

- `entry_grid_double_down_factor`:
  - Quantity of next grid entry is position size times double down factor. For example, if position size is 1.4 and `double_down_factor` is 0.9, then next entry quantity is `1.4 * 0.9 == 1.26`.
  - Also applies to trailing entries.
- `entry_grid_spacing_pct`, `entry_grid_spacing_weight`: 
  - Grid re-entry prices are determined as follows:
    - `next_reentry_price_long = pos_price * (1 - entry_grid_spacing_pct * modifier)`  
    - `next_reentry_price_short = pos_price * (1 + entry_grid_spacing_pct * modifier)`  
  - Where `modifier = (1 + ratio * entry_grid_spacing_weight)`  
  - And where `ratio = wallet_exposure / wallet_exposure_limit`  
- `entry_initial_ema_dist`: 
  - Offset from lower/upper EMA band.
  - Long initial entry/short unstuck close prices are lower EMA band minus offset.
  - Short initial entry/long unstuck close prices are upper EMA band plus offset.
  - See `ema_span_0`/`ema_span_1`.
- `entry_initial_qty_pct`: 
  - `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`

### Trailing Parameters

The same logic applies to both trailing entries and trailing closes.

- `trailing_grid_ratio`: 
  - Set trailing and grid allocations.
  - If `trailing_grid_ratio == 0.0`, grid orders only.
  - If `trailing_grid_ratio == 1.0` or `trailing_grid_ratio == -1.0`, trailing orders only.
  - If `trailing_grid_ratio > 0.0`, trailing orders first, then grid orders.
  - If `trailing_grid_ratio < 0.0`, grid orders first, then trailing orders.
    - For example, `trailing_grid_ratio = 0.3`: trailing orders until position is 30% full, then grid orders for the rest.
    - For example, `trailing_grid_ratio = -0.9`: grid orders until position is (1 - 0.9) == 10% full, then trailing orders for the rest.
    - For example, `trailing_grid_ratio = -0.12`: grid orders until position is (1 - 0.12) == 88% full, then trailing orders for the rest.
- `trailing_retracement_pct`, `trailing_threshold_pct`: 
  - There are two conditions to trigger a trailing order: 1) threshold and 2) retracement.
  - If `trailing_threshold_pct <= 0.0`, threshold condition is always triggered.
  - Otherwise, the logic is as follows, considering long positions:
    - `if highest price since position change > position price * (1 + trailing_threshold_pct)`, the first condition is met.
    - And `if lowest price since highest price < highest price since position change * (1 - trailing_retracement_pct)`, the second condition is met. Place order.
  - Passivbot tracks its own trailing prices, and does not use any special trailing order type from the exchange.
  - Whenever the position changes (add to or partially close), the trailing price tracker is reset.
  - The trailing price tracking is based on 1m OHLCVS, which update on each new whole minute.

### Grid Close Parameters

- `close_grid_markup_range`, `close_grid_min_markup`, `close_grid_qty_pct`: 
  - Take Profit (TP) prices are spread out from:
    - `pos_price * (1 + min_markup)` to `pos_price * (1 + min_markup + markup_range)` for long.
    - `pos_price * (1 - min_markup)` to `pos_price * (1 - min_markup - markup_range)` for short.
  - For example, if `long_pos_price == 100`, `min_markup = 0.01`, `markup_range = 0.02`, and `close_grid_qty_pct = 0.2`, there are at most `1 / 0.2 == 5` TP orders, and TP prices are `[101, 101.5, 102, 102.5, 103]`.
  - Quantity per order is `full pos size * close_grid_qty_pct`.
  - Note that full position size is when position is maxed out. If position is less than full, fewer than `1 / close_grid_qty_pct` may be created.
  - The TP grid is built from the top down:
    - First TP at 103 up to 20% of full position size.
    - Next TP at 102.5 from 20% to 40% of full position size.
    - Next TP at 102.0 from 40% to 60% of full position size.
    - Etc.
  - For example, if `full_pos_size = 100` and `long_pos_size == 55`, then TP orders are `[15@102.0, 20@102.5, 20@103.0]`.
  - If position is greater than full position size, the leftovers are added to the lowest TP order.
    - For example, if `long_pos_size == 130`, then TP orders are `[50@101.0, 20@101.5, 20@102.0, 20@102.5, 20@103.0]`.

### Trailing Close Parameters

- `close_trailing_grid_ratio`: See Trailing Parameters above.
- `close_trailing_qty_pct`: Close quantity is `full pos size * close_trailing_qty_pct`.
- `close_trailing_retracement_pct`: See Trailing Parameters above.
- `close_trailing_threshold_pct`: See Trailing Parameters above.

### Unstuck Parameters

If a position is stuck, the bot will use profits made on other positions to realize losses for the stuck position. If multiple positions are stuck, the stuck position whose price action distance is the lowest is selected for unstucking. 

- `unstuck_close_pct`:
  - Percentage of `full pos size * wallet_exposure_limit` to close for each unstucking order.
- `unstuck_ema_dist`:
  - Distance from EMA band to place unstucking order:
    - `long_unstuck_close_price = upper_EMA_band * (1 + unstuck_ema_dist)`
    - `short_unstuck_close_price = lower_EMA_band * (1 - unstuck_ema_dist)`
- `unstuck_loss_allowance_pct`: 
  - Weighted percentage below past peak balance to allow losses.
  - `loss_allowance = past_peak_balance * (1 - unstuck_loss_allowance_pct * total_wallet_exposure_limit)`
  - For example, if past peak balance was $10,000, `unstuck_loss_allowance_pct = 0.02`, and `total_wallet_exposure_limit = 1.5`, the bot will stop taking losses when balance reaches `$10,000 * (1 - 0.02 * 1.5) == $9,700`.
- `unstuck_threshold`:
  - If a position is bigger than a threshold, consider it stuck and activate unstucking.
  - `if wallet_exposure / wallet_exposure_limit > unstuck_threshold: unstucking enabled`
  - For example, if a position size is $500 and max allowed position size is $1000, then position is 50% full. If `unstuck_threshold == 0.45`, then unstuck the position until its size is $450.  

### Filter Parameters

Coins selected for trading are filtered by volume and noisiness. First, filter coins by volume, dropping a percentage of the lowest volume coins. Then, sort the eligible coins by noisiness and select the top noisiest coins for trading.  

- `filter_relative_volume_clip_pct`: Volume filter; disapprove the lowest relative volume coins. For example, `filter_relative_volume_clip_pct = 0.1` drops the 10% lowest volume coins. Set to zero to allow all.
- `filter_rolling_window`: Number of minutes to look into the past to compute volume and noisiness, used for dynamic coin selection in forager mode.
  - Noisiness is normalized relative range of 1m OHLCVs: `mean((high - low) / close)`.
  - In forager mode, the bot will select coins with highest noisiness for opening positions.

## Live Trading Settings

- `approved_coins`:
	- List of coins approved for trading. If empty, see live.empty_means_all_approved.
		- Backtester and optimizer are live.approved_coins minus live.ignored_coins.
	- May be given as path to external file which is read by Passivbot continuously.
	- May be split into long and short by giving a json on the form:
		- `{"long": ["COIN1", "COIN2"], "short": ["COIN2", "COIN3"]}`
- `auto_gs`: Automatically enable graceful stop for positions on disapproved coins.
  - Graceful stop means the bot will continue trading as normal, but not open a new position after the current position is fully closed.
  - If auto_gs=false, positions on disapproved coins are put on manual mode.
- `coin_flags`:
  - Specify flags for individual coins, overriding values from bot config.
  - For example, `coin_flags: {"ETH": "-sm n -lm gs", "XRP": "-lm p -lc path/to/other_config.json"}` will force short mode to normal and long mode to graceful stop for ETH; it will set long mode to panic and use other config for XRP.
  - Flags:
    - `-lm` or `-sm`: Long or short mode. Choices: [n (normal), m (manual), gs (graceful_stop), p (panic), t (take_profit_only)].
      - Normal mode: passivbot manages the position as normal.
      - Manual mode: passivbot ignores the position.
      - Graceful stop: if there is a position, passivbot will manage it; otherwise, passivbot will not make new positions.
      - Take profit only mode: passivbot will only manage closing orders.
      - Panic mode: passivbot will close the position immediately.
    - `-lw` or `-sw`: Long or short wallet exposure limit.
    - `-lev`: Leverage.
    - `-lc`: Path to live config. Load all of another config's bot parameters except `[n_positions, total_wallet_exposure_limit, unstuck_loss_allowance_pct, unstuck_close_pct]`.
- empty_means_all_approved:
	- If true, will interpret approved_coins=[] as all coins approved.
	- If false, will interpret approved_coins=[] as no coins approved.
- `execution_delay_seconds`: Wait x seconds after executing to exchange.
- `filter_by_min_effective_cost`: If true, will disallow coins where `balance * WE_limit * initial_qty_pct < min_effective_cost`.
  - For example, if exchange's effective minimum cost for a coin is $5, but bot wants to make an order of $2, disallow that coin.
- `forced_mode_long`, `forced_mode_short`: Force all coins long/short to a given mode.
  - Choices: [m (manual), gs (graceful_stop), p (panic), t (take_profit_only)].
- `ignored_coins`:
	- List of coins bot will not make positions on. If there are positions on that coin, turn on graceful stop or manual mode.
	- May be given as path to external file which is read by Passivbot continuously.
	- May be split into long and short by giving a json on the form:
		- `{"long": ["COIN1", "COIN2"], "short": ["COIN2", "COIN3"]}`
- `leverage`: Leverage set on exchange. Default is 10.
- `market_orders_allowed`: If true, allow Passivbot to place market orders when order price is very close to current market price. If false, will only place limit orders. Default is true.
- `max_n_cancellations_per_batch`: Will cancel n open orders per execution.
- `max_n_creations_per_batch`: Will create n new orders per execution.
- `max_n_restarts_per_day`: If the bot crashes for any reason, restart the bot up to n times per day before stopping completely.
- `minimum_coin_age_days`: Disallow coins younger than a given number of days.
- `ohlcvs_1m_rolling_window_days`: How many days worth of OHLCVs for the bot to keep in memory. Reduce this number if RAM consumption becomes an issue.
- `ohlcvs_1m_update_after_minutes`: How many minutes old OHLCVs for a coin may be before the bot will fetch fresh ones from the exchange. Increase this number if rate limiting becomes an issue.
- `pnls_max_lookback_days`: How far into the past to fetch PnL history.
- `price_distance_threshold`: Minimum distance to current price action required for EMA-based limit orders.
- `time_in_force`: Default is Good-Till-Cancelled.
- `user`: Fetch API key/secret from `api-keys.json`.

## Optimization Settings

### Bounds

When optimizing, parameter values are within the lower and upper bounds.

### Other Optimization Parameters

- `compress_results_file`: If true, will compress optimize output results file to save space.
- `crossover_probability`: The probability of performing crossover between two individuals in the genetic algorithm. It determines how often parents will exchange genetic information to create offspring.
- `iters`: Number of backtests per optimize session.
- `mutation_probability`: The probability of mutating an individual in the genetic algorithm. It determines how often random changes will be introduced to the population to maintain diversity.
- `n_cpus`: Number of CPU cores utilized in parallel.
- `population_size`: Size of population for genetic optimization algorithm.
- `scoring`:
  - The optimizer uses two objectives and finds the Pareto front.
  - Finally chooses the optimal candidate based on lowest Euclidean distance to the ideal point.
  - Default values are median daily gain and Sharpe ratio.
  - The script uses the NSGA-II algorithm (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.
  - The fitness function is set up to minimize both objectives (converted to negative values internally).
  - Options: adg, mdg, sharpe_ratio, sortino_ratio, omega_ratio, calmar_ratio, sterling_ratio
  - Examples: ["mdg", "sharpe_ratio"], ["adg", "sortino_ratio"], ["sortino_ratio", "omega_ratio"]

### Optimization Limits

The optimizer will penalize backtests whose metrics exceed the given values. If multiple exchanges are optimized, it will select the worst of them.

- `lower_bound_drawdown_worst`: Lowest drawdown during backtest.
- `lower_bound_equity_balance_diff_mean`: Mean of the difference between equity and balance.
- `lower_bound_loss_profit_ratio`: `abs(sum(losses)) / sum(profit)`
- `equity_balance_diff_neg_max`: greatest distance between balance and equity when equity is less than balance
- `equity_balance_diff_neg_mean`: mean distance between balance and equity when equity is less than balance
- `equity_balance_diff_pos_max`: greatest distance between balance and equity when equity is greater than balance
- `equity_balance_diff_pos_mean`: mean distance between balance and equity when equity is greater than balance
