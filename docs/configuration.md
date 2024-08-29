# Passivbot Parameters Explanation

Here follows an overview of the parameters found in `config/template.json`

## Backtest Settings
- `base_dir`: location to save backtest results
- `end_date`: end date of backtest, e.g. 2024-06-23. Set to 'now' to use today's date as end date
- `exchange`: exchange from which to fetch 1m ohlcv data. Default is Binance
- `start_date`: start date of backtest
- `starting_balance`: starting balance in USD at beginning of backtest
- `symbols`: coins to backtest. If left empty, will use all exchange's coins.

## Bot Settings
### General Parameters for Long and Short
- `ema_span_0`, `ema_span_1`: 
	- spans are given in minutes
	- `next_EMA = prev_EMA * (1 - alpha) + new_val * alpha`
	- where `alpha = 2 / (span + 1)`
	- one more EMA span is added in between span_0 and span_1:
	- `EMA_spans = [ema_span_0, (ema_span_0 * ema_span_1)**0.5, ema_span_1]`
	- these three EMAs are used to make an upper and a lower EMA band:
	- `ema_band_lower = min(emas)`
	- `ema_band_upper = max(emas)`
	- which are used for initial entries and auto unstuck closes
- `n_positions`: max number of positions to open. Set to zero to disable long/short
- `total_wallet_exposure_limit`: maximum exposure allowed.
	- E.g. total_wallet_exposure_limit = 0.75 means 75% of (unleveraged) wallet balance is used.
	- E.g. total_wallet_exposure_limit = 1.6 means 160% of (unleveraged) wallet balance is used.
	- Each position is given equal share of total exposure limit, i.e. `wallet_exposure_limit = total_wallet_exposure_limit / n_positions`.
	- See more: docs/risk_management.md

### Grid Entry Parameters
Passivbot may be configured to make a grid of entry orders, the prices and quantities of which are determined by the following parameters:
- `entry_grid_double_down_factor`:
	- quantity of next grid entry is position size times double down factor. E.g. if position size is 1.4 and double_down_factor is 0.9, then next entry quantity is `1.4 * 0.9 == 1.26`.
	- also applies to trailing entries.
- `entry_grid_spacing_pct`, `entry_grid_spacing_weight`: 
	- grid re-entry prices are determined as follows:
	- `next_reentry_price_long = pos_price * (1 - entry_grid_spacing_pct * modifier)`  
	- `next_reentry_price_short = pos_price * (1 + entry_grid_spacing_pct * modifier)`  
	- where `modifier = (1 + ratio * entry_grid_spacing_weight)`  
	- and where `ratio = wallet_exposure / wallet_exposure_limit`  
- `entry_initial_ema_dist`: 
	- offset from lower/upper ema band.  
	- long_initial_entry/short_unstuck_close prices are lower ema band minus offset  
	- short_initial_entry/long_unstuck_close prices are upper ema band plus offset  
	- See ema_span_0/ema_span_1
- `entry_initial_qty_pct`: 
	- `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`

### Trailing Parameters

Same logic applies to both trailing entries and trailing closes.
- `trailing_grid_ratio`: 
	- set trailing and grid allocations.
	- if `trailing_grid_ratio==0.0`, grid orders only.
	- if `trailing_grid_ratio==1.0 or trailing_grid_ratio==-1.0`, trailing orders only.
	- if `trailing_grid_ratio>0.0`, trailing orders first, then grid orders.
	- if `trailing_grid_ratio<0.0`, grid orders first, then trailing orders.
	- e.g. `trailing_grid_ratio = 0.3`: trailing orders until position is 30% full, then grid orders for the rest.
	- e.g. `trailing_grid_ratio = -0.9`: grid orders until position is (1 - 0.9) == 10% full, then trailing orders for the rest.
	- e.g. `trailing_grid_ratio = -0.12`: grid orders until position is (1 - 0.12) == 88% full, then trailing orders for the rest.
- `trailing_retracement_pct`, `trailing_threshold_pct`: 
	- there are two conditions to trigger a trailing order: 1) threshold and 2) retracement.
	- if `trailing_threshold_pct <= 0.0`, threshold condition is always triggered.
	- otherwise, the logic is as follows, considering long positions:
	- `if highest price since position open > position price * (1 + trailing_threshold_pct)`: 1st condition is met
	- and `if lowest price since highest price < highest price since position open * (1 - trailing_retracement_pct)`: 2nd condition is met. Make order.

### Grid Close Parameters
- `close_grid_markup_range`, `close_grid_min_markup`, `close_grid_qty_pct`: 
	- Take Profit (TP) prices are spread out from
		- `pos_price * (1 + min_markup)` to `pos_price * (1 + min_markup + markup_range)` for long
		- `pos_price * (1 - min_markup)` to `pos_price * (1 - min_markup - markup_range)` for short
		- e.g. if `pos_price==100`, `min_markup=0.01`, `markup_range=0.02` and `close_grid_qty_pct=0.2`, TP prices are [101, 101.5, 102, 102.5, 103]
		- qty per order is `full pos size * close_grid_qty_pct`
		- the TP grid is built from the top down:
			- first TP at 103 up to 20% of full pos size,
			- next TP at 102.5 from 20% to 40% of full pos size,
			- next TP at 102.0 from 40% to 60% of full pos size,
			- etc.

### Trailing Close Parameters

- `close_trailing_grid_ratio`: see Trailing Parameters above
- `close_trailing_qty_pct`: close qty is `full pos size * close_trailing_qty_pct`
- `close_trailing_retracement_pct`: see Trailing Parameters above
- `close_trailing_threshold_pct`: see Trailing Parameters above

### Unstuck Parameters
- `unstuck_close_pct`: 
- `unstuck_ema_dist`: 
- `unstuck_loss_allowance_pct`: 
- `unstuck_threshold`: 

## Live Trading Settings
- `approved_coins`: 
- `auto_gs`: 
- `coin_flags`: 
- `execution_delay_seconds`: 
- `filter_by_min_effective_cost`: 
- `forced_mode_long`: 
- `forced_mode_short`: 
- `ignored_coins`: 
- `leverage`: 
- `max_n_cancellations_per_batch`: 
- `max_n_creations_per_batch`: 
- `minimum_market_age_days`: 
- `noisiness_rolling_mean_window_size`: 
- `pnls_max_lookback_days`: 
- `price_distance_threshold`: 
- `relative_volume_filter_clip_pct`: 
- `time_in_force`: 
- `user`: 

## Optimization Settings
### Bounds
(Note: Bounds are specified for both long and short parameters. Fill in the explanation once, as they apply to both.)

- `close_grid_markup_range`: 
- `close_grid_min_markup`: 
- `close_grid_qty_pct`: 
- `close_trailing_grid_ratio`: 
- `close_trailing_qty_pct`: 
- `close_trailing_retracement_pct`: 
- `close_trailing_threshold_pct`: 
- `ema_span_0`: 
- `ema_span_1`: 
- `entry_grid_double_down_factor`: 
- `entry_grid_spacing_pct`: 
- `entry_grid_spacing_weight`: 
- `entry_initial_ema_dist`: 
- `entry_initial_qty_pct`: 
- `entry_trailing_grid_ratio`: 
- `entry_trailing_retracement_pct`: 
- `entry_trailing_threshold_pct`: 
- `n_positions`: 
- `total_wallet_exposure_limit`: 
- `unstuck_close_pct`: 
- `unstuck_ema_dist`: 
- `unstuck_loss_allowance_pct`: 
- `unstuck_threshold`: 

### Other Optimization Parameters
- `crossover_probability`: 
- `iters`: 
- `mutation_probability`: 
- `n_cpus`: 
- `population_size`: 
- `scoring`: 

### Optimization Limits
- `lower_bound_drawdown_worst`: 
- `lower_bound_equity_balance_diff_mean`: 
- `lower_bound_loss_profit_ratio`: