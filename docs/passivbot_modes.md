# Passivbot Modes

Passivbot has three different ways of calculating entries:  
Recursive Grid Mode, Neat Grid Mode and Clock Mode.

Neat grid mode builds a grid with a pre-specified span.  
Recursive grid mode builds the grid recursively, based on expected new position after previous grid node fill.  
Clock mode builds no grids, but instead waits a duration of time between entries.


## Common Parameters

- ema_span_0: float
- ema_span_1: float
	- spans are given in minutes
	- `next_EMA = prev_EMA * (1 - alpha) + new_val * alpha`
	- where `alpha = 2 / (span + 1)`
	- one more EMA span is added in between span_0 and span_1:
	- `EMA_spans = [ema_span_0, (ema_span_0 * ema_span_1)**0.5, ema_span_1]`
	- these three EMAs are used to make an upper and a lower EMA band:
	- `ema_band_lower = min(emas)`
	- `ema_band_upper = max(emas)`
- enabled: bool
- min_markup: float
- markup_range: float
- n_close_orders: int (if float: int(round(x)))
	- Take Profit (TP) prices are spread out from
		- `pos_price * (1 + min_markup)` to `pos_price * (1 + min_markup + markup_range)` for long
		- `pos_price * (1 - min_markup)` to `pos_price * (1 - min_markup - markup_range)` for short
		- e.g. if `pos_price==100`, `min_markup=0.02`, `markup_range=0.03` and `n_close_orders=7`, TP prices are [102, 102.5, 103, 103.5, 104, 104.5, 105]
		- qty per order is pos size divided by n_close_orders
	- say long, if one TP ask is filled and afterwards price dips below that price level, bot recreates TP grid with reduced qty on each price level
- wallet_exposure_limit: float
	- bot limits pos size to `wallet_balance_in_contracts * wallet_exposure_limit`
	- See more in `docs/risk_management.md`
- backwards_tp: bool  
        - fills the TP grid "backwards" instead of frontwards:  
        - using TP prices from the example above, bot will build the TP grid starting with 105, then 104.g and so on, instead of starting with 102, then 102.5 and so on.


## Parameters common to Neat and Recursive grid modes
- initial_eprice_ema_dist: float
	- if no pos, initial entry price is
		- `ema_band_lower * (1 - initial_eprice_ema_dist)` for long
		- `ema_band_upper * (1 + initial_eprice_ema_dist)` for short
- initial_qty_pct: float
	- `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`
- auto_unstuck parameters: float
	- See `docs/auto_unstuck.md`


## Neat Grid Mode Parameters

- grid_span
	- per uno (0.32 == 32%) distance from initial entry price to last node's price
- eprice_exp_base
	- if 1.0, spacing between all nodes' prices is equal
	- higher than 1.0 and spacing will increase deeper in the grid
- eqty_exp_base
	- if 1.0, qtys will increase linearly deeper in the grid
	- if > 1.0, qtys will increase exponentially deeper in the grid

It is called neat grid mode because the grid is made in a "neat" way.

## Recursive Grid Mode Parameters

- ddown_factor
	- `next_reentry_qty = pos_size * ddown_factor`
	- in recursive grid mode ddown factor is static; in neat grid mode ddown factor becomes dynamic
- rentry_pprice_dist
- rentry_pprice_dist_wallet_exposure_weighting
	- if set to zero, spacing between nodes will be approximately the same
	- if > zero, spacing between nodes will increase in some proportion to wallet_exposure
	- given long,
	- `next_reentry_price = pos_price * (1 - rentry_pprice_diff * modifier)`  
	- where `modifier = (1 + ratio * rentry_pprice_dist_wallet_exposure_weighting)`  
	- and where `ratio = wallet_exposure / wallet_exposure_limit`  

It is called recursive grid mode because the grid is defined recusively by computing each node as if the previous node were filled.


## Clock Mode Parameters

- delay_between_fills_minutes_entry/close
	- delay between entries/closes given in minutes
	- entry delay resets after full pos close
- delay_weight_entry/close
	- delay between clock orders may be reduced, but not increased.
	- if pos size is zero, the timer is reset for entries, but not for closes.

	- the formula is:
	- `modified_delay = delay_between_fills * min(1, (1 - pprice_diff * delay_weight))`
	- where for bids (long entries and short closes):
	- `pprice_diff = (pos_price / market_price - 1)`
	- and for asks (short entries and long closes):
	- `pprice_diff = (market_price / pos_price - 1)`
	- this means (given delay_weights > 0):
	```
	if market_price > pprice_long (upnl is green):
	    entry_delay is unchanged and close_delay reduced
	if market_price < pprice_long (upnl is red):
	    entry_delay is reduced and close_delay is unchanged

	if market_price < pprice_short (upnl is green):
	    entry_delay is unchanged and close_delay is reduced
	if market_price > pprice_short (upnl is red):
	    entry_delay is reduced and close_delay is unchanged
	```

- ema_dist_entry/close
	- offset from lower/upper ema band.  
	- long_entry/short_close price is lower ema band minus offset  
	- short_entry/long_close price is upper ema band plus offset  
	- `clock_bid_price = min(emas) * (1 - ema_dist_lower)`  
	- `clock_ask_price = max(emas) * (1 + ema_dist_upper)`  
	- See ema_span_0/ema_span_1
- qty_pct_entry/close  
	- basic formula is `entry_cost = balance * wallet_exposure_limit * qty_pct`  
- we_multiplier_entry/close
	- similar in function to Recursive Grid mode's ddown_factor
	- entry cost is modified according to:
	- `entry_cost = balance * wallet_exposure_limit * qty_pct * (1 + ratio * we_multiplier)`
	- where `ratio = wallet_exposure / wallet_exposure_limit`

Clock mode uses the take-profit grid common to the other passivbot modes.  
There is no entry grid and no separate auto unstucking mechanism for clock mode.  
It is called Clock Mode because entries and closes are on a timer.


