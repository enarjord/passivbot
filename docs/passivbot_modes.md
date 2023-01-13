# Passivbot Modes

Passivbot has three different ways of generating the grid of entry orders:  
Recursive Grid Mode, Static Grid Mode, Neat Grid Mode and EMAs Mode

Static and Neat grid modes are similar, building a grid with a pre specified span.

Recursive grid mode builds the grid recursively, based on expected new position after previous grid node fill

EMAs mode is not a grid bot, but waits a duration of time between entries.


## Common Parameters

- ema_span_0: float
- ema_span_1: float
	- spans are given in minutes
	- one more EMA is added in between span_0 and span_1
	- `EMA_spans = [ema_span_0, (ema_span_0 * ema_span_1)**0.5, ema_span_1]`
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


## Parameters common to Neat, Static and Recursive grid modes
- initial_eprice_ema_dist: float
	- if no pos, initial entry price is
		- `ema_band_lower * (1 - initial_eprice_ema_dist)` for long
		- `ema_band_upper * (1 + initial_eprice_ema_dist)` for short
- initial_qty_pct: float
	- `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`
- auto_unstuck_ema_dist: float
	- per uno distance from EMA band to place auto unstuck orders.
	- `auto_unstuck_bid_price = lower_EMA_band * (1 - auto_unstuck_ema_dist)`
	- `auto_unstuck_ask_price = upper_EMA_band * (1 + auto_unstuck_ema_dist)`
	- See more in `docs/auto_unstuck.md`
- auto_unstuck_wallet_exposure_threshold: float
	- if set to 0, auto unstuck is disabled
	- auto unstuck mode is triggered when `wallet_exposure >= wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)`
	- auto unstuck entry qty will bring wallet_exposure to wallet_exposure_limit
	- auto unstuck close qty will bring wallet_exposure to `wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)`
	- See more in `docs/auto_unstuck.md`


## Static Grid Mode Parameters

- eprice_exp_base
	- if 1.0 spacing between all nodes' prices is equal
	- higher than 1.0 and spacing will increase deeper in the grid
- eprice_pprice_diff
	- per uno distance from entry price to pos_price if filled
	- a node's qty is determined such that pos price after fill is equal to `entry price * (1 +/- eprice_pprice_diff)`
	- e.g. if long and pos_price is greater than 101.5 and `eprice_pprice_diff=0.015` and `entry_price==100` qty will be such that pos price after node fill is 101.5
	- eprice_pprice_diff is dynamically increased behind the scenes in some proportion to wallet_exposure lest wallet_exposure exceeds wallet_exposure_limit
- grid_span
	- per uno distance from initial entry price to last node's price
- secondary_allocation
	- per uno allocation of allocated funds to a secondary node which is independent of primary grid
	- e.g. if 0.25, primary grid gets 75% of funds, secondary node gets 25%
	- wallet_exposure_limit is always observed
- secondary_pprice_diff
	- per uno distance from pos price after last primary grid node is filled to secondary node
	- e.g. if long and pos price after primary grid's exhaustion is 40 and secondary_pprice_diff is 0.15, secondary grid node's price is `40 * (1 - 0.15) == 34`

It is called static grid mode because the grid is defined as a whole.  
If there already is a position, the grid is reverse engineered by deducing initial entry price  
and assuming partial node fills if the guessed grid is not a close match.

## Recursive Grid Mode Parameters

- ddown_factor
	- `next_reentry_qty = pos_size * ddown_factor`
	- in recursive grid mode ddown factor is static; in static grid mode ddown factor becomes dynamic
- rentry_pprice_dist
- rentry_pprice_dist_wallet_exposure_weighting
	- if set to zero, spacing between nodes will be approximately the same
	- if > zero, spacing between nodes will increase in some proportion to wallet_exposure
	- given long,
	- `next_reentry_price = pos_price * (1 - rentry_pprice_diff * modifier)`  
	- where `modifier = (1 + ratio * rentry_pprice_dist_wallet_exposure_weighting)`  
	- and where `ratio = wallet_exposure / wallet_exposure_limit`  

It is called recursive grid mode because the grid is defined recusively by computing each node as if the previous node were filled.

## Neat Grid Mode Parameters

- eqty_exp_base
	- if 1.0, qtys will increase linearly deeper in the grid
	- if > 1.0, qtys will increase exponentially deeper in the grid
- see static grid mode for the remaining parameters
- parameters secondary_allocation, secondary_pprice_diff and eprice_pprice_diff aren't used in neat grid mode

It is called neat grid mode because the grid is made in a "neater" way than in static grid mode.


## EMAs Mode Parameters

- delay_between_fills_minutes_entry/close
	- delay between entries/closes given in minutes
	- resets after full pos close
- delay_weight_entry/close
	- delay between orders is modified according to: 
	- `max(1, delay_between_fills_mins * (1 - pprice_diff * delay_weight))`
	- where pprice_diff is diff between pos price and market price
- ema_dist_lower/upper
	- offset lower/upper ema band.  See ema_span_0/ema_span_1
- qty_pct_entry/close
	- entry cost = `balance * wallet_exposure_limit * qty_pct`
- we_multiplier_entry/close
	- entry cost is modified according to:
	- `balance * wallet_exposure_limit * qty_pct * (1 + ratio * we_multiplier)`
	- where `ratio = wallet_exposure / wallet_exposure_limit`

EMAs mode uses the pos close logic common to all passivbot modes.  There is no entry grid.
It is called EMAs mode because all entry and close prices are based on EMAs.


