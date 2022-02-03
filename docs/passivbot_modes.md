# Passivbot Modes

Passivbot has two different ways of generating the grid of entry orders:  
Recursive Grid Mode and Static Grid Mode


## Common Parameters

- auto_unstuck_ema_dist: float
	- per uno distance from EMA band to place auto unstuck orders.
	- See more in `docs/auto_unstuck.md`
- auto_unstuck_wallet_exposure_threshold: float
	- if set to 0, auto unstuck is disabled
	- auto unstuck mode is triggered when `wallet_exposure >= wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)`
	- See more in `docs/auto_unstuck.md`
- ema_span_0: float
- ema_span_1: float
	- spans are given in minutes
	- one more EMA is added in between span_0 and span_1
	- `EMA_spans = [ema_span_0, (ema_span_0 * ema_span_1)**0.5, ema_span_1]`
	- `ema_band_lower = min(emas)`
	- `ema_band_upper = max(emas)`
- enabled: bool
- initial_eprice_ema_dist: float
	- if no pos, initial entry price is
		- `ema_band_lower * (1 - initial_eprice_ema_dist)` for long
		- `ema_band_upper * (1 + initial_eprice_ema_dist)` for short
- initial_qty_pct: float
	- `initial_entry_cost = balance * wallet_exposure_limit * initial_qty_pct`
- markup_range: float
- min_markup: float
- n_close_orders: int (if float: int(round(x)))
	- Take Profit (TP) prices are spread out from
		- `pos_price * (1 + min_markup)` to `pos_price * (1 + min_markup + markup_range)` for long
		- `pos_price * (1 - min_markup)` to `pos_price * (1 - min_markup - markup_range)` for short
		- e.g. if `pos_price==100`, `min_markup=0.02`, `markup_range=0.03` and `n_close_orders=7`, TP prices are [102, 102.5, 103, 103.5, 104, 104.5, 105]
- wallet_exposure_limit: float
	- See `docs/risk_management.md`

## Static Grid Mode Parameters

- eprice_exp_base
	- if 1.0 spacing between all nodes is equal
	- higher than 1.0 and spacing will increase deeper in the grid
- eprice_pprice_diff
	- per uno distance from entry price to pos_price if filled
	- a node's qty is determined such that pos price after fill is equal to `entry price * (1 + eprice_pprice_diff)`
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
	- given long,
	- `next_reentry_price = pos_price * (1 - rentry_pprice_diff * modifier)`  
	- where `modifier = (1 + ratio * rentry_pprice_dist_wallet_exposure_weighting)`  
	- and where `ratio = wallet_exposure / wallet_exposure_limit`  

It is called recursive grid mode because the grid is defined recusively by precomputing each node as if the previous node were filled.

