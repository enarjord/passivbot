# passivbot_futures
trading bot running on bybit inverse futures and binance usdt futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchanges bybit inverse and binance usdt futures

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 {exchange}.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate btc (or another coin) in bybit inverse and usdt in binance usdt futures

it is a market maker bot, making a grid of limit orders above and below price

it listens to websocket live stream of trades, and updates its orders continuously

there are two modes, static grid and dynamic grid, set by user in settings

static grid mode places entries at fixed absolute price intervals

dynamic grid mode places entries at a percentage distance from position price, modified by pos_margin / margin_limit

------------------------------------------------------------------

a backtester is included

use backtesting_notes.ipynb in jupyter notebook or jupiter-lab

to iterate multiple settings,

open backtester.py, go to def jackrabbit, adjust starting settings, n_iterations, ranges, etc

and run backtester.py:

`python3 backtester.py exchange your_user_name n_days`



------------------------------------------------------------------

settings, bybit example:

{

    "default_qty": 1.0,                   # entry quantity.
                                          # scalable entry quantity mode:
                                          # if "default_qty" is set to a negative value,
                                          # it becomes a percentage of margin_limit (which is actual account balance if margin_limit is set to -1).
                                          # the bot will calculate entry qty using the following formula:
                                          # default_qty = max(minimum_qty, round_dn(balance_in_terms_of_contracts * abs(settings["default_qty"]), qty_step))
                                          # bybit BTCUSD example:
                                          # if "default_qty"  is set to -0.06, last price is 37000 and wallet balance is 0.001 btc,
                                          # default_qty = 0.001 * 37000 * 0.06 == 2.22.  rounded down is 2.0 usd.
                                          # binance ETHUSDT example:
                                          # if "default_qty" is set to -0.07, last price was 1100 and wallet balance is 60 usdt,
                                          # default_qty = 60 / 1100 * 0.07 == 0.003818.  rounded down is 0.003 eth.
                                          
    "dynamic_grid": True,                 # bot has two modes: dynamic grid and static grid. True for dynamic mode, False for static mode.
    "grid_coefficient": 245.0,            # used in dynamic grid mode.
    "grid_spacing": 0.0026,               # used in dynamic grid mode.
                                          # next entry price is pos_price * (1 +- grid_spacing * (1 + pos_margin / margin_limit * grid_coefficient)).
                                          
    "grid_step": 116.5                    # used in static mode.  absolute price interval.
                                          
    "liq_diff_threshold": 0.02,           # if difference between liquidation price and last price is less than 2%, reduce position by 2% at a loss,
    "stop_loss_pos_reduction": 0.02,      # reduce position by 2% at a loss.
    
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always 100x leverage).
    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "max_markup": 0.0159,                 # evenly distributed between min and max markup, and whose qtys are pos_size // n_close_orders.
    
    "market_stop_loss": true,             # if true will soft stop with market orders, otherwise soft stops with limit orders at order book's higest_bid/lowest_ask
    
    "margin_limit": 0.0015                # used to limit pos size and to modify grid spacing in dynamic mode.
                                          # set margin_limit to -1 to use account balance fetched from exchange as margin limit.
                                          
    "n_close_orders": 20,                 # max n close orders.
    "n_entry_orders": 8,                  # max n entry orders.
    "symbol": "BTCUSD"                    # only one symbol at a time.

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
