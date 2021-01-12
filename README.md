# passivbot_futures
trading bot running on bybit inverse futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchange bybit, binance support maybe added in future

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 {exchange}.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate btc in bybit inverse

it is a market maker bot, making a grid of limit orders above and below price

it listens to websocket live stream of trades, and updates its orders continuously

------------------------------------------------------------------

a backtester is included

use backtesting_notes.ipynb in jupyter notebook or jupiter-lab

to test multiple settings,

open backtester.py, go to def jackrabbit, adjust starting settings, n_iterations, ranges, etc

and run backtester.py:

`python3 backtester.py exchange your_user_name n_days`



------------------------------------------------------------------

settings, bybit example:

{

    "dynamic_grid": True,                 # bot has two modes: dynamic grid and static grid.
    "grid_coefficient": 160.0,            # used in dynamic grid mode.
    "grid_spacing": 0.003,                # used in dynamic grid mode.
                                          # next entry price is pos_price * (1 +- grid_spacing * (1 + pos_margin / account_equity * grid_coefficient))
    "liq_diff_threshold": 0.02,           # if difference between liquidation price and last price is less than 2%, reduce position by 2% at a loss
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always 100x leverage)
    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "max_markup": 0.0159,                 # evenly distributed between min and max markup, and whose qtys are pos_size // n_close_orders
                                          
    "n_close_orders": 20,                 # max n close orders
    "n_entry_orders": 10,                 # max n entry orders
    "default_qty": 1.0,                   # entry quantity
    "stop_loss_pos_reduction": 0.02,      # if difference between liquidation price and last price is less than 2%, reduce position by 2% at a loss
    "symbol": "BTCUSD"                    # only one symbol at a time

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
