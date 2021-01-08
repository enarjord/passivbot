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

supports exchange bybit and binance

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 {exchange}.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate btc in bybit inverse, usdt in binance usdt futures

it is a market maker bot, making a grid of limit orders above and below price

it listens to websocket live stream of trades, and updates its orders continuously

------------------------------------------------------------------

use backtester for research

------------------------------------------------------------------

settings, bybit example:


{
        
        'default_qty': 1.0                  # entry qty
        'grid_step': 25.0                   # grid price spacing
        'leverage': 100.0,                  # leverage (irrelevant in bybit because cross mode in is always 100x leverage)
        'margin_limit': 0.001,              # limits the bot's max allowed pos_size.  set it lower than actual account balance
        'min_markup': 0.0005,               #
        'max_markup': 0.01,                 #
        'n_close_orders': 10,               # max n close orders
        'n_entry_orders': 10,               # max n entry orders
        'symbol': 'BTCUSD'                  # only one symbol at a time

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
