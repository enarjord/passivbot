# passivbot_futures
trading bot running on binance usdt futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

binance account needs futures enabled

add api key and secret as json file in dir `api_key_secret/binance/your_user_name.json`

formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/binance_futures/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 passivbot_futures.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate usdt

it is intended to work with leverage of 75x and higher

if long and price moves up or short and price moves down it will close entire position at given markup

if long and price moves down or short and price moves up it will double down at liquidation price before position is liquidated

position size is doubled after each double down

with initial entry amount of 0.001, size will be 0.002, 0.004, 0.008, 0.016 etc until either position is closed or no more funds are available

more specifically:

if there is no position, it will make small long entry if price / ema < (1 - flashcrash_factor) or small short entry if price / ema > (1 + flashcrash_factor)

if there is a long position, it will make a bid of amount equal to position size and price equal to entry_price * (1 - (1 / leverage) / 2), which is liquidation price at given leverage, and an ask whose amount is equal to position size and price is entry_price * (1 + markup)

if there is a short position, it will make an ask of amount equal to position size and price equal to entry_price * (1 + (1 / leverage) / 2), which is liquidation price at given leverage, and a bid whose amount is equal to position size and price is entry_price * (1 - markup)


