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

will make entries automatically, but will also work with user making manual entries and adding to or removing from positions

it can go both long and short, taking profit at set static markup

if price moves up when short or down when long, it will double down on its position at liquidation price, thus pushing liquidation further away

depending on initial entry amount and funds available in futures wallet, it can double down repeatedly until position is closed or funds run out

position size is doubled after each doubling down

more detailed:

if there is no position, it will make small long entry if price / ema < (1 - flashcrash_factor) or small short entry if price / ema > (1 + flashcrash_factor)

if there is a long position, it will make a bid of amount equal to position size and price equal to entry_price * (1 - (1 / leverage) / 2), which is liquidation price at given leverage, and an ask whose amount is equal to position size and price is entry_price * (1 + markup)

if there is a short position, it will make an ask of amount equal to position size and price equal to entry_price * (1 + (1 / leverage) / 2), which is liquidation price at given leverage, and a bid whose amount is equal to position size and price is entry_price * (1 - markup)

------------------------------------------------------------------

risk may be ascertained by the number of double downs funds allow

given 100x BTCUSDT with initial entry amount of 0.001, then doubling down 9 times builds a position of 0.001 * 2**9 == 0.512

at 20000 usd/btc, that requires margin of around 20000 * 0.512 / 100 == 102.4 usdt

so with 102.4+ usdt available, the bot may double down up to 9 times

at 100x each doubling down pushes liquidation price (1 / 100) / 2 == 0.25% away

so 9 double downs pushes liquidation price 9 * 0.25 == 2.25% away from initial liquidation price

use backtester for further research

------------------------------------------------------------------

a backtester is included

settings:


{

    "ema_span": 1000,             # ema is calculated based on sequence of trades, not ohlcvs.
    
                                  # so 50 trades during one minute is the same as 50 trades during one hour
                                  
    "entry_amount": 0.001,        # initial entry amount
    
    "flashcrash_factor": 0.001,   # if no position, enters long at ema * (1 - flashcrash_factor) and short at ema * (1 + flashcrash_factor)
    
    "leverage": 100,              # leverage.  bot will set leverage to this value at startup
    
    "markup": 0.0014,             # markup does not take fees into account
    
    "symbol": "BTCUSDT"           # only one symbol at a time
    
}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
