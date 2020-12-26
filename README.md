# passivbot_futures
trading bot running on binance usdt futures and bybit inverse futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt ciso8601`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchanges binance and bybit

binance account and api key need futures enabled

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 {exchange}.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate usdt in binance futures, btc in bybit inverse

will make entries automatically, but will also work with user making manual entries and adding to or removing from positions

it works by entering small, then either closing position at static markup or doubling down at liquidation price

depending on initial entry amount and funds available in futures wallet, it will double down repeatedly until position is closed or funds run out

position size is doubled after each doubling down

more detailed:

if there is no position, it will make small long entry if price / ema < (1 - flashcrash_factor) or small short entry if price / ema > (1 + flashcrash_factor)

if there is a long position, it will make a double down bid of amount equal to position size and price equal to entry_price * (1 - (1 / leverage) / 2), and an exit ask whose amount is equal to position size and price is entry_price * (1 + markup)

if there is a short position, it will make a double down ask of amount equal to position size and price equal to entry_price * (1 + (1 / leverage) / 2), and an exit bid whose amount is equal to position size and price is entry_price * (1 - markup)

it listens to websocket live stream of aggregated trades, and updates its orders continuously

------------------------------------------------------------------

risk may be ascertained by the number of double downs funds allows

given binance BTCUSDT with leverage of 100x and initial entry amount of 0.001, doubling down 9 times builds a position of 0.001 * 2**9 == 0.512

at 20000 usd/btc, that requires margin of around 20000 * 0.512 / 100 == 102.4 usdt

so with 102.4+ usdt available, the bot may double down up to 9 times

at 100x each doubling down pushes liquidation price (1 / 100) / 2 == 0.25% away

so 9 double downs pushes liquidation price 9 * 0.25 == 2.25% away from first liquidation price

use backtester for further research

------------------------------------------------------------------

settings:


{

    "ema_spans": [1000, ...],     # emas are calculated based on sequence of trades, not ohlcvs.
    
                                  # so 50 trades during one minute is the same as 50 trades during one hour
                                  
                                  # entry bid is < min(emas) and entry ask is > max(emas)
                                  
    "entry_amount": 0.001,        # initial entry amount
    
    "spread": 0.001,              # if no position, enters long at min(emas) * (1 - spread) and short at max(emas) * (1 + spread)
    
    "leverage": 100,              # leverage.  bot will set leverage to this value at startup
    
    "markup": 0.0015,             # markup does not take fees into account
    
    "symbol": "BTCUSDT"           # only one symbol at a time
    
}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
