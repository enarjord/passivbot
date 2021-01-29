# passivbot_futures
trading bot running on bybit inverse futures and binance usdt futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt`

join telegram group for assistance and discussion of settings and live results

https://t.me/passivbot_futures


------------------------------------------------------------------
change log

2021-01-19
- renamed settings["margin_limit"] to settings["balance"]
- bug fixes and changes in trade data downloading
- if there already is historical trade data downloaded, run the script `rename_trade_data_csvs.py` to rename all files

2021-01-23
- removed static mode
- added indicator ema
- rewrote backtester

2021-01-28
- changed backtesting results formatting
- fixed insufficient margin error
- added possiblility of using more indicators
- many other fixes and changes...

------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchanges bybit inverse and binance usdt futures

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 start_bot.py exchange your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate btc (or another coin) in bybit inverse and usdt in binance usdt futures

it is a market maker bot, making a multiple post only limit orders above and below price

it listens to websocket live stream of trades, and updates its orders continuously

when there is no position, it enters long if price < ema, short if price > ema

if there is a long position, it creates 8 reentry bids below pos price, and up to 20 reduce only asks above pos price

reentry_bid_price = pos_price * (1 - grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))

inversely,

if there is a short position, it creates 8 reentry asks above pos price, and up to 20 reduce only closing bids above pos price

reentry_ask_price = pos_price * (1 + grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))


------------------------------------------------------------------

a backtester is included

go to backtesting_settings/{exchange}/, adjust backtesting_settings.json and ranges.json

run with 

`python3 backtest.py exchange your_user_name`

open backtesting_notes.ipynb in jupyter notebook or jupiter-lab for plotting and analysis


about backtesting settings, binance XMRUSDT example

{

    "session_name": "unnamed_session",
    "exchange": "binance",
    "symbol": "XMRUSDT",
    "n_days": 41,                            # n days to backtest

    "random_starting_candidate": false,      # if false, will use settings given as starting candidate
    "starting_k": 0,                         # k is incremented by 1 per iteration until k == n_jackrabbit_iterations
    "n_jackrabbit_iterations": 200,          # see below for more info on jackrabbit
    
    "min_notional": 1.0,                     # used with binance: entry qty must be greater than min_notional / price
    
    "break_on": [
        ["break on first soft stop", "lambda x: x['type'] == 'soft_stop'"],
        ["neg pnl sum", "lambda x: x['pnl_sum'] < 0.0 and x['progress'] > 0.1"]
    ],
                                             # conditions to break backtest prematurely ["name", if true: break.  x is last trade]

    "inverse": false,                        # inverse is true for bybit, false for binance
    "maker_fee": 0.00018,                    # 0.00018 for binance (with bnb discount), -0.00025 for bybit
    "balance": 10.0,                         # backtest starting balance
    "min_qty": 0.001,                        # minimum allowed contract qty
    "price_step": 0.01,
    "qty_step": 0.001,
    "taker_fee": 0.00036,                    # 0.00036 for binance (with bnb discount), 0.00075 for bybit

}


in ranges.json are defined which settings are to be mutated: [min, max, step]

jackrabbit is a pet name given to a simple algorithm for optimizing settings.

for each iteration, settings are mutated to new values within given range defined in ranges.json.

if the new candidate's backtest yields higher gain than best candidate's backtest,

the superior settings becomes the parent of the next candidate.

the mutation coefficient m determines the mutation range, and is inversely proportional to k, which is a simple counter.

in other words, at first new candidates will vary wildly from the best settings, towards the end they will vary less, "fine tuning" the settings.

------------------------------------------------------------------

about settings, bybit example:

{

    "default_qty": 1.0,                   # entry quantity.
                                          # scalable entry quantity mode:
                                          # if "default_qty" is set to a negative value,
                                          # it becomes a percentage of balance (which is actual account balance if settings["balance"] is set to -1).
                                          # the bot will calculate entry qty using the following formula:
                                          # default_qty = max(minimum_qty, round_dn(balance_in_terms_of_contracts * abs(settings["default_qty"]), qty_step))
                                          # bybit BTCUSD example:
                                          # if "default_qty"  is set to -0.06, last price is 37000 and wallet balance is 0.001 btc,
                                          # default_qty = 0.001 * 37000 * 0.06 == 2.22.  rounded down is 2.0 usd.
                                          # binance ETHUSDT example:
                                          # if "default_qty" is set to -0.07, last price is 1100 and wallet balance is 60 usdt,
                                          # default_qty = 60 / 1100 * 0.07 == 0.003818.  rounded down is 0.003 eth.
    
    "ddown_factor": 0.02,                 # next reentry_qty is max(default_qty, abs(pos_size) * ddown_factor).
                                          # if set to 1.0, each reentry qty will be equal to 1x pos size, i.e. doubling pos size after every reentry.
                                          # if set to 0.0, each reentry qty will be equal to default_qty.
                                          
    "indicator_settings": {"tick_ema": {"span": 10000}},
                                          # indicators may be used to determine long or short initial entry.  they are updated on each websocket trade tick.
                                          # ema is not based on ohlcvs, but calculated based on sequence of raw trades.
                                          # when no pos, bid = min(ema, highest_bid), ask = max(ema, lowest_ask)
                                          # more indicators may be added in future.
                                          
    "grid_coefficient": 245.0,            # next entry price is pos_price * (1 +- grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)).
    "grid_spacing": 0.0026,               # 
                                          
    "liq_diff_threshold": 0.02,           # if difference between liquidation price and last price is less than 2%, reduce position by 2% at a loss,
    "stop_loss_pos_reduction": 0.02,      # reduce position by 2% at a loss.
    
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always max leverage).
    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "max_markup": 0.0159,                 # evenly distributed between min and max markup, and whose qtys are pos_size // n_close_orders.
    
    "market_stop_loss": false,            # if true will soft stop with market orders, if false soft stops with limit orders at order book's higest_bid/lowest_ask
    
    "balance": 0.001,                     # balance bot sees.  used to limit pos size and to modify grid spacing
                                          # set settings["balance"] to -1 to use account balance fetched from exchange as balance.
                                          # if using static balance, binance balance is quoted in usdt, bybit inverse balance is quoted in coin.
                                          
    "n_close_orders": 20,                 # max n close orders.
    "n_entry_orders": 8,                  # max n entry orders.
    "symbol": "BTCUSD"                    # only one symbol at a time.

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ

Nano: nano_1nf3knbhapee5ruwg7i8sqekx3zmifdeijr8495t9kgp3uyunik7b9cuyhf5

EOS: nbt4rhnhpjan

XLM: GDSTC6KQR6BCTA7BH45B3MTSY52EVZ4UZTPZEBAZHJMJHTUQQ5SM57S7

bybit ref:
https://www.bybit.com/invite?ref=PQEGz

binance ref:
https://www.binance.cc/en/register?ref=TII4B07C
