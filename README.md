# passivbot_futures
trading bot running on bybit inverse futures and binance usdt futures

use at own risk

requires python >= 3.8

dependencies, install with pip:

`python3 -m pip install matplotlib pandas websockets ccxt`

discord

https://discord.gg/QAF2H2UmzZ

telegram

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

2021-01-30
- changed backtesting results formatting
- fixed insufficient margin error
- many other fixes and changes...
- added possibility of running same backtest in two or more terminals for better cpu utilization

2021-02-03
- backtester break conditions change
- bug fixes

2021-02-08
- added min_close_qty_multiplier

2021-02-09
- added classic stop loss

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

optional:  specify session name as arg:

`python3 backtest.py exchange your_user_name session_name`

otherwise will use session_name given in backtesting_settings.json

open backtesting_notes.ipynb in jupyter notebook or jupiter-lab for plotting and analysis


about backtesting settings, binance XMRUSDT example

{

    "session_name": "unnamed_session",       # arbitrary name.
    "exchange": "binance",
    "symbol": "XMRUSDT",
    "n_days": 41,                            # n days to backtest

    "random_starting_candidate": false,      # if false, will use settings given as starting candidate
    "starting_k": 0,                         # k is incremented by 1 per iteration until k == n_jackrabbit_iterations
    "n_jackrabbit_iterations": 200,          # see below for more info on jackrabbit
    
    "min_notional": 1.0,                     # used with binance: entry qty must be greater than min_notional / price
    
    "break_on": [
        ["OFF: break on first soft stop",
         "lambda trade, tick: trade['type'] == 'stop_loss'"],
        ["ON: neg pnl sum",
         "lambda trade, tick: trade['pnl_sum'] < 0.0 and trade['progress'] > 0.4"],
        ["ON: liq diff too small",
         "lambda trade, tick: trade['liq_diff'] < 0.02"],
        ["ON: time between consec trades",
         "lambda trade, tick: tick['timestamp'] - trade['timestamp'] > 1000 * 60 * 60 * 24"],
        ["ON: pos price last price diff",
         "lambda trade, tick: calc_diff(trade['price'], tick['price']) > 1.05"]
    ],
                                             # conditions to break backtest prematurely ["name", if true: break.  trade is last trade, tick is last price tick]
                                             # if startswith "OFF", will ignore condition

    "inverse": false,                        # inverse is true for bybit, false for binance
    "maker_fee": 0.00018,                    # 0.00018 for binance (with bnb discount), -0.00025 for bybit
    "starting_balance": 10.0,                # backtest starting balance
    "min_qty": 0.001,                        # minimum allowed contract qty
    "price_step": 0.01,
    "qty_step": 0.001,
    "taker_fee": 0.00036,                    # 0.00036 for binance (with bnb discount), 0.00075 for bybit
    "min_close_qty_multiplier": 0.5,         # min_close_qty = initial_entry_qty * min_close_qty_multiplier


}


in ranges.json are defined which settings are to be mutated: [min, max, step]

jackrabbit is a pet name given to a simple algorithm for optimizing settings.

for each iteration, settings are mutated to new values within given range defined in ranges.json.

if the new candidate's backtest yields higher gain than best candidate's backtest,

the superior settings becomes the parent of the next candidate.

the mutation coefficient m determines the mutation range, and is inversely proportional to k, which is a simple counter.

in other words, at first new candidates will vary wildly from the best settings, towards the end they will vary less, "fine tuning" the settings.

it is possible to run the same backtest in two or more terminals simultaneously.  they will share best candidate and dump results in same file for later analysis.

if you wish to do so, use the same session name for all and be sure to start with only one and let it finish downloading trades and making a trades_list cache before starting the others.

------------------------------------------------------------------

about settings, bybit example:

{

    "entry_qty_pct": 0.05,                # 
                                          # percentage of balance used as initial entry qty.
                                          # the bot will calculate initial entry qty using the following formula:
                                          # initial_entry_qty = max(minimum_qty, round_dn(balance_in_terms_of_contracts * abs(settings["entry_qty_pct"]), qty_step))
                                          # bybit BTCUSD example:
                                          # if "entry_qty_pct"  is set to 0.06, last price is 37000 and wallet balance is 0.001 btc,
                                          # initial_entry_qty = 0.001 * 37000 * 0.06 == 2.22.  rounded down is 2.0 usd.
                                          # binance ETHUSDT example:
                                          # if "entry_qty_pct" is set to 0.07, last price is 1100 and wallet balance is 60 usdt,
                                          # initial_entry_qty = 60 / 1100 * 0.07 == 0.003818.  rounded down is 0.003 eth.
    
    "ddown_factor": 0.02,                 # next reentry_qty is max(initial_entry_qty, abs(pos_size) * ddown_factor).
                                          # if set to 1.0, each reentry qty will be equal to 1x pos size, i.e. doubling pos size after every reentry.
                                          # if set to 0.0, each reentry qty will be equal to initial_entry_qty.
                                          
    "indicator_settings": {
        "tick_ema": {"span": 10000},
        "do_long": true,                  # if true, will allow long positions
        "do_shrt": true                   # if true, will allow short posisions
    },
                                          # indicators may be used to determine long or short initial entry.  they are updated on each websocket trade tick.
                                          # tick ema is not based on ohlcvs, but calculated based on sequence of raw trades.
                                          # when no pos, bid = min(tick_ema, highest_bid), ask = max(tick_ema, lowest_ask)
                                          # more indicators may be added in future.
                                          
    "grid_coefficient": 245.0,            # next entry price is pos_price * (1 +- grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)).
    "grid_spacing": 0.0026,               # 
                                          
    "stop_loss_liq_diff": 0.02,           # if difference between liquidation price and last price is less than 2%, ...
    "stop_loss_pos_price_diff": 0.04,     # ... or if difference between pos price and last price is greater than 4%, reduce position by 2% at a loss,

    "stop_loss_pos_reduction": 0.02,      # reduce position by 2% at a loss.
    
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always max leverage).
    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "max_markup": 0.0159,                 # evenly distributed between min and max markup, and whose qtys are pos_size // n_close_orders.
    "min_close_qty_multiplier": 0.5       # optional setting, will default to 0.0 if not present.
                                          # min_close_qty = max(min_qty, initial_entry_qty * min_close_qty_multiplier)
    
    
    "market_stop_loss": false,            # if true will soft stop with market orders, if false soft stops with limit orders at order book's higest_bid/lowest_ask
    
    "balance_pct": 0.5,                   # if settings["balance_pct"] = 1.0, will use 100% of balance.
                                          # if settings["balance_pct"] = 0.35, will us 35% of balance.
                                          
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
