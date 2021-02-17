# passivbot_futures

**Version: 1.2.0**

trading bot running on bybit inverse futures and binance usdt futures

use at own risk

requires python >= 3.8

dependencies, install with pip:

`python3 -m pip install matplotlib pandas websockets ccxt`

discord

https://discord.gg/QAF2H2UmzZ

telegram

https://t.me/passivbot_futures

for a repository of settings and their backtesting results, see

https://github.com/JohnKearney1/PassivBot-Configurations

for more detailed documentation on this project, see the wiki at:

https://github.com/enarjord/passivbot_futures/wiki

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

2021-02-10
- renamed settings["default_qty"] to settings["entry_qty_pct"]
- settings["entry_qty_pct"] may now also be a positive value
- renamed settings["balance"] to settings["balance_pct"]
- settings["balance_pct"] may now also be a positive value
- added balance_pct to backtester.  backtester will now behave like live bot, taking balance_pct into account
    - actual balance is used for liq price calc, otherwise balance * balance_pct is used

2021-02-12
- added indicator_settings["funding_fee_collect_mode"]
    - if true, will enter long only if predicted fundig rate is negative and enter short only if predicted funding rate is positive
- added indicator rsi (not finished, not active)
- changed entry_qty_pct formula
    - before initial_entry_qty = balance_ito_contracts * entry_qty_pct
    - now    initial_entry_qty = balance_ito_contracts * leverage * entry_qty_pct
- added logging
- added "config_name" and "logging_level" to live settings
- added break_on condition: break if balance + pnl < starting_balance

2021-02-16
- bug fixes v1.0.2
- updated default bybit live settings v1.1.0

2021-02-17 v1.2.0
- bug fixes
- change in backtesting_notes.ipynb
    - automatic plot dump
    - other changes
- backtester now fetches relevant data from exchanges instead of user having to input them manually


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

    "starting_candidate_preference": ["best", "random", "given"],
                                             # starting candidate preference from left to right.
                                             # if best is first and there is a best.json file present, will build on best.
                                             # otherwise, starting candidate will be either random or given, depending which is before the other.
                                             # after first iteration, will build on best regardless of starting_candidate_preference
    
    "starting_k": 0,                         # k is incremented by 1 per iteration until k == n_jackrabbit_iterations
    "n_jackrabbit_iterations": 200,          # see below for more info on jackrabbit
    
    "min_notional": 1.0,                     # used with binance: entry qty must be greater than min_notional / price
    "cross_mode": true,                      # true for cross mode, false for isolated mode
    "max_leverage": 75,                      # max allowed leverage for symbol
    "do_long": true,
    "do_shrt": true,

    
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
                                             # conditions to break backtest prematurely and returns empty list of trades.
                                             # ["name", if true: break.  trade is last trade, tick is last price tick]
                                             # if startswith "OFF", will ignore condition.

    "inverse": false,                        # inverse is true for bybit, false for binance
    "maker_fee": 0.00018,                    # 0.00018 for binance (with bnb discount), -0.00025 for bybit

    "starting_balance": 10.0,                # backtest starting balance
                                             # backtest balance never goes lower than starting balance,
                                             # as if topping up wallet back to starting balance each time balance goes below starting balance
    
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

    "balance_pct": 0.5,                   # if settings["balance_pct"] = 1.0, will use 100% of balance.
                                          # if settings["balance_pct"] = 0.35, will us 35% of balance.
    "config_name": "BTCUSD_default",      # arbitrary name given to settings.
    "cross_mode": true,                   # true for cross, false for isolated.
                                          # use isolated mode with care.  depending on settings, there is high risk of accidental liquidations.

    "entry_qty_pct": 0.005,               # percentage of balance * leverage used as initial entry qty.
                                          # the bot will calculate initial entry qty using the following formula:
                                          # initial_entry_qty = round_dn(balance_in_terms_of_contracts * leverage * abs(settings["entry_qty_pct"]), qty_step)
                                          # bybit BTCUSD example:
                                          # if "entry_qty_pct"  is set to 0.0021, last price is 37000, leverage is 50 and wallet balance is 0.001 btc,
                                          # initial_entry_qty = 0.001 * 37000 * 50 * 0.0021 == 3.885.  rounded down is 3.0 usd.
                                          # binance ETHUSDT example:
                                          # if "entry_qty_pct" is set to 0.07, last price is 1100, leverage is 33 and wallet balance is 40 usdt,
                                          # initial_entry_qty = (40 / 1100) * 33 * 0.07 == 0.084.  rounded down is 0.084 eth.
    
    "ddown_factor": 0.02,                 # next reentry_qty is max(initial_entry_qty, abs(pos_size) * ddown_factor).
                                          # if set to 1.0, each reentry qty will be equal to 1x pos size, i.e. doubling pos size after every reentry.
                                          # if set to 1.5, each reentry qty will be equal to 1.5x pos size.
                                          # if set to 0.0, each reentry qty will be equal to initial_entry_qty.
                                          
    "indicator_settings": {
        "tick_ema": {                     # tick ema is not based on ohlcvs, but calculated based on sequence of raw trades.
            "span": 10000                 # if no pos, bid = min(ema, highest_bid) and ask = max(ema, lowest_ask)
        },                                # if ema span is set to 1.0, ema is always equal to last price, which will disable ema smoothing of initial entries

        "funding_fee_collect_mode": false,# if true, will enter long only if predicted funding fee is < 0.0, and short only if predicted funding fee is > 0.0

        "do_long": true,                  # if true, will allow long positions
        "do_shrt": true                   # if true, will allow short posisions
    },
                                          
    "grid_coefficient": 245.0,            # next entry price is pos_price * (1 +- grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)).
    "grid_spacing": 0.0026,               # 
                                          
    "stop_loss_liq_diff": 0.02,           # if difference between liquidation price and last price is less than 2%, ...
    "stop_loss_pos_price_diff": 0.04,     # ... or if difference between pos price and last price is greater than 4%, reduce position by 2% at a loss,

    "stop_loss_pos_reduction": 0.02,      # reduce position by 2% at a loss.
    
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always max leverage).
    "logging_level": 0,                   # if logging_level > 0,
                                          # will log positions, open orders, order creations and order cancellations in logs/{exchange}/{config_name}.log.

    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "max_markup": 0.0159,                 # evenly distributed between min and max markup, and whose qtys are pos_size // n_close_orders.
    "min_close_qty_multiplier": 0.5       # min_close_qty = max(min_qty, initial_entry_qty * min_close_qty_multiplier)
    
    "market_stop_loss": false,            # if true will soft stop with market orders, if false soft stops with limit orders at order book's higest_bid/lowest_ask
                                          
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

USDT TRC20 (binance): TJr3KYY8Bz7wRU7QLwoYQHk88LcaBJqQN5

bybit ref:
https://www.bybit.com/invite?ref=PQEGz

binance ref:
https://www.binance.cc/en/register?ref=TII4B07C
