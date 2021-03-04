# passivbot_futures

**Version: 2.0.3**

trading bot running on bybit inverse futures and binance usdt futures

use at own risk

requires python >= 3.8

dependencies, install with pip:

`pip install -r requirements.txt`

discord

https://discord.gg/QAF2H2UmzZ

telegram

https://t.me/passivbot_futures

for a repository of settings and their backtesting results, see

https://github.com/JohnKearney1/PassivBot-Configurations

for more detailed documentation on this project, see the wiki at:

https://github.com/enarjord/passivbot_futures/wiki

bybit ref:
https://www.bybit.com/en-US/register?affiliate_id=16464&language=en-US&group_id=0&group_type=1

binance ref:
https://www.binance.cc/en/register?ref=TII4B07C

------------------------------------------------------------------
change log

2021-02-23 v2.0.0_beta
- major update to backtester
- new backtest usage syntax
- other changes

2021-02-27 v2.0.0
- bug fixes
- new default configs for bybit and binance

2021-02-28 v2.0.1
- added optional just-in-time compiling for faster backtesting

2021-03-01 v2.0.2
- more jit'ed calcs

2021-03-02 v2.0.3
- new default bybit config
- behavior change: reentry qtys may now be smaller than initial entry qty
- backtest iterates a numpy array instead of a python list of dicts for reduced ram usage

see `changelog.txt` for earlier changes



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

it is a market maker bot, making multiple post only limit orders above and below price

it listens to websocket live stream of trades, and updates its orders continuously

when there is no position, it enters either long or short depending on indicator settings

if there is a long position, it creates reentry bids below pos price, and reduce-only asks above pos price

reentry_bid_price = pos_price * (1 - grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))

inversely,

if there is a short position, it creates reentry asks above pos price, and reduce-only closing bids below pos price

reentry_ask_price = pos_price * (1 + grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))


------------------------------------------------------------------

a backtester is included

go to `backtest_configs/{config_name}.hjson` and adjust

run with 

`python3 backtest.py {config_name}`

add argument --jit to use numba's just in time compiler for faster backtesting:

`python3 backtest.py {config_name} --jit`

open backtest_notes.ipynb in jupyter notebook or jupiter-lab for plotting and analysis.

jackrabbit is a pet name given to a simple algorithm for optimizing bot's settings.

it iterates many backtests in succession, for each iteration, settings are mutated to new values within given range defined in backtest config.

if the new candidate's backtest yields higher gain than best candidate's backtest,

the superior settings becomes the parent of the next candidate.

the mutation coefficient m determines the mutation range, and is inversely proportional to k, which is a simple counter.

in other words, at first new candidates will vary wildly from the best settings, towards the end they will vary less, "fine tuning" the parameters.

see wiki for more info on backtesting

------------------------------------------------------------------

about live settings, bybit example:

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
            "span": 10000,                # if no pos, bid = min(ema * (1 - spread), highest_bid) and ask = max(ema * (1 + spread), lowest_ask)
            "spread": 0.001
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
https://www.bybit.com/en-US/register?affiliate_id=16464&language=en-US&group_id=0&group_type=1

binance ref:
https://www.binance.cc/en/register?ref=TII4B07C
