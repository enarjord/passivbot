# passivbot_futures

**Version: 3.3.1**

trading bot running on bybit and binance futures using hedge mode when possible

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

2021-03-12 v3.0.0
- binance hedge mode implemented (bybit not yet supported)
- emas removed
- stop loss removed

2021-03-18 v3.0.1
- hedge mode backtester implemented
- emas added back

2021-03-21 v3.1.0
- removed setting min_close_qty_multiplier
- added setting close_qty_pct, which complements entry_qty_pct
- rewrote iter_long/shrt_closes
- fixed memory leak bug

2021-03-23 v3.2.0
- implemented particle swarm optimizationg algorithm, replacing jackrabbit
- bybit hedge mode with inverse futures
- removed config param close_qty_pct
- removed config param balance_pct
- removed config param max_markup
- added config param markup_range

2021-03-25 v3.2.1
- bug fixes
- bybit backtester improvements and bug fixes
- numba is now enabled by default, use --nojit to disable numba
- several renamings

2021-03-30 v3.3.0
- bybit usdt linear perpetual and bybit inverse perpetual markets now supported
- new downloader for historical backtesting data

2021-04-01 v3.3.1
- binance inverse futures coin margined markets now supported


see `changelog.txt` for earlier changes



------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchanges bybit inverse futures and binance usdt futures

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `live_settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 start_bot.py exchange your_user_name`

run in docker: modify command with exchange and user_name in docker-compose and start with `docker-compose up -d` (-d for background run).  all code and files generated are in original git folder.

------------------------------------------------------------------
overview

the bot's purpose is to accumulate tokens over time

it is a market maker bot working in futures markets, making multiple post only limit orders above and below current price

it listens to websocket live stream of trades, and updates its orders continuously

if there is a long position, it creates reentry bids below pos price, and reduce-only asks above pos price

reentry_bid_price = pos_price * (1 - grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))

if there is a short position, it creates reentry asks above pos price, and reduce-only closing bids below pos price

reentry_ask_price = pos_price * (1 + grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))


------------------------------------------------------------------

a backtester is included

go to `backtest_configs/{config_name}.hjson` and adjust

run with 

`python3 backtest.py {config_name}`

will use numba's just in time compiler by default to speed up backtesting

add argument --nojit to disable numba:

`python3 backtest.py {config_name} --nojit`

see wiki for more info on backtesting

------------------------------------------------------------------

about live settings, bybit example:

{

    "config_name": "BTCUSD_default",      # arbitrary name given to settings.

    "qty_pct": 0.005,                     # percentage of balance * leverage used as min order qty.
    
    "ddown_factor": 0.02,                 # next reentry_qty is max(initial_entry_qty, abs(pos_size) * ddown_factor).
                                          # if set to 1.0, each reentry qty will be equal to 1x pos size, i.e. doubling pos size after every reentry.
                                          # if set to 1.5, each reentry qty will be equal to 1.5x pos size.
                                          # if set to 0.0, each reentry qty will be equal to initial_entry_qty.
                                          
    "ema_span": 10000,                    # tick ema is not based on ohlcvs, but calculated based on sequence of raw trades.
    "ema_spread": 0.001                   # if no pos, bid = min(ema * (1 - spread), highest_bid) and ask = max(ema * (1 + spread), lowest_ask)

    "do_long": true,                      # if true, will allow long positions
    "do_shrt": true                       # if true, will allow short posisions
                                          
    "grid_coefficient": 245.0,            # next entry price is pos_price * (1 +- grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)).
    "grid_spacing": 0.0026,               # 
                                          
    "leverage": 100,                      # leverage (irrelevant in bybit because cross mode in is always max leverage).
    "logging_level": 0,                   # if logging_level > 0,
                                          # will log positions, open orders, order creations and order cancellations in logs/{exchange}/{config_name}.log.

    "min_markup": 0.0002,                 # when there's a position, bot makes a grid of n_close_orders whose prices are
    "markup_range": 0.0159,               # evenly distributed between min_markup and (min_markup + markup_range), and whose qtys are pos_size // n_close_orders.
                        
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
