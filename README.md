# passivbot_futures
trading bot running on bybit inverse futures, binance support may be added in future

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3 -m pip install matplotlib pandas websockets ccxt`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

supports exchange bybit

add api key and secret as json file in dir `api_key_secret/{exchange}/your_user_name.json`


formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/{exchange}/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 {exchange}.py your_user_name`

------------------------------------------------------------------
overview

the bot's purpose is to accumulate btc in bybit inverse

will make entries automatically, but will also work with user making manual entries and adding to or removing from positions while the bot is active

it works by entering small, then either closing position at static markup or reentering at intervals

depending on initial entry amount and funds available in futures wallet, it will double down repeatedly until position is closed or funds run out

if there is no position, it will enter long if price < min(emas) * (1 - spread / 2), or short if price > max(emas) * (1 + spread / 2)

it listens to websocket live stream of trades, and updates its orders continuously

------------------------------------------------------------------

use backtester for research

------------------------------------------------------------------

settings:


{

    "ema_spans": [                             # emas are not calculated based on ohlcvs,
        37328.0,                               # but rather on sequence of trades filtered to remove consecutive price duplicates.
        51671.0
    ],
    "ema_spread": -0.000468,                   # if no position, enters long at min(emas) * (1 - spread / 2) and short at max(emas) * (1 + spread / 2)
    "entry_qty_equity_multiplier": 0.00192,    # initial entry qty is equity * entry_qty_equity_multiplier * price
    "entry_qty_scaling_factor": 0.413,         # next entry qty is pos_size * entry_qty_scaling_factor
    "grid_spacing": 0.0047,                    # next entry qty is pos_price (1 +- grid_spacing)
    "initial_equity": 0.001,                   # used in backtesting
    "leverage": 84.0,                          # used in backtesting to determine liquidation price and limit entries.  live leverage is set to cross mode
    "maker_fee": -0.00025,                     # used in backtesting
    "markup": 0.00723,                         # closes entire position at pos_price * (1 +- markup)
    "min_qty": 1.0,                            # used in backtesting
    "price_step": 0.5,                         # used in backtesting
    "qty_step": 1.0,                           # used in backtesting
    "symbol": "BTCUSD"                         # only one symbol at a time

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
