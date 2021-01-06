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

it works by entering small, then either closing position at static markup or reentering at in price intervals

depending on initial entry amount and funds available in futures wallet, it will double down repeatedly until position is closed or funds run out

if there is no position, it will make a bid at order book's highest bid and ask at order book's lowest ask

it listens to websocket live stream of trades, and updates its orders continuously

------------------------------------------------------------------

use backtester for research

------------------------------------------------------------------

settings:


{

        'compounding': False,               # used in backtesting
        'ddown_factor': 0.01,               # next entry_qty is pos_size * ddown_factor
        'grid_spacing': 0.003,              # next entry_price is pos_price * (1 +- grid_spacing * grid_spacing_modifier)
                                            # where grid_spacing_modifier is (1 + pos_margin_to_equity_ratio * grid_spacing_coefficient)
                                            # where pos_margin_to_equity_ratio is (pos_size / pos_price) / (equity * leverage)
        'grid_spacing_coefficient': 20.0,   # the purpose of the coefficient is to increase grid spacing when pos_size is high
        'initial_equity': 0.001,            # used in backtesting.  also limits the bot's max allowed pos_size
        'isolated_mode': False,             # used in backtesting.  cross mode if False.  only tested with cross mode, isolated mode not recommended
        'leverage': 100.0,                  # irrelevant because cross mode in bybit is always 100x leverage
        'liq_modifier': 0.001,              # used in backtesting to simulate mark price for liquidations
        'maker_fee': -0.00025,              # used in backtesting.  bot uses only post_only limit orders -- no takers
        'markup': 0.0019,                   # bot closes any position at pos_price * (1 +- markup)
        'min_qty': 1.0,                     # minimum order quantity.  bybit's minimum is 1.0, user may set higher minimum
        'n_days': 0.0,                      # used in backtesting
        'price_step': 0.5,                  # price step
        'qty_step': 1.0,                    # quantity step
        'symbol': 'BTCUSD'                  # only one symbol at a time

}


------------------------------------------------------------------

feel free to make a donation to show support of the work

XMR: 49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ
