<h1 align="center">
  passivbot
</h1>

![passivbot Version](https://img.shields.io/badge/passivbot-3.3.3-blue)
![GitHub issues](https://img.shields.io/github/issues/enarjord/passivbot)
![GitHub](https://img.shields.io/github/license/enarjord/passivbot)

## Trading bot running on Bybit and Binance Futures using hedge mode when possible

:warning: **Use at own risk** :warning:

## Overview

The bot's purpose is to accumulate tokens over time

It is a market maker bot working in futures markets, making multiple post only limit orders above and below current
price

It listens to websocket live stream of trades, and updates its orders continuously

If there is a long position, it creates reentry bids below pos price, and reduce-only asks above pos price:

`reentry_bid_price = pos_price * (1 - grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))`

If there is a short position, it creates reentry asks above pos price, and reduce-only closing bids below pos price:

`reentry_ask_price = pos_price * (1 + grid_spacing * (1 + (position_margin / wallet_balance) * grid_coefficient))`

In hedge mode, stop loss works like this:

```
if diff(liq_price, last_price) < stop_loss_liq_diff:
    if available margin:
        enter opposite side
    else:
        close same side at a loss
```

In both cases liq price will be pushed away

### Requirements

- Python >= 3.8
- [requirements.txt](requirements.txt) dependencies

### Setup dependencies

```bash
    pip install -r requirements.txt
```

### Usage:

#### Supports exchanges Bybit futures and Binance futures, using hedge mode when supported

1. Add your api key and secret in file [api-keys.json](api-keys.json)
2. ```bash
   python3 start_bot.py {account_name} {symbol} {path/to/config.json}
   ```

Example:

```bash
python3 start_bot.py binance_01 XMRUSDT live_configs/binance_default.json
```

#### Run with docker

Modify command with exchange and user_name in [docker-compose.yml](docker-compose.yml) and start
with `docker-compose up -d` (-d for background run). All code and files generated are in original git folder.

#### Stopping bot

For graceful stopping of the bot, set `do_long`and `do_shrt` both to `false`, and bot will continue as normal, opening
no new positions, until all existing positions are closed.

### Documentation [WIP], see the wiki at:

https://github.com/enarjord/passivbot/wiki

### Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://t.me/passivbot_futures)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

### Resources

- Repository of settings and their backtesting results: https://github.com/JohnKearney1/PassivBot-Configurations

### [Changelog](changelog.md)

### License

Released freely without conditions.
Anybody may copy, distribute, modify, use or misuse for commercial,
non-commercial, educational or non-educational purposes, censor,
claim as one's own or otherwise do whatever without permission from anybody.

## Backtester

A backtester is included

1. go to `backtest_configs/{config_name}.hjson` and adjust it
2. run with `python3 backtest.py {config_name}`

Will use numba's just in time compiler by default to speed up backtesting, add argument `--nojit` to disable numba:

`python3 backtest.py {config_name} --nojit`

See [wiki](https://github.com/enarjord/passivbot/wiki) for more info on backtesting

## Live settings

- [Binance](live_configs/binance_default.json)
- [Bybit](live_configs/bybit_default.json)

## Support the project

### Feel free to make a donation to show support of the work

- XMR: `49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ`

- Nano: `nano_1nf3knbhapee5ruwg7i8sqekx3zmifdeijr8495t9kgp3uyunik7b9cuyhf5`

- EOS: `nbt4rhnhpjan`

- XLM: `GDSTC6KQR6BCTA7BH45B3MTSY52EVZ4UZTPZEBAZHJMJHTUQQ5SM57S7`

- USDT TRC20 (Binance): `TJr3KYY8Bz7wRU7QLwoYQHk88LcaBJqQN5`

### Referrals

- [Bybit](https://www.bybit.com/en-US/register?affiliate_id=16464&language=en-US&group_id=0&group_type=1)
- [Binance](https://www.binance.cc/en/register?ref=TII4B07C)
