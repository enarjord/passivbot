<h1 align="center">
  passivbot
</h1>

![passivbot Version](https://img.shields.io/badge/passivbot-3.5.2-blue)

## Trading bot running on Bybit and Binance Futures

:warning: **Use at own risk** :warning:

## Overview

The bot's purpose is to accumulate tokens over time

It is a market maker bot working in futures markets, making multiple post only limit orders above and below current
price

It listens to websocket live stream of trades, and updates its orders continuously

If possible, it will use hedge mode


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

#### Setting up Telegram

The bot provides interfacing with the bot via Telegram via a telegram bot. In order to set it up, you'll need a telegram
bot token and a chat-id. Once you have those, you can enable teleegram for each individual account that is specified in
the api-keys.json file. There is an example telegram configuration in that file to get started. If a telegram configuration
is not present for an account, telegram is disabled at startup.

For setup instructions, see https://docs.microsoft.com/en-us/azure/bot-service/bot-service-channel-connect-telegram?view=azure-bot-service-4.0

Start a chat with @getmyid_bot in telegram to get chat id.

There are several commands & messages provided via Telegram, please issue a `/help` command in the telegram chat to see
all the options.

### Documentation [WIP], see the wiki at:

https://github.com/enarjord/passivbot/wiki

### Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

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

1. make a backtest_config.hjson file, using `backtest_configs/xmr.hjson` as template
2. run with `python3 backtest.py {path_to_backtest_config.hjson} {path_to_live_config_to_test.json}`

Will use numba's just in time compiler by default to speed up backtesting, add argument `--nojit` to disable numba:

`python3 backtest.py {path_to_backtest_config.hjson} {path_to_live_config_to_test.json} --nojit`

## Optimizer

To optimize a configuration by iterating multiple backtests,

1. make a backtest_config.hjson file, using`backtest_configs/xmr.hjson` as template
2. run with `python3 optimize.py {path_to_backtest_config.hjson}`
3. optionally make optimizer start from given candidate(s) by adding kwarg `--start {path_to_starting_candidate.json}`
   if pointing to a directory, will use all .json files in that directory as starting candidates

See [wiki](https://github.com/enarjord/passivbot/wiki) for more info on backtesting and optimizing

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
