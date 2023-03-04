# Welcome to Passivbot!

![Passivbot](images/pbot_logo_full.svg)

<a class="github-button" href="https://github.com/enarjord/passivbot" data-icon="octicon-star" data-size="large" aria-label="Star enarjord/passivbot on GitHub">Star</a>
<a class="github-button" href="https://github.com/enarjord/passivbot/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork enarjord/passivbot on GitHub">Fork</a>

!!! Warning
    **Use at your own risk! You should never trade with money you cannot or are are afraid to miss.**

    YOU ALONE ARE RESPONSIBLE FOR HOW THE BOT BEHAVES, AND FOR ANY LOSSES THAT MAY BE A RESULT OF USING THE BOT!

    If you don't understand how the bot works, you should not use it. You are always welcome to ask for help or get more
    information on how it works exactly if the documentation does not provide enough information. 

## Overview

Passivbot is a fully automated trading bot built in Python. The goal of Passivbot is to provide a stable, low-risk
trading bot that is able to accumulate profit over time without manual actions.

Passivbot trades in futures and spot markets, using an advanced form of grid trading to place opening and closing orders.

It provides support for automatically optimize configurations, backtest using historic data, and running in live mode.

## Supported exchanges

Currently the exchanges  
**Binance USDT and Coin margined Futures**  
**Binance Spot**  
**Binance US**  
**Bitget USDT and Coin margined Futures**  
**OKX USDT margined perpetuals**  
**Kucoin USDT margined perpetuals**  
**Bybit Derivatives**  
are implemented and supported.  

Not all functionality is supported on all exchanges, depending on the APIs that exchanges expose and development efforts.

!!! Info
    If you would like to have support added for an exchange not supported yet, 
    you can always get in touch to see what the options are.

## Software Requirements

The following requirements are applicable for installing Passivbot:

- Git 2.17 or above
- Python 3.8.x and higher
- Supported OS: 
    - Mac
    - Linux
    - Windows

## Hardware requirements

Passivbot is a very lightweight bot, and can easily be run live on a single-core machine with 1GB of RAM.
While the hardware requirements are very low, you may run into issues when running it on things like a Raspberry Pi.
In case you do, please let us know so we can help out & improve the bot!

Be aware that running an optimize on a long period may require a lot of memory and CPU power. Since Passivbot acts on
trade data (ticks) instead of OHLCV data, the amount of memory is dependent on the amount of trades made in the selected time period.
If you want to optimize on BTC for the last 365 days, it will take significantly more memory than optimizing on a coin
like BTS for example.

For running the optimizer, a minimum of 8gb of memory is recommended.

!!! Warning
    It should be very possible to run multiple bots on a single machine. Be aware however that you may run into other
    limitations like rate limiting of exchanges.

## Support

In case you run into trouble, have questions or would just like to get in touch with the community, you can find
us at [this Discord server](https://discord.gg/QAF2H2UmzZ) or at [this Telegram channel](https://t.me/passivbot_futures).

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

## License

Passivbot is released freely without conditions. Anybody may copy, distribute, modify, use or misuse for commercial, non-commercial, 
educational or non-educational purposes, censor, claim as one's own or otherwise do whatever without permission from anybody.
