![Passivbot](docs/images/pbot_logo_full.svg)

# Trading bot running on Bybit, Binance, OKX, Kucoin and Bitget

:warning: **Use at own risk** :warning:

v6.0.1


## Overview

Passivbot is a cryptocurrency trading bot written in Python, intended to require minimal user intervention.  
It is observed that prices in a market will fluctuate up and down, creating opportunities for capitalizing on the noise.  
The bot's purpose is to automate the harvest.

It operates on spot or futures markets by listening to websocket stream of live trades,
automatically creating and cancelling limit buy and sell orders.

Because passivbot's live behavior is deterministic, it may be simulated on historical price data, using the included backtester.  
Also included is an optimizer, which finds better configurations by iterating thousands of backtests with different candidates.  

The strategy is integrated -- the bot only needs a predefined configuration to run.

## Requirements

- Python >= 3.8
- [requirements.txt](requirements.txt) dependencies

## Pre-optimized configurations

Pre-optimized configurations for Passivbot can be found at https://github.com/JohnKearney1/PassivBot-Configurations.  

See also https://pbconfigdb.scud.dedyn.io/

## Documentation:

For more detailed information about Passivbot, see documentation on https://www.passivbot.com

See also https://www.futuresboard.xyz/guides.html

## Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

## Referrals and Tip Jar

https://accounts.binance.me/en/register?ref=TII4B07C  
https://partner.bybit.com/b/passivbot  
https://partner.bitget.com/bg/Y8FU1W  
https://www.okx.com/join/PASSIVBOT  (20% rebate)  
https://www.kucoin.com/r/rf/QBSFZ5HT  
https://bingx.com/invite/DFONXA  

https://www.buymeacoffee.com/enarjord  

Tether USD (USDT):  
- USDT Solana:  
9hUCHBQA261PU6rUSbwgMoxn8nrdcXPAYgbASRgA8dtm  
- USDT Binance Smart Chain BEP20:  
0x574cad69595fe80c6424ea21988ca0e504cd90cc  
- USDT Matic Polygon:  
0x574cad69595fe80c6424ea21988ca0e504cd90cc  

Monero (XMR):  
49gUQ1jasDK23tJTMCvP4mQUUwndeLWAwSgdCFn6ovmRKXZAjQnVp2JZ2K4UuDDdYMNam1HE8ELZoWdeJPRfYEa9QSEK6XZ

Bitcoin (BTC):  
bc1qcc3kr9gudg35gnsljg64zeaurt0y24pfej36w6

## License

Released freely without conditions.
Anybody may copy, distribute, modify, use or misuse for commercial,
non-commercial, educational or non-educational purposes, censor,
claim as one's own or otherwise do whatever without permission from anybody.
