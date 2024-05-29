![Passivbot](docs/images/pbot_logo_full.svg)

# Trading bot running on Bybit, Binance, OKX, Kucoin, Bitget, BingX and Hyperliquid

:warning: **Used at one's own risk** :warning:

v7.0.0


## Overview

Passivbot is a cryptocurrency trading bot written in Python, intended to require minimal user intervention.  

It operates on perpetual futures derivatives markets, automatically creating and cancelling limit buy and sell orders on behalf of the user. It does not try to predict future price movements, it does not use technical indicators, nor does it follow trends. Rather, it is a contrarian market maker, providing resistance to price changes in both directions, thereby "serving the market" as a price stabilizer.  

Passivbot's behavior may be backtested on historical price data, using the included backtester whose CPU heavy functions are written in Rust for speed. Also included is an optimizer, which finds better configurations by iterating thousands of backtests with different candidates, converging on the optimal ones with an evolutionary algorithm.  

## Strategy

Inspired by the Martingale betting strategy, the robot will make a small initial entry and double down on its losing positions multiple times to bring the average entry price closer to current price action. The orders are placed in a grid, ready to absorb sudden price movements. After each re-entry, the robot quickly updates its closing orders at a set take-profit markup. This way, if there is even a minor market reversal, or "bounce", the position can be closed in profit, and it starts over.  

### Forager
The Forager feature dynamically chooses the most volatile markets on which to open positions. Volatility is defined as the mean of the normalized relative range for the most recent candles (15m candles by default), i.e. `mean((ohlcv.high - ohlcv.low) / ohlcv.close)`.

### Unstucking Mechanism
Passivbot manages underperforming, or "stuck", positions by realizing small losses over time. If multiple positions are stuck, the bot prioritizes positions with the smallest gap between the entry price and current market price for "unstucking". Losses are limited by ensuring that the account balance does not fall under a set percentage below the past peak balance.  


## Installation

To install Passivbot and its dependencies, follow the steps below.

### Step 1: Clone the Repository

First, clone the Passivbot repository to your local machine:

```sh
git clone https://github.com/enarjord/passivbot.git
cd passivbot
```

### Step 2: Create and Activate a Virtual Environment

Create a virtual environment to manage dependencies:

```sh
python3 -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```sh
  .\venv\Scripts\activate
  ```
- On macOS and Linux:
  ```sh
  source venv/bin/activate
  ```

### Step 3: Install Python Dependencies

Install all the required Python dependencies listed in the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Step 4: Build Rust Extensions

Navigate to the `passivbot-rust` directory and build the Rust extensions using `maturin`:

```sh
cd passivbot-rust
maturin develop
```

### Step 5: Add API keys

Make a copy of the api-keys template file:

```sh
cp api-keys.json.example api-keys.json
```

Add your keys to api-keys.json.

### Step 6: Run Passivbot

Make a live configuration file, using `configs/live/example_config.hjson` as a template.

To start the bot, run:

```sh
python src/main.py configs/live/example_config.hjson
```

## Requirements

- Python >= 3.8
- [requirements.txt](requirements.txt) dependencies

## Pre-optimized configurations

Coming soon...

See also https://pbconfigdb.scud.dedyn.io/

## Documentation:

For more detailed information about Passivbot, see documentation files here: [docs/](docs/) or on www.passivbot.com

## Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

## Third Party Links, Referrals and Tip Jar

**Passivbot Manager Service:**  
There is a paid manager service to run Passivbot on the user's behalf:  
www.passivbotmanager.com  

**Referrals:**  
Signing up using these referrals is appreciated:  
https://accounts.binance.me/en/register?ref=TII4B07C  
https://partner.bybit.com/b/passivbot  
https://partner.bitget.com/bg/Y8FU1W  
https://www.okx.com/join/PASSIVBOT  (20% rebate)  
https://www.kucoin.com/r/rf/QBSFZ5HT  
https://bingx.com/invite/DFONXA  

**BuyMeACoffee:**  
https://www.buymeacoffee.com/enarjord  

**Donations:**  
If the robot is profitable, consider donating as showing gratitude for its development:  
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
Anybody may copy, distribute, modify, use or misuse for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do whatever without permission from anybody.
