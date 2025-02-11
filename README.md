![Passivbot](docs/images/pbot_logo_full.svg)

# Trading bot running on Bybit, OKX, Bitget, GateIO, Binance and Hyperliquid

:warning: **Used at one's own risk** :warning:

v7.2.17


## Overview

Passivbot is a cryptocurrency trading bot written in Python and Rust, intended to require minimal user intervention.  

It operates on perpetual futures derivatives markets, automatically creating and cancelling limit buy and sell orders on behalf of the user. It does not try to predict future price movements, it does not use technical indicators, nor does it follow trends. Rather, it is a contrarian market maker, providing resistance to price changes in both directions, thereby "serving the market" as a price stabilizer.  

Passivbot's behavior may be backtested on historical price data, using the included backtester whose CPU heavy functions are written in Rust for speed. Also included is an optimizer, which finds better configurations by iterating thousands of backtests with different candidates, converging on the optimal ones with an evolutionary algorithm.  

## Strategy

Inspired by the Martingale betting strategy, the robot will make a small initial entry and double down on its losing positions multiple times to bring the average entry price closer to current price action. The orders are placed in a grid, ready to absorb sudden price movements. After each re-entry, the robot quickly updates its closing orders at a set take-profit markup. This way, if there is even a minor market reversal, or "bounce", the position can be closed in profit, and it starts over.  

### Trailing Orders
In addition to grid-based entries and closes, Passivbot may be configured to utilize trailing entries and trailing closes.

For trailing entries, the bot waits for the price to move beyond a specified threshold and then retrace by a defined percentage before placing a re-entry order. Similarly, for trailing closes, the bot waits before placing its closing orders until after the price has moved favorably by a threshold percentage and then retraced by a specified percentage. This may result in the bot locking in profits more effectively by exiting positions when the market shows signs of reversing instead of at a fixed distance from average entry price.

Grid and trailing orders may be combined, such that the robot enters or closes a whole or a part of the position as grid orders and/or as trailing orders.

### Forager
The Forager feature dynamically chooses the most volatile markets on which to open positions. Volatility is defined as the mean of the normalized relative range for the most recent 1m candles, i.e. `mean((ohlcv.high - ohlcv.low) / ohlcv.close)`.

### Unstucking Mechanism
Passivbot manages underperforming, or "stuck", positions by realizing small losses over time. If multiple positions are stuck, the bot prioritizes positions with the smallest gap between the entry price and current market price for "unstucking". Losses are limited by ensuring that the account balance does not fall under a set percentage below the past peak balance.  

## Installation

To install Passivbot and its dependencies, follow the steps below.

### Step 1: Clone the Repository

First, clone the Passivbot repository to the local machine:

```sh
git clone https://github.com/enarjord/passivbot.git
cd passivbot
```


### Step 2: Install Rust
Passivbot uses Rust for some of its components. Install Rust by following these steps:

Visit https://www.rust-lang.org/tools/install
Follow the instructions to install Rustup, the Rust installer and version management tool.
After installation, restart the terminal or command prompt.

### Step 3: Create and Activate a Virtual Environment

Create a virtual environment to manage dependencies:

```sh
python3 -m venv venv
```

Activate the virtual environment:

```sh
source venv/bin/activate
```

### Step 4: Install Python Dependencies

Install all the required Python dependencies listed in the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Step 5 (optional): Build Rust Extensions

Passivbot will attempt to build the necessary Rust extensions automatically, but they can also be built manually by navigating to the `passivbot-rust` directory and using `maturin`:

```sh
cd passivbot-rust
maturin develop --release
cd ..
```

If changes in the Rust source are detected, recompilation is needed, which Passivbot will attempt to do automatically when starting. To manually recompile, use the commands given above.

### Step 6: Add API keys

Make a copy of the api-keys template file:

```sh
cp api-keys.json.example api-keys.json
```

Add your keys to api-keys.json.

### Step 7: Run Passivbot

To start the bot with the default settings, run:

```sh
python3 src/main.py -u {account_name_from_api-keys.json}
```

or make a new configuration file, using `configs/template.json` as a template, and start the bot with:


```sh
python3 src/main.py path/to/config.json
```

## Jupyter Lab

Jupyter lab needs to be run in the same virtual environment as the bot. Activate venv (see installation instructions above, step 3), and launch Jupyter lab from the Passivbot root dir with:
```shell
python3 -m jupyter lab
```

## Requirements

- Python >= 3.8
- [requirements.txt](requirements.txt) dependencies

## Pre-optimized configurations

Coming soon...

See also https://pbconfigdb.scud.dedyn.io/

## Documentation:

For more detailed information about Passivbot, see documentation files here: [docs/](docs/)

## Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

## Third Party Links, Referrals and Tip Jar

**Passivbot Manager Service:**  
There is a paid manager service to run Passivbot on the user's behalf:  
www.passivbotmanager.com  

**Passivbot GUI**
A graphical user interface for Passivbot:  
https://github.com/msei99/pbgui

**Referrals:**  
Signing up using these referrals is appreciated:  
https://accounts.binance.com/register?ref=TII4B07C  
https://partner.bybit.com/b/passivbot  
https://partner.bitget.com/bg/Y8FU1W  
https://www.okx.com/join/PASSIVBOT  (20% rebate)  
https://app.hyperliquid.xyz/join/PASSIVBOT  

**Note on Binance**  
To support continued Passivbot development, please use a Binance account which  
1) was created after 2024-09-21 and  
2) either:  
  a) was created without a referral link, or  
  b) was created with referral ID: "TII4B07C".  
                                                                                      
Passivbot receives commissions from trades only for accounts meeting these criteria.  


**BuyMeACoffee:**  
https://www.buymeacoffee.com/enarjord  

**Donations:**  
If the robot is profitable, consider donating as showing gratitude for its development:  

- USDT or USDC Binance Smart Chain BEP20:  
0x4b7b5bf6bea228052b775c052843fde1c63ec530  
- USDT or USDC Arbitrum One:  
0x4b7b5bf6bea228052b775c052843fde1c63ec530  

Bitcoin (BTC) via Strike:  
enarjord@strike.me

## License

Released freely without conditions.
Anybody may copy, distribute, modify, use or misuse for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do whatever without permission from anybody.
