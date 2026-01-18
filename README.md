![Passivbot](docs/images/pbot_logo_full.svg)

# Trading bot running on Bybit, OKX, Bitget, GateIO, Binance, Kucoin and Hyperliquid

:warning: **Used at one's own risk** :warning:

v7.6.1


## Overview

Passivbot is a cryptocurrency trading bot written in Python and Rust, intended to require minimal user intervention.  

It operates on perpetual futures derivatives markets, automatically creating and cancelling limit buy and sell orders on behalf of the user. It does not try to predict future price movements, it does not use technical indicators, nor does it follow trends. Rather, it is a contrarian market maker, providing resistance to price changes in both directions, thereby "serving the market" as a price stabilizer.  

Order planning is computed by a shared Rust orchestrator used by both live trading and backtesting for speed and consistency. Also included is an optimizer, which finds better configurations by iterating thousands of backtests with different candidates, converging on the optimal ones with an evolutionary algorithm.  

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

Passivbot requires **Python 3.12**. Earlier versions are not supported.

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

 **Linux/macOS:**
```sh
python3 -m venv venv
```

 **Windows (Command Prompt or PowerShell):**
```cmd
py -3.12 -m venv venv
```

Activate the virtual environment:

 **Linux/macOS:**
```sh
source venv/bin/activate
```

 **Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

 **Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
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

> **Hint:**  
> To ensure cache folder names are Windows-compatible (even outside Windows), set the environment variable `WINDOWS_COMPATIBILITY=1`.   
> This is only required in certain scenarios, e.g., running under Docker (Linux) while mounting the `caches` folder to a Windows host.

To start the bot with the default settings, run:

```sh
python3 src/main.py -u {account_name_from_api-keys.json}
```

or make a new configuration file, using `configs/template.json` as a template, and start the bot with:


```sh
python3 src/main.py path/to/config.json
```

### Logging

Passivbot uses Python's logging module throughout the bot, backtester, and supporting tools.  
- Use `--debug-level {0-3}` (alias `--log-level`) on `src/main.py` or `src/backtest.py` to adjust verbosity at runtime: `0 = warnings only`, `1 = info`, `2 = debug`, `3 = trace`.  
- Use `--verbose` on `src/main.py` to force debug logging (`--log-level debug`).  
- Persist a default by adding a top-level section to your config: `"logging": {"level": 2}`. The CLI flag always overrides the config value for that run.
- CandlestickManager and other subsystems inherit the chosen level so EMA warm-up, data fetching, and cache behaviour can be inspected consistently.

### Running Multiple Bots

Running several Passivbot instances against the same exchange on one machine is supported. Each process shares the same on-disk OHLCV cache, and the candlestick manager now uses short-lived, self-healing locks with automatic stale cleanup so that one stalled process cannot block the rest. No manual deletion of lock files is required; the bot removes stale locks on startup and logs whenever a lock acquisition times out.

## Jupyter Lab

Jupyter lab needs to be run in the same virtual environment as the bot. Activate venv (see installation instructions above, step 3), and launch Jupyter lab from the Passivbot root dir with:
```shell
python3 -m jupyter lab
```

## Requirements

- Python >= 3.12
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

**Hyperliquid Reference Vault**
Passivbot's default template config running on a Hyperliquid Vault:  
https://app.hyperliquid.xyz/vaults/0x490af7d4a048a81db0f677517ed6373565b42349

**Passivbot GUI**
A graphical user interface for Passivbot:  
https://github.com/msei99/pbgui

**Referrals:**  
Signing up using these referrals is appreciated:  
https://accounts.binance.com/register?ref=TII4B07C  
https://partner.bybit.com/b/passivbot  
https://partner.bitget.com/bg/Y8FU1W  
https://www.okx.com/join/PASSIVBOT  
https://app.hyperliquid.xyz/join/PASSIVBOT  
https://www.kucoin.com/r/broker/CX8QZQJX  

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
- Zcash (ZEC):  
u1jlans93rrqusqx2wp5020aezyt0q22l4tuy7ezkna06fuyaa2gxzremf50wsj3k83a4cm0cncs6zt9urlpte7a3nzvq992z48jxzem455acmhmhhwfwjcjwl8z79vlznla0r3jln6ety565254h96whnllcmepmpqu3ft9hxtqvkn0m7  

Bitcoin (BTC) via Strike:  
enarjord@strike.me

## License
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org/>
