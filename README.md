![Passivbot](docs/images/pbot_logo_full.svg)

# Trading bot running on Bybit, OKX, Bitget, Bitunix, GateIO, Binance, Kucoin, Hyperliquid, WEEX and Lighter

:warning: **Used at one's own risk** :warning:

Latest tagged release: **[v8.1.0](docs/release_notes_v8.1.0.md)**.
These instructions describe the v9 development revision. See
[release status and revision selection](docs/releases.md) for the tagged release and
[Unreleased](CHANGELOG.md#unreleased) for development changes.

> **V9 configuration transition:** this revision accepts the final canonical v9 layout.
> Use the [explicit released-v8 converter](docs/v8_to_v9_migration.md), review semantic
> choices and backtest the result. The v7 strategy and its migration command are removed.

## Overview

Passivbot is a cryptocurrency trading bot written in Python and Rust, intended to require minimal user intervention.  

It operates on perpetual futures derivatives markets, automatically creating and cancelling limit buy and sell orders on behalf of the user. It does not try to predict future price movements or follow trends. Rather, it is a contrarian market maker, using price bands, EMA-derived context, and risk controls to provide resistance to price changes in both directions, thereby "serving the market" as a price stabilizer.

Order planning is computed by a shared Rust orchestrator used by both live trading and backtesting for speed and consistency. Also included is an optimizer, which finds better configurations by iterating thousands of backtests with different candidates, converging on the optimal ones with an evolutionary algorithm.  

Lighter users: see [signer setup and exchange-specific behavior](docs/exchanges/lighter.md).

## Strategy

The default `trailing_martingale` strategy starts with a small entry and can add to a position
as price moves against it, subject to configured exposure and risk limits. Re-entries change the
average entry price, while closes seek a configured profit margin or use trailing confirmation.
Entry sizes, distances, and trailing behavior are configurable.

The `ema_anchor` strategy is also available. See the [bot configuration guide](docs/config.bot.md) for
strategy-specific parameters.

### Trailing Orders
In addition to grid-based entries and closes, Passivbot may be configured to utilize trailing entries and trailing closes.

For trailing entries, the bot waits for the price to move beyond a specified threshold and then retrace by a defined percentage before placing a re-entry order. Similarly, for trailing closes, the bot waits before placing its closing orders until after the price has moved favorably by a threshold percentage and then retraced by a specified percentage. This may result in the bot locking in profits more effectively by exiting positions when the market shows signs of reversing instead of at a fixed distance from average entry price.

Grid and trailing orders may be combined, such that the robot enters or closes a whole or a part of the position as grid orders and/or as trailing orders.

### Forager
The Forager feature dynamically chooses which approved markets may open positions. It ranks eligible markets with configurable weights for quote volume, EMA readiness, 1m log-range volatility and RMS directionality.

### Unstucking Mechanism
Passivbot manages underperforming, or "stuck", positions by realizing small losses over time. If multiple positions are stuck, the bot prioritizes positions with the smallest gap between the entry price and current market price for "unstucking". A continuous net-realized loss memory limits its allowance, with a separate account-wide loss gate for non-panic closes.

## Installation

To install Passivbot and its dependencies, follow the steps below.

Passivbot supports **Python 3.12 and Python 3.14**. Python 3.13 is not supported by the pinned
dependency set.

### Configuration upgrades

Copy the [maintained template](configs/examples/default_trailing_martingale_long.json) to a
private file and review the strategy, market universe, and exposure before use. Follow the
[v8 migration guide](docs/v8_to_v9_migration.md) for supported source versions, explicit spans,
metric formulas, and search-policy acknowledgments. Preserve the original config and matching
checkout for rollback. Normal loading requires a v9 config.

### Step 1: Clone the Repository

First, clone the Passivbot repository to the local machine:

```sh
git clone https://github.com/enarjord/passivbot.git
cd passivbot
```

This checks out the default branch. Use `git clone --branch v9 https://github.com/enarjord/passivbot.git`
to select the v9 development branch explicitly, or follow
[release selection](docs/releases.md) for a fixed tag. Read the docs from the revision you install.


### Step 2: Install Rust
Passivbot uses Rust for some of its components. Install Rust by following these steps:

Visit https://www.rust-lang.org/tools/install
Follow the instructions to install Rustup, the Rust installer and version management tool.
After installation, restart the terminal or command prompt.

### Step 3: Create and Activate a Virtual Environment

Create a virtual environment to manage dependencies:

 **Linux/macOS:**
```sh
# Replace python3.14 with the installed supported executable (python3.12 or python3.14).
PYTHON_BIN=python3.14
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m venv venv
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

### Step 4: Install Passivbot

Choose the install profile that matches your use case:

- **Live-only VPS**: `python3 -m pip install -e .`
- **Backtesting / optimization / research workstation**: `python3 -m pip install -e ".[full]"`
- **Contributing / docs / lint tooling**: `python3 -m pip install -e ".[dev]"`

All profiles build the Rust extension and register the `passivbot` command.

Typical live-only install:

```sh
python3 -m pip install -e .
```

### Step 5 (optional): Build Rust Extensions

Editable installation builds the Rust extension. To rebuild after a Rust change and verify the
loaded artifact against the checkout, follow the [Rust extension runbook](docs/ai/runbooks/rust_extension.md#rebuild-and-verify).

### Step 6: Add API keys

Make a copy of the api-keys template file:

```sh
cp api-keys.json.example api-keys.json
```

Add your keys to `api-keys.json` and restrict access to that file. Keep the file, private configs,
and runtime logs out of commits, public issues, and Docker build contexts.

### Step 7: Run Passivbot

> **Hint:**  
> To ensure cache folder names are Windows-compatible (even outside Windows), set the environment variable `WINDOWS_COMPATIBILITY=1`.   
> This is only required in certain scenarios, e.g., running under Docker (Linux) while mounting the `caches` folder to a Windows host.

Use a reviewed private config with the intended `live.account.user`, explicit side activation,
and an approved coin universe. Inspect it offline with
[`live-config-preflight`](docs/tools.md#monitor-tooling) and backtest it before starting live.

```sh
passivbot live configs/private/live.json
```

To opt into the built-in starter for live trading, add the intended exchange account to your
local `api-keys.json` under the alias `passivbot_default`, then run `passivbot live`. This
starter enables long entries. A live launch can contact the exchange, change account settings,
and place or cancel orders. Use the [offline fake harness](docs/fake_live.md) to test without
an exchange.

With the full install, `passivbot backtest` and `passivbot optimize` also run from the built-in
starter. They use cached market data where available and public data sources otherwise. Suite mode
is opt-in with `--suite`.

The direct-script entrypoints `python3 src/main.py ...`, `python3 src/backtest.py ...`,
and `python3 src/optimize.py ...` also accept the current configuration.

The built-in starter matches the [shipped example](configs/examples/default_trailing_martingale_long.json).
Copy the example to customize it. See [Config Workflow](docs/config_workflow.md) for how bare
commands, explicit config files, and CLI overrides select their settings.

### Logging

Passivbot uses Python's logging module throughout the bot, backtester, and supporting tools.

- Use `--log-level {warning|info|debug|trace}` or `--log-level {0-3}` on `passivbot live` or `passivbot backtest` to adjust verbosity at runtime: `0 = warnings only`, `1 = info`, `2 = debug`, `3 = trace`.
- Use `--verbose` on `passivbot live` to force debug logging (`--log-level debug`).  
- Persist a default by adding a top-level section to your config: `"logging": {"level": 2}`. The CLI flag always overrides the config value for that run.
- `passivbot live` now writes a timestamped logfile under `logs/` by default and refreshes `logs/{user}.log` as a stable alias to the current run. On Windows without symlink privileges, that path is a small pointer file which Passivbot's monitor tooling follows automatically. Control this with `config.logging.persist_to_file`, `config.logging.dir`, and the optional rotation settings in `config.logging`.
- CandlestickManager and other subsystems inherit the chosen level so EMA warm-up, data fetching, and cache behaviour can be inspected consistently.

### Running Multiple Bots

Running several Passivbot instances on one machine is supported when each uses a separate
exchange account or subaccount. Do not run competing bots on the same account; see
[concurrent bot protection](docs/live.md#concurrent-passivbot-protection). Each process shares the same on-disk OHLCV cache, and the candlestick manager now uses short-lived, self-healing locks with automatic stale cleanup so that one stalled process cannot block the rest. No manual deletion of lock files is required; the bot removes stale locks on startup and logs whenever a lock acquisition times out.

## Jupyter Lab

JupyterLab is an optional separate install. Activate the bot virtual environment, then install and
launch it from the repository root:

```shell
python3 -m pip install jupyterlab
python3 -m jupyter lab
```

## Requirements

- Python 3.12 or 3.14 (Python 3.13 is not supported)
- `python3 -m pip install -e .` for live trading only
- `python3 -m pip install -e ".[full]"` for backtesting, optimization, downloader, and advanced tools
- `python3 -m pip install -e ".[dev]"` for contributor tooling on top of the full install

## Example configurations

Start from a maintained [example config](configs/examples/) and follow the
[config workflow](docs/config_workflow.md) to backtest and tune it for your use case.
Community configurations are also available at https://pbconfigdb.scud.dedyn.io/.

## Documentation

For more detailed information about Passivbot, see the [documentation index](docs/index.md).

Useful entry points:

- [Release status and upgrades](docs/releases.md)
- [Installation](docs/installation.md)
- [Config workflow](docs/config_workflow.md)
- [Configuration reference](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Backtesting](docs/backtesting.md)
- [Optimizing](docs/optimizing.md)
- [Metrics reference](docs/metrics.md)
- [Equity Hard Stop Loss](docs/equity_hard_stop_loss.md)
- [Monitor output](docs/monitor.md)

## Support

[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/QAF2H2UmzZ)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/passivbot_futures)

## Third Party Links, Referrals and Tip Jar

**Hyperliquid Reference Vault**
Passivbot's reference long-only config profile running on a Hyperliquid Vault:
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
https://www.weex.com/en/register?vipCode=ppc8

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
