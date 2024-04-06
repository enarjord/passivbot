# Passivbot Multisymbol

Passivbot may be run in multi symbol mode, where one bot handles multiple symbols in the same instance.  

- only recursive grid mode is supported.
- only USDT margined perpetual futures is supported.
- supported exchanges: Bitget, OKX, Bybit, Binance, BingX and Hyperliquid.
- backtesting and optimizing is supported with backtest_multi.py and optimize_multi.py.
- bot will automatically set positions not under active management to graceful stop mode.

## Usage

Use configs/live/example_config.hjson as template and run:
```shell
python3 passivbot_multi.py configs/live/{config_name}.hjson
```  

## Multi symbol auto unstuck

- if a position is stuck, bot will use profits made on other markets to realize losses for the stuck position.
- if multiple positions are stuck, the stuck position whose price action distance is the lowest is selected for unstucking.
- each live config's individual timer based auto unstuck mechanism is disabled.
- only recursive grid configs are supported.
- bot behavior remains the same otherwise

## Backtesting

Modify `configs/backtest/multi.hjson` and run  
```shell
python3 backtest_multi.py
```

## Optimizing

Not yet supported.