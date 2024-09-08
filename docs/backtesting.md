# Backtesting

Passivbot includes a backtester which will simulate the bot's behavior on past price data. Historical 1m candlestick data is automatically downloaded and cached for all coins.

## Usage

```shell
python3 src/backtest.py
```
Or
```shell
python3 src/backtest.py path/to/config.json
```
If no config is specified, it will default to `configs/template.json`

## Backtest Results

Metrics and plots are dumped to `backtests/{exchange}/`.