# Backtesting

Passivbot ships with a backtester that replays historical 1 minute candles. When a coin isn’t cached locally, the backtester fetches data from the exchange archives (see `src/downloader.py`) and stores it under `historical_data/` for reuse.

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

## Backtest CLI args

- `-dp` to disable individual coin plotting.
- `-co` to combine the ohlcv data from multiple exchanges into a single array. Otherwise, backtest for each exchange individually.

For a comprehensive list of CLI args:
```shell
python3 src/backtest.py -h
```
