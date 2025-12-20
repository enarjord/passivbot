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

Standalone runs write metrics and plots to `backtests/{exchange}/timestamp/`. Suite runs collect everything under `backtests/suite_runs/<timestamp>/<scenario>/<exchange>/` and add a top-level `suite_summary.json`.

## Backtest CLI args

- `-dp` to disable individual coin plotting.
- `--suite [y/n]` to override `backtest.suite.enabled` (omit the value to enable, e.g. `--suite`).
- `--suite-config path/to/overrides.json` to merge an additional suite definition onto the base config. Useful when you want to keep suite definitions outside the main config file.

For a comprehensive list of CLI args:
```shell
python3 src/backtest.py -h
```

## Suite Runs

`backtest.suite` lets you evaluate multiple scenario slices in one invocation. Each scenario may override:

- `label`: directory name inside `backtests/suite_runs/<timestamp>/`
- `coins`/`ignored_coins`
- `start_date`/`end_date`
- `exchanges`: restricts which exchanges’ data the scenario can see
- `coin_sources`: force a specific exchange for individual coins (see below)

Top-level suite keys:

- `enabled`: toggles suite mode (overridable via `--suite [y/n]`)
- `include_base_scenario`: prepend an auto-generated scenario that mirrors the base config
- `base_label`: name for the base scenario when included
- `scenarios`: list of dictionaries as described above

If `include_base_scenario=true`, the suite will run **one extra scenario** using the base backtest
settings (date range, coins, exchanges, etc.). For example, two listed scenarios + `include_base_scenario=true`
produces three scenario results.

During a suite run Passivbot prepares one master OHLCV dataset that spans the union of all scenario date ranges and coins, then slices it per scenario so repeated downloads are avoided. Results are written to:

```
backtests/suite_runs/<timestamp>/<scenario_label>/<exchange>/
```

Every suite also receives a `suite_summary.json` containing per-scenario metrics and the aggregated statistics defined in `backtest.suite.aggregate`.
Each scenario’s entry exposes `metrics.stats[metric] = {mean,min,max,std}` so you can inspect exchange-combined performance without digging through `analysis.json` files.

See `configs/examples/suite_example.json` for a minimal multi-scenario setup.

### Comparing Exchanges

To compare performance across exchanges (similar to older “per-exchange analyses”), define one scenario per exchange by using the scenario-level `exchanges` field:

```json
"backtest": {
  "suite": {
    "enabled": true,
    "include_base_scenario": false,
    "aggregate": {"default": "mean"},
    "scenarios": [
      {"label": "binance_only", "exchanges": ["binance"]},
      {"label": "bybit_only", "exchanges": ["bybit"]}
    ]
  }
}
```

Each scenario will then run against only that exchange’s data, and `suite_summary.json` will contain separate `metrics.stats` blocks for `binance_only` and `bybit_only`.

## Coin Sources

When `backtest.combine_ohlcvs` is `true`, the downloader automatically selects the “best” exchange per coin (based on coverage, gaps, and relative volume). To override that choice, add a `coin_sources` map:

```json
"backtest": {
  "coin_sources": {
    "BTC/USDT:USDT": "binance",
    "SOL/USDT:USDT": "bybit"
  }
}
```

Suite scenarios can add more overrides under `coin_sources`; conflicts between scenarios are rejected so every run uses a single consistent exchange assignment. Passivbot currently ships Binance and Bybit archives out of the box. Support for other exchanges exists in the codebase but those feeds may be stale unless you fetch them yourself.
