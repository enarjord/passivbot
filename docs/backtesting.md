# Backtesting

Passivbot ships with a backtester that replays historical 1 minute candles. The
backtester prepares data from the canonical v2 OHLCV store under `caches/ohlcvs/`.
When data is missing there, it imports any matching legacy daily shards before making
targeted remote exchange requests.

Backtesting requires the full install profile:

```shell
python3 -m pip install -e ".[full]"
```

> **GateIO cache note:** If you have existing legacy GateIO OHLCV data in
> `caches/ohlcv/gateio`, delete the folder before a fresh run so Passivbot rebuilds the
> v2 store with base-volume-normalized data.

> **GateIO history note:** GateIO's public 1m OHLCV endpoint only serves a recent window of roughly 10,000 candles. Older GateIO backtests need `backtest.ohlcv_source_dir` data or candles sourced from another exchange; Passivbot marks older GateIO 1m spans unavailable instead of repeatedly retrying rejected requests.

### External OHLCV source dir

You can point `backtest.ohlcv_source_dir` to a pre-populated OHLCV tree. The loader looks under:

```
<ohlcv_source_dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npz
<ohlcv_source_dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npy
```

`<coin_or_symbol>` accepts base coins (e.g., `ETH`) or CCXT-style symbol dirs (e.g., `ETH_USDC:USDC`).

For `.npz` files, the archive must contain a `candles` key with a structured NumPy array having fields `ts` (int64 timestamp), `o` (open), `h` (high), `l` (low), `c` (close), `bv` (base volume). Timestamps should be in milliseconds. For `.npy` files, the array should have columns `[timestamp, open, high, low, close, volume]`.

## Usage

```shell
passivbot backtest
```
Or
```shell
passivbot backtest path/to/config.json
```
If no config is specified, backtesting starts from the in-code schema defaults in `src/config/schema.py`.
The example config `configs/examples/default_trailing_grid_long_npos7.json` mirrors those defaults exactly.
See [Config Workflow](config_workflow.md) for the recommended way to copy and customize configs.

## Backtest Results

Standalone runs write metrics and plots to `backtests/{exchange}/timestamp/`. Suite runs collect everything under `backtests/suite_runs/<timestamp>/<scenario_label>/` and add a top-level `suite_summary.json`.

Each run also writes `dataset.json`, which points to the exact HLCV cache files used for that
run. In notebooks or ad-hoc Python analysis you can load the full artifact bundle like this:

```python
from backtest_artifacts import load_backtest_artifact_workspace

workspace = load_backtest_artifact_workspace("backtests/combined/2026-04-21T19_35_10")
globals().update(workspace)

btc_candles = candles_for_coin("BTC")
fig = plot_fills_for_coin("BTC", start_date="2026-04-01T00:00:00")
```

The workspace includes `config`, `analysis`, `fills`, `balance_and_equity`, `hlcvs`,
`timestamps`, `btc_usd_prices`, `coins`, `market_settings`, `candles_for_coin`, and
`plot_fills_for_coin`.

## HLCV Data Contract

Backtest and optimize data preparation follows one deterministic order:

1. read existing v2 chunks in `caches/ohlcvs/`
2. import matching legacy raw OHLCV shards when they are present
3. fetch only the remaining missing or invalid windows from the exchange

Missing exchange history is recorded as coverage metadata rather than treated as cache
corruption. Newly listed coins may start after the requested start date, and delisted
or migrated coins may end before the requested end date. Small verified source-side
internal gaps may be filled within `backtest.gap_tolerance_ohlcvs_minutes`; larger
internal gaps are repaired or excluded from the returned tradable window rather than
materialized as one continuous tradable span. Backtests still fail when there is no
usable data inside the requested range, no BTC benchmark data, no tradable candles
after warmup, missing/mismatched chunk checksums, conflicting duplicate candles, or
malformed timestamps/OHLCV rows that cannot be normalized.

Prepared final caches under `caches/hlcvs_data/` require a valid manifest. Old
manifest-less final caches are rebuilt. Explicit `backtest.hlcvs_data_dir` override
datasets also require valid manifests and checksums.

## Backtest CLI args

- `-dp` to disable individual coin plotting.
- `--suite [y/n]` to override `backtest.suite_enabled` (omit the value to enable, e.g. `--suite`).
- `--scenarios label1,label2,...` to run only specific scenarios by label (implies `--suite y`).
- `--suite-config path/to/overrides.json` to merge an additional suite definition onto the base config. Useful when you want to keep suite definitions outside the main config file.

The canonical default profile keeps `backtest.suite_enabled = false`. A normal backtest run is
therefore a single run unless you explicitly enable suite mode in the config or via `--suite`.

For a comprehensive list of CLI args:
```shell
passivbot backtest -h
```

## Pre-download OHLCV

Use `passivbot download` when you want to warm the OHLCV cache before a long backtest or optimize
run. It uses the same `backtest.start_date`, `backtest.end_date`, `backtest.exchanges`, and
approved-coin selection flow as the backtester, but stops after preparing cache data.

```shell
passivbot download configs/examples/default_trailing_grid_long_npos7.json
passivbot download --start-date 2025-01-01 --end-date 2025-02-01 --exchanges binance,bybit
```

## Suite Runs

Suite mode evaluates multiple scenario slices in one invocation. Configuration uses a flattened structure directly under `backtest`:

```json
"backtest": {
  "suite_enabled": true,
  "scenarios": [...],
  "aggregate": {"default": "mean"},
  "exchanges": ["binance", "bybit"],
  ...
}
```

Suite mode is off by default in the hardcoded schema and in
`configs/examples/default_trailing_grid_long_npos7.json`. Enable it only when you deliberately
want multi-scenario evaluation.

Each scenario may override:

- `label`: directory name inside `backtests/suite_runs/<timestamp>/`
- `coins`/`ignored_coins`
- `start_date`/`end_date`
- `exchanges`: restricts which exchanges' data the scenario can see
- `coin_sources`: force a specific exchange for individual coins (see below)
- `overrides`: arbitrary config path overrides (e.g., `{"live.long.we": 0.5}`)

Top-level suite keys (directly under `backtest`):

- `suite_enabled`: master toggle for suite mode (overridable via `--suite [y/n]`)
- `scenarios`: list of scenario dictionaries
- `aggregate`: how to combine per-scenario metrics (default: `{"default": "mean"}`)

During a suite run Passivbot prepares master OHLCV datasets only for the scenario
windows that consume them. The combined dataset uses the union of combined scenarios;
single-exchange datasets use the union of scenarios restricted to that exchange. Results
are written to:

```
backtests/suite_runs/<timestamp>/<scenario_label>/
```

Every suite also receives a `suite_summary.json` containing per-scenario metrics and the aggregated statistics defined in `backtest.aggregate`.
Each scenario's entry exposes `metrics.stats[metric] = {mean,min,max,std}` so you can inspect exchange-combined performance without digging through `analysis.json` files.

See [Suite Examples](suite_examples.md) for practical configurations including exchange
comparisons, date range scenarios, and parameter sensitivity testing.

### Data Strategy

The data strategy is determined implicitly by the number of exchanges configured:

- **Single exchange** (1 exchange in `backtest.exchanges`): Data is fetched from that exchange only. Scenario labels include the exchange suffix (e.g., `base/binance`).
- **Combined exchanges** (>1 exchanges): Data is combined from all listed exchanges, selecting the best feed per coin based on coverage and quality. Scenario labels do not include an exchange suffix.

Per-scenario `exchanges` overrides can narrow down which exchanges a scenario sees, but cannot add exchanges not in the base config.

### Comparing Exchanges

To compare performance across exchanges, define one scenario per exchange using the scenario-level `exchanges` field:

```json
"backtest": {
  "suite_enabled": true,
  "aggregate": {"default": "mean"},
  "exchanges": ["binance", "bybit", "gateio"],
  "scenarios": [
    {"label": "binance_only", "exchanges": ["binance"]},
    {"label": "bybit_only", "exchanges": ["bybit"]},
    {"label": "gateio_only", "exchanges": ["gateio"]}
  ]
}
```

Each scenario will then run against only that exchange's data, and `suite_summary.json` will contain separate `metrics.stats` blocks for each exchange scenario.

## Coin Sources

When multiple exchanges are configured (`backtest.exchanges` has >1 entry), the backtester combines OHLCV data and automatically selects the "best" exchange per coin (based on coverage, gaps, and relative volume). To override that choice, add a `coin_sources` map:

```json
"backtest": {
  "coin_sources": {
    "BTC/USDT:USDT": "binance",
    "SOL/USDT:USDT": "bybit"
  }
}
```

Suite scenarios can add more overrides under `coin_sources`; conflicts between scenarios are rejected so every run uses a single consistent exchange assignment.

## Supported Exchanges

The backtester supports OHLCV data from the following exchanges:
- **binance** - Binance USDT-M Futures
- **bybit** - Bybit USDT Perpetuals
- **bitget** - Bitget USDT Perpetuals
- **gateio** - Gate.io USDT Perpetuals

The canonical default template currently uses `binance` and `bybit`. Add `bitget` and/or `gateio`
explicitly in `backtest.exchanges` when you want them included.

## Exchange Name Conventions

Cache paths and output directories use standard exchange names (e.g., `binance`, `bybit`, `gateio`). The ccxt-specific suffixes (`binanceusdm`, `bybitusdt`) are only used internally when communicating with the exchange API. Always use the short names in your configuration files.
