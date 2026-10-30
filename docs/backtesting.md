# Backtesting

Passivbot ships with a backtester that replays historical 1 minute candles. By default,
the backtester prepares data from the canonical v2 OHLCV store under `caches/ohlcvs/`.
When data is missing there, it imports any matching legacy daily shards before making
targeted remote exchange requests. If `backtest.ohlcv_source_dir` is set, Passivbot
reads that caller-managed OHLCV tree directly instead of importing it into the v2 raw
store first.

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

Explicit source directories are treated as caller-managed data. Backtest and optimizer
preparation read them directly and still write the final prepared HLCV dataset cache for
the run, but they do not mirror those raw daily files into `caches/ohlcvs/`.

For `.npz` files, the archive must contain a `candles` key with a structured NumPy array having fields `ts` (int64 timestamp), `o` (open), `h` (high), `l` (low), `c` (close), `bv` (base volume). Timestamps should be in milliseconds. For `.npy` files, the array should have columns `[timestamp, open, high, low, close, volume]`.

## BTC benchmark source

Combined backtests and optimizations try BTC price history from the configured
`backtest.exchanges` in order, except that Binance futures (`binance` /
`binanceusdm`) keeps first priority when configured. Duplicate venues are tried
once. When Binance is not configured, it is the final fallback after the configured
venues. The selected venue is logged and retained as `btc_source_exchange` in
preparation metadata. Candidates must cover both boundaries of the requested BTC
range; shorter histories are skipped instead of extending their first or last price
across unavailable history. Prepared caches from the previous source policy are
rebuilt automatically.

This priority applies in both online and offline modes, including explicit
`backtest.ohlcv_source_dir` inputs. Online preparation may fetch missing data for
the current candidate before trying the next venue; it does not search every
venue's cache before downloading. Offline preparation uses local inputs only and
fails if no candidate supplies the required benchmark. Changing the selected BTC
source can change BTC-denominated metrics and simulations using BTC collateral.

## Offline preparation

Set `backtest.offline: true`, or pass `--offline y`, for backtests and optimization
(including suites and GPU optimization):

```bash
passivbot backtest path/to/config.json --offline y
passivbot optimize path/to/config.json --offline y
```

Offline preparation never refreshes market metadata, listing timestamps, exchange
archives, or candles. Valid cached metadata is accepted regardless of age; its age
and fingerprint are logged. Online remains the default; live trading is unaffected.

Prepare the requested date range on a connected machine, then copy these caches:

- `caches/ohlcvs/`, including its coverage catalog and candle chunks
- `caches/<exchange>/markets.json` and `first_timestamps.json` for each data or
  market-settings exchange
- `caches/first_ohlcv_timestamps_unified*`, including the resolver version and
  exchange-specific symbol provenance files
- Any configured `backtest.ohlcv_source_dir` legacy shards

Include the optimizer's maximum warmup range and BTC benchmark candles. Combined
runs also need inputs for candidate exchanges used in source selection and volume
normalization. Copy a consistent cache snapshot while no downloader is writing it;
SQLite catalog paths must refer to the copied chunks on the receiving machine.

Missing or incompatible metadata, incomplete candles, or unconfirmed range edges
fail with an input/range diagnostic. Refresh those inputs on the connected machine
and copy them again. Offline mode does not replace missing prices or silently shorten
a requested range. Existing confirmed listing boundaries and supported internal-gap
rules still apply. Use a fixed end date for reproducible runs.

Older prepared HLCV caches are verified from raw caches once. Subsequent runs can
reuse the verified prepared dataset. Its manifest retains metadata and consumed-range
fingerprints under `preparation.offline_snapshot`. An explicit `--hlcvs-data-dir`
override must refer to such a verified dataset. `--force-refetch-gaps` conflicts with
offline mode.

## Usage

```shell
passivbot backtest
```
Or
```shell
passivbot backtest path/to/config.json
```
If no config is specified, backtesting starts from the in-code schema defaults in `src/config/schema.py`.
The example config `configs/examples/default_trailing_martingale_long.json` mirrors those defaults exactly.
See [Config Workflow](config_workflow.md) for the recommended way to copy and customize configs.

## Backtest Results

The terminal summary aligns canonical metric names and values in a table, using five
significant digits and `%` for fraction-valued metrics. Ratios retain their raw scale.
Explicitly disabled sides and HSL omit their zero-valued rows when no activity was observed;
enabled features still show zero activity, and unknown applicability remains visible.
Infinite, undefined and invalid values retain their distinct statuses, with reasons beside
the metric. Counts and unfinished recovery tails appear as compact support information.
This presentation does not round or filter `analysis.json`, or change optimizer metrics.

Standalone runs write metrics and plots to `backtests/{exchange}/<session>/`. Suite runs
collect results under `backtests/suite_runs/<session>/` with a top-level `suite_summary.json`.
Session names use this date-sortable UTC format (the suite component appears only for suites):

```text
<timestamp>_<coins>_<source>_<days>days_[suite-<N>sc_]setup-<12 hex>_run-<8 hex>
2026-10-06T14_50_56Z_XMR_combined_1737days_suite-12sc_setup-a71c093be642_run-99449e93
```

Single coins always appear by name. Sets of up to six coins are sorted and joined when their
label fits in 80 characters; larger sets use `<N>_coins`. A suite's coin label covers all
actually evaluated scenarios, and its day count is their overall date envelope, including gaps.
Unsafe or long coin/scenario labels are converted to bounded filesystem components with a
hash suffix; original identifiers remain in configuration and metadata.

`setup-` fingerprints canonical effective execution inputs, resolved dates, loaded market data,
market settings and verified evaluator implementation. `run-` distinguishes individual executions;
full identifiers and setup inputs are saved in `session.json`. Equivalent fingerprints indicate
equivalent recorded setup inputs, not a guarantee of identical results across environments.
Directory creation is exclusive and retries a run-ID collision.

A scenario producing one result writes directly to `<session>/<scenario>/config.json`,
`analysis.json`, `dataset.json` and the other usual artifacts. A scenario producing multiple
independent results uses `<session>/<scenario>/<exchange>/config.json`. There is no inner
timestamp directory. `suite_summary.json` has `layout_version: 2`, actual scenario `started_at`,
`completed_at` and elapsed seconds, and an `artifacts` map of exact paths relative to the suite
root. Use that map to discover results instead of reconstructing filenames.

This layout changes newly generated paths. Existing results are not renamed; there is no
legacy writer option. Scenario `output_path` is now relative to the suite root. Embedding callers
may supply an explicit empty `suite_output_root`; nonempty roots are rejected to prevent mixing runs.

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

### Saved facts and reports

Normal backtests retain exact observations independently of their plots:

- `result.json` versions the result format and identifies its factual artifacts.
- `balance_and_equity.npz` stores integer millisecond clocks and float64 equity/cash
  columns at every native observation. Missing cash observations remain missing.
- `fills.csv`, `analysis.json`, config, dataset and session metadata remain available.
- `hsl_report.json` holds native policy, summaries, lifecycle events and executed-stop
  details. `hsl_trace.npz` stores ordered native numeric plot facts automatically.
- `timings.json` separates startup, preparation, identity, simulation, conversion,
  export and rendering. `reports/report.json` records report selection and completion
  or failure separately from saved simulation facts.

Standalone exchange runs attribute shared startup plus that exchange's measured
preparation and processing; concurrent preparation and earlier result rendering do
not inflate another exchange's preparation stage. Suite artifacts label timings
limited to post-processing explicitly.

The default plots are `reports/equity.png` and, when HSL is enabled,
`reports/hsl.png`. Equity uses log scale by default. A nonpositive observation causes
an explicitly labelled linear fallback so losses are not hidden. Drawdown is computed
from full-resolution strategy equity before view reduction; the original peak and
trough timestamps are retained. Plot sampling never changes native metric support.

```bash
passivbot backtest CONFIG --equity-scale linear
passivbot backtest CONFIG --report-btc
passivbot backtest CONFIG --plots all --equity-csv
passivbot backtest CONFIG --plots none
```

Plot groups are `equity`, `hsl`, `hsl_episodes`, `coin_fills` and `twe`; `all` and
`none` select the complete set or disable rendering. The legacy `--disable_plotting`
switch still suppresses its named groups. Cumulative PnL is no longer generated.
BTC-denominated views, full-history fill plots, stop zooms and dense equity CSV are
optional. The TWE view labels its hourly fill-event averaging population explicitly.

Reports can be regenerated without running the strategy or contacting an exchange:

```bash
passivbot tool backtest-report RUN --equity-scale linear
passivbot tool backtest-report RUN --plots hsl,hsl_episodes --scope BTC:long
passivbot tool backtest-report RUN --plots coin_fills --output-dir RUN/detail
passivbot tool backtest-report RUN --plots none --equity-csv
passivbot tool backtest-dash RUN
```

The analyzer binds only to localhost and prints its URL. It opens metadata first,
then reads requested equity/HSL columns and pages the fill ledger. It supports time
intervals, native HSL controller selection, executed-stop inspection, log/linear
equity and optional BTC denomination. It serves only the selected result and bundled
UI resources, rejects foreign Host/Origin headers, and has no write or exchange API.

The binary files contain no Python objects and are read with pickles disabled.
Native float64 HSL and accounting precision are preserved. Trace
columns retain same-timestamp sequence, observation phase, exposure, raw/EMA/trigger
signals, remembered currency drawdown, budget, UPNL and controller state. Flat idle
periods are omitted, while the first inactive boundary after activity is retained;
plots show those idle intervals as gaps. Lifecycle transitions remain ordered even
when trigger, flatten and restart share a timestamp. Unused sides do not generate
an all-GREEN default plot. An enabled controller with no exposure gets an explicit
empty-observation report rather than an invented zero signal.

`backtest.hsl_detailed_report` retains its opt-in verbose diagnostic row dictionaries;
normal plots do not require it. Metrics-only optimizer evaluations retain neither
plot traces nor verbose rows. Older CSV results and detailed JSON HSL reports remain
readable. A legacy result without native HSL samples cannot supply an exact HSL plot;
its absence is reported explicitly. Full candle/fill plots additionally require the
verified dataset referenced by `dataset.json`; missing caches are not downloaded by
the report tools.

For lazy Python access, use `backtest_results.ResultStore(RUN)`. Its `equity`,
`fills`, `hsl` and `hsl_trace` methods load their respective facts on demand. The
existing `load_backtest_artifact_workspace` remains an eager notebook convenience.
Dense equity CSV is an explicit export under the report output directory and is
not a second default copy of the binary history.

## HLCV Data Contract

Direct or prepared backtest inputs must contain finite high, low, and close prices throughout
each symbol's declared valid window. Internal all-NaN gaps are rejected before simulation;
missing prices never turn a held position's unrealized PnL into zero. Missing rows before a
symbol's listing or after its declared end remain permitted while it is flat. A position still
held when its valuation candle becomes unavailable rejects the run. These checks also apply to
GPU preparation and do not replace the normal data materializer's gap repair policy.

Without `backtest.ohlcv_source_dir`, backtest and optimize data preparation follows one
deterministic order:

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

Final cache identity follows the direction-agnostic data payload: the sorted union
of enabled long and short coins, date window, exchanges, source assignments,
the side age-policy pair, preparation version, and gap tolerance. Long-only, short-only, and both-side runs
therefore reuse the same prepared cache when their effective coin union and other
data inputs match. Age no longer crops the shared candle history: inception is retained as a factual market input,
and Rust applies each side's initial-entry gate at its decision timestamp. Unknown inception fails
a positive age gate without discarding otherwise usable warmup data. Automatic cache recovery
is scoped to coins approved and not ignored on enabled positive-age sides. Optimizer acquisition
uses its reachable-side configuration. Suite preparation resolves source policy once per context
and reuses that frozen membership during execution, including explicit same-file replacements;
zero-age or disabled contexts do not require unused facts from other shared dataset columns.
Side membership remains run-specific; manifests retain the
build request for provenance and explicit dataset replay. Warmup is checked
separately, so a cache with greater warmup coverage may satisfy a smaller request.

### Override final HLCV datasets

Use `backtest.hlcvs_data_dir` or `--hlcvs-data-dir` to replay a specific prepared
dataset under `caches/hlcvs_data/` instead of resolving the dataset from the current
config hash. Override datasets are accepted only when their manifest verifies every
required artifact: `hlcvs`, `timestamps`, `btc_usd_prices`, `coins`, and
`market_specific_settings`.

`backtest.hlcvs_data_override_mode` controls how the current config is matched to
the override dataset:

- `intersection` (default): keep the current config's requested coins and date
  window, clipped to the verified override dataset where necessary. This is the
  conservative mode for rerunning a strategy against a known cache without
  expanding the universe from the cache contents.
- `dataset`: adopt the override dataset's effective coins and timestamp window.
  This is useful for exact artifact replay when the cache itself should define the
  run universe and date range.

## Backtest CLI args

- `-dp` to disable individual coin plotting.
- `--suite [y/n]` to override `backtest.suite_enabled` (omit the value to enable, e.g. `--suite`).
- `--scenarios label1,label2,...` to run only specific scenarios by label (implies `--suite y`).
- `--suite-config path/to/overrides.json` to merge an additional suite definition onto the base config. Useful when you want to keep suite definitions outside the main config file.
- `--hlcvs-data-dir path/to/cache` to replay a verified final HLCV dataset.
- `--hlcvs-data-override-mode intersection|dataset` to choose whether the current
  config or the override dataset defines the effective coin/date universe.

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
passivbot download configs/examples/default_trailing_martingale_long.json
passivbot download --start-date 2025-01-01 --end-date 2025-02-01 --exchanges binance,bybit
```

## Suite Runs

Suite mode evaluates multiple scenario slices in one invocation. Configuration uses a flattened structure directly under `backtest`:

```json
"backtest": {
  "suite_enabled": true,
  "scenarios": [...],
  "reducer": {"default": "mean"},
  "exchanges": ["binance", "bybit"],
  ...
}
```

Suite mode is off by default in the hardcoded schema and in
`configs/examples/default_trailing_martingale_long.json`. Enable it only when you deliberately
want multi-scenario evaluation.

Each scenario may override:

- `label`: directory name inside `backtests/suite_runs/<session>/`
- `coins`/`ignored_coins`
- `start_date`/`end_date`
- `exchanges`: restricts which exchanges' data the scenario can see
- `coin_sources`: force a specific exchange for individual coins (see below)
- `overrides`: arbitrary config path overrides (e.g., `{"live.long.we": 0.5}`)

Top-level suite keys (directly under `backtest`):

- `suite_enabled`: master toggle for suite mode (overridable via `--suite [y/n]`)
- `scenarios`: list of scenario dictionaries
- `reducer`: how to combine per-scenario metrics (default: `{"default": "mean"}`)

During a suite run Passivbot prepares master OHLCV datasets only for the scenario
windows that consume them. The combined dataset uses the union of combined scenarios;
single-exchange datasets use the union of scenarios restricted to that exchange. Results
are written to:

```
backtests/suite_runs/<session>/<scenario_label>/
```

Every suite also receives a `suite_summary.json` containing per-scenario metrics and the aggregated statistics defined in `backtest.reducer`.
Each scenario's entry exposes `metrics.stats[metric] = {mean,min,max,std}` so you can inspect exchange-combined performance without digging through `analysis.json` files.

See [Suite Examples](suite_examples.md) for practical configurations including exchange
comparisons, date range scenarios, and parameter sensitivity testing.

### Data Strategy

The data strategy is determined implicitly by the number of exchanges configured:

- **Single exchange** (1 exchange in `backtest.exchanges`): Data is sourced from that exchange only. A single-result scenario writes directly into its scenario directory.
- **Combined exchanges** (>1 exchanges): Data is combined from all listed exchanges, selecting the best feed per coin based on coverage and quality. Exchange-restricted scenarios can produce independent exchange results; the summary artifact map records their exact locations.

Per-scenario `exchanges` overrides can narrow down which exchanges a scenario sees and can also require extra exchanges that are not listed in the top-level base config. Passivbot expands the prepared dataset set to include every scenario-required exchange before running the suite.

### Comparing Exchanges

To compare performance across exchanges, define one scenario per exchange using the scenario-level `exchanges` field:

```json
"backtest": {
  "suite_enabled": true,
  "reducer": {"default": "mean"},
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
- **gateio** - Gate.io USDT Perpetuals, with the public-history window caveat above

Experimental or narrower-coverage sources:

- **kucoin** - KuCoin Futures archive/CCXT data path. Use `kucoin` in configs and cache paths; Passivbot converts to CCXT's `kucoinfutures` ID internally. Treat KuCoin backtest data as experimental until your intended coins/date windows pass a real data smoke.
- **hyperliquid** - Supported for live and metadata paths; broader historical backtest coverage depends on the available candle source for the requested market universe.
- **lighter** - Lighter USDC perpetuals through bounded CCXT candle requests, with one-way positions
  and base-asset quantities. Use `backtest.exchanges = ["lighter"]`; coverage depends on the
  exchange history available for each market and date range. See the [Lighter guide](exchanges/lighter.md).

The canonical default template currently uses `binance` and `bybit`. Add `bitget` and/or `gateio`
explicitly in `backtest.exchanges` when you want them included.

## Exchange Name Conventions

Cache paths and output directories use standard exchange names (e.g., `binance`, `bybit`, `gateio`, `kucoin`). CCXT-specific IDs such as `binanceusdm`, `bybitusdt`, and `kucoinfutures` are only used internally when communicating with exchange APIs. Always use the short names in your configuration files and external OHLCV source directories.

## Materialized scratch cache locks

Backtests and optimizers serialize scratch allocation and pruning with the persistent
`.materialized.op.flock` file in their materialized cache directory. Do not delete this file:
processes holding an older inode would otherwise stop excluding new owners. The operating system
releases ownership when a worker exits, including after an abnormal exit.

When upgrading from the previous `.materialized.op.lock` directory protocol, stop all older
backtest and optimizer workers before starting the updated version against the same cache.
Concurrent workers using different lock protocols are unsupported. A legacy lock with a confirmed
dead local PID does not block the updated version. An active, foreign, missing, or malformed owner
record is preserved and produces an actionable error. Missing metadata may belong to a worker
paused while creating its lock, so an age limit cannot safely establish that ownership has ended.
After confirming that every older worker has stopped, remove only the legacy
`.materialized.op.lock` directory named in the error and retry. Keep the new `.materialized.op.flock`
file and existing active run directories intact.
