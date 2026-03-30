# Tools

Most tools below require the full install profile:

```shell
python3 -m pip install -e ".[full]"
```

## Pareto dashboard for optimizer runs

`passivbot tool pareto-dash` scans one or more `optimize_results/` directories and launches a Plotly Dash UI with scatter plots, histograms, suite-aware metrics, and CSV export. Use the full install profile (`python3 -m pip install -e ".[full]"`) if those dashboard dependencies are not already installed.

```shell
passivbot tool pareto-dash --data-root optimize_results
```

Pass `--run optimize_results/<timestamp>/` to load a specific run or point it at the entire results directory to browse multiple runs at once.

## Pareto transformations / static plots

`src/tools/pareto_transform.py` converts `all_results.bin` or individual Pareto JSON entries into CSV/JSON summaries for external analysis. The legacy `src/pareto_store.py` still produces quick matplotlib scatter plots if you prefer static images.

```shell
passivbot tool pareto-transform optimize_results/.../all_results.bin --out summary.csv
python3 src/pareto_store.py optimize_results/.../pareto/
```

## Iterative backtester utilities

`src/tools/iterative_backtester.py` and `iterative_history_plot.py` help replay slices of the backtester (or real fills) interactively so you can inspect order-by-order behaviour. Useful when tuning configs by hand.

```shell
passivbot tool iterative-backtester --config configs/your_config.json --symbol BTC/USDT:USDT
passivbot tool iterative-history-plot backtests/.../fills.csv
```

## Historical data helpers

- `passivbot tool pad-historical-daily` – Ensures daily OHLCV shards are present for the downloader when new coins are added.
- `passivbot tool verify-hlcvs-data` – Validates cached OHLCV data (gaps, duplicates) before long optimizations/backtests.
- `passivbot tool streamline-json` – Normalizes/compacts JSON configs (`passivbot tool streamline-json configs/template.json`).

## Repro and diagnostics helpers

`src/repro_harness.py` replays a stored config or Pareto JSON through both the optimizer-evaluation path and the backtest path in one process, then compares the resulting metrics and Rust binary provenance.

```shell
PYTHONPATH=src python3 src/repro_harness.py optimize_results/.../pareto/<hash>.json --json
```

`src/tools/capture_optimize_memory.py` samples process-tree RSS, `/dev/shm`, and host memory state while an optimizer run is active.

```shell
PYTHONPATH=src python3 src/tools/capture_optimize_memory.py --wait --output tmp/optimize_memory.json
```

`src/analysis_visibility.py` is a reusable helper module for resolving `backtest.visible_metrics` against optimize scoring/limits so analysis views can consistently show only the requested metric surface.

## VPS sync helpers

`sync_tar.py` and `vpssync.sh` provide a simple tar-over-ssh/scp workflow for moving configs, logs, backtests, or optimize results between this repo and a VPS tree.

```shell
python3 sync_tar.py push optimize_results/2026-... vps3 /root/passivbot/optimize_results --remote-extract
sh vpssync.sh pull vps3 "logs/20260318*"
```

The archive helper also supports symmetric `pull` and local `extract` modes, including wildcard remote matches.

## Market-cap based approved coins

`src/tools/generate_mcap_list.py` emits a JSON list of coins filtered by market cap and optionally by exchange availability.

```shell
passivbot tool generate-mcap-list -n 80 -m 200 -e binance,bybit -o configs/approved_coins_top80.json
```
