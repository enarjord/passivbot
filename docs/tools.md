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

## Pareto single-candidate explorer

`passivbot tool pareto` reads a Pareto directory of JSON members, optionally filters it with
optimizer-style limit expressions, and selects one candidate using a named decision method.
If you omit the path entirely, it defaults to the newest local `optimize_results/.../pareto`
directory. If you point it at an optimize run directory instead of the nested `pareto/`
directory, it resolves that automatically.

```shell
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto
passivbot tool pareto optimize_results/.../pareto -m reference \
  --target adg_strategy_pnl_rebased=0.001 \
  --target drawdown_worst_hsl=0.25
passivbot tool pareto optimize_results/.../pareto \
  -l 'drawdown_worst_hsl<=0.35' \
  -l 'adg_strategy_pnl_rebased>0.0'
passivbot tool pareto -o sharpe_ratio_strategy_pnl_rebased,adg_strategy_pnl_rebased,peak_recovery_hours_hsl \
  -m ideal
passivbot tool pareto optimize_results/... -m utility \
  --weight adg_strategy_pnl_rebased=4 \
  --weight drawdown_worst_hsl=2 \
  --show-top 5
passivbot tool pareto --json
```

Available methods:

- `knee` - balanced compromise point when you want a compromise selector instead of ideal-point distance
- `reference` - closest to user-specified aspiration targets
- `ideal` - closest to the observed ideal point on the current front; default method
- `utility` - highest weighted normalized utility
- `lexicographic` - strict objective priority order
- `outranking` - simplified PROMETHEE-style pairwise net-flow ranking

The explorer applies limits first, then ranks the retained candidates. It is intended for quickly
promoting one config out of a large Pareto front without opening the dashboard. Its selection
methods are practical decision heuristics for high-dimensional Passivbot fronts, not full formal
multi-criteria decision-analysis implementations.

The output also shows the retained front's ideal point: the best observed value for each active
objective after any `--limit` filters are applied.

`-o` / `--objectives` is not limited to the original `optimize.scoring` list. You can also name
other stored metrics such as `sharpe_ratio_strategy_pnl_rebased` as long as the Pareto JSON
members contain that metric and Passivbot knows whether higher or lower is better.

## Pareto transformations / static plots

`src/tools/pareto_transform.py` converts `all_results.bin` or individual Pareto JSON entries into CSV/JSON summaries for external analysis. The legacy `src/pareto_store.py` still produces quick matplotlib scatter plots if you prefer static images.

```shell
passivbot tool pareto-transform optimize_results/.../all_results.bin --out summary.csv
python3 src/pareto_store.py optimize_results/.../pareto/
```

## Iterative backtester utilities

`src/tools/iterative_backtester.py` and `iterative_history_plot.py` help replay slices of the backtester (or real fills) interactively so you can inspect order-by-order behaviour. Useful when tuning configs by hand.

```shell
passivbot tool iterative-backtester configs/your_config.json --auto-run
passivbot tool iterative-backtester configs/examples/ema_anchor.json --auto-run \
  --override backtest.start_date=2022-01-01 \
  --override backtest.end_date=now \
  --quit-after-run
passivbot tool iterative-history-plot backtests/.../fills.csv
```

## Historical data helpers

- `passivbot tool pad-historical-daily` – Ensures daily OHLCV shards are present for the downloader when new coins are added.
- `passivbot tool verify-hlcvs-data` – Validates cached OHLCV data (gaps, duplicates) before long optimizations/backtests.
- `passivbot tool streamline-json` – Normalizes/compacts JSON configs (`passivbot tool streamline-json configs/examples/default_trailing_grid_long_npos10.json`).

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
