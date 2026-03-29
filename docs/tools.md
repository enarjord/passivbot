# Tools

## Pareto dashboard for optimizer runs

`src/tools/pareto_dash.py` scans one or more `optimize_results/` directories and launches a Plotly Dash UI with scatter plots, histograms, suite-aware metrics, and CSV export. Install Dash/Plotly (`pip install dash plotly`) if they are not already in your environment.

```shell
python3 src/tools/pareto_dash.py --data-root optimize_results
```

Pass `--run optimize_results/<timestamp>/` to load a specific run or point it at the entire results directory to browse multiple runs at once.

## Pareto transformations / static plots

`src/tools/pareto_transform.py` converts `all_results.bin` or individual Pareto JSON entries into CSV/JSON summaries for external analysis. The legacy `src/pareto_store.py` still produces quick matplotlib scatter plots if you prefer static images.

```shell
python3 src/tools/pareto_transform.py optimize_results/.../all_results.bin --out summary.csv
python3 src/pareto_store.py optimize_results/.../pareto/
```

## Iterative backtester utilities

`src/tools/iterative_backtester.py` and `iterative_history_plot.py` help replay slices of the backtester (or real fills) interactively so you can inspect order-by-order behaviour. Useful when tuning configs by hand.

```shell
python3 src/tools/iterative_backtester.py --config configs/your_config.json --symbol BTC/USDT:USDT
python3 src/tools/iterative_history_plot.py backtests/.../fills.csv
```

## Historical data helpers

- `src/tools/pad_historical_daily.py` – Ensures daily OHLCV shards are present for the downloader when new coins are added.
- `src/tools/verify_hlcvs_data.py` – Validates cached OHLCV data (gaps, duplicates) before long optimizations/backtests.
- `src/tools/streamline_json.py` – Normalizes/compacts JSON configs (`python3 src/tools/streamline_json.py configs/template.json`).

## Market-cap based approved coins

`src/tools/generate_mcap_list.py` emits a JSON list of coins filtered by market cap and optionally by exchange availability.

```
python3 src/tools/generate_mcap_list.py -n 80 -m 200 -e binance,bybit -o configs/approved_coins_top80.json
```
