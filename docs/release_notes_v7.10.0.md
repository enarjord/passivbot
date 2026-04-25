# Release Notes for v7.10.0

These notes describe the user-facing changes from `v7.9.1` to `v7.10.0`.

## Highlights
- Added the OHLCV v2 foundation under `caches/ohlcvs/`, with monthly chunk storage, SQLite catalog metadata, legacy cache import, persistent gap tracking, and v2-aware backtest preparation.
- Added `passivbot tool inspect-ohlcvs` for inspecting v2 OHLCV cache coverage, chunk validity, persistent gaps, and recent fetch attempts.
- Updated the canonical schema and mirrored example profile to `configs/examples/default_trailing_grid_long_npos7.json`, with `config_version = v7.10.0`.
- Removed inflated grid re-entry behavior. Grid re-entries are now always normal-or-cropped, while historical inflated order-type ids remain decodable for old fills and restart compatibility.
- Renamed collateral-agnostic strategy-equity metrics to canonical `*_strategy_eq` names, while keeping old `*_strategy_pnl_rebased` and `*_hsl` names as input/result aliases.
- Added day-denominated duration metrics alongside existing hour metrics for high exposure, peak recovery, position held, and position unchanged.
- Added richer backtest artifacts: `dataset.json`, `strategy_equity` in `balance_and_equity.csv.gz`, artifact loading helpers, and single-coin fill plotting for notebooks.

## Upgrade Notes
- Reinstall after upgrading:
  `python3 -m pip install -e .`
  or
  `python3 -m pip install -e ".[full]"`
- If the Rust extension looks stale, rebuild it in the active environment:
  `maturin develop --release`
- New configs should use `config_version = "v7.10.0"`.
- `configs/examples/default_trailing_grid_long_npos10.json` has been replaced by `configs/examples/default_trailing_grid_long_npos7.json`.
- `bot.{long,short}.entry_grid_inflation_enabled` is no longer a supported runtime behavior. Older configs that still contain it are normalized during config loading.
- Canonical optimizer configs should use `*_strategy_eq` metric names and day-duration metrics. Deprecated metric names remain accepted as compatibility aliases.

## What Changed

### OHLCV v2 Foundation
- Added v2 OHLCV chunk storage and catalog modules for backtest-oriented candle preparation.
- Backtests can prepare HLCV payloads from the v2 local store when coverage is already available, while keeping `caches/hlcvs_data/` as the fast reusable prepared bundle cache.
- Legacy daily OHLCV shards, including compressed `.npz` shards, can be imported into the v2 store.
- Persistent exchange-side gaps are tracked so repeated fetch attempts do not endlessly retry known unavailable ranges.
- Combined-exchange backtests now have a v2-aware load path and clearer progress logging while preparing candles.

### Backtest Artifacts and Analysis
- Backtest runs now persist `dataset.json` with the HLCV cache files used by the run.
- Added `src/backtest_artifacts.py` helpers for loading config, analysis, fills, balance/equity, HLCVs, timestamps, BTC/USD prices, and market settings from an artifact directory.
- Notebook workflows can call `load_backtest_artifact_workspace(...)`, `candles_for_coin(...)`, and `plot_fills_for_coin(...)`.
- `balance_and_equity.csv.gz` now includes collateral-agnostic `strategy_equity`.
- Backtest BTC collateral is initialized at the first active trading step instead of during EMA warmup.
- `drawdown_worst_mean_1pct` metrics now derive drawdowns from the full-resolution equity curve before averaging the worst 1% of daily worst drawdowns.

### Metrics and Optimization
- Canonical collateral-agnostic strategy-equity metrics now use `*_strategy_eq`.
- Deprecated `*_strategy_pnl_rebased` and `*_hsl` metric names are accepted as aliases for configs, limits, visibility filters, Pareto tools, and older stored result files.
- `peak_recovery_hours_pnl` now uses net realized PnL (`pnl + fee_paid`) and includes the open tail from the last realized-PnL peak to the end of the backtest.
- Day-duration metric variants were added for high exposure, peak recovery, position held, and position unchanged.
- Suite-mode limit semantics are centralized so `passivbot optimize` and `passivbot tool pareto` resolve omitted `stat=` consistently from `backtest.aggregate`.

### Config and Runtime Defaults
- The hardcoded schema defaults and mirrored example config now use a trailing-grid `n_positions = 7` profile.
- Default approved coins, scenarios, optimizer bounds, scoring, and limits were refreshed.
- Shorts are disabled by default through zero short total wallet exposure.
- Example configs now prefer canonical day-duration metrics where appropriate.

### Order Behavior
- Inflated grid re-entry behavior was removed from current live, backtest, and runtime paths.
- Grid entries remain normal-or-cropped so effective wallet exposure limits are observed without pulling future size forward.
- Historical inflated order-type ids remain readable for old fills and restart compatibility.

## Short Release Summary
Passivbot `v7.10.0` focuses on backtest data infrastructure, clearer strategy-equity metrics, safer grid-entry behavior, and more useful research artifacts. The main operational changes are the new OHLCV v2 foundation, the new `n_positions = 7` default profile, removal of inflated grid re-entries, canonical `*_strategy_eq` metrics, and notebook-friendly backtest artifact helpers.
