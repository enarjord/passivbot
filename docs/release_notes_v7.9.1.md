# Release Notes for v7.9.1

These notes describe the user-facing changes from `v7.9.0` to `v7.9.1`.

## Highlights
- Backtest and live now share a cleaner `pnls_max_lookback_days` contract, including `"all"` support, correct rolling-window behavior, and restored `backtest.visible_metrics` filtering.
- Hyperliquid HIP-3 live support is much more robust: startup and steady-state state discovery are fixed, sizing balance is reconciled against hidden reserve/margin, and dedicated probe tools are now shipped.
- Config and CLI behavior are stricter and clearer: schema-tagged configs, live-owned shared execution settings, inherited live runtime flags exposed on backtest/optimize CLIs, dotted override fixes, and fail-loud validation for invalid `unstuck_ema_dist`.
- Live runtime behavior is more operationally stable: Bybit fill-history storms are reduced, Binance trade-history pagination is fixed, exchange-aware EMA pacing and hourly jitter reduce API bursts, and live runs archive logs by default.
- Container/runtime and tooling support improved substantially: canonical live container contract, env-driven config rendering, better HLCV cache naming, richer Pareto tooling, downloader CLI cleanup, and new Hyperliquid diagnostics tools.

## Upgrade Notes
- Reinstall after upgrading:
  `python3 -m pip install -e .`
  or
  `python3 -m pip install -e ".[full]"`
- If the Rust extension looks stale, rebuild it in the active environment:
  `maturin develop --release`
- `config.backtest.market_orders_allowed`, `config.backtest.market_order_near_touch_threshold`, and `config.backtest.pnls_max_lookback_days` are no longer accepted. Set them under `config.live`.
- `live.pnls_max_lookback_days` now uses one shared contract across live, HSL, plotting, and backtests:
  - `0` = minimal effective native lookback
  - positive float = rolling N-day window
  - `"all"` = full available history
- Invalid `unstuck_ema_dist` boundary values now hard-fail during config validation:
  - `bot.long.unstuck_ema_dist` must be `> -1.0`
  - `bot.short.unstuck_ema_dist` must be `< 1.0`
- The legacy `python src/downloader.py ...` entrypoint is removed. Use `passivbot download ...`.

## What Changed

### Config, CLI, and Runtime Contracts
- Added formal top-level `config_version` schema tagging starting at `v7.9.0`, with canonical defaults and the mirrored example config carrying the schema version and older configs migrating during load.
- Shared live/backtest execution settings now live only under `config.live`, avoiding silent divergence between runtime and backtest behavior.
- CLI/runtime ownership for inherited `live.*` fields is now explicit, so `backtest` and `optimize` help/projected configs expose the correct inherited runtime flags.
- Dotted CLI overrides now create missing intermediate config sections instead of silently failing when a raw config omits them.
- `live.approved_coins` / `live.ignored_coins` now use a canonical shape with explicit per-side support for `"all"`, and legacy `empty_means_all_approved` inputs migrate with warnings instead of remaining part of canonical config.
- `passivbot live -u/--user` and `-pmld/--pnls-max-lookback-days` are restored as curated default-help shorthands.
- `passivbot optimize --help-all` now exposes fixed per-side bot runtime overrides such as `entry_grid_inflation_enabled` and selected HSL runtime flags without turning them into optimizer dimensions.
- Invalid `unstuck_ema_dist` values that would silently disable unstuck now fail loudly in config loading and in optimize-bounds validation.

### Backtest, Optimize, and Metrics
- Backtests now obey the live-owned `pnls_max_lookback_days` setting correctly, and market-order behavior is treated as a correctness fix instead of preserving legacy bug-compatibility.
- Backtest rolling realized-PnL state now actually expires by time for `pnls_max_lookback_days > 0`, so auto-unstuck and realized-loss gating use the true in-window peak/current pair instead of a stale all-time maximum of the rolling series.
- `backtest.visible_metrics` filtering is restored for standalone terminal output: `null` shows optimize-derived metrics, `[]` shows all, and explicit lists add extras without affecting saved `analysis.json`.
- Added a short-term `entry_grid_inflation_enabled` compatibility flag for inflated grid re-entries, with warnings that cropped-only behavior is the forward path.
- Zero-fill backtests no longer crash during analysis/plotting when balance/equity samples exist but no fills were produced.
- Added trade-level metrics such as `win_rate`, `win_rate_w`, and `trade_loss_{max,mean,median}` plus optimizer-facing ratio metrics like `paper_loss_ratio`, `exposure_ratio`, and weighted variants.
- Liquidation reporting now uses an explicit Rust-provided liquidation flag instead of inferring liquidation from drawdown metrics.
- HLCV dataset caches under `caches/hlcvs_data/` now use descriptive directory names with exchange, coin label/count, effective date range, and cache-hash suffix.
- First-timestamp handling for newly listed coins and source-dir fallback behavior were tightened so candle loading clamps to real listing history and falls back safely when source-dir data is non-contiguous.

### Live Runtime Stability and Exchange Fixes
- Fixed `CCXTBot.create_ccxt_sessions()` using generic exchange names instead of futures-specific CCXT ids, which could fetch wrong market sets and cascade-fail updates.
- Fixed Binance trade-history pagination sending future `endTime` and too-tight 7-day windows, eliminating `-4181 "Invalid start time"` issues on sparse symbols.
- Fixed Bybit closed-pnl pagination storms by deriving fill-lookback coverage from durable cache metadata instead of a session-local flag.
- EMA bundle refresh now uses exchange-aware pacing: strict exchanges honor per-symbol delays while zero-delay exchanges keep concurrent refresh behavior.
- Added random hourly `init_markets` jitter so colocated bots do not fire heavy refresh bursts simultaneously.
- `passivbot live` now archives each run to a timestamped log file under `logs/` and keeps `logs/{user}.log` as a stable alias for tooling.

### Hyperliquid and HIP-3
- Hyperliquid stock-perp state sync now uses dex-aware discovery for startup and steady-state reconciliation, including unapproved-symbol live state on the same dex.
- Isolated HIP-3 live trading remains explicitly unsupported; unsupported live state now fails loudly instead of running partially.
- Hyperliquid live sizing now reconciles hidden reserve/margin by restoring:
  - HIP-3 cross-position `marginUsed`
  - Passivbot-managed resting non-reduce-only entry-order reserve
  while still ignoring external/manual orders.
- Added supported Hyperliquid probe tools to the CLI and docs for balance, order-margin, and position-balance diagnostics.

### Tools, Container Runtime, and Operational UX
- Added a canonical live-container runtime contract around `Dockerfile_live`, `container/entrypoint.sh`, env-generated `api-keys.json`, env-driven config rendering, and documented Compose/Railway deployment flow.
- Removed the legacy standalone downloader entrypoint and replaced it with the unified `passivbot download ...` path plus a dedicated `src/ohlcv_download.py`.
- Pareto tooling improved further: `passivbot tool pareto` can default to the newest local Pareto output, accept run or `pareto/` dirs, use stored metrics outside the original objective list when direction is known, and now defaults to the `ideal` selector.
- Added and polished Hyperliquid probe tools as first-class supported diagnostics instead of ad hoc investigation scripts.

## Short Release Summary
Passivbot `v7.9.1` is a runtime-correctness and operator-UX release. The main themes are:
- correct rolling PnL lookback semantics in backtests
- stronger live/backtest config ownership and CLI exposure
- better Hyperliquid HIP-3 state and balance handling
- more stable live exchange refresh behavior
- clearer release/runtime tooling around logs, containers, downloads, and diagnostics
