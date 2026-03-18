# Changelog

All notable user-facing changes will be documented in this file.

## Unreleased

### Added
- **Forager config docs and optimizer bounds** - Added `docs/forager.md` with detailed forager selection behavior, failure policy, and tuning guidance. Canonical config naming now uses `forager_volume_ema_span` / `forager_volatility_ema_span`, and optimizer bounds now include side-specific `forager_score_weights_{volume,ema_readiness,volatility}` knobs.
- **Strategy-PnL rebased + HSL metric families** - Backtest analysis now includes shared `*_strategy_pnl_rebased` growth/quality metrics and `*_hsl` risk metrics, including `peak_recovery_hours_hsl`, so optimizer scoring and limits can use a cleaner collateral-agnostic return family plus HSL-style drawdown/recovery constraints. `backtest.visible_metrics` can also limit which analysis metrics are shown in CLI output without removing any persisted or internally computed metrics. Optimizer startup now logs starting-config dedup statistics, and `optimize.starting_config_twe_multiplier` controls the extra TWEL-scaled seed variant (`1.0` disables it).
- **Side-specific HSL config + metrics** - HSL is now configured per `pside` under `bot.long.hsl_*` and `bot.short.hsl_*`. Live and backtest HSL runtime behavior is now side-specific, while backtest analysis retains global `*_hsl` metrics and also exports `*_hsl_long` / `*_hsl_short` metrics plus side-specific hard-stop trigger/restart counts. `live.hsl_signal_mode` now lets those per-`pside` controllers use either `pside`-specific strategy drawdown signals or a shared `unified` account-level signal.
- **Optimizer provenance + replay diagnostics** - Optimizer results now include runtime provenance for the active Rust extension and related evaluation context, and `src/repro_harness.py` can compare stored metrics against current optimizer-path and backtest-path replays in one command.
- **HSL events per-year metrics** - Backtest HSL analysis now also exports `hard_stop_triggers_per_year` and `hard_stop_restarts_per_year` so runs with different date ranges can be compared more directly without losing the absolute trigger/restart counts.
- **Long/share of net profit metric** - Backtest analysis now also exposes `long_short_profit_ratio`, a clearer alias for the long-side share of total net realized PnL (`1.0` = all net profit from longs, `0.0` = all from shorts, `0.6` = 60/40 long/short split).

### Fixed
- **Optimizer initial-seed memory spike** - Starting-config evaluation no longer queues the entire seed pool at once. Initial evaluations are now bounded by `optimize.max_pending_starting_evals_per_cpu * n_cpus`, which reduces RAM spikes on large seeded runs, especially in suite mode.
- **Rust-owned market-vs-limit execution intent** - Rust orchestrator now decides whether eligible non-panic orders should be emitted as `limit` or `market` using one shared near-touch threshold and market-crossing rules. Live now consumes that Rust execution intent directly, and backtests use the same intent for guaranteed market fills with slippage and taker fees.
- **Live orchestrator fee payload + Hyperliquid hard-stop history replay** - Live orchestrator inputs now include per-symbol `maker_fee` and `taker_fee`, matching the Rust JSON contract used by backtests. Hyperliquid hard-stop equity-history replay now backfills older missing 1m minutes from 5m/15m candles with deterministic intrabar reconstruction instead of relying only on flat carry-forward gaps once the exchange's 5000-candle fetch cap is exceeded.
- **Live HSL status logging** - Live equity hard-stop now emits a periodic INFO-level status summary with tier, distance to RED, drawdown raw/EMA/score, peak metrics, and cooldown context. RED cooldown heartbeat logging is also throttled to avoid repeated spam every few seconds.
- **Backtest taker-fee execution for market fills** - Backtest market executions now charge taker fees instead of maker fees, respect optional `backtest.taker_fee_override`, and record a `liquidity` column (`maker` / `taker`) in `fills.csv`. Simulated market fills remain guaranteed once selected, with execution price shifted by `backtest.market_order_slippage_pct`.
- **Backtest HSL drawdown visualization** - Backtests now output `hard_stop_drawdown.png` alongside the existing summary plots when HSL is enabled. The plot is now driven by the actual Rust HSL traces, so it stays aligned with the current side-specific long/short controller semantics instead of relying on the old account-level config path approximation. `--disable_plotting` also supports a dedicated `hard_stop` plot group.
- **Backtest HSL panic execution and metrics export** - Account-level RED panic now forces panic mode on all symbols/sides in Rust backtests, `panic_close_order_type="market"` is simulated as next-bar taker execution instead of limit-only behavior, and `hard_stop_*` analysis metrics are exported once as shared metrics rather than duplicated into `_usd`/`_btc` variants.
- **Backtest HSL EMA span fallback** - Backtests no longer fail when `bot.{long,short}.hsl_ema_span_minutes` is smaller than `backtest.candle_interval_minutes`. Sub-interval spans now fall back to a one-sample EMA, which disables smoothing and makes the HSL score follow raw drawdown.
- **Backtest market-order slippage config naming** - Renamed `backtest.panic_market_slippage_pct` to `backtest.market_order_slippage_pct` so one backtest slippage knob can cover all simulated market-order execution.
- **HSL no-restart threshold semantics** - Values of `bot.{long,short}.hsl_no_restart_drawdown_threshold` below `hsl_red_threshold` are now clamped up to `hsl_red_threshold` in live, backtest, and optimizer flows. Both live and backtests now use persistent cross-restart HSL drawdown for no-restart latching, while optimizer runs disable terminal no-restart by default via `optimize.fixed_runtime_overrides` so drawdown can be constrained through `*_hsl` metrics without prematurely ending candidate runs.
- **Backtest HSL analysis metrics expanded and clarified** - Added account-level HSL metrics for yellow/orange/red time share, RED halt duration, trigger drawdown, panic-close realized loss, flatten time, and restart-to-retrigger rate. Also renamed the old ambiguous halt-loss metric to `hard_stop_halt_to_restart_equity_loss_pct`.
- **Optimizer constraint visibility** - Pareto logging now reports the top violated constraints and penalties instead of only the aggregate penalty number.
- **One-way forager shortlist eligibility** - In `hedge_mode: false`, coins already occupied on one side no longer consume initial-entry shortlist slots on the opposite side before being blocked.

### Changed
- **BTC-denominated backtest metrics now always use BTC equity** - `*_btc` metrics are now computed from BTC-denominated balance/equity even when `backtest.btc_collateral_cap = 0`, instead of mirroring the USD analysis. This makes metrics like `adg_btc` and `gain_btc` informative as BTC-relative performance measures for cash-collateral runs as well.
- **ADG terminal smoothing simplified** - Backtest `gain`/`adg` now smooth the terminal value by taking the mean of the last up to 3 daily equity samples instead of running an EMA over the full daily-equity series. This preserves end-of-run drawdown smoothing while reducing computation.
- **Optimize-time runtime controls** - Optimizer config now supports `optimize.fixed_params` to lock selected bounds to the current config value and `optimize.fixed_runtime_overrides` to override runtime config values only during optimize evaluations.

## v7.8.4 - 2026-03-06

### Changed
- **Dual balance routing (raw vs hysteresis-snapped)** - Live and orchestrator flows now carry both `balance_raw` (raw wallet balance) and `balance` (hysteresis-snapped balance). Sizing/order-shaping paths use snapped balance, while risk/accounting paths use raw balance (including realized-loss gate peak/floor checks, TWEL entry/auto-reduce gating, and auto-unstuck allowance calculations). This applies consistently across live and backtest via Rust orchestrator input.
- **WEL denominator behavior split by mode** - Live now uses a hard fixed denominator for per-symbol WEL (`total_wallet_exposure_limit / config.bot.{pside}.n_positions`), removing runtime denominator drift from open-position count. Backtests now expose `backtest.dynamic_wel_by_tradability` (default `true`): when enabled, WEL uses tradability-aware denominator growth (`min(n_positions, n_tradable_max)`) based on coins with real candles, and does not shrink after delistings; when disabled, backtests use the same fixed denominator as live.
- **Bulk price fetch for Hyperliquid** - `calc_ideal_orders` now uses a single `allMids` API call to get prices for all symbols instead of individual `get_current_close` calls per symbol (1 call vs ~70). Falls back to per-symbol fetches for non-Hyperliquid exchanges or on error.
- **Sequential margin mode setting for Hyperliquid** - Margin mode and leverage API calls are now sequential with a small delay instead of being fired in parallel, reducing API burst on coin changes.
- **Equity hard-stop framework (live+backtest)** - Added the equity hard-stop framework with configurable thresholds, EMA span in minutes, yellow/orange tier ratios, orange mode selector, panic close order type, plus Rust drawdown/tier state machine support, backtest rolling-peak enforcement using `pnls_max_lookback_days`, and live runtime hooks for tier tracking and RED supervisory flatten-until-confirmed-flat behavior.

### Fixed
- **Bybit fill-event qty inflation on duplicate pages** - `BybitFetcher` now deduplicates `fetch_my_trades` rows by exec id before canonicalization/coalescing, preventing duplicate pagination rows from inflating canonical `qty`, `fees`, and close PnL.
- **Balance peak drift in wrong direction under hysteresis** - Peak reconstruction (`balance + (pnl_cumsum_max - pnl_cumsum_last)`) previously used hysteresis-snapped balance in some paths. Since snapped balance can stay stale while `pnl_cumsum_last` changes fill-by-fill, this made reconstructed peak drift down after profits and up after losses. Peak/PnL-accuracy-sensitive paths now use raw balance (`balance_raw`) consistently.
- **Pytest Rust-module bootstrap fallback** - Test bootstrap now tries the project venv `passivbot_rust` package before falling back to the lightweight stub when tests are launched outside the venv, reducing false failures from missing/incorrect Rust module resolution.
- **`max_ohlcv_fetches_per_minute` ignored when forager slots open** - The rate limit config was only applied when all position slots were full. With open slots (the common case), all candidate symbols were fetched without rate limiting, causing 429 errors on Hyperliquid.
- **Hyperliquid positions+balance double fetch** - `fetch_positions` and `fetch_balance` now share a single API call via a dedup lock instead of making two identical `clearinghouseState` requests per execution cycle.
- **Thundering herd on minute boundary** - `get_candles` no longer force-refreshes all symbols simultaneously when a new minute boundary crosses. A 1-candle staleness tolerance prevents the TTL override that caused all symbols to fetch at once.
- **Candle refresh TTLs aligned to 1-minute finalization** - Active candle refresh TTL raised from 10s to 60s and EMA close TTL from 30s to 60s, matching the actual 1-minute candle finalization interval.
- **Boot stagger for multi-bot setups** - Added `boot_stagger_seconds` config (default 30s for Hyperliquid) to randomize startup delay, preventing simultaneous API bursts when multiple bots share the same IP.
- **Warmup and refresh fetch pacing** - Added configurable `warmup_fetch_delay_ms` (default 200ms for Hyperliquid) with delays between individual symbol fetches during warmup, forager refresh, and active candle refresh loops.
- **Exponential backoff on 429 errors** - WebSocket `watch_orders` uses exponential backoff (up to 30s) on rate limit errors. Execution loop backs off 5s on `RateLimitExceeded`. Hourly `init_markets` catches rate limits with 10s recovery.
- **Fill events pagination abort on repeated rate limits** - `HyperliquidFetcher` now aborts after 5 consecutive rate limit retries with exponential backoff instead of retrying indefinitely.
- **EMA bundle and active candle sweep abort on rate limit** - Both `_load_orchestrator_ema_bundle` and `update_ohlcvs_1m_for_actives` skip remaining symbols when the CandlestickManager's global rate limit backoff is active.
- **Live close-EMA failure handling in orchestrator feed** - `_load_orchestrator_ema_bundle()` no longer silently drops failed/non-finite close EMA spans. It now fails loudly when no prior EMA exists, and otherwise reuses the last successfully computed close EMA for that exact symbol/span with explicit `[ema]` warning logs (including reason, age, and consecutive fallback count).
- **Required 1h log-range EMA handling in orchestrator feed** - `_load_orchestrator_ema_bundle()` now fails loudly when required `h1` log-range spans (from `entry_volatility_ema_span_hours`) are missing or non-finite, instead of deferring to downstream Rust `MissingEma` errors.
- **EMA bundle fetch stability under lock contention** - Orchestrator EMA bundle loading now fetches per-symbol spans serially and drains all symbol task outcomes before re-raising, reducing same-symbol candle-lock contention and eliminating unretrieved sibling-task exception noise.
- **Weighted forager slot ranking** - Forager initial-entry slot filling now uses hard eligibility, coarse `forager_volume_drop_pct` pruning, and weighted normalized ranking across `volume`, `ema_readiness`, and `volatility`. User-facing config now uses `bot.{long,short}.forager_volume_drop_pct` and required `forager_score_weights`; legacy `filter_volume_drop_pct` is migrated in `format_config()`. EMA readiness is measured against the actual offset initial-entry threshold, reducing idle slot occupancy on coins that are far from entry.

### Added
- **Fill events doctor tool** - Added `src/tools/fill_events_doctor.py` to audit cached fill events and auto-repair known Bybit duplicate-fill anomalies without requiring exchange API calls. Bybit startup now runs doctor by default (can be disabled with `PASSIVBOT_FILL_EVENTS_DOCTOR=off`).

## v7.8.3 - 2026-02-24

### Added
- **Global realized-loss gate for close orders** - Added `live.max_realized_loss_pct` (default `0.05`) to block any close order (including WEL/TWEL auto-reduce and unstuck) that would realize losses beyond a peak-balance-relative threshold. Panic closes remain exempt. Live bot now emits `[risk]` warnings when orders are blocked by this gate.

### Fixed
- **False-positive stale Rust extension after identical rebuild** - `sync_installed_extension_into_src()` now updates the local `src/` `.so` mtime when its content (SHA256) already matches the installed site-packages build. Previously the old mtime was preserved, causing `check_and_maybe_compile` to report the extension as stale in a loop even though the binary was current.
- **Peak recovery hours PnL metric** - `peak_recovery_hours_pnl` now computes directly from fill events using gross PnL with strict peak detection (`>` instead of `>=`), instead of reconstructing a cumulative series over the equity index. Fixes inaccurate recovery times when fills were sparse relative to the equity series.
- **Combined OHLCV normalization source selection** - Volume normalization in combined backtests now uses each coin's OHLCV source exchange (`ohlcv_source`) instead of the market-settings exchange when `backtest.market_settings_sources` differs from OHLCV routing.
- **Config template/format preservation** - Added `live.enable_archive_candle_fetch` to the template defaults and ensured `backtest.market_settings_sources` is preserved during config formatting.
- **Live no-fill minute EMA continuity** - When finalized 1m candles are missing because no trades occurred, live runtime now materializes synthetic zero-candles in memory (not on disk), preventing avoidable `MissingEma` loop errors on illiquid symbols. If real candles arrive later, they overwrite synthetic runtime candles and invalidate EMA cache automatically.
- **Suite base scenario inherited all scenario coins** - Scenarios without explicit `coins` (e.g. the `"base"` scenario) fell back to `master_coins` — the union of every scenario's coin list — instead of the original `approved_coins` from the config. Now `apply_scenario` falls back to `base_coins` (the config's `approved_coins`) when a scenario omits its own coin list.
- **Aggregate methods ignored in optimizer scoring and Pareto analysis** - `calc_fitness` always looked up the `_mean` stat for every scoring metric, ignoring the `backtest.aggregate` config (e.g. `"high_exposure_hours_max_long": "max"`). The optimizer now overrides `flat_stats` with correctly aggregated values before computing objectives. The standalone `pareto_store.py` script reads the aggregate config for suite-metric extraction and limit filtering while leaving stored objectives unchanged.
- **Backtest HLCV cache reuse across configs** - Configs that differ only in trading parameters (EMA spans, warmup ratio) now share the same HLCV cache slot. Previously, different EMA spans produced different `warmup_minutes`, which was included in the cache hash, causing unnecessary re-downloads. The cache now uses a ratchet-up strategy: warmup sufficiency is checked at load time, and the cache is overwritten only when a larger warmup is needed.
- **Backtest cache warmup downgrade guard** - Cache saves now keep the highest recorded `warmup_minutes` for a cache slot and skip writes that would downgrade it, reducing refetch churn when multiple runs touch the same cache concurrently.

## v7.8.2 - 2026-02-09

### Added
- **Configurable candle interval** - New `backtest.candle_interval_minutes` setting (default 1) aggregates 1m candles to coarser intervals (e.g., 5m) for faster backtests and optimizer iterations. EMA alphas are automatically adjusted for the interval. Trade-off: intra-interval fill ordering is lost.
- **High-exposure duration metrics** - New backtest metrics `high_exposure_hours_{mean,max}_{long,short}` measuring continuous durations where total wallet exposure exceeded its daily average. Available for optimization scoring and limit checks.
- **Total wallet exposure plot** - Backtests now output `total_wallet_exposure.png` showing long TWE (positive, blue) and short TWE (negative, red) over time.
- **External OHLCV source dir** - New `backtest.ohlcv_source_dir` config option to load 1m candle data from a pre-populated directory tree before falling back to exchange archives. Supports both `.npy` and `.npz` file formats.

### Fixed
- **OHLCV source-dir fallback behavior** - Non-contiguous source-dir candle data now falls back to CandlestickManager instead of propagating gappy series into downstream strict continuity checks.

### Fixed
- **Short-only exposure metrics** - `total_wallet_exposure_max` and related metrics now use absolute values, correctly reporting exposure magnitude for short-only configs where `twe_net` is negative.
- **Timestamp day bucketing** - Backtest analysis now initializes daily bucketing from the first timestamp, preventing a phantom first-day sample when using aggregated candle intervals.
- **Forager fills plots with aggregated candles** - `fills_plots` now use the effective candle stream from the executed backtest, keeping fills aligned when `backtest.candle_interval_minutes > 1`.

### Changed
- **Template config tuning** - Updated `configs/template.json` optimization bounds/scenarios and backtest defaults (`btc_collateral_cap`, `maker_fee_override`, optimize limits).

## v7.8.1 - 2026-02-07

### Fixed
- **Gate.io cache cutoff** - Set `GATEIO_CACHE_CUTOFF_DATE` to 2026-02-07 so stale Gate.io caches are quarantined on startup.

## v7.8.0 - 2026-02-07

### Fixed
- **Live bot candle cache** - Rebuilds candlestick index metadata for the required warmup ranges on startup, preventing stale `index.json` metadata from suppressing candle refreshes.
- **Windows backtest startup** - Avoids importing `resource` at module load, preventing crashes on Windows during backtest/optimizer startup.
- **Legacy cache migration** - Migration now runs once globally and covers all exchanges on first init (not just the first exchange to start), and legacy data is resolved relative to the cache root to avoid unintended copies.
- **Combined OHLCV selection** - `market_settings_sources` no longer expands OHLCV candidates; combined data now uses `backtest.exchanges` plus forced coin sources only.

### Changed
- **Logging** - Reduced INFO/WARNING noise (unsupported market notices now INFO with `[config]`, hedge-mode success logs moved to DEBUG, Bitget OHLCV limit probes moved to DEBUG, KuCoin PnL discrepancy warnings further throttled, large zero-candle warnings now only trigger above 1000). Added `[order]` tag to order plan summaries and extra context for MissingEma errors.

## v7.7.1 - 2026-02-07

### Added
- **Stock perps (HIP-3) support** - Hyperliquid stock perpetuals are now supported, including symbol normalization and routing in combined mode.
- **Pareto host** - Added a lightweight host mode for serving Pareto outputs.

### Fixed
- **Combined HLCV prep** - Fixed `orig_coins` NameError during combined data preparation.

### Changed
- **Logging refinements** - Further reduced INFO noise and improved context across rounds 8–10.
- **Agent docs** - Updated guidance and pitfalls documentation for cross-platform portability.

## v7.7.0 - 2026-01-26

### Fixed
- **Bybit: Missing PnL on some close fills** - Fixed pagination bug in `BybitFetcher._fetch_positions_history()` that caused closed-pnl records to be skipped when >100 records existed in a time window. Now uses hybrid pagination: cursor-based for recent records (no gaps), time-based sliding window for older records.

### Added
- **Fill events now include psize/pprice** - Each fill event is annotated with position size (`psize`) and VWAP entry price (`pprice`) after the fill. Values are computed using a two-phase algorithm and persisted to cache for all exchanges.
- **Logging best practices documentation** - New `docs/ai/log_analysis_prompt.md` with comprehensive logging guidelines, level definitions, and improvement tracking.
- **Exchange API quirks documentation** - New `docs/ai/exchange_api_quirks.md` documenting known exchange-specific limitations and workarounds.
- **Debugging case studies** - New `docs/ai/debugging_case_studies.md` with detailed debugging sessions as reference.

### Changed
- **Logging improvements (7 rounds of refinement)**:
  - Standardized log tags: `[memory]`, `[warmup]`, `[hourly]`, `[fills]`, `[mapping]`, `[candle]`, `[ranking]`, `[mode]`
  - Moved routine API/cache messages from INFO to DEBUG level (CCXT fetch details, cache updates)
  - Moved CCXT API payloads from DEBUG to TRACE level
  - EMA ranking logs now throttled to every 5 minutes (was every cycle)
  - Mode changes throttled to 2 minutes per symbol (reduces forager oscillation noise)
  - KucoinFetcher PnL discrepancy warnings throttled to 1 hour with delta-based deduplication
  - WebSocket reconnection now logs explicit `[ws] reconnecting...` messages
  - Strict mode gaps changed from WARNING to DEBUG (expected for illiquid markets)
  - Persistent gaps changed from WARNING to INFO with throttling
  - Zero-candle synthesis warnings aggregated and rate-limited
- **PnL tracking now uses FillEventsManager exclusively** - Legacy `update_pnls` path removed. FillEventsManager provides more accurate fill tracking with proper event deduplication, canonical schemas, and exchange-specific fetchers for all supported exchanges.
- Fill events are now stored in `caches/fill_events/{exchange}/{user}/` instead of the old `caches/{exchange}/{user}_pnls.json` format. Existing legacy cache files are ignored; FillEventsManager will rebuild from exchange API on first run.
- Unstuck allowances now computed from FillEventsManager data instead of legacy pnls list.
- Trailing position change timestamps now derived from FillEventsManager events.

### Removed
- `--shadow-mode` CLI flag (no longer needed; FillEventsManager is production-ready)
- `live.pnls_manager_shadow_mode` config option
- Legacy `init_pnls`, `update_pnls`, `fetch_pnls` methods in passivbot.py
- Legacy `init_fill_events`, `update_fill_events`, `fetch_fill_events` methods (dead code)
- Shadow mode comparison logging (`_compare_pnls_shadow`, etc.)

### Migration Notes
- **No action required** - FillEventsManager automatically fetches and caches fill data
- Old `{user}_pnls.json` cache files can be safely deleted after upgrading
- If using custom exchange configurations, ensure the exchange's fill fetcher is supported (Binance, Bybit, Bitget, GateIO, Hyperliquid, KuCoin, OKX)

## v7.6.2 - 2026-01-20

### Fixed
- One-way mode now respects disabled sides when choosing initial entry side, preventing a disabled side from blocking entries.
- Startup banner now dynamically calculates width to prevent misaligned borders.
- Bybit leverage/margin mode "not modified" errors now handled gracefully instead of logging full tracebacks.
- Large warmup spans (>2 days) now properly trigger gap-filling via CCXT even when end_ts touches present, fixing issue where thousands of zero-candles were synthesized for historical gaps.
- Windows compatibility: cache folder names now replace `:` with `_` on Windows or when `WINDOWS_COMPATIBILITY=1` env var is set (#547, thanks @FelixJongleur42). **Note:** Existing Windows caches will be orphaned and re-downloaded.
- Pareto dashboard: fixed JavaScript callback errors when switching between tabs (#550, thanks @646826).

### Changed
- Config modification logs now prefixed with `[config]` for easier filtering (e.g., `[config] changed live.user bybit_01 -> gateio_01`).
- Zero-candle synthesis logs are now rate-limited to at most once per minute per symbol, reducing log spam.
- Zero-candle logs now include human-readable UTC timestamps showing which candles were synthesized (e.g., `synthesized 3 zero-candles at 2026-01-19T22:15 to 2026-01-19T22:17`).
- Synthetic candles are now tracked at runtime; when real data arrives for a previously-synthetic timestamp, the EMA cache is automatically invalidated and will be recomputed on next cycle.
- FillEventsManager logs now prefixed with `[fills]` for easier filtering; verbose refresh logs consolidated into single summary line (e.g., `[fills] refresh: events=1311 (+1) | persisted 2 days (2026-01-19, 2026-01-20)`).
- BybitFetcher residual PnL warnings reduced to debug level with compact summary (was logging all order IDs every cycle at WARNING level).
- Health summary now includes realized PnL sum when fills > 0 (e.g., `fills=3 (pnl=+12.50)`).
- Startup banner now shows "TWEL" (Total Wallet Exposure Limit) instead of "Exposure" to clarify it's a limit, not current exposure; long+short mode shows both limits (e.g., `TWEL: L:125% S:85%`).
- Synthetic candle replacement logs now prefixed with `[candle]` for easier filtering.

### Added
- `openpyxl` added to `requirements-live.txt` (required for Bitget archive XLSX parsing).
- `CandlestickManager.needs_ema_recompute(symbol)`: check if EMAs should be recomputed due to synthetic→real data replacement.
- `CandlestickManager.clear_synthetic_tracking(symbol)`: clear synthetic timestamp tracking after warmup completes.
- `live.warmup_jitter_seconds` (default 30): random delay before warmup to prevent API rate limit storms when multiple bots start simultaneously.
- `live.max_concurrent_api_requests` (default null): optional global concurrency limit for CCXT API calls via CandlestickManager's network semaphore.
- `backtest.maker_fee_override` (default null): optional backtest/optimizer maker fee override (part-per-one) to replace exchange-derived fees.
- `live.enable_archive_candle_fetch` (default false): opt-in to use exchange archive data for candle fetching in live bots; disabled by default to avoid potential timeout issues. Backtester always enables archive fetching regardless of this setting.

## v7.6.1 - 2026-01-03

### Testing
- Added comprehensive test coverage for HLCV preparation module (16 tests covering 1,017 lines of production code)
- Added comprehensive orchestrator integration tests (19 tests for order accuracy, edge cases, multi-symbol coordination)
- Added warmup utilities test coverage (20 tests for EMA warmup calculations and edge cases)
- Improved Rust stub in conftest.py with correct parameter signatures and orchestrator JSON API support
- Total: 55 new tests, bringing test suite from ~420 to 477 passing tests

## v7.6.0 - 2026-01-03

### Added
- Shared Pareto core (`pareto_core.py`) with constraint-aware dominance, crowding, and extreme-preserving pruning; reused by ParetoStore.
- Canonical suite metrics payload now shared by backtest and optimizer; suite summaries include the same schema as Pareto members.
- Targeted Pareto tests to ensure consistency.
- KuCoin exchange-config regression tests covering hedge-mode setup and leverage/margin configuration (guards CCXT upgrades).
- Pareto explorer: added configurable “Closest config metrics” dropdown so users can choose which metrics are shown in the Closest Config table, defaulting to scoring/limit metrics.
- `live.balance_override` setting/CLI flag to pin balance to a fixed value instead of fetching from the exchange (off by default).
- Fill events manager: added Gate.io support via ccxt trade fetcher.
- Rust build pipeline: pre-import staleness checks with skip/force/fail flags, shared helpers, and a `scripts/check_rust_extension.py` reporter; added tests for staleness detection.
- Rust compile flow now less noisy in normal operation (debug lock prints removed); compile attempts still logged when rebuilding.
- Balance hysteresis now applied centrally in core bot update_balance; exchange fetch_balance implementations return raw balances.
- Added configurable `live.balance_hysteresis_snap_pct` (default 0.02); set 0.0 to disable balance hysteresis entirely.
- Optimizer: bounds now support optional step size `[low, high, step]` for grid-based optimization; stepped parameters stay on-grid through sampling, crossover/mutation, and Pareto storage.
- Live: added `live.candle_lock_timeout_seconds` to control how long CandlestickManager waits for per-symbol candle locks when multiple bot instances share the same cache (default 10s).
- Rust orchestrator JSON API for unified order planning across live and backtest.
- Backtest HLCV preparation pipeline now routes through CandlestickManager with shared warmup utilities.

### Changed
- Backtest fills now include signed `wallet_exposure` and `twe_long`/`twe_short`/`twe_net` (replacing the previous `total_wallet_exposure` fill column).
- Pareto explorer: default metrics for X/Y/histogram, scenario comparison, param scatter, correlation heatmap, and Closest Config now derive from `config.optimize.scoring` and `config.optimize.limits` instead of first-alphabetical metrics; Closest Config table no longer shows raw *_mean/_min/_max/_std stat columns by default.
- Suite summaries are leaner: redundant metric dumps removed; canonical metrics schema persisted alongside per-scenario timing.
- Pareto pruning preserves per-objective extremes when enforcing max size.
- Hyperliquid combined balance/position caching test isolated stubs to avoid polluting the rest of the suite.
- Separated `fetch_positions` and `fetch_balance` responsibilities across all exchange wrappers (each now returns only positions or only balance) and added `update_positions_and_balance()` helper in the core bot to refresh both concurrently.
- `update_positions_and_balance()` now runs balance and positions concurrently, logs position changes after both complete, and then emits a single balance-change event so equity logging always uses fresh positions.
- KuCoin `get_order_execution_params` now aligns with the latest CCXT payload requirements so orders always include the correct margin/position parameters after the CCXT upgrade.
- Added Pareto regression test to ensure per-metric extremes remain present after front pruning.
- Metric adg_pnl now includes fees paid, effectively making it net pnl instead of gross pnl.
- Risk management docs refreshed and consolidated; new notes on unstucking, WEL/TWEL enforcers, and conditional stop-loss concepts.
- Balance updates now keep the previous value on fetch failures (no more transient zero balances); warnings are logged and the standard restart-on-errors flow handles persistent issues.
- EMA log spam reduced: volume/log-range EMA summaries only emit when rankings change, keeping live logs quieter.
- Suite configuration is canonical under `backtest.suite` for both backtesting and optimizer runs; `optimize.suite` (if present) is ignored and removed during config normalization.
- Live orchestrator compare mode now derives all EMA inputs from a single per-symbol candle snapshot (1m + 1h), reducing redundant candle-lock contention and false compare failures in multi-bot deployments.
- Live order generation now runs exclusively through the Rust orchestrator; legacy Python order planning paths are removed.
