# Changelog

All notable user-facing changes will be documented in this file.

## Unreleased

- Updated the hardcoded schema defaults and mirrored example config to a new trailing-grid `n_positions = 7` profile from `tmp/candidate.json`; the canonical example file is now `configs/examples/default_trailing_grid_long_npos7.json`. Default approved coins, suite scenarios, optimizer bounds, and optimizer scoring/limit templates were refreshed with canonical `*_strategy_eq` metric names and day-based duration metrics while keeping backtest defaults at `candle_interval_minutes = 1`, `end_date = "now"`, and `suite_enabled = false`.
- Added day-denominated backtest analysis metrics mirroring the existing duration metrics: high exposure, peak recovery, position held, and position unchanged outputs now keep their `*_hours*` fields and also expose equivalent `*_days*` fields.
- Backtest `drawdown_worst_mean_1pct` and `drawdown_worst_mean_1pct_strategy_eq` now compute drawdowns from the full-resolution equity curve first, then average the worst 1% of daily worst drawdowns. This better distinguishes isolated max-drawdown spikes from sustained drawdown regimes.
- Backtest BTC collateral is now initialized at the first active trading step instead of at the beginning of EMA warmup data, so warmup-period BTC price movement no longer changes starting account equity.
- Added `strategy_equity` to backtest `balance_and_equity.csv.gz` artifacts so the collateral-agnostic strategy-equity curve is available alongside balance and USD/BTC equity.
- Added `passivbot tool inspect-ohlcvs` for diagnosing the v2 OHLCV store under `caches/ohlcvs/`. The tool can summarize catalog counts and symbols, or inspect one symbol's bounds, chunk validity, persistent gaps, and recent fetch attempts.
- Renamed collateral-agnostic strategy-equity analysis metrics to canonical `*_strategy_eq` names and deprecated the old `*_strategy_pnl_rebased` / `*_hsl` metric names as input aliases. New `analysis.json` output uses canonical names, while optimizer, Pareto, limits, aggregate config, and visibility filters still resolve old stored result keys. `peak_recovery_hours_pnl` now uses net realized PnL (`pnl + fee_paid`) and includes the open tail from the last realized-PnL peak to the end of the backtest.
- Fixed suite-mode limit semantics so `passivbot optimize` and `passivbot tool pareto` now resolve omitted `stat=` the same way: explicit `stat=` still wins, otherwise both defer to `backtest.aggregate.<metric>`, then `backtest.aggregate.default`, then `mean`. This removes the old optimizer-only behavior where `>` silently implied `min` and `<` silently implied `max`.
- Reduced optimizer startup memory pressure when warming from large starting-config sets. Starting configs now stream into quantization instead of being fully materialized up front, and pymoo worker evaluations now reuse per-worker evaluator state plus metrics-only backtests instead of serializing full evaluator payloads and full backtest histories for every candidate.
- Upgraded the pinned `ccxt` dependency from `4.5.22` to `4.5.48` and added a dedicated CCXT upgrade validation workflow with live snapshot capture/diff tooling plus offline contract fixtures for upgrade drift.
- Fixed backtest `pnls_max_lookback_days` rolling realized-PnL reconstruction to match live semantics exactly: both now derive peak/current PnL stats from the active lookback window only by filtering in-window fills and recomputing cumulative realized PnL from that filtered sequence. This fixes overstated auto-unstuck allowance and related risk gating drift caused by the old rebased rolling-peak implementation.
- Fixed all-zero `forager_score_weights` configs to normalize to EMA-readiness-only ranking consistently across Python config prep, Rust selection, and optimizer inputs instead of drifting into ambiguous fallback behavior.
- Stopped hydrating omitted `config.bot.{long,short}` fields from schema-tuned bot defaults in legacy/current configs. Newly omitted feature-style params now hydrate to explicit off/compatibility values with config logs, sparse disabled sides remain loadable, legacy `n_closes` and `min_markup` aliases are preserved, and the Rust parser now fails loudly instead of silently supplying bot-key fallbacks.
- Hyperliquid live balance reconciliation no longer republishes bot-managed resting-order reserve after `fetch_open_orders()`. This removes the old `REST`/`REST+open_orders` balance oscillation path that could trigger self-induced order-size churn.
- Fixed OHLCV cache backfills so earlier requested ranges are no longer silently suppressed just because later shards already exist on disk. CandlestickManager now separates earliest observed cached candles from authoritative exchange-history lower bounds, migrates stale legacy `pre_inception` gaps out of old indexes, and warns when a requested span is clipped by an authoritative start boundary.
- Live bots now watch for newer Passivbot-managed open orders they did not emit during the current runtime and stop after repeated detections within a rolling window. This ignores manual/non-Passivbot orders and older inherited orders, reducing the chance of two Passivbot instances silently competing on the same account indefinitely.
- Staged live bots now route orchestrator latest-price reads through `CandlestickManager`, and `CandlestickManager.get_last_prices()` now uses cheap cache hits plus one bulk ticker snapshot when safe before any per-symbol fallback. This materially reduces staged live market-data call bursts on exchanges like Bybit.
- Live runtime shutdown is now cleaner: Ctrl-C and stop-signal paths stop execution sooner, await cancelled maintainer tasks during shutdown, exit restart cooldowns promptly, and classify Bybit `110001 / order not exists or too late to cancel` as the expected benign cancel race instead of logging a noisy error traceback.
- Fixed CLI `live.approved_coins` / `live.ignored_coins` file overrides so live reload keeps the original file path in `_coins_sources` instead of freezing the first parsed snapshot. Mid-run edits to `-s path/to/file` coin lists now take effect correctly.
- Fixed optimizer Pareto artifact persistence so saved `pareto/*.json` candidates now preserve the exact evaluated bot parameter values instead of being re-rounded again inside `ParetoStore`. This restores replay fidelity between `passivbot tool pareto` selections and standalone `passivbot backtest` runs of the selected file.
- Fixed `passivbot optimize/backtest -cim/--candle-interval-minutes` type handling so integral values stay integers through the Python/Rust backtest boundary. This fixes crashes like `TypeError: 'float' object cannot be interpreted as an integer` when using `-cim 2`.
- Hyperliquid non-unified (`dexAbstraction`) accounts now hard-fail if any HIP-3/non-standard perp symbol appears in effective `approved_coins` or live exchange state. Those symbols now require `unifiedAccount` mode instead of being partially skipped or partially supported.

## v7.9.1 - 2026-04-13
- Removed the legacy `python src/downloader.py ...` entrypoint. Use `passivbot download ...` for OHLCV cache warming.
- Added formal top-level `config_version` schema tagging starting at `v7.9.0`. Canonical defaults and the mirrored example config now carry the schema version, older configs log a migration attempt during load, and the loader upgrades them to the current schema version.
- Backtests now read `market_orders_allowed`, `market_order_near_touch_threshold`, and `pnls_max_lookback_days` from `config.live` only. `config.backtest` no longer accepts those fields, which avoids silent drift between live and backtest behavior.
- Pre-v7.9 backtests did not correctly observe `pnls_max_lookback_days`, and they also did not simulate ordinary market-order execution. v7.9+ treats both as backtest correctness fixes rather than preserving bug-compatibility via migrated `backtest` overrides.
- `live.pnls_max_lookback_days` now uses one consistent contract across live risk logic, HSL, plotting, and backtests: `0` means the minimal effective lookback for that path's native sampling resolution, positive numbers mean that many rolling days, and `"all"` means full available history. Full-history live fill refreshes also stay incremental once the cache is warm instead of forcing a full refetch every cycle.
- Added `bot.{long,short}.entry_grid_inflation_enabled` as a short-term compatibility switch for inflated grid re-entries. The default remains `true` in this release to preserve current behavior, but config parsing now warns that inflated grid re-entries are scheduled for deprecation and that the canonical forward path is cropped-only grid re-entries.
- `passivbot optimize --help-all` now exposes fixed per-side bot runtime overrides for `entry_grid_inflation_enabled`, `hsl_enabled`, `hsl_orange_tier_mode`, and `hsl_panic_close_order_type` without making them optimizer dimensions, and `optimize.bounds` now rejects trying to tune those non-numeric bot fields.
- Restored `backtest.visible_metrics` for standalone backtest terminal output filtering. `null` now shows optimize-derived metrics, `[]` shows all, and explicit lists add extra metrics without affecting the full saved `analysis.json`.
- Fixed `CCXTBot.create_ccxt_sessions()` using the generic exchange name (e.g. `binance`) instead of the futures-specific CCXT id (`binanceusdm`). This caused `load_markets()` to unnecessarily fetch COIN-margined markets from `dapi.binance.com`, and a timeout on that endpoint would cascade-fail all symbol trade fetches and open order updates.
- Fixed `BinanceFetcher._fetch_symbol_trades` sending future `endTime` (now+1h) and using a tight 7-day safety margin (0.1%), causing Binance `-4181 "Invalid start time"` errors for symbols with sparse trades. Removed the +1h extension and widened the margin to 1%.
- Hyperliquid live sizing now compensates for missing cross-margin reserve in `fetch_balance()`: HIP-3 stock-perp positions can restore their hidden `marginUsed`, and Passivbot-managed resting non-reduce-only entry orders can restore reserved margin on both HIP-3 and flat standard perps. This prevents the bot from misreading its own reserved margin as equity loss and churning order sizes in cancel/replace loops, while still ignoring external/manual orders.
- Backtest/optimizer HLCV dataset caches under `caches/hlcvs_data/` now use descriptive directory names with exchange, coin label/count, actual dataset date range, and the cache hash suffix. Existing legacy hash-only cache directories still load unchanged.
- Config validation now hard-fails invalid `bot.long.unstuck_ema_dist <= -1.0` and `bot.short.unstuck_ema_dist >= 1.0` instead of silently disabling auto-unstuck with a non-positive EMA trigger price. The same guard now rejects optimize bounds that would generate those invalid values.
- Fixed Bybit `closed-pnl` pagination storms that caused retCode:10006 rate-limit errors every ~15 minutes. Fill lookback coverage is now derived from `FillEventsManager` cache metadata instead of a session-local flag, so once an open-ended lookback has been checked successfully the bot reuses incremental refreshes across restarts even when the early lookback window legitimately contains no fills.
- Applied exchange-aware EMA bundle pacing in `_load_orchestrator_ema_bundle`. Strict exchanges use the configured inter-symbol delay to avoid hour-boundary candle bursts, while exchanges with zero pacing keep the original concurrent `asyncio.gather` behavior instead of being globally serialized.
- Added random jitter (0–120s) to the hourly `init_markets` cycle so multiple bots on the same VPS don't fire heavy API bursts simultaneously.
- `passivbot live` now persists logs to a timestamped file under `logs/` by default, using `config.logging` for the on/off switch and file-rotation settings, and also refreshes `logs/{user}.log` as a stable alias to the current run for monitor tooling. This makes the built-in live workflow self-logging without needing `run_with_logging.py`.
- Added a canonical live-container runtime contract around `Dockerfile_live`, a thin `container/entrypoint.sh` wrapper, env-generated `api-keys.json` support, env-driven config overrides, and a documented Compose/Railway deployment path that reuses the normal `passivbot live` CLI instead of maintaining platform-specific baked configs.
- Restored `passivbot live --user` / `-u` as the curated shorthand for `live.user`, so existing live-run workflows using `-u account_name` work again and the alias is visible in the default live help output.
- `passivbot live -h` now shows a curated shorthand for `live.pnls_max_lookback_days` as `--pnls-max-lookback-days` / `-pmld` in the default help output instead of exposing it only via `--help-all` and the raw dotted config flag, and the flag now accepts either a non-negative float or `"all"`.
- Added `passivbot tool pareto`, a CLI Pareto front explorer that filters JSON Pareto members with optimizer-style limit expressions, defaults to the newest local `optimize_results/.../pareto` when no path is given, accepts either a run dir or `pareto/` dir, and selects a single candidate using knee, reference-point, ideal-point, weighted utility, lexicographic, or outranking methods with optional shortlist and JSON output. It now also shows the retained front's ideal point, and `-o` / `--objectives` can use stored metrics outside the original `optimize.scoring` list when their min/max direction is known.
- Changed `passivbot tool pareto` to default to the `ideal` selection method instead of `knee`.
- Fixed backtest post-processing for zero-fill runs. When a period produces no fills but still has equity samples, balance/equity resampling now keeps a `DatetimeIndex` and no longer crashes during analysis/plot generation with larger `backtest.balance_sample_divider` values.
- Fixed first-ohlcv timestamp cache handling for newly listed coins. Cached `0.0` entries are now treated as unresolved and refreshed, so optimize/backtest candle downloads correctly clamp fetch start to the coin's actual listing history instead of wasting time paging from much earlier dates.
- Fixed optimizer/backtest liquidation reporting to use an explicit Rust-provided `analysis.liquidated` flag instead of inferring liquidation from `drawdown_worst`, avoiding false positives after runs that made a new equity peak before hitting the liquidation floor.
- Added trade-level backtest metrics for completed positions: `win_rate`, `win_rate_w`, and `trade_loss_{max,mean,median}`. These measure completed-trade outcomes from open-to-flat realized PnL and normalize loss metrics by balance at trade open.
- Added optimizer-facing backtest ratio metrics `paper_loss_ratio`, `paper_loss_mean_ratio`, `exposure_ratio`, and `exposure_mean_ratio`, plus weighted `_w` variants. These measure growth relative to unrealized equity-vs-balance drawdown and actual wallet exposure.
- `live.approved_coins` now supports explicit per-side `"all"` entries such as `{"long": ["BTC"], "short": "all"}`. Missing or explicit empty side values now stay disabled instead of being backfilled from schema defaults. `live.empty_means_all_approved` is no longer part of the canonical config shape; older configs still migrate with a deprecation warning, and globally empty legacy inputs are converted to `approved_coins = "all"`.

### Upgrade Notes
- Reinstall after pulling this release. `passivbot` now validates the active environment and the loaded Rust extension more aggressively, so stale editable installs or stale shell shims are more likely to fail loudly instead of continuing silently. Use `python3 -m pip install -e .` for live-only setups or `python3 -m pip install -e ".[full]"` for backtest/optimize setups, and rebuild with `maturin develop --release` if needed.
- `optimize.backend` now defaults to `pymoo`, so optimization users need the full install profile with the new `pymoo` dependency.
- `configs/template.json` is no longer the canonical starting point. Use `configs/examples/default_trailing_grid_long_npos7.json` or omit the config path to start from the in-code defaults in `src/config/schema.py`.
- The local monitor publisher now ships enabled by default in the canonical schema. Set `monitor.enabled = false` if you do not want snapshot/event files written under `monitor/`.
- `live.max_realized_loss_pct` now defaults to `1.0`, which effectively disables the realized-loss gate unless you set a tighter value explicitly.

### Added
- **Pymoo optimizer backend** - Optimization can now run with `optimize.backend: pymoo` in addition to DEAP, with shared backend dispatch and dedicated backend coverage.
- **Pymoo NSGA-III config is now live** - `optimize.pymoo.algorithm`, nested `optimize.pymoo.shared.*`, and NSGA-III reference-direction settings are now actually honored at runtime, with auto-sized NSGA-III reference directions and `"auto"` per-variable mutation probability support.
- **Repro and sync sidecar tools** - Added `src/repro_harness.py`, `src/analysis_visibility.py`, `src/tools/capture_optimize_memory.py`, root-level `sync_tar.py`, and `vpssync.sh` for replay/debug/VPS workflows.
- **Standalone trailing diagnostics explorer** - Added `src/tools/trailing_diagnostics.py` plus reusable helpers for recomputing next-entry and next-close trailing behavior from `config + monitor snapshot` or manual inputs.
- **HSL events per-year metrics** - Backtest HSL analysis now also exports `hard_stop_triggers_per_year` and `hard_stop_restarts_per_year` so runs with different date ranges can be compared more directly without losing the absolute trigger/restart counts.
- **Fake-live exchange harness for HSL replay** - Added a deterministic `fake` exchange, `src/tools/run_fake_live.py`, and scenario-driven tests/docs so live HSL RED, cooldown restart, terminal halt, and cooldown-position policies can be replayed locally against scripted candles and manual interventions.
- **Opt-in live monitor publisher** - Added a local monitor publisher with on-disk snapshots, event streams, and retained fill/price/candle history, plus basic live bot integration for startup, balance, order, fill, and shutdown events.
- **Read-only monitor relay** - Added a local `monitor-relay` tool exposing monitor snapshots and streamed event/history tails over HTTP and websocket, including recent-message replay on connect.
- **Browser monitor dashboard** - The monitor relay now also serves `GET /dashboard` with a read-only web dashboard that bootstraps from `/snapshot`, stays live via `/ws`, shows summary/focus/positions/trailing/forager/unstuck/recent activity panels, and supports quick focus changes by clicking symbol-bearing rows.
- **Monitor web wrapper** - Added `passivbot tool monitor-web` to reuse or launch the local relay and keep the browser dashboard available from one command.
- **Terminal monitor TUI** - Added a local `monitor-tui` tool consuming the relay for current-state panels, live recent activity, focus cycling, pause/resume, and screen dumps.
- **Monitor dev wrapper** - Added a `monitor-dev` helper that reuses or launches the local relay and opens the terminal monitor with the newest bot log tailed by default.

### Changed
- **Optimizer scoring now has explicit min/max goals** - `optimize.scoring` is normalized to `{metric, goal}` entries, optimizer engines receive minimization-space values internally, and user-facing logging/Pareto tools now show raw metric values with named objectives instead of signed `w_i` fields. Legacy string-list scoring configs and legacy Pareto result files remain readable.
- **Config loading now uses a canonical staged pipeline** - Defaults now come only from in-code schema, omitted CLI configs instantiate schema defaults directly, `load_config()` / `format_config()` normalize to canonical user-facing keys without leaking runtime `filter_*` aliases, runtime aliasing moved into explicit compilation helpers, and the named example profile now lives at `configs/examples/default_trailing_grid_long_npos7.json`.
- **Realized-loss gate now ships disabled by default** - `live.max_realized_loss_pct` now defaults to `1.0`, so the gate is opt-in unless the operator explicitly chooses a tighter peak-relative realized-loss floor.
- **Executable min-cost filtering now matches actual order sizing** - `filter_by_min_effective_cost` now uses the executable minimum entry qty after `qty_step` rounding instead of raw `min_qty/min_cost` metadata, and CCXT markets reporting nonpositive `min_qty` now clamp it to `qty_step`. This fixes GateIO symbols such as `SOL/USDT:USDT` being admitted when the smallest executable order would exceed the intended initial entry size.
- **BTC-denominated backtest metrics now always use BTC equity** - `*_btc` metrics are now computed from BTC-denominated balance/equity even when `backtest.btc_collateral_cap = 0`, instead of mirroring the USD analysis. This makes metrics like `adg_btc` and `gain_btc` informative as BTC-relative performance measures for cash-collateral runs as well.
- **ADG terminal smoothing simplified** - Backtest `gain`/`adg` now smooth the terminal value by taking the mean of the last up to 3 daily equity samples instead of running an EMA over the full daily-equity series. This preserves end-of-run drawdown smoothing while reducing computation.
- **Pymoo NSGA-III population defaults are now auto-sized** - `optimize.population_size: null` now means “use the NSGA-III reference-direction count” for pymoo/NSGA-III runs, and template/config defaults now leave that field null instead of forcing a fixed 500/1000 population.
- **Unified `passivbot` CLI added** - Passivbot now installs a `passivbot` command with subcommands such as `passivbot live`, `passivbot backtest`, `passivbot optimize`, `passivbot download`, and `passivbot tool ...`. Existing direct script entrypoints like `python3 src/main.py ...` remain supported for backwards compatibility.
- **CLI help is now task-oriented by default** - `passivbot live -h`, `passivbot backtest -h`, and `passivbot optimize -h` now show curated, grouped common flags by default, while `--help-all` exposes the full advanced/raw override surface.
- **Install profiles split into `live`, `full`, and `dev`** - `pip install -e .` now targets a lightweight live-trading environment, while `pip install -e ".[full]"` adds backtesting/optimization/tooling dependencies and `pip install -e ".[dev]"` adds contributor-focused docs/lint extras on top.
- **Equity hard-stop config moved under `bot.common`** - Shared HSL settings now live at `bot.common.equity_hard_stop_loss`, with config formatting migrating legacy `live.equity_hard_stop_loss` inputs and optimizer bounds to the new location.
- **Live HSL cooldown interventions are now configurable** - RED cooldown no longer blocks the runtime in one wait path. Live now keeps the halt active while enforcing `live.hsl_position_during_cooldown_policy` (`panic`, `normal`, `manual`, `tp_only`, or `graceful_stop`) until cooldown expires or trading is resumed.
- **Browser monitor is now multi-bot first-class** - The web dashboard now consumes the multiplexed relay feed directly, shows a dense overview for all active bots in one page, and lets operators switch focused bot detail views without separate relay instances or per-bot dashboard sessions.
- **Monitor relay presence is now sticky** - Auto-discovered bots now degrade from `active` to `stale` before being pruned, and the browser overview keeps a stable bot order instead of reshuffling on every freshness blip.
- **HSL cooldown contracts are now documented explicitly** - Added a dedicated cooldown-contract reference covering RED replay, restart, and cooldown-position intervention behavior so operator/runtime expectations are easier to verify against logs.

### Fixed
- **Backtest rolling `pnls_max_lookback_days` peaks now actually expire** - Backtest risk consumers such as auto-unstuck and the realized-loss gate no longer compare the current rolling realized-PnL window against a stale all-time maximum of that rolling series. The Rust backtest now ages rolling realized-PnL state out by time even during fill droughts and uses the true in-window peak/current pair for `pnls_max_lookback_days > 0`.
- **Exchange config refresh now retries per symbol** - Live bots no longer mark exchange-config updates as complete when a symbol fails or hits a rate limit; failed symbols now back off and retry while successful symbols continue to progress.
- **Live forager key mapping** - Live runtime now consistently reads canonical `forager_*` config keys while still exporting Rust orchestrator payload fields under the internal `filter_*` names expected at the Python/Rust boundary.
- **Pymoo optimizer now records results incrementally during each generation** - Completed pymoo evaluations are now drained in the main process as workers finish, immediately written to `all_results.bin` / Pareto storage, and stripped from the generation payload before pymoo continues. This improves progress visibility and avoids retaining full metrics payloads until the entire generation completes.
- **Optimizer multiprocessing now works under the unified CLI on spawn-based platforms** - `passivbot optimize ...` no longer fails at pool startup with a pickling error for the SIGINT worker initializer when launched through the unified CLI on macOS/Python spawn multiprocessing.
- **CLI now guards against wrong-environment `passivbot` launches** - When `VIRTUAL_ENV` or `CONDA_PREFIX` is active but the resolved `passivbot` command is running under a different Python interpreter, the console entrypoint now re-execs into the active environment's `passivbot` script when available, or fails loudly with explicit mismatch diagnostics and install guidance instead of silently running the wrong install.
- **Startup exchange-config timeout handling** - Live startup now gives CCXT exchange sessions a 30s default timeout and retries `update_exchange_config()` on transient network/request timeouts during `init_markets()`, reducing cold-boot failures without suppressing non-retryable errors.
- **Hyperliquid HIP-3 margin-mode detection for `XYZ-...` symbols** - Hyperliquid stock perps exposed by CCXT as `XYZ-...` or `XYZ:...` now correctly force isolated margin mode, preventing erroneous cross-margin config calls that could lead to repeated duplicate entry submissions on stock-perp markets such as `XYZ100`.
- **Hyperliquid HIP-3 state sync for positions and open orders** - Hyperliquid stock-perp positions and open orders now use dex-scoped CCXT queries for HIP-3 symbols instead of relying only on the default `fetch_balance()` / global open-orders routes. This fixes bots repeatedly re-entering because filled HIP-3 positions or resting HIP-3 orders were invisible to local state reconciliation.
- **Hyperliquid HIP-3 isolated trading disabled for now** - Passivbot now treats Hyperliquid HIP-3 as cross-only for live trading until isolated-margin support is properly designed. Cross-capable HIP-3 markets remain tradable in cross mode, isolated-only HIP-3 markets are skipped with warnings, and existing isolated HIP-3 positions or open orders fail startup loudly instead of running in a risky partial-support mode.
- **Stock-perp source-dir resolution in HLCV preparation** - Hyperliquid stock-perp backtests now resolve source-dir symbols against loaded market metadata instead of failing on cache-map casing mismatches such as `xyz:AAPL` vs `XYZ-AAPL/USDC:USDC`.
- **Editable-install Rust freshness checks now find `maturin develop` outputs reliably** - The stale-extension safety check now detects root-level `site-packages/passivbot_rust...so` installs created by `maturin develop`, so `passivbot ...` no longer loops on “stale even after recompilation” while still using an old `src/passivbot_rust...so` shadow copy.
- **Backtest HSL panic execution and metrics export** - Account-level RED panic now forces panic mode on all symbols/sides in Rust backtests, `panic_close_order_type="market"` is simulated as next-bar taker execution instead of limit-only behavior, and `hard_stop_*` analysis metrics are exported once as shared metrics rather than duplicated into `_usd`/`_btc` variants.
- **Rust-owned market-vs-limit execution intent** - Rust orchestrator now decides whether eligible non-panic orders should be emitted as `limit` or `market` using one shared near-touch threshold and market-crossing rules. Live now consumes that Rust execution intent directly, and backtests use the same intent for guaranteed market fills with slippage and taker fees.
- **Backtest taker-fee execution for market fills** - Backtest market executions now charge taker fees instead of maker fees, respect optional `backtest.taker_fee_override`, and record a `liquidity` column (`maker` / `taker`) in `fills.csv`. Simulated market fills remain guaranteed once selected, with execution price shifted by `backtest.panic_market_slippage_pct`.
- **Backtest HSL drawdown visualization** - Backtests now output `hard_stop_drawdown.png` alongside the existing summary plots when account-level HSL is enabled. The new plot shows raw drawdown, EMA-smoothed drawdown, the active HSL trigger score, tier thresholds, and RED-threshold proximity over time. `--disable_plotting` also supports a dedicated `hard_stop` plot group.
- **Backtest HSL EMA span fallback** - Backtests no longer fail when `bot.common.equity_hard_stop_loss.ema_span_minutes` is smaller than `backtest.candle_interval_minutes`. Sub-interval spans now fall back to a one-sample EMA, which disables smoothing and makes the HSL score follow raw drawdown.
- **HSL no-restart threshold semantics** - Values of `bot.common.equity_hard_stop_loss.no_restart_drawdown_threshold` below `red_threshold` are now clamped up to `red_threshold` in live, backtest, and optimizer flows. Stop events now treat `drawdown_raw >= no_restart_drawdown_threshold` as terminal, so setting both thresholds equal makes the first RED halt non-restarting.
- **Backtest HSL analysis metrics expanded and clarified** - Added account-level HSL metrics for yellow/orange/red time share, RED halt duration, trigger drawdown, panic-close realized loss, flatten time, and restart-to-retrigger rate. Also renamed the old ambiguous halt-loss metric to `hard_stop_halt_to_restart_equity_loss_pct`.
- **HLCV fetch logging and cache-root hygiene** - CCXT candle fetch progress logs now include the actual returned candle range (`first`/`last`) instead of only the requested `since`, and CandlestickManager now quarantines invalid root-level daily shard files or `index.json` debris found directly under `caches/ohlcv/{exchange}/{timeframe}` so mixed/corrupt cache roots stop masquerading as symbol data.

## v7.8.4 - 2026-03-06

### Changed
- **Dual balance routing (raw vs hysteresis-snapped)** - Live and orchestrator flows now carry both `balance_raw` (raw wallet balance) and `balance` (hysteresis-snapped balance). Sizing/order-shaping paths use snapped balance, while risk/accounting paths use raw balance (including realized-loss gate peak/floor checks, TWEL entry/auto-reduce gating, and auto-unstuck allowance calculations). This applies consistently across live and backtest via Rust orchestrator input.
- **WEL denominator behavior split by mode** - Live now uses a hard fixed denominator for per-symbol WEL (`total_wallet_exposure_limit / config.bot.{pside}.n_positions`), removing runtime denominator drift from open-position count. Backtests now expose `backtest.dynamic_wel_by_tradability` (default `true`): when enabled, WEL uses tradability-aware denominator growth (`min(n_positions, n_tradable_max)`) based on coins with real candles, and does not shrink after delistings; when disabled, backtests use the same fixed denominator as live.
- **Bulk price fetch for Hyperliquid** - `calc_ideal_orders` now uses a single `allMids` API call to get prices for all symbols instead of individual `get_current_close` calls per symbol (1 call vs ~70). Falls back to per-symbol fetches for non-Hyperliquid exchanges or on error.
- **Sequential margin mode setting for Hyperliquid** - Margin mode and leverage API calls are now sequential with a small delay instead of being fired in parallel, reducing API burst on coin changes.
- **Equity hard-stop framework (live+backtest)** - Added nested equity hard-stop config (now under `bot.common.equity_hard_stop_loss`) with threshold, EMA span in minutes, configurable yellow/orange tier ratios, orange mode selector, panic close order type, plus Rust drawdown/tier state machine module, backtest rolling-peak enforcement using `pnls_max_lookback_days`, and live runtime hooks for tier tracking/latching with RED supervisory flatten-until-confirmed-flat behavior.

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
