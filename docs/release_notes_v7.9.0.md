# Release Notes for v7.9.0

These notes describe the user-facing changes from `master` to `v7.9.0`.

## Highlights
- Equity hard-stop loss is now a full live + backtest feature with Rust-owned runtime handling, richer metrics, and deterministic fake-live replay coverage.
- Optimization now supports `pymoo` as a first-class backend and uses it by default, with explicit objective goals and live NSGA-II / NSGA-III configuration.
- A new `passivbot tool pareto` explorer makes it easier to inspect and select Pareto candidates from local optimize results.
- Config loading now runs through a canonical staged pipeline with in-code schema defaults, compatibility migrations for older config keys, and a new canonical example config path.
- Passivbot now ships a unified `passivbot` CLI with clearer default help, install-profile separation, stronger environment mismatch detection, and stricter Rust freshness checks.
- A built-in monitor stack is now available: publisher, relay, browser dashboard, TUI, and helper wrappers.

## Upgrade Notes
- Reinstall after upgrading.
  `python3 -m pip install -e .`
  or
  `python3 -m pip install -e ".[full]"`
- If Rust looks stale or the wrong binary is being used, run:
  `maturin develop --release`
- `optimize.backend` now defaults to `pymoo`, so optimize users need the full install profile with the new dependency set.
- `configs/template.json` is no longer the canonical starting point. Use `configs/examples/default_trailing_grid_long_npos10.json` or omit the config path to start from schema defaults.
- The canonical schema enables the local monitor by default. Set `monitor.enabled = false` if you do not want snapshot and event files written locally.
- `live.max_realized_loss_pct` now defaults to `1.0`, so the realized-loss gate is opt-in unless you set a tighter value explicitly.

## What's New

### Equity Hard Stop Loss
- Added the account-level equity hard-stop framework to live and backtest, with Rust-managed drawdown state, tier tracking, RED latching, cooldown restart logic, and no-restart handling.
- Added richer HSL metrics and plots, including per-year trigger and restart counts, time-share metrics, halt and restart metrics, panic-close metrics, and `hard_stop_drawdown.png`.
- Added deterministic fake-live replay tooling and scenarios so RED halt, cooldown behavior, manual intervention, and restart policies can be tested locally.
- Clarified HSL semantics around no-restart threshold clamping, sub-interval EMA fallback, and market panic execution.

### Optimization and Pareto Tooling
- Added the `pymoo` backend alongside DEAP and made it the default optimizer backend.
- Activated real runtime support for nested `optimize.pymoo.*` settings, including `algorithm`, shared crossover and mutation settings, and NSGA-III reference-direction configuration.
- Changed optimizer scoring to explicit `{metric, goal}` specs instead of implicit sign-based weights, while keeping legacy string-list configs readable.
- Added auto-sized NSGA-III population defaults when `optimize.population_size` is `null`.
- Added `passivbot tool pareto`, a CLI Pareto explorer for filtering local Pareto candidates with optimizer-style limit expressions and selecting a single candidate using methods such as knee, reference-point, ideal-point, weighted utility, lexicographic, or outranking selection.
- The Pareto explorer can now default to the newest local `optimize_results/.../pareto`, accept either a run directory or a `pareto/` directory, show the retained front's ideal point, and use stored metrics outside the original `optimize.scoring` list when their direction is known.

### Configs, CLI, and Install Flow
- Moved canonical defaults to `src/config/schema.py` and made schema defaults the no-config-path behavior for `live`, `backtest`, and `optimize`.
- Added staged config normalization, runtime compilation helpers, migration logging, and backward-compatibility renames for older config keys.
- Replaced the old implicit template workflow with `configs/examples/default_trailing_grid_long_npos10.json`.
- Added the unified `passivbot` CLI, curated default help, `--help-all`, and install-profile guidance for `live`, `full`, and `dev`.
- Restored `passivbot live --user` / `-u` as the curated shorthand for `live.user`, and added a curated shorthand for `live.pnls_max_lookback_days` as `--pnls-max-lookback-days` / `-pmld` in the default live help.
- Added environment mismatch detection so stale shell shims and wrong-entrypoint installs fail loudly or re-exec into the active environment.
- Tightened Rust extension freshness checks to reduce silent stale-binary runs.

### Monitoring and Diagnostics
- Added the local monitor publisher for bot snapshots, event streams, and retained fill, price-tick, and candle history.
- Added the read-only monitor relay with websocket streaming and recent-message replay.
- Added the browser dashboard, terminal TUI, `monitor-web`, and `monitor-dev`.
- Added standalone trailing diagnostics tooling for recomputing trailing-entry and trailing-close behavior from saved snapshots or manual input.
- Added repro and sync sidecar tools for investigation and deployment workflows.

### Backtest and Runtime Behavior
- Rust now owns more of the market-vs-limit execution intent, and live plus backtest now consume the same shared logic.
- Market fills in backtests now use taker fees, optional taker-fee override support, and explicit maker/taker liquidity labeling in `fills.csv`.
- Backtest BTC metrics now always use BTC equity instead of mirroring USD analysis when `btc_collateral_cap = 0`.
- ADG terminal smoothing now uses the last up to three daily samples instead of an EMA over the full run.
- Executable min-cost filtering now uses actual rounded executable size rather than raw market metadata, improving forager tradability filtering.
- First-timestamp cache handling for newly listed coins is more robust, avoiding fetches from invalid early dates.

### Exchange and Market Coverage
- Hyperliquid HIP-3 handling is more robust across margin-mode detection, state sync, and source-dir resolution for stock-perp backtests.
- Isolated HIP-3 live trading is explicitly blocked for now instead of partially supported.

## Discord Forum Announcement Draft
Passivbot v7.9.0 is out.

This release covers the full user-facing diff from the current `master` branch to v7.9.0, with major changes across live trading, backtesting, optimization, config handling, CLI UX, and monitoring.

Important upgrade note first:
- After pulling, reinstall Passivbot in your active environment.
- Live-only: `python3 -m pip install -e .`
- Backtest / optimize / research: `python3 -m pip install -e ".[full]"`
- If Rust looks stale, run `maturin develop --release`

Main changes in v7.9.0:
- Equity hard-stop loss is now a full live + backtest feature with Rust-owned runtime state, cooldown handling, improved RED supervision, richer metrics, and deterministic fake-live replay scenarios.
- Optimization now supports `pymoo` as a first-class backend and uses it by default. NSGA-II / NSGA-III settings under `optimize.pymoo.*` are now actually honored, and NSGA-III population sizing can be auto-derived from reference directions.
- Optimizer scoring is now explicit via `{metric, goal}` entries instead of implicit signed weights. Legacy scoring configs still load.
- A new `passivbot tool pareto` explorer can filter local Pareto candidates with optimizer-style limit expressions and select a single config using knee, reference-point, ideal-point, weighted utility, lexicographic, or outranking methods.
- Config loading now uses a canonical staged pipeline with in-code schema defaults, compatibility migrations for older keys, and a new canonical example config at `configs/examples/default_trailing_grid_long_npos10.json`.
- Passivbot now has a unified `passivbot` CLI with cleaner default help, `--help-all`, install-profile separation, restored `passivbot live -u`, and much stronger environment / stale-extension detection.
- A new local monitoring stack is included: publisher, relay, browser dashboard, TUI, `monitor-web`, and `monitor-dev`.
- Rust and Python execution behavior are better aligned across live and backtest, especially for market-vs-limit order intent, HSL panic behavior, and market-fill fee modeling.
- Backtests now include better HSL plots and metrics, cleaner BTC-relative analysis, taker-fee handling for market executions, and improved newly-listed-coin timestamp handling.
- Hyperliquid HIP-3 handling is more robust, while isolated HIP-3 live trading is now blocked explicitly until it is properly supported.

Behavior changes to be aware of:
- `optimize.backend` now defaults to `pymoo`
- `configs/template.json` is no longer the canonical starting point
- the canonical schema enables the local monitor by default
- `live.max_realized_loss_pct` now defaults to `1.0`, so the realized-loss gate is opt-in unless you set it explicitly

## Telegram Announcement Draft
Passivbot v7.9.0 is out.

Key changes:
- full live + backtest equity hard-stop framework with richer metrics and fake-live replay tooling
- `pymoo` optimizer backend added and now the default
- new `passivbot tool pareto` explorer for filtering and selecting Pareto candidates
- canonical staged config pipeline with schema defaults and new example config path
- unified `passivbot` CLI with better help and stricter environment / Rust freshness checks
- new monitor stack: publisher, relay, web dashboard, and TUI
- better live/backtest alignment for market-vs-limit execution and market-fill fee handling

Important after upgrading:
- reinstall Passivbot in your active env
- use `python3 -m pip install -e ".[full]"` if you run optimize/backtest
- rebuild Rust with `maturin develop --release` if needed

Also note:
- `optimize.backend` now defaults to `pymoo`
- `configs/template.json` is no longer the canonical starting point
- monitor is enabled by default in the schema
- `live.max_realized_loss_pct` now defaults to `1.0`
