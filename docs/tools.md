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
directory by lexicographic run-directory name, considering only runs whose `pareto/`
subdirectory contains at least one `*.json` candidate. If you point it at an optimize run
directory instead of the nested `pareto/` directory, it resolves that automatically.

```shell
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto
passivbot tool pareto optimize_results/.../pareto -m reference \
  --target adg_strategy_eq=0.001 \
  --target drawdown_worst_strategy_eq=0.25
passivbot tool pareto optimize_results/.../pareto \
  -l 'drawdown_worst_strategy_eq<=0.35' \
  -l 'adg_strategy_eq>0.0'
passivbot tool pareto -o sharpe_ratio_strategy_eq,adg_strategy_eq,strategy_eq_recovery_days_max \
  -m ideal
passivbot tool pareto optimize_results/... -m utility \
  --weight adg_strategy_eq=4 \
  --weight drawdown_worst_strategy_eq=2 \
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

`passivbot tool pareto-explorer` is a CLI alias for the same tool.

The output also shows the retained front's ideal point: the best observed value for each active
objective after any `--limit` filters are applied.

`-o` / `--objectives` is not limited to the original `optimize.scoring` list. You can also name
other stored metrics such as `sharpe_ratio_strategy_eq` as long as the Pareto JSON
members contain that metric and Passivbot knows whether higher or lower is better.

## Pareto transformations / static plots

`src/tools/pareto_transform.py` converts `all_results.bin` or individual Pareto JSON entries into CSV/JSON summaries for external analysis. The legacy `src/pareto_store.py` still produces quick matplotlib scatter plots if you prefer static images.

```shell
passivbot tool pareto-transform optimize_results/.../all_results.bin --out summary.csv
python3 src/pareto_store.py optimize_results/.../pareto/
```

`passivbot tool pareto-analyze` inspects a Pareto front's config and metric distributions,
including the strongest metric-to-metric correlations among the displayed metrics.
`passivbot tool pareto-analysis` is an alias for the same command. `passivbot tool
pareto-compress` selects a compact representative subset from a front, and
`passivbot tool merge-paretos` merges Pareto fronts into starting configs.

```shell
passivbot tool pareto-analyze optimize_results/.../pareto
passivbot tool pareto-compress optimize_results/.../pareto 8 --output-dir selected_pareto_8
passivbot tool merge-paretos optimize_results/run_a/pareto optimize_results/run_b/pareto
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

- `passivbot download` – Pre-warm the v2 OHLCV store using the same config/date/exchange selection as backtesting.
- `passivbot tool inspect-ohlcvs` – Inspect v2 OHLCV cache metadata and gaps.
- `passivbot tool pad-historical-daily` – Ensures daily OHLCV shards are present for the downloader when new coins are added.
- `passivbot tool verify-hlcvs-data` – Validates prepared HLCV datasets and coverage metadata before long optimizations/backtests.
- `passivbot tool streamline-json` – Normalizes/compacts JSON configs (`passivbot tool streamline-json configs/examples/default_trailing_martingale_long.json`).
- `passivbot tool cache-integrity-doctor` – Read-only local cache smoke report for missing roots, file counts/sizes, cache-family summaries, report-only warm-cache readiness evidence, v2 candle coverage/gap evidence with interior/boundary gap summaries, fill/HSL metadata evidence, and empty or corrupt JSON/NDJSON/NPY artifacts.
- `passivbot tool candle-doctor` – Audits legacy `caches/ohlcv/...` shards for corruption, stale index entries, and legacy-format issues; add `--fix` to apply automatic repairs before importing into the v2 store.
- `passivbot tool migrate-historical-data` – Converts legacy `historical_data/ohlcvs_<exchange>/...` shards into the current `caches/ohlcv/...` layout.

## Fill Events Tooling

`passivbot tool fill-events-dash` launches a Dash UI for inspecting cached fill-event history,
PnL, symbol-level details, cache health, and CSV export.

```shell
passivbot tool fill-events-dash --users bybit_01
```

`passivbot tool fill-events-doctor` audits cached fill-event anomalies and optionally repairs them.

```shell
passivbot tool fill-events-doctor --exchange bybit --user bybit_01
passivbot tool fill-events-doctor --exchange bybit --user bybit_01 --repair
```

## Monitor Tooling

Monitor commands are documented in detail in [monitor.md](monitor.md). The CLI surface is:

- `passivbot tool monitor-relay`
- `passivbot tool monitor-web`
- `passivbot tool monitor-tui`
- `passivbot tool monitor-dev`
- `passivbot tool live-config-preflight` emits a read-only offline JSON report for one live
  config, summarizing identity hints, HSL settings, universe counts, forager slots/staleness,
  cache-related live settings, cache root hints, and config-only cache readiness attention
  signals without contacting exchanges or scanning local artifacts. Add
  `--compare BASELINE_CONFIG.json` to include a bounded, read-only diff of risk-relevant
  HSL, universe, forager, identity, cache-setting, and derived cache-readiness changes
  between two local configs.
- `passivbot tool hsl-startup-preview` emits a read-only offline JSON preview for one live
  config plus optional local monitor events. It reports configured HSL settings and latest
  local HSL status/cooldown observations when present, while explicitly marking current
  drawdown and startup panic-order prediction unavailable unless a future slice adds safe
  local replay inputs.
- `passivbot tool live-event-query` validates and queries local structured monitor event
  segments. It is read-only and does not contact exchanges. Use `--event-type`,
  `--level`, `--cycle-id`, ID filters, `--symbol`, `--pside`, `--tag`,
  `--debug-profile`, `--data-eq`, and time-window filters to reconstruct
  incident slices. Use `--exchange EXCHANGE` and
  `--user USER` to focus one monitor account and prune unrelated monitor paths before
  scanning; `--bot-id` remains available for event-envelope bot IDs. Directory scans
  read `current.ndjson` by default; add `--include-rotated` for complete history
  queries after monitor event rotation. For repeated recent-window queries over
  large current monitor segments, `--event-tail-lines N` bounds parsing to the last N rows
  from each event file; the default `0` keeps full event validation.
- `passivbot tool live-smoke-report` summarizes local live monitor events and text logs for
  operator smoke-test evidence. Use `--summary` for bounded event groups and log matches, or
  `--brief` for top-level counters suitable for repeated VPS smoke loops. Use
  `--section SECTION` one or more times to emit selected top-level smoke-report
  sections plus common smoke metadata after the full/summary/brief projection is
  selected, for example `--brief --section fill_refresh`.
  Full, summary, and brief output include text-log scan bounds and bounded
  text-log window counters so time-windowed smoke evidence shows the configured
  file/tail/match limits plus how many log lines were considered, skipped
  before/after the window, or dropped by the unparseable-line policy.
  For repeated recent-window smoke checks over large current monitor segments,
  `--event-tail-lines N` bounds monitor event parsing to the last N rows from
  each event file; the default `0` keeps full monitor-event validation. Brief
  output includes the structured event-window `enabled` flag plus the bounded
  event-window counters.
  Startup timing summaries include report-only budget projections from prior local p95
  phase baselines when enough monitor evidence exists, and the summary/brief projections
  surface bounded startup timing counters for repeated smoke loops.
  Existing structured fill-refresh summaries are summarized as
  `fill_refresh_health` in full/summary output and `fill_refresh` in brief output,
  including status/error counts and whether a bot had failed refreshes that later
  recovered inside the report window.
  Existing structured EMA readiness degradation events are summarized as
  `ema_readiness_health` in full/summary output and `ema_readiness` in brief output,
  including latest candidate/unavailable counts and bounded reason/error group evidence.
  Existing staged execution readiness degradations are summarized as
  `staged_readiness_health` in full/summary output and `staged_readiness` in brief output,
  including latest missing/invalid surface counts and bounded completed-candle mismatch evidence.
  Existing structured execution/order-wave events are summarized as
  `execution_health` in full/summary output and `execution` in brief output,
  including create/cancel/confirmation outcome counters and bounded write/confirmation groups.
  Existing structured shutdown events are also summarized as `shutdown_events`,
  so recent Ctrl+C/restart behavior can be inspected from the same smoke output.
  With `--supervisor-config`, the read-only process section compares configured
  `passivbot live` commands to the local process table and reports matched, missing,
  duplicate-command, and extra/orphan-like live processes. This does not prove tmux pane
  ownership; the fields are command-match diagnostics intended to make stale process leftovers
  obvious before any restart orchestration. Shutdown orchestration should follow an explicit
  escalation ladder as policy only: graceful Ctrl+C/request stop, bounded wait, a second
  graceful signal when warranted, SIGTERM, then SIGKILL. The smoke report never sends those
  signals.
- `passivbot tool live-incident-bundle` writes a local `.tar.gz` evidence
  bundle with monitor event reports, problem-event reports, smoke evidence,
  redacted log excerpts, monitor snapshots, runtime metadata, and optional
  bounded event segments. The returned report and manifest include a compact
  smoke execution summary with only aggregate create/cancel/confirmation
  counters. When invoked from outside the repository with an absolute monitor
  path, the manifest infers git metadata from the monitor tree when possible.
  Manifest git status is limited to tracked changes so local untracked configs
  and monitor artifacts do not dominate the bundle metadata. It is read-only
  and does not contact exchanges. Use
  `--performance-report` to embed a bounded `performance_report.json` artifact
  plus compact manifest summary using the bundle's time, bot/exchange/user,
  debug-profile, rotated-file, tail-line, and per-bot file bounds. Use
  `--smoke-section SECTION` one or more times to keep only selected top-level
  sections from the embedded full `smoke_report.json` plus common smoke
  metadata, for example `--smoke-section fill_refresh_health`. Use
  `--restart-smoke-plan` with `--supervisor-config` to include a
  `restart_smoke_plan.json` dry-run artifact in the bundle; the embedded plan
  is generated by `live-restart-smoke-plan`, remains non-executing, and carries
  planned smoke/config-preflight commands for follow-up review. Its planned
  smoke command uses the restart planner's bounded scan defaults rather than
  the incident bundle's event/log scan settings.
- `passivbot tool live-restart-smoke-plan` emits a read-only dry-run restart
  plan from a tmuxp-style supervisor config. The planned smoke command defaults
  to a brief compact smoke command with `--event-tail-lines 2000` and
  `--max-event-files-per-bot 2`, plus `--max-log-files 8`,
  `--log-tail-lines 1200`, and `--max-log-matches 20`, for repeated operator
  loops; use `--event-tail-lines 0`, `--max-event-files-per-bot 0`,
  `--max-log-files 0`, `--log-tail-lines 0`, or `--max-log-matches 0` in the
  planner to omit those explicit bounds from the generated smoke command. Use
  `--summary-smoke-report` for bounded groups or `--full-smoke-report` for the
  full smoke report. Use `--smoke-section SECTION` one or more times to add
  focused `live-smoke-report --section` values to the planned smoke command and
  matching `live-incident-bundle --smoke-section` values to the planned failure
  evidence bundle. Use `--log-window-unparsed-policy drop` to make both planned
  commands drop unparseable text-log lines without in-window timestamp context;
  the default planned policy is `keep`.
  The planner does not execute the restart, send signals,
  invoke tmux, run SSH, pull git, start bots, contact exchanges, or load
  credentials. The plan also includes a bounded `live-incident-bundle` command
  for failure evidence, reusing the same event/log scan limits and disabling
  event-segment copying by default. The planned failure bundle includes
  `--performance-report` and `--restart-smoke-plan`, so the archive carries
  bounded performance timing/readiness evidence plus the non-executing restart
  plan that produced it; use `--incident-bundle-output PATH` to choose the planned
  bundle path. The default planned bundle path is a
  timestamped `/tmp/passivbot_incident_bundle_restart_smoke_<utc>.tar.gz`
  path to avoid overwriting prior evidence. The pre-restart readiness phase also
  includes one deduplicated `live-config-preflight` command for each configured
  live config path found in the supervisor config, plus a skip count for any
  configured live command whose config path could not be derived. Use planner
  `--summary` when you only need the bot count, phase names, preflight commands,
  smoke command, and incident-bundle command without every per-bot phase detail.
- `passivbot tool live-performance-report` summarizes local live monitor event timings for
  operator performance analysis. It is read-only and does not contact exchanges. Use
  `--recent-minutes` for a time window, `--summary` for a bounded operator projection, and
  `--bot EXCHANGE/USER`, `--exchange EXCHANGE`, or `--user USER` to focus one account or
  exchange. Use `--section SECTION` one or more times to emit selected top-level report
  sections plus common metadata, for example `--summary --section fill_refresh`; use
  `--section all` for the default full output. The report includes
  decision-boundary lag groups showing how far after the
  whole-minute boundary cycles reached start, Rust planning, action planning, writes,
  confirmations, and completion. It also includes an input-staleness section derived from
  existing packet, snapshot, EMA, and Rust-call events, covering account packet age at
  snapshot build plus snapshot/EMA age at the Rust call boundary when those events are present.
  For `snapshot_to_rust`, the report uses exact event-envelope `cycle_id` matches when
  available; for legacy/current `snapshot.built` events that only carry a planning snapshot
  epoch in `data.cycle_id`, it uses the latest preceding snapshot in the same bot/restart scope
  and exposes exact-vs-latest match counters.
  Newer `snapshot.built` events also expose bounded surface-age and market-snapshot age
  summaries, allowing the report to break down stale planning inputs without exposing market
  prices.
  The startup-readiness section summarizes latest per-bot startup phase timings and HSL
  replay state from existing lifecycle/replay events. The `slowest_blockers` section ranks
  non-diagnostic timing and staleness groups by observed duration so the largest trading-impact
  delays are visible without scanning every timing group. The `operation_durations` section
  collates startup, cycle, state-refresh, remote-call, HSL replay, cache, decision-boundary,
  input-staleness, fill-refresh, execution, and shutdown timing groups into one bounded table with operation
  category, trading-impact, blocking-scope, and timing-kind counters. The `resource_pressure` section
  summarizes whitelisted process and event-pipeline health fields from existing
  `health.summary` events, including RSS, memory percent, file descriptors, load average,
  event queue depth, dropped-event counters, and sink-error counters with count, latest,
  min, mean, median, p95, and max values without exposing raw account or financial
  payloads. The `shutdown_latency` section summarizes existing
  lifecycle shutdown events, including per-stage cumulative elapsed time and final shutdown
  duration, without copying shutdown error text. The `execution_timing` section derives
  aggregate exchange-action latency groups from existing order-wave, order create/cancel,
  and confirmation events with missing/unpaired counters, without exposing raw order payloads.
  The `hsl_replay_profile` section derives bounded HSL replay work/progress summaries from
  existing `hsl.replay.*` events, including pair counts, timeline rows, rows/s, estimated
  dense pair-row work, observed progress percentage, and startup-blocking elapsed time where
  available. The `cache_warmup` section derives bounded warm-cache reuse, cold-path,
  candle cache load, and candle cache flush summaries from existing cache events without
  exposing raw cache paths or payloads. The `fill_refresh` section derives bounded
  fill-cache refresh status, coverage, retry, count, and elapsed summaries from existing
  `fills.refresh_summary` events without exposing raw error text, debug payloads, fill ids,
  order ids, account values, or coverage payloads. The `forager_ema_readiness` section derives bounded
  forager selection, forager feature-unavailable, EMA unavailable, and EMA fallback summaries
  from existing events without exposing raw EMA error text, top-score payloads, account values,
  or cache paths.

## Exchange Helpers

`passivbot tool fetch-balance` is a lightweight credential smoke test that loads one user from
`api-keys.json`, instantiates the matching ccxt exchange, and prints the raw balance payload.

```shell
passivbot tool fetch-balance --user bybit_01
```

Ticker probes inspect CCXT ticker support and latency without placing orders:

- `passivbot tool ticker-probe` checks ticker capability behavior for one exchange/user context.
- `passivbot tool ticker-endpoint-probe` compares CCXT ticker endpoint latency across configured
  users. When authenticated probes are enabled, its JSON includes an `account_critical_health`
  summary for the read-only balance, positions, and open-orders endpoints required before live
  exchange actions. Use `--account-only` for the lower-impact balance/positions/open-orders probe,
  which still loads markets to resolve an open-orders fallback symbol, or `--skip-my-trades` to
  omit the fill-history endpoint. When the first-symbol `fetch_my_trades` probe is enabled,
  `fill_history_health` summarizes that sample's success, latency, trade count, newest timestamp,
  page count, terminal pagination reason, and shape without emitting raw trade/order ids. The default
  `--fill-history-pages=1` keeps the low-impact single-call behavior; raising it performs a bounded,
  explicit pagination sample for coverage diagnostics. `rate_limit_health` estimates request pressure
  from the calls the probe already made,
  including public/private/concurrent call counts and CCXT `rateLimit` metadata; it is a planning
  estimate, not proof of exchange-side rate-limit consumption. `endpoint_latency_health` groups
  already-recorded endpoint attempts by endpoint, including open-orders fallback attempts and
  fill-history pages, with success/failure counts and latency summaries. `exchange_surface_health`
  adds exchange/user-level notes from those same outcomes, such as open-orders symbol fallback,
  unsupported time sync, fill-history terminal pagination reason, and OHLCV tail shape, without
  making additional exchange calls. The probe also includes
  `time_sync_health` from read-only `fetch_time` clock-skew checks when the exchange supports it;
  use `--skip-time-sync` to omit that call. When OHLCV probing is enabled,
  `candle_freshness_health` summarizes the existing 1m candle tail results without making
  additional exchange calls. Keep authenticated runs low-rate when a live bot is using the same
  account.

## Hyperliquid live probes

These probes were added to investigate live Hyperliquid balance/state quirks, especially HIP-3
stock perps. They are intended to be reused for future live diagnostics instead of creating new
one-off scripts.

- `passivbot tool hyperliquid-balance-probe` is read-only. It fetches one wallet balance and prints
  a normalized summary.
- `passivbot tool hyperliquid-abstraction-probe` is read-only. It inspects Hyperliquid account
  abstraction mode and related metadata for one configured user.
- `passivbot tool hyperliquid-order-margin-probe` is mutating. It places one tiny post-only order,
  snapshots balance changes, then cancels it.
- `passivbot tool hyperliquid-position-probe` is mutating. It can open/flatten a tiny position and
  optionally place resting entry or reduce-only close orders to inspect live state transitions.

The mutating probes require `--yes` and are intended for test wallets or deliberately tiny live
positions only.

```shell
passivbot tool hyperliquid-balance-probe --user hyperliquid_01
passivbot tool hyperliquid-order-margin-probe --user hyperliquid_01 --symbol BTC/USDC:USDC --yes
passivbot tool hyperliquid-position-probe --user hyperliquid_01 --symbol XYZ-SP500/USDC:USDC --yes
passivbot tool hyperliquid-position-probe --user hyperliquid_01 --symbol XYZ-SP500/USDC:USDC --flatten-only --yes
```

Recommended usage:

- Start with `hyperliquid-balance-probe` to confirm the wallet and baseline balance fields.
- Use `hyperliquid-order-margin-probe` to inspect how a single resting order changes
  `accountValue`, `withdrawable`, and related fields.
- Use `hyperliquid-position-probe` only when you need to inspect live position-margin behavior,
  startup-state handling, or combined position-plus-order effects.

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
