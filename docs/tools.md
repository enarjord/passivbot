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
passivbot tool pareto optimize_results/... --scenario opt_2025_to_present
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

For suite optimization results, `--scenario LABEL` projects every saved Pareto member onto that
scenario's stored metric values, applies CLI limits in that scenario context, rebuilds a
nondominated sub-front using the original `optimize.scoring` goals, and then runs the selected
decision method. Suite artifacts store one mean value per scenario and metric, so explicit
non-`mean` limit statistics are unavailable in this mode. The rebuilt front covers only members of
the saved aggregate-suite Pareto front: candidates already discarded by the optimizer cannot be
recovered, so it is not the complete scenario Pareto front across all evaluated candidates.

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

## Trailing parameter inspector

`passivbot tool trailing-inspect` explains the effective `trailing_martingale` entry and close
thresholds for a hypothetical position. It is offline and read-only. It shows each wallet-exposure
and volatility contribution, the threshold boundary, the retracement distance from the running
extreme, the nominal confirmation price if the reversal starts exactly at the threshold, and the
order-reference price used after both conditions pass.

Without `--config`, the command uses the Rust-owned strategy defaults. With `--config`, it loads the
selected side's canonical `bot.<side>.strategy.trailing_martingale` parameters. Individual flags
override either source. Percent values use config ratios, so `0.01` means 1%.

```shell
passivbot tool trailing-inspect \
  --symbol COIN --side long \
  --position-size 150 --position-price 20 \
  --wallet-exposure 0.6 --effective-wallet-exposure-limit 0.9 \
  --volatility-ema-1m 0.007 --volatility-ema-1h 0.0033

passivbot tool trailing-inspect \
  --config configs/examples/default_trailing_martingale_long.json \
  --side long --position-price 20 \
  --wallet-exposure 0.6 --effective-wallet-exposure-limit 0.9 \
  --volatility-ema-1m 0.007 --volatility-ema-1h 0.0033 \
  --entry-threshold-base-pct 0.02 --json
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

## Crash finder

`passivbot tool crash-finder` builds larger discovery candles from local v2 1m OHLCV data and
scans them for severe crash windows. The discovery timeframe is parameterized with `--timeframe`
and defaults to `1h`; `4h` and `12h` are useful alternatives for slower market-wide moves. The
scanner groups valid source rows once, while retaining their order inside each candle so a low
before a later high is not misclassified as a crash.

Full scans verify cache chunk checksums and write raw event/cluster CSVs plus
`backtest.scenarios` suites. Use `--clusters-csv` only to regenerate suites cheaply when event
discovery does not need to be repeated; this fast path cannot find crashes in newly downloaded
candles.
When a sibling `scanned_ranges.csv` is present, suite generation drops coins whose cached data range
does not overlap the generated scenario date window. A targeted scenario is omitted if no coins
remain, rather than inheriting the suite's base coin universe.

```shell
passivbot tool crash-finder \
  --root caches/ohlcvs \
  --exchange binance \
  --exchange bybit \
  --source-timeframe 1m \
  --timeframe 1h \
  --threshold -0.10 \
  --pre-days 14 \
  --post-days 60 \
  --scenario-force-normal both \
  --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios

passivbot tool crash-finder \
  --clusters-csv crash_finder_results/crash_clusters.csv \
  --pre-days 14 \
  --post-days 60 \
  --scenario-force-normal both \
  --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios
```

Useful suite controls:

- `--scenario-kind market-wide` keeps broad market crashes only.
- `--scenario-kind coin-focused` keeps non-market-wide crashes, including isolated multi-coin
  clusters such as a DEXE/M event.
- `--scenario-kind single-coin` keeps strict one-coin crashes only.
- `--write-filtered-suites` writes all three filtered suites alongside the main suite.
- `--scenario-force-normal long|short|both` emits per-coin `forced_mode_* = "normal"` overrides
  only for idiosyncratic non-market-wide crash coins. Market-wide scenario coins are not forced.
  If a merged scenario would contain more than two forced coins, the tool splits it into repeated
  scenario windows with at most two forced coins per scenario.
- `--scenario-merge-overlaps` merges scenarios whose generated date windows overlap, preserving
  the earliest start, latest end, union of coins, and worst-severity label.

See `docs/ai/runbooks/crash_discovery.md` for the complete discovery, download, generation, and
validation workflow.

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
- `passivbot tool hsl-replay-benchmark` runs a bounded deterministic in-memory fixture through
  the current coin-HSL history replay initializer. It reports machine-readable stage timings,
  profiled timeline-rows/s and pair-rows/s, replay counters, fixture and final-state hashes,
  and side-effect counters. It never contacts an
  exchange or reads/writes live cache, monitor, latch, or state artifacts. Use
  `--minutes`, `--symbols`, and `--iterations` to change only the bounded synthetic workload.
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
  selected, for example `--brief --section fill_refresh` or
  `--section repository` for checkout metadata only.
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
  Existing structured cache events are summarized as `cache_health` in
  full/summary output and `cache` in brief output, including warm-cache reuse,
  cold-path, cache load, and cache flush counters without exposing raw cache
  paths or payloads.
  Existing structured shutdown events are also summarized as `shutdown_events`,
  so recent Ctrl+C/restart behavior can be inspected from the same smoke output.
  With `--supervisor-config`, the read-only process section compares configured
  `passivbot live` commands to the local process table and reports matched, missing,
  duplicate-command, and extra/orphan-like live processes. This does not prove tmux pane
  ownership; the fields are command-match diagnostics intended to make stale process leftovers
  obvious before any restart orchestration. Full, summary, and brief process output also
  includes bounded current state counts, uninterruptible-sleep count, and CPU, memory, and RSS
  totals/maxima/reporting counts from the same local process scan. These fields are observational;
  missing platform fields remain null and do not change the smoke verdict. Shutdown orchestration
  should follow an explicit
  escalation ladder as policy only: graceful Ctrl+C/request stop, bounded wait, a second
  graceful signal when warranted, SIGTERM, then SIGKILL. The smoke report never sends those
  signals.
- `passivbot tool live-process-report` runs the same bounded process-table and
  optional supervisor-config checks without entering the smoke report's monitor-event or
  text-log paths. It does not access credential stores, contact a network or exchange, control
  processes, or write files. Use `--samples N --interval-s SECONDS` for bounded persistence and
  recovery evidence; the existing maxima are enforced. The JSON `safety` object declares this
  capability boundary, and the command exits nonzero when the process/config verdict has hard
  failures. Use `--brief` for aggregate-only process/config/resource/state/sampling counters
  without command, account, path, PID, or per-process rows.
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
  `--performance-section SECTION` with `--performance-report` to keep only
  selected top-level performance sections plus common performance metadata in
  the embedded artifact and compact manifest summary. When
  `--restart-smoke-plan` is also used, the same performance section filters are
  passed into the embedded restart plan's planned failure bundle command. Use
  `--smoke-section SECTION` one or more times to keep only selected top-level
  sections from the embedded full `smoke_report.json` plus common smoke
  metadata, for example `--smoke-section fill_refresh_health`. Use
  `--restart-smoke-plan` with `--supervisor-config` to include a
  `restart_smoke_plan.json` dry-run artifact in the bundle; the embedded plan
  is generated by `live-restart-smoke-plan`, remains non-executing, and carries
  planned smoke/config-preflight commands for follow-up review. The returned
  report and manifest include the embedded restart plan's selected smoke and
  performance sections plus the planned smoke and follow-up bundle command
  summaries, process-signal safety, and execution-policy summaries as compact
  provenance. They also include the restart plan's compact warning and issue
  summaries, so malformed or partial restart plans are visible without
  extracting `restart_smoke_plan.json`, plus the compact timeout-escalation
  ladder summary. Its planned smoke command uses the restart planner's bounded
  scan defaults rather than the incident bundle's event/log scan settings.
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
  plan that produced it. Use `--performance-section SECTION` one or more times
  to add matching `live-incident-bundle --performance-section` values to the
  planned failure bundle command; use `--incident-bundle-output PATH` to choose
  the planned bundle path. The default planned bundle path is a
  timestamped `/tmp/passivbot_incident_bundle_restart_smoke_<utc>.tar.gz`
  path to avoid overwriting prior evidence. The pre-restart readiness phase also
  includes one deduplicated `live-config-preflight` command for each configured
  live config path found in the supervisor config, plus a skip count for any
  configured live command whose config path could not be derived. Use
  `--target-session-name SESSION` to append the exact local-only
  `live-restart-target-report` gate to that phase. The planner defaults to
  three samples five seconds apart; use `--target-samples N` (2-5) and
  `--target-interval-s SECONDS` (greater than 0, at most 30) to change the
  bounded stability window.
  The command is only emitted, never run. Omitting the exact session keeps the
  plan valid and explicitly marks the stable target gate unconfigured. Use
  `--summary` when you only need the bot count, phase names, preflight commands,
  target-preflight verdict requirement, smoke command, and incident-bundle
  command without every per-bot phase detail.
- `passivbot tool live-restart-target-report SUPERVISOR_CONFIG --session-name
  SESSION` performs the local-only exact-target preflight required before any
  future restart executor may signal a pane. It joins expected supervisor
  window names with canonical tmux pane IDs and read-only pane metadata plus the
  existing process/config verdict, proves ownership by requiring the matched bot
  process PID or PPID to equal the pane PID, and fails on missing, duplicate, or
  unconfigured panes and mismatched ownership in the exact confirmed session.
  Other sessions such as `misc` are ignored. The report is bounded and never
  signals, starts, or controls processes, contacts a network or exchange,
  accesses credential stores, or writes files. Use `--samples N` with
  `--interval-s SECONDS` to require the same pane ID, pane PID, matched bot PID,
  and ownership proof across a bounded pre-action window; any hard-red sample
  or identity change makes the report fail.
  The sampled identity also includes an opaque SHA-256 fingerprint of the
  complete parsed supervisor window/command/config contract, computed from
  private canonical commands before public report redaction or truncation. The
  report never emits the command content, and fails closed when the contract is
  unavailable, malformed, or changes during the pre-action sample window.
  Each resolved target also classifies whether the matched bot is a child of
  the pane PID and therefore has a candidate relaunch path through that exact
  pane after the bot exits. The report exposes the bounded relaunch method,
  supervisor-config command source, and mandatory post-stop pane recheck, but
  never emits the configured command or assumes the pane is ready before that
  recheck. Direct pane-process targets remain valid ownership matches but are
  explicitly relaunch-unready. The executor requires
  `relaunch_ready_targets == resolved_targets` in addition to the hard-green
  stable sampling verdict.
- `passivbot tool live-restart-executor SUPERVISOR_CONFIG --session-name
  SESSION --expected-repository-head COMMIT
  --expected-rust-source-fingerprint SHA256
  --expected-supervisor-fingerprint SHA256 --execute` gracefully
  restarts only the exact local tmux targets proven by the same bounded target
  contract. Before target sampling it requires the exact caller-confirmed
  40-character Git commit, zero tracked changes while preserving untracked
  artifacts, the expected fingerprint recorded by the repository-owned Rust
  build verification, and a loaded Rust extension whose source stamp exactly
  matches that fingerprint and the checked-out Rust build inputs. This explicit
  fingerprint matters because the build inputs include the host-local ignored
  `Cargo.lock`, so identical Git heads need not produce the same fingerprint.
  Each runtime check rehashes those inputs after extension verification so an
  ignored input cannot change inside the check unnoticed. It repeats that
  runtime contract immediately before the first signal and before relaunch. It
  also requires the caller-confirmed
  full-command fingerprint, takes an immediate action snapshot, and sends one
  Ctrl-C round to exact pane IDs. After a bounded exact-PID exit wait, it scans
  for unexpected or duplicate live processes, verifies the complete session
  pane set, parent PIDs, window identities, and shell-ready exited panes, then
  immediately re-reads the runtime contract, private supervisor snapshot, and
  process set before typing commands only into eligible panes. Final startup
  and multi-sample target verification must retain the same fingerprint.
  The executor does not SSH, pull code, contact exchanges directly, write files
  directly, use broad process-pattern signals, or apply SIGTERM/SIGKILL. A
  timeout or changed runtime/process/pane contract is reported as manual
  recovery; targets that did exit may be relaunched only when every post-stop
  check is still exact. The relaunched live bots resume their configured
  exchange access and normal runtime file writes. Run the read-only target
  report first, retain the Rust fingerprint from build verification, and pass
  both exact fingerprints; command content is never emitted.
- `passivbot tool live-restart-smoke-evidence TARGET_REPORT_JSON
  SMOKE_REPORT_JSON --expected-repository-head COMMIT
  --expected-supervisor-fingerprint SHA256 --expected-targets N` evaluates two
  already-generated full JSON reports against one bounded post-restart evidence
  contract. It requires stable multi-sample exact targets, the caller-confirmed
  private supervisor fingerprint, a clean exact repository head, bounded
  monitor and text-log windows with no hard evidence, complete shutdown event
  counts, and distinct startup timing coverage for the expected targets. The
  event and text-log bounds are compared as exact validated epoch-millisecond
  values and projected without count clamping. Both retained hard log matches
  and contextless hard-looking matches suppressed by the opt-in drop policy
  must be well-formed zeroes.
  result is bounded and sanitized: it does not copy paths, commands, bot names,
  symbols, log samples, or arbitrary input payloads. The command only reads the
  two named local files; it does not run report producers, SSH, signal or start
  processes, contact a network or exchange, load credentials, or write files.
  Use `--compact` for single-line JSON. Exit status is zero only when every hard
  gate passes; non-hard attention evidence remains visible without making the
  verdict red.
- `passivbot tool live-restart-smoke-collect SUPERVISOR_CONFIG MONITOR_ROOT
  --session-name SESSION --expected-repository-head COMMIT
  --expected-supervisor-fingerprint SHA256 --expected-targets N --since-ms MS
  --until-ms MS` directly composes the existing exact target sampler, full
  bounded-window smoke report, and sanitized evidence evaluator in memory. The
  caller-confirmed head, private fingerprint, target count, and exact window are
  never derived from the reports being checked. Managed rotated event filenames
  provide interval boundaries so the
  collector reads only overlapping segments plus explicit coverage evidence;
  malformed names, missing predecessor coverage, more than eight selected
  segments per bot, more than 128 selected segments total, or more than 128 MiB
  of selected event data fail closed before event content scanning. Its output
  includes only aggregate selection completeness/count/scan-byte evidence and
  code-owned issue counts, never segment paths. Process
  inspection is owned by the target report rather than
  duplicated in smoke collection, and contextless log rows use the existing
  drop policy whose dropped hard-looking count must remain zero. The command
  emits no raw target or smoke report and writes no intermediate file. It may
  run bounded local `tmux`, `ps`, and `git` inventory reads; it does not SSH,
  pull/build, contact a network or exchange, access credentials, signal/start
  processes, or perform force escalation. Use `--compact` for single-line JSON.
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
  delays are visible without scanning every timing group. Base timing groups and their
  `operation_durations`/`slowest_blockers` projections include one bounded `latest_ids` mapping
  of report-safe canonical event-envelope IDs when available (excluding per-action IDs),
  allowing direct correlation with `live-event-query` without exposing free-form payloads.
  The mapping uses the same legacy snapshot-ID normalization as `live-event-query` and
  selects the latest sample by stable event position (`ts`, `seq`, path, and line).
  The `operation_durations` section
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
  Event-pipeline health also attributes due monitor-retention work to fixed inventory,
  age-unlink, and byte-cap-unlink totals/maxima with visited, candidate, and successful
  deletion counts. These fields are diagnostic only and do not change retention policy or
  report verdicts.
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
