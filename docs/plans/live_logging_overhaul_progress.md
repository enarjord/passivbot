# Live Logging Overhaul Progress

This file is the running progress ledger for `docs/plans/live_logging_overhaul_plan.md`.
Keep the plan as the architecture and decision record; update this file as PRs
merge, live smoke evidence changes, or new gaps are discovered.

## Update Policy

- Update this file when a logging-overhaul slice is merged to `v8`.
- Update it when a merged slice is deployed to VPS5 and smoke-tested.
- Keep entries factual and compact: PR/commit, scope, validation, and remaining
  gap.
- Do not use this file for design churn; unresolved design details belong in the
  plan or a focused handoff doc.

## Current Status

Last updated: 2026-07-09.

Current `origin/v8` head:

- `aff9e864` after PR #1162, `Report exchange config refresh performance`.

Current logging-overhaul head:

- `aff9e864` after PR #1162, `Report exchange config refresh performance`
  (latest merged logging-overhaul slice).

Current work:

- Branch `codex/v8-exchange-config-refresh-recovery` adds latest-per-bot status,
  latest-failed-bot, and recovered-bot aggregates to the existing smoke and
  performance `exchange.config_refresh` projections. Historical failures remain
  visible, but a later successful refresh is no longer presented as unresolved.
  The slice is read-only and does not change smoke verdicts, exchange calls,
  refresh behavior, order/risk logic, restart orchestration, or trading behavior.

Current review gate:

- Composer has been stopped/retired from this loop. The normal review gate is
  now Claude Opus 4.8 + Hermes + Grok 4.5 + CI. For low-risk docs/tooling-only
  slices, a degraded gate may still be used after repeated reviewer absence,
  but that exception must be called out in the progress evidence.

Retuned goal boundary:

- The active loop remains the live logging/observability overhaul. Backlog work
  belongs in this loop when it directly improves structured diagnosis, operator
  smoke evidence, incident reconstruction, or the ability to finish the logging
  overhaul.
- This loop should keep adding newly discovered operational gaps, non-urgent
  bugs, and room-for-improvement notes to
  `docs/plans/live_ops_improvement_backlog.md`, but implementation from that
  backlog should stay selective: prioritize items that make the logging
  overhaul easier to validate, deploy, and use.
- Trading-behavior bugs discovered by the new observability, including HSL
  panic/cooldown contract issues and HSL startup replay latency, should be
  tracked in the backlog and handled as separate focused trading-path PRs unless
  they block observability validation.
- Prefer single-agent implementation for tightly scoped observability slices.
  Use sub-agents only for isolated offline investigations or independent PRs
  with explicit no-SSH/no-merge boundaries.

VPS5 deployment status:

- Repository pulled through PR #1162 at `aff9e864`.
- PR #1162 added bounded `exchange_config_refresh` health and elapsed-timing
  groups to `live-performance-report`, excluding raw exchange error text. It
  merged after Hermes, Claude Opus 4.8, and Grok 4.5 approved the current head
  and CI was green. VPS5 was pulled with `git pull --autostash --ff-only origin
  v8`, preserving the pre-existing tracked Rust change and local config/tmp
  artifacts. No bot restart was performed because the slice was read-only
  report projection plus docs. A fresh one-minute smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, `missing_expected_count=0`, and zero
  failed remote/account-critical calls. A bounded 24-hour focused report scanned
  24 event files and returned `ok=true`, `error_count=0`, with 14 real refresh
  events across all five bots: 13 succeeded and one Kucoin `RequestTimeout`.
  Kucoin subsequently succeeded, Binance had three successes, and no raw error
  text appeared. GateIO had the slowest observed refresh at `max_ms=78920`.
- Repository pulled through PR #1161 at `aa9e6623`.
- PR #1161 added top-level event-pipeline queue, drop, sink-error, degraded,
  and unhealthy-bot aggregates to `live-performance-report`
  `resource_pressure`. It merged after Hermes, Claude Opus 4.8, and Grok 4.5
  approved the current head and CI was green. VPS5 was pulled with
  `git pull --autostash --ff-only origin v8`, preserving the pre-existing
  tracked Rust change and local config/tmp artifacts. No bot restart was
  performed because the slice was read-only report projection plus docs. The
  first bounded two-minute smoke saw one recovered Kucoin balance
  `RequestTimeout`: the same group already had a later successful balance
  fetch. A fresh one-minute smoke then reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`, and
  `logs.hard_matches=0`. A focused 30-minute `resource_pressure` report
  returned `ok=true`, `error_count=0`, `resource_bots=4`, queue depth `0`,
  zero drop/sink/degraded counters, and zero unhealthy bots.
- Repository pulled through PR #1157 at `441d9fe5`.
- PR #1157 added read-only aggregate resource-pressure sample-age fields to
  `live-performance-report`, exposing `latest_event_age_ms_max` and
  `latest_event_age_reporting_bots` from existing per-bot `health.summary`
  timestamps. It merged after Hermes approved the current head, Claude Opus
  4.8 green-lighted the rebased head, Grok 4.5 re-approved the current head,
  Codex reviewer posted green, and CI was green. VPS5 was pulled with
  `git pull --autostash --ff-only origin v8`, preserving the pre-existing
  tracked Rust-format/local config state. No bot restart was performed because
  the slice was read-only report projection plus docs. A bounded 2-minute
  summary smoke reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  repository head `441d9fe5`. A focused 30-minute `resource_pressure` report
  on VPS5 monitor data reported `ok=true`, `error_count=0`,
  `resource_total=9`, `resource_bots=5`,
  `latest_event_age_ms_max=872783`, and
  `latest_event_age_reporting_bots=5`.
- Repository pulled through PR #1156 at `4ac5b309`.
- PR #1156 added the read-only smoke-report counterpart for resource-pressure
  sample age, exposing per-group `latest_event_age_ms`, aggregate
  `latest_event_age_ms_max`, and reporting-bot count from existing
  `health.summary` event timestamps. It merged after Hermes approved, Claude
  Opus 4.8 green-lighted the current head, Grok 4.5 re-approved the rebased
  head, Codex reviewer posted green, and CI was green. VPS5 was pulled with
  `git pull --autostash --ff-only origin v8`, preserving the pre-existing
  tracked Rust-format/local config state. No bot restart was performed because
  the slice was read-only report projection plus docs. A bounded 2-minute
  summary smoke reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  repository head `4ac5b309`. The 2-minute brief resource-pressure window had
  no health-summary samples, so `latest_event_age_ms_max=null` and
  `latest_event_age_reporting_bots=0`; a focused 30-minute `resources` section
  confirmed four reporting bots with `latest_event_age_ms_max=789479` and a
  sample group age of `583466`.
- Repository pulled through PR #1154 at `141e88db`.
- PR #1154 added read-only `latest_event_age_ms` freshness metadata to
  `live-performance-report` `resource_pressure` groups. It merged after Hermes
  approved, Claude Opus 4.8 green-lighted the current head, Cursor/Grok
  approved, Codex reviewer posted green, and CI was green. VPS5 was pulled with
  `git pull --autostash --ff-only origin v8` to preserve the pre-existing
  tracked Rust-format/local config state. No bot restart was performed because
  the slice was read-only report projection plus docs. A bounded 2-minute smoke
  with supervisor process check reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `logs.hard_matches=0`, and repository head `141e88db`. Non-hard attention
  remained active HSL replay and EMA readiness evidence. The short smoke window
  contained no `health.summary` resource-pressure samples; a focused 30-minute
  `live-performance-report --section resource_pressure` check reported
  `ok=true`, `resource_pressure.total=2`, `resource_pressure.bots=1`, and
  `latest_event_age_ms=624447` on the Hyperliquid resource-pressure group.
- Repository pulled through PR #1153 at `84191c02`.
- PR #1153 kept no-data `live-smoke-report --brief` resource-pressure minimum
  fields as null/absent instead of coercing them to zero. It merged after
  Hermes approved, Claude Opus 4.8 green-lighted the current head, Cursor/Grok
  approved, and CI was green. VPS5 was pulled with `git pull --autostash
  --ff-only origin v8` to preserve the pre-existing tracked Rust-format/local
  config state. No bot restart was performed because the slice was read-only
  report projection plus docs. A bounded 2-minute smoke with supervisor process
  check reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  repository head `84191c02`. Non-hard attention remained active HSL replay and
  EMA readiness evidence; the short smoke window contained no
  `health.summary` resource-pressure samples, so `resource_pressure.total=0`.
- Repository pulled through PR #1152 at `80ea2ea2`.
- PR #1152 added optional psutil-backed system memory and swap pressure fields
  to existing `health.summary` events and projected them through
  smoke/performance `resource_pressure` output. It merged after Claude Opus 4.8
  green-lighted the current head, Hermes approved, Cursor/Grok approved, and CI
  was green. VPS5 was pulled with `git pull --autostash --ff-only origin v8`
  to preserve the pre-existing tracked Rust-format/local config state, then
  bots were restarted from `/root/bots_vps5.yaml` because the slice changed the
  live health-summary producer. All five old bot processes required exact
  live-process SIGTERM after two Ctrl-C grace windows; `tmuxp load -d
  /root/bots_vps5.yaml` started all five configured bots. Immediate 2-minute
  smoke reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `fill_refresh.failed=0`, `logs.hard_matches=0`, and repository head
  `80ea2ea2`. A focused 5-minute `resource_pressure` smoke also reported
  `ok=true`, `hard_failures=0`, one `health.summary` event, and live projected
  host pressure fields: `latest_system_memory_percent_max=96.4`,
  `latest_system_memory_available_bytes_min=35811328`, and
  `latest_swap_percent_max=49`.
- Repository pulled through PR #1151 at `1a624989`.
- PR #1151 added bounded health-summary scheduling lag telemetry to existing
  `health.summary` events and projected it through smoke/performance
  `resource_pressure` output. It merged after Claude Opus 4.8 and Grok 4.5
  green-lighted the current head, Hermes approved, and CI was green. VPS5 was
  pulled with `git pull --autostash --ff-only origin v8` to preserve the
  pre-existing tracked Rust-format/local config state, then bots were restarted
  from `/root/bots_vps5.yaml` because the slice changed the live
  health-summary producer. Binance exited in the first Ctrl-C window; Kucoin,
  GateIO, OKX, and Hyperliquid required exact live-process SIGTERM before
  reload. `tmuxp load -d /root/bots_vps5.yaml` started all five configured
  bots. Immediate 2-minute smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`,
  `logs.hard_matches=0`, and repository head `1a624989`. A focused 5-minute
  `resource_pressure` smoke also reported `ok=true`, `hard_failures=0`, and
  two `health.summary` resource-pressure events. The new
  `health_summary_lag_ms` field was not present in that short post-restart
  window because first post-start health summaries intentionally omit it.
- Repository pulled through PR #1042 at `06f04070`.
- PR #1042 added an opt-in `cache` live-event debug profile for existing cache
  load, flush, and warmup-decision events. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `06f04070` without restarting running bots because the slice was
  observability-only. Bounded smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard log matches,
  no event-pipeline drops or sink errors, and only known non-hard ZEC HSL
  cooldown attention.
- Repository pulled through PR #1041 at `c5ba5f5d`.
- PR #1041 made `live-incident-bundle --restart-smoke-plan` expose the embedded
  restart-smoke plan's compact timeout-escalation ladder summary in the returned
  incident-bundle report and manifest summary. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `c5ba5f5d` without restarting running bots because the slice was read-only
  tooling. Bounded smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard log matches,
  no event-pipeline drops or sink errors, and only known non-hard ZEC HSL
  cooldown attention. A focused VPS incident-bundle check proved the manifest's
  embedded restart-smoke summary includes a four-row non-executing
  `timeout_escalation_ladder`, `ok=true`, and still omits config-preflight raw
  command lists.
- Repository pulled through PR #1040 at `030d77aa`.
- PR #1040 made `live-incident-bundle --restart-smoke-plan` expose the embedded
  restart-smoke plan's compact warning and issue summaries in the returned
  incident-bundle report and manifest summary. It merged after Claude and Hermes
  approved with no findings and CI was green; Cursor was absent, so the
  low-risk read-only tooling degraded gate was used. VPS5 checkout was updated
  to `030d77aa` without restarting running bots because the slice was read-only
  tooling. Bounded smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard log matches,
  no event-pipeline drops or sink errors, and only known non-hard ZEC HSL
  cooldown attention. A focused VPS incident-bundle check proved the manifest's
  embedded restart-smoke summary includes `warnings.count=7`, `issues.count=0`,
  warning items, and still omits config-preflight raw command lists.
- Repository pulled through PR #1039 at `f761063f`.
- PR #1039 made `live-incident-bundle --restart-smoke-plan` expose the embedded
  restart-smoke plan's process-signal safety and execution-policy summaries in
  the returned incident-bundle report and manifest summary. It merged after
  Claude and Hermes approved with no findings and CI was green; Cursor was
  absent, so the low-risk read-only tooling degraded gate was used. VPS5
  checkout was updated to `f761063f` without restarting running bots because
  the slice was read-only tooling. Bounded smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, zero
  hard log matches, and only known non-hard ZEC HSL cooldown attention. A
  focused VPS incident-bundle check proved the manifest's embedded
  restart-smoke summary includes `forbid_broad_process_pattern_signals=true`,
  `execute_flag=not_implemented`, `future_execution_requires_review=true`, and
  still omits config-preflight raw command lists.
- Repository pulled through PR #1038 at `d840c008`.
- PR #1038 made `live-incident-bundle --restart-smoke-plan` expose the embedded
  restart-smoke plan's planned smoke and follow-up incident-bundle command
  summaries in the returned incident-bundle report and manifest summary. It
  merged after Claude and Hermes approved with no findings and CI was green;
  Cursor was absent, so the low-risk read-only tooling degraded gate was used.
  VPS5 checkout was updated to `d840c008` without restarting running bots
  because the slice was read-only tooling. Bounded smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, zero
  hard log matches, no event-pipeline drops or sink errors, and only known
  non-hard ZEC HSL cooldown attention. A focused VPS incident-bundle check
  proved the manifest's embedded restart-smoke summary includes both planned
  command summaries with `execute=false`, preserves selected performance
  filters, and still omits config-preflight raw command lists.
- Repository pulled through PR #1037 at `b688fed9`.
- PR #1037 made `live-incident-bundle --restart-smoke-plan` expose the embedded
  restart-smoke plan's selected smoke/performance sections in the returned
  incident-bundle report and manifest summary. It merged after Claude and Hermes
  approved with no findings and CI was green; Cursor was absent, so the
  low-risk read-only tooling degraded gate was used. VPS5 checkout was updated
  to `b688fed9` without restarting running bots because the slice was read-only
  tooling. Bounded smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard log matches,
  no event-pipeline drops or sink errors, and only known non-hard ZEC HSL
  cooldown attention.
- Repository pulled through PR #1036 at `7c12497f`.
- PR #1036 made `live-incident-bundle --restart-smoke-plan` propagate selected
  `--performance-section` filters into the embedded restart plan's planned
  failure-bundle command. It merged after Claude and Hermes approved with no
  findings and CI was green; Cursor was absent, so the low-risk read-only
  tooling degraded gate was used. VPS5 checkout was updated to `7c12497f`
  without restarting running bots because the slice was read-only tooling. A
  focused VPS incident-bundle check proved embedded `restart_smoke_plan.json`
  had `inputs.performance_sections=["startup_readiness"]` and the planned
  incident-bundle command included `--performance-section startup_readiness`.
  Bounded smoke reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  clean tracked repository state, zero hard log matches, and only known
  non-hard ZEC HSL cooldown plus EMA-readiness attention.
- Repository pulled through PR #1035 at `9b83ca2a`.
- PR #1035 made `live-restart-smoke-plan --performance-section` pass selected
  performance-report sections through to the planned failure
  `live-incident-bundle --performance-report` command. It merged after Claude
  and Hermes approved with no findings and CI was green; Cursor was absent, so
  the low-risk read-only tooling degraded gate was used. VPS5 checkout was
  updated to `9b83ca2a` without restarting running bots because the slice was
  read-only tooling. A focused VPS planner check proved the generated
  incident-bundle command includes both `--performance-section
  startup_readiness` and `--performance-section hsl_replay_profile`, and the
  bounded smoke reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  clean tracked repository state, zero hard log matches, no event-pipeline
  drops or sink errors, and only known non-hard ZEC HSL cooldown attention.
- Repository pulled through PR #1034 at `f068d9a4`.
- PR #1034 made `live-smoke-report --section` accept base metadata selectors
  such as `repository`, `monitor`, and `event_window`. It merged after Claude
  and Hermes approved with no findings and CI was green; Cursor was absent, so
  the low-risk read-only tooling degraded gate was used. VPS5 checkout was
  updated to `f068d9a4` without restarting running bots because the slice was
  read-only tooling. A focused VPS check proved `--section repository` now
  returns clean checkout metadata directly, and the settled bounded smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked
  repository state, zero hard log/problem/process failures, and only known
  non-hard ZEC HSL cooldown attention on binance/gateio/okx.
- Repository pulled through PR #1033 at `0a225f10`.
- PR #1033 added `live-incident-bundle --performance-section` for embedded
  performance reports. It merged after Claude and Hermes approved with no
  findings and CI was green; Cursor was absent, so the low-risk read-only
  tooling degraded gate was used. VPS5 checkout was updated to `0a225f10`
  without restarting running bots because the slice was read-only tooling. A
  first 2-minute smoke was hard-red from a transient Kucoin
  `maintain_hourly_cycle` open-orders traceback, while process and event
  pipeline checks were green. After the transient log window rolled forward, a
  settled 2-minute smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard
  log/problem/process failures, and only known non-hard ZEC HSL cooldown
  attention on binance/gateio/okx.
- Repository pulled through PR #1032 at `0f6611ae`.
- PR #1032 updated `live-restart-smoke-plan` so planned post-failure incident
  bundles include the existing `--performance-report` artifact by default. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `0f6611ae` without restarting running bots
  because the slice was read-only tooling. A first 5-minute smoke observed one
  transient Kucoin `RequestTimeout` degradation, and a tighter settled
  follow-up smoke was hard-green with five configured bots running, clean
  tracked repository state, and only known non-hard HSL cooldown/status plus
  stale dropped Kucoin traceback attention.
- Repository pulled through PR #1031 at `7d001553`.
- PR #1031 added opt-in `live-incident-bundle --performance-report`, embedding
  the existing read-only performance report artifact and compact summary in
  incident bundles with compatible bundle bounds. It merged after Claude and
  Hermes approved with no findings and CI was green. VPS5 checkout was updated
  to `7d001553` without restarting running bots because the slice was read-only
  tooling. Post-deploy smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard
  log/problem/process failures, and only known non-hard HSL cooldown/status
  attention plus dropped stale Kucoin traceback fragments under the drop
  policy. A focused bundle smoke verified `performance_report.json` was present
  in the generated archive.
- Repository pulled through PR #1030 at `a5faeaa8`.
- PR #1030 added `live-performance-report --debug-profile`, letting
  performance summaries scope to events enriched by one live-event debug
  profile. It merged after Claude approval, green CI, and repeated absent
  Hermes/Cursor polling under the degraded low-risk tooling gate; Hermes later
  approved the merged SHA with no findings. VPS5 checkout was updated to
  `a5faeaa8` without restarting running bots because the slice was read-only
  tooling. Post-deploy smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero hard
  log/problem/process failures, and only known non-hard HSL cooldown/status
  attention.
- Repository pulled through PR #1029 at `65a01d27`.
- PR #1029 added `live-incident-bundle --debug-profile` and passed the filter
  through embedded event, problem-event, and time-window reports plus manifest
  metadata. It merged after Claude and Hermes approved with no findings and CI
  was green; Cursor did not post before the degraded low-risk tooling gate was
  used. VPS5 checkout was updated to `65a01d27` without restarting running bots
  because the slice was read-only tooling. Post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state,
  zero hard log/problem/process failures, and only known non-hard HSL
  cooldown/status attention.
- Repository pulled through PR #1028 at `87754dc2`.
- PR #1028 added the read-only `live-event-query --debug-profile` filter. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `87754dc2` without restarting running bots
  because the slice was read-only tooling. Smoke stayed hard-green with all five
  configured bots running and only known non-hard HSL cooldown/status attention.
- Repository pulled through PR #1027 at `9c555384`.
- PR #1027 added the opt-in `startup` debug profile for existing
  `bot.startup_timing` events. It merged after Claude and Hermes approved with
  no findings and CI was green. VPS5 checkout was updated to `9c555384`
  without restarting running bots because default startup behavior and console
  output are unchanged. Post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state,
  zero event-pipeline drops/sink errors, and only known non-hard HSL
  cooldown/status attention.
- Repository pulled through PR #1026 at `2de5a6af`.
- PR #1026 added the opt-in `state` debug profile for existing
  `state.refresh_timing` and `state.refresh_progress` events. It merged after
  Claude and Hermes approved with no findings and CI was green. VPS5 checkout
  was updated to `2de5a6af` without restarting running bots because the slice
  was observability-only. Recent smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state, zero event-pipeline
  drops/sink errors, and only known non-hard HSL cooldown/status attention.
- Repository pulled through PR #1025 at `7f8a1942`.
- PR #1025 added the opt-in `forager` debug profile for existing
  `forager.selection` and `forager.feature_unavailable` events. It merged after
  Claude and Hermes approved with no findings and CI was green. VPS5 checkout
  was updated to `7f8a1942` without restarting running bots. Post-deploy smoke
  reported `ok=true`, `hard_failures=0`, clean tracked repository state, five
  configured bots running, and no event-pipeline drops or sink errors.
- Repository pulled through PR #1024 at `59c36ada`.
- PR #1024 added allowlisted `risk_events.latest_data` to brief smoke output.
  It merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `59c36ada` without restarting running bots.
  Post-deploy smoke reported `ok=true`, `hard_failures=0`, clean tracked
  repository state, and five configured bots still running.
- Repository pulled through PR #1020 at `85080299`.
- PR #1020 added a bundle-level `result` verdict to
  `live-incident-bundle` `manifest.json`, sharing the returned report's
  total `ok` and `hard_failures` calculation. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `85080299` without restarting running bots because the slice was report-only.
  The first post-deploy smoke caught an unrelated transient Kucoin
  `cycle.degraded` market snapshot miss still inside the 20-minute window; a
  Kucoin-focused event query then showed subsequent clean `cy_32` candle work
  with hundreds of successful remote calls and no errors. After the transient
  aged out, standard 20-minute smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=85080299`, no failed remote calls, no failed
  account-critical remote calls, and no hard log matches. A focused
  incident-bundle verification confirmed `manifest.result` matched the bundle
  result.
- Repository pulled through PR #1019 at `b741f49b`.
- PR #1019 added top-level smoke verdict fields to `live-incident-bundle`
  `manifest.json`: `ok`, `attention`, `hard_failures`, and `attention_count`.
  It merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `b741f49b` without restarting running bots
  because the slice was report-only. Recent-window smoke with stale unparsed
  traceback fragments dropped reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=b741f49b`, no failed remote calls, no failed
  account-critical remote calls, and no process hard failures. A focused
  incident-bundle verification confirmed `manifest.json` included the four
  smoke verdict fields and that they matched `smoke_report.json`.
- Repository pulled through PR #1018 at `9488f2ad`.
- PR #1018 added bounded repository and monitor smoke summaries to
  `live-incident-bundle` returned JSON and `manifest.json`, making checkout
  cleanliness and monitor event-count context visible in bundle-level triage.
  It merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `9488f2ad` without restarting running bots
  because the slice was report-only. Recent-window smoke with stale unparsed
  traceback fragments dropped reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=9488f2ad`, no failed remote calls, and no process hard
  failures. A focused incident-bundle verification confirmed `manifest.json`
  included `smoke_report.repository` and `smoke_report.monitor`, with
  repository dirty state false, zero tracked changes, and zero monitor errors.
- Repository pulled through PR #1017 at `c493ea5d`.
- PR #1017 added bounded text-log and event-window smoke summaries to
  `live-incident-bundle` `manifest.json`. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `c493ea5d` without restarting running bots because the slice was
  report-only. Recent-window smoke with stale unparsed traceback fragments
  dropped reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean
  tracked repository state at `repository.head=c493ea5d`, and no process hard
  failures. A focused incident-bundle verification confirmed `manifest.json`
  included `smoke_report.logs` and `smoke_report.event_window` with log hard
  matches at zero and the event window enabled.
- Repository pulled through PR #1016 at `8d2499ad`.
- PR #1016 added the bounded process smoke summary to `live-incident-bundle`
  returned JSON and `manifest.json`. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `8d2499ad` without restarting running bots because the slice was
  report-only. Recent-window smoke after the fast-forward reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state at
  `repository.head=8d2499ad`, and no process hard failures. A focused
  incident-bundle verification confirmed `manifest.json` included
  `smoke_report.processes` with `expected_total=5`, `matched_expected=5`,
  no missing/duplicate/unexpected processes, and zero config-check issues.
- Repository pulled through PR #1015 at `f02d53a9`.
- PR #1015 added smoke verdict source breakdowns and recovered problem-event
  counts to `live-incident-bundle` returned JSON and `manifest.json`. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `f02d53a9` without restarting running bots
  because the slice was report-only. Smoke after the fast-forward reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state at `repository.head=f02d53a9`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and no
  hard log matches. A focused incident-bundle verification confirmed the
  returned report and `manifest.json` both included `hard_failure_sources`,
  `attention_sources`, `recovered_problem_events`, and `problem_events`.
- Repository pulled through PR #1014 at `d5a9680f`.
- PR #1014 added the bounded `problem_events` smoke summary to
  `live-incident-bundle` returned JSON and `manifest.json`, including hard and
  non-hard problem-event type histograms. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `d5a9680f` without restarting running bots because the slice was report-only.
  Smoke after the fast-forward reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=d5a9680f`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and no
  hard log matches. A focused incident-bundle verification confirmed the
  returned report and `manifest.json` both included the new `problem_events`
  section with hard/non-hard event-type histogram keys.
- Repository pulled through PR #1013 at `bc0eb4fe`.
- PR #1013 split structured problem-event type histograms into hard and
  non-hard counts in `live-smoke-report --summary` and `--brief`, making mixed
  smoke attention easier to triage without opening grouped event rows. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `bc0eb4fe` without restarting running bots
  because the slice was report-only. Smoke after the fast-forward reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state at `repository.head=bc0eb4fe`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and no
  hard log matches. The new `problem_events.hard_event_types` and
  `problem_events.non_hard_event_types` fields were visible; current attention
  remained non-hard EMA-readiness, candle coverage, cycle degraded, and HSL
  cooldown status.
- Repository pulled through PR #1012 at `b8cfe90b`.
- PR #1012 added a bounded `problem_events.non_hard` count to
  `live-smoke-report --summary` and `--brief`, making non-fatal structured
  attention easier to distinguish from hard problem events without changing
  smoke verdict policy. It merged after Claude and Hermes approved with no
  findings and CI was green. VPS5 checkout was updated to `b8cfe90b` without
  restarting running bots because the slice was report-only. Smoke after the
  fast-forward reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  clean tracked repository state at `repository.head=b8cfe90b`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `fill_refresh.failed=0`, and no hard log matches. The new
  `problem_events.non_hard` field was visible with known non-hard
  EMA-readiness and HSL cooldown attention.
- Repository pulled through PR #1011 at `57a52e80`.
- PR #1011 added CLI smoke-section aliases so brief names such as
  `fill_refresh`, `hsl_replay`, and `remote_calls` resolve to their embedded
  full-report smoke sections. It merged after Claude and Hermes approved with
  no findings and CI was green. VPS5 checkout was updated to `57a52e80`
  without restarting running bots because the slice was report-only/tooling
  only. Smoke after the fast-forward reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=57a52e80`, no failed remote calls, no failed
  account-critical remote calls, `fill_refresh.failed=0`, and no hard log
  matches. Remaining attention came from known non-hard EMA readiness and ZEC
  HSL cooldown status. A focused incident-bundle verification confirmed
  `--smoke-section fill_refresh` selected embedded `fill_refresh_health`,
  omitted unrelated embedded full sections, and preserved
  `manifest.filters.smoke_sections=["fill_refresh"]`.
- Repository pulled through PR #1010 at `2f91372a`.
- PR #1010 added bounded data-plane smoke projections to
  `live-incident-bundle` returned JSON and `manifest.json`: remote calls,
  account-critical remote calls, fill refresh, startup timings, and HSL replay.
  It merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `2f91372a` without restarting running bots
  because the slice was report-only. Smoke after the fast-forward reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state at `repository.head=2f91372a`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and no
  hard log matches. Remaining attention came from known non-hard EMA readiness
  and ZEC HSL cooldown status. A focused incident-bundle verification confirmed
  `remote_calls`, `account_critical_remote_calls`, `fill_refresh`,
  `startup_timings`, and `hsl_replay` were present in both the returned report
  and `manifest.json`.
- Repository pulled through PR #1009 at `60ed8f60`.
- PR #1009 added bounded operational smoke projections to
  `live-incident-bundle` returned JSON and `manifest.json`: exchange config
  refresh, staged readiness, event-pipeline health, and shutdown events. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `60ed8f60` without restarting running bots
  because the slice was report-only. Smoke after the fast-forward reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `fill_refresh.failed=0`, and no hard log matches. A focused
  incident-bundle verification confirmed all four operational manifest
  sections were present.
- Repository pulled through PR #1008 at `52a2fbd3`.
- PR #1008 added the bounded `ema_readiness` brief smoke projection to
  `live-incident-bundle` returned JSON and `manifest.json`, reusing the same
  safe EMA readiness summary surfaced by brief smoke reports. It merged after
  Claude and Hermes approved with no findings and CI was green. VPS5 checkout
  was updated to `52a2fbd3` without restarting running bots because the slice
  was report-only. A focused incident-bundle verification confirmed
  `manifest.smoke_report.ema_readiness` was present. Final short-window smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `fill_refresh.failed=0`. A
  transient Kucoin hourly market-refresh timeout appeared in a wider window but
  aged out; subsequent Kucoin events showed successful state/fill/candle work.
- Repository pulled through PR #1007 at `9d3b4051`.
- PR #1007 added the bounded `risk_events` brief smoke projection to
  `live-incident-bundle` returned JSON and `manifest.json`, reusing the same
  safe risk-event summary surfaced by brief smoke reports. It merged after
  Claude and Hermes approved with no findings and CI was green. VPS5 checkout
  was updated to `9d3b4051` without restarting running bots because the slice
  was report-only. Smoke after the fast-forward reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`, and
  `fill_refresh.failed=0`. A focused incident-bundle verification confirmed
  `manifest.smoke_report.risk_events` was present.
- Repository pulled through PR #1006 at `68c3fe22`.
- PR #1006 added bounded `risk_events.attention_groups` to
  `live-smoke-report --brief`, prioritizing HSL RED/cooldown/raw-red and
  panic-mode context over routine latest-event ordering while emitting only
  safe group metadata. It merged after Claude and Hermes approved with no
  findings and CI was green. VPS5 checkout was updated to `68c3fe22` without
  restarting running bots because the slice was report-only. Smoke after the
  fast-forward reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  clean tracked repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `fill_refresh.failed=0`. The
  new `risk_events.attention_groups` field was visible and surfaced current
  ZEC cooldown evidence.
- Repository pulled through PR #1005 at `f54dae3e`.
- PR #1005 added bounded `risk_events.latest_groups` to
  `live-smoke-report --brief`, sourced from existing structured risk events and
  limited to safe metadata fields. VPS5 smoke after deploy reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state at
  `repository.head=f54dae3e`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `fill_refresh.failed=0`. The
  new field was visible in smoke output. A wider risk-events-only smoke
  confirmed current HSL cooldown/raw-red evidence, but also showed that
  latest-time ordering can bury older RED/cooldown context behind newer routine
  green HSL status groups, motivating the current attention-groups follow-up.
- Repository pulled through PR #985 at `947b75b1`.
- PR #985 projected existing `execution.create_skipped` low-balance create-skip
  events into the structured console/text sinks and made the legacy
  `[balance] too low` line fallback-only. It merged after Claude and Hermes
  approved with no findings and CI was green. VPS5 checkout was updated to
  `947b75b1` on disk without restarting running bots because the slice was
  console-only and HSL replay remained active. Smoke after the fast-forward
  reported `ok=true`, `hard_failures=0`, five running live processes, clean
  tracked repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `fill_refresh.failed=0`;
  remaining attention was non-hard active HSL replay and Hyperliquid EMA
  readiness diagnostics.
- Repository pulled through PR #984 at `b15da359`.
- PR #984 suppressed duplicate legacy balance and position-change stdlib
  console lines when the structured live-event console path is active. It
  merged after Claude and Hermes approved with no findings and CI was green.
  VPS5 checkout was updated to `b15da359` on disk without restarting running
  bots because the slice was console-only and HSL replay remained active. Smoke
  after the fast-forward reported `ok=true`, `hard_failures=0`, five running
  live processes, clean tracked repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `fill_refresh.failed=0`;
  remaining attention was non-hard active HSL replay and EMA/readiness
  diagnostics.
- Repository pulled through PR #983 at `f6700c5cd`.
- PR #983 suppressed legacy order-wave complete/settled stdlib console lines
  when the structured live-event console path is active. It merged after Claude
  and Hermes approved with no findings, CI was green, and local focused tests
  passed. VPS5 checkout was updated to `f6700c5cd` on disk without restarting
  running bots because the slice was console-only and HSL replay was still
  active.
- Repository pulled through PR #982 at `c7a89c9c`.
- PR #982 made flat coin-mode HSL cooldown finalizations informational instead
  of critical when no exchange close was needed. It merged after Claude and
  Hermes approved with no findings and CI was green. VPS5 bots were restarted
  and left running; settled smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=c7a89c9c`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and
  `logs.hard_matches=0`. Remaining attention was non-hard active HSL replay
  and EMA-unavailable/staged-refresh diagnostics.
- Repository pulled through PR #972 at `f789dccc`.
- PR #972 projected existing `bot.startup_timing` events into the default live
  event console/text sinks and kept trading behavior unchanged. It merged after
  Hermes and Claude approved with no findings, CI was green, and Cursor did not
  post during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited in the first graceful window; Binance, GateIO, and OKX exited during
  the longer graceful window; Kucoin still required SIGTERM after the full wait
  window.
- VPS5 settled smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=f789dccc`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and
  `logs.hard_matches=0`. Remaining attention was non-hard active HSL replay on
  the four forager bots plus Hyperliquid EMA-unavailable/staged-refresh
  diagnostics.
- Focused log inspection confirmed the new structured startup projection was
  active on VPS5, for example Hyperliquid emitted `[boot] succeeded
  phase=account-ready ... reason=startup_phase_ready` and related
  active-candle/startup/market/full-warmup phase lines.
- Repository pulled through PR #971 at `45cd1d7e`.
- PR #971 made the structured live event console projection default-on for
  `passivbot live` while preserving explicit config/env opt-outs. It merged
  after Claude and Hermes approved the amended head, CI was green, and no open
  PRs remained.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited in the first graceful window; Binance, Kucoin, GateIO, and OKX exited
  during the longer graceful window without SIGTERM.
- VPS5 settled smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=45cd1d7e`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and
  `logs.hard_matches=0`. Remaining attention was non-hard active HSL replay on
  the four forager bots plus Hyperliquid EMA-unavailable/staged-refresh
  diagnostics.
- Focused log inspection confirmed the default console projection was active:
  Hyperliquid emitted structured `[trailing]` and `[unstuck]` summaries with
  threshold, retracement, allowance, and distance fields, while legacy duplicate
  text lines still remained for some account-state changes pending later
  cleanup.
- Repository pulled through PR #970 at `ac425a1f0`.
- PR #970 refined existing trailing/unstuck event-console summaries, routed the
  already-throttled `unstuck.selection` event to the opt-in console/text sinks,
  and kept trading behavior unchanged. It merged after Claude and Hermes
  approved with no findings and CI was green; Cursor did not post during the
  bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited during the first graceful window; Binance, GateIO, and OKX exited
  during the longer graceful window; Kucoin still required SIGTERM after the
  full wait window.
- VPS5 settled smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=ac425a1f`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and
  `logs.hard_matches=0`. Remaining attention was non-hard active HSL replay on
  three forager bots and Hyperliquid EMA-unavailable diagnostics.
- The deployed VPS5 configs did not set `logging.live_event_console`, so PR
  #970's new console projection was present in the event-console path but not
  visible in the default legacy console logs on that host.
- Repository pulled through PR #969 at `426f3ae0b`.
- PR #969 projected existing `health.summary` events into the opt-in live event
  console/text sinks with compact operator summaries. It kept periodic health
  as an INFO-level event matching the existing legacy `[health]` line, kept
  degraded execution-loop error bursts immediate, and did not change health
  producer timing, trading behavior, exchange I/O, or resource counters. It
  merged after Hermes approved, Claude approved, and CI was green; Cursor did
  not post during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited within the first graceful window; Binance, GateIO, and OKX exited
  during the longer graceful window; Kucoin still required SIGTERM after the
  full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked repository state at
  `repository.head=426f3ae0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  `fill_refresh.failed=0`. The event pipeline saw `health.summary` with no
  degraded/dropped/sink errors. Remaining attention was non-hard Hyperliquid
  EMA-unavailable diagnostics and active HSL replay.
- Repository pulled through PR #968 at `9ff2b8582`.
- PR #968 projected existing initial-entry distance gate blocked/cleared,
  min-effective-cost entry skip, and realized-loss gate deferral events into
  the opt-in live event console/text sinks with compact operator summaries. It
  merged after Hermes and Claude approved and CI was green; Cursor did not post
  during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, Kucoin, GateIO, and OKX were still
  present after 25 seconds, and all exited gracefully within the longer wait
  window without SIGTERM.
- VPS5 smoke immediately after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=9ff2b858`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `logs.hard_matches=0`.
  A later smoke observed one Hyperliquid routine fill-prefetch `RequestTimeout`
  which recovered on subsequent refreshes while fill coverage remained ready.
  The final short-window smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, clean tracked state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `fill_refresh.failed=0`, and
  `logs.hard_matches=0`. Remaining attention was non-hard Hyperliquid EMA
  unavailable diagnostics for cache-only stock symbols and active HSL replay on
  two bots.
- Repository pulled through PR #967 at `ca686cf5a`.
- PR #967 projected existing `risk.mode_changed` and `hsl.transition` events
  into the opt-in live event console/text sinks with compact operator
  summaries. It merged after Hermes and Claude approved and CI was green; Cursor
  did not post during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, Kucoin, GateIO, and OKX were still
  present after the first 25-second grace window, then all exited within the
  longer graceful wait without SIGTERM.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=ca686cf5`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `logs.hard_matches=0`.
  A second smoke after another five minutes stayed hard-green with the same
  core signals. Remaining attention was non-hard startup state: active HSL
  replay workers on four bots, Hyperliquid EMA-unavailable diagnostics for
  cache-only stock symbols, staged refresh progress, and shutdown-slow history
  from the restart.
- Repository pulled through PR #966 at `47f84d8c5`.
- PR #966 projected existing `fill.ingested`, `position.changed`, and
  `balance.changed` events into the opt-in live event console/text sinks with
  compact operator summaries. It merged after Hermes approved, Claude approved
  with the docs sensitivity note addressed, and CI was green; Cursor did not
  post during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=47f84d8c`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `logs.hard_matches=0`.
  Remaining attention was non-hard startup state: active HSL replay workers,
  Hyperliquid EMA-unavailable diagnostics for cache-only stock symbols, staged
  refresh progress, and shutdown-slow history from the restart.
- Repository pulled through PR #965 at `191af58ab`.
- PR #965 enriched existing trailing and unstuck console summaries with
  distance-to-threshold, distance-to-retracement, unstuck target distance, and
  the already-computed monitor runtime hint for the next unstuck candidate. It
  merged after Hermes approved, Claude approved with no findings, and CI was
  green; Cursor did not post during the bounded wait.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=191af58a`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `logs.hard_matches=0`.
  Remaining attention was non-hard startup state: active HSL replay workers,
  Hyperliquid EMA-unavailable diagnostics for cache-only stock symbols, staged
  refresh progress, and shutdown-slow history from the restart.
- A focused monitor-event check confirmed deployed `trailing.status` events for
  Hyperliquid include the new distance ratio fields. The observed
  `unstuck.status` event had no next-candidate hint because no close-unstuck
  candidate was active in that window; that path is covered by focused tests.
- Repository pulled through PR #964 at `b9a3110e8`.
- PR #964 projected existing `forager.selection` events into the opt-in live
  event console/text sinks with a compact 5-minute-throttled operator summary.
  It merged after Claude and Hermes approved with CI green.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=b9a3110e`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and `logs.hard_matches=0`.
  Remaining attention was non-hard startup state: four active HSL replay
  workers, Hyperliquid EMA-unavailable diagnostics for cache-only stock
  symbols, staged refresh progress, and shutdown-slow history from the restart.
- Repository pulled through PR #963 at `f54c92147`.
- PR #963 refined the trailing `selected_mode` classifier so
  `close_auto_reduce_wel_long` reports `auto_reduce` instead of generic `grid`.
  It merged after Claude and Hermes approved with CI green.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=f54c9214`, `account_critical_remote_calls.failed=0`,
  and `logs.hard_matches=0`. The focused follow-up report showed
  `remote_calls.failed=0`; remaining attention was non-hard active HSL replay,
  Hyperliquid EMA-unavailable diagnostics for cache-only stock symbols, staged
  refresh progress, and shutdown-slow history from the restart.
- The deployed Hyperliquid `XYZ-MU/USDC:USDC` long `trailing.status` event now
  reports `selected_mode=auto_reduce` for
  `order_type=close_auto_reduce_wel_long`, with threshold/retracement fields
  still present and `triggered=false`.
- Repository pulled through PR #962 at `675e93ba`.
- PR #962 kept trailing-martingale diagnostics visible when the current next
  order is grid/non-trailing, exposed `selected_mode` in `trailing.status`, and
  kept `triggered` false for non-trailing next orders. It merged after Claude
  and Hermes approved with CI green.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=675e93ba`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  `logs.attention_matches=0`. Remaining attention was non-hard: active HSL
  replay workers, Hyperliquid EMA-unavailable diagnostics for cache-only stock
  symbols, and shutdown-slow history from the restart.
- The deployed Hyperliquid `XYZ-MU/USDC:USDC` long `trailing.status` event now
  emits `diagnostics_supported=true` with threshold/retracement fields instead
  of `active_unsupported`. It also showed `order_type=close_auto_reduce_wel_long`
  with `selected_mode=grid`, motivating the current label-refinement follow-up.
- Repository pulled through PR #961 at `23552121`.
- PR #961 added Rust-aligned `trailing_grid_v7` monitor diagnostics so the same
  position-status stream can explain v7 compatibility positions instead of
  reporting that v7 trailing diagnostics are unsupported. The helper calls
  existing Rust v7 order functions and reports selected mode, WEL ratio,
  threshold/retracement status, and projected retracement price without
  changing order generation.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Hyperliquid
  exited quickly after Ctrl-C; Binance, GateIO, and OKX exited within the longer
  graceful window; Kucoin still required SIGTERM after the full wait window.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=23552121`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  `logs.attention_matches=0`. Remaining attention was non-hard: four active HSL
  replay workers, Hyperliquid EMA-unavailable diagnostics for cache-only stock
  symbols, and shutdown-slow history from the restart.
- The deployed event stream still reported the Hyperliquid `XYZ-MU/USDC:USDC`
  long position as `active_unsupported` because that bot uses
  `trailing_martingale` and the current helper suppresses diagnostics when the
  selected next order is not trailing. This is the reason for the current
  `codex/v8-trailing-waiting-diagnostics` follow-up.
- Repository pulled through PR #960 at `c4a23aa9`.
- PR #960 added structured `trailing.status` events for active trailing
  positions, projected `trailing.status` plus existing `unstuck.status` into the
  opt-in event console, and kept unsupported strategy diagnostics explicit until
  strategy-specific Rust diagnostics exist. It was merged after Claude and
  Hermes approved with CI green. Cursor had not posted a review after repeated
  polling, so this was treated as a reviewer-availability exception for an
  observability-only live-path slice; no reviewer findings were outstanding.
- Bots were restarted from `/root/bots_vps5.yaml` and left running. Graceful
  Ctrl-C shutdown improved for Binance but Kucoin, GateIO, OKX, and
  Hyperliquid still needed SIGTERM after the full Ctrl-C wait window; this
  remains evidence for the existing shutdown/backlog work.
- VPS5 smoke after restart reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=c4a23aa9`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `logs.hard_matches=0`, and
  `logs.attention_matches=0`. Attention was limited to four active HSL replay
  workers during startup, with no failed HSL replay bots.
- The deployed monitor event stream emitted a `trailing.status` event for the
  Hyperliquid `XYZ-MU/USDC:USDC` long position. It was correctly marked
  `active_unsupported` because the Rust-aligned `trailing_grid_v7` diagnostics
  follow-up had not merged yet.
- Repository pulled through PR #958 at `a989a3f2`.
- PR #958 refreshed v8 long-side default values in Rust metadata, Python schema,
  and the default example config. It was not a logging-overhaul slice, but it
  advanced `origin/v8` after PR #957 and was therefore deployed before the next
  logging branch was created. Running bots were not restarted because the
  deployed change only affects defaults for newly generated/loaded configs and
  the active VPS5 bots use explicit configs.
- PR #958 passed Hermes + Claude + CI. A VPS5 2-minute bounded smoke after the
  pull reported `ok=true`, `hard_failures=0`, clean tracked repository state at
  `a989a3f2`, all five configured bots matched, config checks green, zero
  failed remote calls, and zero failed account-critical remote calls. Remaining
  attention came from known non-hard EMA readiness and HSL cooldown/status
  groups.
- Repository pulled through PR #957 at `d038c405`.
- PR #957 added the disabled-by-default
  `logging.live_event_console` / `PASSIVBOT_LIVE_EVENT_CONSOLE` opt-in path for
  the existing structured-event `ConsoleSummarySink`, and then narrowed candle
  remote-fetch callback installation so console-only mode does not create
  unused remote-call event traffic. The slice was observability-only and did not
  add exchange calls, cache mutation, readiness gates, smoke verdict changes,
  process signaling, order logic, risk logic, or trading behavior.
- PR #957 passed Claude + Hermes + CI. Cursor did not post during the bounded
  wait, so the merge used a documented reviewer-availability exception rather
  than claiming full three-reviewer coverage. Local validation covered
  `tests/test_live_event_bus.py`, focused monitor pipeline tests, py_compile for
  touched files, `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `df10b379` to `d038c405` without bot restart because the new
  console sink is disabled by default. A VPS5 2-minute bounded smoke after
  deploy reported `ok=true`, `hard_failures=0`, clean tracked repository state
  at `d038c405`, all five configured bots matched, config checks green, zero
  failed remote calls, and zero failed account-critical remote calls. Remaining
  attention came from known non-hard EMA readiness and HSL cooldown/status
  groups.
- Repository pulled through PR #956 at `df10b379`.
- PR #956 limited `live-incident-bundle` manifest git status to tracked changes
  via `git status --short --untracked-files=no`, preserving tracked dirty-tree
  evidence while keeping local untracked configs and monitor artifacts out of
  the manifest status block. The slice was read-only incident-bundle tooling and
  did not add event producers, exchange calls, cache mutation, readiness gates,
  smoke verdict changes, process signaling, console routing, order logic, risk
  logic, or trading behavior.
- PR #956 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `69603923` to `df10b379` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A VPS5 2-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `df10b379`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and a populated brief `execution`
  section with recent create/cancel/confirmation events. A focused
  incident-bundle run from outside the repo with absolute `/root/passivbot`
  monitor/log paths reported `ok=true` and populated `smoke_report.execution`;
  archive inspection confirmed `manifest.git.cwd=/root/passivbot`,
  `manifest.git.branch=v8`, `manifest.git.head=df10b379...`, and
  `manifest.git.status_short=""` despite the known untracked local config and
  monitor artifacts. Remaining attention came from known non-hard EMA/HSL/candle
  coverage groups.
- Repository pulled through PR #955 at `69603923`.
- PR #955 made `live-incident-bundle` infer manifest git metadata from the
  monitor tree when invoked from outside the repo with an absolute monitor
  path, while preserving explicit `cwd` behavior and remote URL userinfo
  redaction. The slice was read-only incident-bundle tooling and did not add
  event producers, exchange calls, cache mutation, readiness gates, smoke
  verdict changes, process signaling, console routing, order logic, risk logic,
  or trading behavior.
- PR #955 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `3425a14a` to `69603923` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A VPS5 2-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `69603923`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and a populated brief `execution`
  section with recent create/cancel/confirmation events. A focused
  incident-bundle run from outside the repo with absolute `/root/passivbot`
  monitor/log paths reported `ok=true`, returned populated
  `smoke_report.execution`, and archive inspection confirmed
  `manifest.git.cwd=/root/passivbot`, `manifest.git.branch=v8`, and
  `manifest.git.head=69603923...`. Remaining attention came from known non-hard
  EMA/HSL cooldown groups.
- Repository pulled through PR #954 at `3425a14a`.
- PR #954 added a compact `execution` summary to `live-incident-bundle` report
  and manifest output, sourced from the existing `live-smoke-report`
  `execution_health` section. The projection includes only aggregate execution
  counters and event/status/outcome maps. The slice was read-only
  incident-bundle tooling and did not add event producers, exchange calls,
  cache mutation, readiness gates, smoke verdict changes, process signaling,
  console routing, order logic, risk logic, or trading behavior.
- PR #954 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `8981464a` to `3425a14a` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A VPS5 2-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `3425a14a`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and the new brief `execution` section
  present with zero recent execution events. A focused incident-bundle run with
  `--smoke-section execution_health --no-event-segments` reported `ok=true`
  and returned `smoke_report.execution` with aggregate zero counters for the
  quiet sample window; archive inspection confirmed
  `manifest.smoke_report.execution` was present. Remaining attention came from
  known non-hard EMA/HSL cooldown groups.
- Repository pulled through PR #953 at `8981464a`.
- PR #953 added an `execution_health` section to `live-smoke-report` full and
  summary output and an `execution` line to brief output, deriving aggregate
  execution-write health from existing `order_wave.*` and `execution.*`
  structured events. The slice was read-only smoke-report tooling and did not
  add event producers, exchange calls, cache mutation, readiness gates, smoke
  verdict changes, process signaling, console routing, order logic, risk logic,
  or trading behavior.
- PR #953 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_smoke_report.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `07ee5860` to `8981464a` without bot restart because the
  deployed change was read-only smoke-report tooling. The five configured bots
  were left running.
- A VPS5 1-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `8981464a`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and the new brief `execution` section
  present with zero recent execution events. Remaining attention came from known
  non-hard EMA/HSL cooldown groups.
- Repository pulled through PR #952 at `07ee5860`.
- PR #952 made `live-restart-smoke-plan` planned failure-bundle commands
  self-contained by adding `live-incident-bundle --restart-smoke-plan` and the
  same `--restart-smoke-window-minutes` value. The slice was read-only restart
  planner / incident-bundle tooling and did not add restart execution, process
  signaling, tmux calls, SSH/git operations, exchange calls, event producers,
  smoke verdict changes, console routing, order logic, risk logic, or trading
  behavior.
- PR #952 passed Hermes + Claude + CI. It was merged under the documented
  degraded low-risk tooling gate after Cursor absence. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `a5a3a83f` to `07ee5860` without bot restart because the
  deployed change was read-only restart planner / incident-bundle tooling. The
  five configured bots were left running.
- A VPS5 1-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `07ee5860`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and only known non-hard EMA/HSL
  cooldown attention groups. A focused restart-plan run confirmed planned
  failure-bundle commands include both `--restart-smoke-plan` and
  `--restart-smoke-window-minutes 30`, while staying `execute=false` /
  `plan_only=true`.
- Repository pulled through PR #951 at `a5a3a83f`.
- PR #951 added opt-in `live-incident-bundle --restart-smoke-plan`, requiring
  `--supervisor-config`, to embed a non-executing `restart_smoke_plan.json`
  artifact and value-safe restart-plan summary in incident bundles. The
  embedded plan uses the restart planner's bounded smoke defaults instead of
  inheriting the incident bundle's event/log scan settings. The slice was
  read-only incident-response tooling and did not add restart execution,
  process signaling, tmux calls, SSH/git operations, exchange calls, event
  producers, smoke verdict changes, console routing, order logic, risk logic,
  or trading behavior.
- PR #951 passed Hermes + Claude + CI after an amendment fixed the embedded
  plan scan defaults and removed raw config-preflight command strings from the
  bundle result/manifest summary. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_restart_smoke_plan.py`,
  py_compile for touched files, `git diff --check`, and an added-line
  silent-handling scan.
- VPS5 pulled from `48919ea1` to `a5a3a83f` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A VPS5 1-minute bounded smoke after deploy reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `a5a3a83f`, all five
  configured bots matched, config checks green, zero failed remote calls, zero
  failed account-critical remote calls, and only known non-hard EMA/HSL
  cooldown attention groups. A focused incident-bundle run with
  `--restart-smoke-plan --restart-smoke-window-minutes 1 --no-event-segments`
  reported `ok=true`, five bots in the embedded restart plan, two config
  preflight commands counted, zero skipped config paths, and archive inspection
  confirmed `restart_smoke_plan.json` was present.
- Repository pulled through PR #950 at `48919ea1`.
- PR #950 added `skipped_without_config_path_count` to the full and summary
  `live-restart-smoke-plan` `config_preflight` output, keeping planned restart
  readiness evidence explicit when a configured live command has no derivable
  config path. The slice was read-only planner tooling and did not add restart
  execution, process signaling, tmux calls, SSH/git operations, exchange calls,
  event producers, smoke verdict changes, console routing, order logic, risk
  logic, or trading behavior.
- PR #950 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `c7b09bf8` to `48919ea1` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan /root/bots_vps5.yaml --smoke-section
  fill_refresh_health --log-window-unparsed-policy drop --summary --compact`
  run reported `ok=true`, five configured bots, six planned phases,
  `execute=false`, zero issues, two deduplicated config-preflight commands, and
  `skipped_without_config_path_count=0`. A follow-up 1-minute smoke reported
  `ok=true`, `hard_failures=0`, clean tracked repository state at `48919ea1`,
  all five configured bots matched, config checks green, zero failed remote
  calls, zero failed account-critical remote calls, and only known non-hard
  EMA/HSL cooldown attention groups.
- Repository pulled through PR #949 at `c7b09bf8`.
- PR #949 added deduplicated planned `live-config-preflight ... --compact`
  commands to `live-restart-smoke-plan` pre-restart readiness and summary
  output. The slice was read-only dry-run planner tooling and did not add
  restart execution, process signaling, tmux calls, SSH/git operations, exchange
  calls, event producers, smoke verdict changes, console routing, order logic,
  risk logic, or trading behavior.
- PR #949 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `79478e1f` to `c7b09bf8` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan /root/bots_vps5.yaml --smoke-section
  fill_refresh_health --log-window-unparsed-policy drop --summary --compact`
  run reported `ok=true`, five configured bots, six planned phases,
  `execute=false`, zero issues, and two deduplicated config-preflight commands
  for the configured forager and tradfi configs. A follow-up 1-minute smoke
  reported `ok=true`, `hard_failures=0`, clean tracked repository state at
  `c7b09bf8`, all five configured bots matched, config checks green, zero
  failed remote calls, zero failed account-critical remote calls, and only known
  non-hard EMA/HSL cooldown attention groups.
- Repository pulled through PR #948 at `79478e1f`.
- PR #948 added `live-restart-smoke-plan --log-window-unparsed-policy`, passing
  the existing `keep`/`drop` text-log window policy through to both planned
  evidence commands: `live-smoke-report` and `live-incident-bundle`. The slice
  was read-only dry-run planner tooling and did not add restart execution,
  process signaling, tmux calls, SSH/git operations, exchange calls, event
  producers, smoke verdict changes, console routing, order logic, risk logic, or
  trading behavior.
- PR #948 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `123d59d6` to `79478e1f` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan /root/bots_vps5.yaml --smoke-section
  fill_refresh_health --log-window-unparsed-policy drop --summary --compact`
  run reported `ok=true`, five configured bots, six planned phases,
  `execute=false`, zero issues, and both planned evidence commands carrying
  `--log-window-unparsed-policy drop`. A follow-up 1-minute smoke reported
  `ok=true`, `hard_failures=0`, clean tracked repository state at `79478e1f`,
  all five configured bots matched, zero failed remote calls, zero failed
  account-critical remote calls, text-log `unparsed_policy=drop`, and only known
  non-hard EMA/HSL cooldown attention groups.
- Repository pulled through PR #947 at `123d59d6`.
- PR #947 made `live-restart-smoke-plan --smoke-section` apply to both planned
  evidence commands: `live-smoke-report --section` and
  `live-incident-bundle --smoke-section`. The slice was read-only dry-run
  planner tooling and did not add restart execution, process signaling, tmux
  calls, SSH/git operations, exchange calls, event producers, smoke verdict
  changes, console routing, order logic, risk logic, or trading behavior.
- PR #947 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `fddd5491` to `123d59d6` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan /root/bots_vps5.yaml --smoke-section
  fill_refresh_health --summary --compact` run reported `ok=true`, five
  configured bots, six planned phases, `execute=false`, zero issues, and
  matching focused evidence commands: `live-smoke-report ... --section
  fill_refresh_health` and `live-incident-bundle ... --smoke-section
  fill_refresh_health`. A follow-up 1-minute smoke reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `123d59d6`, all five
  configured bots matched, zero failed remote calls, zero failed
  account-critical remote calls, and only known non-hard EMA/HSL cooldown
  attention groups.
- Repository pulled through PR #946 at `fddd5491`.
- PR #946 added `live-incident-bundle --smoke-section`, allowing the embedded
  full `smoke_report.json` inside an incident bundle to keep selected
  top-level smoke-report sections plus common metadata while preserving the
  compact incident-bundle summary from the unfiltered smoke report. The slice
  was read-only incident-response tooling and did not change event producers,
  exchange calls, cache mutation, readiness gates, smoke verdict policy,
  console routing, order logic, risk logic, or trading behavior.
- PR #946 passed CI plus fresh Cursor and Claude approval at amended head
  `5106af2c`. Hermes approved the original larger head and did not post a
  second-pass review after the narrow amendment; the amendment only kept the
  bundle summary unprojected while projecting the archived `smoke_report.json`.
  Local validation covered `tests/test_live_incident_bundle.py`, py_compile for
  touched files, `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `1a7c83ee` to `fddd5491` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A VPS5 focused incident-bundle smoke using `--smoke-section
  fill_refresh_health` reported `ok=true`, `hard_failures=0`, and preserved
  compact `logs`/`processes` summary fields. Archive inspection confirmed the
  embedded `smoke_report.json` recorded
  `filters.smoke_sections=["fill_refresh_health"]`, included
  `fill_refresh_health`, and omitted `logs` as intended. A follow-up 1-minute
  smoke reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `fddd5491`, all five
  configured bots matched, zero failed remote calls, zero failed
  account-critical remote calls, and only known non-hard `ema.unavailable`
  attention groups.
- Repository pulled through PR #945 at `1a7c83ee`.
- PR #945 added `live-restart-smoke-plan --smoke-section`, allowing the
  plan-only restart smoke command to include focused `live-smoke-report
  --section` values while recording the selected sections in plan inputs. The
  slice was read-only dry-run planner tooling and did not add restart
  execution, process signaling, tmux calls, SSH/git operations, exchange calls,
  event producers, smoke verdict changes, console routing, order logic, risk
  logic, or trading behavior.
- PR #945 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `42999f19` to `1a7c83ee` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan /root/bots_vps5.yaml --smoke-section
  fill_refresh --summary --compact` run reported `ok=true`, five configured
  bots, six planned phases, `execute=false`, zero issues, and a planned smoke
  command containing `--brief --section fill_refresh --compact`.
- The first post-deploy 2-minute smoke surfaced one real transient GateIO
  `cycle.degraded` hard event from `RequestTimeout` in cycle `cy_2130`,
  involving one `authoritative_balance` timeout and one candle `fetch_ohlcv`
  timeout for `POL/USDT:USDT`. A settled 1-minute smoke then reported
  `ok=true`, `hard_failures=0`, clean tracked repository state at `1a7c83ee`,
  all five configured bots matched, no failed remote calls, and no failed
  account-critical remote calls.
- Repository pulled through PR #944 at `42999f19`.
- PR #944 added `live-smoke-report --section`, allowing focused top-level
  smoke-report sections plus common smoke metadata after the full, summary, or
  brief projection is selected. The slice was read-only operator/report tooling
  and did not change event producers, exchange calls, cache mutation, readiness
  gates, smoke verdict policy, console routing, order logic, risk logic, or
  trading behavior.
- PR #944 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_smoke_report.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `2dab5af0` to `42999f19` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A 2-minute smoke after deploy reported `ok=true`, `hard_failures=0`, clean
  tracked repository state at `42999f19`, all five configured bots matched, no
  failed account-critical remote calls, and `fill_refresh` populated with nine
  succeeded and zero failed refreshes. A focused smoke report using
  `--brief --section fill_refresh` then returned only common smoke metadata plus
  `fill_refresh`, with `fill_refresh.total=9` and `failed=0`.
- Repository pulled through PR #943 at `2dab5af0`.
- PR #943 added `live-performance-report --section`, allowing focused
  top-level report sections plus common metadata after the full or summary
  projection. The slice was read-only operator/report tooling and did not
  change event producers, exchange calls, cache mutation, readiness gates,
  console routing, order logic, risk logic, or trading behavior.
- PR #943 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `a85427bf` to `2dab5af0` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A 2-minute smoke after deploy reported `ok=true`, `hard_failures=0`, clean
  tracked repository state at `2dab5af0`, all five configured bots matched, no
  failed account-critical remote calls, and `fill_refresh` populated with eight
  succeeded and zero failed refreshes. A focused 10-minute performance report
  using `--summary --section fill_refresh` then returned only common metadata
  plus `fill_refresh`, with `fill_refresh.total_events=38`,
  `failed_groups=0`, and `latest_failed_groups=0`.
- Repository pulled through PR #942 at `a85427bf`.
- PR #942 added a `fill_refresh` section to `live-performance-report`, derived
  only from existing `fills.refresh_summary` events. It also includes
  `fills_refresh.elapsed` in the report's performance and operation-duration
  tables with `blocks_or_delays_hsl_readiness` impact classification. The slice
  did not change event producers, exchange calls, cache mutation, readiness
  gates, console routing, order logic, risk logic, or trading behavior.
- PR #942 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `462afac7` to `a85427bf` without bot restart because the
  deployed change was read-only performance-report tooling. The five configured
  bots were left running.
- A 2-minute smoke after deploy reported `ok=true`, `hard_failures=0`, clean
  tracked repository state at `a85427bf`, all five configured bots matched, no
  failed account-critical remote calls, and the new brief `fill_refresh`
  projection populated with seven succeeded and zero failed refreshes. A focused
  10-minute performance report then reported `ok=true`,
  `fill_refresh.total_events=38`, `failed_groups=0`,
  `latest_failed_groups=0`, and
  `operation_durations.operation_category_counts.fill_refresh=4`.
- Repository pulled through PR #941 at `462afac7`.
- PR #941 added bounded `fill_refresh_health` projections to
  `live-smoke-report` full/summary output and concise `fill_refresh` counters
  to brief output, derived only from existing off-console
  `fills.refresh_summary` monitor events. The projection reports status/error
  counts, latest failed bots, recovered groups, and bounded group detail without
  changing smoke verdict policy, event producers, exchange calls, cache
  mutation, readiness gates, console routing, order logic, risk logic, or
  trading behavior.
- PR #941 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_smoke_report.py`, py_compile for touched files,
  `git diff --check`, and an added-line silent-handling scan.
- VPS5 pulled from `f576ad35` to `462afac7` without bot restart because the
  deployed change was read-only smoke-report tooling. The five configured bots
  were left running.
- The first post-deploy 5-minute smoke surfaced one unrelated GateIO
  `cycle.degraded` hard event from a transient `RequestTimeout`. Event-query
  evidence showed the failed `authoritative_balance` fetch in cycle `cy_2043`
  recovered in cycle `cy_2044` with balance, positions, and open-orders all
  succeeding. A settled 1-minute smoke then reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `462afac7`, all five
  configured bots matched, no account-critical remote failures, no text-log
  hard or attention matches, and the new `fill_refresh` brief section populated
  with four succeeded and zero failed refreshes.
- Repository pulled through PR #940 at `f576ad35`.
- PR #940 made the default planned incident-bundle output path generated by
  `live-restart-smoke-plan` timestamped under `/tmp`, while preserving the
  explicit `--incident-bundle-output` override. This prevents repeated planner
  runs from pointing at the same default tarball path. The slice did not change
  restart execution, signal behavior, process discovery, smoke-report
  semantics, exchange calls, event producers, cache mutation, readiness gates,
  console routing, order logic, risk logic, or trading behavior.
- PR #940 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan.
- VPS5 pulled from `e3c616d4` to `f576ad35` without bot restart because the
  deployed change was read-only planner output tooling. The five configured
  bots were left running.
- A VPS5 `live-restart-smoke-plan --summary --compact` run against
  `/root/bots_vps5.yaml` completed with `ok=true`, five configured bots, six
  phase names, `execute=false`, zero issues, and a timestamped planned
  incident-bundle output path such as
  `/tmp/passivbot_incident_bundle_restart_smoke_20260701_090751_232703.tar.gz`.
- The first 5-minute smoke after deploy surfaced one unrelated Hyperliquid
  `fills.refresh_summary` hard event with `RequestTimeout`. Follow-up event
  queries showed later fill refreshes succeeding, and settled 2-minute and
  5-minute smokes reported `ok=true`, `hard_failures=0`, clean tracked
  repository state at `f576ad35`, all five configured bots matched, no
  missing/extra/duplicate live processes, no failed account-critical remote
  calls, and no text-log hard or attention matches. Remaining attention stayed
  in known non-hard EMA readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #939 at `e3c616d4`.
- PR #939 added a read-only `live-restart-smoke-plan --summary` projection for
  repeated operator loops. The projection keeps bot count/names, phase names,
  planned smoke and incident-bundle commands, execution policy, signal-safety
  strategy, escalation command counts, and warning/issue counts without dumping
  every per-bot phase. The slice did not alter full restart plan generation,
  planned commands, smoke/incident-bundle behavior, execution policy, process
  signaling, tmux/SSH/git behavior, event producers, exchange calls, cache
  mutation, readiness gates, console routing, order logic, risk logic, or
  trading behavior.
- PR #939 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan.
- VPS5 pulled from `8ffb9a73` to `e3c616d4` without bot restart because the
  deployed change was read-only planner projection tooling. The five
  configured bots were left running.
- A VPS5 `live-restart-smoke-plan --summary` run against
  `/root/bots_vps5.yaml` completed with `ok=true`, five configured bots, six
  phase names, `execute=false` for both planned commands, zero issues, and
  bounded smoke/incident-bundle commands carrying `--event-tail-lines 2000`,
  `--max-event-files-per-bot 2`, `--max-log-files 8`,
  `--log-tail-lines 1200`, `--max-log-matches 20`, and `--compact`.
- A bounded 5-minute VPS5 smoke after the pull reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `e3c616d4`, five
  matched expected live processes, no missing/extra/duplicate live processes,
  no failed account-critical remote calls, `logs.max_files=8`,
  `logs.tail_lines=1200`, and `logs.max_matches=20`. Remaining attention came
  from known non-hard EMA readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #938 at `8ffb9a73`.
- PR #938 made the read-only `live-restart-smoke-plan` output include a
  bounded `live-incident-bundle` evidence command for non-clean restart/smoke
  cases. The planned command reuses the same recent-window, event-tail,
  per-bot event-file, log-file, log-tail, and log-match bounds as the planned
  smoke command, defaults to `--no-event-segments` and `--compact`, and remains
  plan-only with `execute=false`. The slice did not execute bundles, send
  signals, invoke tmux, run SSH, pull git, start bots, contact exchanges, load
  credentials, add event producers, mutate caches, change readiness gates,
  route console output, or alter order/risk/trading behavior.
- PR #938 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan.
- VPS5 pulled from `6b38c7dd` to `8ffb9a73` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan` run against `/root/bots_vps5.yaml` completed
  with `ok=true`, five configured bots, `execute=false`, and a new
  `post_failure_incident_bundle` phase. The planned incident-bundle command
  contained `--supervisor-config /root/bots_vps5.yaml`, `--recent-minutes 5`,
  `--no-event-segments`, `--event-tail-lines 2000`,
  `--max-event-files-per-bot 2`, `--max-log-files 8`,
  `--log-tail-lines 1200`, `--max-log-matches 20`, and `--compact`.
- A bounded 5-minute VPS5 smoke after the pull reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `8ffb9a73`, five
  matched expected live processes, no missing/extra/duplicate live processes,
  no failed account-critical remote calls, `logs.max_files=8`,
  `logs.tail_lines=1200`, and `logs.max_matches=20`. Remaining attention came
  from known non-hard EMA readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #937 at `6b38c7dd`.
- PR #937 made the read-only `live-incident-bundle` compact result expose the
  embedded smoke report's bounded text-log scan summary under
  `smoke_report.logs`, and added the log scan bounds/policy to the incident
  bundle manifest filters. The slice projected existing smoke metadata only;
  it did not change log/event scan behavior, smoke verdict policy, bundle
  file selection beyond manifest/result metadata, event producers, exchange
  calls, cache mutation, readiness gates, console routing, order logic, risk
  logic, or trading behavior.
- PR #937 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan.
- VPS5 pulled from `0da21e9c` to `6b38c7dd` without bot restart because the
  deployed change was read-only incident-bundle reporting. The five configured
  bots were left running.
- A bounded 5-minute VPS5 smoke after the pull reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `6b38c7dd`, five
  matched expected live processes, no missing/extra/duplicate live processes,
  no failed account-critical remote calls, `logs.max_files=8`,
  `logs.tail_lines=1200`, and `logs.max_matches=20`. Remaining attention came
  from known non-hard EMA readiness and HSL status/cooldown diagnostics.
- A focused VPS5 incident-bundle smoke wrote
  `/tmp/passivbot_incident_bundle_smoke_937.tar.gz` with `--no-event-segments`
  and the same bounded event/log scan arguments; its compact output reported
  `ok=true`, `hard_failures=0`, five expected live processes, and the new
  `smoke_report.logs` projection with `max_files=8`, `tail_lines=1200`,
  `max_matches=20`, `files_scanned=8`, and no hard or attention text-log
  matches.
- Repository pulled through PR #936 at `0da21e9c`.
- PR #936 made the read-only `live-restart-smoke-plan` generated smoke command
  include `--max-log-files 8` by default, with `0` as an escape hatch to omit
  that explicit smoke-report override. This completed the generated
  smoke-command log bound set alongside `--log-tail-lines 1200` and
  `--max-log-matches 20`. The slice was plan-only operator tooling and did not
  execute restarts, send signals, invoke tmux, run SSH, pull git, start bots,
  contact exchanges, load credentials, add event producers, mutate caches,
  change readiness gates, write monitor events, route console output, or alter
  order/risk/trading behavior.
- PR #936 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan.
- VPS5 pulled from `94f88dba` to `0da21e9c` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan` run against `/root/bots_vps5.yaml` completed
  with `ok=true`, five configured bots, `execute=false`, and planned smoke
  commands containing `--event-tail-lines 2000`,
  `--max-event-files-per-bot 2`, `--max-log-files 8`,
  `--log-tail-lines 1200`, `--max-log-matches 20`, `--brief`, and
  `--compact`. The planned command was then run as a bounded 5-minute brief
  smoke; it reported `ok=true`, `hard_failures=0`, clean tracked repository
  state at `0da21e9c`, five matched expected live processes, no
  missing/extra/duplicate live processes, no failed account-critical remote
  calls, `logs.max_files=8`, `logs.tail_lines=1200`, and
  `logs.max_matches=20`. Remaining attention came from known non-hard EMA
  readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #935 at `94f88dba`.
- PR #935 made `live-smoke-report` full, summary, and brief `logs` output
  expose the configured text-log scan bounds: `max_files`, `tail_lines`, and
  `max_matches`. The slice was read-only smoke-report projection tooling and
  did not change log file discovery, line scanning, match classification,
  smoke verdict policy, event routing, exchange access, cache/state behavior,
  or order/risk/trading behavior.
- PR #935 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_smoke_report.py`, py_compile for touched files,
  `git diff --check`, and a diff-only silent-handling scan. A broad
  touched-file silent scan still reports pre-existing patterns in the large
  smoke-report module, but the diff added no new forbidden default-get or
  silent-exception patterns.
- VPS5 pulled from `3e1335d` to `94f88dba` without bot restart because the
  deployed change was read-only smoke-report tooling. The five configured bots
  were left running.
- A bounded 5-minute brief smoke at `94f88dba` completed with `ok=true`,
  `hard_failures=0`, clean tracked repository state, five matched expected
  live processes, no missing/extra/duplicate live processes, and no failed
  account-critical remote calls. The new log-bound metadata was present in
  brief output: `logs.max_files=8`, `logs.tail_lines=1200`, and
  `logs.max_matches=20`. Remaining attention came from known non-hard EMA
  readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #934 at `3e1335d`.
- PR #934 made the read-only `live-restart-smoke-plan` generated smoke command
  include `--log-tail-lines 1200` and `--max-log-matches 20` by default, with
  `0` escape hatches to omit those explicit smoke-report overrides. The slice
  was plan-only operator tooling and did not execute restarts, send signals,
  invoke tmux, run SSH, pull git, start bots, contact exchanges, load
  credentials, add event producers, mutate caches, change readiness gates,
  write monitor events, route console output, or alter order/risk/trading
  behavior.
- PR #934 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `0fd30b6b` to `3e1335d` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan` run against `/root/bots_vps5.yaml` completed
  with `ok=true`, five configured bots, `execute=false`, and planned smoke
  commands containing `--event-tail-lines 2000`,
  `--max-event-files-per-bot 2`, `--log-tail-lines 1200`,
  `--max-log-matches 20`, `--brief`, and `--compact`. The planned command was
  then run as a bounded 5-minute brief smoke; it reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `3e1335d`, five matched
  expected live processes, no missing/extra/duplicate live processes, no failed
  account-critical remote calls, `event_tail_limited_files=6`,
  `event_files_skipped_by_limit=0`, `logs.files_scanned=8`, and
  `logs.window.lines_considered=34`. Remaining attention came from known
  non-hard EMA readiness and HSL status/cooldown diagnostics.
- Repository pulled through PR #933 at `0fd30b6b`.
- PR #933 made the read-only `live-restart-smoke-plan` generated smoke command
  include `--event-tail-lines 2000` and `--max-event-files-per-bot 2` by
  default, with `0` escape hatches for full event validation. The slice was
  plan-only operator tooling and did not execute restarts, send signals, invoke
  tmux, run SSH, pull git, start bots, contact exchanges, load credentials, add
  event producers, mutate caches, change readiness gates, write monitor events,
  route console output, or alter order/risk/trading behavior.
- PR #933 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_restart_smoke_plan.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `908c955e` to `0fd30b6b` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan` run against `/root/bots_vps5.yaml` completed
  with `ok=true`, five configured bots, `execute=false`, and planned smoke
  commands containing `--event-tail-lines 2000`,
  `--max-event-files-per-bot 2`, `--brief`, and `--compact`. The planned
  command was then run as a 5-minute brief smoke; it reported `ok=true`,
  `hard_failures=0`, clean tracked repository state at `0fd30b6b`, five
  matched expected live processes, no missing/extra/duplicate live processes,
  no failed account-critical remote calls, `event_tail_limited_files=6`, and
  `event_files_skipped_by_limit=0`. Remaining attention came from known
  non-hard EMA readiness, HSL status/cooldown, and staged-readiness diagnostics.
- Repository pulled through PR #932 at `908c955e`.
- PR #932 made the read-only `live-restart-smoke-plan` generated smoke command
  default to `live-smoke-report --brief --compact`, with explicit
  `--summary-smoke-report` and `--full-smoke-report` overrides. The slice was
  plan-only operator tooling and did not execute restarts, send signals, invoke
  tmux, run SSH, pull git, start bots, contact exchanges, load credentials, add
  event producers, mutate caches, change readiness gates, write monitor events,
  route console output, or alter order/risk/trading behavior.
- PR #932 passed Hermes + Claude + Cursor + CI after a reviewer-requested CLI
  boolean cleanup. Local validation covered `tests/test_live_restart_smoke_plan.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `28b0687d` to `908c955e` without bot restart because the
  deployed change was read-only planner tooling. The five configured bots were
  left running.
- A VPS5 `live-restart-smoke-plan` run against `/root/bots_vps5.yaml` completed
  with `ok=true`, five configured bots, `execute=false`, and planned smoke
  commands containing `--brief --compact`. The planned command was then run as
  a 5-minute brief smoke; it reported `ok=true`, `hard_failures=0`, clean
  tracked repository state at `908c955e`, five matched expected live processes,
  no missing/extra/duplicate live processes, and no failed account-critical
  remote calls. Remaining attention came from known non-hard EMA readiness,
  HSL status, and staged-readiness diagnostics.
- Repository pulled through PR #931 at `28b0687d`.
- PR #931 capped incident-bundle raw event-segment fallback copying with
  `--max-event-files-per-bot`, while preserving exact matched report segment
  paths. It exposes event-segment selection and limit metadata in compact
  output and bundled manifests. The slice was read-only incident-bundle tooling
  and did not add event producers, exchange calls, cache mutation, readiness
  gates, console routing, monitor writes, order/risk logic, or trading
  behavior.
- PR #931 passed Hermes + Claude + Cursor + CI. A reviewer-requested regression
  test now proves active file caps do not prune matched incident evidence.
  Local validation covered `tests/test_live_incident_bundle.py`, py_compile for
  touched files, `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `103a4d11` to `28b0687d` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A targeted all-rotated incident-bundle smoke forced the fallback segment-copy
  path with `--event-type no.such.event`, `--max-event-files-per-bot=2`, and
  `--event-tail-lines=200`. It completed with `ok=true`,
  `hard_failures=0`, `event_segments.selection=fallback_discovered_paths`,
  `event_segments.files=12`, `included=12`, `event_files_before_limit=941`,
  and `event_files_skipped_by_limit=929`. A standard bounded 5-minute smoke at
  `28b0687d` also completed with `ok=true`, `hard_failures=0`, clean tracked
  repository state, all five live bot processes running, and no failed
  account-critical remote calls. Remaining attention came from known non-hard
  EMA readiness, HSL cooldown, and candle coverage diagnostics.
- Repository pulled through PR #930 at `103a4d11`.
- PR #930 added `live-smoke-report --max-event-files-per-bot` and the matching
  `build_live_smoke_report()` option for fair, opt-in per-events-directory caps
  in smoke-report event scanning. The slice was read-only smoke/report tooling
  and did not add event producers, exchange calls, cache mutation, readiness
  gates, console routing, monitor writes, order/risk logic, or trading
  behavior.
- PR #930 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_smoke_report.py`, `tests/test_live_incident_bundle.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `109c4342` to `103a4d11` without bot restart because the
  deployed change was read-only smoke/incident tooling. The five configured bots
  were left running.
- A bounded 5-minute smoke at `103a4d11` completed with `ok=true`,
  `hard_failures=0`, clean tracked repository state, all five live bot
  processes running, and no failed account-critical remote calls. A bounded
  all-rotated incident-bundle smoke with `--max-event-files-per-bot=2`
  completed successfully with `event_report.files_scanned=12`,
  `event_report.event_window.event_files_before_limit=945`,
  `event_report.event_window.event_files_skipped_by_limit=933`,
  `time_window.files_scanned=12`, and the embedded smoke report carrying the
  same per-bot file cap metadata. This verified the embedded smoke-report gap
  from PR #929, but event-segment copying still needs the same cap on its
  fallback-discovered path when no report has matched exact segment paths.
- Repository pulled through PR #929 at `109c4342`.
- PR #929 added `live-incident-bundle --max-event-files-per-bot` and the
  matching `build_live_incident_bundle()` option for fair, opt-in
  per-events-directory caps in the embedded event report, problem-event
  report, and time-window report. It reuses the shared event-query cap helper,
  reports limit metadata in compact output and bundled JSON, and records the
  cap in manifest filters. The slice was read-only incident-bundle/event-query
  tooling and did not add event producers, exchange calls, cache mutation,
  readiness gates, console routing, monitor writes, order/risk logic, or
  trading behavior.
- PR #929 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `98e653ba` to `109c4342` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A bounded 5-minute smoke at `109c4342` completed with `ok=true`,
  `hard_failures=0`, clean tracked repository state, all five live bot
  processes running, and no failed account-critical remote calls. A first
  all-rotated incident-bundle smoke with `--max-event-files-per-bot=2` was
  interrupted after exposing a remaining slow path: the embedded smoke report
  still received `include_rotated=true` without a file-count cap. The follow-up
  branch `codex/v8-smoke-report-per-bot-file-limit` addresses that embedded
  smoke-report gap.
- Repository pulled through PR #928 at `98e653ba`.
- PR #928 added `live-event-query --max-event-files-per-bot` and the matching
  `build_event_report()` option for fair, opt-in per-events-directory rotated
  file caps. It shares the same current-first/newest-mtime helper used by
  `live-performance-report`, reports `event_file_limit_scope=per_bot`,
  group count, before/after file counts, skipped file count, and selection
  order in `event_window`, and keeps default scans unchanged. The slice was
  read-only event-query/performance-report tooling and did not add event
  producers, exchange calls, cache mutation, readiness gates, console routing,
  monitor writes, order/risk logic, or trading behavior.
- PR #928 passed Hermes + Claude + Cursor + CI on the amended head after a
  reviewer-requested de-duplication cleanup. Local validation covered
  `tests/test_live_event_query.py`, `tests/test_live_performance_report.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `1ff596d1` to `98e653ba` without bot restart because the
  deployed change was read-only query/report tooling. The five configured bots
  were left running.
- A bounded all-rotated event-query probe at `98e653ba` completed with
  `ok=true`, `include_rotated=true`, `max_event_files_per_bot=2`,
  `event_file_limit_scope=per_bot`, `event_file_limit_groups=6`,
  `event_files_before_limit=942`, `event_files_skipped_by_limit=930`, and
  `files_scanned=12`. A bounded 5-minute smoke completed with `ok=true`,
  `hard_failures=0`, clean tracked repository state, all five live bot
  processes running, and no failed account-critical remote calls. Attention
  remained from existing non-hard EMA-readiness and HSL-cooldown problem
  events. This verified fair per-bot event-query bounding, but
  `live-incident-bundle` still needs to expose the same cap for its embedded
  event-query reports.
- Repository pulled through PR #927 at `1ff596d1`.
- PR #927 added `live-performance-report --max-event-files-per-bot`, an
  opt-in fair event-segment scan bound for multi-bot monitor roots. It keeps
  the existing global `--max-event-files` mode, makes the two modes mutually
  exclusive, groups by event directory, prefers `current.ndjson` first and then
  newest rotated segments by mtime per group, and reports
  `event_file_limit_scope=per_bot`, group count, before/after file counts, and
  skipped file count in `event_window`. The slice was read-only report tooling
  and did not add event producers, exchange calls, cache mutation, readiness
  gates, console routing, monitor writes, order/risk logic, or trading
  behavior.
- PR #927 passed Cursor + Hermes + CI. Claude did not comment after repeated
  polling, so it was merged under the documented low-risk degraded gate, with
  the rationale recorded on the PR. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `ec9f47fe` to `1ff596d1` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A bounded all-rotated performance report at `1ff596d1` completed in about
  12 seconds with `ok=true`, `include_rotated=true`,
  `max_event_files_per_bot=2`, `event_file_limit_scope=per_bot`,
  `event_file_limit_groups=6`, `event_files_before_limit=947`,
  `event_files_skipped_by_limit=935`, and `files_scanned=12`. A bounded
  5-minute smoke completed with `ok=true`, `hard_failures=0`, clean tracked
  repository state, all five live bot processes running, and no failed
  account-critical remote calls. Attention remained from existing non-hard
  EMA-readiness and HSL-cooldown problem events plus one non-hard OKX candle
  fetch timeout that later recovered. This verified fair per-bot performance
  report bounding, but `live-event-query` still lacks the same per-bot file cap
  for bounded rotated incident/debug queries.
- Repository pulled through PR #926 at `ec9f47fe`.
- PR #926 added `live-performance-report --max-event-files`, an opt-in global
  event-segment scan bound for smoke/debug reports. It prefers
  `current.ndjson` files first, then newest rotated segments by mtime, and
  reports `event_file_limit_scope=global`, before/after file counts, skipped
  file count, and selection order in `event_window`. The slice was read-only
  report tooling and did not add event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior.
- PR #926 passed Cursor + Hermes + CI on the amended head. Claude did not
  comment after repeated polling, so it was merged under the documented
  low-risk degraded gate, with the rationale recorded on the PR. Local
  validation covered `tests/test_live_performance_report.py`, py_compile for
  touched files, `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `dc0ad4dd` to `ec9f47fe` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A bounded 5-minute smoke at `ec9f47fe` completed with `ok=true`,
  `hard_failures=0`, clean tracked repository state, all five live bot
  processes running, no failed remote calls, and no failed account-critical
  remote calls. Attention remained from existing EMA-readiness and HSL-cooldown
  problem events. A bounded all-rotated performance report completed in about
  13 seconds with `ok=true`, `include_rotated=true`, `max_event_files=12`,
  `event_files_before_limit=944`, `event_files_skipped_by_limit=932`,
  `files_scanned=12`, and `event_tail_methods` showing both `seek_tail` and
  `sequential_gzip_tail`. This verified the file-count bound works, but the
  global cap can still drop whole bots if the cap is too low; the next slice
  should add a per-bot fair file cap for fleet-wide smoke/debug reports.
- Repository pulled through PR #925 at `dc0ad4dd`.
- PR #925 added `live-performance-report --event-tail-lines`, an opt-in
  per-segment row bound for smoke/debug reports. Plain current segments can seek
  from the file end; compressed rotated segments may still require sequential
  decompression. The slice was read-only report tooling and did not add event
  producers, exchange calls, cache mutation, readiness gates, console routing,
  monitor writes, order/risk logic, or trading behavior.
- PR #925 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `652b019e` to `dc0ad4dd` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A bounded current-segment performance report at `dc0ad4dd` completed with
  `ok=true`, `include_rotated=false`, `event_tail_lines=500`,
  `event_tail_limited_files=6`, `event_tail_methods.seek_tail=6`,
  `records_total=2389`, and `files_scanned=6`. A 5-minute smoke rerun completed
  with `ok=true`, `hard_failures=0`, clean tracked repository state, all five
  configured bots matched, no failed remote calls, and no failed
  account-critical remote calls. An attempted all-rotated performance report
  remained too slow because many compressed rotated segments still had to be
  opened; the next slice should add an explicit newest-segment file-count bound
  for smoke/debug reports.
- Repository pulled through PR #924 at `652b019e`.
- PR #924 added aggregate startup phase timing counters to
  `live-performance-report` `startup_readiness`, exposing bounded phase counts
  plus elapsed and since-previous timing summaries from existing
  `bot.startup_timing` events. It also routes startup phase labels through a
  whitelist before they appear in `startup_readiness` or
  `operation_durations`. The slice was read-only report tooling and did not add
  event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, order/risk logic, or trading behavior.
- PR #924 passed Hermes + Claude + Cursor + CI after fixing the
  `operation_durations` phase-label leak found by Hermes. Local validation
  covered `tests/test_live_performance_report.py`, py_compile for touched
  files, `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `31ef4d40` to `652b019e` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A 5-minute smoke at `652b019e` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked
  repository state, no failed remote calls, and no failed account-critical
  remote calls. Current monitor segments had no startup timing events, so
  `startup_readiness` showed the new aggregate keys as empty. An attempted
  full rotated `live-performance-report --include-rotated` verification was
  interrupted after it proved too slow for smoke use, which motivated the next
  bounded performance-report scan slice.
- Repository pulled through PR #923 at `31ef4d40`.
- PR #923 added aggregate market snapshot staleness counters to
  `live-performance-report` `input_staleness`, exposing snapshot observations,
  count/symbol/missing summaries, max/mean/configured age summaries,
  configured-age excess, missing-symbol totals, and bounded source labels from
  existing `snapshot.built` events. The slice was read-only report tooling and
  did not add event producers, exchange calls, cache mutation, readiness gates,
  console routing, monitor writes, order/risk logic, or trading behavior.
- PR #923 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `82b1990c` to `31ef4d40` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A 5-minute smoke at `31ef4d40` completed with `ok=true`, `hard_failures=0`,
  all five configured bots matched, clean tracked repository state, no failed
  remote calls, and no failed account-critical remote calls. A VPS5
  `live-performance-report --summary` verified
  `input_staleness.market_snapshot` in live output with 67 observations, max
  age 1128 ms, no missing symbols, and source labels
  `fetch_tickers`/`hyperliquid_hip3_asset_ctx`.
- Repository pulled through PR #922 at `82b1990c`.
- PR #922 added aggregate HSL replay stage/status counters to
  `live-performance-report` `hsl_replay_profile`, exposing
  active/completed/failed replay bot counts and active/latest replay stage
  counts from existing sanitized `hsl.replay.*` events. The slice was read-only
  report tooling and did not add event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior.
- PR #922 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_performance_report.py`, py_compile for touched files,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `5808f679` to `82b1990c` without bot restart because the
  deployed change was read-only report tooling. The five configured bots were
  left running.
- A 5-minute smoke at `82b1990c` completed with `ok=true`, `hard_failures=0`,
  all five configured bots matched, clean tracked repository state, no failed
  remote calls, and no failed account-critical remote calls. A VPS5
  `live-performance-report --summary` verified the new HSL replay counters in
  live output, including `active_bot_count=1`,
  `active_stage_counts.price_history_symbol_fetch_started=1`, and
  `latest_status_counts.active=1` for the existing Kucoin replay evidence.
- Repository pulled through PR #921 at `5808f679`.
- PR #921 projected `problem_event_report.file_discovery` and
  `problem_event_report.event_window` into compact incident-bundle output so
  scoped problem-event discovery/window metadata can be verified without
  opening the tarball. The slice was read-only incident-bundle tooling and did
  not add event producers, exchange calls, live execution, report verdict
  changes, console routing, order/risk logic, or trading behavior.
- PR #921 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `570875ab` to `5808f679` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A 5-minute smoke at `5808f679` completed with `ok=true`, `hard_failures=0`,
  all five configured bots matched, clean tracked repository state, no failed
  remote calls, and no failed account-critical remote calls. A focused OKX
  incident-bundle smoke verified compact output with
  `problem_event_report.file_discovery.bot_path_pruning_applied=true`,
  `problem_event_report.files_scanned=1`, `scope_pruned=5`, and
  `problem_event_report.event_window` populated from the recent-window query.
- Repository pulled through PR #920 at `570875ab`.
- PR #920 projected `time_window.files_scanned` and
  `time_window.file_discovery` into compact incident-bundle output so focused
  recent-window bundle scope can be verified without opening the tarball. The
  slice was read-only incident-bundle tooling and did not add event producers,
  exchange calls, live execution, report verdict changes, console routing,
  order/risk logic, or trading behavior.
- PR #920 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  py_compile for touched files, and `git diff --check`.
- VPS5 pulled from `989b81c9` to `570875ab` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A 5-minute smoke at `570875ab` completed with `ok=true`, `hard_failures=0`,
  all five configured bots matched, clean tracked repository state, no failed
  remote calls, and no failed account-critical remote calls. A focused OKX
  incident-bundle smoke verified compact output with
  `time_window.files_scanned=1`, path-pruned discovery metadata, and one
  included event segment for `monitor/okx/okx_faisal/events/current.ndjson`.
- Repository pulled through PR #919 at `989b81c9`.
- PR #919 applied incident-bundle query scope filters to
  `time_window_report.json`, `timeline.txt`, and matched event-segment
  selection, using one shared event-query predicate across event reports and
  time-window reports. The slice was read-only incident-bundle/event-query
  tooling and did not add event producers, exchange calls, live execution,
  report verdict changes, console routing, order/risk logic, or trading
  behavior.
- PR #919 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `946d0757` to `989b81c9` without bot restart because the
  deployed change was read-only incident-bundle/event-query tooling. The five
  configured bots were left running.
- A 5-minute smoke at `989b81c9` using `--event-tail-lines 1000`,
  `--processes`, and `/root/bots_vps5.yaml` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked repository
  state, no failed remote calls, and no failed account-critical remote calls.
  A focused OKX incident-bundle smoke verified the time-window scope fix:
  `time_window_report.json` had `time_window_matched=9`, paths only under
  `monitor/okx/okx_faisal/events/current.ndjson`, exchange/user only
  `okx/okx_faisal`, `event_segments.files=1`, and `timeline_has_binance=false`.
- Repository pulled through PR #918 at `946d0757`.
- PR #918 added incident-bundle query scope filters for level, exchange, user,
  bot id, remote-call group, side, source, component, tag, and data equality.
  The slice was read-only incident-bundle tooling and did not add event
  producers, exchange calls, live execution, report verdict changes, console
  routing, order/risk logic, or trading behavior.
- PR #918 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `29d026a` to `946d0757` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A 5-minute smoke at `946d0757` using `--event-tail-lines 1000`,
  `--processes`, and `/root/bots_vps5.yaml` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked repository
  state, no failed remote calls, and no failed account-critical remote calls.
  A focused OKX incident-bundle smoke verified the new query filters against
  live monitor files: `event_report.file_discovery.bot_path_pruning_applied`
  was `true`, `scope_pruned=5`, `event_report.query_matched_events=2`, and
  `problem_event_report.matched_events=2`.
- Repository pulled through PR #917 at `29d026a`.
- PR #917 embedded `problem_event_report.json` in incident bundles by default,
  using the same shared structured problem-event predicate as
  `live-smoke-report` and `live-event-query --problem-events`. The slice was
  read-only incident-bundle tooling and did not add event producers, exchange
  calls, live execution, report verdict changes, console routing, order/risk
  logic, or trading behavior.
- PR #917 passed Hermes + Claude + Cursor + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `0f32aeff` to `29d026a` without bot restart because the
  deployed change was read-only incident-bundle tooling. The five configured
  bots were left running.
- A 5-minute smoke at `29d026a` using `--event-tail-lines 1000`,
  `--processes`, and `/root/bots_vps5.yaml` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked repository
  state, no failed remote calls, and no failed account-critical remote calls.
  A bounded incident-bundle smoke over the same live monitor tree completed
  with `ok=true`, `hard_failures=0`, `problem_event_report.enabled=true`,
  `problem_event_report.matched_events=45`, zero copied event-segment bytes
  because `--no-event-segments` was used, and verified
  `problem_event_report.json` in the archive.
- Repository pulled through PR #916 at `0f32aeff`.
- PR #916 added `live-event-query --problem-events` and
  `--hard-problem-events`, using the same shared structured problem-event
  predicate as `live-smoke-report`. The slice was read-only event-query tooling
  and did not add event producers, exchange calls, live execution, report
  verdict changes, console routing, order/risk logic, or trading behavior.
- PR #916 used the documented degraded review gate: Claude remained absent
  across repeated poll cycles, while Hermes approved the amended head, CI was
  green, and the merge was clean. Local validation covered
  `tests/test_live_event_query.py`, `tests/test_live_smoke_report.py`,
  py_compile for touched files and tests, `git diff --check`, and a
  touched-file silent-handling scan.
- VPS5 pulled from `aef82af9` to `0f32aeff` without bot restart because the
  deployed change was read-only event-query tooling. The five configured bots
  were left running.
- A 5-minute smoke at `0f32aeff` using `--event-tail-lines 1000`,
  `--processes`, and `/root/bots_vps5.yaml` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked repository
  state, no failed remote calls, and no failed account-critical remote calls.
  A focused `live-event-query --problem-events --trace-summary` over the same
  recent window matched the same known non-hard EMA readiness and HSL status
  attention groups shown by smoke.
- Repository pulled through PR #915 at `aef82af9`.
- PR #915 projected bounded, value-safe `problem_events.groups` and
  `event_types` into `live-smoke-report --brief`, with
  `event_types_truncated` and `groups_truncated` flags. The slice was read-only
  smoke-report tooling and did not add event producers, exchange calls, live
  execution, report verdict changes, console routing, order/risk logic, or
  trading behavior.
- PR #915 used the documented degraded review gate: Claude remained absent
  across repeated poll cycles, while Hermes approved the amended head, CI was
  green, and the merge was clean. Local validation covered
  `tests/test_live_smoke_report.py`, py_compile for touched files and tests,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `9ff335e4` to `aef82af9` without bot restart because the
  deployed change was read-only smoke-report tooling. The five configured bots
  were left running.
- A 5-minute brief smoke at `aef82af9` using `--event-tail-lines 1000`,
  `--processes`, and `/root/bots_vps5.yaml` completed with `ok=true`,
  `hard_failures=0`, all five configured bots matched, clean tracked repository
  state, no failed remote calls, and no failed account-critical remote calls.
  The new brief projection made the remaining `attention=true` immediately
  attributable to bounded non-hard `problem_events.groups`: EMA readiness
  groups plus HSL cooldown status groups. A focused `live-event-query` for
  recent `hsl.status` events confirmed the HSL attention came from expected
  cooldown-active and green status events, not a hard deploy failure.
- Repository pulled through PR #914 at `9ff335e4`.
- PR #914 moved incident-bundle event-segment SHA-256 calculation behind actual
  segment inclusion. Disabled event segments and byte-budget-skipped segments no
  longer hash large monitor event files just to write an excluded manifest
  entry. The slice was read-only incident-bundle/report tooling and did not add
  event producers, exchange calls, live execution, report verdict changes,
  console routing, order/risk logic, or trading behavior.
- PR #914 passed Claude + Hermes + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files and tests,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `d3f3264c` to `9ff335e4` without bot restart because the
  deployed change was read-only incident-bundle/report tooling. The five
  configured bots were left running.
- A bounded incident-bundle smoke at `9ff335e4` using `--recent-minutes 2`,
  `--event-tail-lines 1000`, `--no-event-segments`, and `--no-trace-report`
  completed in `9.77s` with `ok=true`, `hard_failures=0`,
  `monitor_snapshots=12`, six seek-tailed current segments, roughly `16.18MB`
  skipped by byte, `matched_events=652`, and `event_segments.included=0`.
- A fresh brief smoke at `9ff335e4` completed in `4.33s` with `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, no
  failed remote calls, and no failed account-critical remote calls. It still
  returned `attention=true` from structured non-hard problem events, including
  EMA-readiness and event-pipeline signals, but brief mode exposed only the
  problem-event counts. The next useful reporting slice is to project bounded,
  value-safe problem-event groups into brief output so smoke-loop attention is
  immediately actionable.
- Repository pulled through PR #913 at `d3f3264c`.
- PR #913 added opt-in `--event-tail-lines` support to incident bundles and
  moved monitor event row tailing into a shared helper used by
  `live-event-query`, `live-smoke-report`, and the incident-bundle time-window
  report. Plain NDJSON current segments seek from file end; compressed segments
  remain sequential. The slice was read-only report/query tooling and did not
  add event producers, exchange calls, live execution, report verdict changes,
  console routing, order/risk logic, or trading behavior.
- PR #913 passed Claude + Hermes + CI on the amended head. Local validation
  covered `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`,
  and `tests/test_live_smoke_report.py`, py_compile for touched files and tests,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `0eb29545` to `d3f3264c` without bot restart because the
  deployed change was read-only report/query tooling. The five configured bots
  were left running.
- A first full incident-bundle smoke was manually interrupted after roughly
  three minutes, then component timing localized the remaining cost away from
  event-row parsing: `live-event-query --recent-minutes 2 --event-tail-lines
  1000` completed in `5.12s`, and `live-smoke-report --brief
  --recent-minutes 2 --event-tail-lines 1000` completed in `5.58s`. Both
  reported `event_tail_methods={"seek_tail": ...}` and skipped roughly `24.8MB`
  by byte with no hard failures.
- A retried bounded incident-bundle smoke at `d3f3264c` using
  `--recent-minutes 2`, `--event-tail-lines 1000`, `--no-event-segments`, and
  `--no-trace-report` completed in `15.51s` with `ok=true`, `hard_failures=0`,
  `monitor_snapshots=12`, `event_report.files_scanned=6`, six seek-tailed
  current segments, roughly `26MB` skipped by byte, `matched_events=953`, and
  `event_segments.included=0`. This exposed a follow-up optimization: the
  disabled event-segment manifest still hashes matched segment files even when
  segment copying is disabled.
- Repository pulled through PR #912 at `0eb29545`.
- PR #912 added `--recent-minutes` to `passivbot tool live-incident-bundle`,
  resolving it to the existing `since_ms` event/log time-window contract. The
  slice was read-only incident-bundle/report tooling and did not add event
  producers, exchange calls, live execution, report verdict changes, console
  routing, order/risk logic, or trading behavior.
- PR #912 passed Claude + Hermes + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, py_compile for touched files and tests,
  `git diff --check`, and a touched-file silent-handling scan. The optional
  passivbot CLI dispatch test was not collected because the local Rust
  extension freshness guard tripped on import.
- VPS5 pulled from `7c5c96f4` to `0eb29545` without bot restart because the
  deployed change was read-only incident-bundle/report tooling and docs. The
  five configured bots were left running.
- A bounded incident-bundle smoke at `0eb29545` using `--recent-minutes 2`,
  `--no-event-segments`, and `--no-trace-report` reported `ok=true`,
  `hard_failures=0`, `time_window.enabled=true`, `matched_events=906`,
  `events_truncated=true`, `event_report.files_scanned=6`, and matching
  discovery counts in the event report and segment manifest
  (`candidate_files=3750`, `event_segments=950`, `rotated_skipped=944`,
  `scope_pruned=0`). A fresh 2-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- Repository pulled through PR #911 at `7c5c96f4`.
- PR #911 projected bounded event-file discovery metadata into incident-bundle
  event-report summaries and event-segment manifests, using the same
  metadata-returning discovery helper as the query, smoke, and performance
  report tools. The slice does not add event producers, exchange calls, live
  execution, report verdict changes, console routing, order/risk logic, or
  trading behavior.
- PR #911 passed Claude + Hermes + CI. Local validation covered
  `tests/test_live_incident_bundle.py`, `tests/test_live_event_query.py`, and
  `tests/test_live_smoke_report.py`, py_compile for touched files and tests,
  `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `ac949f03` to `7c5c96f4` without bot restart because the
  deployed change was read-only incident-bundle/report tooling and docs. The
  five configured bots were left running.
- A bounded incident-bundle smoke at `7c5c96f4` with event segments disabled
  reported `ok=true`, `hard_failures=0`, `event_report.files_scanned=6`, and
  matching `file_discovery` blocks in `event_report` and `event_segments`
  (`candidate_files=3751`, `event_segments=952`, `rotated_skipped=946`,
  `scope_pruned=0`). A fresh 2-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- Repository pulled through PR #910 at `ac949f03`.
- PR #910 projected bounded event-file discovery metadata into
  `live-performance-report` full and summary output, using the same public
  metadata-returning discovery helper as `live-event-query` and
  `live-smoke-report`. The slice does not add event producers, exchange calls,
  live execution, report verdict changes, console routing, order/risk logic, or
  trading behavior.
- PR #910 passed Claude + Hermes + CI. Local validation covered the full
  `tests/test_live_performance_report.py` and `tests/test_live_event_query.py`
  suites, py_compile for touched files and tests, `git diff --check`, and a
  touched-file silent-handling scan.
- VPS5 pulled from `c8c51d73` to `ac949f03` without bot restart because the
  deployed change was read-only report tooling and docs. The five configured
  bots were left running.
- A concise 2-minute performance-report smoke at `ac949f03` reported
  `ok=true`, `error_count=0`, `warning_count=0`, `files_scanned=6`, and
  `file_discovery` with `candidate_files=3748`, `event_segments=949`,
  `rotated_skipped=943`, `scope_pruned=0`, and no bot-id path pruning. A fresh
  2-minute brief smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
- Repository pulled through PR #909 at `c8c51d73`.
- PR #909 projected bounded event-file discovery metadata into
  `live-smoke-report` full, summary, and brief output by using the public
  metadata-returning event discovery helper added after PR #908. The slice does
  not add event producers, exchange calls, live execution, smoke verdict
  changes, or trading behavior.
- PR #909 passed Claude + Hermes + CI. Local validation covered the full
  `tests/test_live_smoke_report.py` and `tests/test_live_event_query.py`
  suites, py_compile for touched files and tests, `git diff --check`, and a
  touched-file silent-handling scan.
- VPS5 pulled from `0c0024a3` to `c8c51d73` without bot restart because the
  deployed change was read-only report/query tooling and docs. The five
  configured bots were left running.
- A fresh 2-minute brief smoke at `c8c51d73` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. The brief smoke included the new
  `monitor.file_discovery` block with `candidate_files=3748`,
  `event_segments=949`, `rotated_skipped=943`, `scope_pruned=0`, and no bot-id
  path pruning, confirming the shareable projection contains only bounded
  counts/flags and no file paths.
- Repository pulled through PR #908 at `0c0024a3`.
- PR #908 added bounded `file_discovery` metadata to
  `live-event-query` reports, while preserving the existing
  `discover_event_files()` list-returning API. The slice does not add event
  producers, exchange calls, live execution, smoke verdict changes, or trading
  behavior.
- PR #908 passed Claude + Hermes + CI. Local validation covered the full
  `tests/test_live_event_query.py` suite, py_compile for touched query files
  and tests, `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `f792f889` to `0c0024a3` without bot restart because the
  deployed change was read-only query tooling and docs. The five configured
  bots were left running.
- Focused VPS5 query smoke showed bounded discovery metadata:
  `--bot-id binance/binance_01` over monitor root completed with `ok=true`,
  `files_scanned=1`, `file_discovery.bot_path_pruning_applied=true`,
  `candidate_files=3749`, `event_segments=951`, `rotated_skipped=945`, and
  `scope_pruned=5`, with no parser errors. A bounded 2-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, clean tracked repository state at `0c0024a3`,
  `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- Repository pulled through PR #907 at `f792f889`.
- PR #907 added conservative `live-event-query --bot-id` path pruning for
  path-shaped bot ids such as `exchange/user`, while preserving full-scan
  behavior for opaque event-level bot ids such as `bot_1`. The slice does not
  add event producers, exchange calls, live execution, smoke verdict changes,
  or trading behavior.
- PR #907 passed Claude + Hermes + CI. Local validation covered the full
  `tests/test_live_event_query.py` suite, py_compile for touched query files
  and tests, `git diff --check`, and a touched-file silent-handling scan.
- VPS5 pulled from `b7b34758` to `f792f889` without bot restart because the
  deployed change was read-only query tooling and docs. The five configured
  bots were left running.
- Focused VPS5 query smoke showed the new path-pruning behavior:
  `--bot-id binance/binance_01` over monitor root completed with `ok=true`,
  `files_scanned=1`, and `event_tail_limited_files=1`; an opaque
  `--bot-id bot_opaque_no_match` query preserved full-scan behavior with
  `files_scanned=4` and no matched events. A bounded 2-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, clean tracked repository state at `f792f889`,
  `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- Repository pulled through PR #906 at `b7b34758`.
- PR #906 added read-only `live-event-query` filters for envelope labels
  `source`, `component`, and `side`, using the existing repeated/
  comma-separated filter contract. The slice also made `source` visible in
  compact query records and folded the PR #903/#904 progress evidence into the
  same real observability PR, replacing the closed docs-only PR #905.
- PR #906 passed Claude + Hermes + CI on the final head. Local validation
  covered the full `tests/test_live_event_query.py` suite, py_compile for
  touched query files and tests, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `1dd115cc` to `b7b34758` without bot restart because the
  deployed change was read-only query tooling and docs. The five configured
  bots were left running.
- Focused VPS5 query smoke on Binance current monitor events showed the new
  filter echo and no parser errors: `--source executor --component order_wave`
  returned `ok=true` with `files_scanned=1`, and `--side buy,sell` returned
  `ok=true` with one matched `entry.initial_distance_gate_blocked` event.
  A bounded 2-minute brief smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state
  at `b7b34758`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- Broad parallel VPS5 monitor scans over the full root were interrupted after
  about 90 seconds. Follow-up smoke used single-bot/current-segment paths and
  completed in under 13 seconds. Future VPS smoke probes should avoid parallel
  broad monitor scans unless the query path is intentionally being stress
  tested.
- Repository pulled through PR #903 at `1dd115cc`.
- PR #903 added read-only `risk_events.hsl_status` projections to
  `live-smoke-report` full, summary, and brief output, derived only from
  existing `hsl.status` monitor events. The projection summarizes HSL status
  totals, bot/symbol counts, tier counts, signal-mode counts, and bounded
  closest-to-red labels.
- PR #903 fixed issue #904 before merge by filtering shareable summary
  `risk_events.groups[].latest_data` through a fixed value-safe whitelist.
  Shareable summary and brief reports do not expose raw drawdown, distance, or
  threshold magnitudes; the local full report keeps detailed HSL magnitudes for
  local diagnostics.
- PR #903 passed Claude + Hermes + CI. Local validation covered the focused
  risk smoke-report regression, the full `tests/test_live_smoke_report.py`
  suite, py_compile for touched files, `git diff --check`, and a touched-file
  silent-handling scan.
- VPS5 pulled from `852d4b89` to `1dd115cc` without bot restart because the
  deployed change was read-only smoke-report projection code. The five
  configured bots were left running.
- A fresh 2-minute brief smoke after the pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at `1dd115cc`,
  `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. The brief report included
  `risk_events.hsl_status` with `tier_counts={"red":4}`, `bots=3`, and a
  bounded symbol sample of `ZEC/USDT:USDT`; no magnitude fields were present in
  the shareable brief output.
- Repository pulled through PR #901 at `9b3c29ad`.
- PR #901 added a read-only `live-smoke-report --brief` projection for
  `hsl_replay.max_completed_elapsed_ms`, derived only from existing sanitized
  completed HSL replay groups and the same bounded elapsed-field policy used
  for active replay timing. The slice does not add event producers, exchange
  calls, cache mutation, readiness gates, smoke verdict changes, console
  routing, order/risk logic, or trading behavior.
- PR #901 passed Claude + Hermes + CI. Local validation covered the focused HSL
  replay smoke-report regression, the full `tests/test_live_smoke_report.py`
  suite, py_compile for touched files, `git diff --check`, and a
  touched-file silent-handling scan.
- VPS5 pulled from `e1fcb038` to `9b3c29ad` without bot restart because the
  deployed change was read-only smoke-report projection code. The five
  configured bots were left running.
- A fresh 2-minute brief smoke after the pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at `9b3c29ad`,
  `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. The short sampled window had
  `hsl_replay.total=0`, so the new completed-HSL replay brief field was not
  populated by live data in that smoke.
- Repository pulled through PR #899 at `e1fcb038`.
- PR #898 was docs-only retuned-loop progress tracking. PR #899 added a
  read-only `live-smoke-report --brief` projection for the worst active HSL
  replay elapsed time, latest-event age, and active stage counts from existing
  sanitized HSL replay groups. PR #899 does not add event producers, exchange
  calls, cache mutation, readiness gates, smoke verdict changes, console
  routing, order/risk logic, or trading behavior.
- PR #899 had one reviewer finding on the first head: the brief
  `max_active_latest_elapsed_ms` field under-reported active replay time by
  reading only `latest_elapsed_ms`. The final head fixed this by taking the
  maximum across the bounded HSL elapsed fields already exposed by the summary
  projection, and updated the regression test. Hermes approved the fixed head;
  CI was green. Claude posted comment-type approvals on both final heads, but
  not formal `APPROVED`-state reviews, so PR #898 and PR #899 used the
  degraded low-risk docs/tooling gate.
- VPS5 pulled from `aebc3667` to `e1fcb038` without bot restart because the
  deployed changes were docs plus read-only smoke-report projection code. The
  five configured bots were left running.
- A fresh 2-minute brief smoke after the pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at `e1fcb038`,
  `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. The short sampled window had
  `hsl_replay.active_bots=0`, so the new active HSL replay brief fields were
  not populated by live data in that smoke. Local tests and Hermes review
  validated the active-HSL fixture path.
- Repository pulled through PR #897 at `aebc3667`.
- PR #897 was docs-only progress/backlog tracking for the
  `exchange.config_refresh` smoke projection and did not change runtime code.
- Bots were restarted from `/root/bots_vps5.yaml` after PR #897 so running
  processes would load the PR #894 event producer and the PR #896 smoke-report
  projection. Binance, GateIO, OKX, and Hyperliquid stopped within the bounded
  restart procedure; Kucoin did not exit within a 180-second Ctrl+C observation
  window and required SIGTERM before reload. All five configured bots were
  restarted and left running.
- A fresh 2-minute brief smoke after the restart reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at `aebc3667`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Remaining attention was non-hard: two active HSL replay bots and one
  `ema.unavailable` group.
- A 20-minute summary smoke returned `ok=false` only because the window
  included real GateIO risk/HSL CRITICAL text lines for ZEC:
  `entering HSL coin RED supervisor loop` and `HSL[long:ZEC/USDT:USDT] RED
  stop finalized`. Structured events also showed the same state as
  `hsl.status` `cooldown_active`; process liveness, repo state, remote calls,
  and account-critical remote calls were otherwise clean. This reinforces that
  HSL RED/cooldown is a real risk signal, not a deploy/tooling failure.
- The `exchange.config_refresh` query over the post-restart monitor events
  still returned `matched_events=0`, and both 2-minute and 20-minute smoke
  projections reported `exchange_config_refresh.total=0`. This is inconclusive
  for the Binance hourly `-4084` maintenance traceback because the sampled
  windows have not yet proven an hourly refresh occurrence after the restarted
  processes loaded the producer.
- The same restart/smoke re-confirmed the coin-HSL startup latency gap. In a
  20-minute window, Binance, OKX, and Kucoin still had active HSL replay groups:
  Binance was at pair `10/27` after `645s`, OKX at pair `22/28` after
  `1522s`, and Kucoin was still in `price_history_symbol_fetch_started` after
  `1541s`. GateIO completed enough replay to enter its ZEC cooldown path, with
  `full-warmup` reported at about `1843s`.
- A later 5-minute summary smoke on the same deployment remained clean for
  processes, repository state, remote-call totals, and account-critical
  remote-call totals, but was hard-red from real runtime signals: OKX finalized
  `HSL[long:ZEC/USDT:USDT]` RED without an exchange order after a slow coin-HSL
  replay, and Hyperliquid emitted one `fills.refresh_summary`
  `fill_refresh_failed` event after several successful fill refreshes. Focused
  OKX evidence showed coin-HSL replay completed after `1904.391s`,
  `hsl-ready=2462.54s`, then `hsl.red_finalized_without_order`,
  `hsl.red_triggered`, and `hsl.cooldown_started` for ZEC at
  `17828078249xx`. This reinforces both the HSL startup-latency backlog item
  and the value of the structured risk/exchange-health smoke sections.
- Repository pulled through PR #896 at `53b8accb`.
- PR #896 added read-only `live-smoke-report` projections for
  `exchange.config_refresh` events: full report
  `exchange_config_refresh_health`, summary limited groups, and brief
  `exchange_config_refresh` counters. The projection intentionally excludes
  raw free-text `data.error` and carries only bounded labels, `error_type`,
  status/reason counts, and timing fields. It does not change smoke verdict
  logic or text-log classification.
- PR #896 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation covered the full
  `tests/test_live_smoke_report.py` suite, py_compile for touched files, and
  `git diff --check`.
- After deploying PR #896 to VPS5 without bot restart, a 5-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, clean tracked repository state at `53b8accb`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`, and the
  new `exchange_config_refresh` brief section present with `total=0`. This is
  expected because the bots have not restarted since PR #894, so the new event
  producer is not loaded by running processes yet. This short smoke window also
  did not span an hourly Binance config refresh occurrence, so it is not
  evidence that the intermittent `-4084` text-log traceback is resolved.
- Repository pulled through PR #894 at `796ceb38`.
- PR #894 added the off-console/text structured event
  `exchange.config_refresh` for hourly maintenance `init_markets` refresh
  success/failure. Failures carry bounded sanitized error text, `error_type`,
  `context=maintain_hourly_cycle`, `operation=init_markets`, elapsed timing,
  and distinct success/failure reason codes. The wrapper re-raises the original
  exception and has an extra best-effort guard so event emission cannot mask
  refresh success or the original refresh failure.
- PR #894 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation covered
  `tests/test_exchange_config_refresh_event.py`, `tests/test_live_event_bus.py`,
  `tests/test_live_event_registry_docs.py`, py_compile for touched Python
  files, and `git diff --check`.
- After deploying PR #894 to VPS5, the bots were not restarted. All five
  configured `passivbot live` processes remained running on the previously
  loaded Python code while the repository was fast-forwarded to `796ceb38`.
  Live emission evidence for `exchange.config_refresh` therefore remains
  pending until the next planned bot restart and hourly maintenance cycle.
- Repository pulled through PR #893 at `3be95aca`.
- PR #893 was docs-only progress/backlog tracking for the PR #892 deploy and
  the discovered Binance hourly hedge-mode/config-refresh traceback gap.
  No bot restart or smoke was needed beyond confirming all five configured
  `passivbot live` processes remained running.
- Repository pulled through PR #892 at `7e7ce16f`.
- PR #892 added read-only `passivbot tool live-event-query --level`, so local
  structured monitor-event queries can be scoped by envelope severity and
  composed with existing timeline, trace-summary, order-trace, and cycle-trace
  views. The slice did not add event producers, exchange calls, cache mutation,
  readiness gates, console routing, order/risk logic, or trading behavior.
- PR #892 passed the normal review gate on the final head: Claude approved
  `e456ced6`, Hermes approved `e456ced6`, and CI was green. Local validation
  covered `tests/test_live_event_query.py`, py_compile for touched Python
  files, `git diff --check`, and a touched-file silent-handling scan.
- After deploying PR #892 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. A 10-minute `live-event-query --level
  warning,error,critical --trace-summary --event-tail-lines 5000` smoke on
  `v8@7e7ce16f` returned `ok=true`, `error_count=0`, `matched_events=33`,
  and `trace_summary.levels.warning=33`. A parallel 5-minute brief smoke
  showed all five configured bots matched, clean tracked repository state,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`, but
  returned `ok=false` because Binance logged an existing non-risk traceback
  from hourly hedge-mode config refresh (`binanceusdm -4084 Method is not
  allowed currently`). That operational gap is tracked in
  `docs/plans/live_ops_improvement_backlog.md`.
- Repository pulled through PR #890 at `1498abc9c`.
- PR #890 exposed the structured event-window `enabled` flag in
  `live-smoke-report --brief`, matching the full report and the existing
  `logs.window.enabled` projection. The slice did not add event producers,
  exchange calls, cache mutation, readiness gates, console routing, order/risk
  logic, or trading behavior.
- PR #890 passed the normal review gate: Claude approved head `f655a8c`,
  Hermes approved head `f655a8c`, and CI was green. Local validation covered
  `tests/test_live_smoke_report.py`, compileall for touched files,
  `git diff --check`, and an added-diff silent-handling scan.
- After deploying PR #890 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@1498abc9` returned
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  clean tracked repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and the new
  `event_window.enabled=true` brief projection. Remaining attention was
  non-hard EMA readiness and HSL/unstuck status groups.
- Repository pulled through PR #888 at `4b435d33e`.
- PR #888 exposed already-computed text-log window counters in
  `live-smoke-report --brief`, making concise smoke output show whether
  log-derived hard/attention counts came from a time-windowed scan and how many
  log lines were considered, skipped before/after the window, unparseable, or
  dropped by the unparseable-line policy. The slice did not add event
  producers, exchange calls, cache mutation, readiness gates, console routing,
  order/risk logic, or trading behavior.
- PR #888 passed the normal review gate: Claude approved head `fd4f918`,
  Hermes approved head `fd4f918`, and CI was green. Local validation covered
  `tests/test_live_smoke_report.py`, compileall for touched files,
  `git diff --check`, and an added-diff silent-handling scan.
- After deploying PR #888 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@4b435d33` returned
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  clean tracked repository state, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and the new `logs.window` brief
  projection with `enabled=true`, `lines_considered=28`,
  `lines_skipped_before=1730`, `unparsed_ts=56`, and
  `unparsed_policy=keep`. Remaining attention was non-hard EMA readiness and
  HSL/unstuck status groups.
- Repository pulled through PR #886 at `60c79c3a4`.
- PR #886 exposed already-computed `startup_timings` in
  `live-smoke-report --summary` and `--brief`. The full report already had the
  sanitized startup timing evidence; this slice made slow restart phases visible
  in concise smoke-loop projections without adding event producers, exchange
  calls, cache mutation, readiness gates, console routing, order/risk logic, or
  trading behavior.
- PR #886 passed the normal review gate: Claude approved head `87a3aa5`,
  Hermes approved head `87a3aa5`, and CI was green. Local validation covered
  `tests/test_live_smoke_report.py`, compileall for touched files,
  `git diff --check`, and an added-diff silent-handling scan.
- After deploying PR #886 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A no-window brief smoke proved the new
  `startup_timings` projection was present, but returned `ok=false` because it
  included older text-log hard matches. A 5-minute brief smoke on
  `v8@60c79c3a` returned `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`, and the
  new `startup_timings` brief key. Remaining attention was non-hard EMA
  readiness, HSL/unstuck status, and text-log attention groups.
- Repository pulled through PR #885 at `d0a8e0da5`.
- PR #885 added opt-in `passivbot tool live-event-query --event-tail-lines N`
  for repeated recent-window event queries over large current monitor segments.
  The default remains full event validation; bounded query output reports
  `event_tail_lines`, `event_tail_limited_files`, and
  `event_tail_skipped_lines` in `event_window`. The slice did not add event
  producers, exchange calls, cache mutation, readiness gates, console routing,
  order/risk logic, or trading behavior.
- PR #885 passed the normal review gate: Claude approved head `f9550f8f`,
  Hermes approved head `f9550f8f`, and CI was green. Local validation covered
  `tests/test_live_event_query.py`, compileall for touched files,
  `git diff --check`, and a touched-diff silent-handling scan.
- After deploying PR #885 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. A full-validation 5-minute `hsl.status` query on
  `v8@d0a8e0da` reported `ok=true`, `error_count=0`, and
  `matched_events=40` in 6.69s. The same query with
  `--event-tail-lines 5000` reported `ok=true`, `error_count=0`,
  `matched_events=38`, and exposed `event_tail_limited_files=4` in 7.04s.
  A smaller `--event-tail-lines 500` smoke reported `ok=true`,
  `matched_events=37`, `event_tail_limited_files=4`, and
  `event_tail_skipped_lines=9941` in 3.26s, proving bounded-evidence metadata
  is visible on live artifacts.
- Repository pulled through PR #883 at `5ca270800`.
- PR #883 added opt-in `passivbot tool live-smoke-report --event-tail-lines N`
  for repeated recent-window smoke checks over large current monitor segments.
  The default remains full monitor-event validation; bounded runs report
  `event_tail_lines`, `event_tail_limited_files`, and
  `event_tail_skipped_lines` in `event_window`, so reduced evidence is explicit.
  The slice did not add event producers, exchange calls, cache mutation,
  readiness gates, console routing, order/risk logic, or trading behavior.
- PR #883 passed the normal review gate: Claude approved head `16b8c419`,
  Hermes approved head `16b8c419`, and CI was green. Local validation covered
  `tests/test_live_smoke_report.py`, compileall for touched files,
  `git diff --check`, and the silent-handling scan on touched Python files.
- After deploying PR #883 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A full-validation 5-minute brief smoke on
  `v8@5ca27080` completed in 16.74s and reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  clean tracked repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. The same 5-minute brief smoke with
  `--event-tail-lines 5000` completed in 12.94s, reported the same hard-green
  health counters, and exposed `event_tail_limited_files=6` plus
  `event_tail_skipped_lines=6919` in `event_window`. Known non-hard attention
  remained from EMA readiness and HSL/unstuck status groups.
- Repository pulled through PR #881 at `197d74942`.
- PR #881 made `live-smoke-report` reuse one monitor-event parse for both
  monitor validation/summary and windowed smoke aggregates. This removed the
  previous `build_event_report()` plus `_scan_events()` double parse for the
  same monitor segments. The slice did not add event producers, exchange calls,
  cache mutation, readiness gates, console routing, order/risk logic, or
  trading behavior.
- PR #881 passed the normal review gate: Claude approved head `520b3916`,
  Hermes approved head `520b3916`, and CI was green. Local validation covered
  `tests/test_live_smoke_report.py`, `tests/test_live_event_query.py`,
  compileall for touched files, and `git diff --check`.
- After deploying PR #881 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A no-log 1-minute brief smoke improved from the
  pre-merge 15-19s range to 10.6s while reporting `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  A 5-minute brief smoke with logs enabled completed in 15.4s after the same
  command had previously hit a 30s timeout wrapper; it reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Known non-hard attention remained from EMA readiness and HSL/unstuck status
  groups. Current monitor segments still required row-level skipping of roughly
  25k-27k older events, so tail scanning or a lightweight local index remains a
  possible future operator-tooling optimization.
- Repository pulled through PR #880 at `74a07640`.
- PR #880 was progress-ledger-only and recorded the deployment/smoke evidence
  for PR #879. VPS5 was pulled without restarting bots; all five configured
  `passivbot live` processes remained running. Before PR #881, a 5-minute brief
  smoke on `v8@74a07640` hit a 30s timeout wrapper, which motivated the
  single-pass smoke-report slice.
- Repository pulled through PR #879 at `72d450ba6`.
- PR #879 made read-only `live-event-query` output use the same path-resolved
  exchange/user labels that filtering already uses for legacy monitor rows.
  Compact query events, cycle-trace timelines, and trace-summary exchange/user
  counters now remain self-describing even when older rows lack embedded
  exchange/user fields. The slice did not add event producers, exchange calls,
  cache mutation, readiness gates, console routing, order/risk logic, or trading
  behavior.
- PR #879 passed the normal review gate: Claude approved head `558bca5e`,
  Hermes approved head `558bca5e`, and CI was green. Local validation covered
  the full `tests/test_live_event_query.py` suite, compileall for touched files,
  `git diff --check`, and the silent-handling scan on touched files.
- After deploying PR #879 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. A focused Gate.io HSL status query over the last
  30 minutes completed under a 20-second timeout wrapper and showed path-scoped
  labels in both compact output and trace summary:
  `query.events[0].exchange=gateio`, `query.events[0].user=gateio_01`,
  `trace_summary.exchanges.gateio=115`, and
  `trace_summary.users.gateio_01=115`.
- A 5-minute brief smoke on `v8@72d450ba` reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state, `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Known non-hard attention remained from EMA readiness and HSL status groups.
- Repository pulled through PR #877 at `1c9d05036`.
- PR #877 added read-only `live-event-query` file-level window pruning: when
  `--since-ms` or `--recent-minutes` is set, files whose filesystem mtime is
  strictly before the query window are skipped before opening, and
  `event_window.files_skipped_before_window` reports the count. Row-level time
  filtering remains authoritative for all scanned files. The slice did not add
  event producers, exchange calls, cache mutation, readiness gates, console
  routing, order/risk logic, or trading behavior.
- PR #877 passed the normal review gate: Claude approved head `47a1ec71`,
  Hermes approved head `47a1ec71`, and CI was green. Local validation covered
  the full `tests/test_live_event_query.py` suite, compileall for touched files,
  `git diff --check`, and the silent-handling scan on touched files.
- After deploying PR #877 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. The previously slow Gate.io/ZEC rotated query
  completed under a 20-second timeout wrapper, scanning 4 Gate.io files and
  reporting `files_skipped_before_window=160`; it found no matching ZEC
  `hsl.status` rows in the 180-minute window. No probe process was left running.
- A 5-minute brief smoke on `v8@1c9d0503` reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state, `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Known non-hard attention remained from EMA readiness and HSL/unstuck status
  groups.
- Repository pulled through PR #875 at `fda17fad1`.
- PR #875 added read-only `passivbot tool live-event-query --exchange` and
  `--user` filters plus conservative monitor-root path pruning, so incident
  queries can focus one VPS account without scanning unrelated bot directories
  when the root has the standard `<monitor>/<exchange>/<user>/events` layout.
  Direct file and direct `events/` directory workflows remain readable and use
  row-level filtering. The slice did not add event producers, exchange calls,
  cache mutation, readiness gates, console routing, order/risk logic, or trading
  behavior.
- PR #875 passed the normal review gate: Claude approved head `6d70eacd`,
  Hermes approved head `6d70eacd`, and CI was green. Local validation covered
  the full `tests/test_live_event_query.py` suite, compileall for touched files,
  `git diff --check`, and the silent-handling scan on touched files.
- After deploying PR #875 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@fda17fad` reported
  `ok=true`, `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  clean tracked repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Known non-hard attention remained
  from EMA readiness and HSL status groups.
- Follow-up gap: a focused Gate.io/ZEC query using exchange/user scoping,
  `hsl.status`, `data_eq symbol=ZEC/USDT:USDT`, `--recent-minutes 180`, and
  `--include-rotated` still ran for roughly two minutes before manual
  interruption. No probe process was left running afterward. This shows
  exchange/user path pruning is useful but not sufficient for rotated incident
  queries; the next operator tooling slice should add safer time-bounded
  rotated scanning or an event index.
- Repository pulled through PR #874 at `f09c79887`.
- PR #874 was progress-ledger-only and recorded the HSL anchor/query evidence
  after PR #873. No live bot restart was needed. All five configured
  `passivbot live` processes remained running after the deploy. A 5-minute
  brief smoke on `v8@f09c79887` reported `ok=true`, `hard_failures=0`,
  `hard_failure_sources.total=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state,
  `account_critical_remote_calls.failed=0`, and one non-account-critical
  remote-call failure. Known non-hard attention remained from EMA readiness and
  HSL status/cooldown groups.
- Repository pulled through PR #873 at `20351283`.
- PR #873 added `passivbot tool live-event-query --data-eq key=value`, a
  repeated top-level live-event `data` equality filter. It lets operators query
  exact event subsets such as `stop_event_anchor_source=current_time_fallback`
  or `tier=red` without changing event producers or forcing payload data into
  compact output unless `--include-data` is requested. The slice is
  query/tooling-only and does not add exchange calls, cache mutation, readiness
  gates, console routing, order/risk logic, or trading behavior.
- PR #873 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation covered the full `tests/test_live_event_query.py`
  suite, focused data-filter API/CLI tests, compileall for touched files,
  `git diff --check`, and the silent-handling scan on touched files.
- After deploying PR #873 to VPS5, the bots were not restarted because the
  change was read-only query tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@20351283` reported
  `ok=true`, `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  clean tracked repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. A live event-query probe against
  VPS5 monitor artifacts with `--event-type hsl.status --data-eq tier=red`
  returned matching ZEC red cooldown events and reported `ok=true`, proving the
  new filter works on current monitor output. Remaining attention came from
  known non-hard EMA readiness groups and existing HSL status/cooldown groups.
- Repository pulled through PR #872 at `c45537c1`.
- PR #872 added the bounded `risk_events.hsl_flat_finalization_anchors`
  aggregate to `live-smoke-report` full, summary, and brief outputs. It rolls
  up existing `hsl.red_finalized_without_order` anchor-source labels and
  current-time-fallback counts, making fallback-heavy flat HSL finalizations
  visible at window level without opening individual event payloads. The slice
  is report-only and does not add event producers, exchange calls, cache
  mutation, readiness gates, console routing, order/risk logic, or trading
  behavior.
- PR #872 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation covered the full `tests/test_live_smoke_report.py`
  suite, focused smoke-report summary/brief CLI tests, compileall for touched
  files, and `git diff --check`.
- After deploying PR #872 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@c45537c1` reported
  `ok=true`, `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  clean tracked repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. A focused 180-minute brief smoke
  confirmed the new `hsl_flat_finalization_anchors` section was present; no
  flat-finalization anchor events were present in that sampled window
  (`total=0`). Remaining attention came from known non-hard EMA readiness
  groups and existing HSL status/cooldown groups.
- Repository pulled through PR #870 at `b6cd3b9e`.
- PR #870 added bounded anchor provenance to existing
  `hsl.red_finalized_without_order` events and `live-smoke-report` risk-event
  projections. Flat HSL finalizations now distinguish `panic_fill`,
  `provided_stop_event`, and `current_time_fallback` anchors without changing
  stop timestamp selection, cooldown math, latch writes, exchange calls,
  readiness gates, order creation, or console routing.
- PR #870 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation in a detached worktree covered
  `tests/test_live_smoke_report.py`, focused coin-HSL flat-finalization anchor
  tests, the existing panic-fill timestamp safeguard test, compileall for
  touched files, and `git diff --check`.
- After deploying PR #870 to VPS5, the bots were not restarted because the
  change was observability-only. All five configured `passivbot live` processes
  remained running. A 5-minute brief smoke on `v8@b6cd3b9e` reported
  `ok=true`, `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  clean tracked repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from
  known non-hard EMA readiness groups and existing HSL cooldown/status groups.
- Repository pulled through PR #868 at `a5af777c`.
- PR #868 made the already-emitted `hsl.red_finalized_without_order` event
  visible in the operator-facing report surfaces. `live-smoke-report` now
  includes it in `risk_events` with bounded provenance fields only, and
  `live-performance-report` now includes it in `risk_activity` as envelope
  grouping metadata. The smoke projection explicitly filters drawdown ratio
  fields for this event, so the shareable report carries stop/cooldown
  timestamps, booleans, and counts without account/risk magnitude values. The
  slice did not add event producers, exchange calls, cache mutation, readiness
  gates, console routing, or trading behavior.
- PR #868 passed the normal review gate: Claude approved the amended
  value-safety head at `ebeb9caf`, Hermes approved the same head, and CI was
  green. Local validation covered the full smoke/performance report test files
  (`94 passed`), compileall for touched files, and `git diff --check`.
- After deploying PR #868 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A 5-minute brief smoke on `v8@a5af777c` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Known non-hard attention remained from EMA readiness groups, existing HSL
  cooldown/status groups, and one non-hard staged-readiness group.
- Repository pulled through PR #866 at `d44c5132`.
- PR #866 added the active HSL replay age projection to
  `live-performance-report` startup readiness summaries. It derives the field
  from existing HSL replay profile data and reports `latest_event_age_ms` for
  active replay stages without adding event producers, exchange calls, cache
  mutation, readiness gates, console routing, or trading behavior.
- PR #866 passed the normal review gate: Claude approved the current head,
  Hermes approved, and CI was green. Local validation for the slice covered the
  focused live performance report tests and `git diff --check`.
- After deploying PR #866 to VPS5, the bots were not restarted because the
  change was read-only report tooling. All five configured `passivbot live`
  processes remained running. A later settled 5-minute smoke on `v8@d44c5132`
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`; known non-hard attention remained
  from EMA readiness groups, existing HSL cooldown groups, and one GateIO
  `RequestTimeout` text-log match.
- Repository pulled through PR #865 at `f63823a0`.
- PR #865 added the structured `hsl.red_finalized_without_order` event for HSL
  RED supervisor paths that finalize cooldown after authoritative state proves
  the pside/symbol is already flat and no exchange close order is needed. The
  event is routed away from console/text by default and carries bounded
  correlation fields: stop/cooldown timestamps, flat confirmation count,
  blocking-order counts, no-exchange-close booleans, and HSL drawdown summary
  ratios. It does not change finalization conditions, panic supervision, order
  creation, cooldown math, forced modes, or exchange writes.
- PR #865 passed the normal review gate on the current rebased head: Claude
  approved the rebase at `8032495d`, Hermes approved the same head, and CI was
  green. Local validation for the slice covered the focused coin-HSL
  finalization tests, event-bus route/reason-code tests, registry-doc tests,
  compileall for touched files, and `git diff --check`.
- After deploying PR #865 to VPS5, the bots were not restarted. All five
  configured `passivbot live` processes were still running. A 5-minute settled
  smoke reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=f63823a0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Known non-hard attention remained
  from EMA readiness groups and ZEC HSL cooldown groups. The new event was not
  expected to appear without a new flat RED finalization transition.
- Repository pulled through PR #864 at `66ac3a1f`.
- PR #864 updated this progress ledger after PR #863 and caught the ledger up
  through PRs #859-#863. It was docs-only, passed CI, and was approved by both
  Claude and Hermes.
- Repository pulled through PR #863 at `cf7e5d25`.
- PR #863 added report-only active-age projections for non-terminal HSL replay
  groups. `live-smoke-report` and `live-performance-report` now include the age
  since the latest active replay event as `active_latest_event_age_ms` plus
  `latest.derived.latest_event_age_ms`, making stuck active replay stages
  visible without raw NDJSON inspection. The slice changed only report
  projections and tests; it did not change event producers, exchange calls,
  HSL replay behavior, readiness gates, console routing, or trading behavior.
- PR #863 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Hermes noted two unrelated full-suite smoke-report test
  failures reproduced on clean `origin/v8` in that environment; Claude's clean
  worktree run of the touched performance/smoke tests passed.
- After deploying PR #863 to VPS5, the bots were not restarted because this was
  read-only report tooling. All five configured `passivbot live` processes were
  still running. A 5-minute settled smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=cf7e5d25`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Known non-hard attention remained
  from EMA readiness groups and ZEC HSL cooldown groups.
- A focused 30-minute performance-report extraction validated the new active HSL
  replay age field on VPS5. Kucoin had an active HSL replay group at
  `hsl_price_history_symbol_fetch_started` for `ASTER/USDT:USDT`, and the
  report projected both `active_latest_event_age_ms=1352835` and
  `latest.derived.latest_event_age_ms=1352835`. The same report showed
  `startup_readiness.hsl_replay_active_count=1`, confirming the field is
  populated for active, non-terminal replay stages.
- Ledger catch-up since PR #858:
  - PR #859 updated this progress ledger after the HSL history-build progress
    smoke.
  - PR #860 added HSL price-history fetch progress events, including bounded
    per-symbol start/completed progress for the expensive price-history stage.
  - PR #861 fixed live-event-query bot path filtering so monitor paths such as
    `gateio/gateio_01` are matched consistently.
  - PR #862 made HSL replay query/report output more usable by exposing
    latest-stage elapsed information for active HSL history/replay work.
  - PR #863 added the current active-event age projection described above.
- Repository pulled through PR #858 at `8c908e72`.
- PR #858 added opt-in structured HSL history-build progress events inside
  `get_balance_equity_history` for HSL replay callers. The new
  `hsl.replay.progress` reason codes are `hsl_history_inputs_loaded`,
  `hsl_history_empty`, `hsl_price_history_fetch_started`,
  `hsl_price_history_fetch_completed`, `hsl_timeline_replay_started`, and
  `hsl_timeline_replay_completed`. Payloads are limited to counts, timestamps,
  and elapsed timings; no balances, prices, equity, realized PnL, sizes, raw
  fills, or candle rows are emitted.
- PR #858 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. Local validation covered the HSL history pacing regression
  test, unsupported historical-symbol skip test, event-registry docs test,
  coin-mode HSL tests, compileall for touched files, and `git diff --check`.
- After deploying PR #858 to VPS5, all five configured bots were restarted from
  `/root/bots_vps5.yaml` and left running on `v8@8c908e72`. The follow-up
  read-only smoke reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- The first VPS5 event query after the restart found
  `hsl_history_inputs_loaded` and `hsl_price_history_fetch_started` for the
  Binance, GateIO, Kucoin, and OKX forager bots. It had not yet found
  `hsl_price_history_fetch_completed` or either timeline replay stage in that
  early window, so the remaining startup bottleneck is now localized to HSL
  price/candle history fetching before dense timeline replay.
- Shutdown observation from the same deploy: Hyperliquid stopped promptly on
  the first `killbots` signal, while the four HSL forager bots remained alive
  after roughly 20 seconds during HSL replay and required a second signal. This
  reinforces the existing shutdown-responsiveness backlog item; it was not a
  PR #858 trading-behavior regression.
- Repository pulled through PR #857 at `7c199df1`.
- PR #856 added structured `hsl.raw_red_pending` diagnostics for coin-mode HSL
  cases where raw drawdown is beyond red but EMA-confirmed drawdown has not
  crossed red. The event is routed away from console/text by default and keeps
  payloads to bounded HSL metrics/status fields.
- PR #857 was a trading-safety interruption, not a logging slice: historical
  HSL replay now preserves a panic-marker cooldown only when reconstructed RED
  metrics confirm the historical panic marker. Incomplete marker metrics fail
  loudly, and ignored markers are logged/observable instead of silently keeping
  a stale bad cooldown alive.
- After deploying PRs #856/#857 to VPS5, all five configured bots were
  restarted and left running on `v8@7c199df1`. A read-only smoke reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`, and
  `processes.config_checks.ok=true`.
- The same smoke showed the remaining live startup-readiness gap more
  precisely: Binance, Kucoin, GateIO, and OKX had emitted
  `hsl.replay.started` but not `hsl.replay.progress`/`completed` after the
  restart window, meaning they were still blocked inside HSL balance/equity
  history assembly before the per-pair replay loop. That gap is now the driver
  for the next observability slice.
- Current ledger catch-up through PR #854:
  - PR #836 updated this progress ledger after the realized-loss gate event
    slice.
  - PR #837 added structured `entry.min_effective_cost_blocked` events for
    min-effective-cost entry skips, routed away from console/text by default and
    covered by event-registry and live-event tests.
  - PR #838 added structured initial-entry distance gate blocked/cleared events
    with throttled producer behavior and focused regression tests.
  - PR #839 was a trading-safety interruption, not a logging slice: it guards
    account-level `unified`/`pside` HSL replay when balance override is active
    and anchors replay realized PnL to the lookback window.
  - PR #840 documented the HSL false-panic recovery follow-up in the live ops
    backlog and incident report.
  - PR #841 added preflight detection for unsafe account-level HSL replay with
    balance override, aligned with the runtime guard from PR #839.
  - PR #842 added structured
    `risk.entry_cooldown_delta_anchored` events and registry/report support for
    entry-cooldown anchor updates.
  - PR #843 was a trading-safety interruption, not a logging slice: it
    clarified the VPS3 HSL false-panic incident status.
  - PR #844 updated this progress ledger after the HSL safety slices.
  - PR #845 added structured `hsl.replay.failed` events, smoke/performance
    report projections, and the terminal startup guard event for unsafe
    account-level HSL replay with balance override.
  - PR #846 added structured `execution.create_skipped` events for the
    pre-create market-distance guard, using reason code
    `limit_order_create_market_distance` and bounded group/sample payloads.
  - PR #847 updated this progress ledger after the market-distance guard event
    slice.
  - PR #848 added structured `execution.create_skipped` events for the
    remaining pre-create market snapshot skip paths: non-refreshable invalid
    planning snapshots, failed market snapshot refresh, and stale market
    snapshots after refresh. The slice uses reason codes
    `pre_create_planning_snapshot_invalid` and
    `pre_create_market_snapshot_unavailable`, bounded allowlisted details, and
    no raw exception text.
  - PR #849 updated this progress ledger after the pre-create snapshot event
    slice.
  - PR #850 added structured `fills.refresh_summary` events for fill-cache
    doctor startup reports, legacy-cache quarantine actions, and rebuild-start
    decisions. The slice uses reason codes `fill_cache_doctor_report`,
    `fill_cache_quarantined`, and `fill_cache_rebuild_started`, with bounded
    scalar fields and no backup paths or raw doctor report blobs.
  - PR #851 updated this progress ledger after the fill-cache doctor startup
    event slice.
  - PR #852 added structured `market.snapshot_diagnostic_skipped` events for
    noncritical market snapshot diagnostic skips. The slice keeps the existing
    warning log and boolean helper behavior unchanged, routes the event away
    from console/text by default, and emits only bounded sanitized context,
    error type, and error text.
  - PR #853 updated this progress ledger after the market-snapshot diagnostic
    skip event slice.
  - PR #854 added a read-only smoke-report config/process check for the known
    unsafe `balance_override` plus account-level HSL replay contract. The
    check parses running `passivbot live` commands, resolves local configs,
    hard-fails smoke only for `unified`/`pside` HSL with an active balance
    override, and redacts override values from public process/report command
    surfaces.
- Repository pulled through PR #854 at `2fb9ffc5`.
- Bots were not restarted for PRs #853/#854. PR #853 was docs-only and PR #854
  was read-only smoke-report tooling; all five configured `passivbot live`
  processes remained running.
- PR #854 passed the normal review gate after two value-safety amendments:
  Claude approved the current head, Hermes approved the current head, and CI
  was green. Local validation before PR creation covered focused
  `tests/test_live_smoke_report.py`, the incident-bundle process-status
  integration test, compileall for touched Python files, and `git diff --check`.
- Immediate read-only VPS5 smoke after the PR #854 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=2fb9ffc5`, `remote_calls.failed=1`,
  `account_critical_remote_calls.failed=0`, and
  `processes.config_checks.ok=true` with `issues_count=0`. Remaining attention
  came from known non-hard EMA readiness and structured problem-event groups.
- Repository pulled through PR #850 at `13b0044d`.
- Bots were not restarted for PR #850 because the change is an
  observability-only event producer beside existing fill-cache doctor startup
  decisions. All five configured `passivbot live` processes remained running.
- PR #850 passed the normal review gate after addressing Hermes's important
  finding about doctor-performed `quarantine_legacy_files`: Claude approved the
  current head, Hermes approved the current head, and CI was green. The slice
  reuses the existing off-console `fills.refresh_summary` route and checked
  best-effort emitter, so event sink/pipeline failures do not affect cache
  repair, quarantine, rebuild, or startup behavior.
- Immediate read-only VPS5 smoke after the PR #850 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=13b0044d`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from known
  non-hard EMA readiness and HSL/unstuck status groups.
- A follow-up read-only VPS5 smoke after a short observation window again
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=13b0044d`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. All five configured bots were left
  running.
- Repository pulled through PR #848 at `446049aa`.
- Bots were not restarted for PR #848 because the change is an
  observability-only event producer beside already-existing pre-create
  `return []` decisions. All five configured `passivbot live` processes
  remained running.
- PR #848 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice reuses the existing off-console
  `execution.create_skipped` route and existing best-effort create-filter
  emitter, so event sink/pipeline failures do not affect order filtering.
- Immediate read-only VPS5 smoke after the PR #848 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=446049aa`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from known
  non-hard EMA readiness and HSL/unstuck status groups.
- A follow-up read-only VPS5 smoke after a short observation window again
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, clean tracked repository
  state at `repository.head=446049aa`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. All five configured bots were left
  running.
- VPS5 was already deployed at `b207bb42` with all five configured bots running.
  A 10-minute read-only smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected=[]`, clean
  tracked repository state, and `account_critical_remote_calls.failed=0`.
  Non-hard attention came from one recovered Hyperliquid candle timeout, six
  Hyperliquid EMA unavailable symbols, and active HSL replay on Binance, GateIO,
  and OKX.
- The same smoke re-confirmed the separate HSL coin replay performance/safety
  gap: after several minutes of startup, Binance, GateIO, and OKX were still in
  `hsl.replay.progress` pair replay at roughly 10-13% of estimated dense
  pair-row work. That gap remains tracked outside the logging-overhaul stream as
  live startup readiness/performance work.
- Repository pulled through PR #846 at `d2eedb4e`.
- Bots were not restarted for PR #846 because the change is an observability-only
  event producer for an already-made pre-create skip decision, and the guard is
  only exercised when a create candidate is beyond the configured market-distance
  threshold. All five configured `passivbot live` processes remained running.
- PR #846 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice reuses the existing off-console
  `execution.create_skipped` route and existing best-effort create-filter
  emitter, so event sink/pipeline failures do not affect order filtering.
- A 5-minute read-only VPS5 smoke after the PR #846 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=d2eedb4e`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from known
  non-hard EMA readiness and HSL cooldown/status groups.
- Repository pulled through PR #835 at `17962106`.
- Bots were not restarted for PR #835 because the change is an observability-only
  structured event and report aggregation slice. All five configured
  `passivbot live` processes remained running.
- PR #835 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice emits `risk.realized_loss_gate_blocked` beside the
  existing throttled realized-loss-gate warning, routes the event away from
  console/text by default, and includes it in the read-only `risk_activity`
  performance-report summary using envelope labels only.
- A 5-minute smoke after the PR #835 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=17962106`, and all hard failure sources at zero. Remaining
  attention came from known non-hard structured problem events.
- Repository pulled through PR #833 at `ac495065`.
- Bots were not restarted for PRs #832/#833. PR #832 was a small CLI
  version-flag improvement; PR #833 was read-only performance-report tooling.
  All five configured `passivbot live` processes remained running.
- PR #833 passed the normal review gate after a rebase: Claude approved the
  current head, Hermes approved the current head, and CI was green. The slice
  adds value-safe risk/HSL/unstuck activity summaries to the read-only live
  performance report from existing structured event envelope labels only.
- A 5-minute smoke after the PR #833 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=ac495065`, and all hard failure sources at zero. Remaining
  attention came from known non-hard structured problem events.
- A 180-minute performance report after deploy confirmed the new
  `risk_activity` section populated on VPS5 with `total_events=606`,
  `event_types={"hsl.status": 587, "unstuck.status": 19}`,
  `total_groups=7`, and `bot_count=4`. Reported fields were bounded event
  labels/counts only; no risk/account numeric payload values were surfaced.
- Repository pulled through PR #829 at `fb2268af`.
- Bots were not restarted for PR #829 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #829 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice adds value-safe account-state activity summaries to
  the read-only live performance report from existing `fill.ingested`,
  `position.changed`, and `balance.changed` events only.
- A 5-minute smoke after the PR #829 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=fb2268af`, and all hard failure sources at zero. Remaining
  attention came from known non-hard structured problem events.
- Repository pulled through PR #826 at `eec38e60`.
- Bots were not restarted for PRs #826/#827 because the changes were docs-only
  and read-only performance-report tooling. All five configured
  `passivbot live` processes remained running.
- PR #827 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice adds bounded execution terminal outcome counts to the
  read-only live performance report from existing execution event types only.
- PR #826 passed the normal review gate after the reviewer nit was addressed:
  Claude approved the current rebased head, Hermes approved the same docs patch
  and dry-ran it into current `origin/v8`, and CI was green. The slice updates
  the logging guide debug-profile summary to match the current registry,
  including `forager`.
- A 5-minute smoke after the PR #827 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=38f3a9e3`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard structured problem events.
- A 5-minute smoke after the PR #826 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=eec38e60`, and all hard failure sources at zero. Remaining
  attention came from known non-hard structured problem events.
- Repository pulled through PR #824 at `d1b3ca04`.
- Bots were not restarted for PR #824 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #824 passed the normal review gate: Claude approved and carried approval
  over the clean rebase, Hermes approved the same own-delta, CI was green, and
  the final patch-id matched the reviewed pre-rebase patch. The slice derives
  startup debug-profile visibility from existing `bot.started` / `bot.ready`
  events only.
- A 5-minute smoke after the PR #824 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=d1b3ca04`, and `hsl_replay_health.active_bots=0`. Remaining
  attention came from known non-hard live readiness diagnostics.
- A 120-minute performance report after deploy confirmed the
  `startup_readiness.debug_profile_counts` field is present. No retained
  startup lifecycle rows were present in that current-window report, so the
  field was `{}` and no live debug profiles were inferred.
- Repository pulled through PR #822 at `09f145f4`.
- Bots were not restarted for PRs #821/#822 because the changes were docs and
  tests only. All five configured `passivbot live` processes remained running.
- PR #821 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice documents the stable `EventTypes` registry and adds a
  docs-sync test so future event-type changes cannot drift silently from
  `docs/ai/live_event_registry.md`.
- PR #822 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice documents `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` and
  adds a docs-sync test so future debug-profile additions/removals require a
  matching registry doc update.
- A 5-minute smoke after the PR #822 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=09f145f4`, and `hsl_replay_health.active_bots=0`. Remaining
  attention came from known non-hard live readiness diagnostics.
- Repository pulled through PR #819 at `c7bc5924`.
- Bots were not restarted for PR #819 because the change was read-only smoke
  report tooling. All five configured `passivbot live` processes remained
  running.
- PR #819 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice derives `hsl_replay_health` from existing
  `hsl.replay.*` monitor events and adds no event producers, exchange calls,
  cache mutation, readiness gates, order/risk logic, monitor writes, console
  routing, or trading behavior.
- A settled 10-minute smoke after the PR #819 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=c7bc5924`, `hsl_replay_health.active_bots=0`, and no
  missing configured processes. Remaining attention came from known non-hard
  EMA readiness groups.
- Wider 20- and 45-minute smokes intentionally captured startup/risk history:
  `hsl_replay_health` showed three completed coin-HSL replays, with Binance
  about `1.00M` applied rows in `1445s`, GateIO about `954k` rows in `1467s`,
  and OKX completing similarly slowly. The same wider windows showed HSL RED
  ZEC supervisor/finalized risk lines for Binance, GateIO, and OKX, not a
  software crash. This confirms the new smoke section works and reinforces that
  coin-HSL startup replay latency remains the highest-priority safety/perf gap.
- Repository pulled through PR #817 at `4dc6ef79`.
- Bots were not restarted for PR #817 because the change was read-only
  event-query tooling. All five configured `passivbot live` processes remained
  running.
- PR #817 passed local validation and CI; the slice adds `--tag` filtering to
  `passivbot tool live-event-query` and `build_event_report()` so event,
  timeline, trace-summary, order-trace, and cycle-trace reports can be scoped by
  structured live-event tags. It adds no event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, or trading
  behavior.
- PR #810 was merged at `ddf1f7fe` and carried snapshot/cycle IDs through
  `DiagnosticEvent` and `snapshot.built` diagnostic emission. This improves
  cycle/snapshot trace correlation for existing diagnostic events without
  changing trading behavior.
- PR #809 was an interleaved non-logging v8 configuration/defaults update,
  merged at `54e6b335`. It is noted here only because it advanced `v8` between
  logging slices.
- Repository pulled through PR #815 at `404063c6`.
- Bots were not restarted for PR #815 because the change was read-only
  event-query tooling. All five configured `passivbot live` processes remained
  running.
- PR #815 was merged under the documented degraded low-risk tooling gate after
  repeated Claude absence. Hermes approved current head `fe9f0fdc` with no
  findings and CI was green; the slice is query-only and derives source,
  component, tag, exchange, and user counters from already-persisted monitor
  event rows/envelopes. It adds no event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, or trading
  behavior.
- A 3-minute smoke after the PR #815 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state at `repository.head=404063c6`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard live readiness diagnostics.
- Repository pulled through PR #813 at `11c1a847`.
- Bots were not restarted for PR #813 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #813 was merged under the documented degraded low-risk tooling gate after
  repeated Claude absence. Hermes approved current head `a98c168ca` with no
  findings and CI was green; the slice is report-only, derives values from
  existing `snapshot.built` event data, and does not add event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior.
- A 5-minute smoke after the PR #813 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state at `repository.head=11c1a847`,
  `account_critical_remote_calls.failed=0`, and one non-hard general
  `remote_calls.failed=1`. Remaining attention came from known non-hard live
  readiness diagnostics.
- A focused 30-minute performance report confirmed the new
  `input_staleness.snapshot_market_stale_count` field populated on VPS5. It
  reported `snapshots_seen=151`, `snapshot_surface_age_rows=906`,
  `snapshot_market_summaries_seen=151`, `snapshot_market_stale_count=0`, and
  `total_groups=52`. The top groups were existing account/input age groups, so
  the market-snapshot excess-age projection added no new stale rows in that
  window.
- Repository pulled through PR #812 at `8e4712f6`.
- Bots were not restarted for PR #812 because the change was docs-only. All
  five configured `passivbot live` processes remained running.
- PR #812 was merged under the documented degraded low-risk docs gate after
  repeated Claude absence. Hermes approved current head `38da2ccf` with no
  findings and CI was green.
- A compact smoke after the PR #812 pull reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  remaining attention only from known non-hard EMA readiness events.
- Repository pulled through PR #811 at `52f85e08`.
- Bots were not restarted for PR #811 because the change was read-only
  event-query tooling. All five configured `passivbot live` processes remained
  running.
- PR #811 was merged under the documented degraded low-risk tooling gate after
  repeated Claude absence. Hermes approved current head `6ca8a602` with no
  findings and CI was green; the slice is query-only, adds no event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior.
- A 5-minute time-windowed smoke after the PR #811 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state at
  `repository.head=52f85e08`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from
  known non-hard EMA readiness problem events.
- Repository pulled through PR #807 at `2bb89cfa`.
- Bots were not restarted for PR #807 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #807 passed the normal review gate: Claude approved, Hermes approved, and
  CI was green. The slice corrected `snapshot_to_rust` report correlation only;
  no event producers, exchange calls, cache mutation, readiness gates, console
  routing, or trading behavior changed.
- A 10-minute time-windowed smoke after the PR #807 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state at
  `repository.head=2bb89cfa`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard EMA readiness and HSL cooldown
  groups.
- A focused 30-minute performance report confirmed the corrected
  `snapshot_to_rust` values on VPS5: Binance p95 `1449ms`, GateIO p95
  `1477ms`, OKX p95 `1712ms`, and Hyperliquid p95 `564ms`. The same report
  showed `snapshot_to_rust_latest_snapshot_matches=154`,
  `snapshot_to_rust_exact_matches=0`, and one missing snapshot match at the
  time-window boundary.
- Repository pulled through PR #805 at `3d6e3fa7`.
- Bots were not restarted for PR #805 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #805 was merged under the documented degraded low-risk tooling gate after
  repeated Claude absence. Hermes approved current head `4573fe59` with no
  findings and CI was green; the slice is report-only, derives values from
  already-collected timing groups, and does not add event producers, exchange
  calls, cache mutation, readiness gates, console routing, or trading behavior.
- A 5-minute time-windowed smoke after the PR #805 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state at
  `repository.head=3d6e3fa7`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard EMA readiness and HSL cooldown
  groups.
- A focused 30-minute performance report confirmed the new
  `operation_durations` section populated on VPS5. It reported `total_groups=161`
  across cache, cycle, decision-boundary, input-staleness, remote-call, and
  state-refresh categories. The top groups showed `input_staleness.snapshot_to_rust`
  as the largest observed delay in that window, with the expected
  `delays_cycle_decision` blocking scope. No raw payloads, account values, or
  exchange response bodies were surfaced.
- Repository pulled through PR #803 at `07f8e759`.
- Bots were not restarted for PR #803 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- PR #803 was merged under the documented degraded low-risk tooling gate after
  repeated Claude absence. Hermes approved current head `80bc42fd` with no
  findings and CI was green; the slice is report-only, derives values from
  existing `health.summary` events, and does not add event producers, exchange
  calls, cache mutation, readiness gates, console routing, or trading behavior.
- A 5-minute time-windowed smoke after the PR #803 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state at
  `repository.head=07f8e759`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard EMA readiness and HSL cooldown
  groups.
- A focused 30-minute performance report confirmed the updated
  `resource_pressure` section populated on VPS5. It reported `total=8` health
  summary events across four bots, and sample groups for Hyperliquid and GateIO
  showed whitelisted process/event-pipeline fields with `count`, `latest`,
  `min`, `mean`, `median`, `p95`, and `max` values, including RSS, load
  averages, loop duration, event queue depth, sink-error totals, dropped-event
  totals, and worker state. No raw account or financial payload fields were
  surfaced.
- Repository pulled through PR #801 at `1fc77413`.
- Bots were not restarted for PR #801 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- A 10-minute time-windowed smoke after the PR #801 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, `processes.ok=true`, and clean tracked repository
  state at `repository.head=1fc77413`. The same window still showed non-hard
  EMA readiness and HSL cooldown attention plus one recovered non-hard Kucoin
  candle timeout; account-critical remote calls had `failed=0`.
- A focused 10-minute performance report confirmed the new
  `forager_ema_readiness` section populated on VPS5. It reported
  `total_events=140`, with `ema.fallback_used=41`, `ema.unavailable=56`, and
  `forager.selection=43`, grouped across Binance, GateIO, OKX, and
  Hyperliquid. Sample groups showed bounded forager selection counts,
  selected-symbol samples, EMA unavailable reason counters, and EMA fallback
  counters without raw EMA error text or account/cache payloads.
- Repository pulled through PR #799 at `cb034e82`.
- Bots were not restarted for PR #799 because the change was read-only
  performance-report tooling. All five configured `passivbot live` processes
  remained running.
- A 2-minute smoke after the PR #799 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. The same window still showed non-hard EMA readiness
  and HSL cooldown attention; those are live-state diagnostics, not deploy
  hard failures.
- A focused 5-minute performance report confirmed the new `cache_warmup`
  section populated on VPS5 for all five bots. The section reported
  `total_events=614`, with `cache.load.completed=469`,
  `cache.flush.completed=140`, and `cache.warmup_decision=5`. Sample groups
  showed OKX, Binance, and GateIO with bounded candle load/flush rows,
  source/reason counters, warmup cold-path decisions, and elapsed summaries.
- Repository pulled through PR #797 at `87f22840`.
- Bots were not restarted for PRs #796/#797 because the changes were docs and
  read-only performance-report tooling. All five configured `passivbot live`
  processes remained running.
- A 2-minute smoke after the PR #797 pull reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected_count=0`.
- A focused 45-minute performance report confirmed the new
  `hsl_replay_profile` section populated on VPS5. It showed four active HSL
  replay profiles: GateIO about `1.25M` estimated dense pair-rows and `56.2%`
  observed work, Binance about `1.12M` and `62.4%`, OKX about `1.17M` and
  `62.9%`, and Kucoin with only the start event in the sampled window. This
  validates the report slice and re-confirms that coin-HSL full replay remains
  the dominant startup safety/performance gap.
- Repository pulled through PR #696 at `d850daf5`.
- Bots were restarted from `/root/bots_vps5.yaml` after PR #677 and left
  running. The old process set stopped after about 36 seconds before the tmuxp
  reload.
- `tmuxp load` returned a nonzero attach error because SSH had no terminal, but
  it created the `passivbot` session and all five configured bot processes.
- VPS5 post-restart smoke with text logs disabled and
  `--recent-minutes 2 --supervisor-config /root/bots_vps5.yaml` reported
  `ok=true`, `hard_failures=0`, `hard_problem_event_count=0`,
  `expected_total=5`, `matched_expected=5`, and `missing_expected=[]`.
- A wider post-restart smoke saw a real GateIO HSL ZEC long RED finalization
  during startup/replay and one non-hard Kucoin candle `RequestTimeout`; the
  narrower settled-window smoke confirmed no continuing hard failures.
- PRs #670 and #671 were pulled to VPS5 without bot restart because they only
  changed read-only smoke-report tooling. A later smoke with text logs enabled
  and `--recent-minutes 2 --supervisor-config /root/bots_vps5.yaml` reported
  `ok=true`, `hard_failures=0`, `hard_problem_event_count=0`,
  `logs.hard_matches=0`, `matched_expected=5`, and `missing_expected=[]`.
- PR #673 was also pulled to VPS5 without bot restart. A later 5-minute smoke
  with text logs and process matching enabled reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `remote_call_failures.total=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new `risk_events` section surfaced GateIO ZEC long
  HSL RED cooldown directly as `hsl.status` `cooldown_active`.
- PR #675 was pulled to VPS5 without bot restart. A 2-minute smoke with
  `--log-window-unparsed-policy drop` and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `remote_call_failures.total=0`, `matched_expected=5`, and
  `missing_expected=[]`.
- PR #677 was pulled to VPS5 and bots were restarted. Repeated settled
  2-minute smokes with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `matched_expected=5`, and `missing_expected=[]`. One intermediate Kucoin
  authoritative account-refresh timeout was visible in structured events and
  recovered without intervention. Remaining smoke attention came from non-hard
  EMA readiness / staged-execution degradation and GateIO ZEC HSL cooldown, not
  from the PR #677 logging migration.
- PRs #678 and #679 were pulled to VPS5 without bot restart because they only
  changed docs and read-only smoke-report tooling. A later 2-minute smoke with
  `--log-window-unparsed-policy drop`, text logs enabled, and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. One non-hard Kucoin candle `RequestTimeout` remained
  visible in `remote_call_failures`. The smoke output now includes bounded
  `problem_events.latest_data` for persistent non-hard `ema.unavailable` and
  `cycle.degraded` groups, and stale unparseable traceback/header fragments are
  filtered by timestamp context instead of causing false recent-window matches.
- PR #680 was pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #682 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The first smoke after deploy surfaced a real
  transient Kucoin authoritative account-refresh `RequestTimeout` as one hard
  problem group. A later settled 2-minute smoke with
  `--log-window-unparsed-policy drop`, text logs enabled, and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `matched_expected=5`, and `missing_expected=[]`. The new
  `problem_event_groups` section grouped the remaining known non-hard
  EMA/cycle/HSL attention without requiring inspection of individual event
  samples.
- PR #683 was pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #684 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The pre-fix smoke showed the tool could still
  hard-fail on a stale Kucoin traceback header when the tailed log slice started
  in the middle of an old traceback with no timestamp context. After PR #684, a
  1-minute smoke with `--log-window-unparsed-policy drop`, text logs enabled,
  and `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `matched_expected=5`, and `missing_expected=[]`.
- PR #686 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The smoke report confirmed
  `repository.branch=v8`, `repository.head=9e898019`,
  `repository.dirty=false`, `tracked_changes=0`, and all five configured bots
  running. The same smoke surfaced repeated Kucoin authoritative REST
  `RequestTimeout` events unrelated to the tooling change.
- PR #688 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `remote_call_timings.total=637`, `matched_expected=5`, and
  `missing_expected=[]`. The new timing groups surfaced slow-but-successful
  candle remote fetches, including Kucoin `M/USDT:USDT` p95/max 48047ms.
- PR #690 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `remote_call_health.total=445`, `matched_expected=5`, and
  `missing_expected=[]`. The new health groups showed the highest-volume
  terminal remote-call categories by bot/component/kind/surface, including
  slow-but-successful Binance authoritative fill fetches and candle fetch
  groups.
- PR #692 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new top-level `remote_call_health` summary was
  present with `total=390`, `succeeded=389`, `failed=1`, `throttled=0`,
  `failure_pct=0`, and `throttled_pct=0`.
- PR #694 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new `account_critical_remote_call_health`
  summary was present with `total=126`, `succeeded=126`, `failed=0`,
  `throttled=0`, `failure_pct=0`, and `throttled_pct=0`.
- PR #696 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 2-minute compact summary smoke with text
  logs, `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml`
  process matching reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `logs.attention_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, `remote_calls.total=169`, `remote_calls.succeeded=169`,
  `account_critical_remote_calls.total=58`, and
  `account_critical_remote_calls.succeeded=58`. Remaining `attention=true`
  came from known non-hard EMA readiness / staged-execution / HSL cooldown
  groups.
- PR #681 was merged after Claude + Hermes approval and green CI, then deployed
  to VPS5 with a bot restart because it added live state-refresh event
  producers. The final rebase changed only progress-doc context relative to
  Claude's reviewed code. The restart left all five configured bots running;
  Kucoin needed a second Ctrl+C before exit. A settled 2-minute smoke reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `account_critical_remote_calls.total=35`, `succeeded=35`,
  `remote_calls.total=321`, `succeeded=321`, `matched_expected=5`, and
  `missing_expected=[]`. A direct event query confirmed
  `state.refresh_timing` and `state.refresh_progress` events on VPS5. A wider
  smoke also surfaced a real GateIO ZEC long HSL RED finalization during
  startup/replay, unrelated to the state-refresh event slice.
- PRs #698 and #699 were merged after current-head Claude + Hermes approval and
  green CI, then pulled to VPS5 without bot restart because they only changed
  read-only smoke-report tooling. Immediate post-pull smokes caught real
  transient Kucoin authoritative `RequestTimeout` bursts at 13:16Z and 13:19Z;
  a later settled 2-minute compact summary smoke on `d5639813` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, `missing_expected=[]`,
  `account_critical_remote_calls.total=81`, `succeeded=81`,
  `remote_calls.total=230`, and `succeeded=230`. Remaining attention came from
  known non-hard EMA/staged readiness and GateIO ZEC HSL cooldown groups.
- PR #701 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 without bot restart because it only changed read-only
  probe tooling and docs. The pull left all five configured bots running and
  repository state clean at `91d5db9f`. Immediate smoke was red from real,
  unrelated Kucoin live issues: authoritative balance/positions/open-orders
  timeouts, a timestamp/nonce recovery, and a later fill-refresh timeout. The
  new probe summary was exercised with a one-repeat authenticated
  `ticker-endpoint-probe` on `binance_01`; it produced
  `account_critical_health.total=3`, `succeeded=2`, `failed=1`, with balance
  and positions successful and `fetch_open_orders()` failing as `ExchangeError`.
  This validates the summary projection and exposes a follow-up for
  lower-impact/account-only probing and exchange-specific open-orders shape.
- PR #703 was merged after Claude + Hermes approval on the code delta, green CI,
  and a docs-only amendment that clarified `--account-only` still loads markets
  for open-orders symbol fallback. It was pulled to VPS5 without bot restart.
  A one-repeat authenticated `ticker-endpoint-probe --account-only` on
  `binance_01` reported `account_critical_health.total=3`, `succeeded=3`,
  `failed=0`; Binance `fetch_open_orders()` first failed as `ExchangeError`,
  then succeeded through `mode=symbol_fallback`. A follow-up 2-minute smoke at
  `a2f93456` reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `matched_expected=5`, and `missing_expected=[]`.
- PR #704 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only updated progress docs. A compact
  smoke at `65589e71` reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`.
- After `/root/bots_vps5.yaml` was updated to run the new HSL forager config on
  Binance, Kucoin, GateIO, and OKX while leaving `hyperliquid_tradfi`
  unchanged, a 10-minute smoke at `65589e71` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. One non-hard Kucoin candle `RequestTimeout` recovered
  in subsequent candle calls. Remaining attention came from known non-hard
  Hyperliquid tradfi EMA/staged readiness events.
- PR #705 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changed read-only smoke-report
  tooling. A 10-minute `--brief` smoke at `4e752000` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=606`, `remote_calls.failed=1`,
  `account_critical_remote_calls.total=90`, and
  `account_critical_remote_calls.failed=0`. The single remote failure was a
  non-hard Kucoin candle timeout; no account-critical failure or hard problem
  event was present.
- PR #706 was merged under the low-risk docs gate after Hermes approval and
  green CI, then pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #707 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 and deployed with a bot restart because it changed live
  HSL console projection. The first Ctrl+C round stopped all five bot panes
  cleanly within 20 seconds; `tmuxp load -d /root/bots_vps5.yaml` then started
  all five configured bots. A settled 5-minute `--brief` smoke at `f384edbe`
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=329`, `remote_calls.failed=0`,
  `account_critical_remote_calls.total=57`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from known
  non-hard Hyperliquid tradfi EMA/staged readiness events. No `hsl.status`
  risk events occurred in the 5-minute smoke window, so the new coin-HSL console
  projection was not exercised on VPS5 yet.
- PR #709 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 and deployed with a bot restart because it adds startup
  fill-cache structured events. The first Ctrl+C round stopped Binance, but
  Kucoin, GateIO, OKX, and Hyperliquid remained as orphaned live processes after
  two Ctrl+C rounds; SIGTERM cleared them before reload. `tmuxp load -d
  /root/bots_vps5.yaml` then started all five configured bots. A settled
  5-minute `--brief` smoke at `71479c61` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=186`, `remote_calls.failed=0`,
  `account_critical_remote_calls.total=30`, and
  `account_critical_remote_calls.failed=0`. Direct monitor-file checks found
  `fills.refresh_summary` `fill_cache_ready` events for all five bots. Remaining
  attention came from known non-hard Hyperliquid tradfi EMA/candle readiness.
- PR #711 was merged after current-head Claude + Hermes approval and green CI,
  then deployed to VPS5 with a bot restart because it adds an exchange
  time-sync live event producer. During shutdown, the first Ctrl+C round stopped
  Binance only; a second Ctrl+C stopped OKX and Hyperliquid, while Kucoin and
  GateIO needed SIGTERM before reload. A settled 5-minute `--brief` smoke at
  `0fa6269b` reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, `remote_calls.failed=0`,
  and `account_critical_remote_calls.failed=0`. Remaining attention came from
  known non-hard Hyperliquid tradfi EMA/cycle readiness.
- PR #712 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 without bot restart because it only changes read-only
  smoke-report tooling. A settled 5-minute `--brief`/`--summary` smoke at
  `51ba92a3` reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`. The
  new process diagnostics reported
  `classification_source=local_process_table_command_match`,
  `tmux_pane_ownership=not_available_from_process_table`,
  `duplicate_configured_command_matches_count=0`, and
  `extra_passivbot_live_processes_count=0`.
- PRs #714 and #715 were merged under the low-risk tooling/docs degraded gate
  after repeated Claude absence. Hermes had approved the PRs before the final
  current-base rebases, CI was green on the rebased heads, and local focused
  tests passed after conflict resolution. They were pulled to VPS5 without bot
  restart because they only changed read-only operator tooling and docs. A
  settled 5-minute `--brief` smoke at `7b12d4b2` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `repository.dirty=false`, and zero
  duplicate/extra live process matches. A bounded summary showed only known
  non-hard EMA readiness, HSL cooldown, and staged-readiness attention. The new
  `shutdown_events` section was present with `total=0` because no bot restart
  occurred. A read-only `live-config-preflight` smoke against
  `configs/forager_3pos_hsl_2026-06-26.json` returned `ok=true` with one
  warning for missing short-side bot config, and did not contact exchanges.
- PRs #719 and #720 were merged under the low-risk tooling/observability
  degraded gate after Hermes approval, green CI, and local focused validation.
  Claude did not return before merge. #719 added a plan-only restart-smoke
  planner and #720 added the opt-in `ema` live-event debug profile. Both were
  deployed to VPS5 by fast-forward pull without bot restart because #719 is
  read-only operator planning and #720 is opt-in structured event enrichment.
  A 10-minute brief smoke at `27f597be` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `repository.dirty=false`, and no
  duplicate/extra live process matches. Remaining attention came from existing
  non-hard structured problem events and HSL status. A VPS5 dry-run
  `live-restart-smoke-plan` against `/root/bots_vps5.yaml` reported `ok=true`,
  `dry_run=true`, `execution_available=false`, five planned bots, no issues,
  and rejected operations including SSH, tmux signaling, process signaling, git
  checkout/pull/fetch, `passivbot live`, exchange/API calls, and credential
  loading.
- PRs #723-#724 and #728/#730/#732 added opt-in debug-profile enrichment for
  remote calls, candles, fills, HSL, and execution events. They were deployed
  to VPS5 without bot restart where appropriate because profiles are opt-in and
  default live behavior and console volume were unchanged. Settled brief smoke
  reports after those pulls showed all five expected bots running with no hard
  failures, no log hard matches, no failed account-critical remote calls, and
  only known non-hard EMA/staged-readiness or HSL attention.
- Claude's retrospective audit for already-merged PRs #713-#722 was posted on
  PR #723. Current `v8` contains follow-up coverage: PR #726 addressed
  shutdown-message echo, EMA debug-profile scoping, and Rust debug best-effort
  isolation; PR #734 addressed shareable path hygiene, flat config reporting,
  and HSL preview scalar/path hardening; and the cache-doctor candle coverage
  path now reports length mismatches using canonical row-count authority.
- PR #735 was pulled to VPS5 without bot restart because it only changed
  read-only startup-budget smoke projections. A later smoke initially surfaced
  real HSL RED events and one OKX active-symbol forager EMA readiness handoff
  error unrelated to the smoke-tooling change; a settled follow-up smoke
  reported `ok=true`, no hard failures, no log hard matches, all five expected
  bots matched, and no failed remote/account-critical calls. PR #736 recorded
  the EMA-readiness handoff gap in the live-ops backlog.
- PR #737 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 and deployed with a bot restart because it changed live forager EMA
  readiness behavior. The first Ctrl+C round left Kucoin running; a follow-up
  SIGINT stopped it, but the restart shell killed itself because the process
  match also matched the shell command. The stale tmux session was then killed
  and `/root/bots_vps5.yaml` was reloaded, leaving all five configured bots
  running. Settled 2-minute and 5-minute smokes at `e3429ee9` both reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- PR #739 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes plan-only restart tooling
  and docs. A settled 5-minute smoke at `87e53dac` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- PR #741 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes read-only active probe
  tooling and docs. A 5-minute compact smoke at `d4c28058` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`,
  `account_critical_remote_calls.failed=0`, and clean tracked repository
  state. A one-repeat `ticker-endpoint-probe --account-only` on `binance_01`
  validated the new `time_sync_health` output with `total=1`, `succeeded=1`,
  `failed=0`, `unsupported=0`, and `max_abs_clock_skew_ms=14`.
- PR #743 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes read-only active probe
  projections and docs. A 5-minute compact smoke at `1fe1292b` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and clean tracked repository
  state. A one-repeat public-only `ticker-endpoint-probe` on `binance_01` for
  `BTC/USDT:USDT` validated `candle_freshness_health.total_symbols=1`,
  `succeeded_symbols=1`, `failed_symbols=0`, `current_incomplete_symbols=1`,
  and `worst_symbol=BTC/USDT:USDT`.
- PR #745 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes read-only active probe
  projections and docs. A compact smoke at `4130155e` reported `ok=true`,
  `hard_failures=0`, all five expected bots running, clean tracked repository
  state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. A one-repeat authenticated
  `ticker-endpoint-probe` on `binance_01` for `BTC/USDT:USDT`, with
  order-book/OHLCV/time-sync probes skipped, validated
  `fill_history_health.total=1`, `succeeded=1`, `failed=0`,
  `latest_symbol=BTC/USDT:USDT`, and `latest_trade_count=0`.
- PR #747 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes read-only active probe
  projections and docs. A brief smoke at `74270454` reported `ok=true`,
  `hard_failures=0`, all five expected bots running, clean tracked repository
  state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. A one-repeat authenticated
  `ticker-endpoint-probe` on `binance_01` for `BTC/USDT:USDT`, with
  order-book/OHLCV/time-sync probes skipped, validated
  `rate_limit_health.observed_call_count=12`, `public_call_count=6`,
  `private_call_count=5`, `concurrent_request_count=1`,
  `exchange_rate_limit_ms=50`, and `estimated_min_serial_ms=600`.
- PR #778 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only adds read-only performance-report
  tooling and docs. A 5-minute summary smoke at `fe946c6c` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected=[]`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  `passivbot tool live-performance-report --recent-minutes 180 --compact`
  also returned `ok=true`, `issues=[]`, and redacted `~/passivbot/...` paths.
  Its slowest groups made the current startup/HSL latency explicit:
  OKX full-warmup about `2552s`, Binance full-warmup about `2419s`,
  GateIO full-warmup about `2409s`, and Binance `startup.hsl` about `2017s`.
- PR #779 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only extends read-only
  performance-report tooling and docs. A filtered VPS5 performance summary for
  `--exchange binance` returned `ok=true`, `filters.exchanges=["binance"]`,
  `events_skipped=11699`, and bounded slowest groups led by Binance
  `startup.full-warmup` about `2419s`, `startup.market` about `2134s`,
  `startup.hsl` about `2017s`, and `hsl_replay.elapsed` about `1503s`.
  A 5-minute summary smoke at `0d742a4f` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, all five expected bots matched,
  no account-critical remote-call failures, and one non-hard OKX candle
  timeout visible in `remote_calls`.
- PR #780 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only extends read-only
  performance-report tooling and docs. A filtered VPS5 performance summary for
  `--exchange binance` returned `ok=true`, `decision_boundary_lag.cycles=7`,
  `cycles_with_write=0`, and bounded lag groups led by
  `decision_boundary.cycle_completed` p95 about `98s`,
  `action_planned`/`rust_returned` p95 about `93s`, and `cycle_started` p95
  about `53s`. A 5-minute summary smoke at `f70434f3` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, all five expected bots matched,
  no failed remote calls, and no account-critical remote-call failures.
- PR #781 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only extends read-only
  performance-report tooling and docs. The new `input_staleness` section was
  present in a filtered Binance report and showed high snapshot-to-Rust age
  directly: `snapshot_to_rust` p95 about `192s`, account packet age at snapshot
  p95 about `38-42s`, and EMA-bundle age p95 about `5s`. The first smoke after
  deploy caught a real transient GateIO authoritative positions
  `RequestTimeout`; a later settled 2-minute smoke at `cdb2f381` reported
  `ok=true`, `hard_failures=0`, all five expected bots matched, no text-log
  hard matches, no failed remote calls, and no failed account-critical remote
  calls.
- PR #782 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 with a bot restart because it changes live HSL replay event
  producers. Shutdown was bounded but Kucoin/GateIO required a second Ctrl+C;
  the session was reloaded from `/root/bots_vps5.yaml` and all five expected
  bots were left running. Immediate and settled smokes at `b5e08986` reported
  `ok=true`, `hard_failures=0`, no text-log hard matches, no failed remote
  calls, and no failed account-critical remote calls. A direct
  `live-event-query` confirmed new `hsl.replay.progress` fields on VPS5,
  including `held_pairs`, `cooldown_pairs`, `required_pairs`,
  `timeline_rows`, `applied_rows`, `total_applied_rows`, `rows_per_second`,
  `is_held_pair`, and `is_cooldown_pair`.

## Phase Checklist

| Area | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| Phase 0: event contract and routing basics | Done enough to build on | `LiveEvent`, routing, pipeline, monitor-backed sink, schema/query constants | Keep registry stable; avoid ad-hoc event names in new slices |
| Phase 1: event bus around existing structured events | Mostly done | Cycle, data packet, snapshot, planning unavailable, Rust orchestrator, order wave, fill/state events | Continue tightening tests as new producers migrate |
| Phase 2: data gatherer events | Mostly done | Account remote-call cohorts, candle tail/coverage, fill refresh summaries, cache load/flush, warmup/startup timing | Not every exchange/network call is instrumented; richer remote-call payload summaries remain incremental |
| Phase 3: Rust planning and payload refs | Partially done | Rust orchestrator called/returned events, redacted error hardening, action/planning summaries | Full raw-ref retention/debug policy still limited |
| Phase 4: order lifecycle and risk transitions | Mostly done | Order wave lifecycle, create/cancel/confirmation events, HSL/risk mode events, HSL replay failure events | Expand WEL/TWEL/unstuck transition coverage as those paths are touched |
| Phase 5: migrate meaningful text logs | Partially started | Some noisy EMA console output already reduced; PR #646 improves event-projected console summaries for already-routed execution events; PR #707 restores throttled coin-mode HSL position status console lines from existing `hsl.status` metrics; PR #709 mirrors fill-cache startup readiness into off-console `fills.refresh_summary` events; PR #711 mirrors CCXT timestamp/nonce recovery into off-console `exchange.time_sync` events; PR #846 mirrors pre-create market-distance guard skips into off-console `execution.create_skipped` events; PR #848 mirrors pre-create planning/market snapshot skips into off-console `execution.create_skipped` events; PR #850 mirrors fill-cache doctor startup report/quarantine/rebuild decisions into off-console `fills.refresh_summary` events | Migrate high-value stdlib logs to structured-event projections without increasing console noise |
| Phase 6: gatekeeper integration | Pending | Gatekeeper remains a planned producer | Instrument gate decisions once gatekeeper work resumes |
| Operator tools | In progress | `live-event-query`, trace summaries, order trace reconstruction, cycle trace reconstruction, time-window filters, severity-level filters, problem-event filters, event-file discovery metadata, `live-smoke-report` startup baselines/process liveness/remote-call failures/remote-call timings/remote-call health groups and top-level totals/account-critical health/risk-events/execution-health/shutdown-events/time windows/unparseable-log policy/brief smoke counters/brief problem-event groups/supervisor duplicate-extra process diagnostics, incident bundle trace/process/time-window/problem-event reports and query-scope filters, ID filters, `ticker-endpoint-probe` account-critical/time-sync/candle-freshness/fill-history-sample/rate-limit health summaries and account-only mode, `live-config-preflight` offline config summaries, `live-performance-report` timing aggregation with summary/filter, decision-boundary, input-staleness including market snapshot staleness, startup phase timing summaries, HSL replay pair/rate/stage summaries, forager/EMA readiness, cache warmup, resource-pressure percentiles, and unified operation-duration support | Cross-bot incident workflow, safe restart orchestration, bounded historical performance-report scans, active probe expansion beyond current endpoint/freshness summaries |
| Operational restart goals | Split to adjacent work | PR #619 shutdown progress; PR #622 warm-cache startup; PR #656/#668 cache integrity smoke doctor | Continue separate reviewed PRs for shutdown/warmup/cache proof improvements |

## Current Work

### In Progress: HSL Replay ETA Smoke Report

- Branch: `codex/v8-smoke-hsl-replay-eta`.
- Scope: read-only live smoke report projection and tests.
- Result: `live-smoke-report` derives the same HSL replay remaining row and ETA
  estimates from existing sanitized `hsl.replay.*` monitor events, and exposes
  max active primary remaining rows/time in the compact brief summary. The full
  `hsl_replay_health` groups keep conservative dense estimates plus
  required-pair estimates when available.
- Expected validation: focused and full live-smoke-report tests,
  py_compile, `git diff --check`, and an added-line silent-handling scan. No
  event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, process signaling, smoke verdict policy, order/risk
  logic, or trading behavior should change.

## Merged Slices

### PR #979: HSL Replay ETA Performance Report

- Branch: `codex/v8-hsl-replay-eta-report`.
- Scope: read-only live performance report projection and tests.
- Result: `live-performance-report` derives HSL replay remaining row estimates
  and ETA from existing sanitized `hsl.replay.*` monitor events. The report
  keeps the existing conservative dense pair-row estimate and adds a
  required-pair estimate when `required_pairs` is present, using
  `rows_per_second` only when the event already supplies a positive finite rate.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered the full live-performance-report test file,
  py_compile, `git diff --check`, an added-line silent-handling scan, and a
  local compact CLI smoke.
- VPS5 evidence: deployed to `v8` at `b35becbc` without bot restart because the
  change is read-only tooling. Smoke stayed hard-green with all five configured
  bots running. The deployed performance report showed new remaining/ETA fields
  for active Binance, GateIO, and OKX HSL pair replay; Kucoin was still in
  price-history symbol fetch, so no row-rate ETA was available there.

### PR #976: Trailing Status Risk Activity Report

- Branch: `codex/v8-live-performance-trailing-risk-activity`.
- Scope: read-only live performance report projection and tests.
- Result: `live-performance-report` includes existing `trailing.status` events
  in the bounded `risk_activity` section, so trailing/waiting position state can
  be found beside HSL and unstuck risk-state events without opening the full
  event stream. The projection uses event-envelope labels and bounded symbol
  samples only; threshold/retracement prices and detailed event payload values
  remain out of the shareable report.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered the full live-performance-report test file,
  py_compile, `git diff --check`, and an added-line silent-handling scan.
- VPS5 evidence: deployed to `v8` at `1823d3e1` with all five configured bots
  restarted and left running. Repeated smoke checks stayed hard-green with no
  failed remote calls or account-critical remote calls. The final performance
  report showed `risk_activity` populated with deployed `trailing.status`
  events.

### PR #924: Startup Phase Readiness Summary

- Branch: `codex/v8-startup-readiness-summary`.
- Scope: read-only live performance report tooling and tests.
- Result: `passivbot tool live-performance-report` now adds bounded aggregate
  startup phase timing counters to `startup_readiness`, so operators can see
  cross-bot startup phase elapsed and since-previous summaries without opening
  every per-bot phase row. Startup phase labels are whitelisted before they
  appear in either `startup_readiness` or `operation_durations`.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head
  after the Hermes finding about raw startup phase labels in
  `operation_durations` was fixed. Local validation covered focused
  live-performance-report tests plus py_compile and `git diff --check`. No
  event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, order/risk logic, or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `652b019e` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, no failed remote
  calls, and no failed account-critical remote calls. Current monitor segments
  had no startup timing rows, so the new `startup_readiness` aggregate fields
  were present but empty.

### PR #923: Market Snapshot Staleness Summary

- Branch: `codex/v8-market-snapshot-staleness-summary`.
- Scope: read-only live performance report tooling and tests.
- Result: `passivbot tool live-performance-report` now adds aggregate market
  snapshot staleness counters to `input_staleness`, so operators can see
  snapshot observations, missing symbol totals, configured-age excess, bounded
  source labels, and age/count summaries without opening each `snapshot.built`
  event.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused live-performance-report tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `31ef4d40` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, no failed remote
  calls, no failed account-critical remote calls, and
  `live-performance-report --summary` showed
  `input_staleness.market_snapshot` from live monitor data.

### PR #922: HSL Replay Profile Stage Summary

- Branch: `codex/v8-hsl-replay-profile-stage-summary`.
- Scope: read-only live performance report tooling and tests.
- Result: `passivbot tool live-performance-report` now adds aggregate HSL
  replay stage/status counters to `hsl_replay_profile`, so operators can see
  active/completed/failed replay bots and current active replay stages without
  manually scanning per-bot groups.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused live-performance-report tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `82b1990c` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, and
  `live-performance-report --summary` showed the new HSL replay aggregate
  counters from existing monitor data.

### PR #921: Incident Bundle Problem Report Discovery Summary

- Branch: `codex/v8-incident-problem-discovery-summary`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now projects
  `problem_event_report.file_discovery` and
  `problem_event_report.event_window` into the compact incident-bundle result
  so scoped problem-event queries can be verified without opening the archive.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused incident-bundle and event-query tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `5808f679` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, and focused OKX
  incident-bundle output showing `problem_event_report.files_scanned=1` plus
  path-pruned discovery/window metadata.

### PR #920: Incident Bundle Time-Window Discovery Summary

- Branch: `codex/v8-incident-window-discovery-summary`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now projects
  `time_window.files_scanned` and
  `time_window.file_discovery` into the compact incident-bundle result so
  focused bundle scoping can be verified without opening the archive.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused incident-bundle and event-query tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `570875ab` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, and focused OKX
  incident-bundle output showing `time_window.files_scanned=1` plus path-pruned
  discovery metadata.

### PR #919: Incident Bundle Time-Window Scope Filters

- Branch: `codex/v8-incident-window-scopes`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` applies query scope
  filters to `time_window_report.json`, `timeline.txt`, and matched event
  segment selection, while preserving the existing behavior where a cycle-id
  bundle keeps surrounding time-window context instead of filtering the window
  down to that cycle only.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.
- Review evidence: Hermes, Claude, Cursor, and CI approved the branch.
- VPS5 evidence: deployed at `989b81c9` without bot restart because this is
  read-only tooling. Smoke stayed hard-green with all five bots matched, and a
  focused OKX bundle verified scoped time-window paths, exchange/user values,
  timeline, and event-segment selection from the live monitor tree.

### PR #918: Incident Bundle Query Scope Filters

- Branch: `codex/v8-incident-query-scopes`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now exposes additional
  event-query scope filters for the bundled event reports:
  `--level`, `--exchange`, `--user`, `--bot-id`,
  `--remote-call-group-id`, `--side`, `--source`, `--component`, `--tag`, and
  `--data-eq`. The filters are passed to both `event_report.json` and
  `problem_event_report.json`, recorded in `manifest.json`, and preserve the
  existing smoke/process/log behavior.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.
- Review evidence: Hermes, Claude, Cursor, and CI approved the branch.
- VPS5 evidence: deployed at `946d0757` without bot restart because this is
  read-only incident-bundle tooling. Smoke stayed hard-green with all five bots
  matched, and a focused OKX bundle verified scoped event/problem reports from
  the live monitor tree. That smoke also showed the remaining gap addressed by
  the next slice: the time-window report still scanned broader root context.

### PR #917: Incident Bundle Problem Event Report

- Branch: `codex/v8-incident-problem-query`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now embeds
  `problem_event_report.json` by default. The report is built with the shared
  `live-event-query --problem-events` predicate, honors the bundle's existing
  cycle/symbol/status/reason/time/tail filters, includes a trace summary, and
  can be disabled with `--no-problem-report` for compact bundles. Event segment
  selection now considers the problem-event report too, so bundles keep the raw
  segment needed to reconstruct smoke attention rows.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.

### Foundation Before PR #619

- Added the first live event pipeline pieces and monitor projections.
- Emitted structured events for forager/EMA summaries, planning defer summaries,
  HSL replay lifecycle, planning symbol state, EMA bundle starts, planned actions,
  and confirmation timeouts.
- These commits established the initial cycle/planning/execution event chain used
  by later query and smoke tooling.

### PR #619: Shutdown Progress

- Branch: `codex/v8-shutdown-progress`.
- Scope: adjacent operations improvement, not logging core.
- Result: improved shutdown progress and bounded shutdown cancel grace coverage.
- Follow-up: continue shutdown interruption work outside logging-only PRs.

### PR #621: Live Event Query Helper

- Branch: `codex/v8-live-event-query-helper`.
- Scope: shared live event query schema constants and initial query helper.
- Result: provided a stable base for later CLI filters and incident tooling.

### PR #622: Startup Warm Cache

- Branch: `codex/v8-startup-warm-cache`.
- Scope: adjacent operations improvement.
- Result: improved live startup warm-cache reuse.
- Follow-up: continue cache proof and warmup optimization separately from
  observability-only slices.

### PR #623: Live Event Query Scope

- Branch: `codex/v8-live-event-query-scope`.
- Scope: bounded live event query directory scans and rotated scan defaults.
- Result: query helper became safer on VPS-sized monitor trees.

### PR #624: EMA Console Noise

- Branch: `codex/v8-ema-console-noise`.
- Scope: console cleanup.
- Result: reduced forager EMA console noise while keeping diagnostics available
  through structured/debug paths.

### PR #625: Candle Tail Event

- Branch: `codex/v8-candle-tail-event`.
- Scope: candle/EMA readiness observability.
- Result: emitted structured candle tail projection events.

### PR #626: Event Query Filter

- Branch: `codex/v8-live-event-query-filter`.
- Scope: query tooling.
- Result: added event-type filtering to `passivbot tool live-event-query`.

### PR #627: Warmup Cache Decision Event

- Branch: `codex/v8-warmup-cache-event`.
- Scope: startup/warmup observability.
- Result: emitted structured warmup cache decision events.

### PR #628: Startup Timing Event

- Branch: `codex/v8-startup-timing-event`.
- Scope: startup timing observability.
- Result: emitted startup timing events.

### PR #629: Cache Load Events

- Branch: `codex/v8-cache-load-events`.
- Scope: cache instrumentation.
- Result: emitted candle cache load events and hardened payload building.

### PR #630: Cache Load Event Throttle

- Branch: `codex/v8-cache-load-event-throttle`.
- Scope: high-volume policy.
- Result: throttled cache load events to keep structured output bounded.

### PR #631: Cache Flush Events

- Branch: `codex/v8-cache-flush-events`.
- Scope: cache instrumentation.
- Result: emitted cache flush events.

### PR #632: HSL Transition Events

- Branch: `codex/v8-hsl-transition-events`.
- Scope: risk/HSL observability.
- Result: emitted HSL red finalization events and fixed event dedupe between
  episodes.

### PR #633: Risk Mode Events

- Branch: `codex/v8-risk-mode-events`.
- Scope: risk mode observability.
- Result: emitted risk mode change events and covered halted HSL mode events.

### PR #634: Candle Coverage Events

- Branch: `codex/v8-candle-coverage-events`.
- Scope: candle coverage audit observability.
- Result: emitted candle coverage audit events.

### PR #635: Fill Refresh Events

- Branch: `codex/v8-fill-refresh-events`.
- Scope: fill refresh observability.
- Result: emitted fill refresh summary events and covered fill refresh resync
  summaries.

### PR #636: Rust Orchestrator Event Hardening

- Branch: `codex/v8-rust-orchestrator-event-hardening`.
- Scope: event emission safety.
- Result: hardened Rust orchestrator event emission and redacted orchestrator
  error events.

### PR #637: Live Ops Improvement Backlog

- Branch: `codex/v8-live-ops-improvement-backlog`.
- Scope: process tracking.
- Result: created the living operations improvement backlog and clarified live
  event query backlog work.

### PR #638: Live Event Query Filters

- Branch: `codex/v8-live-event-query-filters`.
- Scope: query tooling.
- Result: added additional live event query filters.

### PR #639: Live Smoke Report Tool

- Branch: `codex/v8-live-smoke-report-tool`.
- Scope: operator tooling.
- Result: added read-only live smoke report tooling for monitor/log inspection.

### PR #640: Health Summary Events

- Branch: `codex/v8-health-summary-events`.
- Scope: health observability.
- Result: emitted structured health summary events.

### PR #641: Live Incident Bundle

- Branch: `codex/v8-live-incident-bundle`.
- Scope: incident tooling.
- Result: added live incident bundle tool and redacted monitor snapshots.
- VPS5 evidence: bundle smoke created an archive successfully; tool returned
  attention because live GateIO HSL RED risk events were present, not because
  bundle generation failed.

### PR #642: Live Event Query ID Scopes

- Branch: `codex/v8-live-event-query-id-scopes`.
- Scope: query tooling.
- Result: added `bot_id`, `snapshot_id`, `plan_id`, `action_id`,
  `remote_call_group_id`, and related ID filters; timeline rendering now uses
  shared event ID keys.
- VPS5 evidence: deployed to VPS5 at `ad36d8ea`; `--remote-call-group-id`
  returned correlated Kucoin authoritative remote-call events.

### PR #643: Health Resource Pressure

- Branch: `codex/v8-health-resource-pressure`.
- Scope: health observability.
- Result: enriched structured `health.summary` events with resource pressure and
  live event pipeline counters.
- Review evidence: Cursor, Hermes, and Claude approved current head
  `d34241a4`; CI green; local targeted tests passed before merge.
- VPS5 evidence: pending pull/restart/smoke.

### PR #644: Logging And Ops Progress Tracking

- Branch: `codex/v8-live-logging-progress-tracker`.
- Scope: process tracking.
- Result: added this progress ledger and converted the live operations backlog
  into a living checklist with per-item statuses and a merged-work log.

### PR #645: Reason-Code Registry Slice

- Branch: `codex/v8-reason-code-registry-slice`.
- Scope: event taxonomy and drift prevention.
- Result: added shared `EventTags` and `ReasonCodes` registries for common live
  event tags/reason codes, migrated representative emitters without changing
  emitted strings, and documented the registry rule.

### PR #646: Console Event Summaries

- Branch: `codex/v8-console-event-summaries`.
- Scope: Phase 5 console/text projection.
- Result: improved `format_console_event()` with compact operator-facing tags
  and typed summaries for order waves, order writes, confirmation results, and
  Rust planning returns. Routes and console event volume were unchanged.

### PR #648: Live Event Trace Summaries

- Branch: `codex/v8-live-event-trace-summary`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --trace-summary` to aggregate
  matched live events by event type, level, status, reason code, ID scopes,
  symbol/side, and order-wave/action coverage. Summary counts cover all matched
  events even when `--limit` truncates the returned event sample.

### PR #649: Startup Timing Baselines In Smoke Report

- Branch: `codex/v8-startup-phase-budgets`.
- Scope: adjacent operations observability.
- Result: `passivbot tool live-smoke-report` now summarizes existing
  `bot.startup_timing` monitor events into latest per-phase timings and rolling
  median/p95/min/max baselines. Latest details are redacted before smoke-report
  or incident-bundle output.

### PR #651: Live Event Order Trace View

- Branch: `codex/v8-live-event-order-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --order-trace` to reconstruct
  order-wave/action lifecycles from existing structured execution events. The
  view groups by `order_wave_id` and `action_id`, reports event/status/reason
  counts, confirmation events, symbol/pside/side sets, and bounded event
  samples with shortened order/client-order references.

### PR #652: Order Trace Progress Update

- Branch: `codex/v8-progress-after-order-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #651 merged.

### PR #653: Live Event Registry Documentation

- Branch: `codex/v8-reason-code-registry-docs`.
- Scope: event taxonomy documentation and drift prevention.
- Result: added `docs/ai/live_event_registry.md` for stable event tags and
  reason codes, linked it from the AI docs router/logging guide, and added a
  doc drift test that compares documented values to `EventTags`/`ReasonCodes`.

### PR #654: Live Event Cycle Trace View

- Branch: `codex/v8-live-event-cycle-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --cycle-trace` to reconstruct
  matched events by `cycle_id`. Each cycle contains bounded timeline samples,
  aggregate trace summaries, and nested order traces using the existing order
  lifecycle reconstruction.

### PR #655: Cycle Trace Progress Update

- Branch: `codex/v8-progress-after-cycle-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PRs #652-#654 merged.

### PR #656: Local Cache Integrity Doctor

- Branch: `codex/v8-cache-integrity-doctor-slice`.
- Scope: adjacent operations tooling.
- Result: added `passivbot tool cache-integrity-doctor`, a read-only local
  cache smoke report for root presence, aggregate file/size counts, empty
  files, and corrupt JSON/NDJSON/NPY artifacts. This is an initial cache-doctor
  slice; it does not yet prove warm-cache coverage or HSL/fill metadata
  compatibility.

### PR #658: Cache Doctor Progress Update

- Branch: `codex/v8-progress-after-cache-doctor`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #656 merged.

### PR #659: Incident Bundle Trace Reports

- Branch: `codex/v8-incident-bundle-traces`.
- Scope: incident tooling.
- Result: `passivbot tool live-incident-bundle` now embeds existing
  `live-event-query` trace-summary and order-trace reports in `event_report.json`
  by default, includes cycle-trace reconstruction when scoped to `--cycle-id`,
  and supports `--no-trace-report` for compact bundles.
- VPS5 evidence: deployed to VPS5 at `27931c81`; a read-only bundle smoke on
  monitor data produced a tarball containing `trace_summary`, `order_trace`, and
  `cycle_trace` sections. The tool returned attention because the embedded
  smoke report saw existing GateIO HSL RED and EMA readiness degradation.

### PR #661: Live Smoke Process Status

- Branch: `codex/v8-smoke-process-status`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` can now include an optional
  read-only `processes` section. With `--supervisor-config`, tmuxp-style
  expected `passivbot live` commands are compared against running live
  processes and missing expected bots become smoke hard failures. Incident
  bundles pass the same process snapshot through `smoke_report.json` when
  requested.
- VPS5 evidence: deployed to VPS5 at `72b3d931`; read-only smoke using
  `/root/bots_vps5.yaml` matched all five expected bots and left them running.
  The overall smoke exit remained nonzero because Kucoin authoritative state
  fetches had recent `RequestTimeout` events, not because process liveness
  failed.

### PR #663: Remote-Call Failure Smoke Summary

- Branch: `codex/v8-smoke-remote-call-summary`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `remote_call_failures` aggregate section built from existing
  `remote_call.failed` monitor events. Groups are keyed by
  bot/reason/surface/error type/component and include latest redacted failure
  context.
- VPS5 evidence: deployed to VPS5 at `45b0cf9e`; read-only smoke using
  `/root/bots_vps5.yaml` still matched all five expected bots and now exposed
  Kucoin authoritative endpoint timeouts directly in the smoke output:
  positions=9, open_orders=7, balance=7.

### PR #665: Live Smoke Report Time Window

- Branch: `codex/v8-smoke-report-time-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` can scope structured monitor
  events with `--since-ms`, `--until-ms`, or `--recent-minutes`, and reports
  explicit event-window counters. Incident bundles pass the same window into
  embedded smoke reports. Text-log scanning remains intentionally unchanged.

### PR #666: Create Filter/Defer Events

- Branch: `codex/v8-execution-defer-events`.
- Scope: execution/order lifecycle observability.
- Result: create-order pre-exchange filter/defer decisions now emit bounded
  structured events after the existing gates decide. The execution gates remain
  authoritative, event emission is best-effort, and the new routes stay off the
  default console/text projection.

### PR #667: Live Event Query Time Window

- Branch: `codex/v8-event-query-time-window`.
- Scope: operator query tooling.
- Result: `passivbot tool live-event-query` can scope matched structured events
  with `--since-ms`, `--until-ms`, or `--recent-minutes`. The same scoped event
  set feeds query output, timeline, trace summary, order trace, and cycle trace
  views, with explicit event-window counters.

### PR #668: Cache Doctor Family Summary

- Branch: `codex/v8-cache-doctor-family-summary`.
- Scope: adjacent operations tooling.
- Result: `passivbot tool cache-integrity-doctor` now includes per-root and
  aggregate cache-family summaries plus family tags on reported issues. This is
  still read-only diagnostics and does not decide whether live trading may reuse
  a warm cache.
- VPS5 evidence: deployed to VPS5 at `734c2de0`. A settled 2-minute smoke after
  restart reported all five configured bots running and no hard problem events.

### PR #670: Smoke Report Timestamped Log Window

- Branch: `codex/v8-smoke-report-log-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` applies `since_ms`/`until_ms` and
  `--recent-minutes` windows to parseable ISO-UTC text log lines as well as
  structured monitor events. Unparseable log lines remain visible and are
  counted in `logs.window.unparsed_ts`.
- VPS5 evidence: deployed to VPS5 at `b74d12be` without bot restart. Smoke
  showed `logs.window.lines_skipped_before=2000`, proving stale parseable log
  lines were excluded, while two current Kucoin websocket warning lines were
  still false-classified as hard due to the older traceback matcher.

### PR #671: Smoke Report Traceback Prose Filter

- Branch: `codex/v8-smoke-report-traceback-pattern`.
- Scope: operator smoke tooling.
- Result: smoke-report text-log matching now treats only real Python traceback
  headers (`Traceback (most recent call last):`) as traceback signals, avoiding
  hard/attention matches for operational prose such as "suppressing callback
  traceback".
- VPS5 evidence: deployed to VPS5 at `34f63799` without bot restart. A
  2-minute smoke with text logs enabled reported all five configured bots
  running, `logs.hard_matches=0`, and no hard problem events.

### PR #673: Live Smoke Risk Event Summary

- Branch: `codex/v8-live-smoke-risk-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `risk_events` aggregate built from existing structured HSL/risk events. It
  groups by bot, event type, symbol, pside, and reason, keeps the latest compact
  risk fields such as tier, mode, drawdown score, distance to red, and cooldown
  timing, and does not change smoke `ok`/`attention`/hard-failure policy.
- VPS5 evidence: deployed to VPS5 at `2697ff48` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no log hard
  matches, and exposed GateIO ZEC long HSL RED cooldown in `risk_events`.

### PR #675: Smoke Log Unparsed Policy

- Branch: `codex/v8-smoke-log-unparsed-policy`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` and embedded incident-bundle smoke
  reports now accept `--log-window-unparsed-policy keep|drop`. The default
  `keep` preserves prior behavior; opt-in `drop` suppresses only non-signal
  unparseable text-log lines when a log window is active. Signal-bearing
  unparseable lines, including Python traceback headers, remain visible and can
  still make smoke hard-fail.
- VPS5 evidence: deployed to VPS5 at `3aa1e7a7` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no log matches, no remote-call failures, and
  `logs.window.unparsed_policy=drop`.

### PR #676: Smoke Log Unparsed Policy Progress

- Branch: `codex/v8-progress-after-unparsed-policy`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #675 merged and VPS5 smoke confirmed the unparseable-log window policy.

### PR #677: Execution Loop Error Burst Event

- Branch: `codex/v8-execution-loop-error-burst-event`.
- Scope: Phase 5 text-log-to-event migration.
- Result: the existing execution-loop error burst warning now emits a bounded
  structured `health.summary` event with reason code
  `execution_loop_error_burst` before the existing stdlib warning. Emission is
  best-effort, uses the existing warning threshold, redacts/caps latest error
  text, and does not change restart/backoff/trading behavior or default console
  volume.
- Review evidence: Claude and Hermes approved head `409f5d8e`; focused pytest,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `eda7cb2f` with a full bot restart. Three
  smoke windows reported all five configured bots running, no hard failures, no
  log hard matches, and no missing expected processes. Settled windows still
  showed non-hard EMA readiness / staged-execution degradation and GateIO ZEC
  HSL cooldown, which remain separate operational signals.

### PR #678: Execution Burst Progress Update

- Branch: `codex/v8-progress-after-execution-burst-event`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #677 merged and VPS5 smoke confirmed the execution-loop error burst event
  migration.

### PR #679: Smoke Problem Event Context

- Branch: `codex/v8-smoke-problem-event-data`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded,
  allowlisted `latest_data` for relevant problem-event groups such as
  `ema.unavailable` and `cycle.degraded`, with recursive redaction and payload
  bounds. The text-log scanner also uses the last parsed timestamp as context
  for unparseable continuation lines inside active windows, so stale traceback
  fragments after old errors are skipped while current traceback signals remain
  preserved.
- Review evidence: Hermes approved the original and amended delta; CI was
  green; focused smoke-report tests, compileall, and `git diff --check` passed
  before merge. Claude did not return during repeated polls, and Composer had
  been retired, so this docs/tooling-only slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `ff714b61` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no text-log hard or
  attention matches, and exposed useful `latest_data` for the remaining
  non-hard EMA/cycle readiness groups.

### PR #680: Smoke Problem Context Progress

- Branch: `codex/v8-progress-after-smoke-problem-context`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #679 merged and VPS5 smoke confirmed the bounded problem-event context.

### PR #682: Smoke Problem Event Groups

- Branch: `codex/v8-smoke-problem-event-groups`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now reports top-level
  `problem_event_count` and bounded `problem_event_groups` aggregates grouped
  by bot, event type, reason code, status, hard flag, symbol, and position
  side. Existing bounded `problem_events` samples remain available for detail.
- Review evidence: Hermes approved head `048e8595c`; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
  Claude did not return during repeated polls, and Composer had been retired,
  so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `998d7c9c` without bot restart. The first
  smoke surfaced a real transient Kucoin authoritative account-refresh timeout;
  a later settled 2-minute smoke reported all five configured bots running, no
  hard failures, no hard problem events, no log matches, no remote-call
  failures, and grouped the remaining known non-hard EMA/cycle/HSL attention in
  `problem_event_groups`.

### PR #683: Smoke Grouping Progress Update

- Branch: `codex/v8-progress-after-smoke-grouping`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #682 merged and VPS5 smoke confirmed grouped problem-event summaries.
- Review evidence: Hermes approved head `3fa8d819`; CI was green; `git diff
  --check` passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this docs-only slice used the degraded gate.

### PR #684: Contextless Traceback Smoke Filter

- Branch: `codex/v8-smoke-drop-contextless-traceback`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --log-window-unparsed-policy drop`
  now skips unparseable log lines without in-window timestamp context, avoiding
  stale hard failures when the inspected log tail begins in the middle of an old
  traceback. The direct smoke-report and incident-bundle help text now describes
  the context-based drop behavior.
- Review evidence: Hermes approved original head `f01996ec` with one minor
  help-text mismatch, then approved the fixed head `04ca717`; CI was green;
  focused smoke-report tests, compileall, and `git diff --check` passed before
  merge. Claude did not return during repeated polls, and Composer had been
  retired, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `dff40001` without bot restart. A
  1-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and no remote-call failures.

### PR #686: Smoke Repository Metadata

- Branch: `codex/v8-smoke-repo-metadata`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a best-effort
  `repository` block with worktree root, branch, head, full head, tracked-only
  dirty status, and tracked change count. Git lookup failures remain
  observational and do not affect smoke hard-failure accounting.
- Review evidence: Hermes approved head `b03f4139`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `9e898019` without bot restart. The smoke
  report confirmed `repository.branch=v8`, `repository.head=9e898019`,
  `repository.dirty=false`, `tracked_changes=0`, and all five configured bots
  running. The same smoke surfaced a separate Kucoin operational issue:
  repeated authoritative balance/positions/open-orders `RequestTimeout` events
  with 98-140s staged refresh wall times and websocket ping timeouts.

### PR #688: Remote-Call Timing Smoke Summary

- Branch: `codex/v8-smoke-remote-call-timings`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_timings` groups for terminal remote calls that expose elapsed
  median, p95, min, max, latest ids, and latest elapsed time. The section is
  observational only and does not affect `ok`, `attention`, or trading
  behavior.
- Review evidence: Hermes approved head `9945a3d3`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `11f7d142` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no hard problem
  events, no log hard or attention matches, no remote-call failures, and
  `remote_call_timings.total=637`.

### PR #690: Remote-Call Health Smoke Summary

- Branch: `codex/v8-smoke-remote-call-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_health` groups that roll up terminal remote calls by
  bot/component/kind/surface, with success/failure/throttle counts, failure and
  throttle percentages, latency summaries, reason/error counts, and bounded
  affected-symbol samples. Throttled terminal buckets are derived from
  `event_type`, while differing raw statuses remain auxiliary context.
- Review evidence: Hermes first found that `remote_call.throttled` events with
  raw `status="deferred"` were not counted as throttles, then approved fixed
  head `dc99378a`; CI was green; focused smoke-report and incident-bundle
  tests, compileall, and `git diff --check` passed before merge. Claude did
  not return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `b150176f` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, no remote-call failures, and `remote_call_health.total=445`.

### PR #692: Remote-Call Health Top-Level Totals

- Branch: `codex/v8-smoke-remote-call-health-totals`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes top-level
  `remote_call_health` success, failure, throttle, failure-percent, and
  throttle-percent totals in addition to the existing bounded per-group health
  details. This keeps operator smoke summaries scannable without changing
  `ok`, `attention`, event producers, or trading behavior.
- Review evidence: Hermes first found that the new aggregate failure/throttle
  counters could be overwritten by per-group counters, then approved fixed head
  `ac4afe3f`; CI was green; focused smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge. Claude did not return
  during repeated polls, and Composer had been retired, so this read-only
  tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `c8ce4880` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and top-level `remote_call_health` totals:
  `total=390`, `succeeded=389`, `failed=1`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #694: Account-Critical Remote-Call Health

- Branch: `codex/v8-smoke-authoritative-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes
  `account_critical_remote_call_health`, a filtered view of terminal
  authoritative balance, position, open-order, and Hyperliquid split
  account-state surfaces. It reuses the existing bounded remote-call health
  summarizer while excluding broader candle/fill traffic.
- Review evidence: Hermes approved head `bebbb3f6`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, `git diff --check`, and
  the touched-file silent-handling audit passed before merge. Claude did not
  return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `3299c1ca` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and account-critical health totals:
  `total=126`, `succeeded=126`, `failed=0`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #696: Concise Live Smoke Summary

- Branch: `codex/v8-smoke-report-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --summary` now projects the full
  report down to high-signal smoke fields: health booleans/counters,
  repository state, monitor totals, event/log windows, process summary,
  bounded problem groups, remote-call/account-critical health, and risk events.
  `--compact` can be combined with `--summary` for short machine-readable
  output. Full report generation and exit-code behavior are unchanged.
- Review evidence: Hermes approved head `f1efbe45`; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge. Claude did not return during
  repeated polls, and Composer had been retired, so this read-only tooling
  slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `d850daf5` without bot restart. A
  2-minute compact summary smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported all five configured bots running, no hard failures, no log
  hard or attention matches, account-critical calls `total=58`,
  `succeeded=58`, and all terminal remote calls `total=169`,
  `succeeded=169`.

### PR #681: Staged Refresh Timing Events

- Branch: `codex/v8-staged-refresh-events`.
- Scope: state refresh observability.
- Result: staged refresh timing and progress logs now emit structured
  `state.refresh_timing` and `state.refresh_progress` events with bounded
  timing, plan, residual, and pending-surface data. Emission is best-effort and
  does not change refresh behavior or console log text.
- Review evidence: Claude approved code-identical head `bddf5a4c`, Hermes
  approved rebased head `4be87ce7`, and CI was green; focused
  state-refresh/smoke tests, compileall, and `git diff --check` passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `9a52d3a9` with a bot restart. A settled
  2-minute smoke reported `ok=true`, no hard failures, no log hard/attention
  matches, all five configured bots running, and all account-critical/terminal
  remote calls succeeding. A direct event query confirmed
  `state.refresh_timing` and `state.refresh_progress` events. A wider smoke
  surfaced a real GateIO ZEC HSL RED finalization during startup/replay,
  unrelated to the event slice.

### PR #698: Smoke Repository Root Redaction

- Branch: `codex/v8-smoke-report-redact-repo-root`.
- Scope: operator smoke tooling privacy.
- Result: `live-smoke-report` now redacts current-home, `/root`,
  `/home/<user>`, and `/Users/<user>` prefixes from the serialized
  `repository.root` field while continuing to run git commands against the real
  resolved repository path. Incident-bundle `smoke_report.json` inherits the
  same safer display field.
- Review evidence: current-head Claude and Hermes approved head `7c7368f3`;
  CI was green; focused smoke-report/incident-bundle tests, compileall, and
  `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 as part of the later `d5639813` pull without
  bot restart. The settled smoke reported clean repository state on `v8`, all
  five configured bots running, and no hard failures.

### PR #699: Dropped Unparsed Smoke Log Signal Counters

- Branch: `codex/v8-smoke-report-dropped-unparsed-counters`.
- Scope: operator smoke tooling.
- Result: when `--log-window-unparsed-policy drop` suppresses a contextless
  unparseable log line that still matches attention/hard patterns, smoke
  reports now expose dropped attention/hard counters and include dropped
  attention signals in `attention_count`. Dropped contextless fragments remain
  excluded from `hard_failures`, preserving the stale-tail false-positive
  suppression from PR #684.
- Review evidence: current-head Claude and Hermes approved rebased head
  `4e2fcee7`; CI was green; full smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `d5639813` without bot restart. Two
  immediate smokes caught real Kucoin authoritative `RequestTimeout` bursts; a
  later settled 2-minute compact summary smoke reported `ok=true`, no hard
  failures, no log hard/attention matches, all five configured bots running,
  account-critical calls `total=81`, `succeeded=81`, and all terminal remote
  calls `total=230`, `succeeded=230`.

### PR #709: Fill Cache Ready Event

- Branch: `codex/v8-fills-cache-ready-event`.
- Scope: Phase 5 startup fill-cache observability.
- Result: startup fill-cache readiness now emits a structured
  `fills.refresh_summary` event with reason code `fill_cache_ready`, source
  `startup`, refresh mode `cache_load`, elapsed time, event count, and optional
  history scope. The existing console line remains unchanged and the structured
  event route stays off console/text.
- Review evidence: current-head Claude and Hermes approved head `f5838dfea`,
  and CI was green. Focused live-event, fill-cache init/update, monitor emitter,
  compileall, and `git diff --check` checks passed before merge.
- VPS5 evidence: deployed to VPS5 at `71479c61` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls. Direct monitor-file checks found
  `fill_cache_ready` events for Binance, GateIO, Hyperliquid, Kucoin, and OKX.

### PR #711: Exchange Time-Sync Event

- Branch: `codex/v8-exchange-time-sync-event`.
- Scope: Phase 5 exchange timestamp/nonce recovery observability.
- Result: CCXT timestamp/nonce recovery now emits bounded
  `exchange.time_sync` events for recovery success or unavailable exchange
  hooks. The event route stays off console/text, and event emission is
  best-effort so it cannot mask the original recovery path.
- Review evidence: current-head Claude and Hermes approved head `225a0e2b8`,
  and CI was green. Focused live-event, exchange time-sync recovery, execution
  loop timestamp-error, compileall, and `git diff --check` checks passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `0fa6269b` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls.

### PR #712: Supervisor Process Diagnostics

- Branch: `codex/v8-smoke-supervisor-process-diagnostics`.
- Scope: operator smoke tooling.
- Result: `live-smoke-report --supervisor-config` now classifies expected
  process matches, duplicate configured-command matches, and extra/orphan-like
  `passivbot live` processes from bounded local process-table metadata. The
  report explicitly states that tmux pane ownership is not available from this
  read-only process-table classifier.
- Review evidence: Claude first found the no-RSS `ps` fallback row parser
  dropped valid process rows, then approved fixed head `da39c8af`; Hermes
  approved the same fixed head, CI was green, and focused smoke-report and
  incident-bundle tests plus `git diff --check` passed before merge.
- VPS5 evidence: pulled to VPS5 at `51ba92a3` without bot restart. A settled
  5-minute brief/summary smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, no failed
  account-critical remote calls, and zero duplicate/extra live process matches.

### PR #714: Live Config Preflight Tool

- Branch: `codex/v8-live-config-preflight`.
- Scope: adjacent operator tooling.
- Result: added `passivbot tool live-config-preflight`, a read-only offline JSON
  report for one live config covering identity hints, HSL settings, approved and
  ignored universe counts with bounded samples, forager slots/staleness, and
  cache-related live settings. The tool does not load credentials, contact
  exchanges, or enforce startup policy.
- Review evidence: Hermes approved the original head `b2557db0`; after PR #713
  merged, the branch was rebased to `b51a15a2` with the same tool code plus
  resolved progress-doc context. CI was green, local focused preflight tests
  passed, and `git diff --check` passed. Claude did not return during repeated
  polls, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A
  `live-config-preflight --compact` smoke against
  `configs/forager_3pos_hsl_2026-06-26.json` returned `ok=true`, reported
  bounded approved/ignored coin samples and HSL/forager/cache settings, and
  surfaced one warning for missing short-side bot config.

### PR #715: Shutdown Event Smoke Summary

- Branch: `codex/v8-smoke-shutdown-summary`.
- Scope: operator smoke tooling.
- Result: existing `bot.stopping`, `bot.shutdown.stage`, and `bot.stopped`
  structured events are now summarized as `shutdown_events` in the full,
  `--summary`, and `--brief` smoke-report projections. The change is passive and
  does not add shutdown control, process signaling, or trading behavior.
- Review evidence: Hermes approved heads `01574b4d` and `7dfa6c4a`; after
  PR #714 merged, the branch was rebased to `befade50d` with only current-base
  changelog/tool-doc context added underneath. CI was green, local
  smoke-report/incident-bundle tests and compileall passed, and `git diff
  --check` passed. Claude did not return during repeated polls, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A settled
  5-minute brief/summary smoke reported no hard failures, no log hard matches,
  all five configured bots running, no failed remote/account-critical calls,
  clean tracked repository state, and `shutdown_events.total=0` because no
  restart occurred.

### PR #719: Live Restart Smoke Plan Tool

- Branch: `codex/v8-live-restart-smoke-plan`.
- Scope: adjacent operator restart/smoke tooling.
- Result: added `passivbot tool live-restart-smoke-plan`, a read-only dry-run
  planner for the repeated live restart/smoke routine. The tool parses a
  tmuxp-style supervisor config through the existing sanitized smoke-report
  parser and emits structured plan metadata, per-bot phases, repo checks, smoke
  command wiring, timeout/escalation guidance, and explicit non-execution
  policy. `--execute` is rejected; the tool does not SSH, invoke tmux, signal
  processes, pull code, start bots, contact exchanges, or load credentials.
- Review evidence: Hermes approved head `c7c4ec09`; CI was green; local
  focused restart-plan, smoke-report, CLI dispatch, compileall, and `git diff
  --check` validation passed before merge. Claude did not return before merge,
  so this plan-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart as part of
  the PR #719/#720 deploy. A dry-run plan against `/root/bots_vps5.yaml`
  reported `ok=true`, `dry_run=true`, `execution_available=false`, five planned
  bots, no issues, and the expected rejected operations.

### PR #720: EMA Live Event Debug Profile

- Branch: `codex/v8-ema-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `ema` as a `logging.live_event_debug_profiles` /
  `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` profile, including `ema-readiness`
  aliases. When `ema` is enabled, existing `ema.unavailable` events include
  bounded parsed EMA type, span, and inner reason summaries.
  Default events remain compact, console output is unchanged, and no
  exchange/order/risk behavior changed.
- Review evidence: Hermes approved original head `1a9a53218`; after PR #719
  merged, the branch was rebased to `b8be04654` with the same code delta, CI was
  green, and local `tests/test_live_event_bus.py`,
  `tests/test_passivbot_monitor.py`, compileall, and `git diff --check`
  validation passed. Claude did not return before merge, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported all five configured bots running, no hard failures, no log hard
  matches, no failed remote calls, no failed account-critical remote calls, and
  clean tracked repository state.

### PR #722: Cache Doctor Candle Coverage Evidence

- Branch: `codex/maxwell-cache-integrity-doctor-13a`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now derives v2 candle
  coverage windows, valid row counts, suspicious interior gap samples, and
  non-enforcing warm-cache evidence labels from local `.valid.npy` artifacts.
  It remains read-only and does not change cache materialization, startup, or
  trading behavior.
- Review evidence: Hermes approved head `43f6d17b`; CI was green; Maxwell ran
  focused cache-doctor tests, compileall, `git diff --check`, and a touched-file
  silent-handling audit before opening the PR. Claude did not return during the
  merge window, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed as part of the `09ae3773` pull without bot restart.
  The 10-minute brief smoke reported all five configured bots running, clean
  tracked repository state, no hard failures, no log hard matches, no failed
  remote calls, and no failed account-critical remote calls.

### PR #727: Cache Doctor Warm-Cache Readiness Evidence

- Branch: `codex/maxwell-cache-warm-readiness`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now adds report-only
  `warm_cache_readiness` summaries derived from already-scanned candle, fill,
  and HSL/risk cache metadata. The readiness projection is explicitly
  non-enforcing and does not change startup or trading behavior.
- Review evidence: Claude and Hermes approved head `155e3640`; CI was green;
  focused cache-doctor tests, compileall, `git diff --check`, and
  touched-file silent-handling audit passed before merge. A parent-side
  temporary-worktree validation also passed the focused test/check set.

### PR #723: Remote-Call Debug Profile

- Branch: `codex/v8-remote-call-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `remote_calls` debug-profile enrichment for candle remote-fetch
  and authoritative state-fetch events. Enrichment is bounded to key shape,
  selected timing/correlation fields, status/surface/kind, and counts; default
  events remain unchanged, console output is unchanged, and no raw payloads are
  added.
- Review evidence: Hermes approved head `e78d79b0`; CI was green; focused
  remote-call profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return during the merge window, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `09ae3773` without bot restart because the
  profile is opt-in and no VPS config enabled it. The same 10-minute brief
  smoke reported no hard failures, all expected bots matched, and zero failed
  remote/account-critical calls. Remaining attention came from known non-hard
  EMA/staged-readiness and HSL cooldown events.

### PR #724: Candle Debug Profile

- Branch: `codex/v8-candle-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `candles` debug-profile enrichment to existing
  `candle.tail_projected` and `candle.coverage_checked` events, exposing bounded
  key-shape, timeframe/window, and missing-coverage counters without raw candle
  arrays or default console/log changes.
- Review evidence: Hermes approved head `4454fd09`; CI was green; focused
  candle debug-profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return before merge, so this opt-in observability slice
  used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `f3969dbc` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported `ok=true`, no hard failures, all five expected bots matched, clean
  tracked repository state, no failed remote calls, and no failed
  account-critical remote calls.

### PR #726: Reviewer Follow-Ups

- Branch: `codex/v8-reviewer-followups`.
- Scope: Claude retrospective follow-up for already-merged low-risk
  observability/tooling slices.
- Result: redacted shareable live-ops path fields consistently, removed
  shutdown event message echo from smoke summaries, scopes EMA debug enrichment
  to the `ema` profile only, and keeps Rust debug sample construction
  best-effort inside the event-emitter path.
- Review evidence: Claude and Hermes approved head `8f4465a9`, CI was green,
  and local focused tests plus the broader live-event/monitor/smoke/preflight
  suite, compileall, `git diff --check`, and touched-file silent-handling audit
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `41961266` without bot restart because the
  slice is observability/tooling-only. A 10-minute brief smoke reported
  `ok=true`, no hard failures, all five expected bots matched, clean tracked
  repository state, no failed remote calls, and no failed account-critical
  remote calls. Remaining attention came from known non-hard EMA/staged
  readiness and HSL status/cooldown events.

### PR #728: Fills Debug Profile

- Branch: `codex/v8-fills-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `fills` debug-profile enrichment to existing
  `fills.refresh_summary` and `fill.ingested` events with bounded coverage,
  count, and key-shape metadata. Default events and console output stay
  unchanged, and no raw fill IDs, source IDs, or payload values are emitted.
- Review evidence: Claude and Hermes approved the original head `4c58a718`;
  after PR #727 merged, the branch was rebased to `8ba083f6`, CI was green, and
  both reviewers confirmed the rebased patch was unchanged. Local focused tests,
  the broader live-event/monitor suite, compileall, and `git diff --check`
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `5714d36d` without bot restart because the
  slice is opt-in and no VPS config enabled it. The first 10-minute smoke was
  red only because text logs contained a fresh OKX ccxt-pro websocket callback
  traceback after a reconnect; structured monitor events had no hard failures,
  all five expected bots matched, the repository was clean, and remote-call /
  account-critical failures were zero. A settled 2-minute follow-up smoke
  reported `ok=true`, no hard failures, no log hard matches, all five expected
  bots matched, clean tracked repository state, no failed remote calls, and no
  failed account-critical remote calls.

### PR #730: HSL Debug Profile

- Branch: `codex/v8-hsl-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `hsl` debug-profile enrichment to existing HSL
  status, transition, replay, red-trigger, and cooldown events with bounded
  event key, metric key, and HSL latch/cooldown state-shape metadata. Default
  HSL events and console output remain unchanged.
- Review evidence: Hermes approved head `7cfb80ac`, CI was green, and local
  focused tests, the broader HSL/live-event suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed. Claude did
  not return before merge, so this opt-in observability slice used the degraded
  gate.
- VPS5 evidence: deployed together with PRs #731 and #732 at `9bc2c37f`.
  Settled 5-minute smoke returned `ok=true`, no hard failures, no log hard
  matches, all five expected bots matched, clean tracked repository state, no
  failed remote calls, no failed account-critical remote calls, and no monitor
  errors or warnings. Remaining attention was non-hard Hyperliquid EMA/candle
  readiness.

### PR #732: Execution Debug Profile

- Branch: `codex/v8-execution-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `execution` debug-profile enrichment to existing
  order-wave, order-write, create-filter, and confirmation events with bounded
  key-shape/counter metadata. Default execution event payloads and console
  output remain unchanged, and raw order payload values are not added.
- Review evidence: Hermes found and then approved the raw-order-value leak fix
  at `cb1f2c23`; after rebasing onto current `v8`, CI was green and focused
  execution/live-event tests, compileall, and `git diff --check` passed. Claude
  did not return before merge, so this opt-in observability slice used the
  degraded gate.
- VPS5 evidence: deployed together with PRs #730 and #731 at `9bc2c37f`.
  Settled 5-minute smoke returned `ok=true`, no hard failures, no log hard
  matches, all five expected bots matched, clean tracked repository state, no
  failed remote calls, no failed account-critical remote calls, and no monitor
  errors or warnings.

### PR #734: Retrospective Tool Hygiene

- Branch: `codex/v8-retro-tool-hygiene`.
- Scope: Claude retrospective follow-up for already-merged operator tooling.
- Result: collapsed user/deploy prefixes in shareable restart/preflight/HSL
  preview output, resolved grouped and flat bot-side config keys in preflight
  reports, and kept HSL startup-preview event data scalar and allowlisted.
- Review evidence: Claude and Hermes approved; CI was green; focused
  preflight/HSL preview/restart-plan tests, compileall, `git diff --check`, and
  touched-file audits passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  operator tooling. Existing bots were left running.

### PR #735: Startup Budget Smoke Projections

- Branch: `codex/maxwell-startup-budget-smoke`.
- Scope: adjacent startup/warmup observability.
- Result: `live-smoke-report` now projects existing `bot.startup_timing` events
  into report-only elapsed and per-phase budget evidence using recent local p95
  baselines. The slice does not enforce budgets or change startup behavior.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed without bot restart. A settled follow-up smoke
  reported all five expected bots running, no hard failures, no log hard
  matches, no failed remote calls, and no failed account-critical calls.

### PR #736: EMA Readiness Handoff Backlog

- Branch: `codex/v8-backlog-ema-readiness-race`.
- Scope: living backlog / progress tracking.
- Result: recorded the OKX/AAVE active-symbol forager EMA readiness handoff gap
  as live-ops backlog item #17 and noted that the first remediation must
  preserve the no-fabricated-ranking-values contract.
- Review evidence: Claude and Hermes approved; CI was green before merge.
- VPS5 evidence: pulled to VPS5 without bot restart; all five configured bots
  remained running.

### PR #737: Active Forager EMA Carry-Forward

- Branch: `codex/v8-forager-active-ema-projection`.
- Scope: live forager EMA readiness hardening for active/normal symbols.
- Result: active/normal forager symbols may use bounded cached real-candle
  quote-volume and log-range EMA values when the current EMA read is transiently
  unavailable, while candidate-only symbols still become unavailable and
  active/normal symbols without cached values still fail loudly. The fix keeps
  open-tail projection valid for close EMA readiness only in forager mode, so
  qv/log-range ranking inputs do not come from projected open-tail values.
- Review evidence: Hermes first found that cached metric fallback and open-tail
  projection shared one eligibility map; fixed head `6965783e` split cached
  metric and projection eligibility and added a regression covering projected
  qv/log-range values differing from cached real-candle values. Claude and
  Hermes approved the fixed head; CI was green; focused EMA tests, compileall,
  `git diff --check`, and a diff-only silent-handling audit passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `e3429ee9` with a bot restart. Settled
  2-minute and 5-minute smokes reported all five expected bots running, no hard
  failures, no log hard matches, no failed remote calls, no failed
  account-critical remote calls, and clean tracked repository state.

### PR #739: Restart Plan Process Signal Safety

- Branch: `codex/v8-restart-plan-process-safety`.
- Scope: adjacent operator restart/smoke planning.
- Result: `passivbot tool live-restart-smoke-plan` now includes a
  `process_signal_safety` contract that warns future restart automation away
  from broad `pkill -f` / `pgrep -f` live-bot process matches and toward exact
  tmux panes or exact canonical process rows. The plan remains read-only and
  execution-unavailable.
- Review evidence: Claude and Hermes approved; CI was green; focused
  restart-plan tests, compileall, `git diff --check`, and the diff-only
  touched-file silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is plan-only
  operator tooling. A settled 5-minute smoke at `87e53dac` reported all five
  expected bots running, no hard failures, no log hard matches, no failed
  remote calls, no failed account-critical remote calls, and clean tracked
  repository state.

### PR #741: Ticker Probe Time Sync Health

- Branch: `codex/v8-ticker-probe-clock-skew`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now records bounded
  `fetch_time` clock-skew evidence in each repeat and summarizes it as
  per-user and collection-level `time_sync_health`. Unsupported exchanges are
  counted separately from true failures, and `--skip-time-sync` omits the extra
  read-only time call.
- Review evidence: Hermes first found that inherited CCXT `fetch_time` methods
  can exist when `has["fetchTime"]` is false or missing. The fixed head
  `6d24b8b7` gates on `has["fetchTime"] is True`, remaps `NotSupported` to
  unsupported/skipped, and adds a regression proving the unsupported inherited
  method is not called. Claude and Hermes approved the fixed head; CI was
  green; focused ticker-probe tests, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `d4c28058` reported all five expected bots
  running, no hard failures, no log hard matches, no failed account-critical
  remote calls, and clean tracked repository state. A real Binance
  `--account-only` probe produced `time_sync_health.total=1`,
  `succeeded=1`, `failed=0`, `unsupported=0`, and `max_abs_clock_skew_ms=14`.

### PR #743: Ticker Probe Candle Freshness Health

- Branch: `codex/v8-ticker-probe-candle-freshness`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `candle_freshness_health` from the existing 1m OHLCV tail probe results. The
  summary reports symbol success/failure counts, current-incomplete candle
  counts, last-candle age statistics, and the worst-age symbol without making
  additional exchange calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `1fe1292b` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  real Binance public-only probe for `BTC/USDT:USDT` produced
  `candle_freshness_health.total_symbols=1`, `succeeded_symbols=1`,
  `failed_symbols=0`, `current_incomplete_symbols=1`, and
  `worst_symbol=BTC/USDT:USDT`.

### PR #745: Ticker Probe Fill History Health

- Branch: `codex/v8-ticker-probe-fill-history-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `fill_history_health` from the existing first-symbol `fetch_my_trades`
  sample. The summary reports success/failure counts, latency, trade count,
  newest timestamp, side/symbol shape, and id/order presence counts without raw
  trade/order ids or raw fill payloads. It intentionally does not add fill
  pagination calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `4130155e` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `fill_history_health.total=1`, `succeeded=1`, `failed=0`,
  `latest_symbol=BTC/USDT:USDT`, and `latest_trade_count=0`.

### PR #747: Ticker Probe Rate Limit Health

- Branch: `codex/v8-ticker-probe-rate-limit-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `rate_limit_health` from existing probe outcomes and CCXT
  `rateLimit`/`enableRateLimit` metadata. The summary reports observed
  public/private/concurrent call counts, endpoint counts, configured sleep, and
  an estimated minimum serial duration without adding exchange calls or
  enforcing throttles.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A brief smoke at `74270454` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `rate_limit_health.observed_call_count=12`, `public_call_count=6`,
  `private_call_count=5`, `concurrent_request_count=1`,
  `exchange_rate_limit_ms=50`, and `estimated_min_serial_ms=600`.

### PR #749: Ticker Probe Fill Pagination Sample

- Branch: `codex/v8-ticker-probe-fill-pagination-sample`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` keeps the default
  one-call first-symbol `fetch_my_trades` sample, and adds opt-in bounded
  pagination through `--fill-history-pages` and `--fill-history-page-limit`.
  The probe records only page/count/timestamp/latency summaries and terminal
  pagination reason, with no raw trade/order ids.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A 5-minute smoke and a settled 3-minute smoke at `16c25149`
  reported all five expected bots running, no hard failures, no log hard
  matches, no failed remote calls, no failed account-critical remote calls, and
  clean tracked repository state. A one-repeat authenticated Binance probe for
  `BTC/USDT:USDT` with `--fill-history-pages 2 --fill-history-page-limit 2`
  validated `fill_history_health.total=1`, `succeeded=1`, `failed=0`,
  `latest_call_count=1`, `latest_page_count=1`,
  `latest_terminal_reason=short_page`, and
  `rate_limit_health.endpoint_counts.fetch_my_trades_first_symbol=1`.

### PR #751: Ticker Probe Endpoint Latency Health

- Branch: `codex/v8-ticker-probe-endpoint-latency-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `endpoint_latency_health` from already-recorded outcomes. The summary groups
  endpoint attempts, including open-orders fallback attempts and fill-history
  pages, by endpoint/category with success/failure counts, latency summaries,
  error-type counts, and slowest endpoint metadata.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A settled smoke at `4eef3572` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `endpoint_latency_health.endpoint_count=11`, `total=12`, `succeeded=11`,
  `failed=1`, `slowest.endpoint=load_markets`, and the expected Binance
  all-symbol open-orders warning as one `fetch_open_orders` failure while
  account-critical health remained successful through symbol fallback.

### PR #753: Ticker Probe Exchange Surface Health

- Branch: `codex/v8-ticker-probe-exchange-surface-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `exchange_surface_health` from already-recorded open-orders, time-sync,
  fill-history, and OHLCV-tail outcomes. The summary adds exchange/user notes
  for surface quirks such as open-orders symbol fallback, unsupported time sync,
  fill-history terminal pagination reason, and OHLCV tail shape without adding
  exchange calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A one-repeat authenticated Binance probe for `BTC/USDT:USDT`
  validated `exchange_surface_health.notes=[fill_history_short_page,
  open_orders_all_symbols_failed, open_orders_symbol_fallback_required]`,
  `open_orders.mode_counts.symbol_fallback=1`,
  `fill_history.terminal_reasons.short_page=1`, and collection-level exchange
  note counts. A settled 5-minute smoke at `0f1afc49` reported all five
  expected bots running, no hard failures, no log hard/attention matches, no
  failed remote calls, no failed account-critical remote calls, and clean
  tracked repository state.

### PR #755: Live Smoke EMA Readiness Health

- Branch: `codex/v8-smoke-ema-readiness-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `ema_readiness_health` full/summary groups and brief `ema_readiness`
  counters from existing `ema.unavailable` events. The new projection reports
  event count, affected bot count, latest candidate/unavailable totals, bounded
  reason counts, error-type counts, latest cycle IDs, and compact allowlisted
  EMA event data without changing smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and local real-data
  summary/brief smoke checks passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  smoke-report tooling. The first 10-minute smoke after deploy caught a real
  Binance `InvalidNonce`/timestamp-window recovery in authoritative positions
  and the related text-log warning; later authoritative calls succeeded. A
  settled 2-minute brief smoke at `5d9f3a5f` reported all five expected bots
  running, no hard failures, no log hard/attention matches, no failed remote
  calls, no failed account-critical remote calls, and clean tracked repository
  state. The new `ema_readiness` counters reported `total=11`, `bots=4`,
  `latest_candidate_unavailable_total=31`, and `latest_unavailable_total=112`,
  making persistent non-hard EMA readiness degradation visible without full
  problem-event inspection.

### PR #756: EMA Gate Modes For Trailing Martingale Entries

- Branch: `codex/v8-ema-entry-gate-mode`.
- Scope: adjacent Rust strategy/runtime behavior, not a logging-overhaul slice.
- Result: trailing-martingale entry EMA gating is now a fixed config policy
  with `disabled`, `initial`, `reentry`, and `all` modes. The default remains
  `initial`; unstuck EMA gating also gained an explicit fixed toggle defaulting
  to enabled.
- Review evidence: Claude and Hermes approved; CI was green; the PR author ran
  `cargo check`, rebuilt the Rust extension, ran targeted Python suites, and
  added real backtest/optimize smoke evidence. My final read agreed with the
  reviewers that Claude's remaining notes were future-maintenance concerns, not
  current merge blockers.
- VPS5 evidence: deployed after PR #757. The VPS checkout pulled to
  `19b34138`; the Rust extension was rebuilt in `/root/passivbot/venv` with
  `PATH=/root/.cargo/bin:/root/passivbot/venv/bin:$PATH` and
  `VIRTUAL_ENV=/root/passivbot/venv`. Bots were then restarted from
  `/root/bots_vps5.yaml` and left running. An immediate 3-minute smoke and a
  settled 5-minute smoke both reported all five expected bots matched, no hard
  failures, no log hard/attention matches, no failed remote calls, no failed
  account-critical calls, and clean tracked repository state. The settled smoke
  at `19b34138` reported `remote_calls.total=336`,
  `account_critical_remote_calls.total=46`, and only non-hard EMA readiness
  attention (`ema_readiness.total=4`, `bots=1`,
  `latest_candidate_unavailable_total=0`).

### PR #759: Live Smoke Staged Readiness Health

- Branch: `codex/v8-smoke-staged-readiness-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `staged_readiness_health` full/summary groups and brief `staged_readiness`
  counters from existing staged `cycle.degraded` events. The new projection
  reports affected bot count, latest missing/invalid staged-surface totals,
  missing/invalid surface groups, completed-candle mismatch counts, latest
  cycle IDs, and compact allowlisted cycle-degraded event data without changing
  smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and a local real-data
  brief smoke projection passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  smoke-report tooling. A 5-minute brief smoke at `74a52ede` reported all five
  expected bots running, no hard failures, no log hard/attention matches, no
  failed remote calls, no failed account-critical calls, and clean tracked
  repository state. The new `staged_readiness` counters reported `total=4`,
  `bots=1`, `latest_missing_surface_total=1`, and
  `latest_invalid_surface_total=1`, making staged `completed_candles` style
  readiness degradation visible without full problem-event inspection.

### PR #760: Staged Readiness Deploy Progress

- Branch: `codex/v8-progress-after-staged-readiness-smoke`.
- Scope: docs-only progress update.
- Result: recorded PR #759 review, merge, deploy, and VPS5 smoke evidence in
  this progress ledger and the live-ops backlog.
- Review evidence: Claude and Hermes approved; CI was green.
- VPS5 evidence: pulled to `31d42ea3` without bot restart because the slice is
  docs-only. The first 5-minute smoke was red from real HSL ZEC long RED
  finalizations on OKX, GateIO, and Binance, not from the docs change; the
  structured `risk_events` section and text-log hard matches both surfaced the
  RED finalizations. A settled 2-minute follow-up smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `logs.attention_matches=0`,
  `matched_expected=5`, `missing_expected=[]`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining non-hard attention was
  visible through `ema_readiness` and `staged_readiness`; the settled
  `staged_readiness` projection reported `total=5`, `bots=3`,
  `latest_missing_surface_total=3`, and `latest_invalid_surface_total=3`.
  A later 5-minute smoke still reported `ok=true`, all five bots matched, no
  log matches, no failed remote/account-critical calls, and clean repository
  state, while `staged_readiness` had grown to `total=17`, `bots=4`,
  `latest_missing_surface_total=5`, and `latest_invalid_surface_total=5`.

### PR #762: Completed Candle Fallback Shape Recovery

- Branch: `codex/v8-staged-readiness-target-change`.
- Scope: narrow staged-readiness runtime fix driven by the PR #759/#760 smoke
  signal.
- Result: completed-candle preconditions now compare canonical
  `(symbol, completed_timestamp)` targets instead of exact signature tuple
  shape, so a stamped bounded `tail_gap_fallback` signature may recover to
  normal cache coverage without causing a spurious staged-planner defer. Symbol
  set changes, target timestamp changes, and genuinely missing/stale completed
  candles still defer.
- Review evidence: Claude and Hermes approved; CI was green; focused staged
  planner tests, `tests/test_live_smoke_report.py -k staged_readiness`,
  compileall, `git diff --check`, and the full
  `tests/test_passivbot_balance_split.py` file passed locally.
- VPS5 evidence: deployed to `d9188b64` with a bot restart because this changed
  live Python runtime code. Four bots stopped after the first exact-pane
  Ctrl+C; Kucoin needed a second exact-pane Ctrl+C before the stale `passivbot`
  tmux session was killed and reloaded from `/root/bots_vps5.yaml`. Immediate
  2-minute smoke reported `ok=true`, all five bots matched, no hard/log
  failures, no failed remote/account-critical calls, and
  `staged_readiness.total=0`. A settled 5-minute smoke also reported `ok=true`,
  all five bots matched, clean repository state, no hard/log failures, no
  failed remote/account-critical calls, and `staged_readiness.total=0`.

### PR #764: Cache Doctor Metadata Compatibility Evidence

- Branch: `codex/v8-cache-doctor-compat`.
- Scope: adjacent read-only cache diagnostics from the live-ops backlog.
- Result: `passivbot tool cache-integrity-doctor` now reports deeper
  metadata compatibility evidence for local candle/fill/HSL caches. Candle
  `index.json` known-gap metadata is classified by no-trade vs unclassified
  reasons, fill current-contract evidence distinguishes proven coverage from
  current-contract-but-unproven coverage, and HSL/risk metadata reports HSL
  artifact/timestamp compatibility fields. The final follow-up commit made
  mixed no-trade/unclassified candle gaps explicitly partial and still
  `candle_synthetic_no_trade_evidence_unproven`.
- Review evidence: Claude and Hermes approved the final head; CI was green;
  `tests/test_cache_integrity_doctor.py`, compileall, and `git diff --check`
  passed locally for the follow-up.
- VPS5 evidence: deployed as part of merged `v8` `5275ab75` without bot
  restart because this is read-only local tooling.

### PR #765: Live Smoke Event Pipeline Health

- Branch: `codex/v8-smoke-pipeline-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `event_pipeline_health` full/summary groups and brief `event_pipeline`
  counters from existing `health.summary` event-pipeline counters. The new
  projection reports latest queue depth, unfinished queue work, dropped event
  counts, sink-error counts, degraded count, worker-not-alive count, and
  stopping count without changing smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, and `git diff --check`
  passed locally.
- VPS5 evidence: deployed at `5275ab75` without bot restart because this is
  read-only smoke-report tooling. A 5-minute smoke confirmed the new brief
  `event_pipeline` field was present, but no recent matching health-summary
  sample was in that short window. A 30-minute smoke reported
  `event_pipeline.total=1`, `bots=1`, `latest_dropped_total=0`,
  `latest_sink_error_total=0`, `latest_worker_not_alive_count=0`, and clean
  tracked repository state. The same smoke was red from live risk state and
  unrelated runtime events: HSL red/cooldown events, CRITICAL HSL text logs,
  and an earlier Hyperliquid fill-refresh timeout. All five expected bots were
  still running, and remote/account-critical call summaries reported no
  failures.

### PR #767: Live Smoke Risk Log Classification

- Branch: `codex/v8-smoke-risk-log-classification`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now splits text-log attention and
  hard matches into risk/HSL-related and non-risk buckets. Full, summary, and
  brief reports include `risk_attention_matches`, `risk_hard_matches`,
  `non_risk_attention_matches`, and `non_risk_hard_matches`; bounded log match
  samples now include `category=risk|general`. Smoke verdict logic is unchanged:
  risk/HSL CRITICAL lines still count in `hard_matches` and still make smoke
  red.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed locally.
- VPS5 evidence: deployed at `b07d5166` without bot restart because this is
  read-only smoke-report tooling. A 5-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`,
  `logs.risk_hard_matches=0`, `logs.non_risk_hard_matches=0`, all five
  expected bots matched, clean tracked repository state, no failed remote
  calls, and no failed account-critical remote calls. Remaining attention was
  non-hard EMA readiness and HSL status.

### PR #769: Live Smoke Verdict Source Breakdown

- Branch: `codex/v8-smoke-source-breakdown`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now exposes
  `hard_failure_sources` and `attention_sources` in full, summary, and brief
  reports. The source maps identify monitor parse errors, invalid event rows,
  structured hard/problem events, text-log matches, dropped unparsed attention
  matches, and process hard failures. Smoke verdict logic is unchanged:
  `hard_failures`, `attention_count`, and `ok` use the same accounting as
  before.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed locally.
- VPS5 evidence: deployed at `b789e146` without bot restart because this is
  read-only smoke-report tooling. A 5-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`, all five expected bots
  matched, clean tracked repository state, no failed remote calls, no failed
  account-critical remote calls, and no text-log attention or hard matches.
  The remaining attention was explicitly attributed to
  `attention_sources.problem_events=101`, with non-hard EMA readiness and HSL
  status events visible in the existing summaries.

### PR #772: Live Config Cache Readiness Preflight

- Branch: `codex/v8-live-config-cache-readiness`.
- Scope: read-only config preflight tooling.
- Result: `passivbot tool live-config-preflight` now includes config-only
  cache readiness/root-hint reporting for candles, fills, and HSL/risk surfaces,
  plus bounded compare deltas. The report explicitly marks cache artifacts as
  not scanned and startup policy as not enforced, so it does not claim coverage,
  touch local caches, contact exchanges, or change live startup/trading
  behavior.
- Review evidence: Claude and Hermes approved; CI was green; targeted
  `tests/test_live_config_preflight.py`, py_compile/compileall, a compact CLI
  smoke, and `git diff --check` passed.
- VPS5 evidence: deployed at `5fcb39cd` without bot restart because this is
  read-only preflight tooling. A 5-minute summary smoke reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `logs.attention_matches=0`, all five expected bots
  matched, clean tracked repository state, no failed remote calls, and no
  failed account-critical remote calls. Remaining attention came from known
  non-hard EMA readiness and HSL cooldown/status groups.

### PR #775: Event Pipeline Health Aggregation Regression

- Branch: `codex/v8-fake-live-observability-test`.
- Scope: offline regression test and backlog ledger only.
- Result: `tests/test_live_smoke_report.py` now covers multi-bot
  `event_pipeline_health` aggregation from existing `health.summary` events,
  including queue depth, dropped events, sink errors, degraded counts, and
  worker-liveness. The PR changed no production runtime files and does not alter
  event routing, live bots, exchange calls, or trading behavior.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, targeted event-pipeline aggregation tests,
  py_compile/compileall, and `git diff --check` passed.
- VPS5 evidence: not deployed or smoke-tested separately because the merged
  slice changed only tests and `docs/plans/live_ops_improvement_backlog.md`.
  The latest runtime-bearing deploy/smoke evidence remains PR #772 at
  `5fcb39cd`.

### PR #773: Cache Doctor Candle Boundary Gap Summary

- Branch: `codex/v8-cache-doctor-boundary-summary`.
- Scope: read-only cache-integrity tooling and tests.
- Result: `passivbot tool cache-integrity-doctor` now exposes bounded candle
  boundary-gap summaries so operators can distinguish full coverage from
  edge-window gaps without opening cache metadata manually. The slice does not
  change live trading behavior, candle loading behavior, or exchange calls.
- Review evidence: Claude and Hermes approved after the boundary summary fix;
  CI was green.
- VPS5 evidence: deployed as part of merged `v8` `54a909b`. Bots were
  restarted and left running. A 5-minute compact smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, no failed remote or
  account-critical remote calls, `matched_expected=5`, and
  `missing_expected=[]`.

### PR #774: Structured Unstuck Events

- Branch: `codex/v8-unstuck-events`.
- Scope: live event producers for unstuck state transitions and tests.
- Result: unstuck-related live state now emits structured events through the
  event pipeline, preserving best-effort observability semantics and avoiding
  trading-behavior changes. Smoke-report value-safety was fixed before merge.
- Review evidence: Claude and Hermes approved after the value-safety fix; CI
  was green.
- VPS5 evidence: deployed as part of merged `v8` `54a909b`. The same
  post-restart smoke showed all five configured bots running with no hard
  failures, no text-log hard matches, and no failed account-critical remote
  calls. Shutdown/restart events from the deployment were visible through the
  structured smoke summaries.

### PR #778: Live Performance Report

- Branch: `codex/v8-live-performance-report`.
- Scope: read-only operator performance tooling and performance/readiness
  checklist docs.
- Result: `passivbot tool live-performance-report` now scans local monitor
  event NDJSON and aggregates startup, cycle, state-refresh, remote-call, and
  HSL replay timing groups with min/max/mean/median/p95, trading-impact labels,
  bounded group output, path redaction, and no exchange calls or live behavior
  changes.
- Review evidence: Claude and Hermes approved after the path-redaction fix; CI
  was green; targeted performance-report, event-query, and smoke-report tests,
  py_compile, a local compact CLI smoke, and `git diff --check` passed.
- VPS5 evidence: deployed at `fe946c6c` without bot restart because this is
  read-only tooling. A 5-minute summary smoke reported `ok=true`,
  `hard_failures=0`, no text-log hard matches, all five expected bots matched,
  no failed remote calls, and no failed account-critical remote calls. The new
  performance report returned `ok=true`, scanned all current monitor event
  streams, and surfaced multi-thousand-second startup/full-warmup/HSL groups as
  the dominant current performance gap.

### PR #779: Live Performance Report Summary Filters

- Branch: `codex/v8-live-performance-report-summary`.
- Scope: read-only performance-report filtering, summary projection, tests, and
  docs.
- Result: `passivbot tool live-performance-report` now supports
  `--summary`, `--bot`, `--exchange`, and `--user`, with explicit
  skipped-event filter accounting and bounded summary output. This keeps
  repeated operator performance checks concise without changing event emission,
  exchange calls, caches, or trading behavior.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact filtered CLI smoke
  passed.
- VPS5 evidence: deployed at `0d742a4f` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned `ok=true`
  and showed HSL/startup latency as the dominant Binance performance cost. A
  5-minute summary smoke reported all five expected bots running with no hard
  failures, no text-log hard matches, and no failed account-critical remote
  calls.

### PR #780: Live Decision Boundary Lag Report

- Branch: `codex/v8-live-performance-decision-lag`.
- Scope: read-only performance-report decision-boundary lag aggregation, tests,
  and docs.
- Result: `passivbot tool live-performance-report` now includes
  `decision_boundary_lag`, aggregating per-bot lag from the relevant whole
  minute boundary to cycle start, Rust call/return, action planning,
  order-wave/write/confirmation events when present, and cycle completion. The
  report keeps cycle ids internal and surfaces only aggregate timing groups.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact filtered CLI smoke
  passed.
- VPS5 evidence: deployed at `f70434f3` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned `ok=true`
  and showed decision-boundary lag groups directly; the same smoke pass kept
  all five configured bots running with no hard failures and no failed
  account-critical remote calls.

### PR #781: Live Input Staleness Report

- Branch: `codex/v8-live-performance-input-staleness`.
- Scope: read-only performance-report input-staleness aggregation, tests, and
  docs.
- Result: `passivbot tool live-performance-report` now includes
  `input_staleness`, aggregating account packet age at snapshot build plus
  snapshot/EMA-bundle age at the Rust call boundary when existing monitor
  events provide enough proof. Joins are keyed by bot plus cycle generation so
  reused cycle IDs after restart do not cross-link old snapshot/EMA state.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact CLI smoke passed.
- VPS5 evidence: deployed at `cdb2f381` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned
  `ok=true` with `input_staleness` groups visible; a settled smoke after a
  transient GateIO timeout reported all five expected bots running with no hard
  failures and no failed account-critical remote calls.

### PR #782: HSL Replay Timing Fields

- Branch: `codex/v8-hsl-replay-timing-fields`.
- Scope: HSL coin replay structured-event payload fields and tests only.
- Result: `hsl.replay.progress`/`completed` payloads now expose bounded replay
  context for performance work: held/cooldown/required pair counts,
  `timeline_rows`, `applied_rows`, `total_applied_rows`, `skipped_pairs` on
  completion, `rows_per_second`, `full_elapsed_s`,
  `startup_blocking_elapsed_s`, and per-pair held/cooldown booleans. The
  current full replay remains startup-blocking; true protective elapsed timing
  still belongs to the future protective/full replay split.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  focused HSL tests, py_compile, `git diff --check`, and silent-handling audit
  of touched files passed locally.
- VPS5 evidence: deployed at `b5e08986` with a controlled restart from
  `/root/bots_vps5.yaml`. Immediate and settled smokes reported all five
  expected bots running with no hard failures and no failed account-critical
  remote calls. Direct event query showed the new HSL replay fields in live
  Binance, GateIO, and OKX replay progress events.

### PR #783: Progress Update After HSL Timing

- Branch: `codex/v8-progress-after-hsl-timing`.
- Scope: docs-only progress/readiness ledger update after PRs #781 and #782.
- Result: progress docs now record the deployed input-staleness and HSL replay
  timing slices, their VPS5 smoke evidence, and the remaining gap that true
  `protective_elapsed_s` still requires a protective/full replay split.
- Review evidence: Claude and Hermes approved with no findings; CI was green.
- VPS5 evidence: deployed at `ef75d210` without bot restart because this was a
  docs-only update. The pull was a clean fast-forward and all five configured
  `passivbot live` processes remained running afterward.

### PR #784: Startup Readiness Performance Summary

- Branch: `codex/v8-live-startup-readiness-summary`.
- Scope: read-only live performance report startup-readiness aggregation,
  docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `startup_readiness`, derived from existing lifecycle/startup timing/HSL replay
  events. The summary is generation-scoped across restarts, bounded by
  `group_limit`, and copies only a fixed whitelist of HSL replay fields.
- Review evidence: CI was green. Hermes approved the fixed head after a
  generation-reset finding was addressed. Claude approved the slice and its
  only bounding/value-safety suggestions were addressed with implementation
  changes and regression tests. Local validation covered performance-report,
  event-query, and smoke-report tests, py_compile, `git diff --check`, a
  silent-handling audit, and a compact filtered CLI smoke.
- VPS5 evidence: deployed at `f763a85a` without bot restart because this is
  read-only report tooling. A focused Binance performance summary returned
  `ok=true` and showed populated `startup_readiness`, including completed HSL
  replay timing and startup phase timings. The immediate smoke caught a real
  transient Binance `InvalidNonce` authoritative open-orders failure; a settled
  2-minute smoke then reported `ok=true`, `hard_failures=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `matched_expected=5`, and `missing_expected=[]`.

### PR #787: Live Performance Slowest Blockers View

- Branch: `codex/v8-live-performance-slowest-blockers`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `slowest_blockers`, a bounded cross-section ranking derived from existing
  performance, decision-boundary, and input-staleness metric groups. The view
  excludes diagnostics-only/observability groups and copies only the existing
  bounded metric fields plus `source_section` and `blocking_scope`.
- Review evidence: CI was green. Claude approved the current head with no
  findings. Hermes approved the equivalent pre-rebase code delta; the final
  rebase only moved the branch over the already-merged progress-doc update.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, silent-handling audit, and a compact
  filtered CLI smoke.
- VPS5 evidence: deployed at `002fb965` without bot restart because this is
  read-only report tooling. A focused Binance performance report returned
  `ok=true` and showed `slowest_blockers` populated, with startup warmup and
  HSL-related timings ranked above lower-impact groups. A 2-minute smoke
  reported `ok=true`, `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`.

### PR #789: Live Performance Resource Pressure Report

- Branch: `codex/v8-live-performance-resource-pressure`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `resource_pressure`, derived only from existing `health.summary` events. The
  section aggregates whitelisted process and event-pipeline fields such as RSS,
  memory percent, open file descriptors, load averages, loop duration, event
  queue depth, dropped-event counters, sink-error counters, degraded count, and
  event-pipeline worker state. It does not surface raw account, balance,
  equity, PnL, or other financial health payload fields.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Claude noted one optional non-blocking efficiency nit: the report accumulator
  stores value lists for min/max/mean, which is acceptable for this one-shot
  offline report and consistent with nearby report accumulators. Local
  validation covered performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, silent-handling audit, and a compact
  filtered CLI smoke.
- VPS5 evidence: deployed at `bc6e7f1d` without bot restart because this is
  read-only report tooling. A compact smoke reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused performance summary showed
  `resource_pressure` populated for GateIO and Hyperliquid with RSS, load
  average, loop duration, event queue, sink-error, and worker-state fields.

### PR #791: Live Performance Shutdown Latency Report

- Branch: `codex/v8-live-performance-shutdown-latency`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `shutdown_latency`, derived only from existing `bot.stopping`,
  `bot.shutdown.stage`, and `bot.stopped` events. The section summarizes
  per-stage cumulative shutdown elapsed time and final total shutdown duration
  while keeping the data out of trading blocker rankings and without copying
  shutdown error text.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, silent-handling audit, and a compact
  CLI performance-report smoke.
- VPS5 evidence: deployed at `a04bc1ed` without bot restart because this is
  read-only report tooling. A compact smoke reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused performance summary showed
  `shutdown_latency` present; it was empty in the recent window because no
  shutdown lifecycle events occurred during that window.

### PR #793: Live Performance Execution Timing Report

- Branch: `codex/v8-live-performance-execution-timing`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `execution_timing`, derived only from existing order-wave, order create/cancel,
  and confirmation events. The section reports bounded exchange-action latency
  groups plus `starts_seen`, `terminals_seen`, `timing_observations`,
  `missing_id_counts`, `unpaired_terminal_counts`, and `pending_start_counts`.
  Pairing keys are used only internally; raw order payloads, action ids, and
  client-order ids are not surfaced.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, and a compact synthetic CLI
  performance-report smoke.
- VPS5 evidence: deployed at `b5fc245b` without bot restart because this is
  read-only report tooling. All five configured `passivbot live` processes
  remained running after pull. A 5-minute smoke reported `hard_failures=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `matched_expected=5`, and `missing_expected=[]`. A 180-minute performance
  summary returned `ok=true` and showed `execution_timing` present but empty
  because no order-wave/write events occurred in that sampled window. Existing
  slowest blockers were input-staleness and cycle-boundary lag groups, not
  execution writes.

### PR #796: Live Performance Readiness Checklist

- Branch: `codex/v8-live-performance-readiness-checklist`.
- Scope: docs-only performance/readiness plan structure.
- Result: `docs/plans/live_performance_readiness_goals.md` now has a short
  current-priority checklist and definition of done for fast but correct
  readiness, HSL protective startup latency, replay profiling, exact
  optimization, checkpoints, warm restarts, shutdown latency, and
  observability-vs-trading boundaries.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  `git diff --check -- docs/plans/live_performance_readiness_goals.md` passed.
- VPS5 evidence: deployed at `8bba0641` without bot restart because this was
  docs-only. A 2-minute brief smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected_count=0`.

### PR #797: HSL Replay Profile Performance Report

- Branch: `codex/v8-live-performance-hsl-profile`.
- Scope: read-only live performance report HSL replay profiling, docs, and
  tests.
- Result: `passivbot tool live-performance-report` now includes
  `hsl_replay_profile`, derived only from existing `hsl.replay.*` events. The
  section whitelists bounded replay metadata and derives estimated dense,
  required, held, and cooldown pair-row work, observed applied rows/progress
  percentage, elapsed timing, and startup-blocking timing when available.
  Trading behavior, exchange calls, event emission, and raw HSL/account payloads
  are unchanged.
- Review evidence: CI was green. Claude approved the rebased head with no
  findings. Hermes approved the identical report slice before the clean rebase;
  the only old-vs-new tree difference was the separately approved PR #796 docs
  context. Local validation covered `tests/test_live_performance_report.py`,
  adjacent event-query and smoke-report tests, py_compile, `git diff --check`,
  a local compact CLI performance-report smoke, and a silent-handling scan of
  touched report/test files.
- VPS5 evidence: deployed at `87f22840` without bot restart because this is
  read-only report tooling. A 2-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected_count=0`. A focused 45-minute performance report returned
  `ok=true` and showed `hsl_replay_profile` populated for Binance, GateIO,
  OKX, and Kucoin. The populated groups made the current dense full-replay
  problem explicit: Binance/GateIO/OKX were still in coin-HSL replay after
  roughly `15-17m`, with about `1.1M-1.25M` estimated dense pair-rows each and
  only about `56-63%` observed work in the sampled window.

### PR #799: Cache Warmup Performance Report

- Branch: `codex/v8-live-performance-cache-warmup`.
- Scope: read-only live performance report cache/warmup projection, docs, and
  tests.
- Result: `passivbot tool live-performance-report` now includes
  `cache_warmup`, derived only from existing `cache.warmup_decision`,
  `cache.load.completed`, and `cache.flush.completed` events. The section
  summarizes bounded warm-cache reuse/cold-path decisions, candle cache
  load/flush rows, reason/source counters, symbol/timeframe samples, and
  elapsed timing where present. Trading behavior, exchange calls, cache
  mutation, event producers, and console routing are unchanged.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Both reviews verified that the section is report-only and uses explicit
  scalar/counter whitelists that exclude raw cache paths, raw payloads, account
  values, and secrets. Local validation covered performance-report,
  event-query, and smoke-report tests, py_compile, `git diff --check`, local
  compact CLI performance-report smoke, and a silent-handling scan of touched
  report/test files.
- VPS5 evidence: deployed at `cb034e82` without bot restart because this is
  read-only report tooling. A 2-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused 5-minute performance report returned
  `ok=true` and showed `cache_warmup` populated for all five bots, with
  `total_events=614`, `cache.load.completed=469`,
  `cache.flush.completed=140`, and `cache.warmup_decision=5`. Sample groups
  showed OKX, Binance, and GateIO warmup cold-path decisions plus bounded
  candle load/flush row counts and elapsed summaries.

### PR #801: Forager EMA Readiness Performance Report

- Branch: `codex/v8-live-performance-forager-ema-readiness`.
- Scope: read-only live performance report forager/EMA readiness projection,
  docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `forager_ema_readiness`, derived only from existing `forager.selection`,
  `forager.feature_unavailable`, `ema.unavailable`, and `ema.fallback_used`
  events. The section summarizes bounded forager selection counts,
  feature-unavailable counts, EMA unavailable reason/error-type counters, EMA
  fallback counters, pside/status/reason-code counters, configured age/budget
  fields where present, and bounded symbol samples. Trading behavior, exchange
  calls, cache mutation, event producers, and console routing are unchanged.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Both reviews verified the section is report-only, standalone, and value-safe.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, a local compact CLI
  performance-report smoke, and a silent-handling scan of touched report/test
  files. Tests explicitly inject and reject raw top scores, raw EMA error text,
  API-key markers, balance/equity fields, raw payload markers, and local paths.
- VPS5 evidence: deployed at `1fc77413` without bot restart because this is
  read-only report tooling. A 10-minute time-windowed smoke reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 10-minute performance
  report returned `ok=true` and showed `forager_ema_readiness` populated with
  `total_events=140`, including `ema.fallback_used=41`, `ema.unavailable=56`,
  and `forager.selection=43`. The section grouped current readiness evidence
  across Binance, GateIO, OKX, and Hyperliquid.

### PR #803: Resource Pressure Percentiles

- Branch: `codex/v8-resource-pressure-percentiles`.
- Scope: read-only live performance report resource-pressure projection, docs,
  and tests.
- Result: `passivbot tool live-performance-report` `resource_pressure` field
  stats now include `count`, `median`, and `p95` in addition to the prior
  latest/min/max/mean values. Integer-only health series remain integer-valued,
  while fractional fields such as load averages and memory percentage keep
  bounded decimal precision. The section continues to derive only from existing
  `health.summary` events and continues to use the existing whitelist of
  process and event-pipeline fields.
- Review evidence: CI was green. Hermes approved current head `80bc42fd` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered performance-report, event-query, and smoke-report tests, py_compile,
  `git diff --check`, a local compact CLI performance-report smoke, and a
  silent-handling scan of touched report/test files. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior changed.
- VPS5 evidence: deployed at `07f8e759` without bot restart because this is
  read-only report tooling. A 5-minute time-windowed smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 30-minute performance
  report returned `ok=true` and showed `resource_pressure` populated with
  `total=8` health summary events across four bots. Sample groups confirmed
  resource fields now include count/latest/min/mean/median/p95/max values
  without surfacing raw account or financial payload fields.

### PR #805: Operation Duration Performance Summary

- Branch: `codex/v8-live-performance-operation-durations`.
- Scope: read-only live performance report operation-duration projection,
  docs, and tests.
- Result: `passivbot tool live-performance-report` now includes an
  `operation_durations` section that collates existing `performance`,
  `decision_boundary_lag`, `input_staleness`, `execution_timing`, and
  `shutdown_latency` timing groups into one bounded table. Each row carries the
  source section, operation category, timing kind, trading-impact label, and
  blocking scope so the operator can compare startup, cycle, state-refresh,
  remote-call, HSL replay, cache, decision-boundary, input-staleness,
  execution, and shutdown delays from one surface. The section reuses existing
  bounded aggregates and does not copy raw event payloads.
- Review evidence: CI was green. Hermes approved current head `4573fe59` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered performance-report, event-query, and smoke-report tests, py_compile,
  `git diff --check`, a local compact CLI performance-report smoke, and a
  silent-handling scan of touched report/test files. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior changed.
- VPS5 evidence: deployed at `3d6e3fa7` without bot restart because this is
  read-only report tooling. A 5-minute time-windowed smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 30-minute performance
  report returned `ok=true` and showed `operation_durations` populated with
  `total_groups=161` across cache, cycle, decision-boundary, input-staleness,
  remote-call, and state-refresh categories. The top observed groups were
  `input_staleness.snapshot_to_rust` delays in the `delays_cycle_decision`
  scope. A follow-up investigation found those multi-minute
  `snapshot_to_rust` durations were a report correlation artifact: planning
  snapshot epochs in `snapshot.built.data.cycle_id` were being treated as live
  event cycle IDs. The report now uses exact envelope cycle IDs when present
  and otherwise falls back to the latest preceding snapshot in the same
  bot/restart scope, with match counters.

### PR #807: Snapshot-to-Rust Correlation Fix

- Branch: `codex/v8-live-performance-snapshot-correlation`.
- Scope: read-only live performance report correlation fix, docs, and tests.
- Result: `passivbot tool live-performance-report` now correlates
  `snapshot_to_rust` timing from `snapshot.built` to
  `rust_orchestrator.called` by exact live-event envelope cycle ID when
  available, and otherwise falls back to the latest preceding snapshot in the
  same bot/restart scope. It also reports exact-match, latest-snapshot-match,
  missing, and ambiguous counters so report consumers can see when fallback
  correlation was used. No event producers, exchange calls, cache mutation,
  readiness gates, console routing, or trading behavior changed.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered the focused live performance report tests,
  py_compile, `git diff --check`, and a compact local report smoke.
- VPS5 evidence: deployed at `2bb89cfa` without bot restart because this is
  read-only report tooling. A 10-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 30-minute performance
  report confirmed corrected `snapshot_to_rust` values on VPS5: Binance p95
  `1449ms`, GateIO p95 `1477ms`, OKX p95 `1712ms`, and Hyperliquid p95
  `564ms`, with `snapshot_to_rust_latest_snapshot_matches=154`,
  `snapshot_to_rust_exact_matches=0`, and one missing snapshot match at the
  time-window boundary.

### PR #811: Legacy Snapshot ID Query Fallback

- Branch: `codex/v8-event-query-snapshot-data-fallback`.
- Scope: read-only `live-event-query` compatibility for legacy
  `snapshot.built` rows written before snapshot IDs were promoted into the
  structured event envelope.
- Result: `_event_ids()` now derives `snapshot_id` from
  `_live_event.data.snapshot_id` when `ids.snapshot_id` is absent, so existing
  ID filters, compact output, timelines, trace summaries, order traces, and
  cycle traces can match older snapshot events. The fallback intentionally does
  not promote `data.cycle_id`, because legacy `snapshot.built.data.cycle_id`
  is a planning snapshot epoch rather than a live cycle ID.
- Review evidence: CI was green. Hermes approved current head `6ca8a602` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered event-query tests, the adjacent CLI dispatch test, py_compile, and
  `git diff --check`. No event producers, exchange calls, cache mutation,
  readiness gates, console routing, or trading behavior changed.
- VPS5 evidence: deployed at `52f85e08` without bot restart because this is
  read-only query tooling. A 5-minute time-windowed smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard EMA readiness problem events.

### PR #812: Event-Query Snapshot Fallback Progress

- Branch: `codex/v8-progress-after-event-query-fallback`.
- Scope: docs-only progress-ledger update for PR #811.
- Result: recorded the legacy snapshot-ID query fallback, its degraded
  low-risk merge gate, and VPS5 smoke evidence in this ledger. No code,
  tooling, event producers, exchange calls, cache mutation, readiness gates,
  console routing, or trading behavior changed.
- Review evidence: CI was green. Hermes approved current head `38da2ccf` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk docs gate.
- VPS5 evidence: deployed at `8e4712f6` without bot restart because this is
  docs-only. A compact smoke after the pull reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  remaining attention only from known non-hard EMA readiness events.

### PR #813: Market Snapshot Staleness Performance Report

- Branch: `codex/v8-market-snapshot-staleness-report`.
- Scope: read-only live performance report input-staleness projection and
  tests.
- Result: `passivbot tool live-performance-report` now derives
  `input_staleness.snapshot_market_stale_count` from existing
  `snapshot.built.data.market_snapshot_summary` rows and adds
  `input_staleness.market_snapshot.configured_excess` timing groups when a
  symbol's observed `max_age_ms` exceeds configured `configured_max_age_ms`.
  The bounded summary includes the stale-count field. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior changed.
- Review evidence: CI was green. Hermes approved current head `a98c168ca` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered the full `tests/test_live_performance_report.py` suite, py_compile,
  `git diff --check`, and a focused silent-handling scan of touched report/test
  files.
- VPS5 evidence: deployed at `11c1a847` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `account_critical_remote_calls.failed=0`, and one non-hard
  general `remote_calls.failed=1`. A focused 30-minute performance report
  returned `ok=true` and showed `snapshots_seen=151`,
  `snapshot_surface_age_rows=906`, `snapshot_market_summaries_seen=151`,
  `snapshot_market_stale_count=0`, and `total_groups=52`; no market snapshot
  excess-age group was present in that window.

### PR #815: Event-Query Trace Taxonomy Summary

- Branch: `codex/v8-event-query-trace-taxonomy`.
- Scope: read-only `live-event-query` trace-summary taxonomy projection and
  tests.
- Result: `passivbot tool live-event-query --trace-summary` now includes
  source, component, tag, exchange, and user counters for matched structured
  live events. Tags are read from existing monitor rows and optional embedded
  live-event tags, deduplicated per event before counting. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, monitor
  writes, or trading behavior changed.
- Review evidence: CI was green. Hermes approved current head `fe9f0fdc` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered `tests/test_live_event_query.py`, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched files.
- VPS5 evidence: deployed at `404063c6` without bot restart because this is
  read-only query tooling. A 3-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.

### PR #810: Snapshot IDs in Diagnostic Events

- Branch: `codex/v8-snapshot-built-envelope-ids`.
- Scope: live diagnostic event correlation IDs and tests.
- Result: `DiagnosticEvent` can now carry `cycle_id` and `snapshot_id`, and the
  live planning path passes the current live cycle ID plus planning snapshot ID
  into `snapshot.built` diagnostics. This narrows the gap between legacy
  diagnostic producers and the structured live-event envelope without changing
  trading behavior.
- Review evidence: CI was green. Local validation covered focused event-bus and
  planning snapshot tests, the adjacent event-bus, balance-split, and
  performance-report suites, py_compile, `git diff --check`, and a source-stamp
  verification of the shared local Rust extension. No order/risk/cache/exchange
  behavior changed.

### PR #817: Event-Query Tag Filtering

- Branch: `codex/v8-event-query-tag-filter`.
- Scope: read-only `live-event-query` tag filtering and tests.
- Result: `passivbot tool live-event-query` and `build_event_report()` now
  accept tag filters so event, timeline, trace-summary, order-trace, and
  cycle-trace views can be scoped by structured live-event tags. The tag filter
  applies to already-persisted event rows only.
- Review evidence: CI was green. Local validation covered
  `tests/test_live_event_query.py`, adjacent CLI dispatch tests, py_compile for
  touched files, `git diff --check`, and a silent-handling scan with no matches.
  No event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, or trading behavior changed.

### PR #819: HSL Replay Smoke Summary

- Branch: `codex/v8-smoke-hsl-replay-summary`.
- Scope: read-only smoke-report summary for existing HSL replay monitor events.
- Result: `passivbot tool live-smoke-report` now includes `hsl_replay_health`
  derived from existing `hsl.replay.started`, `hsl.replay.progress`, and
  `hsl.replay.completed` rows. The summary shows per-bot active/completed/failed
  state, bounded started/loaded/progress/completed samples, dense pair-row work
  estimates, observed work percentage, and elapsed timing. Active replay counts
  as attention, not a hard failure.
- Review evidence: CI was green. Claude approved the initial commit and carried
  approval over a test-only hardening commit. Hermes approved both the initial
  commit and the follow-up delta. Local validation covered the smoke-report
  suite, adjacent CLI dispatch tests, py_compile, `git diff --check`, and a
  silent-handling scan of touched files.
- VPS5 evidence: deployed at `c7bc5924` without bot restart because this is
  read-only smoke tooling. A settled 10-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, and `hsl_replay_health.active_bots=0`. Wider
  20- and 45-minute smokes confirmed the new section captured completed
  coin-HSL startup replays and showed their long elapsed times; the hard
  entries in those wider windows were intended HSL RED ZEC risk events rather
  than software failures.

### PR #821: Event Type Registry Docs

- Branch: `codex/v8-live-event-type-registry-docs`.
- Scope: docs/test-only live event registry synchronization.
- Result: `docs/ai/live_event_registry.md` now documents the stable
  `EventTypes` values in addition to tags and reason codes, and
  `tests/test_live_event_registry_docs.py` asserts that the documented event
  type section exactly matches `src/live/event_bus.py`. This makes event-type
  filters and query-facing names easier to review and prevents future registry
  drift.
- Review evidence: CI was green. Claude approved the initial head and carried
  approval over the clean rebase. Hermes approved the rebased head with no
  findings. Local validation covered the focused registry-doc test and
  `git diff --check`. No runtime code, event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, or trading
  behavior changed.
- VPS5 evidence: deployed at `81baddfa` without bot restart because this is
  docs/test-only. A 10-minute smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  and `hsl_replay_health.active_bots=0`.

### PR #822: Debug Profile Registry Docs

- Branch: `codex/v8-live-event-debug-profile-docs`.
- Scope: docs/test-only live event debug-profile registry synchronization.
- Result: `docs/ai/live_event_registry.md` now documents
  `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`, the `all`/disabled-value semantics,
  and the current bounded debug-profile names. The registry-doc test now asserts
  that the documented profile list matches `LIVE_EVENT_DEBUG_PROFILES`.
- Review evidence: CI was green. Claude approved and Hermes approved with no
  findings. Local validation covered the focused registry-doc test and
  `git diff --check`. No runtime code, event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, or trading
  behavior changed.
- VPS5 evidence: deployed at `09f145f4` without bot restart because this is
  docs/test-only. A 5-minute smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  and `hsl_replay_health.active_bots=0`.

### PR #824: Startup Debug Profile Performance Report

- Branch: `codex/v8-startup-debug-profile-report`.
- Scope: read-only live performance report startup-readiness projection.
- Result: `passivbot tool live-performance-report` now includes
  `startup_readiness.debug_profiles` per bot and
  `startup_readiness.debug_profile_counts` in aggregate, derived from existing
  `bot.started` / `bot.ready` event data. Values are filtered through
  `LIVE_EVENT_DEBUG_PROFILES`, so arbitrary or secret-looking event-data strings
  are ignored instead of reflected into the report.
- Review evidence: CI was green. Claude approved and carried approval over the
  clean rebase. Hermes approved the same own-delta with no findings. Local
  validation covered the full `tests/test_live_performance_report.py` suite,
  py_compile for touched files, `git diff --check`, a silent-handling scan of
  touched files, and a patch-id check proving the rebased own-delta matched the
  reviewed patch. No event producers, exchange calls, cache mutation, readiness
  gates, order/risk logic, console routing, monitor writes, or trading behavior
  changed.
- VPS5 evidence: deployed at `d1b3ca04` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, and `hsl_replay_health.active_bots=0`. A 120-minute
  performance report confirmed the new `debug_profile_counts` field is present,
  with no startup lifecycle rows retained in that current-window report.

### PR #827: Execution Terminal Outcome Report

- Branch: `codex/v8-execution-outcome-report`.
- Scope: read-only live performance report execution-timing projection and
  tests.
- Result: `passivbot tool live-performance-report` `execution_timing` now
  includes `terminal_outcome_counts`, derived only from fixed existing
  execution terminal event types. The counters report bounded labels such as
  `create.succeeded`, `cancel.ambiguous_terminal`, and
  `confirmation.satisfied` even when timing correlation is missing or unpaired.
  No order/action ids, raw order payloads, exchange data, account values, event
  producers, exchange calls, cache mutation, readiness gates, order/risk logic,
  console routing, monitor writes, or trading behavior changed.
- Review evidence: CI was green. Claude approved and Hermes approved with no
  findings. Local validation covered the full
  `tests/test_live_performance_report.py` suite, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched report/test files.
- VPS5 evidence: deployed at `38f3a9e3` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, and `account_critical_remote_calls.failed=0`.

### PR #826: Debug Profile Logging Guide

- Branch: `codex/v8-debug-profile-guide-docs`.
- Scope: docs-only logging-guide alignment.
- Result: `docs/ai/logging_guide.md` now points supported live-event debug
  profile names to `docs/ai/live_event_registry.md`, describes current
  profile-family behavior, and states that debug summaries must not copy raw
  exchange/account payloads, credentials, or unbounded row data. The final
  reviewed patch includes `forager` in the bounded profile-family list.
- Review evidence: CI was green. Claude approved the current rebased head.
  Hermes approved the docs patch after the `forager` reviewer nit was fixed and
  dry-ran the patch into current `origin/v8`. Local validation covered
  `git diff --check`. No runtime code, event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed at `eec38e60` without bot restart because this is
  docs-only. A 5-minute smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  and all hard failure sources at zero.

### PR #829: Account State Change Performance Report

- Branch: `codex/v8-live-performance-state-changes`.
- Scope: read-only live performance report account-state activity projection
  and tests.
- Result: `passivbot tool live-performance-report` now includes
  `account_state_changes`, derived only from existing `fill.ingested`,
  `position.changed`, and `balance.changed` events. The section summarizes
  event counts by bot and event type plus bounded status, reason, symbol,
  pside, side, and component counters. It deliberately ignores event `data`,
  so balances, equity, sizes, prices, PnL, fees, order ids, fill ids, raw
  payloads, and client-order ids are not surfaced.
- Review evidence: CI was green. Claude approved and Hermes approved with no
  findings. Local validation covered the full
  `tests/test_live_performance_report.py` suite, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched report/test files.
  No event producers, exchange calls, cache mutation, readiness gates,
  order/risk logic, console routing, monitor writes, or trading behavior
  changed.
- VPS5 evidence: deployed at `fb2268af` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, and all hard failure sources at zero.

### Draft Slice: Order Wave Console Dedupe

- Branch: `codex/v8-dedupe-order-wave-console`.
- Scope: observability-only console routing cleanup for order-wave lifecycle
  summaries.
- Result: when the structured live-event console path is active, legacy stdlib
  `[order] wave complete` and `[order] wave settled` lines are suppressed so
  operators see the structured execution/confirmation summaries without
  duplicate legacy lines. If the structured console path is unavailable or
  disabled, legacy order-wave lines remain the fallback.
- Local validation: targeted order-wave console tests and live-event console
  formatter tests passed, plus `py_compile` for touched Python files.

### Draft Slice: HSL Replay Stale Smoke Classification

- Branch: `codex/v8-hsl-replay-stale-smoke`.
- Scope: read-only smoke-report refinement for existing HSL replay monitor
  events.
- Triggering evidence: after PR #987 was merged and VPS5 was fast-forwarded to
  `c021376d`, a smoke run stayed non-green because GateIO had one recovered
  `cycle.degraded` from market snapshots aging just past the 10s readiness
  limit, and KuCoin showed an active coin-HSL replay with no completion event.
  Follow-up event queries showed GateIO completed later cycles, while KuCoin
  was still startup-blocked in HSL replay after long candle-history fetches.
- Intended result: make active HSL replay groups easier to classify in smoke
  output by adding stale-progress and long-running active counts from existing
  bounded monitor data. This is observability-only and does not change HSL,
  order, candle, exchange, readiness, or risk behavior.
- Result: PR #988 added the stale and long-running active HSL replay
  classification, was reviewed, merged to `v8`, and deployed to VPS5. The
  first post-deploy smoke helped identify Kucoin as stopped after a terminal
  startup validation failure, which moved the next slice from observability to
  a narrow trading-path fix.

### Critical Live Safety Gap: Coin-HSL Startup Replay Latency

- Discovery: Binance VPS5 startup on 2026-06-26 showed coin-mode HSL history
  reconstruction loading `symbols=24 pairs=24 rows=43201 fills=2704` at
  `16:19:33Z`, then completing at `16:46:37Z` after applying `985965` rows in
  `1623.4s`. The XLM protective panic close was not posted until `16:48:06Z`.
- Current code shape: coin-mode startup blocks before bot READY, builds a dense
  all-symbol minute timeline, then serially replays that timeline once per
  `coin+pside` pair. This makes a held coin wait behind unrelated coins and
  scales roughly as `timeline_minutes * pairs`.
- Priority: this is now the highest-value live safety item outside the logging
  overhaul. The next trading-path PR should preserve exact HSL semantics while
  making currently held positions protective-ready before broad/full replay
  finishes. Fresh initial entries may remain blocked until full replay is
  complete.

### Draft Slice: Coin-HSL Startup Replay Scope Narrowing

- Branch: `codex/v8-hsl-coin-flat-history-upnl`.
- Triggering evidence: after PR #988 was deployed to VPS5 at `d2021419`,
  KuCoin was no longer running. Its latest log ended in terminal startup
  failure during `equity_hard_stop_initialize_coin_from_history`:
  `get_balance_equity_history()['timeline'][]['unrealized_pnl_by_coin_pside']`
  missing required coin HSL symbol `AVAX/USDT:USDT`. The same startup had spent
  more than an hour in coin-HSL history reconstruction and candle fetch locks
  even though KuCoin had no current positions/open orders.
- Root contract adjustment: coin-HSL startup replay should be strict for
  current-position symbols and historical panic-close cooldown symbols, because
  those can affect immediate protective action. Flat non-panic historical fill
  symbols should not block startup or force candle-price replay; they remain
  available to runtime coin-HSL through the PnL manager once the bot is active.
- Intended result: faster and less brittle coin-HSL startup for flat accounts,
  while preserving hard validation for held symbols and cooldown reconstruction.
- Result: PR #989 was reviewed, merged to `v8`, and deployed to VPS5. Kucoin
  was restarted and passed the prior AVAX missing-UPnL failure. HSL price
  replay no longer loaded candle history for flat non-panic symbols, but the
  dense row replay still took about `1237.7s` and the bot reached READY after
  about `1445s`. The next trading-path slice should reduce or bypass the serial
  `timeline_minutes * pairs` replay for immediate protective readiness.

### Draft Slice: Recovered Time-Sync Smoke Classification

- Branch: `codex/v8-smoke-recovered-time-sync`.
- Scope: read-only smoke-report classification for existing
  `cycle.degraded` and `exchange.time_sync` monitor events.
- Triggering evidence: after PR #989 was deployed and Kucoin reached READY,
  VPS5 smoke was temporarily hard-red because the 10-minute window still
  contained a first-cycle `InvalidNonce` `cycle.degraded` event. The following
  `exchange.time_sync` event succeeded, subsequent cycles continued, and the
  same smoke window became green once the recovered event aged out.
- Intended result: keep unrecovered timestamp/nonce cycle errors hard, but
  classify same-cycle successful `exchange.time_sync` recovery as a recovered
  problem event in detailed, summary, and brief smoke output. This changes only
  report classification; it does not add exchange calls, change live recovery,
  or alter trading behavior.
- Result: PR #990 was reviewed, merged to `v8`, and deployed to VPS5. After the
  transient Kucoin maintenance timeout aged out of the requested 10-minute
  window, VPS5 smoke was green on `bed4f285` with all five expected bots
  running and no hard problem/log/process failures.

### Draft Slice: Text Log Match Context

- Branch: `codex/v8-log-match-context`.
- Scope: read-only smoke-report text-log projection for existing log matches.
- Triggering evidence: the post-PR #990 VPS5 smoke briefly stayed hard-red
  because Kucoin logged a transient `maintain_hourly_cycle` exchange timeout
  with traceback fragments. The structured event stream already classified the
  config-refresh failure as a warning, but the hard text-log matches only showed
  `Traceback (most recent call last):` without the timestamped context line in
  the compact match entry.
- Intended result: attach the nearest timestamped log context line to matched
  unparseable traceback/error lines. This should make future smoke reports
  explain what subsystem emitted a hard text-log fragment without suppressing
  or down-classifying the hard match.
- Result: PR #991 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 together with PR #992 at `03f2fc10`.

### Draft Slice: EMA Readiness Reason Smoke Summary

- Branch: `codex/v8-smoke-ema-reason-summary`.
- Scope: read-only smoke-report projection for existing `ema.unavailable`
  events.
- Triggering evidence: current VPS5 smoke was green but still showed non-hard
  EMA-readiness attention. A focused `live-event-query` revealed useful
  structured detail already present in the events, including
  `cache_only_fetch_failed`, `never_fetched_cache_only`, and candidate error
  type groups. Operators should not need a second event-query just to identify
  the dominant EMA-readiness reason in concise smoke output.
- Intended result: aggregate latest EMA-readiness candidate reason counts,
  unavailable reason counts, and candidate error-type counts into the full,
  summary, and brief smoke-report projections. This changes only report output;
  it does not add event producers, exchange calls, readiness gates, EMA
  behavior, order logic, risk logic, or trading behavior.
- Expected validation: focused EMA-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #992 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `03f2fc10`. The first post-deploy smoke showed a
  transient GateIO positions `RequestTimeout` from an existing running bot; a
  follow-up after the event aged out reported `ok=true`, `hard_failures=0`,
  five expected bots matched, clean tracked repository state, zero
  account-critical failures, no hard log matches, and the new EMA-readiness
  reason maps visible in brief output.

### Draft Slice: Remote Call Failure Cause Smoke Summary

- Branch: `codex/v8-smoke-remote-call-failure-summary`.
- Scope: read-only smoke-report projection for existing `remote_call.*`
  health events.
- Triggering evidence: the post-PR #991/#992 VPS5 smoke briefly went hard-red
  from one GateIO `remote_call.failed` / `cycle.degraded` pair. The brief smoke
  showed remote-call failure counts, but identifying that the failing surface
  was `authoritative_positions` and the error type was `RequestTimeout`
  required a separate `live-event-query`.
- Intended result: aggregate failed remote-call reason codes, surfaces, logical
  call kinds, and error types into full, summary, and brief smoke-report
  projections. This changes only report output; it does not add event
  producers, exchange calls, readiness gates, order logic, risk logic, or
  trading behavior.
- Expected validation: focused remote-call smoke-report tests, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #993 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `32e80518`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, clean tracked
  repository state, and non-hard attention from known EMA/HSL status events.

### Draft Slice: HSL Proximity Smoke Summary

- Branch: `codex/v8-smoke-hsl-proximity-summary`.
- Scope: read-only smoke-report projection for existing `hsl.status` risk
  events.
- Triggering evidence: the post-PR #993 VPS5 smoke showed non-hard HSL status
  attention and a `risk_events.hsl_status.closest_to_red` list, but concise
  summary/brief output omitted proximity context. Operators could see which
  symbols were closest to red, but not how close they were without a full event
  query.
- Intended result: include a bounded normalized HSL red-proximity percentage in
  summary and brief closest-to-red samples while continuing to suppress raw
  drawdown-space thresholds/distances and raw drawdown internals.
  This changes only report output; it does not add event producers, exchange
  calls, readiness gates, HSL behavior, order logic, risk logic, or trading
  behavior.
- Expected validation: focused HSL risk smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #994 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `a071492a`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, clean tracked
  repository state, and the new normalized `red_proximity_pct` values visible
  in `risk_events.hsl_status.closest_to_red`.

### Draft Slice: HSL Cooldown Smoke Summary

- Branch: `codex/v8-smoke-hsl-cooldown-summary`.
- Scope: read-only smoke-report projection for existing `hsl.status` cooldown
  events.
- Triggering evidence: the post-PR #994 VPS5 smoke showed
  `risk_events.hsl_status.tier_counts.red > 0`, but `closest_to_red` contained
  only green symbols. A focused `live-event-query` showed RED cooldown
  `hsl.status` events for ZEC with `tier=red`, `reason_code=cooldown_active`,
  `cooldown_remaining_seconds`, and `cooldown_until_ms`, but no drawdown
  distance metrics. Operators should not need an event query to see which RED
  cooldown targets explain the red tier count.
- Intended result: include bounded active HSL cooldown target samples in full,
  summary, and brief smoke-report risk summaries. This changes only report
  output; it does not add event producers, exchange calls, readiness gates, HSL
  behavior, order logic, risk logic, monitor writes, console routing, or
  trading behavior.
- Expected validation: focused HSL cooldown smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #995 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `1a39d587`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked
  repository state, and active HSL cooldown targets visible in
  `risk_events.hsl_status.cooldown_active`.

### Draft Slice: Staged Readiness Surface Smoke Summary

- Branch: `codex/v8-smoke-staged-surface-summary`.
- Scope: read-only smoke-report projection for existing staged-readiness
  `cycle.degraded` events.
- Triggering evidence: the post-PR #995 VPS5 smoke showed
  `staged_readiness.total=1` and `latest_missing_surface_total=1`, but the
  brief output did not name which surface was missing. A focused
  `live-event-query` showed KuCoin deferred Rust order calculation with
  `missing=["completed_candles"]` and
  `defer_reason=staged_planner_inputs_not_fresh`.
- Intended result: include bounded missing/invalid staged-readiness surface
  maps in full, summary, and brief smoke-report output. This changes only
  report output; it does not add event producers, exchange calls, readiness
  gates, staged execution behavior, order logic, risk logic, monitor writes,
  console routing, or trading behavior.
- Expected validation: focused staged-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #996 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `fae2b0b8`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked
  repository state, and the five configured live bots still running.

### Draft Slice: HSL Raw-Red-Pending Smoke Summary

- Branch: `codex/v8-smoke-hsl-raw-red-pending-summary`.
- Scope: read-only smoke-report projection for existing `hsl.raw_red_pending`
  events.
- Triggering evidence: the post-PR #996 VPS5 smoke showed
  `problem_events.event_types.hsl.raw_red_pending=1`, but the concise
  `risk_events` section did not identify the pending target. A focused
  `live-event-query` showed KuCoin `ASTER/USDT:USDT` long with
  `reason_code=hsl_raw_red_pending_ema_confirmation`, `tier=yellow`, raw
  drawdown already beyond red, and EMA drawdown still below red by
  `ema_gap_to_red`.
- Intended result: include bounded HSL raw-red-pending target samples in full,
  summary, and brief smoke-report risk output, with normalized
  `red_proximity_pct` and `ema_gap_to_red_pct` but without raw drawdown,
  threshold, balance, or payload internals. This changes only report output; it
  does not add event producers, exchange calls, readiness gates, HSL behavior,
  order logic, risk logic, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused raw-red-pending smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #997 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `f7e49f37`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The raw-red-pending event had
  aged out of the 10-minute smoke window by the deployment check.

### Draft Slice: EMA-Readiness Symbol Samples In Smoke Summary

- Branch: `codex/v8-smoke-ema-symbol-samples`.
- Scope: read-only smoke-report projection for existing `ema.unavailable`
  monitor events.
- Triggering evidence: the post-PR #997 VPS5 smoke still showed EMA-readiness
  attention by reason, but identifying affected symbols required a separate
  `live-event-query`. The underlying `ema.unavailable` events already included
  bounded `candidate_unavailable_groups` and `unavailable_reasons` symbol
  samples.
- Intended result: include bounded symbol samples by EMA unavailable reason in
  full, summary, and brief smoke-report output. The projection must omit
  `example_error` prose and payload extras from the concise fields. This
  changes only report output; it does not add event producers, exchange calls,
  readiness gates, EMA behavior, order logic, risk logic, monitor writes,
  console routing, or trading behavior.
- Expected validation: focused EMA-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #998 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `d5c239e9`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new bounded EMA symbol
  fields were visible in VPS5 brief smoke output.

### Draft Slice: Remote-Call Latency Samples In Brief Smoke

- Branch: `codex/v8-smoke-remote-latency-brief`.
- Scope: read-only brief smoke-report projection over existing
  `remote_call_health.groups`.
- Triggering evidence: post-PR #998 VPS5 brief smoke showed healthy remote-call
  counts but hid latency details, while summary output showed slow surfaces
  such as account-critical open-orders calls above 16s and candle remote fetch
  groups above 30s. Operators had to run the larger summary to see which
  bot/surface was slow.
- Intended result: add a bounded `slowest` list to brief `remote_calls` and
  `account_critical_remote_calls`, containing bot, kind/surface, count,
  failed/throttled counts when nonzero, max/p95/latest elapsed milliseconds,
  and latest symbol when present. This changes only report output; it does not
  add event producers, exchange calls, remote-call behavior, readiness gates,
  order logic, risk logic, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused brief smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #999 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `8e6aaf4b`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new brief `slowest` rows
  exposed slow surfaces such as OKX candle fetches near 35s and account-critical
  open-orders/balance calls above 10s.

### Draft Slice: Staged Readiness Reason And Timing Smoke Summary

- Branch: `codex/v8-smoke-staged-degraded-timing`.
- Scope: read-only smoke-report projection over existing `cycle.degraded` and
  `planning.unavailable` staged-readiness events.
- Triggering evidence: post-PR #999 VPS5 brief smoke showed
  `staged_readiness.total=1` and `latest_missing_surface_total=0`, while a
  focused event query showed KuCoin degraded because Rust order calculation was
  deferred with `defer_reason=staged_planner_inputs_not_fresh`,
  `reason_code=staged_execution_precondition`, and long `timings_ms` such as
  `market_state=303339`. The brief smoke did not surface those reason/timing
  fields, so the operator still needed a separate event query to understand why
  planning degraded.
- Intended result: include bounded reason-code counts, latest defer-reason
  counts, latest context counts, and max latest timing fields in full, summary,
  and brief staged-readiness smoke output. This changes only report output; it
  does not add event producers, exchange calls, readiness gates, staged
  execution behavior, order logic, risk logic, monitor writes, console routing,
  or trading behavior.
- Expected validation: focused staged-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1000 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `1a73776f`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. No staged-readiness events
  appeared in the post-deploy smoke window, so the new fields were not
  exercised live yet; the test fixture covers both `cycle.degraded` and
  `planning.unavailable` shapes.

### Draft Slice: HSL Replay Active Samples In Brief Smoke

- Branch: `codex/v8-smoke-hsl-replay-active-samples`.
- Scope: read-only brief smoke-report projection over existing
  `hsl_replay_health.groups`.
- Triggering evidence: after PR #1000 deployment, VPS5 smoke remained
  `ok=true` with zero hard failures, but HSL replay stayed active on four bots
  and became long-running on all four. Brief smoke showed only aggregate counts,
  max elapsed/event-age values, and `active_stage_counts={"pair_replay": 4}`.
  A focused `live-event-query` was still required to learn which bot, symbol,
  pair index, progress, and ETA were responsible.
- Intended result: add a bounded `active` list to brief `hsl_replay`, including
  bot, stage, signal mode, symbol/pside, latest elapsed/event age, stale and
  long-running flags, pair counters, row progress, row rate, observed work
  percentages, and estimated remaining rows/time. This changes only report
  output; it does not add event producers, exchange calls, HSL replay behavior,
  startup gating, order logic, risk logic, monitor writes, console routing, or
  trading behavior.
- Expected validation: focused HSL replay smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1001 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `1c6f5882`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new brief `active` rows were
  visible for OKX, KuCoin, Binance, and Gateio. Follow-up evidence showed the
  primary required-work estimate could be `0` while dense pair replay was still
  active, so the next slice should expose dense and required remaining work
  separately.

### Draft Slice: HSL Replay Dense Progress In Brief Smoke

- Branch: `codex/v8-smoke-hsl-replay-dense-progress`.
- Scope: read-only brief smoke-report projection over existing
  `hsl_replay_health.groups` and `latest.derived` fields.
- Triggering evidence: after PR #1001 deployment, VPS5 brief smoke remained
  `ok=true` with zero hard failures and the active HSL replay rows were visible,
  but several rows showed `estimated_remaining_rows=0` while the stage remained
  `pair_replay` and dense progress was well below 100%. The primary estimate was
  reflecting required-position work, not dense replay work.
- Intended result: add bounded dense-vs-required remaining row/time fields to
  active HSL replay rows: `estimated_dense_remaining_rows`,
  `estimated_dense_remaining_ms`, `estimated_required_remaining_rows`, and
  `estimated_required_remaining_ms`. This changes only report output; it does
  not add event producers, exchange calls, HSL replay behavior, startup gating,
  order logic, risk logic, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused HSL replay smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1002 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `433ec363`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new active rows clearly
  exposed dense replay work still remaining while required replay work was
  already complete. Follow-up evidence showed the aggregate
  `max_active_estimated_remaining_rows/ms` fields still reflected only the
  primary required estimate and could therefore remain zero during active dense
  replay.

### Draft Slice: HSL Replay Dense Max Aggregates In Brief Smoke

- Branch: `codex/v8-smoke-hsl-replay-dense-aggregates`.
- Scope: read-only brief smoke-report projection over existing
  `hsl_replay_health.groups` and `latest.derived` fields.
- Triggering evidence: after PR #1002 deployment, VPS5 brief smoke showed four
  active long-running HSL pair-replay workers. Each active row had
  nonzero `estimated_dense_remaining_rows/ms`, but the top-level
  `max_active_estimated_remaining_rows=0` and
  `max_active_estimated_remaining_ms=0` because those legacy aggregate fields
  use the primary required-position estimate.
- Intended result: keep the legacy primary max fields for compatibility, and
  add dense-vs-required aggregate max fields:
  `max_active_estimated_dense_remaining_rows`,
  `max_active_estimated_dense_remaining_ms`,
  `max_active_estimated_required_remaining_rows`, and
  `max_active_estimated_required_remaining_ms`. This changes only report
  output; it does not add event producers, exchange calls, HSL replay behavior,
  startup gating, order logic, risk logic, monitor writes, console routing, or
  trading behavior.
- Expected validation: focused HSL replay smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1003 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `c0238f0d`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, and the five configured live bots still running. The new dense max
  aggregate fields were visible: dense remaining rows/time were nonzero while
  the legacy primary required-work max fields correctly remained zero. A wider
  45-minute smoke window also exposed an older KuCoin NEAR HSL RED hard text
  match; finding the actual log lines required rerunning the larger summary
  report.

### Draft Slice: Brief Log Match Samples

- Branch: `codex/v8-smoke-brief-log-match-samples`.
- Scope: read-only brief smoke-report projection over existing sanitized text
  log matches.
- Triggering evidence: after PR #1003 deployment, the short post-deploy smoke
  was green, but a wider window reported `log_hard_matches=2` from an older
  KuCoin NEAR HSL RED event. The brief output showed only counts, so the
  operator had to rerun summary sections to identify the log path, line, and
  text.
- Intended result: keep the full `matches` list out of `--brief`, but include
  bounded `hard_samples` and `attention_samples` under `logs`, sourced from the
  same redacted match objects already emitted by the summary report. This
  changes only report output; it does not add log scanning, event producers,
  exchange calls, HSL behavior, order/risk logic, monitor writes, console
  routing, or trading behavior.
- Expected validation: focused log-sample smoke-report tests, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1004 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `291a8711`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, and the five configured live bots still running. A wider logs-only
  smoke verified the new bounded `hard_samples`/`attention_samples` fields and
  showed basename-only log paths for recent HSL RED log lines. The same smoke
  also showed several recent HSL RED/cooldown/mode events in structured
  `risk_events`, but brief output still required reading specialized HSL
  status fields or a summary rerun to see the latest risk event rows.

### Draft Slice: Brief Risk Event Samples

- Branch: `codex/v8-smoke-brief-risk-event-samples`.
- Scope: read-only brief smoke-report projection over existing summarized
  `risk_events.groups`.
- Triggering evidence: after PR #1004 deployment, VPS5 brief smoke showed
  HSL cooldown and RED context through `risk_events.hsl_status`, but recent
  structured rows such as `hsl.red_finalized_without_order`,
  `risk.mode_changed`, and `unstuck.status` were only visible in the larger
  summary `risk_events.groups` output.
- Intended result: add bounded `risk_events.latest_groups` to brief output with
  safe identifiers only: bot, event type, reason code, status, level,
  symbol/pside, component, count, and latest timestamp. Do not include
  `latest_data` or raw drawdown/balance/order payload fields. This changes only
  report output; it does not add event producers, exchange calls, HSL behavior,
  order/risk logic, monitor writes, console routing, or trading behavior.
- Expected validation: focused risk-event smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Event Query Rotation Warning

- Branch: `codex/v8-event-query-rotation-warning`.
- Scope: read-only `passivbot tool live-event-query` report output and docs.
- Triggering evidence: a VPS5 Kucoin ASTER HSL RED/cooldown incident was
  visible in text logs and recoverable from rotated monitor event segments with
  `--include-rotated`, but a default filtered `live-event-query` scan over
  `current.ndjson` returned zero matches after event rotation. The same query
  also showed scan-order trace bounds when current segments were read before
  older rotated files.
- Intended result: keep current-only directory scans as the default for speed,
  but emit a warning issue when a filtered query skips rotated event segments.
  Also report trace-summary `first_ts`/`last_ts` as chronological min/max
  timestamps rather than scan-order first/last. No event producers, monitor
  writes, exchange calls, or trading behavior change.
- Expected validation: focused `tests/test_live_event_query.py`, `py_compile`,
  `git diff --check`, added-line silent-handling scan, and a read-only VPS5
  rotated query proving the HSL incident is recoverable with `--include-rotated`.
- Result: PR #1022 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `c090fd5b`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, and five configured live bots still running. A read-only Kucoin ASTER
  current-only query now emits `current_only_rotated_segments_skipped`, while
  `--include-rotated` recovers the HSL RED/cooldown incident events.

### Draft Slice: Dropped Unparsed Log Samples

- Branch: `codex/v8-smoke-dropped-unparsed-samples`.
- Scope: read-only `passivbot tool live-smoke-report` summary/brief projection
  over existing log-window dropped-unparsed counters.
- Triggering evidence: after PR #1022 deployment, VPS5 brief smoke was green
  but reported `dropped_unparsed_attention_matches=1` and
  `dropped_unparsed_hard_matches=1` with `--log-window-unparsed-policy drop`.
  The brief output showed the counts but not the dropped line, so an operator
  could not tell whether the dropped hard-looking signal was a stale traceback
  fragment or a fresh line cut off by the log tail.
- Intended result: keep the existing verdict policy and drop behavior, but
  retain bounded redacted samples for contextless attention/hard-looking lines
  dropped by the unparsed-line policy. Project them into summary and brief
  output with basename-only paths in brief, matching existing log sample
  conventions. This changes only report output; it does not add log scanning,
  event producers, exchange calls, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused dropped-unparsed smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, added-line
  silent-handling scan, and a read-only VPS5 smoke showing dropped samples when
  the condition is present.
- Result: PR #1023 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `8aab137c`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, clean tracked repository state, and five
  configured live bots still running. The current VPS5 log window did not
  contain dropped unparsed hard/attention matches after deployment, so the new
  sample fields were absent as expected when the condition is not present.
  The same smoke showed `unstuck.status` in brief `risk_events.latest_groups`,
  but without compact `latest_data`; a follow-up event query showed the
  underlying event had useful allowlisted state such as `changed`,
  `status_counts`, and `over_budget_sides`.

### Draft Slice: Brief Risk Latest Data

- Branch: `codex/v8-smoke-risk-brief-latest-data`.
- Scope: read-only `passivbot tool live-smoke-report --brief` projection over
  existing summarized `risk_events.groups` and `risk_events.attention_groups`.
- Triggering evidence: after PR #1023 deployment, VPS5 brief smoke exposed an
  `unstuck.status` latest group for `hyperliquid/hyperliquid_tradfi`, but the
  brief row did not explain whether the state changed, which status counts were
  present, or whether any side was over budget. A focused `live-event-query`
  showed the existing monitor event already contained safe compact fields:
  `changed=true`, `status_counts`, and `over_budget_sides`, alongside raw
  per-side allowance details that should remain out of brief smoke output.
- Intended result: include only allowlisted `latest_data` keys in brief risk
  groups, covering HSL mode/status/finalization context and unstuck status
  summaries. Continue excluding raw balances, drawdown internals, price-distance
  details, nested per-side allowance maps, and arbitrary event payload keys.
  This changes only report output; it does not add event producers, exchange
  calls, HSL behavior, order/risk logic, monitor writes, console routing, or
  trading behavior.
- Expected validation: focused risk-event smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`,
  added-line silent-handling scan, and a read-only VPS5 smoke after merge
  proving the compact fields appear when the condition is present.
- Result: PR #1024 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `59c36ada` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  and five configured bots still running. The deployed brief `risk_events`
  output now includes allowlisted `latest_data` for HSL cooldown and green
  status rows, proving the compact state projection is active while keeping raw
  drawdown/balance/allowance details out of brief output.

### Draft Slice: Forager Debug Profile

- Branch: `codex/v8-forager-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `forager.selection` and `forager.feature_unavailable` events.
- Triggering evidence: `forager` is already part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface, but the existing forager event emitters did not add any
  profile-specific debug shape when that profile was enabled.
- Intended result: when `forager` debug is enabled, add bounded count and
  key-shape metadata to existing forager events: candidate/eligible/selected
  counts, unavailable sample counts, top-score key shape, slot state, and
  related decision counters. Do not add raw score values beyond the existing
  default bounded top-score sample, do not change default forager events, and do
  not change console output, event routing, selection behavior, exchange calls,
  order/risk logic, monitor writes, or trading behavior.
- Expected validation: focused forager debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1025 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `7f8a1942` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: State Debug Profile

- Branch: `codex/v8-state-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `state.refresh_timing` and `state.refresh_progress` events.
- Triggering evidence: `state` is part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface, but only startup events and live performance reports exposed profile
  state. Existing state-refresh events did not add profile-specific debug shape
  when the profile was enabled.
- Intended result: when `state` debug is enabled, add bounded plan/pending
  counts, surface key lists, slowest refreshed surface, and timing scalar
  summaries to existing state refresh timing/progress events. Do not add raw
  account payloads, exchange responses, credentials, event routes, console
  output, refresh behavior, exchange calls, order/risk logic, monitor writes,
  or trading behavior.
- Expected validation: focused state debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1026 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `2de5a6af` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: Startup Debug Profile

- Branch: `codex/v8-startup-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `bot.startup_timing` events.
- Triggering evidence: `startup` is part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface. Startup lifecycle events already expose enabled profiles and
  performance reports summarize them, but the startup timing events themselves
  did not add profile-specific debug shape when the profile was enabled.
- Intended result: when `startup` debug is enabled, add bounded phase, timing,
  and details-shape metadata to existing startup timing events. Do not duplicate
  raw startup details into the debug block, and do not change default startup
  timing payloads, console output, event routing, startup behavior, exchange
  calls, order/risk logic, monitor writes, or trading behavior.
- Expected validation: focused startup debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1027 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `9c555384` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: Cache Debug Profile

- Branch: `codex/v8-cache-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `cache.load.completed`, `cache.flush.completed`, and `cache.warmup_decision`
  events.
- Triggering evidence: restart/warm-cache startup speed has been a repeated
  live-ops concern, and cache events plus performance-report cache aggregation
  already exist. Unlike `startup`, `state`, `forager`, and other debug-profile
  event families, cache events did not yet support profile-specific bounded
  shape metadata when operators enable a focused profile during restart
  investigations.
- Intended result: when `cache` debug is enabled, add bounded key/count/source
  metadata to existing cache load, flush, and warmup-decision events. Keep raw
  candle rows, file paths, cache contents, event routes, console output, cache
  behavior, startup behavior, exchange calls, monitor writes, order/risk logic,
  and trading behavior unchanged.
- Expected validation: focused cache debug-profile monitor test, live-event
  debug-profile normalization and registry-doc tests, `py_compile`,
  `git diff --check`, and the standard added-line silent-handling scan.
- Result: PR #1042 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `06f04070` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard ZEC HSL cooldown attention.

### Draft Slice: Cache Health Smoke Summary

- Branch: `codex/v8-smoke-cache-health`.
- Scope: read-only smoke-report projection over existing `cache.load.completed`,
  `cache.flush.completed`, and `cache.warmup_decision` events.
- Triggering evidence: `live-performance-report` already summarizes cache
  warmup/load/flush behavior, and PR #1042 made cache events debug-profile
  enriched on demand, but repeated VPS smoke loops still did not expose
  warm-cache reuse, cold-path counts, cache load rows, or cache flush rows
  directly.
- Intended result: add `cache_health` to full/summary smoke output and `cache`
  to brief/section aliases, exposing bounded cache counters and latest compact
  event data. Do not expose raw cache paths, raw cache payloads, candle rows, or
  arbitrary payload keys. Do not add event producers, exchange calls, cache
  behavior, startup behavior, console routing, monitor writes, order/risk logic,
  or trading behavior.
- Expected validation: focused cache smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Event Query Debug Profile Filter

- Branch: `codex/v8-event-query-debug-profile`.
- Scope: read-only operator tooling for the completed live-event debug-profile
  surface.
- Triggering evidence: the documented debug-profile event surface is now
  complete, but querying those events requires remembering the generic
  `--data-eq debug_profile=...` predicate. A first-class filter reduces
  operator friction during incident reconstruction.
- Intended result: add `passivbot tool live-event-query --debug-profile` as a
  shortcut for matching `event.data.debug_profile`, with query metadata
  reporting the selected profile names. Keep existing `--data-eq` behavior
  unchanged. Do not add event producers, exchange calls, monitor writes, console
  routing, startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused API and CLI live-event-query tests, full
  `tests/test_live_event_query.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Incident Bundle Debug Profile Filter

- Branch: `codex/v8-incident-debug-profile-filter`.
- Scope: read-only incident-bundle tooling.
- Triggering evidence: PR #1028 made `live-event-query --debug-profile` a
  first-class filter, but incident bundles still require the generic
  `--data-eq debug_profile=...` predicate to package the same focused evidence.
- Intended result: add `passivbot tool live-incident-bundle --debug-profile`
  and pass it through event reports, problem-event reports, time-window reports,
  and manifest filter metadata. Keep existing `--data-eq` behavior unchanged.
  Do not add event producers, exchange calls, monitor writes, console routing,
  startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused incident-bundle CLI/API test, full
  `tests/test_live_incident_bundle.py`, `py_compile`, `git diff --check`, and
  the standard added-line silent-handling scan.

### Draft Slice: Performance Report Debug Profile Filter

- Branch: `codex/v8-performance-debug-profile-filter`.
- Scope: read-only performance-report tooling.
- Triggering evidence: `live-event-query` and `live-incident-bundle` can now
  scope reports to one debug profile, but `live-performance-report` still
  aggregates all events for timing/readiness summaries even when an operator is
  investigating one enriched profile.
- Intended result: add `passivbot tool live-performance-report --debug-profile`
  and filter events at the same scan boundary as bot/exchange/user filters,
  recording the selected profiles and skipped-event count in report metadata.
  Do not add event producers, exchange calls, monitor writes, console routing,
  startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused performance-report CLI filter test, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

### Draft Slice: Incident Bundle Performance Report Artifact

- Branch: `codex/v8-incident-performance-report`.
- Scope: read-only incident-bundle/performance-report tooling.
- Triggering evidence: performance reports now expose startup, HSL replay,
  readiness, operation duration, and debug-profile scoped timing summaries, but
  incident bundles do not yet package that artifact with the rest of the local
  evidence.
- Intended result: add opt-in
  `passivbot tool live-incident-bundle --performance-report`, writing a
  `performance_report.json` artifact and compact manifest/command summary using
  compatible bundle bounds: time window, include-rotated, event-tail lines,
  max-event-files-per-bot, bot/exchange/user, and debug-profile. Keep it opt-in
  to avoid extra default scan work. Do not add event producers, exchange calls,
  monitor writes, console routing, startup behavior, order/risk logic, or
  trading behavior.
- Expected validation: focused incident-bundle API/CLI tests, full
  `tests/test_live_incident_bundle.py`, `py_compile`, `git diff --check`, and
  the standard added-line silent-handling scan.

### Draft Slice: Restart Smoke Performance Bundle Evidence

- Branch: `codex/v8-restart-smoke-performance-bundle`.
- Scope: read-only restart-smoke plan command generation and operator docs.
- Triggering evidence: PR #1031 made performance reports available inside
  incident bundles, but the restart-smoke plan's post-failure bundle command
  did not request that artifact. Failed restart/smoke investigations are exactly
  where startup, HSL replay, readiness, operation-duration, and resource-pressure
  summaries are most useful.
- Intended result: include `live-incident-bundle --performance-report` in the
  planned post-failure incident bundle command emitted by
  `passivbot tool live-restart-smoke-plan`. The command remains non-executing
  plan output and inherits the existing bounded restart-smoke bundle settings:
  recent window, event tail lines, per-bot event-file cap, log bounds,
  supervisor process checks, `--no-event-segments`, and `--compact`.
  Do not execute restarts, contact exchanges, add event producers, write monitor
  events, alter console routing, or change order/risk/trading behavior.
- Expected validation: focused restart-smoke plan tests, full
  `tests/test_live_restart_smoke_plan.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

## Current Next Steps

1. Prioritize a separate trading-path PR for coin-HSL startup replay latency:
   held-position protective readiness must be bounded, exact where data is
   available, and observable before full historical replay of unrelated coins.
   PRs #819, #988, and #989 improve observability and remove one flat-history
   validation failure, but they do not solve the remaining dense replay
   latency.
2. Continue collecting smoke evidence with the new source breakdown and
   risk-vs-general log-match counters before changing any verdict policy. If
   future HSL RED/cooldown episodes make smoke red, the report can now show
   whether the red state came from structured hard events, risk/HSL log lines,
   non-risk software log failures, monitor parse/row failures, or process
   health.
3. Continue Phase 5/6 by adding the next high-value event producer or debug
   profile slice without increasing default console noise. Likely candidates
   are more order/risk transition coverage or focused profile refinements only
   when live diagnostics need deeper evidence.
4. Continue active read-only exchange health probes beyond account-critical
   basics. PR #701 added account-critical health summaries and PR #703 added
   `--account-only` plus symbol fallback for open-orders. PR #741 added
   clock-skew health. PR #743 added candle freshness health. PR #745 added
   fill-history sample health. PR #747 added rate-limit pressure estimates.
   PR #749 added opt-in bounded fill pagination sampling. PR #751 added
   `endpoint_latency_health` summaries from existing probe outcomes, including
   open-orders fallback attempts and fill-history pages. PR #753 added
   `exchange_surface_health` notes over the existing endpoint outcomes.
   Remaining useful slices should be driven by a concrete live exchange gap
   rather than broad probe expansion.
5. Continue monitoring long-running HSL replay. After PR #1000 deployment, the
   replay was still active and progressing on four bots, but brief smoke did
   not identify the active bot/symbol/pair progress without a separate
   `live-event-query`.
6. Start the live restart/smoke automation slice if operational workflow speed
   becomes the higher leverage next step.
7. Continue cache-doctor refinements in separate adjacent PRs: deeper metadata
   compatibility checks and synthetic/no-trade assumptions.

### Draft Slice: Health Summary CPU Percent

- Branch: `codex/v8-health-cpu-percent`.
- Scope: observability producer plus existing performance-report projection for
  periodic `health.summary` resource-pressure events.
- Triggering evidence: `live-performance-report` already summarizes
  `resource_pressure`, and the performance readiness plan calls for CPU/load,
  RSS, open-FD, event queue, and monitor sink telemetry. RSS, memory percent,
  load averages, open FDs, and event-pipeline counters were already emitted, but
  process CPU percent was still absent from the source event.
- Intended result: add a cached, non-blocking process `cpu_percent` probe when
  `psutil` is available, omit the first priming sample, include subsequent
  samples in `health.summary`, and aggregate them through the existing
  `resource_pressure` performance report field. Do not add exchange calls,
  monitor writes beyond the existing periodic health event, console routing
  changes, order/risk logic, restart behavior, or trading behavior.
- Expected validation: focused health-summary payload test, focused
  live-performance-report resource-pressure test, `py_compile`, `git diff
  --check`, and the standard added-line silent-handling scan.
- Result: PR #1148 was reviewed by Hermes, Claude Opus 4.8, and Grok 4.5 on
  current head `5dc295f8`, merged to `v8` as `63c60022`, and deployed to VPS5.
  Because it changed the live `health.summary` producer, VPS5 was pulled to
  `63c60022` and the five configured bots were restarted from
  `/root/bots_vps5.yaml`. The first bounded smoke was hard-red from a transient
  Kucoin account-critical balance `RequestTimeout`; a later 2-minute smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and only non-hard EMA readiness plus
  one active Kucoin HSL replay attention item. A focused 10-minute
  `resource_pressure` report returned `ok=true` with health summaries from three
  bots and no event-pipeline drops or sink errors, but VPS5's live venv did not
  have `psutil`, so `cpu_percent` was omitted. That packaging gap is being
  handled by the follow-up live requirement slice.

### Draft Slice: Live psutil Requirement

- Branch: `codex/v8-live-psutil-requirement`.
- Scope: live packaging plus progress evidence.
- Triggering evidence: PR #1148 added a cached CPU-percent probe and the
  existing health summary already had optional `psutil`-backed RSS,
  memory-percent, and open-FD probes. VPS5 deploy proved the live venv installed
  from `requirements-live.txt` lacks `psutil`, so those probes safely return
  `None` and the new `cpu_percent` field is not emitted in standard live
  installs.
- Intended result: include the already-pinned `psutil` dependency in
  `requirements-live.txt` so fresh live installs and refreshed VPS live envs can
  emit the resource-pressure probes. Do not change health-summary fallback
  behavior, event schemas, monitor routing, console output, exchange calls,
  order/risk logic, restart behavior, or trading behavior.
- Expected validation: requirement parsing/import smoke, focused health-summary
  CPU probe tests, `py_compile` for the touched live monitor/report modules if
  needed, `git diff --check`, and the standard added-line silent-handling scan.
- Result: PR #1149 was reviewed by Hermes, Claude Opus 4.8, and Grok 4.5 on
  current head `16c05d8`, merged to `v8` as `1bb6f620`, and deployed to VPS5.
  The VPS5 checkout was pulled to `1bb6f620`, the live venv installed
  `psutil==5.9.8` from `requirements-live.txt`, and the five configured bots
  were restarted from `/root/bots_vps5.yaml`. A 2-minute post-restart smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`, with only non-hard EMA readiness,
  state refresh-progress, and active HSL replay attention. A focused
  `health.summary` query then showed `cpu_percent=3.9` for
  `hyperliquid/hyperliquid_tradfi`, and a 30-minute
  `live-performance-report --section resource_pressure` showed
  `cpu_percent.latest=20.4`, `count=2`, and no event-pipeline drops or sink
  errors.

### Draft Slice: Resource Pressure Smoke Projection

- Branch: `codex/v8-smoke-resource-pressure`.
- Scope: read-only smoke-report projection over existing periodic
  `health.summary` resource-pressure fields.
- Triggering evidence: PRs #1148 and #1149 made CPU, memory, RSS, open-FD, and
  load evidence available in deployed `health.summary` events and
  `live-performance-report --section resource_pressure`, but standard repeated
  smoke loops still did not expose those process-pressure signals directly.
- Intended result: add `resource_pressure` to full and summary smoke reports,
  plus a compact `resource_pressure` brief projection and `resources` section
  alias, using only existing `health.summary` events. Keep output bounded to
  per-bot latest values and aggregate latest max/total counters. Do not add
  event producers, exchange calls, monitor writes, console routing, restart
  behavior, order/risk logic, or trading behavior.
- Expected validation: focused smoke-report resource-pressure tests,
  `py_compile`, `git diff --check`, and the standard added-line
  silent-handling scan.
- Result: PR #1150 was reviewed by Hermes, Claude Opus 4.8, and Grok 4.5 on
  current head `c2e68817`, merged to `v8` as `58308f32`, and deployed to VPS5
  without restarting running bots because the slice was read-only report
  projection over existing events. VPS5 was pulled to `58308f32` with
  `--autostash`, preserving a pre-existing local rustfmt-only tracked diff in
  `passivbot-rust/src/equity_hard_stop_loss.rs`. A settled 2-minute smoke with
  dropped-unparsed log policy reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and event-pipeline dropped/sink
  errors at zero; non-hard attention remained EMA readiness, one stale dropped
  traceback sample, and `unstuck.status`. The brief smoke exposed the new
  `resource_pressure` projection with one reporting bot, `cpu_percent` max
  `15.1`, RSS total `61177856`, and open FDs total `16`. A focused 30-minute
  `resource_pressure` section showed four reporting bots, `cpu_percent` max
  `21.5`, RSS total `363417600`, and open FDs total `61`; the section command
  exited red only because the report also carried unrelated hard-event and
  dirty-repository markers.

### Draft Slice: Health Summary Scheduling Lag

- Branch: `codex/v8-health-loop-lag`.
- Scope: observability producer plus existing smoke/performance report
  projections for periodic `health.summary` resource-pressure events.
- Triggering evidence: resource-pressure reports now show CPU/load, memory,
  RSS, open FDs, event queue depth, dropped event counters, and sink errors, but
  the performance checklist still lacked loop-lag-style heartbeat evidence.
  Existing `last_loop_duration_ms` measures the previous cycle body and does
  not prove whether periodic health summaries themselves are being delayed.
- Intended result: add non-negative `health_summary_lag_ms` to
  `health.summary` after the first heartbeat, measuring elapsed time beyond the
  configured health-summary interval. Project it through
  `live-performance-report` resource-pressure stats and `live-smoke-report`
  resource-pressure full/summary/brief output. Do not add exchange calls,
  monitor files beyond the existing periodic health event, order/risk logic,
  restart behavior, or trading behavior.
- Expected validation: focused health-summary payload/scheduler tests, focused
  smoke/performance resource-pressure tests, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

### Draft Slice: Resource Pressure Sample Age

- Branch: `codex/v8-resource-pressure-event-age`.
- Scope: read-only performance-report projection over existing
  `health.summary` resource-pressure events.
- Triggering evidence: PRs #1148 through #1153 made resource-pressure values
  available and visible in performance/smoke reports, but
  `live-performance-report --section resource_pressure` exposed only each bot's
  `latest_ts`, not a directly comparable age. During bounded post-deploy smoke
  windows, operators need to distinguish recent pressure samples from stale or
  absent samples without manually subtracting timestamps.
- Intended result: add `latest_event_age_ms` to each performance-report
  `resource_pressure` group, derived from the report timestamp and the group's
  latest `health.summary` event timestamp. Keep existing field statistics,
  event parsing, monitor writes, console output, smoke-report output, exchange
  calls, restart behavior, order/risk logic, and trading behavior unchanged.
- Expected validation: focused resource-pressure performance-report tests, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.
- Result: PR #1154 was reviewed by Hermes, Claude Opus 4.8, Grok 4.5, and
  Codex on current head, merged to `v8` as `141e88db`, and deployed to VPS5
  without restarting bots because the slice was read-only report projection.
  A bounded smoke reported process/repository health green, and a focused
  30-minute `live-performance-report --section resource_pressure` check proved
  one Hyperliquid group with `latest_event_age_ms=624447`.

### Draft Slice: Resource Pressure Sample Age Aggregate

- Branch: `codex/v8-resource-pressure-age-aggregate`.
- Scope: read-only performance-report projection over existing
  `health.summary` resource-pressure events.
- Triggering evidence: PR #1154 made per-bot sample age visible in
  `live-performance-report`, but operators still need a compact top-level
  maximum age and reporting-bot count to compare the freshness of the whole
  resource-pressure section without scanning all groups.
- Intended result: add aggregate `latest_event_age_ms_max` and
  `latest_event_age_reporting_bots` fields to `live-performance-report`
  `resource_pressure`, derived from existing per-group ages. Keep existing
  field statistics, event parsing, monitor writes, console output, smoke-report
  output, exchange calls, restart behavior, order/risk logic, and trading
  behavior unchanged.
- Expected validation: focused resource-pressure performance-report tests, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.
