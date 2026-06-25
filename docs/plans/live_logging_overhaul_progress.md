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

Last updated: 2026-06-25.

Current `origin/v8` logging-overhaul head:

- `72b3d931` merge of PR #661, `Add live smoke process status`.

VPS5 deployment status:

- Repository pulled through PR #661 at `72b3d931`.
- Bots were left running after the pull; PR #661 was a read-only tooling slice
  and did not require bot restart.
- VPS5 process-status smoke with `/root/bots_vps5.yaml` reported
  `expected_total=5`, `matched_expected=5`, `missing_expected=[]`,
  and `scan_error=null`.
- Overall smoke still returned attention/hard failure because current monitor
  events include Kucoin authoritative balance/positions/open-orders
  `RequestTimeout` failures. The process-status check itself was green.

## Phase Checklist

| Area | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| Phase 0: event contract and routing basics | Done enough to build on | `LiveEvent`, routing, pipeline, monitor-backed sink, schema/query constants | Keep registry stable; avoid ad-hoc event names in new slices |
| Phase 1: event bus around existing structured events | Mostly done | Cycle, data packet, snapshot, planning unavailable, Rust orchestrator, order wave, fill/state events | Continue tightening tests as new producers migrate |
| Phase 2: data gatherer events | Mostly done | Account remote-call cohorts, candle tail/coverage, fill refresh summaries, cache load/flush, warmup/startup timing | Not every exchange/network call is instrumented; richer remote-call payload summaries remain incremental |
| Phase 3: Rust planning and payload refs | Partially done | Rust orchestrator called/returned events, redacted error hardening, action/planning summaries | Full raw-ref retention/debug policy still limited |
| Phase 4: order lifecycle and risk transitions | Mostly done | Order wave lifecycle, create/cancel/confirmation events, HSL/risk mode events | Expand WEL/TWEL/unstuck transition coverage as those paths are touched |
| Phase 5: migrate meaningful text logs | Partially started | Some noisy EMA console output already reduced; PR #646 improves event-projected console summaries for already-routed execution events | Migrate high-value stdlib logs to structured-event projections without increasing console noise |
| Phase 6: gatekeeper integration | Pending | Gatekeeper remains a planned producer | Instrument gate decisions once gatekeeper work resumes |
| Operator tools | In progress | `live-event-query`, trace summaries, order trace reconstruction, cycle trace reconstruction, `live-smoke-report` startup baselines and process liveness, incident bundle trace/process reports, ID filters | Cross-bot incident workflow and safe restart orchestration |
| Operational restart goals | Split to adjacent work | PR #619 shutdown progress; PR #622 warm-cache startup; PR #656 cache integrity smoke doctor | Continue separate reviewed PRs for shutdown/warmup/cache proof improvements |

## Merged Slices

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

## Current Next Steps

1. Continue Phase 5 by migrating one high-value stdlib text log family to
   structured-event projection without increasing default console noise.
2. Add read-only exchange health probes or smoke summaries for account-critical
   endpoint timeouts, especially after the latest Kucoin authoritative-fetch
   timeouts on VPS5.
3. Start the live restart/smoke automation slice if operational workflow speed
   becomes the higher leverage next step.
4. Continue cache-doctor refinements in separate adjacent PRs: cache-family
   metadata, coverage windows, suspicious gaps, and warm-cache readiness.
