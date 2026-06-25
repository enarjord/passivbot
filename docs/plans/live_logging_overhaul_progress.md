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

- `09fd305b` merge of PR #643, `Enrich live health resource summaries`.

VPS5 deployment status:

- Deployed through PR #642 at `ad36d8ea`.
- PR #643 is merged to `v8`; VPS5 pull/restart and health-summary smoke are next.

## Phase Checklist

| Area | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| Phase 0: event contract and routing basics | Done enough to build on | `LiveEvent`, routing, pipeline, monitor-backed sink, schema/query constants | Keep registry stable; avoid ad-hoc event names in new slices |
| Phase 1: event bus around existing structured events | Mostly done | Cycle, data packet, snapshot, planning unavailable, Rust orchestrator, order wave, fill/state events | Continue tightening tests as new producers migrate |
| Phase 2: data gatherer events | Mostly done | Account remote-call cohorts, candle tail/coverage, fill refresh summaries, cache load/flush, warmup/startup timing | Not every exchange/network call is instrumented; richer remote-call payload summaries remain incremental |
| Phase 3: Rust planning and payload refs | Partially done | Rust orchestrator called/returned events, redacted error hardening, action/planning summaries | Full raw-ref retention/debug policy still limited |
| Phase 4: order lifecycle and risk transitions | Mostly done | Order wave lifecycle, create/cancel/confirmation events, HSL/risk mode events | Expand WEL/TWEL/unstuck transition coverage as those paths are touched |
| Phase 5: migrate meaningful text logs | Not started as a dedicated phase | Some noisy EMA console output already reduced; many text logs now have event equivalents | Redesign console as a projection of structured events; migrate high-value stdlib logs first |
| Phase 6: gatekeeper integration | Pending | Gatekeeper remains a planned producer | Instrument gate decisions once gatekeeper work resumes |
| Operator tools | In progress | `live-event-query`, `live-smoke-report`, incident bundle, ID filters | Richer cycle replay and cross-bot incident workflow |
| Operational restart goals | Split to adjacent work | PR #619 shutdown progress; PR #622 warm-cache startup | Continue separate reviewed PRs for shutdown/warmup improvements |

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

## Current Next Steps

1. Pull merged `v8` on VPS5 and restart bots so PR #643 runtime health-summary
   fields are emitted by live processes.
2. Verify `health.summary` includes resource pressure and event-pipeline fields
   without increasing console noise or trading behavior.
3. Continue Phase 5 with a small console-projection/text-log migration slice, or
   add the next high-value operator query/replay helper if live smoke shows that
   debugging still needs better cycle reconstruction.
