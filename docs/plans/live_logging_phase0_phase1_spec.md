# Live Logging Phase 0/1 Implementation Spec

This is the pre-merge implementation handoff for the first logging-overhaul
branch. It expands `docs/plans/live_logging_overhaul_plan.md` into concrete
Phase 0/1 work only. It should be implemented after
`codex/v8-fill-history-coverage-bootstrap` is accepted into `v8`, on a fresh
branch from updated `v8`.

## Scope

Goal: introduce one behavior-neutral live event pipeline and route a small first
event set through it.

Do not migrate broad logging call sites in Phase 1. Do not change trading
decisions, freshness gates, order generation, exchange writes, or risk behavior.

## Current Implementation Status

The initial `codex/v8-live-event-pipeline-phase1` branch is Phase 1a, not the
full Phase 1 target. It currently delivers the event envelope, route table,
redaction helpers, bounded pipeline, monitor-backed diagnostic adapter, monitor
publisher thread-safety, shutdown flush/close, and tests for monitor relay/TUI
compatibility.

Still pending before calling Phase 1 complete:

- a first-class structured NDJSON sink for full `LiveEvent` envelopes
- console sink wiring from the event stream
- lifecycle, cycle, Rust-orchestrator, and order-wave instrumentation
- top-level `cycle_id` propagation for reconstructing a full order wave
- broad migration of existing direct monitor/log call sites

## Module Layout

Preferred first layout:

- `src/live/event_bus.py`
  - `LiveEvent`
  - `LiveEventContext`
  - `LiveEventPipeline`
  - `emit_event(...)`
  - route lookup and throttling helpers
- `src/live/event_sinks.py` only if `event_bus.py` becomes too large during
  implementation.
- `tests/test_live_event_bus.py`
  - envelope serialization
  - route decisions
  - redaction
  - queue overflow
  - sink degradation
- `tests/test_live_event_pipeline_integration.py`
  - fake-live cycle chain through the first instrumented events.

Keep `passivbot.py` integration thin. Live code should call a facade or domain
helper, not sink implementations.

## Phase 0: Contract And Routing Table

Phase 0 is docs/schema/test scaffolding. No runtime behavior change.

Deliverables:

1. `LiveEvent` dataclass or typed mapping with stable fields:
   - `schema_version`
   - `event_id`
   - `event_type`
   - `ts_ms`
   - `monotonic_ms`
   - `level`
   - `source`
   - `component`
   - `tags`
   - `exchange`
   - `user`
   - `bot_id`
   - `symbol`
   - `pside`
   - `side`
   - `order_id`
   - `client_order_id`
   - `cycle_id`
   - `snapshot_id`
   - `plan_id`
   - `action_id`
   - `order_wave_id`
   - `remote_call_id`
   - `remote_call_group_id`
   - `status`
   - `reason_code`
   - `message`
   - `data`
   - `raw_ref`
   - `raw_hash`
2. `LiveEventContext` with immutable/copy-on-write helpers for adding ids.
3. Event registry constants for the Phase 1 event set:
   - `bot.started`
   - `bot.ready`
   - `bot.stopping`
   - `bot.stopped`
   - `cycle.started`
   - `cycle.completed`
   - `cycle.degraded`
   - `data_packet.updated`
   - `snapshot.built`
   - `planning.unavailable`
   - `rust_orchestrator.called`
   - `rust_orchestrator.returned`
   - `order_wave.completed`
   - `sink.degraded`
4. Routing table mapping event type/level to:
   - console projection on/off
   - structured NDJSON sink on/off
   - monitor projection on/off
   - text logfile projection on/off
   - throttle key and interval
   - raw payload policy
5. Redaction helper and tests. Default deny:
   - API keys
   - signatures
   - auth headers
   - passwords/passphrases
   - cookies
   - raw account identifiers not already user-facing.

## Phase 1: Event Bus Around Existing Structured Events

Phase 1 adds runtime plumbing but remains behavior-neutral.

Deliverables:

1. `LiveEventPipeline`
   - synchronous `emit(...)` facade
   - bounded queue for disk/monitor sinks
   - dedicated writer thread
   - nonblocking or bounded-time enqueue
   - drop counters by route/event type
   - `sink.degraded` when queue or disk fails
   - clean shutdown/flush method with short timeout
2. Structured sink backed by current monitor storage:
   - write `LiveEvent` envelopes to `monitor/<exchange>/<user>/events/current.ndjson`
   - preserve rotation, compression, retention, manifest, and max-total-byte
     behavior from `MonitorPublisher`
   - keep compatibility for existing monitor relay consumers during migration
3. Compatibility adapter for `DiagnosticEvent`:
   - `kind` -> `event_type`
   - `payload` -> `data`
   - existing `symbol`/`pside` passthrough
   - existing monitor output must still be visible to relay/TUI/web.
4. Console summary sink for the first small event set:
   - lifecycle
   - cycle started/completed/degraded
   - planning unavailable summary
   - Rust orchestrator summary
   - order wave completed summary
   - sink degraded.
5. First instrumentation points:
   - live startup/stopping/stopped lifecycle
   - planning cycle start/end
   - current `data_packet.updated`
   - current `snapshot.built`
   - current `planning_unavailable`
   - Rust orchestrator input/output summaries and hashes
   - order wave summary.

## Raw Payload Policy

Default:

- store summary and hash only
- no full raw exchange payloads
- no full raw Rust payloads

DEBUG or targeted diagnostic mode:

- allow redacted raw Rust input/output refs with short retention
- allow selected raw exchange refs only for explicit targets
- enforce byte caps before writing.

TRACE/firehose:

- allowed only with explicit config/env opt-in
- short retention
- clear `sink.degraded` if raw refs are dropped due to caps.

## Default Budgets

Initial defaults per bot:

- structured event stream: 100-250 MB
- human text projection: 25-50 MB
- raw refs: disabled; when enabled, 50-100 MB with short retention
- queue size: start conservative and test with overflow, for example 10k events
  or a byte-budgeted equivalent.

All limits must be configurable after Phase 1 stabilizes. Phase 1 may use
defaults plus environment overrides to avoid config schema churn.

## Console Contract

Console is an operator view, not the source of truth.

Show at INFO:

- lifecycle
- fills and realized PnL
- position/balance changes
- order create/cancel/results
- HSL/WEL/TWEL/unstuck transitions
- staged entries blocked by high-level gates
- trailing orders waiting for threshold/retracement
- compact forager selection/unavailable summaries
- terminal errors and degraded states.

Hide at INFO:

- per-span EMA diagnostics
- full exchange payloads
- full Rust payloads
- routine poll/fetch loops
- repeated candidate-only forager misses
- high-frequency cache maintenance without state change.

## Test Acceptance

Unit tests:

- event envelope serializes to stable JSON
- default ids/timestamps are present
- context propagation copies ids correctly
- route table sends events to expected sinks
- redaction removes sensitive keys recursively
- raw refs are disabled by default
- queue overflow increments counters and emits/degrades observably
- sink write failure does not raise into trading caller.

Integration tests:

- fake cycle emits a coherent chain by `cycle_id`
- existing `DiagnosticEvent` emits equivalent monitor-compatible events
- structured NDJSON segments are valid after rotation
- console projection snapshots match expected compact strings
- shutdown flush does not hang past timeout.

Manual VPS smoke after implementation:

- DEBUG off console remains tail-safe
- CPU does not materially regress on 1 vCPU
- monitor relay still serves `/health`, `/snapshot`, `/dashboard`, and `/ws`
- one order wave can be reconstructed from structured events by `cycle_id`.

## First Branch Prompt

Use this as the narrow handoff prompt for a fresh implementation agent:

```text
Implement Phase 0/1 of docs/plans/live_logging_overhaul_plan.md and
docs/plans/live_logging_phase0_phase1_spec.md on a fresh branch from updated v8.
Keep behavior unchanged. Do not migrate broad logging call sites. Add the
LiveEvent contract, event pipeline, bounded writer, monitor-backed structured
sink, small console projection sink, DiagnosticEvent compatibility adapter, and
tests proving serialization, routing, redaction, overflow/degradation, monitor
compatibility, and one fake cycle chain. Do not make logging a trading control
plane; sink failures must degrade observability only.
```
