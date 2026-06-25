# Live Logging Overhaul Plan

## Goal

Make live observability a first-class part of the Python live bot's data-gatherer,
payload-builder, orchestrator, and executor role.

Rust remains the source of truth for trading logic. Python remains responsible for
fetching exchange data, proving freshness and coverage, building Rust payloads,
executing returned orders, and explaining every degraded, deferred, skipped, or
executed decision. The logging overhaul should make that explanation cheap to
collect, bounded on a small VPS, and reliable enough for post-incident debugging.

## Current State

The problem is fragmentation, not absence. The branch already has several useful
observability islands:

1. Standard Python logging from `src/logging_setup.py`.
   Console/file logs are operator-readable and already use familiar text tags
   such as `[order]`, `[ema]`, `[candle]`, `[risk]`, `[fills]`, `[state]`,
   `[health]`, and `[forager]`. They are not backed by one structured event
   envelope, so fields, ids, reason codes, and throttling differ by call site.
2. Monitor artifacts from `src/monitor_publisher.py`.
   `MonitorPublisher.record_event()` writes rotated NDJSON envelopes and already
   supports snapshots, history streams, manifests, compression, retention, and
   byte caps. It is the best current persistence base, but it only covers a
   subset of live behavior and writes synchronously from the caller.
3. Diagnostic event wrappers in `src/live/events.py`.
   `DiagnosticEvent` gives `data_packet.updated`, `snapshot.built`, and
   `planning_unavailable` a typed path into monitor events, but the envelope is
   narrow: `kind`, `tags`, `payload`, `ts_ms`, `symbol`, `pside`.
4. Structured freshness/planning models in `src/live/`.
   `DataPacketMetadata`, `FreshnessLedger`, `PlanningSnapshot`, and
   `PlanningAvailability` already model much of the information a proper event
   stream needs.
5. Domain-specific instrumentation hooks.
   `CandlestickManager` has remote fetch callback hooks. Fill refresh has fetch
   timing and coverage metadata. Order waves, staged refresh, HSL, EMA readiness,
   and forager selection already log useful details, but not through one event
   contract.

Recent live work on VPS5 and the local Bybit bot showed why this matters:

- A one-candle open-tail EMA miss needed logs from normal text output, monitor
  events, candle health diagnostics, and code inspection to reconstruct.
- GateIO HSL startup failed terminally, but the causal chain crossed fill
  replay, candle replay, balance/equity timeline generation, and HSL replay.
- VPS5 CPU and memory pressure made it clear that observability must be bounded,
  sampled, and moved off the hot path where possible.
- Console logs are useful for operators, but the full forensic record needs more
  detail than a human can tail during normal operation.
- Repeated live restarts showed that shutdown and warm-cache startup need their
  own explicit contracts. A shutdown signal should propagate into long candle,
  fill, account-refresh, warmup, and HSL replay work so exit is prompt and
  bounded. A short-downtime restart should prove cached coverage and reuse it
  where safe instead of repeating cold-start work unnecessarily.

## Target Shape

Build one live event pipeline with multiple sinks. Do not merge all outputs into
one physical file. Instead, emit one canonical event and project it differently
for each audience.

Primary components:

1. `src/live/event_bus.py`
   A small facade for creating, enriching, routing, throttling, and enqueueing
   events. Live code calls `emit(...)` or domain helpers, not each sink directly.
2. `LiveEvent`
   A stable structured event envelope.
3. `LiveEventContext`
   Per-cycle and per-action context propagated through the live loop.
4. Sinks
   Console, text logfile, structured NDJSON, monitor projection, history streams,
   and optional raw payload store.
5. Routing table
   One declarative policy mapping event types and levels to sinks, retention,
   throttling, and console formatting.

The event bus must not become a trading control plane. Event emission must never
decide whether an order is safe. If sinks fail or fall behind, emit/surface
`sink.degraded`, increment counters, and preserve trading behavior.

## Event Envelope

Each event should include a stable envelope:

- `schema_version`
- `event_id`
- `event_type`
- `ts_ms`
- `monotonic_ms`
- `level`: `trace`, `debug`, `info`, `warning`, `error`, `critical`
- `source`: module or subsystem, for example `live`, `candles`, `fills`,
  `orders`, `risk`, `monitor`
- `component`: narrower producer, for example `staged_refresh`,
  `orchestrator_payload`, `order_wave`, `hsl_coin_replay`
- `tags`: stable hierarchy such as `["risk", "hsl", "coin"]`
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
- `remote_call_id`
- `remote_call_group_id`
- `status`: `started`, `succeeded`, `failed`, `deferred`, `skipped`,
  `recovered`, `degraded`
- `reason_code`
- `message`: short human text
- `data`: bounded JSON object with event-specific fields
- `raw_ref`: optional reference to a redacted raw payload artifact
- `raw_hash`: hash of the raw payload when persisted or intentionally omitted

Compatibility mapping:

- Current monitor `kind` maps to `event_type`.
- Current monitor `payload` maps to `data`.
- Current text tags map to `tags`.
- `_authoritative_refresh_epoch` should become or feed `cycle_id`.
- `PlanningSnapshot.snapshot_id` should pass through unchanged.

## Event Registry

Start with a small event registry and expand as components migrate. Initial
stable names:

- `bot.started`
- `bot.ready`
- `bot.stopping`
- `bot.stopped`
- `cycle.started`
- `cycle.completed`
- `cycle.degraded`
- `remote_call.started`
- `remote_call.succeeded`
- `remote_call.failed`
- `remote_call.throttled`
- `cache.load.started`
- `cache.load.completed`
- `cache.flush.started`
- `cache.flush.completed`
- `cache.flush.degraded`
- `data_packet.updated`
- `snapshot.built`
- `planning.unavailable`
- `planning.defer_summary`
- `planning.symbol_state`
- `forager.selection`
- `forager.feature_unavailable`
- `ema.bundle.started`
- `ema.bundle.completed`
- `ema.fallback_used`
- `ema.unavailable`
- `candle.coverage_checked`
- `candle.tail_projected`
- `hsl.replay.started`
- `hsl.replay.progress`
- `hsl.replay.completed`
- `hsl.transition`
- `rust_orchestrator.called`
- `rust_orchestrator.returned`
- `action.planned`
- `order_wave.started`
- `order_wave.completed`
- `execution.create_sent`
- `execution.create_succeeded`
- `execution.create_failed`
- `execution.create_rejected`
- `execution.cancel_sent`
- `execution.cancel_succeeded`
- `execution.cancel_failed`
- `execution.cancel_ambiguous_terminal`
- `execution.ambiguous`
- `execution.confirmation_requested`
- `execution.confirmation_satisfied`
- `execution.confirmation_timeout`
- `fill.ingested`
- `position.changed`
- `balance.changed`
- `sink.degraded`

Names should be stable and dotted. Text prefixes can remain for console
readability, but they should be projections of the event registry.

## Sinks

### Console Sink

Human-readable, low-volume, tail-safe. INFO defaults should include:

- lifecycle start/ready/stop
- real exchange writes and results
- fills and realized PnL
- balance and position changes
- mode changes
- HSL/WEL/TWEL/unstuck transitions
- execution-loop errors and terminal startup failures
- blocking/degraded states that explain why the bot waits
- bounded fallback use
- periodic health summaries

Keep out of INFO by default:

- every remote poll
- every EMA update
- full Rust payloads
- full exchange payloads
- repeated candidate-only forager misses
- high-frequency cache maintenance with no state change

Default console output should be compact summaries fed from structured events,
for example:

```text
[cycle] 2026-06-22T12:00:01Z id=cy_abc ready symbols=42 rust=8ms reconcile=3ms plan=create:2 cancel:1
[gate] cy_abc deferred creates=5 stale_ema=3 pending_config=2 protective=allowed
[execute] wave=17 cancel=1 ok create=2 ok elapsed=642ms confirm=open_orders requested
[forager] long selected=3/40 unavailable=7 budget_skipped=12 max_age=180s
[risk] BTCUSDT long RED panic activated dd=12.4% threshold=10.0%
```

The full event details remain on disk. Console is a view, not the source of
truth.

### Text File Sink

Same human projection as console, with bounded size rotation. This remains useful
for existing tail workflows and simple VPS debugging.

### Structured Event Sink

Authoritative forensic stream. Reuse and evolve `MonitorPublisher`'s current
rotated NDJSON/event-history machinery rather than inventing a parallel
`logs/events.jsonl` format.

Changes needed:

- accept the full `LiveEvent` envelope
- keep `current.ndjson` compatibility during migration
- move writes off the trading hot path via a bounded queue and dedicated writer
- preserve manifest, compression, retention, and byte caps
- emit/drop counters and `sink.degraded` if the queue overflows or disk fails

### Monitor Sink

Curated live subset for relay/TUI/web. Keep `state.latest.json` as a projection,
not a replacement for the event stream. The monitor should be able to answer
"what is the bot doing now?" while structured events answer "what happened?".

### History Streams

High-volume domain streams should stay physically separate because their volume
and retention needs differ:

- fills
- orders
- positions
- balances
- completed candles
- candle fetches
- EMA bundles or EMA summaries
- Rust payload raw refs

### Raw Payload Store

Full raw exchange responses and full Rust input/output payloads should be stored
only under explicit policy:

- default: store hashes and compact summaries
- debug: store selected raw refs for targeted components
- trace/firehose: store full redacted payloads with short retention and byte caps

Raw artifacts must be redacted before persistence. API keys, signatures,
secrets, auth headers, and sensitive account identifiers must never be written.

## High-Volume Policy

The event bus needs event-type-specific sampling and suppression. Suppression
must itself be observable.

Defaults:

- EMA updates: one `ema.bundle.completed` summary per symbol set/cycle. Per-span
  events only at TRACE or targeted debug.
- Forager rankings: compact top-N and selected/incumbent/replacement detail at
  INFO/DEBUG; full ranking only structured DEBUG/TRACE.
- Remote calls: all failures and slow calls; successful calls summarized by
  cohort at INFO/DEBUG with individual details in structured events.
- Candle cache maintenance: summaries by symbol/timeframe; full shard details in
  DEBUG/TRACE.
- Planning unavailable: throttle repeats by `(symbol, pside, order_class,
  reason_code)` and emit suppression counts.
- HSL replay: start/progress/completion summaries, not per-row events.

## Correlation Model

Post-incident reconstruction depends on correlation ids.

- `cycle_id`: one live-loop cycle or staged refresh epoch.
- `snapshot_id`: planning snapshot id already produced by `PlanningSnapshot`.
- `remote_call_group_id`: concurrent fetch cohort.
- `remote_call_id`: individual request.
- `plan_id`: one Rust planning invocation.
- `action_id`: one planned order action or decision.
- `order_wave_id`: one create/cancel reconciliation wave.

These ids should connect:

1. why data was fetched
2. which remote calls were made
3. which data packets changed
4. which snapshot was built
5. which Rust payload was sent
6. which Rust output came back
7. which actions were planned
8. which exchange writes happened
9. how confirmations/fills changed state

For any exchange write, the structured stream should make the following chain
traceable:

1. Rust ideal order
2. Python executable order
3. gate decision
4. submitted exchange payload
5. exchange response
6. local open-order update
7. confirmation refresh request
8. confirmation satisfied, timed out, or ambiguous terminal state

## Remote Call Instrumentation

Every exchange/network call should eventually emit:

- `remote_call.started`
- `remote_call.succeeded` or `remote_call.failed`

Fields:

- exchange
- endpoint or logical operation
- method
- params hash and bounded params summary
- caller/component
- reason_code, for example `startup_account_ready`, `hsl_replay_candles`,
  `forager_refresh_stalest`, `order_create`, `order_cancel`
- isolated vs concurrent via `remote_call_group_id`
- start/end timestamps
- elapsed ms
- timeout/retry/rate-limit metadata
- payload size/hash/raw_ref when available
- completeness/coverage result when applicable

`CandlestickManager.remote_fetch_callback` should be wired into this pipeline
rather than staying a dead-end hook.

## Rust Orchestrator Instrumentation

The call to Rust should become a first-class event chain:

- `rust_orchestrator.called`
  Summary plus raw ref/hash for the full input payload.
- `rust_orchestrator.returned`
  Summary plus raw ref/hash for the output.
- `planning.symbol_state`
  Per-symbol non-tradable/deferred/reason state, compacted and throttled.
- `action.planned`
  Per order or compact batch summary with reason and source.

Raw Rust payload retention can be much more useful than raw exchange payloads
because it captures the complete decision surface passed to the pure Rust order
computer.

## Cache And Disk Instrumentation

Cache load/flush events should explain:

- symbol/timeframe/data kind
- requested range
- loaded/written range
- row counts
- bytes read/written
- elapsed ms
- overlap detected
- overwrite/merge behavior
- synthetic vs real rows
- shard/index path or sanitized relative path
- error and recovery policy

This is especially important for candle and fill-history issues, where runtime
behavior often depends on local cache coverage.

## Operational Restart Goals

These are adjacent behavior/performance goals discovered while smoke-testing the
logging work. They should be implemented as reviewed slices with tests and live
smoke, not hidden inside observability-only PRs.

1. Shutdown contract.
   Ctrl-C or process stop should set one shutdown intent that long-running live
   paths observe quickly: candle warmup/fetch, fill refresh, account refresh,
   HSL replay, background maintainers, executor waits, and lock waits. Work that
   is not needed for safe cleanup should be cancelled or abandoned cleanly, while
   session close and event/monitor flush still get a short bounded deadline.
   Structured events should record `bot.stopping`, interrupted component,
   cleanup duration, cancelled task counts, and any bounded cleanup timeout.
2. Warm-cache fast restart.
   If downtime is short and local cache metadata proves coverage, startup should
   take a delta path instead of repeating cold-start warmup/replay. This must not
   skip fresh account-critical state or weaken HSL/stateless safety. The bot
   should emit structured startup evidence explaining which cached surfaces were
   reused, which were refreshed, which coverage proofs were accepted/rejected,
   and why cold-start work was still required.

These goals depend on the event stream being good enough to prove whether an
exit or restart was slow because of exchange I/O, cache coverage, HSL replay,
lock contention, or intentional safety policy.

## Migration Plan

### Phase 0: Contract And Routing Table

No behavior change.

- Freeze the initial `LiveEvent` envelope.
- Create the event registry and routing table.
- Define console defaults, structured retention defaults, raw payload policy,
  and redaction rules.
- Document how current monitor `kind` events map to `event_type`.

### Phase 1: Event Bus Around Existing Structured Events

No trading behavior change.

- Add `src/live/event_bus.py`.
- Add `LiveEvent`, `LiveEventContext`, and `emit_event(...)`.
- Wrap current `DiagnosticEvent` or replace it with a compatibility adapter.
- Route `data_packet.updated`, `snapshot.built`, and `planning_unavailable`
  through the bus.
- Add queue-backed structured sink using current `MonitorPublisher` storage.
- Emit sink health counters and `sink.degraded`.
- Add fake-live tests proving one cycle emits a coherent event chain.
- Emit minimal `cycle.started` and `cycle.completed` summaries.

### Phase 2: Data Gatherer Events

No trading behavior change.

- Instrument staged account refresh cohorts.
- Wire candle remote fetch callback into `remote_call.*`.
- Instrument fill refresh request stats and coverage decisions.
- Instrument cache load/flush summaries.
- Add call ids and call group ids.
- Add tests for concurrent vs isolated remote-call reconstruction.

### Phase 3: Rust Planning And Payload Raw Refs

No trading behavior change.

- Emit `rust_orchestrator.called` and `rust_orchestrator.returned`.
- Persist raw Rust payloads under debug/raw-ref policy.
- Emit planning summaries and per-symbol unavailable reasons from the same
  snapshot context.
- Add payload redaction/hash tests and retention tests.

### Phase 4: Order Lifecycle And Risk Transitions

No trading behavior change.

- Convert order wave summaries into structured events.
- Instrument create/cancel sent/succeeded/failed/ambiguous/confirmed lifecycle.
- Route HSL/WEL/TWEL/unstuck transitions through the bus.
- Keep existing text output stable via the console sink.

For order writes, acceptance requires events for:

- create sent/succeeded/failed/rejected/ambiguous
- cancel sent/succeeded/failed
- cancel ambiguous terminal state
- confirmation requested
- confirmation satisfied or timed out

### Phase 5: Migrate Meaningful Text Logs

Mostly observability cleanup.

- Move high-value ad-hoc `logging.*` call sites to structured events.
- Bridge the remaining stdlib logs into the event stream at low fidelity where
  useful.
- Avoid chasing every low-value log site if it adds risk or churn.
- Remove duplicate text once facade projections cover that component.

### Phase 6: Gatekeeper Integration

After the event bus exists.

- Feed gatekeeper diagnostics into the same pipeline.
- Treat gatekeeper output as one producer, not as a separate logging system.
- Console shows only gatekeeper decisions that affect execution or explain
  blocked/degraded behavior.
- Structured events retain full diagnostic context subject to volume policy.

## Recommended First Implementation Milestone

Milestone 1 should be a unified local event stream, not a broad conversion of
all logging call sites.

Companion pre-implementation docs:

- `docs/plans/live_logging_phase0_phase1_spec.md`
- `docs/plans/live_logging_migration_audit.md`

Scope:

- `LiveEvent`
- `LiveEventContext`
- `LiveEventPipeline`
- queue-backed NDJSON structured sink using monitor storage
- console summary sink for the small first event set
- context ids for cycle, plan, and order wave
- bridge existing monitor events
- instrument only:
  - planning cycle start/end
  - existing `data_packet.updated`, `snapshot.built`, `planning_unavailable`
  - Rust orchestrator input/output summaries and raw refs under policy
  - reconciliation summary
  - execution wave summary

Why this first:

- It is small enough to review and merge safely.
- It establishes schema, routing, sink behavior, and backpressure policy before
  instrumenting every exchange endpoint.
- Rust input/output capture is the highest-value artifact for live/backtest
  alignment and for debugging why a given ideal order existed.
- It gives immediate replay value: one cycle can be reconstructed by `cycle_id`
  before the full remote-call and cache instrumentation arrives.

Milestone 2 should then instrument remote calls, candle/EMA readiness, forager
features, and cache load/flush behavior.

## Validation Strategy

- Unit tests for envelope defaults, routing, redaction, queue overflow, and sink
  degradation.
- Unit tests for payload digest/raw-ref behavior and context propagation across
  `cycle_id`, `remote_call_group_id`, `plan_id`, and `order_wave_id`.
- Fake-live integration tests for a full chain:
  `cycle.started -> remote_call.* -> data_packet.updated -> snapshot.built ->
  rust_orchestrator.* -> action.planned -> execution.* -> fill.ingested`.
- Fake exchange tests for concurrent candle fetch batches: one batch id, one
  request/response pair per call, and latency on every result.
- Rust orchestrator tests: input captured, output captured, and exceptions
  captured with input digest.
- Gatekeeper tests: stale market snapshot defers normal creates, panic close can
  use its reduced freshness contract, candidate-only stale EMA excludes the
  candidate, and active/normal stale required EMA fails or defers loudly according
  to its contract.
- Executor tests: partial create response emits an ambiguous event, terminal
  rejection emits a non-ambiguous rejected event, and confirmation requested is
  emitted before confirmation refresh.
- Snapshot tests for console formatting and suppression counters.
- Rotation/retention tests using small byte limits.
- A replay utility that loads one structured event directory and reconstructs a
  planning cycle by ids.
- Manual VPS smoke with DEBUG off to verify console remains tail-safe and CPU
  does not materially regress.
- Operational verification that one live order can be traced from Rust output to
  exchange result, rotated NDJSON segments remain valid, and monitor relay still
  serves snapshots/events.

## Branching And Rollout Recommendation

Do not implement the logging overhaul directly on the current hardening branch
unless the branch is first merged into `v8`.

Recommended flow:

1. Finish validating `codex/v8-fill-history-coverage-bootstrap`.
2. Merge or fast-forward the accepted hardening commits into `v8`, because the
   current `v8` baseline is known to be noisier and buggier than this branch.
3. Fork a new logging branch from updated `v8`, for example
   `codex/v8-live-event-pipeline-phase1`.
4. Implement Phase 0 and Phase 1 only in the first PR/commit series.
5. Review each phase separately before adding broader instrumentation.

Reasoning:

- The logging overhaul should be behavior-preserving and reviewable.
- Starting from old `v8` would force agents to rediscover already-fixed live
  bugs and would make test/probe results harder to interpret.
- Starting from the current hardening branch without merging risks building a
  major observability redesign on a branch that still has unrelated live fixes
  under review.
- Phase boundaries let reviewers verify that observability does not silently
  change execution behavior or add unacceptable VPS load.

## Settled Design Decisions

These decisions should guide the first implementation unless later live evidence
forces a revision.

1. Implement the pipeline in a new live logging module or package, not inside
   `passivbot.py`. The intended first home is `src/live/event_bus.py` or a small
   `src/live/logging/` package if the sink/router split grows.
2. Console output is a sink fed from the structured event stream. Aside from
   early startup/bootstrap messages, default console text should not contain
   material that is absent from the event stream.
3. Console output is operator-facing and trading-focused. It should hide
   internal technical detail by default and emphasize fills, position changes,
   balance changes, order create/cancel results, HSL/WEL/TWEL/unstuck
   transitions, staged entries blocked by high-level gates, trailing orders
   waiting for threshold/retracement, and compact health summaries.
4. EMA, candle, and forager internals belong in structured events. Console INFO
   should summarize only user-relevant effects, such as entries deferred,
   forager unavailable counts, stale market data, or degraded readiness. Per-span
   EMA diagnostics are DEBUG/TRACE material.
5. Raw Rust input/output payloads default to summary plus hash. Full redacted raw
   refs are enabled only by explicit DEBUG/targeted diagnostic policy with short
   retention.
6. Raw exchange payloads are never default-on. They are allowed only in targeted
   DEBUG/TRACE sessions with strict redaction, byte caps, and short retention.
   Default events keep hashes and bounded summaries.
7. Reuse and evolve monitor storage as the canonical structured sink instead of
   creating a parallel `events/` tree. Monitor snapshots remain projections of
   the structured event stream.
8. Use a dedicated writer thread with a bounded queue. Sink failure or overflow
   emits `sink.degraded` and drop counters but must not change trading behavior.
9. Conservative default disk budgets per bot on a small VPS:
   - structured event stream: roughly 100-250 MB
   - human text projection: roughly 25-50 MB
   - raw payload refs: disabled by default; when enabled, roughly 50-100 MB with
     short retention
   These limits should be configurable.
10. Migrate meaningful decision/action stdlib logs first: lifecycle, cycle
    summary, Rust planning, reconciliation, order lifecycle, fills, positions,
    balance, HSL/WEL/TWEL/unstuck, forager selection summaries, and
    blocking/degraded states. Bridge remaining stdlib logs temporarily and avoid
    churn on low-value maintenance chatter.
11. Phase 1 should include only a minimal event validator/query helper that can
    validate rotated NDJSON and reconstruct a single `cycle_id` chain. A richer
    replay CLI should wait until remote-call and cache instrumentation exist.
12. Gatekeeper diagnostics should be one producer in the same event pipeline, not
    a separate logging system. Console shows gate decisions that affect
    execution; structured events retain the full reason tree within volume
    policy.
13. Add a compact `live.logging` config surface eventually. Phase 1 may use
    conservative defaults and environment overrides to avoid schema churn.
    Durable knobs should include level, structured retention, console verbosity,
    raw payload policy, redaction mode, and queue size.
14. Event emission is behavior-neutral. Logging failures may degrade
    observability, but they must not block order execution, alter risk decisions,
    or become a trading control plane.
