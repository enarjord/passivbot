# Live Logging Migration Audit

This audit groups existing logging and monitor call sites into migration buckets
for the future event-pipeline branch. It is intentionally not an implementation
diff.

## Current Surfaces

1. Text logging
   - `src/logging_setup.py`
   - root Python logger
   - timestamped files under `logs/`
   - stable `logs/{user}.log` alias
   - optional rotation.
2. Monitor persistence
   - `src/monitor_publisher.py`
   - `events/current.ndjson`
   - `history/*.current.ndjson`
   - `state.latest.json`
   - manifest, rotation, compression, retention, and max-total-byte pruning.
3. Monitor relay/dashboard/TUI
   - `src/monitor_relay.py`
   - read-only consumer of monitor artifacts
   - should remain transport/projection, not the event contract owner.
4. Diagnostic event shim
   - `src/live/events.py`
   - `DiagnosticEvent(kind, tags, payload, ts_ms, symbol, pside)`
   - currently emits into `_monitor_record_event`.
5. Direct monitor events
   - `_monitor_record_event(...)` calls in live startup, data packet updates,
     fill ingestion, executor events, and related runtime points.
6. Domain text logs in `src/passivbot.py`, `src/live/`, `src/exchanges/`, and
   `src/candlestick_manager.py`.

## Phase 1: Convert Or Bridge First

These are high-value and already close to structured state:

- `DiagnosticEvent` in `src/live/events.py`
  - replace or wrap with `LiveEvent` compatibility adapter.
- `data_packet.updated`
  - source: current data-packet diagnostics
  - event type: `data_packet.updated`
  - console: no, except summary/degraded state.
- `snapshot.built`
  - event type: `snapshot.built`
  - include `snapshot_id`, readiness counts, and surface ages.
- `planning_unavailable`
  - event type: `planning.unavailable`
  - throttle by reason/symbol/pside/order class
  - console only as compact blocked/degraded summary.
- Rust orchestrator call
  - event types: `rust_orchestrator.called`, `rust_orchestrator.returned`
  - default data: symbol count, payload hash, output hash, elapsed ms, order
    counts, diagnostics counts
  - raw refs only under debug policy.
- Order wave summary
  - event type: `order_wave.completed`
  - source current executor/reconciler summaries
  - console yes.
- Lifecycle
  - event types: `bot.started`, `bot.ready`, `bot.stopping`, `bot.stopped`
  - console yes.
- Sink health
  - event type: `sink.degraded`
  - console warning with suppression.

## Phase 2: Instrument Next

These are important but should wait until the event bus is proven:

- Remote calls
  - exchange/account surfaces: balance, positions, open orders, fills
  - market surfaces: tickers, OHLCV, order book/ticker snapshots
  - event types: `remote_call.started`, `remote_call.succeeded`,
    `remote_call.failed`, `remote_call.throttled`
  - group concurrent batches with `remote_call_group_id`.
- Candlestick manager fetch callbacks
  - wire existing callback hook into remote call events
  - include timeframe, requested range, returned range, row counts, cache impact.
- EMA readiness
  - event types: `ema.bundle.started`, `ema.bundle.completed`,
    `ema.unavailable`, `ema.fallback_used`
  - console only aggregate user-facing effects.
- Forager ranking and selection
  - event types: `forager.selection`, `forager.feature_unavailable`
  - console compact selected/incumbent/unavailable summary.
- Cache load/flush
  - event types: `cache.load.*`, `cache.flush.*`
  - console no unless degraded.

## Phase 3/4: Convert After Core Pipeline

- HSL/WEL/TWEL/unstuck transitions
  - event type: `hsl.transition` plus later risk-specific names if useful
  - console yes, concise and trading-relevant.
- Fill ingestion
  - event type: `fill.ingested`
  - console yes for new fills.
- Position/balance changes
  - event types: `position.changed`, `balance.changed`
  - console yes.
- Order lifecycle detail
  - event types:
    - `execution.create_sent`
    - `execution.create_succeeded`
    - `execution.create_failed`
    - `execution.create_rejected`
    - `execution.cancel_sent`
    - `execution.cancel_succeeded`
    - `execution.cancel_failed`
    - `execution.cancel_ambiguous_terminal`
    - `execution.confirmation_requested`
    - `execution.confirmation_satisfied`
    - `execution.confirmation_timeout`
  - acceptance: trace any exchange write from Rust ideal order to confirmation.

## Keep As Early Bootstrap Or Plain Text Initially

These may remain outside the event stream until the pipeline is initialized:

- initial CLI argument/config load failures
- very early logging setup errors
- startup banner
- fatal import/build/runtime environment failures before bot identity is known.

If possible, later bridge these as `bot.started` or `bot.startup_failed` once
bot identity exists.

## Demote Or Suppress In Console

These are useful for diagnostics but should not stay default INFO console noise:

- per-span EMA recovery/fallback chatter
- repeated candidate-only forager misses
- routine candle cache/index maintenance with no state change
- successful high-frequency remote poll calls
- successful cache loads/flushes
- websocket reconnect debug loops unless user action is needed
- raw CCXT request/response payloads outside TRACE.
- executor create/cancel anomaly payloads. Bounded structured execution events
  are authoritative; text fallback may include only counts, sanitized symbols,
  order type, reason code, and exception type when the structured console is
  unavailable. Never print raw order dictionaries, exchange responses, or
  exception messages from these paths.

Structured events may still retain compact summaries or DEBUG details.

## Keep As DEBUG/TRACE Structured Events

- full forager ranking lists
- full Rust payload raw refs
- raw exchange request/response refs
- candle shard merge details
- cache overlap repair details
- per-symbol/per-span EMA details
- low-level websocket callback details.

## Migration Guardrails

- Do not remove text logs before the event projection covers the same operator
  information.
- Do not make monitor relay/dashboard availability part of live trading.
- Do not emit events from Rust strategy logic in Phase 1; Python captures the
  Rust boundary.
- Do not serialize secrets.
- Do not let event emission exceptions escape into trading-critical code.
- Do not use logging migration as a reason to weaken fail-loud contracts.

## Suggested Review Checklist

For every migrated call site:

1. Is the event type in the registry?
2. Are `exchange`, `user`, ids, symbol/pside, status, and reason code populated
   when available?
3. Is console output a projection of the event?
4. Is INFO tail-safe?
5. Is full diagnostic detail still available in structured DEBUG/TRACE or raw
   refs when enabled?
6. Does sink failure leave trading behavior unchanged?
