# Market Data, Candle, And EMA Events

Topic file for [structured live events](../live_events.md). The envelope, payload budget,
redaction, and validation rules there apply to every event below.

## Candle Persistence Observer Diagnostics

Candle disk-load and persisted-batch observer diagnostics retain only their code-owned stage, a
bounded exception type, and stable control-flow action: `return_from_cache_load_observer`,
`continue_to_cache_flush_observer`, `return_from_persist_observer`, or
`return_from_cache_flush_observer`. They never retain exception messages, unsafe class names, URLs,
credentials, tokens, or tracebacks. Monitor persistence and cache-flush reporting remain
independently best-effort; redaction does not change callback placement, throttling, suppression,
completed disk persistence/load behavior, event schemas, or trading behavior.

## Market Snapshot Diagnostic Skips

`market.snapshot_diagnostic_skipped` records noncritical position-change and balance diagnostics
that could not obtain the required live market snapshot. The event retains only bounded context,
exception type, cycle correlation, and its stable status/reason; it never retains arbitrary
exception text, request details, account values, credentials, URLs, payloads, or tracebacks.

The event is the sole normal console/text warning when the structured console is available. Its
projection is bounded to the normal 240-character record budget. One bounded stdlib warning remains
only when the event emitter or structured console owner is unavailable. The structured event,
legacy warning, and emitter-failure DEBUG diagnostic use the same hostile-metadata-safe bounded
exception type and retain neither the raw class name nor exception text. Event or sink failure must
not alter position/balance refresh, state mutation, scheduling, retries, planning, orders, risk, or
the caller's existing decision to continue after this noncritical diagnostic failure.

Adjacent market-snapshot provider primary-fetch, missing-symbol retry, cache-sink, fallback,
pre-create gate, event-emitter, and eligibility-trace diagnostics retain only bounded exception types
and stable code-owned actions. They never retain exception messages, unsafe exception class names,
tracebacks, request details, credentials, or arbitrary payload fragments. Redaction must not change
provider fallback order, optional completed candle fallback, fail-closed create filtering,
entry-block attribution, cache-sink suppression, trace isolation, or exception cause chaining.

Observer-only position and balance diagnostics follow the same rule. A non-market-snapshot balance
equity observer failure records only a bounded exception type and the
`preserve_balance_anchors_and_schedule` action. A position-change observer failure records only a
bounded exception type and a stable continuation action: `return_updated_positions` for a
standalone positions refresh, `continue_balance_reconciliation` for a combined positions/balance
refresh, or `continue_balance_update` for staged refresh. A non-cancellation progress logger
failure records only a bounded type and `stop_progress_logger`; explicit cancellation still
propagates. These diagnostics do not change refresh propagation, balance anchors, scheduling,
reconciliation, balance handling, cleanup, timing, fallback values, or event schemas.

## Coin-List Refresh Failure Diagnostics

The outer approved/ignored coin-list refresh failure diagnostic retains only a bounded exception
type and the `return_from_coin_list_refresh` action. It never retains exception messages, unsafe
exception class names, URLs, credentials, tokens, or tracebacks. This redaction does not alter the
existing return, list/universe/eligibility update semantics, scheduling, risk, HSL, or trading
behavior.

When unrealized-PnL aggregation cannot calculate a fetched position, its ERROR diagnostic retains
only a bounded exception type and the `return_zero` action. It immediately returns the existing
`0.0` fallback without retaining exception text, unsafe class metadata, URLs, credentials, or a
traceback; price fetching, iteration, balance/equity handling, state, scheduling, risk, HSL, and
trading behavior are unchanged.

Foreign-writer safety shutdown keeps its terminal stop behavior when best-effort data-maintainer
cleanup fails. The ERROR diagnostic retains only a bounded exception type and the stable
`continue_foreign_writer_stop` action; it never retains exception text, unsafe exception class
names, URLs, credentials, or tracebacks. This diagnostic redaction does not alter stop flags,
maintainer invocation, foreign-writer detection, or the terminal exception.

## EMA Diagnostic Redaction

`ema.unavailable` and `ema.fallback_used` retain only code-owned reason classifications, bounded
EMA types (`m1_close`, `m1_volume`, `m1_log_range`, or `h1_log_range`), exception types, symbols,
spans, ages, fallback counts, and cycle correlation. Their normal and debug payloads never retain
arbitrary exception messages, URLs, credentials, raw fallback reasons, or reason fragments parsed
from exception text. Console formatting derives EMA identity from an explicit safe field rather
than inspecting an exception message. Malformed symbols and classifications normalize to bounded
sentinels, unsupported metric names are omitted, and non-finite spans are omitted. Adjacent EMA
failure logs retain only code-owned context and exception type.

The structured event owns the normal warning only after its emitter reports success and the
structured console owner is available. Success requires queue admission and no synchronous console
sink failure; these owner events defer console/text projection until after queue admission so a
rejected event cannot produce both projections. Otherwise, the bounded legacy warning remains and
follows the same redaction boundary. Event or sink failure must not alter EMA calculation, cached
fallback selection, candidate availability, scheduling, retries, planning, orders, risk, or caller
control flow. Downstream smoke and incident redaction remains defense in depth, not the primary
payload boundary.

The generated value reference is `../../generated/live_event_registry.md`.

## Candle Remote-Fetch Diagnostics

Candle manager callbacks are sanitized before any live-event, HLCV, or fake-live consumer receives
them. CCXT and archive diagnostics retain code-owned stage, symbol/timeframe, attempt/status/timing,
bounded exception type, parameter keys, and a SHA-256 URL hash where needed for correlation. They
exclude exception text and repr, raw request URLs, and request-parameter values.

`remote.call_started`, `remote.call_succeeded`, `remote.call_failed`, and
`remote.call_throttled` preserve start/result correlation after this redaction, including concurrent
archive calls keyed by URL hash. Repeated consumer-side sanitization must be idempotent. Redaction
and sink failure cannot change fetches, retries, backoff, rate-limit classification, archive
availability, cache behavior, or trading decisions.

## WebSocket Reconnect Diagnostics

`websocket.reconnect` retains bounded reconnect count, retry delay, reason classification,
rate-limit state, warning visibility, DEBUG stack-frame-emission state, and optional exception type.
The human warning/debug projection may retain the same classifications and bounded stack depth, but
must not render frame labels, line values, the exception value, raw reason text, request URLs,
responses, or formatted exception-value traceback. A stack diagnostic is marked emitted only when
DEBUG logging accepts it. Reconnect cadence, retry behavior, and connector control flow are
independent of observability delivery.

## EMA Preparation Timings

`ema.bundle.completed` includes an optional `timings` object, without changing its DEBUG,
structured/monitor-only routing. `elapsed_ms` measures bundle wall time since the start event;
`symbol_elapsed_ms` sums symbol preparation durations, and `symbol_count` counts attempted symbols.
`slowest_symbols` retains at most eight rows in descending elapsed time (symbol breaks ties), with
`symbols_omitted` reporting the remainder. `stage_totals` includes all attempted symbols, not just
that sample. Older producers omit `timings`; consumers must not interpret its absence as zero.

Rows and totals use `<stage>_ms` and `<stage>_calls` for candle loading (`candles`), disk loading
(`disk_load`), fetch-lock acquisition (`fetch_lock_wait`, excluding held work and release), projected
open-tail metrics (`projection`), EMA arithmetic (`ema_compute`), and remote page-fetch
operations including internal retries (`remote_fetch`). `remote_spacing_sleep` also records `_requested_ms`, separately from
observed elapsed sleep. Stages absent from a completed symbol were not attempted; failed attempts
contribute elapsed time and counts if the bundle subsequently completes. A failed/cancelled bundle
does not emit a completion event.

These are inclusive, potentially nested measurements, not an additive partition. Concurrent symbol
work can exceed bundle wall time; candle loading includes nested disk, lock, and remote work, and
remote fetch time includes spacing and other retry/rate-limit waits inside that call. Timers do not
profile every operation or explain unmeasured time. Timing state is scoped to the current bundle's
symbol tasks, bounded independently of span/candle counts, and never controls scheduling, readiness,
EMA values, or order construction.
