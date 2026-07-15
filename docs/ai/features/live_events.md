# Structured Live Events

## Contract

`src/live/event_bus.py` owns event types, tags, reason codes, debug profiles, envelope semantics,
and redaction helpers. Use registry constants for query-facing or repeated values. Do not add
aliases for temporary branch-local spellings.

Event emission and sink failures are observability-only unless a feature contract explicitly
requires durable publication. Diagnostic producers must not mutate order lists, execution results,
eligibility, replay order, or runtime decisions.

The generated value reference is `../generated/live_event_registry.md`.

## Fill Console Projection

`fill.ingested` is the canonical normal-fill event. Every new fill remains present in structured
and monitor sinks even when its human projection is suppressed. `operator_visible=false` affects
only console and durable text sinks; it must not change fill accounting, fill history, health
counters, PnL enrichment, or trading behavior.

Batches of 20 fills or fewer project one bounded human line per fill. Larger batches suppress only
those per-fill console/text lines and emit one `fills.ingested_summary` line with the batch count,
the net realized PnL over fills whose PnL is known, the number of known-PnL fills, and the pending-PnL
count. An all-pending batch must not present its known PnL as zero. The legacy direct logger is a
fallback only when the structured live-event console is disabled or its pipeline is absent. Runtime
sink degradation remains isolated by the event pipeline and must not activate dual writing.

## Fresh-Entry Eligibility

Completed normal live plans emit `entry.initial_eligibility` to structured and monitor sinks. The
producer observes the existing Rust plan, reconciliation, and local pre-connector filters; it does
not create another trading gate.

Per symbol/position-side outcome precedence is:

1. `eligible`: selected for the connector-bound create list immediately before invocation
2. `blocked_candidate`: an initial candidate existed but did not reach that boundary
3. `already_satisfied`: reconciliation found an exact or tolerance-equivalent open order
4. `protective_only`: protective actions existed without an initial candidate
5. `no_candidate`: no initial candidate; stable default reason `rust_no_initial_candidate`

Stable per-record reason-count values are:

- `batch_capacity`
- `conversion_zero_or_duplicate`
- `debug_mode`
- `exact_reconciliation_match`
- `freshness_creation_guardrail`
- `hsl_replay_pending`
- `initial_entry_distance_gate`
- `limit_order_create_market_distance`
- `low_balance`
- `malformed_actual_orders`
- `mode_filter`
- `order_match_tolerance`
- `pending_exchange_config`
- `pre_create_market_snapshot_unavailable`
- `pre_create_planning_snapshot_invalid`
- `protective_actions_only`
- `recent_execution`
- `rust_no_initial_candidate`
- `state_change_detected`
- `trailing_unavailable`
- `unclassified_candidate` for an initial candidate missed by known observation points
- `reason_overflow` when the bounded reason-key limit is exceeded

`eligible` does not claim connector invocation, exchange acceptance, or acknowledgement. Samples
are deterministic and bounded; payloads contain no order price, quantity, raw payload, path,
secret, or exception text. Planning failure, deferral, shutdown interruption, or diagnostic failure
omits the event rather than publishing a misleading candidate-free result.

## Connector Call Boundary

`execution.create_connector_call_started` and `execution.cancel_connector_call_started` are
emitted immediately before the concrete `cca.create_order` or `cca.cancel_order` coroutine. They
prove only that local execution reached the call site—not that bytes were sent or the exchange
accepted an action.

Payloads use stable action/method/route values plus bounded order shape and correlation IDs. They
exclude raw connector params, vault addresses, URLs, responses, exception text, paths, and
arbitrary payloads. These events are distinct from pre-call submission intent and terminal exchange
outcome events.

The fixed fields are `action=create|cancel`,
`connector_method=cca.create_order|cca.cancel_order`, and
`connector_route=base|hyperliquid|okx`. Normal plan calls retain cycle, `order_wave_id`, and
`action_id` correlation.

## HSL Replay Timing

For coin-mode `hsl.replay.completed`, `full_elapsed_s` is total replay time;
`protective_elapsed_s` and `startup_blocking_elapsed_s` measure held-pair protective readiness;
`replay_loop_elapsed_s` covers the replay loop itself.

Cache events use `hsl.replay.cache` with `cache_status=hit|miss|rejected`. Cache misses and
rejections are non-authoritative performance outcomes and fall back to exchange-derived replay.
Pair progress exposes `applied_rows`/`total_applied_rows` and scan-cost fields
`scanned_rows`/`total_scanned_rows`/`scanned_rows_per_second`/`pair_elapsed_s`.
`is_held_pair`, `is_cooldown_pair`, and `pair_idx` expose deterministic
held/cooldown/remaining ordering without controlling it. `stage=held_protective_ready` records
bounded ready/pending pairs after the held batch; remaining pairs continue until the full-replay
terminal event.

## Eligibility And Market Compatibility

`forager.eligibility_changed` is a bounded structured/monitor-only record of approved/ignored
membership changes. It does not retain config paths, raw sources, or full lists.

`config.market_compatibility` records configured symbols removed by existing market filters.
Approved-symbol incompatibility is degraded; ignored-symbol incompatibility is skipped. HIP-3
account-mode fatal events may request a bounded best-effort flush before the existing startup error,
but event failure never suppresses that error. Isolated-only market events remain observational and
must not alter filtering or existing-position handling.

## Exchange Configuration Outcomes

`exchange.config_refresh` covers both periodic market refreshes and bounded connector-local
configuration outcomes. Per-symbol connector events use the existing success/failure reason codes
and distinguish the observed result in `data.outcome`:

- `confirmed`: the exchange returned normally; this does not claim that a setting changed
- `unchanged`: the exchange explicitly reported that the requested setting already matched
- `failed`: the request raised or returned a connector-classified failure

Connector-local payloads include only bounded `context`, `operation`, `outcome`, optional numeric
`response_code`, `error_type`, and the envelope's `symbol`. They exclude raw responses, exception
text, request parameters, URLs, and tracebacks. An explicit unchanged outcome is DEBUG; confirmed
success remains INFO until the connector can prove whether it changed state; failures retain their
existing operator-visible warning or error. The event route remains structured/monitor-only, and
event emission failure must not change exchange configuration control flow or results.

## Dynamic Registry Values

- `authoritative_reason_code(surface)` produces `authoritative_<surface>`.
- `sink_failed_reason_code(name)` produces `<name>_sink_failed`.

## Pipeline Timing Semantics

`health.summary` pipeline timings describe observed work; they are not a partition that callers may
sum indiscriminately. Parent service and maintenance timings are inclusive, while child phase
timings attribute selected work inside those parents and may be non-exhaustive. Counts represent
actual attempts or runs. A failed attempt remains counted and contributes measured elapsed time;
work that was not attempted has a zero count.

Consumers must distinguish an observed zero from unavailable historical evidence. Producers omit
fields that were not captured by that runtime version, and reports preserve that absence rather than
converting it to zero. Totals may be summed across compatible windows; maxima remain maxima.

## Validation

- Generated registry values match code.
- Payloads are bounded and redacted.
- Sink failure does not change the observed behavior.
- Correlation IDs connect plan, wave, action, connector-call, and terminal outcomes where present.
- Event claims do not exceed the boundary at which they are emitted.

## Key Code And Tests

- `src/live/event_bus.py`
- `src/live/event_producers.py`
- `tests/test_live_event_registry_docs.py`
- event-family tests under `tests/`
