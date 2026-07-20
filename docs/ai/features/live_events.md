# Structured Live Events

## Contract

`src/live/event_bus.py` owns event types, tags, reason codes, debug profiles, envelope semantics,
and redaction helpers. Use registry constants for query-facing or repeated values. Do not add
aliases for temporary branch-local spellings.

Event emission and sink failures are observability-only unless a feature contract explicitly
requires durable publication. Diagnostic producers must not mutate order lists, execution results,
eligibility, replay order, or runtime decisions.

## Balance Composition Diagnostics

`balance.changed` may carry an optional `balance_composition` object from the
same already-fetched authoritative balance response. It is diagnostic metadata
only: it must not affect scalar balance calculation, refresh cadence, planning,
orders, risk, readiness, or execution scheduling.

The object contains a stable `status` (`available`, `unavailable`, or
`malformed`), bounded source/reason classification, `count`, `retained`,
`truncated`, and at most eight deterministic `asset_balances` rows. Rows retain
only connector-proven asset, total/free/used amount, USD value, unrealized PnL,
explicit liability, collateral-enabled state, and bounded field provenance.
Missing or non-finite values are absent; a connector must not infer debt,
price, or a neutral zero. Raw account payloads, arbitrary response keys,
addresses, credentials, and internal composition signatures are never emitted.
Any legacy raw-only balance apply replaces a previous composition with an
explicit unavailable state; it must never pair stale asset rows with a fresh
aggregate balance.

Generic staged refresh retains the already-fetched raw balance response only
long enough to derive data-packet provenance metadata, separately from the
bounded composition object and normalized scalar balance. Events may expose
the resulting hash/reference but never the raw response itself.

Hyperliquid contributes rows only for proven unified responses with
`info.balances` as a list. Each retained row uses only its `coin` asset and
finite signed `total`, with field provenance; no hold/free/used, liability,
price, USD value, collateral, or HIP-3 position inference is permitted.
Non-unified responses are explicitly unavailable, while malformed unified
shapes are malformed diagnostics. This parser uses the already-fetched balance
response and does not alter scalar balance extraction or staged refresh calls.

Balance publication occurs on the existing aggregate raw/snapped transition or
on a changed full normalized composition signature, including a change in an
omitted row. Console admission remains based solely on snapped-balance
materiality. Visible balance lines may append a sanitized sample of at most two
retained assets; composition-only changes remain structured/text durable but
stay off the console.

`sink.degraded` identifies the failed sink, stable failure reason, and exception type. It must not
retain exception text, request URLs, credentials, response bodies, paths, or other arbitrary values
from the sink exception. Sink counters and pipeline timings remain available through health
snapshots independently of the exception payload.

## Market Snapshot Diagnostic Skips

`market.snapshot_diagnostic_skipped` records noncritical position-change and balance diagnostics
that could not obtain the required live market snapshot. The event retains only bounded context,
exception type, cycle correlation, and its stable status/reason; it never retains arbitrary
exception text, request details, account values, credentials, URLs, payloads, or tracebacks.

The event is the sole normal console/text warning when the structured console is available. Its
projection is bounded to the normal 240-character record budget. One bounded stdlib warning remains
only when the event emitter or structured console owner is unavailable. Event or sink failure must
not alter position/balance refresh, state mutation, scheduling, retries, planning, orders, risk, or
the caller's existing decision to continue after this noncritical diagnostic failure.

## EMA Diagnostic Redaction

`ema.unavailable` and `ema.fallback_used` retain only code-owned reason classifications, bounded
EMA types (`m1_close`, `m1_volume`, `m1_log_range`, or `h1_log_range`), exception types, symbols,
spans, ages, fallback counts, and cycle correlation. Their normal and debug payloads never retain
arbitrary exception messages, URLs, credentials, raw fallback reasons, or reason fragments parsed
from exception text. Console formatting derives EMA identity from an explicit safe field rather
than inspecting an exception message.

When structured console ownership is unavailable, the bounded legacy warning follows the same
redaction boundary. Event or sink failure must not alter EMA calculation, cached fallback
selection, candidate availability, scheduling, retries, planning, orders, risk, or caller control
flow. Downstream smoke and incident redaction remains defense in depth, not the primary payload
boundary.

The generated value reference is `../generated/live_event_registry.md`.

## Runtime Identity

Each live `Passivbot` instance creates one immutable runtime identity before its
monitor and fill manager are initialized. Startup writes it once to
`caches/runtime/<exchange>/<user>/<run_id>.json`, logs one bounded hash-only
summary, emits `runtime.started`, and includes the same identity in `bot.started`,
monitor manifests, and monitor snapshots.

The identity binds a run id and start timestamp to the Passivbot version, Python
Git commit and tracked-dirty state, canonical config SHA-256, embedded Rust
source fingerprint, Rust crate version, and SHA-256 of the extension actually
loaded by Python. It must not include raw config, commands, absolute paths,
credentials, or exchange payloads. Manifest and event publication failures are
observability-only and do not change trading behavior.

The Rust artifact hash and embedded source fingerprint are distinct evidence:
the artifact cannot embed its own final hash, while the build-time source
fingerprint identifies the Rust input used to produce it.

## Startup Timing Budgets

`live.startup_phase_budgets` may define optional diagnostic budgets for the canonical
`bot.startup_timing` phases `account`, `active-candle`, `full-warmup`, `hsl`, `market`, and
`startup`. Each phase may carry `elapsed_ms`, measured from process startup, and
`since_previous_ms`, measured from the preceding startup timing mark. Configured values are copied
to the existing event as `elapsed_budget_ms` and `since_previous_budget_ms` with
`budget_source=config`.

These fields are reporting metadata only. They must not delay, fail, skip, or otherwise control
startup, readiness, exchange access, order construction, or trading. Smoke reports prefer a valid
event-carried configured budget for that dimension and retain the prior local p95 projection only
when no configured value is present. Malformed event-carried configured metadata is reported as
`invalid_budget`; it must not be silently replaced by a historical baseline.

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

## Open-Orders Snapshot Deltas

`open_orders.snapshot_delta` replaces the aggregate INFO lines for open-order snapshot additions or
removals larger than 20 orders. It emits at most one event for each qualifying aggregate direction,
with `direction=added|removed` and `order_count` as its only payload fields. The event is routed to
structured, monitor, console, and durable text sinks at INFO with the `[order]` console tag.

Snapshots of 20 orders or fewer retain the existing per-order logging behavior. The event is
diagnostic-only: emission or sink failure must not affect reconciliation, unexpected-change
classification, guardrails, confirmations, balance handling, or follow-up refreshes. It never
retains order rows, identifiers, symbols, prices, raw payloads, exception text, or samples.
When the structured console is disabled or unavailable, the same bounded count retains one legacy
`[order]` INFO fallback; the two console paths never intentionally write the same observation.

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

Pre-create planning-snapshot and market-snapshot gate failures use
`execution.create_skipped` as the sole normal console/text warning when the structured console is
available. The event retains the code-owned reason and message plus bounded order/symbol counts,
stage, safe detail fields, and exception type; it excludes raw exception text and arbitrary
payloads. The legacy warning remains only when the event emitter or structured console owner is
unavailable. Event or sink failure must not change refresh attempts, entry-block attribution, the
returned create list, or any planning, execution, order, and risk decision.

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

Connector-local payloads include only bounded `context`, `operation`, `outcome`, optional
digit-only `response_code`, `error_type`, and the envelope's `symbol`. They exclude raw responses, exception
text, request parameters, URLs, and tracebacks. An explicit unchanged outcome is DEBUG; confirmed
success remains INFO until the connector can prove whether it changed state; failures retain their
existing operator-visible warning or error. The event route remains structured/monitor-only, and
event emission failure must not change exchange configuration control flow or results.

## Execution-Loop Incidents

An execution-loop failure publishes a bounded `error.bot` record and an equivalent first-occurrence
operator signature. The stable diagnostic fields are operation/source, exception type, optional
bounded status/code and endpoint, action, and cycle. This family excludes raw exception text,
request URLs, response payloads, and traceback values. Stack frames may be retained only in the
protected DEBUG text path and must not include the exception value.

Equivalent repeats use `health.summary` with the execution-error-burst reason. Its latest-failure
fields are `latest_error_type`, optional `latest_status`, `latest_code`, and `latest_endpoint`; it
must not retain `latest_error`. These projections are observability-only: error counting,
timestamp recovery, restart thresholds, backoff, and trading behavior remain owned by the existing
execution-loop policy.

## Smoke Hard-Problem Evidence

Full live smoke reports keep the existing bounded latest mixed `problem_events` sample and expose
an independent `hard_problem_events` object. Its `count` is the authoritative full-window hard
count; `retained` and `truncated` describe the bounded chronological `sample`. Later non-hard
attention therefore cannot remove every hard classification while the report remains red.

The hard-only sample uses the caller's existing `max_problem_events` bound and the same compact,
redacted event projection as the mixed sample. It is reporting evidence only and must not change
problem classification, recovery matching, smoke verdicts, process control, or live behavior.
Concise summary, brief, and incident-bundle compact smoke metadata retain the same object shape,
with an independently bounded value-safe sample and `retained`/`truncated` recalculated for that
projection.

## Cycle Degradation

`cycle.degraded` retains the stable reason code, bounded exception type, cycle correlation, safe
operational details, and phase timings needed to explain an incomplete live cycle. Exception text,
request URLs, request/response payloads, tracebacks, and unknown caller fields are excluded from
this event, including nested spelling variants and generic execution-loop failures or fill-history
coverage deferrals. Only the event family's bounded classification, timing, counter, authoritative
barrier, and staged-readiness fields are retained. The existing execution-loop incident family
remains the bounded source for safe status, code, endpoint, and action classifications.

This payload boundary is observability-only. Exception propagation, retries, time-sync recovery,
restart thresholds, fill-history refresh requests, cycle deferral, and trading behavior are
unchanged.

## Trailing And Unstuck Status Materiality

`trailing.status` and `unstuck.status` retain one complete observation every five minutes while the
corresponding status producer is active. `operator_visible=false` suppresses only console and
durable text projection; structured and monitor sinks still receive the event. Missing visibility
metadata remains operator-visible. The first observation, a qualitative transition, a material
numeric transition, and the existing hourly reminder use `operator_visible=true`. The `changed`
field is true only for the first or materially changed observation, not for an unchanged hourly
reminder.

Unstuck allowance materiality uses five-percent relative hysteresis against the last
operator-visible baseline. Status, over-budget crossing, configured overrides, candidate symbol,
and existing target fields remain qualitative transition inputs. Trailing visibility is owned per
`symbol`/`pside`/`kind` item so one changing position does not expose unrelated repeats. Its
threshold and retracement ratios are material at an absolute ratio delta of `0.0005` (0.05
percentage points); threshold and retracement prices are material at a relative delta of `0.005`
(0.5 percent). Qualitative status, support, and trigger flags remain immediate.

These thresholds affect human projection only. They must not change Rust planning, order emission,
position handling, unstuck eligibility, trailing calculations, the five-minute observation cadence,
or event payload values. Legacy direct INFO logging remains a fallback only when the structured
console is unavailable and follows the same producer-owned admission decision.

When a trailing observation is operator-visible, its console/text projection must remain within the
normal 240-character record budget. Keep the action/status, `symbol`/`pside`, selected mode,
threshold and retracement gate states and material values, current price, and available correlation
while using compact labels. Complete diagnostics remain in structured/monitor data; compaction must
not remove payload fields or change admission, cadence, or behavior.

## Memory Snapshots

`resource.memory_snapshot` records the existing material memory diagnostic when it first becomes
available or when absolute process-RSS movement reaches the producer's 25 percent boundary. It does
not add a collection pass, timer, threshold, cache traversal, task inspection, or trading control.
Routine samples remain available only in the protected DEBUG text path.

The bounded payload contains finite RSS bytes and percent delta; cache bytes, candle count, symbol
count, and at most three symbol samples; timeframe-cache bytes, range count, and at most three
symbol/timeframe samples; and task total, pending count, and at most four sanitized task-name/count
entries. Samples use strict length and character allowlists and exclude URLs, query strings,
exception text, paths, raw payloads, and secrets.

The event is the sole normal console/text owner when the structured console is available. Its
compact projection reports aggregate RSS, cache, timeframe-cache, and task counts and must satisfy
the normal 240-character budget. The legacy compact INFO line is a fallback only when the event
console or emitter is unavailable. Event or sink failure remains isolated and must not change the
producer's prior-RSS state, cadence, collection, cache state, tasks, or trading behavior.

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
