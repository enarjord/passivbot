# Structured Live Events

## Contract

`src/live/event_bus.py` owns event types, tags, reason codes, debug profiles, envelope semantics,
and redaction helpers. Use registry constants for query-facing or repeated values. Do not add
aliases for temporary branch-local spellings. Follow the
[event-stream-first logging policy](../logging_policy.md#event-stream-first) for new or changed
diagnostics and their console projections.

Event emission and sink failures are observability-only unless a feature contract explicitly
requires durable publication. Diagnostic producers must not mutate order lists, execution results,
eligibility, replay order, or runtime decisions.

## Event-Family Contracts
Event-family contracts live in topic files. Open only the family being changed:

| Family | Contract |
|---|---|
| Account, Balance, And Risk Events | [`live_events/account_and_risk.md`](live_events/account_and_risk.md) |
| Market Data, Candle, And EMA Events | [`live_events/market_data.md`](live_events/market_data.md) |
| Planning, Order, And Execution Events | [`live_events/execution.md`](live_events/execution.md) |
| Lifecycle, Health, And Incident Events | [`live_events/lifecycle_and_health.md`](live_events/lifecycle_and_health.md) |

## Signed Price Diagnostics

`position.changed.data.signed_entry_distance_fraction` uses the new average entry price and the
diagnostic last price. Monitor position rows use the same
[signed entry distance](../numeric_conventions.md#quantities) convention with their snapshot last
price. Positive means favorable for either position side. Closed or flat positions and observations
without a usable entry or last price omit the field. A measured breakeven is zero. Historical
`price_action_distance` fields use the former adverse-positive convention.

Trailing `current_price_change_from_threshold_fraction` and
`current_price_change_from_retracement_fraction` report `current / reference - 1`.
Monitor `next_target_price_offset_fraction` and `next_unstuck_trigger_price_offset_fraction`
report `target / current - 1`; positive means the target is above the current price for either
position side. These are scale-one signed fractions (`0.01` means 1%), with the existing unavailable
values retained. Console and dashboard percentage formatting is unchanged.

## Numeric Presentation Units

Owned diagnostic fields ending in `_percent` use scale 100: `2.5` means 2.5%, without a numeric
rescaling from their previous abbreviated spelling. This includes replacement price/quantity
differences (absolute relative differences), changes from a balance peak (signed), memory changes,
execution-distance samples, progress, budget usage and failure shares.
CPU, memory and budget usage may exceed 100 where the producer's existing contract permits it.
Percentile qualifiers such as `p95` describe selection, not a percent-valued output. External
library field names remain unchanged; report producers and consumers use the owned names together.

## Canonical Payload Budget

Each `LiveEvent` constructor makes a non-mutating, JSON-compatible bounded copy of its canonical
`data` mapping before the event reaches any sink. Under-limit data is preserved apart from existing
sensitive-key redaction. The copy has a maximum nesting depth of six, 64 items per list, 128 keys
per mapping, 256 inspected mapping entries, 512 characters per string, 128 characters per key,
4,096 traversed nodes, and 32 KiB of serialized `data` including metadata. Strings retain their
prefix with a stable truncation suffix; lists retain prefixes without marker objects; excess keys
and items are omitted.

Sensitive keys redact before their values are retained. Cycles, over-depth values, unsupported
values, and node-limit values become stable classification strings; arbitrary object rendering is
not retained. When any boundary changes data, the reserved root `_live_event_budget` object records
only versioned aggregate counters. A caller cannot supply or override that object. The metadata
contains no source paths, payload fragments, or other caller values.

The same bounded `data` appears in `to_dict()` and the monitor event envelope. This construction-time
budget is revalidated by serialization, monitor projection, context replacement, and pipeline
emission so post-construction caller mutation cannot bypass redaction or size limits. Revalidation
preserves existing aggregate budget metadata without double-counting it. The boundary is
diagnostic-only: it does not alter event identity, schema fields, routes, sinks, caller returns, or
trading behavior. It applies only to canonical `LiveEvent.data`; direct legacy `MonitorPublisher`
history, order, and raw-fill storage contracts remain unchanged.

## Runtime Identity

Each live `Passivbot` instance creates one immutable runtime identity before its
monitor and fill manager are initialized. Startup writes it once to
`caches/runtime/<exchange>/<user>/<run_id>.json`, logs one bounded hash-only
summary, emits `runtime.started`, and includes the same identity in `bot.started`,
monitor manifests, and monitor snapshots.

The identity binds a run id and start timestamp to the Passivbot version, Python
Git commit and tracked-dirty state, canonical config SHA-256 (excluding the top-level
`_transform_log` loader audit trail), embedded Rust
source fingerprint, Rust crate version, and SHA-256 of the extension actually
loaded by Python. It must not include raw config, commands, absolute paths,
credentials, or exchange payloads. Manifest and event publication failures are
observability-only and do not change trading behavior.

The Rust artifact hash and embedded source fingerprint are distinct evidence:
the artifact cannot embed its own final hash, while the build-time source
fingerprint identifies the Rust input used to produce it.

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
- `src/live/event_emitters.py`
- `tests/test_live_event_registry_docs.py`
- event-family tests under `tests/`
