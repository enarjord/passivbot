# Lifecycle, Health, And Incident Events

Topic file for [structured live events](../live_events.md). The envelope, payload budget,
redaction, and validation rules there apply to every event below.

## Emitter Failure Diagnostics

Best-effort live event emitters remain isolated from their callers. If event construction,
sanitization, or publication fails, the developer DEBUG diagnostic may retain bounded code-owned
event, action, stage, symbol, or position-side context and a bounded exception type. It must not
retain the caught exception value, traceback, request URL, response body, credential, or arbitrary
payload fragment. Non-built-in string metadata, invalid type names, and identifier-shaped names
containing credential markers normalize to `Error` before the safe type is bounded. Redacting the
diagnostic must not change the emitter's return value, exception isolation, event routing, retries,
scheduling, HSL/risk behavior, or trading behavior.
The shared diagnostic-event legacy monitor fallback and diagnostic-step wrapper follow this same
boundary: their DEBUG logs retain the code-owned event kind or step label plus a bounded exception
type, never the exception value or raw class metadata.
The same redaction applies when the event-adjacent HSL coin-status human projection fails; the
structured `hsl.status` event must still be attempted independently. Other non-event HSL
diagnostics are outside this contract.

Startup monitor-publisher construction failures retain the existing ERROR `[monitor]` projection,
while live-event-pipeline installation failures retain their existing WARNING `[monitor]` or
`[event]` projection. These startup-observer diagnostics retain only a bounded exception type and
the stable `continue_without_monitor_publisher` or `continue_without_live_event_pipeline` action;
they never retain exception messages, unsafe class names, URLs, credentials, tokens, or tracebacks.
Their monitor-enabled behavior, pipeline-install condition, assignments, startup continuation, and
all runtime and trading behavior remain unchanged.

## Planning Snapshot Readiness

Captured planning-snapshot readiness failures use the existing `planning.unavailable` diagnostic
with `defer_reason=captured_planning_snapshot_unavailable`. Its grouped `invalid` rows describe the
rejected cohort, never a later provider refresh. Retain up to eight details per surface and twelve
symbols within a missing-symbol detail; `invalid_counts` gives each full per-surface detail count,
and `symbols_count` gives the full size of a sampled symbol list. These observations do not renew
inputs or authorize an exchange action.

Ordinary staged readiness failures before snapshot capture retain the original failed check under
`defer_reason=staged_planner_inputs_not_fresh`, using the same diagnostic bounds. Epoch failures
record the observed and required epoch. Quote failures retain the affected symbol and available
age details even if the provider has refreshed by the time the owner processes the failure.
When account facts invalidate ordinary preparation or reconciliation, the diagnostic includes
`changed_account_facts`. Its labels identify positions, account generation, open orders,
strategy balance, or required fills. The event carries the labels without account values.

## Live Read Progress And Failures

An HSL freshness loss without RED, unavailable account inputs, unavailable scopes, or
a candle source failure uses a single compact warning. Its full prior observation
remains in the structured event. Recovery and status transitions retain their existing
visibility.

`ema.rms_read_failed` records bounded RMS candle-read exception types and a single
symbol sample, without exception text. The first failure per exception type and each
following five-minute summary project as warnings. Intervening occurrences remain
structured DEBUG observations. The summary's `failures` counts the current failure
and equivalent failures since the previous warning.

`state.refresh_progress` retains every previously recorded observation. The first
slow refresh per plan and pending surface shape is visible at INFO, with equivalent
refreshes coalesced across cycles for five minutes. On the next visible occurrence,
`suppressed_refreshes` counts similar refreshes since the last INFO record.

## Startup Timing Budgets

`live.startup.startup_phase_budgets` may define optional diagnostic budgets for the canonical
`bot.startup_timing` phases `account`, `active-candle`, `full-warmup`, `hsl`, `market`, and
`startup`. Each phase may carry `elapsed_milliseconds`, measured from process startup, and
`since_previous_milliseconds`, measured from the preceding startup timing mark. Configured values are copied
to the existing event as `elapsed_budget_ms` and `since_previous_budget_ms` with
`budget_source=config`.

These fields are reporting metadata only. They must not delay, fail, skip, or otherwise control
startup, readiness, exchange access, order construction, or trading. Smoke reports prefer a valid
event-carried configured budget for that dimension and retain the prior local p95 projection only
when no configured value is present. Malformed event-carried configured metadata is reported as
`invalid_budget`; it must not be silently replaced by a historical baseline.

## Shutdown Failure Diagnostics

`bot.shutdown.stage` failure records retain the code-owned stage, bounded exception type, elapsed
time, and applicable task-count or timeout context. The event and its human fallback exclude
exception messages, request URLs, response text, credentials, and traceback values. Failure to
deliver the structured stage may emit one DEBUG fallback with only the stage and bounded exception
type. Event-pipeline close failures retain the same bounded type in their warning. These diagnostics
do not alter task cancellation, execution-loop waits, session closing, shutdown timing, process
control, or event-pipeline isolation. Per-maintainer cancellation failures and the legacy cleanup
helper apply the same type-only boundary.

## Execution-Loop Incidents

An execution-loop or startup failure publishes a bounded `error.bot` record and an equivalent
operator signature. The stable diagnostic fields are incident id, operation/source, exception
type, optional bounded status/code and endpoint, stage, innermost origin, action, and cycle. A
companion private `error.bot.detail` record uses the same incident id and retains the bounded full
exception-chain frame sequence at normal logging levels so a rare failure remains diagnosable
without a DEBUG restart. The detail contains normalized file/function/line values but no exception
text, locals, source lines, request URLs, or response payloads. Startup records use the exact
pre-wrapper exception, including deterministic validation failures later wrapped as fatal process
errors. The normal console keeps only the compact operator signature; its omission of the frame
chain is a readability and volume decision, not a public/privacy boundary. Traceback projection is
entirely observability-only: hostile or malformed exception accessors produce a bounded
`projection_failed` detail instead of replacing the original error, its counter update,
restart-threshold handling, startup propagation, or backoff.

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

## Readiness and owner timing presentation
Console health includes fill age/pending confirmation, active held trailing input wait count/age and a bounded scope sample, last completed-pass/write age, and HSL ordinary preparation age when pending. Unknown ages remain unknown. The last write timestamp is retained independently of pruned execution retry caches. Wait tracking retains up to 256 stable scopes; the current held wait count includes untracked scopes, which are explicitly labeled, and their ages are not invented. Trailing recovery phase totals attribute time to the last observed blocker until the next poll (`fill_confirmation`, `candle_input`, `other_input`); they include polling, candle finalization and retrieval, and are not exclusive network/CPU profiles. Configured pacing and idle loop waits are not readiness waits.

`cycle.completed` identifies `execution_owner`. For `hsl`, it completes one finite protective pass, not its overlapping background ordinary preparation. `timings_ms` reports authoritative refresh, protection and ordinary submission performed in that pass. Optional `ordinary_prepare_elapsed_ms` measures preparation start through retrieval of its completed task (including polling delay); `ordinary_pending_age_ms` describes an unfinished task separately. Early-return, shutdown-shortened and I/O-failed passes do not emit successful completions; completion describes the finite owner pass, not proof that every trading input or background task is ready. Only finite nonnegative numeric `timings_ms.authoritative` values bypass sensitive-key redaction.

Optional `ordinary_admission` describes the latest rejected ordinary guard in the completed pass.
`handoff` is `ready_at_entry`, `same_cycle`, or `after_refresh_retry`. A pass that refreshes and
then accepts an expired ready-at-entry plan retains its initial rejection observation and sets
`ordinary_executed` as usual. The existing rejection reason is unchanged. The create and cancel
counts are measured after reconciliation and before connector admission. Each of the three account
receipt rows reports `age_ms`, a `clock_status` of `fresh`, `stale`, or `invalid`, and whether
confirmation is pending. Invalid clocks have a null age. These bounded observations use the guard's
timestamp and expose only counts, clocks, and classifications.

For the HSL owner, `ordinary_executed` records whether an ordinary plan passed account admission
and returned from the execution path during that completed pass, matching the owner's existing
return flag. It includes accepted empty or sub-millisecond executions whose integer timing can be
zero. Absent or rejected plans report false. This is not exchange acknowledgement or proof of
successful writes; other owners omit this optional field unless explicitly supplied.

`candle.websocket_status` retains individual CCXT receive failures and observed recovery/retirement. Console coalescing is confined to receive-stage network errors; ingestion/programming failures retain their scoped warnings. Receive recovery is recorded before downstream ingestion. Durable terminal observations survive console overflow; watcher generation ownership prevents an old retirement from clearing a replacement receive failure. Recovery covers observed failed subscriptions, not venue connectivity or REST readiness, and overflow is labeled partial. No reconnect, fallback or trading policy depends on presentation state. Candle `scope=warmup_cache` summaries describe cache windows; held close coverage and trailing input waits describe current trading impact.

Bitunix's typed malformed-frame fallback sets `receive_cause=malformed_kline` on the durable
receive event. The first such fallback in a receive episode is visible, and the connector's
periodic drop and fallback summary covers repeats. Other receive failures keep an independent
first warning and repeat interval within the shared receive episode.

Console prefixes include bounded exchange/user identity; full user identity is also in durable events and health summaries. Timestamped command archives include a unique run suffix; stable per-user log aliases remain unchanged. Routine KuCoin re-fetches are DEBUG, while slow fetches and newly persisted fills retain their existing operator visibility.
