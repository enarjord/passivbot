# Planning, Order, And Execution Events

Topic file for [structured live events](../live_events.md). The envelope, payload budget,
redaction, and validation rules there apply to every event below.

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

A completed fill refresh can attach `refresh_work` to its structured summary. It counts received
rows, normalized duplicates, applied or ignored overlap rows, PnL observations, duplicate days
verified against disk, and actual day-file replacements. The replay-reuse flag describes the
same fetch. No fill payload or identifier is included.

Fill refresh failures retain code-owned source, refresh mode, coverage, retry, timing, and count
context plus a bounded exception type. The structured `fills.refresh_summary`, exchange-specific
fetcher and manager request-timing lines, staged authoritative remote-call failure, blocking and
routine-prefetch fallbacks, startup initialization boundary, fill-coverage retry, and HSL flatten
confirmation must not retain exception text, request URLs, response bodies, credentials, or
exception-value tracebacks. Metadata-capture and HSL progress-emitter failures at the same caller
boundary follow the same classification-only rule. Redaction is diagnostic-only: exception
propagation, retry and coverage behavior, fill accounting, planning, risk, and trading behavior
remain unchanged.

Cache-read/doctor failures and the outer process failure projection follow the same rule so a
preserved exception cause cannot be exposed by an unsanitized startup traceback. The blocking
failure line may also retain validated numeric status/code and a code-owned endpoint label.
Time-sync and exchange recovery classification may inspect complete exception text in bounded
temporary chunks and traverse both cause/context edges within a fixed node budget, but that
inspection must not retain the text, broaden caller-specific marker case rules, or replace the
original failure. Control-flow-only class-name inspection follows the same non-retaining rule so
legacy recovery heuristics do not require projecting untrusted class names.
Monitor error events use the same hostile-metadata-safe exception-type boundary; diagnostic
publication must not replace the original refresh exception.

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

## Close Replacement Timing

`order.close_replacement_timing` records timing for a tracked ordinary limit close replacement
after an unambiguous cancel acknowledgment. The event identifies the symbol and position side,
with optional synthetic `cancel_wave_id` and `create_wave_id` for joining execution events.
These numeric wave IDs correspond to the `ow_{id}` execution event IDs. When available,
`create_action_index` identifies the specific create within its wave.
`elapsed_ms` is the time from the cancel acknowledgment to the observation. Milestones use
the same origin:

- `account_confirmed_ms` records the first observed post-cancel epoch readiness covering
  balance, positions, open orders, and fills. A subsequent change can require reconfirmation.
- `first_plan_ms` records the first validated, reconciled plan with a unique ordinary create
  candidate for the same symbol and position side, before waiting for executor handoff.
- `create_admitted_ms` records executor admission of the candidate that reached transport.

Milestones are omitted when unobserved. When present, `last_create_defer` identifies a recent
create-side guard for this position side. `last_missing_surfaces` lists recent missing account
surfaces from the ordinary planner. These observations do not assign elapsed time to a blocker.

`outcome=create_sent` is recorded once at guarded submission. Placement acknowledgments
retain their existing event. Neither ends account observations for the tracked replacement.
After the delay threshold, the first authoritative observation of zero resting close quantity
reports `still_waiting` with `resting_coverage=zero`. Ambiguous order evidence can report
`coverage_unconfirmed` with `resting_coverage=unknown` once per replacement. These observations
can follow guarded submission.

A current post-cancel positions and open-orders cohort may report `resting_close_observed` with
`resting_coverage=full` when proven resting close quantity covers the current position, or
`resting_coverage=partial` when it covers a known smaller quantity. For float roundoff, the
comparison allows only the adjacent positive float below position size and only when the
gap is below half a venue quantity step if the step is known. A partial close can be
intentional. Both observations end replacement tracking. Only full coverage clears an earlier
warning with a resolved log. The other terminal observations, `position_closed` and
`no_close_in_plan`, end tracking without claiming resting coverage. Each tracked replacement
produces at most one terminal observation.

The event goes to structured and monitor sinks. It carries no prices, quantities, or venue
order IDs. Tracking and publication are best effort and leave order admission, account
confirmation, reconciliation, and retry scheduling unchanged.

## Order-Replacement Churn

The account-wide replacement-churn policy emits structured, monitor-only summaries:

- `order.churn_evidence` reports one bounded per-plan aggregation of RAM-history association
  reasons, churn-evidenced order count, tracked symbol count, generation, and RAM-history clearing.
  History starts empty and is cleared after planning failure, gate disablement, or a symbol leaving
  the current ideal/active/open-order/nonzero-position universe. Flat position placeholders retained
  by account-state normalization do not keep a departed symbol's churn history alive.
- `order.churn_admission` reports one bounded final-admission aggregation after exchange
  configuration and the final fresh-market check, including rolling create-attempt count, threshold,
  decision reasons, and sampled market distances. The `uncovered_close_exempt`
  decision identifies the sole reduce-only limit close in a Bitunix cancel-first
  scope admitted over the economy allowance with confirmed zero resting close coverage.
- `order.churn_actions_accounted` records logical create attempts debited immediately before the
  connector create call, plus the resulting rolling count. Configuration writes and cancellations
  are excluded from the window; the retained `action_kind` field is therefore `create`.
- `execution.cancel_first_barrier` reports create deferral after a stale-order cancellation. In
  effective hedge mode the scope is symbol and position side; in one-way mode it is the whole
  symbol because long and short share one net exchange position. Unrelated scopes may continue in
  the same wave; an unscoped cancellation conservatively defers all ordinary creates. The stable
  `account_cancel_first_barrier` reason-code name is retained for event-consumer compatibility.
  `execution.cancel_deferred` reports cancellation batch truncation.

These events are diagnostic projections only. The causal history, rolling counter, and
cancel-first state are updated directly in the planner and executor, and sink failure cannot alter
the selected exchange actions.

Reconciliation trace, churn-summary, normalization, and malformed-open-order diagnostics retain
only bounded exception types. They exclude exception text, untrusted exception class names, URLs,
and traceback data; this diagnostic redaction does not change churn availability, malformed-order
guardrails, planned orders, or exchange-action gating.

## Order-Sort Price-Fetch Diagnostics

Market-price fetch failures while sorting orders retain bounded symbol context, a bounded exception
type, and the stable `preserve_original_order` action. They use the existing empty-price-map path to
return the original order object sequence without retaining exception messages, unsafe exception
class names, URLs, credentials, tokens, or tracebacks. Fetch parameters, missing-price handling,
complete-price sorting, reconciliation, planning, and trading behavior remain unchanged.

## Rust Orchestrator Returned Events

Failed `rust_orchestrator.returned` events retain their existing timing, input hash, cycle and
remote-call correlation, error level/status/tags, and one bounded exception type used for both
`reason_code` and `data.error_type`. They omit exception text, URLs, credentials, and unsafe class
metadata. This diagnostic redaction does not change Rust orchestration, event ordering or
isolation, caller propagation, connector calls, or trading behavior.

## Fresh-Entry Eligibility

Rust-owned `forager.selection` events report the candidate and eligible counts, whether ranking
was required, sampled exclusions with feature and reason, and any startup deferral.
The event warns when a symbol is excluded for the same reason across ten consecutive plans.
It also warns when a side with free slots and candidates has neither a working initial
entry nor a newly opened position for `logging.forager_no_entries_warning_minutes`. This includes
sides whose candidates are all excluded. Startup deferral and active cooldown on every selected
candidate reset the stall timer. The monitor snapshot exposes an active entry stall. These
observations do not affect order permissions.

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
- `account_cancel_first_barrier`
- `limit_order_create_market_distance`
- `low_balance`
- `malformed_actual_orders`
- `mode_filter`
- `order_match_tolerance`
- `order_churn_gate`
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

The console create-candidate summary lists orders selected before final write admission.
Candidates can still be skipped. Use the connector call and outcome events to distinguish
local admission, exchange acknowledgement, and failed or skipped orders.

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

## Terminal Execution Outcomes

Terminal and ambiguous execution-order outcome events retain their existing envelope correlation,
status, reason, action, bounded order metadata, result summaries, and optional bounded debug
profiles. When an outcome carries an exception object or result, its payload may retain bounded
`error_type`, numeric status/code, a constrained exchange error label, and a bounded sanitized
reason extracted from a structured exchange error payload. Exact top-level mappings, their
`info` mapping, and up to eight exact OKX per-order `data` mappings use the same sanitizer;
per-order OKX fields take precedence over generic envelope details. Hyperliquid's error response
string and the first error among up to eight per-order statuses are also recognized. An exact
unrecognized API-wallet response receives a stable label and actionable wording. This
structured rejection extraction applies only to failed or ambiguous/degraded outcomes; successful
outcomes must not project connector-native success codes or messages as errors. This operator-facing reason is
deliberately retained because the exchange's rejection explanation is often required to repair a
live order failure. It omits raw response envelopes, unsafe exception class metadata, URLs,
credentials, sensitive-marked values, long token-like values, and tracebacks. Live event and text
logs remain private operational artifacts and must not be published without review. This
diagnostic projection does not alter event routing, emitter isolation, executor retries or
re-raises, exchange calls, or trading behavior.

`execution.create_rejected` reports a terminal Bitget UTA insufficient-margin rejection with
`data.error_code="25203"` and `data.retry_delay_seconds`, the new retry window for the event's
symbol and position side. The console warning projects that duration from the event.
`execution.create_skipped` with `reason_code=exchange_margin_retry` reports a subsequent attempt
withheld before submission. It carries the scoped symbol in `data.symbols`, `data.position_side`,
and `data.retry_remaining_seconds` measured from the process-local monotonic deadline when the
batch filter or connector withheld the attempt. Skips retain DEBUG console visibility.

`execution.create_health` reports degradation after five unacknowledged create outcomes,
recovery on acceptance, or the fatal stop decision. Its bounded fields distinguish total
failed creates from unconfirmed outcomes and include elapsed seconds, the affected write
action, the next action, and the latest safe failure details where available. A confirmed
API-wallet rejection may stop on a create or a cancel with zero preceding create failures.

## Fill Confirmation Fallbacks

`fill.position_price_tolerance_used` records the accepted bounded numeric difference
between a reconstructed fill after-state price and the authoritative exchange position
price. It includes the effective connector-aware price tick, absolute delta, delta in
ticks, and applied tolerance; it contains no raw fill or exchange payload. The event is
published only after identity, size, refresh-generation, and price predicates clear.
The existing warning remains the console/text projection, so the structured event itself
does not produce a second console line.

## Eligibility And Market Compatibility

`forager.eligibility_changed` is a bounded structured/monitor-only record of approved/ignored
membership changes. It does not retain config paths, raw sources, or full lists.
Monitor market entries label a forager candidate only when the same live
`is_approved` predicate admits its coin and position side, including minimum
market age. Ranking-feature unavailability is refreshed from the active
orchestrator EMA preparation path on every bundle load. Candidate-only EMA
failures remain distinct from flat active/normal-symbol degradation; monitor
tradability must not relabel the latter as a candidate-only failure.

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

Failure payloads include only bounded `context`, `operation`, `outcome`, optional digit-only
`response_code`, `error_type`, and the envelope's `symbol`. Periodic refresh records may also
include `started_ms` and `elapsed_ms`; current connector-local per-symbol outcomes do not supply
those timing fields. These payloads omit raw responses, exception text, request parameters, URLs,
and tracebacks. An explicit unchanged outcome is DEBUG; confirmed success remains INFO until the
connector can prove whether it changed state; failures retain their existing operator-visible
warning or error. The event route remains structured/monitor-only, and event emission failure must
log only a bounded exception type, preserve exchange configuration control flow and results, and
never replace an original configuration exception.
