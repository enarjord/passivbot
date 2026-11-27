# Account, Balance, And Risk Events

Topic file for [structured live events](../live_events.md). The envelope, payload budget,
redaction, and validation rules there apply to every event below.

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
omitted row. Deferred HSL account reports publish one initial balance snapshot, then raw/snapped or
composition transitions, using presentation-only anchors and passive cached equity (unknown
when current inputs are absent). Publication performs no fetches or execution scheduling. Presentation anchors advance only
after successful event enqueue or console fallback, so transient observer failures can retry.
Balance-log equity requires complete held-position coverage and confirmed account state. It
may reuse quotes already obtained for position logging within its 60-second diagnostic allowance,
including completed-candle fallback; it performs no additional fetch. Trading and risk retain their
own stricter freshness checks. `equity_estimated`, `equity_valuation_source` (market_quote,
completed_candle, mixed, or flat), and `equity_observation_age_ms` describe the valuation. Age is
since observation, not the underlying candle timestamp. Older or candle-derived estimates are
labelled on the console. Missing, invalid, stale, or pending inputs leave equity unavailable with
`equity_unavailable_reason`; partial PnL is never published as total equity.

Console admission thereafter remains based solely on snapped-balance materiality. Visible balance lines may append a sanitized sample of at most two
retained assets; composition-only changes remain structured/text durable but
stay off the console.

`sink.degraded` identifies the failed sink, stable failure reason, and exception type. It must not
retain exception text, request URLs, credentials, response bodies, paths, or other arbitrary values
from the sink exception. Sink counters and pipeline timings remain available through health
snapshots independently of the exception payload.

Queue-full and pipeline-closing degradation diagnostics retain a canonical registered event type
only. Unsafe, malformed, sensitive-marker-bearing, or unregistered labels normalize to
`unknown_event` in the drop counter, message, and `dropped_event_type`; canonical event labels are
unchanged. Monitor-delivery and generic emitter fallback warnings follow the same bounded event-type
and exception-type rule. These diagnostics remain isolated from event routing, queue semantics,
sink timing/counters, caller return values, and trading behavior.

## Risk input readiness
`risk.input.status` reports live balance-input readiness: `deferred` blocks ordinary planning,
`succeeded` reports recovery, and `failed` reports terminal exhaustion of the local attempt budget.
The producer in `live/risk_input_recovery.py` emits code-owned reason/action strings and bounded
numeric diagnostics: current raw/sizing balances, first invalid replay timestamp/balance, invalid
row count, replay start/end, `retry_count` (failed attempts including the initial failure),
`max_attempts`, and `retry_delay_seconds`. Non-finite numbers become null. Each counted failure
emits a warning, except the final failure which emits an error with `action=stop_without_restart`
and zero delay. The first and final failures also attach a bounded `traceback` frame chain and
log its frame-only text representation. Raw payloads, exception text, and locals are excluded.
Failed event emission falls back to standard logging. The pipeline also logs these recovery
records directly if its configured console sink raises, since an accepted event alone does not
prove console delivery. Healthy console delivery produces one attempt record.
Readiness polls inside the retry deadline neither increment the count nor repeat the warning.
Recovery is immediate after the owning operation succeeds. See `../equity_hard_stop_loss.md` for
retry, terminal-stop, and protection policy.

## HSL Observations

HSL emits `hsl.status` with `engine=hsl` and one bounded aggregate per
qualitative scope/action/availability/estimate or raw-RED/EMA-pending change, including stale/current observation recovery. This is passive observation, never a gate
or retained trading permission. Numeric metrics refresh after each protective execution wave in the
monitor snapshot even when no new status event is emitted. Sink/projection failure cannot inhibit
risk evaluation or exchange execution.

Connector admission evaluates current risk without diagnostic projection or synchronous event sinks.
Decision captures are separate from reporting. A protective wave reports after its order work,
including when no exit work remains, so synchronous HSL sinks cannot age its inputs before a write.
Diagnostic expiry uses the actual held-position mark timestamps consumed by evaluated scopes;
quotes cached for disabled scopes cannot expire another scope's display. The bounded legacy
aggregate preserves last RED attention, while a stale or failed GREEN observation is labelled
stale or unavailable instead of advertising current GREEN. Per-scope rows retain the last observed
native decision alongside explicit freshness.

The hsl monitor `hsl` section has `schema_version=1`, `signal_mode`, `observation_status`,
`captured_at_ms`, `age_ms`, current account availability, complete scope counts and up to 128 scoped
rows. RED scopes come first, then unavailable, raw-RED/EMA-pending and estimated scopes. `omitted_scopes` discloses
truncation. Status events carry at most three rows; console summaries carry counts. A top-level aggregate tier
keeps RED visible to existing risk reports and startup previews without inventing an aggregate
drawdown score for independent scopes. Rows identify
symbol/position side where applicable, native action, GREEN/RED or diagnostic inactive status,
raw/EMA/selected drawdown, configured threshold, RED/flat evidence times and approximation reasons.
There is one portfolio scope in unified mode; side and coin modes retain their native topology.
Unavailable input is never displayed as GREEN. Removed legacy tiers do not reappear in hsl
payloads.

The optional `candle_sources` diagnostic describes failures in the latest already-acquired HSL
source batches, independently of decision freshness. It contains the complete `failure_count`,
up to 16 deterministic `failures` rows (symbol, timeframe, fetch/cache stage and bounded exception
type), and `omitted_failures`. Failed cache fallbacks precede fetch failures before sampling, so
terminal source loss remains distinguishable from a fetch failure with successful cache recovery.
Exception text and raw exchange payloads are never included. Source
failure changes and recovery emit a status transition, including changes in omitted rows; repeated
unchanged failures do not. The console shows one compact source cause when no current scope
unavailability reason takes priority, including in the legacy/sink-error console fallback. This
view neither fetches inputs nor changes risk reasons,
readiness, retry policy or trading authority.

An optional bounded `unavailable_scope` retains the first required-input cause before scope
sampling, even when RED rows fill the event or monitor sample. Both consoles preserve that cause
ahead of candle failures. Structured console records reserve space for counts and the primary
cause, then include account, stale, scope and approximation context only while the complete
record fits 240 characters; omitted console context remains in the structured observation.

Smoke reports and dashboard event summaries consume these scoped status rows directly,
including current raw-RED/EMA-pending loss and halted terminal timestamps. Complete
`action_counts` and `raw_pending_scope_count` accompany sampled rows; omitted scope
counts remain explicit, so samples do not imply complete per-symbol coverage. Expired
observations are retained as historical evidence but do not advertise active cooldown,
current proximity or pending RED. Historical event tapes may still be read in their old
schema; the current controller does not emit the retired event lifecycle.

An observation becomes visibly stale when its captured input TTL expires (including the exact
retained position timestamp used by evaluation), account confirmation is
pending, or its account generation changes. The last decision remains labeled as an observation;
monitor reads do not initialize an HSL owner, perform I/O, or reevaluate risk. Diagnostic state is
replaced after each protective wave and is not persisted as restart authority. A first projection
failure is explicitly diagnostic-unavailable, and unavailable-scope fallback logs retain warning
severity when event emission fails or the configured console sink reports a write failure.

## Trailing And Unstuck Status Materiality

`trailing.status` retains one complete observation every five minutes while its producer is
active, with an hourly unchanged reminder. `unstuck.status` is observed in the normal execution
loop and deferred-planning path: the first observation and semantic changes print immediately,
unchanged observations remain structured every five minutes, and unchanged INFO reminders use
20 minutes. `operator_visible=false` suppresses only console and durable text projection.
The `changed` field excludes unchanged reminders.

Realized-close diagnostics report proposed and final quantities, execution route, estimated
loss, binding cap and remaining headroom. The gate policy carries a boolean and a nullable
cap, including an active zero cap. The Rust debug profile includes bounded
`close_allocation_sample` rows in the orchestrator return event. Live transport reports
`realized_close_admission_deferred` when the current plan lacks account-wide reservation
authority. `risk.realized_close_recovery` reports the pending reservation count when an
exact terminal lookup has no usable evidence, is temporarily unavailable, or is unsupported.
The same event reports duplicate exchange order IDs, conflicting fill identities, fills
aggregated across multiple order IDs, and fills exceeding the submitted quantity. It also
reports resting orders with unknown reduce-only status or execution type, missing
remaining quantity, contradictory side, or duplicate client IDs. Lookup diagnostics
repeat at most once every five minutes. Reservation warnings repeat at most once per
reason within the same interval.

Rust classifies held positions at the actual unstuck gates and reports allowance, exposure/EMA
thresholds, projected post-fill exposure, eligibility and proposal details. Final native orders determine `selected`;
`loss_gate_blocked`, `superseded`, and `finalization_filtered` retain the proposal without
implying submission. Python
retains at most three samples per side, aggregate reason counts, allowance range (coin overrides
may differ), and plan age. No completed plan is `awaiting_plan`; observations at least five
minutes old are `stale_plan`, with no current selection displayed. The open-order flag is
account-wide exchange-observed state, separate from native selection. Neither grants execution
authority. Observations perform no exchange fetch and failures remain isolated from trading.
Visibility checkpoints advance only after accepted event enqueue or successful legacy console
projection, allowing a rejected initial or changed event to retry. Diagnostic collection is enabled
for one-shot live/JSON planning; the reused-workspace backtest path skips it.

Unstuck allowance movement uses five-percent relative hysteresis against the last visible
baseline, including accumulated movement and immediate zero/positive crossings. Candidate-only
churn and numeric movement print at most every five minutes; reason/count and open-order
transitions remain immediate. Ordinary price and target movement do not trigger INFO. Its
console uses bounded records per side, plus one watch/proposal record when available.

Trailing visibility is owned per
`symbol`/`pside`/`kind` item so one changing position does not expose unrelated repeats. Its
threshold and retracement ratios are material at an absolute ratio delta of `0.0005` (0.05
percentage points); threshold and retracement prices are material at a relative delta of `0.005`
(0.5 percent). Qualitative status, support, and trigger flags remain immediate.

These thresholds affect human projection only. They must not change Rust planning, order emission,
position handling, unstuck eligibility, trailing calculations, the observation cadence,
or event payload values. Legacy direct INFO logging remains a fallback only when the structured
console is unavailable and follows the same producer-owned admission decision.

When a trailing observation is operator-visible, its console/text projection must remain within the
normal 240-character record budget. Keep the action/status, `symbol`/`pside`, selected mode,
threshold and retracement gate states and material values, current price, and available correlation
while using compact labels. Complete diagnostics remain in structured/monitor data; compaction must
not remove payload fields or change admission, cadence, or behavior.
