# Monitor Persistence

## Contract

`src/monitor_publisher.py` owns monitor appends, snapshots, manifest checkpoints, rotation,
retention, and recovery. Persistence failures stay visible and trading decisions remain independent.
The [relay](monitor_relay.md) reads the published files.

Snapshots, event rows, and history rows require strict JSON. Reject NaN and infinity, including
after NumPy conversion, through the existing persistence-failure policy. Failed snapshots keep
the previous snapshot and checkpoint. Failed event or history serialization appends no partial
row. Event rejection precedes sequence reservation and rotation. Uncertain append I/O reserves
its sequence to prevent reuse. The candle history cursor advances through persisted rows and stops
at the first failure, allowing corrected data to retry without a gap. Later valid writes recover.
Preserve unknown values and trading inputs when serialization fails.

## Snapshot Production

A serialized background maintainer refreshes `state.latest.json` independently of planning.
Successful and degraded cycles may also request a flush. The publisher interval throttles all
writes, and the independent maintainer has a five-second minimum cadence. Snapshot writes run off
the event loop. The maintainer captures the event sequence before writing, so event publication
can advance during the write.

HSL account and health equity use fresh committed balance and positions plus cached held-symbol
quotes. Missing or stale inputs give null equity. Reporting performs no exchange I/O.

### Trailing and Strategy Inputs

Monitor trailing projections distinguish available, disabled or not required, and unavailable
extrema or arithmetic overflow. Unavailable prices and dependent price changes are null with
explicit diagnostic status and reason. [Structured events](live_events.md) preserve that
classification and dashboards show unknown for null or absent values. Native sentinels and
order results keep their trading semantics. Each synchronous section build resolves strategy
settings once per symbol and position side, then discards them so the next snapshot sees config
and coin-override changes.

Market `strategy_inputs` records canonical per-side 1m/1h log-range spans (including fractional
spans and symbol overrides), units, nullable prepared values, availability reasons, and the last
validated ordinary-input capture time and age. Age describes that prepared bundle. Observed zero
is available, while absent, malformed, or non-finite values are null. The
`market.*.tradable_basis=market_and_prepared_inputs_only` flag describes point readiness.
Order permission and candle freshness follow their respective trading input contracts.

Canonical coefficients determine input consumption independently for entry and close. A missing
dimension with all consuming threshold and retracement coefficients zero is `not_required`,
with `required=false` and a null observation. Only inert native diagnostic transport may use zero.
Any nonzero consuming coefficient requires the prepared value. Missing inputs omit dependent
native calculations with explicit `entry_diagnostic_status` and `close_diagnostic_status`.
Each branch retains its classification, so independent diagnostics and other snapshot sections
survive failures.

The [trailing diagnostic tool](trailing_diagnostics.md) rejects unavailable or non-finite snapshot
inputs and accepts observed log-range zeros. It accepts zero transport for `not_required` or
missing legacy M1 metadata only after independently proving zero consumption from current canonical
configuration, including exact-symbol overrides. Without an exchange market resolver, unresolved
override aliases prevent that proof. Import performs no exchange lookup or guessed mapping.

### Ordinary Planning and Progress

`ordinary` observes the last validated Rust computation and its exact ordinary HSL capture.
`hsl` observes the latest fast protective computation. Ordinary rows retain native input and
effective modes, active and initial-entry flags, captured HSL permission, and bounded
unavailability classifications. `stage=computed_native_plan` precedes final receipt acceptance.
Write admission, submission, and execution remain independent.

`input_receipts` projects the immutable `PlanningSnapshot` used by Rust, including required quote
symbols, receipt counts, and oldest quote time under the captured quote TTL. Balance, positions,
and open-order times use the captured account TTL. Missing, future, incomplete, or mismatched
receipts give unknown or stale observations. Required surface epochs are checked. Fills and
completed candles retain coverage and generation requirements. Quote age applies to quotes.

Capture age, input expiry, account generation, and pending confirmations distinguish current from
stale observations. Captured HSL scopes also use the position, mark, and actual balance
observation times consumed before native evaluation. A required missing timestamp gives unknown
age. Disabled or uncaptured HSL scopes cannot substitute for the ordinary quote cohort. A later
balance refresh, market cache update, or replacement planning snapshot cannot renew the retained
computation. Prioritize held and restricted rows within the 128-row limit. Retain complete mode,
scope, and restricted-held counts plus the omitted count. Health summaries report ordinary status,
age, and held restriction count separately from strategy close-wait conditions.

`health.ordinary_progress` records accepted plans after reconciliation returns, including valid
no-ops. This measures reconciliation progress. It keeps the ages of the last acceptance and
completed outer cycle, and the span of outer cycle completions without acceptance. Unknown ages
are null. The last defer after acceptance retains known phase, reason, and missing input surface.

`ordinary.pending_confirmations` reports required account surfaces and requested and observed
epochs. `last_create_filter` and `last_cancel_first_barrier` show the last observed bounded reason,
count, wave id, required surfaces, and age, including after confirmation settles. Progress and
confirmation capture are independent of optional event sinks. These outcomes exclude raw event
data, order details, and exception messages.

## Errors and Restart

`record_error` retains event kind, tags, known code-owned source and stage, and bounded exception
type. Exclude arbitrary caller strings and numbers, raw diagnostic aliases, exception messages,
request URLs, responses, and tracebacks. Use an explicitly sanitized event or protected developer
path for more detail. Keep persistence behavior and caller control flow independent.

Private monitor history may retain the sanitized
[`error.bot.detail` incident record](live_events.md#execution-loop-incidents), correlated to
`error.bot` at normal logging levels. Its frame projection excludes credentials. Rotation and
retention bound that diagnostic data.

During an internal restart, a failed or timed-out event-pipeline close may leave a worker holding
the publisher lock. Abandon that process-local publisher and let startup recover any missing
checkpoint. Exchange-client closure and replacement-bot startup proceed without synchronous
publisher close.

## Recovery Framing

Sequences increase monotonically within a monitor root across unclean restarts. Start at the
maximum of the manifest watermark and checksummed segment recovery metadata. Event rows carry a
`_recovery` trailer whose checksum binds envelope and sequence. Recover sequence numbers only
from verified envelopes. Queries exclude the trailer.

Recovery must:

- remain bounded for rows larger than a scan chunk
- accept no-final-newline and CRLF framing
- ignore malformed, torn, altered, or checksum-mismatched rows
- ignore payload fields resembling envelope or recovery metadata
- retain the manifest watermark when no higher valid row exists
- tolerate older rows without recovery trailers

Framing changes require restart tests proving an appended but uncheckpointed event cannot cause
sequence reuse.

## Checkpoints And Retention

Ordinary appends may coalesce manifest checkpoints to snapshot cadence. Initialization,
rotation boundaries, successful snapshots, and close force checkpoints. Failed due checkpoints
remain dirty for retry.

Retention protects current files, counts regular files in managed roots, and deletes only direct
non-current candidates using strict age and oldest-first cap pruning. Races must preserve that
deletion domain.

Window-bounded smoke selection may use `current_segment_started_ms` from a bounded manifest as
coverage proof when no rotated predecessor exists. Missing, oversized, malformed, or
later-than-window evidence makes coverage unavailable.

## Validation

- manifest, row, and trailer corruption and oversized rows
- uncheckpointed restart without sequence reuse
- checkpoint retry, rotation, and relay discovery
- retention order, protection, accounting, races, and symlinks
- disk-full behavior, concurrency, and event/report regressions

## Key Code And Tests

- `src/monitor_publisher.py`
- `src/live/event_bus.py`
- `tests/test_monitor_publisher.py`
- `tests/test_live_event_bus.py`
