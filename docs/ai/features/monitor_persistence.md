# Monitor Persistence

## Contract

`src/monitor_publisher.py` owns monitor appends, snapshots, manifest checkpoints, rotation,
retention, and startup recovery. Persistence failures stay visible but never alter trading behavior.

`state.latest.json` is refreshed by a serialized background maintainer, independent
of whether the trading loop reaches a valid planning snapshot. Successful and
degraded cycles may also request a flush; concurrent requests serialize and the
publisher's configured interval remains the write throttle. The independent
maintainer uses a five-second minimum cadence to keep degraded state visible
without turning diagnostic persistence into a high-frequency trading dependency.
Trailing diagnostics resolve strategy settings once per symbol and position side within
one synchronous section build. Those resolved settings are discarded afterward; configuration
and coin-override changes must be observed by the next snapshot.

Market `strategy_inputs` reports canonical per-side 1m/1h log-range spans (including fractional
spans and symbol overrides), units, nullable prepared values, availability reasons and the last
validated ordinary-input capture time/age. A real observed zero remains available; absent,
malformed or non-finite values are `null`, never zero placeholders. Capture age describes the
last prepared bundle, not independent proof of candle freshness. Missing inputs omit dependent
native trailing calculations and retain explicit `entry_diagnostic_status` and
`close_diagnostic_status`; diagnostic failure cannot discard unrelated snapshot sections.
Complete canonical coefficients determine consumption separately for entry and close. An absent
dimension whose consuming threshold/retracement coefficients are all zero is `not_required`, with
`required=false` and a null observed value; only its inert native diagnostic transport may use
zero. A nonzero consuming coefficient requires the actual prepared value. Available observed zero
remains `available`, even for an optional dimension. Branch diagnostic payloads retain their own
input classifications, so entry unavailability cannot suppress an independent close calculation.
The snapshot-based trailing diagnostic tool rejects unavailable/non-finite indicator imports
instead of replacing them with manual-wizard zeros; genuine observed log-range zeros remain valid.
It accepts `not_required` transport only after independently confirming zero consumption from the
current canonical configuration, including exact-symbol overrides. Without an exchange market
resolver, authored override aliases that cannot be matched exactly conservatively prevent that
zero transport; the importer neither guesses a coin mapping nor fetches market data.
The same proof is required when an older snapshot omits M1 metadata entirely. Missing data with
nonzero or unknown current consumption is rejected; legacy absence is not an implicit zero.
`market.*.tradable_basis=market_and_prepared_inputs_only` explicitly limits the existing point
readiness flag; it does not grant order permission.

The separate `ordinary` section observes the last validated Rust computation and its exact ordinary HSL
capture, independently of the latest fast protective `hsl` observation. It retains native input
and effective modes, active/initial-entry flags, captured HSL permission and bounded unavailability
classifications. Its `stage=computed_native_plan` precedes the HSL owner's final receipt acceptance
and does not imply submission or execution; current write admission remains independent. All rows
are observational, never execution authority. Capture age, input expiry,
account generation changes and pending confirmations distinguish current from stale observations.
Expiry includes the actual balance observation time retained before native evaluation alongside
the consumed position and mark observations. A later unchanged balance refresh cannot renew an
older computation's captured receipt; a wave lacking that timestamp has unknown input age.
At most 128 held/restricted-prioritized rows accompany complete mode/scope/restricted-held counts
and an explicit omitted count. Health summaries show the ordinary observation status/age and held
restriction count separately from strategy close-wait conditions.

`ordinary.pending_confirmations` reports current required account surfaces with requested and
observed epochs. `last_create_filter` and `last_cancel_first_barrier` retain only bounded reason,
count, wave id, required surface names and observation age. These are labelled last observed
outcomes, including after confirmation settles; they do not imply an active barrier. Capture is
independent of optional event sink success. Raw event data, order details and exception messages
are not copied into this projection.

`record_error` retains the event kind, tags, known code-owned source/stage classifications, and a
bounded exception type. It must not persist arbitrary caller strings or numbers, exception messages,
request URLs, responses, tracebacks, or caller-supplied raw diagnostic aliases. More detailed
incident context belongs in an explicitly sanitized event or protected developer path; persistence
behavior and caller control flow remain unchanged.

Private normal-run monitor history may retain an explicitly sanitized incident detail event. The
startup and execution-loop `error.bot.detail` schema stores a bounded normalized frame chain,
correlated to its compact `error.bot` summary by incident id, while excluding exception text,
locals, source lines, URLs, payloads, and credentials. Existing rotation and retention limits bound
this diagnostic data; its production does not depend on DEBUG logging.

During an internal bot restart, a failed or timed-out event-pipeline close may leave its worker
holding the publisher lock. The restart path therefore abandons that process-local publisher
without calling its synchronous close; startup recovery owns any missing final checkpoint. This
bounded observability fallback must not delay exchange-client closure or replacement-bot startup.

Event sequences are monotonic within a monitor root, including across unclean restart. The startup
watermark is the maximum of the manifest and checksummed segment recovery metadata. Never infer an
envelope sequence from payload bytes or an invalid row.

HSL account and health equity are passive observations of fresh committed balance and
positions plus cached held-symbol quotes. Missing or stale inputs produce `null` equity rather
than a startup placeholder or a partial portfolio sum; reporting performs no exchange I/O.

## Recovery Framing

Event rows carry a `_recovery` trailer whose checksum binds the envelope to its sequence. It is
persistence metadata, not a query-facing field.

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

Ordinary appends may coalesce manifest checkpoints to the snapshot cadence. Initialization,
rotation boundaries, successful snapshots, and close force checkpoints. Failed due checkpoints
remain dirty for retry.

Retention protects current files, counts regular files in managed roots, and deletes only direct
non-current candidates using strict age and oldest-first cap pruning. Races must not broaden the
deletion domain.

Window-bounded smoke selection may use `current_segment_started_ms` from the
bounded manifest as coverage proof when no rotated predecessor exists. Missing,
oversized, malformed, or later-than-window evidence remains unavailable rather
than treating `current.ndjson` as complete by assumption.

## Validation

- manifest/row/trailer corruption and oversized rows
- uncheckpointed restart without sequence reuse
- checkpoint retry, rotation, and relay discovery
- retention order, protection, accounting, races, and symlinks
- disk-full behavior, concurrency, and event/report regressions

## Key Code And Tests

- `src/monitor_publisher.py`
- `src/live/event_bus.py`
- `tests/test_monitor_publisher.py`
- `tests/test_live_event_bus.py`
