# Monitor Persistence

## Contract

`src/monitor_publisher.py` owns monitor appends, snapshots, manifest checkpoints, rotation,
retention, and startup recovery. Persistence failures stay visible but never alter trading behavior.

Event sequences are monotonic within a monitor root, including across unclean restart. The startup
watermark is the maximum of the manifest and checksummed segment recovery metadata. Never infer an
envelope sequence from payload bytes or an invalid row.

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
