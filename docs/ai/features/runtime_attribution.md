# Runtime Attribution

## Contract

`passivbot tool runtime-attribution` reads local fill caches, monitor history,
runtime manifests, structured startup events, and bounded text startup logs. It
does not contact exchanges, inspect processes, mutate caches, or control bots.

The report keeps two claims separate:

1. `first_ingestion` is `recorded` only when a fill itself carries
   `first_ingested_by_runtime` provenance. This proves which runtime first
   persisted that fill locally, not which runtime submitted its order.
2. `producer_attribution` correlates the exchange fill timestamp with observed
   runtime start windows. Even a single matching window is a candidate, sets
   `proven=false`, and requires client-order IDs plus contemporaneous
   order/execution logs for stronger attribution.

Legacy fills without embedded provenance remain explicitly unattributed. The
tool never assigns them to the runtime that happened to read or re-save them.
Legacy startup banners can establish a runtime window without establishing a
version. A bounded startup-log run-id prefix merges with a complete
manifest/event identity only when it is the producer's exact 12-character,
lowercase-hex `uuid4().hex[:12]` form; exchange, user, and prefix agree
uniquely; and their start timestamps differ by at most two seconds. Ambiguous,
incomplete, malformed, or out-of-bound identities remain separate observations.

## Inputs And Output

Defaults scan:

- `caches/fill_events` for canonical daily fill arrays
- `caches/runtime` for immutable per-run manifests
- `monitor` for manifests, structured events, and fill history
- `logs` for startup banners and bounded runtime identity lines

Repeat `--exchange`, `--user`, or `--symbol` to filter the report. Use
`--trailing-only` to isolate trailing order types. `--since-ms` and `--until-ms`
bound exchange fill timestamps. File count, fill-record count, per-file bytes,
and total scanned bytes fail closed; parse warnings are capped and report only
path and stable error classification, not file contents.

`--fail-on-unattributed` exits 1 if any selected fill lacks recorded
first-ingestion provenance. It does not treat a candidate producer window as
proof.

```bash
passivbot tool runtime-attribution --exchange bybit --user account_01 --compact
passivbot tool runtime-attribution --trailing-only --fail-on-unattributed
```

## Validation

- recorded versus legacy first-ingestion classification
- runtime-window candidate boundaries across consecutive starts
- runtime manifest, monitor event, bounded startup log, and fill-history parsing
- fill deduplication across cache and monitor history
- trailing-only filters and fail-closed scan limits

## Key Code

- `src/live/runtime_attribution.py`
- `src/tools/runtime_attribution.py`
- `tests/test_runtime_attribution.py`
