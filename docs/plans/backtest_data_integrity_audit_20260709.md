# Backtest Data Integrity + Artifact-Build Performance Audit — 2026-07-09

Deep audit of the market-data pipeline feeding backtests and the optimizer, following the
pre-audit survey (`pre_audit_survey_20260709.md`, target 2). Four parallel deep traces:
materializer→Rust handoff, timestamp/boundary conventions, prepare_hlcvs error paths and
cross-process coherence, and artifact-build performance. Key findings were independently
re-verified against source before inclusion. Verdicts: CONFIRMED (traced end-to-end with a
concrete failure scenario), PLAUSIBLE (mechanism verified, impact depends on external behavior),
REFUTED (lead was wrong; mitigation documented).

Complementary to `backtester_performance_optimization_plan.md` (Rust sim loop); this audit covers
the Python preparation path.

---

## Part 1 — Data-integrity findings (ranked)

### I-1. AlphaVantage provider misfiles all candles by 4-5 hours (TZ/DST-dependent) — CONFIRMED, HIGH

`src/tradfi_data.py:301-306`: Alpha Vantage returns US-Eastern wall-clock strings; the code parses
them into a **naive** datetime and calls `.timestamp()`, which interprets them in the **host's
local timezone**. The comment acknowledges the simplification. The error is a whole number of
minutes, so every downstream alignment assert passes (`write_rows`'s `% 60000` check,
`standardize_gaps` boundary flooring), and the day-window filter at
`candlestick_manager.py:5150` silently drops the candles pushed outside the UTC day.

Failure: every stock-perp backtest using `tradfi.provider == "alphavantage"` gets prices shifted
4-5 h (host-TZ- and DST-dependent), with edge candles dropped, no error raised. Scope: opt-in
provider (default is yfinance), but silently wrong when selected.

Other providers verified correct: Finnhub (epoch-seconds `//1000` matches docs; endpoint now
premium-gated so effectively dead), Polygon (ms passthrough), Alpaca (tz-aware ISO), yfinance
(tz-aware index). None of the conversions have tests.

Fix: `dt = datetime.strptime(...).replace(tzinfo=ZoneInfo("America/New_York"))` before
`.timestamp()`. Add unit tests for all provider conversions.

### I-2. Interior data gap silently discards the shorter real segment — CONFIRMED, MEDIUM-HIGH

`src/backtest_dataset_materializer.py:281-297`: a coin's tradable window is
`_longest_contiguous_valid_span` computed before gap-filling. When an interior gap (e.g. exchange
outage) splits a coin's history into segments, only the longest contiguous segment is used; the
shorter **real** segment is excluded from the backtest entirely. No warning fires —
`warn_hlcv_valid_range_coverage` (`src/backtest.py:1878-1923`) only covers leading/trailing/
outside-range cases, never interior loss. The synthetic fill written into the gap is never
consumed on this path (wasted I/O; see P-8).

Failure: coin with 1000 + 4000 bar segments around an outage backtests on 4000 bars; 1000 bars of
genuine history vanish silently.

Fix: warn when `source_valid_count` materially exceeds the longest-contiguous span; skip the dead
fill when `synthetic_gaps_tradable` is false.

### I-3. Synthetic *tradable* candles in the stock-perps (`xyz:`) path — BY DESIGN (maintainer decision 2026-07-09); document instead of fix

`_fill_sparse_hlcv_gaps` (`backtest_dataset_materializer.py:36-49`) forward-fills interior gaps
with flat h=l=c=prev_close, v=0 candles; with `synthetic_gaps_tradable=True` (only
`ohlcv_source_dir`/`xyz:` stock coins) they land inside the tradable window and Rust treats them
as tradable.

**Maintainer decision: keep them tradable.** Rationale (verified against the code):

- The live venue (Hyperliquid HIP-3 xyz) trades 24/7 with no evening/weekend halt; third-party
  RTH-only data is used only because real xyz candle history is short. Marking closure windows
  non-tradable would be *less* faithful to live, and keeping them tradable avoids
  tradable/non-tradable bookkeeping in the Rust backtester.
- Fill leakage is minimal by construction: fills use strict inequalities
  (`order_filled`, `backtest.rs:4630-4642` — `low < bid` / `high > ask`), so orders at exactly
  `prev_close` never fill on a flat candle. Only orders priced strictly inside the flat price
  fill — and such crossing orders would also fill immediately on the live 24/7 venue at
  ≈prev_close, so the backtest is consistent (marginally pessimistic: filled at the limit price
  rather than at market).

Documented modeling caveats (accepted, to be written into stock-perps docs):

1. Flat-at-prev-close understates overnight/weekend drift of HIP-3 prices; triggers (HSL,
   trailing, unstuck) that would fire overnight in live cannot fire in backtest, and the next
   real candle absorbs the whole move as one discontinuity.
2. Volume=0 for ~73% of minutes (RTH ≈ 390/1440) dilutes rolling volume metrics. Forager
   `volume_score`/`volume_drop_pct` (`coin_selection.rs`) will systematically rank xyz coins
   below crypto in mixed universes and may drop them via the volume filter — worth a targeted
   check if stocks + forager are combined.
3. EMA spans run over synthetic flat minutes, so configs optimized on synthetic-filled history
   may shift behavior as real 24/7 xyz history replaces it (and there is a regime seam where
   third-party data hands over to real xyz data).

Action: document the model in `docs/stock_perps.md` + `docs/ai/features/stock_perps.md`; surface
per-coin synthetic-fill share in backtest output (coverage metadata already computes it).

### I-4. False "short tail" confirmation silently clips recent-end backtests for 7 days — CONFIRMED mechanism, MEDIUM-HIGH

`_fetch_coin_range_into_v2_store` (`hlcv_preparation.py:3287-3334`): a truncated trailing tail is
only accepted after two attempts with identical boundary/request-end/last-ts — good idea — but
nothing requires the two attempts to be temporally separated. Once confirmed, the missing tail is
marked a **persistent** v2 gap (`_mark_sparse_fetch_gaps`, `:3452-3467`) with
`next_retry_at = now + 7d` (`ohlcv_catalog.py:14,261-264`). An exchange publishing delay observed
twice at the same `last_ts` therefore becomes a week-long silent clip: subsequent recent-end
backtests take the partial-coverage path (`_usable_partial_coverage_window`,
`hlcv_preparation.py:2213-2219`, reached at :1772/:1869/:1986/:2046 and NOT gated by
`allow_partial_window`) and end each affected coin at the false boundary, logged only at INFO.

Fix: require corroborating attempts to be separated in time (distinct `created_at` beyond a
minimum interval), and use a much shorter `next_retry_at` for `trailing_unavailable` gaps whose
end is recent.

### I-5. Leading/internal gaps become persistent on a single observation — CONFIRMED, MEDIUM

Same code path (`hlcv_preparation.py:3244-3271,3351-3369,3457-3467`): unlike the corroborated
trailing tail, `leading_unavailable` and `internal_gap` are stamped `persistent=True` on the first
fetch that observes them. A transient partial response with a hole exceeding tolerance blocks or
partial-covers the coin for 7 days. Mitigated by the 7-day retry; most exchanges omit no-trade
minutes so internal holes are usually genuine.

Fix: apply the same two-observation corroboration to leading/internal gaps.

### I-6. Kucoin cross-page holes recorded as permanent, never-expiring `no_trades` gaps — CONFIRMED, MEDIUM

`candlestick_manager.py:845,4263-4266,2880-2898`: with `_record_payload_gaps_as_known` (kucoin
only), holes **between pagination pages** — not just within a single response — are recorded with
reason `no_trades` at max retry count. `no_trades` is non-expiring (`:177-178`), so a pagination
stall/outage permanently masks a real minute-range; only `force_refetch_gaps` ever re-checks.

Fix: only intra-payload holes deserve `no_trades`; classify between-page holes with an expiring
reason (`auto_detected`).

### I-7. Wrong-but-nonzero cached first-timestamp can clip a coin's start — PLAUSIBLE, MEDIUM

`hlcv_preparation.py:571-586,1192-1196`: the inception probe tries `since=2018-01-01` then falls
back to `since=2020-01-01`. If an exchange rejects/empties the far-back probe for a coin that
launched in between, the too-late first-ts is cached persistently and, via
`adjusted_start_ts = max(..., first_ts_guess)`, overrides the earlier unified inception. The clip
prevents ever fetching the earlier data that would self-correct it.

Fix: treat probe results later than the unified inception with suspicion (re-probe or prefer
unified); expire cached first-timestamps.

### I-8. `read_range` verifies against a catalog snapshot taken outside the chunk lock — CONFIRMED, MEDIUM (fails loud)

`ohlcv_store.py:315,363-364,415-430`: chunk checksums are snapshotted via `list_chunks` before
the per-chunk lock is acquired. A concurrent rewrite between snapshot and lock makes the fresh
body hash mismatch the stale expected checksum → spurious
`ValueError("checksum mismatch")`; the repair path re-lists, finds nothing corrupt, and aborts the
whole prep (`hlcv_preparation.py:2100-2108`). Two workers preparing the same coin can kill a run
with a bogus corruption error. No silent bad data.

Fix: re-read the chunk record from the catalog inside the chunk lock before verifying.

### I-9. `_verified_checksums` false hit on coarse-mtime filesystems — PLAUSIBLE, LOW

`ohlcv_store.py:146-156`: the cache key includes `mtime_ns`+size+expected checksum, which defeats
staleness on modern filesystems. On coarse-mtime filesystems (FAT/NFS), an in-place same-size
rewrite within one mtime tick could produce a silent stale read (chunks are fixed-size and writes
are in-place memmap flushes, not tmp+rename).

### Refuted leads (mitigations verified — don't spend time here)

- **NaN edges reaching Rust price/EMA math**: refuted. `_validate_hlcvs_valid_windows_from_mss`
  (`backtest.py:709-722`, on all three prepare paths) hard-fails on non-finite values inside any
  valid window; Rust double-guards every hot read (`coin_is_valid_at` + `is_finite`), EMA update
  early-returns on non-finite, delist-mid-backtest force-closes. The `compute_bands` NaN-panic in
  `backtest.rs` is unreachable from this pipeline. (Residual: aggregated `candle_interval>1`
  arrays are not re-validated after aggregation — low risk.)
- **Boundary off-by-ones store↔catalog↔materializer**: refuted. Inclusive-`end_ts` + `+1`
  exclusive-slice convention is uniform; month-chunk seams compose with no drop/duplicate; traced
  concretely across a Jan→Feb boundary.
- **`ensure_millis_df` unit heuristic**: effectively defended. Realistic misguesses are caught by
  `write_rows`' minute-alignment and month-bounds asserts (crash, not corruption). Worth adding an
  explicit epoch-ms sanity assert, but not currently exploitable with real data shapes.
- **`canonicalize_daily_ohlcvs` bfill / `fill_gaps_in_ohlcvs` / `attempt_gap_fix_ohlcvs`**: not
  reachable from the live pipeline. First is tool-only (`pad_historical_daily`), other two are
  dead in production (test-only callers). Candidates for deletion.
- **Stale-lock sweeper double-hold**: refuted. Sweeper only unlinks a lockfile it can itself
  acquire (holder dead); watchdog logs but never force-releases. Verified by existing tests.
- **Failed inception probe drops coin**: refuted. `0.0` sentinel is treated as falsy everywhere
  and degrades to the unified inception.
- **Whole-run silent truncation on fetch errors**: refuted. Transient network/corrupt-chunk
  exceptions propagate to hard errors; `standardize_gaps` marks synthetic rows `valid=False`.

---

## Part 2 — Artifact-build performance findings (ranked by expected wall-clock win)

Context: the persistent cache (`caches/hlcvs/…/hlcvs.npy.gz`, keyed by `get_cache_hash`,
`backtest.py:1421`) is correctly keyed and shared; suite shares one dataset across scenarios via
SharedMemory; the optimizer builds once per exchange. No spurious-rebuild bug exists. The
slowness is in (a) cache-hit load/verify and (b) cold-build serialization.

### P-1. Cache hit gzip-decompresses the multi-GB hlcvs artifact TWICE and full-hashes it once

`backtest.py:1571` → `verify_hlcvs_manifest` → `_verify_array_artifact`
(`hlcvs_manifest.py:247-248`) fully decompresses `hlcvs.npy.gz` and sha256-hashes it; then
`backtest.py:1611` decompresses the same file again for actual use. Single-threaded gzip at
~200-400 MB/s over a 10-20 GB array ≈ 30-60 s per pass, twice, plus the hash pass. This is the
dominant cache-hit cost. Fix: load once, pass the in-memory array to the verifier. Risk: none
(same bytes, same check).

### P-2. Manifest verification re-hashes the full array on every load

`hlcvs_manifest.py:233-262`; `hash_logical_array` (`:39-47`) additionally does
`ascontiguousarray(...).tobytes()` — a full-size extra copy — before hashing. Fix: store
size+mtime_ns in the manifest and skip the deep hash when unchanged (same trick as
`OhlcvStore._checksum_cache_keys_for_paths`); keep full hash behind a verify flag. Drop the
`tobytes` copy (hash a memoryview). Risk: low-moderate (weakens local tamper detection; flag it).

### P-3. Cold build resolves coins fully serially in the single-exchange path

`try_prepare_hlcvs_v2_local` (`hlcv_preparation.py:1178-1222`) awaits `_resolve_v2_store_range`
per coin in a plain for-loop, while the combined path already gathers under
`asyncio.Semaphore(6)` (`:3957-3985`). Hundreds of coins → hundreds of serialized
I/O+checksum round-trips. Fix: mirror the combined path's semaphore+gather; push chunk
copy/hash into `asyncio.to_thread` (hashlib/numpy release the GIL). Prerequisites: P-6/P-7.
Impact: multi-minute → sub-minute on wide coin sets.

### P-4. Cold build reads every coin's chunk bodies twice

Resolve phase (`_resolve_v2_store_range` → `store.read_range`) copies full chunk bodies but the
caller consumes only timestamps ends + valid flags (`hlcv_preparation.py:1219-1220`); materialize
then re-reads identical chunks (`copy_range_into`). Fix: valid-only read in the resolve phase, or
thread the materialized range through. Roughly halves cold-build read I/O.

### P-5. Per-chunk checksum copies the whole chunk before hashing; verification cache is per-process

`_compute_chunk_checksum` (`ohlcv_store.py:374-390`) does `ascontiguousarray(body).tobytes()` —
e.g. ~8 GB copied + hashed per cold process for 300 coins × 36 months. Fix: hash the mmap
buffer via memoryview (no copy, identical hash); optionally persist verified
(checksum, mtime_ns, size) in the catalog so unchanged chunks skip re-hashing across processes.

### P-6. New sqlite connection per catalog operation

`OhlcvCatalog._connect` (`ohlcv_catalog.py:70-73`) opens a fresh connection for every
list/get/register call — thousands over a wide cold build. Fix: cache one connection per
instance (lock-guarded / thread-local if P-3 lands).

### P-7. Read path takes an exclusive portalocker write-lock per chunk

`_copy_chunk_into_range` (`ohlcv_store.py:363`) wraps read-only copies in the chunk *write* lock —
thousands of serialized FS lock ops per cold build, and it defeats P-3's parallelism. Fix:
shared/read lock for pure reads, keep exclusive locks for `write_rows`/invalidate. (Also the
natural place to fix I-8: re-read the catalog record inside the lock.)

### P-8. `hlcvs[:] = np.nan` writes the entire memmap before copying real data

`backtest_dataset_materializer.py:231,404`: full multi-GB write, mostly overwritten immediately.
Fix: NaN-fill only the complement of each coin's copy window. Moderate care needed (assert
coverage in tests). Also: skip the dead interior fill when not tradable (see I-2).

### Missing observability

There is no timing around manifest verify vs gzip decompress (the P-1/P-2 hotspots) — only a
combined "Seconds to load cache" log (`backtest.py:1969`). Add split timers around
`verify_hlcvs_manifest`, `try_prepare_hlcvs_v2_local`, and `materialize` first, to confirm the
ranking on real datasets before/after each fix.

### Cheap wins (< 1 h each)

P-1 (delete the second decompress), P-5's memoryview hash (also in `hash_logical_array`), P-6
(reuse sqlite connection), split timing logs.

---

## Part 3 — Test-coverage gaps surfaced by the audit

- TradFi provider unit/timezone conversions: zero tests (`test_stock_perps.py` only checks the
  factory). `test_tradfi_providers.py` is a manual live-API script.
- Rust-side consequence of `synthetic_gaps_tradable=True` (I-3): untested at the boundary.
- Interior-gap coverage loss (I-2): tests assert the truncation happens, none flag it as loss.
- Short-tail false-positive across runs / 7-day persistence (I-4): the corroboration test
  (`test_ohlcv_v2_prepare.py:281-340`) doesn't cover it.
- Single-observation persistent leading/internal gaps (I-5), kucoin cross-page `no_trades` (I-6).
- No two-process tests for lock sweep or `_verified_checksums` staleness (I-8, I-9).
- `legacy_data_migrator.py`: no tests at all.

---

## Suggested action-plan slices (in order)

1. **Quick integrity fix**: I-1 AlphaVantage timezone (one-line + provider conversion tests).
2. **Quick perf wins**: P-1 + P-5 memoryview + P-6 + split timers (all low-risk, measurable).
3. **Gap-classification hardening**: I-4/I-5/I-6 (corroboration separation, expiring reasons,
   shorter retry for recent tails) + tests.
4. **Coverage-loss visibility**: I-2 warning + skip dead fill; I-3 document the synthetic-candle
   model (maintainer decision: keep tradable) + surface per-coin synthetic share.
5. **Cold-build parallelism**: P-3/P-4/P-7 together (they interlock), fixing I-8 in passing.
6. **Cleanups**: delete dead `fill_gaps_in_ohlcvs`/`attempt_gap_fix_ohlcvs`, epoch-ms sanity
   assert in `ensure_millis_df`, cache-hash coverage for gap-fill flags (materializer finding 5).

Every perf slice must preserve outputs exactly (same bytes, same hashes) per the measurement
protocol in `backtester_performance_optimization_plan.md`.
