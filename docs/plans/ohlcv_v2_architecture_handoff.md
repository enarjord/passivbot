# OHLCV V2 Architecture Handoff

## Purpose

Redesign Passivbot's OHLCV download, persistence, gap tracking, and backtest payload assembly so
that:

- remote fetching is smarter and faster
- real exchange-side gaps are tracked explicitly and not retried forever
- backtests and optimizer runs reuse canonical local data instead of rebuilding exact-query caches
- optimizer workers continue to share one read-only dataset without per-worker copies
- local assembly into the Rust-ready payload is at least as fast as, and preferably faster than,
  the current `caches/hlcvs_data/` path when no remote fetch is needed

This handoff is written for a fresh agent with no prior thread context.

## Current State

### What exists today

1. Persistent symbol/timeframe OHLCV storage and fetch orchestration:
   - [candlestick_manager.py](/Users/eiriknarjord/repos/passivbot-3/src/candlestick_manager.py)
   - handles:
     - disk shard storage
     - remote pagination
     - retry/backoff
     - known-gap metadata
     - live OHLCV access

2. Backtest HLCV preparation:
   - [hlcv_preparation.py](/Users/eiriknarjord/repos/passivbot-3/src/hlcv_preparation.py)
   - computes warmup-adjusted ranges
   - fetches per coin
   - unifies candles into dense arrays for Rust

3. Exact-slice backtest dataset cache:
   - [backtest.py](/Users/eiriknarjord/repos/passivbot-3/src/backtest.py)
   - persists one cache directory per coin set / date range / config hash under `caches/hlcvs_data/`
   - loads quickly once built
   - duplicates data already present in the canonical OHLCV cache

### Practical problems today

1. There are effectively two cache layers:
   - persistent OHLCV shards
   - persistent exact-query backtest datasets

2. The exact-query cache is wasteful.
   - If one run requests a superset of coins and dates, a later subset request still builds a new
     cache even though the required data already exists.

3. Assembly into the Rust payload is too slow when reading from the canonical OHLCV cache.
   - Even when nothing needs to be fetched remotely, `hlcv_preparation.py` still spends too much
     time re-reading, transforming, and packing data.

4. Gap semantics are not formalized enough at the storage/catalog level.
   - The downloader can still spend effort retrying missing intervals that are real exchange-side
     gaps.

5. The current backtest cache shape is good for load speed but bad for reuse granularity.
   - `caches/hlcvs_data/` is fast to load
   - but too specific to exact query slices

## What The User Wants

These points were explicitly requested and should be treated as hard design constraints:

1. One canonical OHLCV cache root under `caches/ohlcvs/`.
2. Backtest range planning must consider:
   - requested date range
   - EMA warmup
   - per-coin first available timestamp
   - minimum coin age
3. Missing data should be fetched concurrently with:
   - exchange-aware rate limits
   - backoff
   - retry policies
4. Real source-side gaps must be remembered so the system does not retry them forever.
5. Backtester should use `1m` candles only and derive higher timeframes by resampling.
6. Live bot must still support direct remote `1h` fetching/storage for its own startup/runtime use
   cases.
7. Optimizer parallel backtesting must remain memory efficient.
   - all OHLCV data loaded once
   - child workers attach to shared read-only data
   - no per-worker duplication explosions
8. Local assembly from disk into a Rust-ready payload must be at least as fast as the current
   `caches/hlcvs_data/` approach when remote fetching is not needed.

## Non-Goals

1. Do not rewrite Rust backtest behavior as part of this storage redesign.
2. Do not move exchange OHLCV fetching into Rust in phase 1.
3. Do not force live bot to derive all `1h` data from `1m`.
4. Do not introduce a slower, query-oriented storage format if it degrades backtest startup speed.
5. Do not preserve the exact current `caches/hlcvs_data/` persistence model as the primary long-term
   solution.

## Core Design Decision

Separate the system into two layers:

1. Canonical persistent OHLCV store
2. Ephemeral shared backtest materialization

These must not be the same thing.

### 1. Canonical persistent OHLCV store

This is the durable local truth for:

- downloaded candles
- timeframe-specific storage (`1m`, and optionally direct `1h` for live)
- known gaps and fetch provenance
- first/last available timestamps

This store is optimized for:

- incremental updates
- cheap range slicing
- mmap-friendly local reads

### 2. Ephemeral shared backtest materialization

This is a dense, Rust-ready payload created once per run or optimizer session.

It is optimized for:

- very fast Rust input loading
- read-only sharing across worker processes
- zero per-worker copies

This replaces the performance purpose of `caches/hlcvs_data/` without keeping exact-query
persistent caches as the main architecture.

## Recommended Storage Format

### Persistent metadata catalog

Use SQLite:

- path: `caches/ohlcvs/catalog.sqlite`
- stores chunk metadata, gap state, symbol bounds, and fetch history

Reason:

- small mutable metadata fits SQLite well
- atomic updates and indexing are easy
- query needs are relational, not large-columnar

### Persistent candle bodies

Use fixed-width, mmap-friendly monthly chunk files, not Parquet, for the backtest hot path.

Reason:

- the user explicitly requires local assembly speed comparable to current `hlcvs_data`
- Parquet is good analytically, but likely slower for Passivbot's repeated dense-array assembly path
- fixed-offset month files make slice lookup and copying cheap

Recommended backtest canonical body format:

- one file per `exchange / timeframe / symbol / year / month`
- `float32[rows, 4]` body storing:
  - `high`
  - `low`
  - `close`
  - `volume`
- separate validity mask file
- timestamps are implicit from month start and row offset

### Ephemeral shared materialization

Use `np.memmap` files for the final dense backtest payload:

- easy to create in the parent process
- read-only attach in worker processes
- explicit and cross-platform
- good match for the current Rust input contract

## On-Disk Layout

### Canonical store

```text
caches/ohlcvs/
  catalog.sqlite
  data/
    {exchange}/
      1m/
        {symbol}/
          2026/
            04.npy
            04.valid.npy
      1h/
        {symbol}/
          2026/
            04.npy
            04.valid.npy
```

### Ephemeral materialized payload

```text
caches/ohlcvs/materialized/
  {run_id}/
    hlcvs.dat
    timestamps.dat
    btc_usd_prices.dat
    mss.json
    meta.json
```

The `materialized/` area is runtime-oriented and may be cleaned up aggressively.

## Chunk File Contract

### Backtest canonical `1m` chunk

Body file:

- path: `.../{YYYY}/{MM}.npy`
- dtype: `float32`
- shape: `[minutes_in_month, 4]`
- columns in fixed order:
  - `high`
  - `low`
  - `close`
  - `volume`

Validity file:

- path: `.../{YYYY}/{MM}.valid.npy`
- dtype: `bool`
- shape: `[minutes_in_month]`
- `true` means candle data is present locally for that minute

Optional sidecar metadata/debug file:

- if used at all, it must not be authoritative
- SQLite remains the metadata source of truth
- sidecars are for integrity/debugging only, not required state

### Why fixed-size month grids

This is a deliberate performance choice.

It allows:

- direct row offset calculation:
  - `offset = (ts - month_start_ts) // interval_ms`
- no timestamp arrays in each chunk
- cheap mmap slicing
- cheap in-place updates for the current open month

## Handling Incomplete Months

Incomplete months are normal and should not require special-case storage formats.

### Cases

1. Coin starts mid-month
- rows before inception are invalid in the validity mask

2. Requested backtest ends mid-month
- the materializer slices only the requested range

3. Current calendar month is still ongoing
- the chunk remains `open`
- future rows are invalid
- newly available candles patch the existing month file in place

### Month lifecycle

1. `open`
- current month
- writable
- new candles can be appended/patched
- retryable gaps may still be filled

2. `sealed`
- historical month
- normally immutable
- only rewritten by explicit repair or force-refetch flows

The implementation may keep the previous month open briefly after rollover, but the default intent
should be:

- current month: open
- prior months: sealed unless explicit repair is requested

## Gap Semantics

The system must distinguish these cases explicitly:

1. Not yet in existence
- future rows in the current month
- pre-listing rows before symbol inception

2. Missing locally and retryable
- local fetch failure
- not yet attempted
- transient source issue

3. Verified source-side absence
- exchange outage
- market closed
- no archive available
- no candles exist on exchange for that interval

The validity mask only tells whether candle data exists locally.
The reason/provenance belongs in the SQLite catalog.

### Gap policy

Retryable gaps:

- can be retried by normal refresh flows
- use bounded retry counts and backoff

Persistent verified gaps:

- must not be retried indefinitely
- only retried if user explicitly requests force-refetch

This extends the existing good ideas already present in
[candlestick_manager.py](/Users/eiriknarjord/repos/passivbot-3/src/candlestick_manager.py#L2461)
into a formal storage/catalog contract.

## SQLite Catalog Contract

Recommended tables:

### `symbols`

Tracks high-level availability bounds.

Fields:

- `exchange`
- `timeframe`
- `symbol`
- `first_ts`
- `last_ts`
- `updated_at`

### `chunks`

Tracks physical chunk files.

Fields:

- `exchange`
- `timeframe`
- `symbol`
- `year`
- `month`
- `body_path`
- `valid_path`
- `start_ts`
- `end_ts`
- `rows`
- `status`
- `schema_version`
- `checksum`
- `updated_at`

### `gaps`

Tracks explicit missing intervals and retry policy.

Fields:

- `exchange`
- `timeframe`
- `symbol`
- `start_ts`
- `end_ts`
- `reason`
- `persistent`
- `retry_count`
- `last_attempt_at`
- `next_retry_at`
- `note`

### `fetch_log`

Tracks remote-fetch history for observability and debugging.

Fields:

- `exchange`
- `timeframe`
- `symbol`
- `start_ts`
- `end_ts`
- `attempt`
- `outcome`
- `latency_ms`
- `created_at`

## Functional Components

## `OhlcvCatalog`

Responsibilities:

- gap lookup
- symbol bounds lookup
- chunk registration
- gap persistence
- fetch attempt recording

Suggested API sketch:

```python
class OhlcvCatalog:
    def get_symbol_bounds(self, exchange: str, timeframe: str, symbol: str) -> tuple[int | None, int | None]: ...
    def get_gaps(self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int): ...
    def register_chunk(...): ...
    def mark_gap(...): ...
    def clear_retryable_gap(...): ...
```

## `OhlcvStore`

Responsibilities:

- open month chunks
- create missing chunks
- patch body/valid arrays
- slice month ranges efficiently
- manage open vs sealed chunks

Suggested API sketch:

```python
class OhlcvStore:
    def open_month(self, exchange: str, timeframe: str, symbol: str, year: int, month: int): ...
    def read_range(self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int): ...
    def write_rows(self, exchange: str, timeframe: str, symbol: str, rows): ...
    def seal_month(self, exchange: str, timeframe: str, symbol: str, year: int, month: int): ...
```

## `OhlcvPlanner`

Responsibilities:

- compute effective required ranges per coin
- account for:
  - requested start/end
  - EMA warmup
  - minimum coin age
  - per-coin first available timestamp
  - local chunk availability derived from `chunks` plus chunk validity masks
  - persistent gaps
- emit fetch plans and materialization plans

Suggested outputs:

- `BacktestPlan`
- `FetchPlan`

## `OhlcvFetcher`

Responsibilities:

- fulfill uncovered intervals only
- use async concurrency
- enforce exchange-aware rate limiting
- preserve the good retry/backoff behavior of the current
  [candlestick_manager.py](/Users/eiriknarjord/repos/passivbot-3/src/candlestick_manager.py#L3010)
- write canonical chunks and catalog updates directly

This should remain Python orchestration.

Reason:

- remote OHLCV fetching is I/O-bound
- current Python async model is already a good fit
- native-code speedups matter much less here than in local hot-path assembly

## `BacktestDatasetMaterializer`

Responsibilities:

- read canonical `1m` chunks for all required coins
- build:
  - `timestamps`
  - dense `hlcvs[time, coin, field]`
  - `btc_usd_prices`
  - market-specific settings payload
- write them once to read-only memmap-backed files
- expose a small handle that workers can attach to

This is the most performance-critical piece of the redesign.

## Shared-Memory / Shared-File Contract For Optimizer

This requirement is hard.

Optimizer backtesting must not duplicate the OHLCV payload per worker.

### Recommended contract

Parent process:

1. compute backtest plan
2. fulfill missing local data if needed
3. materialize one dense dataset into memmap files
4. pass only a lightweight dataset handle to child workers

Child workers:

1. reopen memmap files read-only
2. do not heap-copy OHLCV payloads
3. keep only small config/runtime state locally

### `SharedDatasetHandle`

Suggested fields:

```python
@dataclass
class SharedDatasetHandle:
    hlcvs_path: str
    timestamps_path: str
    btc_usd_prices_path: str
    hlcvs_shape: tuple[int, int, int]
    hlcvs_dtype: str
    timestamps_shape: tuple[int]
    timestamps_dtype: str
    btc_shape: tuple[int]
    btc_dtype: str
    coins: list[str]
    mss_path: str
    exchange: str
```

This must become the optimizer/runtime contract, not incidental process-memory inheritance.

## Performance Requirements

These are explicit acceptance criteria, not aspirations.

### Local no-fetch path

When all required OHLCV data is already present on disk:

1. v2 must assemble the Rust-ready payload at least as fast as the current
   `caches/hlcvs_data/` load path for comparable datasets
2. v2 should preferably be faster
3. v2 must not regress optimizer startup due to per-worker copies

This requirement is the main reason the canonical backtest storage should use mmap-friendly fixed
width chunk files instead of a slower decode-heavy format.

### Remote fetch path

When missing data must be fetched:

1. fetching should remain bounded by exchange safety limits
2. retry logic must remain exchange-aware
3. real exchange-side gaps must be recorded and stopped from infinite retry loops
4. local writes must be incremental rather than rebuilding entire datasets

## Assembly Algorithm

The materializer should do the minimum work necessary.

### Target payload

Produce the same logical payload the Rust backtester already expects:

- `timestamps`
- `hlcvs[time, coin, field]`
- `btc_usd_prices`
- `mss`

### Recommended assembly sequence

1. Compute global time grid once.
2. Create output memmaps at final target shapes.
3. Initialize `hlcvs` with `NaN`.
4. For each coin:
   - compute the required month files
   - mmap the body and validity arrays
   - compute source offsets and destination offsets
   - copy contiguous valid blocks directly into final output
   - compute `first_valid_index`, `last_valid_index`, and warmup-derived trade start index
5. Build `timestamps` once as a direct arithmetic range.
6. Build or align `btc_usd_prices` once using the same grid.

### Important implementation detail

Avoid pandas in the hot path.

The current slow path pays too much overhead for:

- DataFrame construction
- conversions
- temporary `.npy` files
- repeated high-level transformations

The v2 hot path should be mostly:

- mmap open
- offset math
- NumPy copy into final memmap

## Backtest vs Live Timeframe Policy

### Backtest

- canonical historical timeframe: `1m`
- `1m` is the authoritative historical truth for backtesting
- all higher timeframes are derived from canonical `1m`
- backtest has no requirement to fetch or trust remote `1h`

### Live

- direct `1h` remote fetch/storage remains supported
- needed because:
  - some exchanges have shallow `1m` history access
  - live startup should not need deep `1m` reconstruction for every symbol
  - some live indicators use `1h`

### Authority rule

- this is one storage system with two policies
- `1m` is canonical historical truth
- direct `1h` is a live-only optimization cache
- direct `1h` is not authoritative against `1m`
- direct `1h` may be rebuilt or discarded at any time without affecting backtest/history correctness

## Migration Plan

### Phase 1: Prove the hot path with the smallest viable slice

1. Add:
   - `ohlcv_store.py`
   - `backtest_dataset_materializer.py`
   - minimal SQLite catalog support inside one small metadata module
2. Start writing fetched OHLCV data into `caches/ohlcvs/`
3. Build a no-fetch local materialization path from canonical chunks
4. Keep current `caches/hlcvs_data/` backtest path unchanged as the comparison baseline

Goal:

- prove the fast local assembly path and shared-memory contract before introducing broader
  abstractions

### Phase 2: Add the minimal fetch/gap planning needed for correctness

1. Add minimal planning around:
   - effective date range
   - warmup
   - min coin age
   - symbol inception clipping
2. Add gap persistence in SQLite
3. Add incremental fetch/patch flows for open months

Goal:

- validate correctness for partial local coverage and real source-side gaps

### Phase 3: Switch optimizer/backtest to shared materialized payloads

1. Parent process creates one shared dataset handle per exchange/universe
2. Worker processes reopen memmaps read-only
3. Remove reliance on per-query persistent exact-slice caches for normal operation

Goal:

- preserve and harden shared-memory optimizer efficiency

### Phase 4: Decommission legacy exact-slice persistence

1. stop writing new `caches/hlcvs_data/` datasets as the primary path
2. keep optional compatibility tools for migration/verification
3. update verification tooling to inspect `caches/ohlcvs/` and materialized payloads

Goal:

- one canonical persistent store
- one ephemeral shared materialization layer

## Testing Plan

### Unit tests

1. Month chunk offset math
2. Partial month handling
3. Open-month patching when new data arrives
4. Gap-state transitions:
   - retryable
   - persistent verified
   - force-refetch override
5. Range planning computation:
   - requested range
   - warmup
   - min coin age
   - per-coin inception clipping

### Integration tests

1. No-fetch local assembly from canonical chunks into Rust-ready payload
2. Optimizer worker attach to shared memmap payload without extra copies
3. Mixed coin universe with partial local chunks and real gaps
4. Live direct `1h` fetch/storage remains functional

### Performance tests

These are mandatory for accepting the redesign.

1. Compare current `hlcvs_data` load time vs v2 no-fetch assembly time
2. Compare optimizer resident memory before/after for equivalent workloads
3. Compare remote-fill throughput for a representative multi-coin backtest fetch

## Docs To Update When Implemented

1. `CHANGELOG.md`
2. user-facing backtest/data-cache documentation
3. any tool docs that currently point at `caches/hlcvs_data/`
4. developer docs for the new `caches/ohlcvs/` layout and gap semantics

## Recommended First Slice

Implement the minimum vertical slice that proves the hardest requirement first:

1. new month-chunk store for backtest `1m`
2. SQLite catalog with:
   - symbol bounds
   - chunk registry
   - gaps
3. materializer that builds a shared memmap Rust payload from local chunks
4. benchmark against current `hlcvs_data` local load path

Do **not** start with:

- a broad planner/fetcher abstraction split
- a full downloader rewrite
- Parquet adoption
- broad live-path rewrites

The redesign succeeds or fails on the fast local assembly path. Prove that first, then widen the
scope.
