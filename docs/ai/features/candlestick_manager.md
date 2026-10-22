# Candlestick Manager

## Contract

1. Prefer existing local data before remote calls. Backtest/optimization `backtest.offline`
   forbids every remote preparation path, including metadata and listing refreshes. Accept
   valid stale metadata, preserve confirmed exchange-side coverage boundaries, and fail on
   missing required local inputs. The task-local policy must never enable offline behavior
   in live callers. Verified prepared datasets retain input provenance.
2. Backtest preparation reads the current OHLCV store first, then uses targeted remote fetches
   when offline policy permits them. Importing legacy historical data is an explicit offline
   `import-historical-data` command with separate dry-run and execute modes; startup does not
   convert or silently consume a legacy store. Imports preserve sources and already-valid target
   rows. Dry runs leave the target tree unchanged, including locks and SQLite sidecars. Validate
   sources and all overlapping destination chunks before the first row write, then revalidate each
   chunk under its write lock; this is not a transaction across months. Reject historical inputs
   whose volume units are ambiguous. Current-store integrity and recovery checks remain active.
   For missing Binance futures 1m data in the current v2 path, remote source priority is Binance
   Vision monthly archives, Binance Vision daily archives, then CCXT. More than seven days of
   missing bars in an eligible closed month selects the monthly archive. Eligible days with more
   than 1,000 missing bars select daily archives. Small gaps, recent days, archive failures, and
   any gaps remaining inside a successful archive are repaired through CCXT.
3. Treat exchange-side late starts and early ends as coverage metadata, not local corruption.
4. Fill only internal gaps within the configured tolerance. Larger internal gaps must be repaired,
   excluded from the returned tradable window, or fail; do not make them tradable via synthetic rows.
5. Synthesize zero-candles only for verified gaps where the downstream consumer requires a dense array.
6. Remote-fetch diagnostics retain only bounded exception type, URL hash, parameter keys,
   operation/stage, symbol/timeframe, attempt/status/timing, and correlation. Manager callbacks,
   HLCV progress logs, archive fetch/day logs, structured events, and fake-live traces do not retain
   exception text or repr, raw request URLs, or request-parameter values. This diagnostic boundary
   must not alter retry, rate-limit classification, exception propagation, or cache behavior.
7. Local storage diagnostics for migration, cleanup, locks, indexes, shards, disk/cache health,
   inception metadata, and deferred index writes retain bounded exception type instead of exception
   text, repr, or exception-value traceback. Redaction must not alter cache contents, migration and
   cleanup behavior, lock handling, retries, fallbacks, or exception propagation.
   The periodic disk-coverage audit stamps its check timestamp before auditing; an audit failure is
   type-only and does not prevent the regular market refresh or normal cycle sleep.
8. Direct live lifecycle diagnostics for completed-close fallback, startup/index/background
   warmup, forager refresh, active refresh, and refresh-cap handling retain bounded exception type
   instead of exception text or exception-value traceback. Redaction must not alter warmup,
   cancellation, lock retry, refresh scheduling, fallback values, readiness, or trading behavior.
9. Direct live health and trailing diagnostics retain bounded exception type instead of exception
   text in health-window construction, health summaries, trailing fetch failures, freshness
   readiness, and tail-gap projection logging. Redaction must not alter fallback values,
   symbol/position-side availability, trailing deferral, readiness results, or trading behavior.

## Non-Obvious Details

Span-based warmup acquisition rounds positive `bot.portfolio.warmup.max_minutes` caps upward to
whole minutes in live and backtest paths; `0.5` means one minute, never uncapped. Null remains
uncapped. Round only the acquisition window, preserving fractional EMA spans and naturally shorter
requirements. Native hourly acquisition retains whole-hour alignment, and independent RMS history
requirements are unchanged. Positive integer and fractional caps are ceilings rather than minimum
requests; only uncapped background candidate refresh retains its default minimum window. Direct
live ranking readers pass floating EMA spans through to the candle manager without rounding them.

Ordinary native higher-timeframe reads reuse verified persisted rows and fetch only missing spans,
with one source bucket of overlap on each side. Overlapping requests are coalesced; the overlap
retains within-response evidence for KuCoin's no-tick gaps. Pages persisted before a deadline or
cancellation remain useful to the next acquisition, including after manager restart.
Pagination stops when a nonempty response tail repeats or regresses. Best-effort reads
retain all successful pages, including corrections in the stalled page. Accumulated output
is sorted and unique by timestamp, with later pages winning corrections. Forced strict
reads report the incomplete acquisition through the
existing terminal-page failure contract. A moving request cursor alone is not progress. Forced
`max_age_ms=0` reads still refetch the full requested range. Missing buckets remain unavailable;
this acquisition optimization does not relax full-coverage EMA readiness or fabricate candles.

1. Runtime synthetic candles are not always persisted to disk shards.
2. Real candles replacing synthetic candles must trigger EMA cache invalidation.
3. Gap semantics differ within-page vs between-page boundaries.
4. Staged-live open-ended tail gaps use bounded, provisional in-memory EMA projection for close,
   log-range, and quote-volume inputs. Projection computes temporary values as if missing tail
   minutes were no-trade candles, but does not persist open-tail synthetic candles or normal EMA
   cache entries.
5. That projection contract is for paths where no-trade tail projection is explicitly allowed. Live
   forager candidate ranking has a narrower stale-tail contract: close EMA readiness may use
   bounded flat-close projection, but quote-volume and log-range ranking inputs should carry
   forward the latest known EMA value with age/source metadata instead of appending unknown
   zero-volume or zero-range tail minutes. Zero volume/log-range is valid for verified no-trade
   continuity gaps, not for unknown stale tails caused by refresh budget or REST delay.
6. Projection is stateless per read. Real candles always win on the next read, and bounded internal
   gaps use the non-persistent synthetic gap path with replacement/invalidation tracking for EMA
   reads while remaining unavailable to ordinary candle consumers. Unresolved gap metadata wholly
   inside the configured open-tail projection interval does not invalidate that projection.
   Internal eligibility is measured from the complete still-uncovered contiguous span, including
   portions outside the current EMA query. Authoritative or verified-zero rows inside older broad
   retry metadata split the original outage into its remaining uncovered spans; stale metadata
   bounds alone must not make a short residual gap unavailable.
   The open-tail age bound applies uniformly, including stock perps; venue category must not create
   an unbounded synthetic tail exception.
   Trailing-extrema reconstruction may append the same temporary flat rows only after proving dense
   coverage from the first whole minute following the latest fill through the last cached candle.
   A missing reset boundary, internal minute, or tail beyond the bound remains unavailable. The
   temporary rows never enter candle shards or manager caches, so a delayed real high or low
   replaces the projection on the next cycle. Each trailing consumer records structured projection
   context by symbol and position side, including the authoritative and projected bounds, projected
   candle count, and consecutive-use count; authoritative recovery clears that runtime context.
7. Binance monthly and daily archive requests are parallel within each tier, verify the published
   SHA-256 sidecar before parsing, and write only invalid v2 rows. Monthly archives are attempted
   only after Binance's first-Monday publication window plus a buffer; daily archives exclude the
   current day and two preceding complete UTC days.
8. Live forager planning is cache-only for inactive candidates. Its background refresher must warm
   every consumed native candle surface, including 1m inputs and native 1h log-range inputs with a
   nonzero strategy weight, using the same per-symbol requirements and explicit warmup cap as the
   live EMA bundle.
   Cache-only EMA carry-forward must read the index and candle range for the requested native
   timeframe. When the newest native 1h bucket has only just become due, the previous cached 1h
   EMA may remain eligible only for the candidate staleness window computed from the same
   refreshable surface count used by the background refresher and measured on the candle manager's
   active live/replay clock; the one-hour bucket interval itself is not counted as one hour of
   refresh lateness. Cached carry-forward requires full requested-window coverage and must not
   populate the normal active-strategy EMA cache.
   Tail-only gaps remain eligible within the configured candidate staleness window; missing basis
   and internal gaps do not. A known-gap retry cooldown suppresses only the recorded missing
   prefix; newly finalized candles after that gap and unrelated internal gaps remain fetchable,
   while every fetch path excludes the deferred prefix. Unknown `auto_detected` and
   `fetch_failed` rows remain absent until authoritative candles replace them, including when a
   retry is due or the current request disallows remote fetching, and must not become synthetic
   zero-volume continuity candles. Day-coalesced historical fetches split around deferred ranges
   rather than contacting the venue for them. Forced 1m and native higher-timeframe candidate
   refreshes use best-effort stale reads so a successful partial sparse response can continue into
   gap repair. A terminal empty page, or a successful refresh which leaves an empty or unchanged
   open tail, applies a bounded per-surface retry delay; these expected background outcomes are
   DEBUG diagnostics, while required active reads and unexpected refresh failures remain loud. An
   overlap page that already covers the requested end is complete, not terminal-empty. Partial authoritative
   recovery stamps the unresolved remainder with a new retry time, and deferred exclusions remain
   compact timestamp intervals even for large historical gaps.
   Live EMA reads are the deliberate exception for gaps already bounded by later authoritative
   candles: they may use non-persistent zero-volume continuity rows so sparse no-trade intervals
   do not permanently block strategy inputs. The timestamps remain unresolved and retryable;
   delayed authoritative rows replace the provisional values and invalidate affected EMA caches.
   This exception is selected explicitly by the consumer: completed-candle forager ranking metrics
   for current, remote-enabled candidates may bridge bounded internal gaps, while cache-only
   candidate reads remain strict. This is approximate ranking continuity, not evidence that the
   missing rows were fetched. Synthetic replacement tracking follows the manager's live/replay
   clock so delayed authoritative rows invalidate provisional replay EMAs deterministically.
   Existing known-gap and refresh diagnostics expose repair state. Live orchestration derives
   ranking-input consumption from canonical gap state and emits activation/recovery transitions
   per symbol and metric without a second per-span provenance state machine or use counter.
   Open-ended tails use the separate bounded projection policy below.
   Refresh budgets count
   symbol/timeframe fetches, health scans are bounded and rotated across cycles, interleave each
   candidate's 1m and native 1h health surfaces,
   keep discovered-but-unfetched stale surfaces pending, and charge one token immediately before
   each actual fetch attempt rather than reserving tokens for a batch which may be cut short by
   the wall-time cap. The remaining wall time is divided across the remaining selected surfaces;
   a fast refresh leaves its unused share available to later work, while one slow sparse-history
   refresh cannot consume the entire cycle and starve the rest of the batch. A timed-out candidate
   surface receives a short in-memory retry delay so one blocked symbol cannot monopolize
   successive refresh cycles. Exhausting an assigned scheduler time share is an expected DEBUG
   yield; a timeout raised by the candle operation itself remains a warning. Completed-candle health reports
   whether missing minutes are currently refreshable separately
   from raw coverage. Verified internal `no_trades` continuity and unresolved known gaps whose retry
   cooldown is active do not spend background REST budget. This scheduler classification never
   turns an unresolved gap into tradable coverage: raw `coverage_ok` remains false, and a newly
   finalized suffix outside the deferred range remains refreshable. Adjacent missing ranges with
   different retry epochs remain separate, even when their reason matches, so fresh evidence never
   inherits an older range's retry cooldown. Refresh scheduling prioritizes
   never-attempted 1m fetches before native 1h backfills. Staleness targets count
   only surfaces handled by this background
   refresher, excluding urgent active symbols. A native 1h range with a fresh tail and only an
   unavailable leading prefix remains nontradable and is retried at most once per 24 hours after a
   successful nonempty fetch which still proves the same requested leading-prefix gap; changed
   requirements, empty results, partial pagination failures, and other failed fetches remain
   eligible for normal retry. Dense post-listing history which is shorter than the configured 1h
   warmup remains unavailable until enough real buckets exist; pre-listing hours are never
   synthesized. A zero OHLCV network budget disables candidate fetches even when entry slots are
   open.
   When enabled and supported by CCXT Pro, proven-final public 1m WebSocket rows for flat forager
   candidates are persisted through the same canonical candle path as REST rows. Because CCXT may
   repeat a sliding cache, the first nonempty snapshot of each watcher session only primes
   provenance. A changed row may correct its existing canonical timestamp, but extending canonical
   history requires a fresh successor timestamp proving a trusted preceding bucket closed;
   processing time is not post-boundary transport provenance. The current in-progress minute is
   rejected, an existing canonical basis is required, and WebSocket silence and reconnect gaps remain
   missing. A later changed row for the same timestamp overwrites the candle and invalidates affected
   EMA state. WebSocket canonical reads, merge-before-write reads, and persistence verification
   propagate shard read failures, including failures masked by another legacy or primary source.
   Missing shards are distinct from unreadable history. Strict canonical reads refresh source
   discovery; strict merges read the actual write target under the fetch lock rather than trusting
   a cached directory listing. WebSocket shard persistence must be read-verified before the row is
   exposed to cache and EMA readers, including where an immutable legacy shard shadows primary storage. REST remains
   the complete fallback for startup basis, historical and internal gaps,
   prolonged silence, reconnect recovery, and a configured periodic integrity audit. Audits force a
   bounded REST overlap even while the persisted WebSocket tail is current; a successful REST
   omission alone does not disprove a validated WebSocket candle. Repeated stream or ingestion errors
   enter a bounded cooldown while REST continues, then retry automatically. The subscription
   reconciler remains alive when the transport is configured but no side is yet in forager mode, so
   runtime mode transitions are handled without restart. Dynamic subscriptions include only sides
   currently using forager mode, follow their flat approved universe, and are removed when a symbol
   enters the urgent active-candle universe. Removal requests the CCXT Pro unsubscribe while the
   watcher still owns its pending `watch_ohlcv` future, lets that watcher consume the transport's
   unsubscribe wake-up, and uses cancellation only as a bounded fallback. Restart cleanup awaits
   this owner-managed teardown before closing exchange clients. An internal restart request only
   unwinds to that cleanup owner; it does not pre-cancel maintainers and then cancel them again.
   Removed watchers remain in the owner task map until retirement returns, so cancellation cannot
   detach them from outer cleanup.
   Bulk and singleton unsubscribe calls, the graceful watcher wait, and the post-cancellation wait
   all use hard deadlines which do not await cancellation-resistant connector work. An uncooperative
   unsubscribe task is cancelled, retained for exception consumption, and abandoned without
   blocking teardown. A cancellation-resistant watcher is likewise abandoned without
   blocking reconciliation and remains marked retiring until it actually terminates, preventing
   it from resuming ingestion beside its replacement.
   A forced native higher-timeframe refresh bypasses in-memory range and complete-disk
   short-circuits so a partial cached range cannot consume budget without retrying the exchange.
   Fresh remote rows overwrite matching disk rows, but partial remote results retain any existing
   disk coverage without entering the reusable range or EMA caches. Affected higher-timeframe EMA
   cache entries are invalidated by changes to processed history. Higher-timeframe EMAs require
   full coverage of their required one-span window; an older optional bootstrap prefix is separate.

   A flat symbol selected by forager remains eligible for symbol-scoped required-EMA degradation
   even while one of its entry orders is resting. The unavailable symbol is sent to Rust as
   nontradable for that planning cycle, which removes its ideal orders and lets normal
   reconciliation cancel the resting entry. An open order alone must not promote a flat,
   dynamically selected symbol into the account-fatal required-input path. This degradation is
   side-aware: every normal side must be dynamically forager-selected; fixed or explicitly normal
   sides retain their strict readiness contract. A side disabled by zero entry capacity is not an
   implicit normal side merely because the symbol is active on the opposite side. Strategy EMA
   maps and Rust's `tradable` flag are
   nevertheless symbol-scoped: once every normal side is proven dynamically managed, any missing
   required strategy EMA degrades the whole flat symbol rather than fabricating a partial bundle.
   Missing forager ranking features remain side-scoped because their affected side is authoritative
   and carried separately from strategy EMA maps. They remain conditional diagnostics, rather than
   symbol-wide EMA-unavailable state, until Rust reports that ranking was required for that side.
   Dynamic-management eligibility is retained in
   memory independently of side-scoped cancellation permission when Rust's symbol-level
   nontradable result changes both sides to the configured manual stop mode. This lets the next
   identical missing-ranking cycle remain degraded without authorizing cancellation of an
   unaffected side's resting entry; an explicit operator `manual` or ordinary `tp_only` override
   still revokes the retained eligibility. Reconciliation preserves only the affected side's
   proven forager entry cancellation, identified by exchange or client order ID. Every entry
   observed while the side is fully managed is bot-owned under the live ownership contract, even
   if the user originally submitted it. Authorization survives EMA recovery and is rederived on
   later cycles only while the same order ID remains, so a rejected or ambiguous first
   cancellation is retried. Orders which first appear after the side enters `manual` or ordinary
   `tp_only` are not authorized by matching symbol/side, price, or quantity. It does not weaken
   manual ownership for other sides, closes, or creations.
   Eligibility does not depend on prior membership in `active_symbols`: Rust evaluates the flat
   forager universe before selecting active coins. A temporary bot-managed entry override such as
   HSL `graceful_stop`, `panic`, or `tp_only_with_active_entry_cancellation` retains this
   degradation/cancellation behavior; `manual` and ordinary `tp_only` continue to protect
   operator-owned entries. Held positions
   retain their stricter readiness contract.
9. WEEX live warmups use exchange-specific hybrid pagination: bounded 100-row historical windows
   followed by the recent endpoint only when its 999 finalized-row tail covers the remainder. This
   supports deep-enough 1m and 1h live EMA, trailing, and HSL restart windows without enabling WEEX
   bulk backtest-data download.
10. Native higher-timeframe EMAs require full coverage of their required one-span window on every
    exchange; an optional older bootstrap prefix may be shorter. WEEX additionally requires exact
    aligned coverage for the required 1m EMA window because its recent endpoint
    silently tail-anchors responses. HSL restart and live trailing restart reconstruction may fetch
    1m candles first, then cover only the older leading prefix with 5m, 15m, and 1h candles. The
    finest available source wins and source counts remain visible. Normally only coarse buckets
    ending at or before the first available 1m candle are eligible. One coarse bucket may straddle
    that boundary when every overlapping exact 1m minute is already present; only synthetic rows
    before the precision boundary are eligible, while exact 1m rows retain precedence. The missing
    prefix is expanded separately, ending at the first exact candle's open clamped to the normalized
    coarse range. Full-bucket and boundary expansion share finite-value checks, reversed-extrema
    normalization and endpoint clamping. Coarse extrema outside
    the complete overlap and already-selected finer rows' ranges must belong to the missing prefix
    and are preserved in its remaining synthetic rows. Extrema already present in finer rows are
    not moved backward; whether they also occurred in the missing prefix is unknown. Coarse volume
    retains its proportional approximation. This bridges retention-boundary seams without allowing
    coarse data to conceal an internal exact-era gap.

    Trailing still requires a nonempty exact 1m suffix and its existing dense post-fill coverage
    and bounded open-tail checks. The first post-fill minute remains the exact reset boundary;
    deterministic expansion may clip the beginning of a coarse bucket there but never emits an
    earlier timestamp. Coarse OHLC preserves the old prefix's highs and lows, while the dependent
    `min_since_max` and `max_since_min` values are explicitly approximate because a higher-timeframe
    candle cannot prove whether its high or low occurred first. Failure to obtain the exact suffix
    keeps the affected position side unavailable rather than treating an entirely coarse window as
    current trailing state.
11. Quote-volume EMA is derived from normalized CCXT base volume and typical price
    (`base_volume * (high + low + close) / 3`). It is an approximation when an exchange, including
    WEEX, does not expose raw quote turnover through unified OHLCV.
12. Persisting an authoritative 1m row trims that timestamp from any stale
    `known_gaps` range, splitting partially covered ranges while preserving the
    retained retry metadata. Hyperliquid may publish authoritative flat,
    zero-volume candles after an initially sparse recent response. Recent
    bounded tail-sized Hyperliquid gaps therefore retry on a time-spaced schedule
    rather than exhausting the ordinary retry count in consecutive live cycles.
    Large missing-basis ranges retain the ordinary persistent-gap schedule. The
    known-gap retry decision runs before refresh, ordinary present, tail-completion,
    and targeted gap fetches so a deferred tail cannot consume REST calls through
    an earlier path. Forced overlap refreshes also split around deferred internal
    gaps. A failed retry of a persistent recent gap retains the persistent retry
    cadence, and all retry metadata uses the manager's active live/replay clock.
    Missing rows remain unavailable to ordinary candle consumers until an authoritative row
    arrives. Live strategy EMA reads may provisionally bridge a later-bracketed internal gap with
    non-persistent flat zero-volume rows only when the gap is no wider than
    `live.candles.max_active_candle_tail_gap_minutes`; current remote-enabled forager ranking reads use the
    same bounded internal-gap rule, while cache-only forager ranking carry-forward remains
    unavailable across an unresolved internal gap.
    Complete rows in the supplied EMA window remain
    authoritative even if stale known-gap metadata still names their timestamps. Recording or
    extending a 1m gap invalidates cached 1m EMA and open-tail projection values. An overlap refresh
    which retries a due gap
    stamps every unresolved remainder before later repair stages run, preventing a
    second attempt in the same request. Historical pagination flushes deferred
    partial-page index writes before propagating terminal-empty failure.
    Gap normalization also preserves proof-specific subranges: verified terminal
    reasons never expand into adjacent retryable minutes, and an overlap gives
    terminal evidence authority only over the timestamps it actually covers.
    Repeated terminal empty-page failures and successful refreshes which make no
    open-tail progress for a forager candle surface use a bounded in-memory retry
    delay without converting missing data into candles. Expected background
    sparse-tail outcomes are DEBUG diagnostics; unexpected failures remain loud.
    A partial authoritative response similarly defers and excludes its unresolved
    remainder until the next eligible retry.
13. Open-tail projection requests only the metrics its caller may consume.
    Forager projection requests close EMAs only; quote-volume and log-range
    ranking inputs continue to come from current or bounded cached real candles.
    Identical projections within one finalized bucket reuse a bounded in-memory
    result keyed by the cached-tail timestamp, requested spans, and projection
    horizon. The key also includes the content-bearing shard and known-gap state
    for the requested window, so another process writing the shared cache
    invalidates stale projections. Candle content, synthetic provenance, or
    known-gap coverage changes invalidate local entries; metadata-only refresh
    writes do not force a full recomputation.
    Compatible current and bounded cache-only EMA spans share one candle-window
    load per metric policy. Cache-only coverage is validated independently per
    span, preserving complete shorter spans when a longer requested window is
    incomplete. Complete windows use a vectorized continuity check instead of
    rebuilding every unchanged real row in Python.
    Latest-value EMA calculations use a scalar recurrence rather than allocating
    a full output series. Full EMA series remain available to callers that need
    every intermediate value. Live provisional internal-gap tolerance is separate from the
    simulation-only backtest gap tolerance.
14. KuCoin omits kline buckets with no ticks. For native timeframes above 1m, gaps bounded by real
    candles in the same successful payload, absent from that raw payload, and no wider than the
    fixed 120-minute live connector policy are materialized as flat zero-volume candles before
    persistence. The simulation-only `backtest.gap_tolerance_ohlcvs_minutes` setting remains owned
    by historical data preparation and does not alter live readiness. A bucket present in the raw
    payload but rejected by candle validation is a
    continuity barrier: it and later omitted buckets remain unavailable until another accepted real
    candle establishes the close. A rejected row whose timestamp cannot be identified disables
    synthesis for that payload page and evicts cached placeholders between the page's accepted
    bounds. If fewer than two accepted timestamps exist, the remaining requested range is treated as
    unavailable. Expansion is bounded to the requested range, and a later real exchange candle
    always overwrites a persisted synthetic bucket. If a later payload contains a rejected real row
    at a cached sparse-placeholder timestamp, the placeholder is evicted so the bucket becomes
    unavailable and the timeframe index bounds are recomputed from the remaining shards. Leading,
    trailing, failed-fetch, oversized, and unproven between-page gaps remain unavailable. For 1m
    gaps initially classified as `auto_detected` or `fetch_failed`, a targeted retry may expand to
    the nearest cached real candle on each side as soon as those boundaries are available; this
    proof does not wait for the ordinary retry count to become persistent. Only when one successful
    raw payload returns both boundaries while omitting the intervening timestamps may that exact
    range be promoted to verified `no_trades` continuity. Empty, one-sided, terminal, or rejected
    payloads do not prove the gap and start a separate five-minute contextual-proof cooldown. A
    contiguous missing span may be covered by multiple adjacent retry records; verification requires
    complete metadata coverage and every unverified fragment's contextual retry to be due. Verified
    no-trade fragments may share the proof window, but terminal and uncovered fragments cannot.
    Ordinary historical repair recognizes the union of deferred records without merging their
    independent retry clocks or refetching the surrounding cached history. Coverage uses one sorted
    metadata snapshot per check. Live present and historical scans index overlapping
    records without merging their retry clocks, and rebuild the read index when canonical
    metadata changes locally or through a shared-cache writer. Contextual proof is scheduled only when both real bounds and the
    overlap fit one request page; wider gaps remain unavailable under ordinary retry policy. Ordinary
    missing-range retries retain their existing independent schedule.
15. Urgent active-candle refresh records and reports incomplete symbol coverage but does not itself
    gate the whole planner cycle. Canonical EMA consumers determine symbol/order-class readiness;
    unavailable values remain absent, never neutralized. Account surfaces and fresh market
    snapshots retain their separate execution barriers.
16. A `NaN` returned by a completed-candle EMA API represents an empty or incomplete EMA window.
    The live consumer may recover it only through the explicitly bounded open-tail projection or
    carry-forward policies above; without such a recovery, it becomes symbol- or action-scoped
    unavailability. It is never passed to Rust or replaced by an unbounded or fabricated value.
    Positive or negative infinity is not an availability sentinel and remains a fatal invalid
    calculation result before any fallback.

Cache paths use `to_standard_exchange_name()` rather than raw CCXT identifiers such as
`binanceusdm` or `kucoinfutures`.

## Failure Semantics And Risks

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Stale known-gap metadata should guide retries but must expire; the current default retry horizon is 7 days.
   Recently missing bounded Hyperliquid tail rows use a shorter, time-spaced live
   retry policy because the venue may publish an authoritative no-trade row later.
5. Forager ranking drift if projected open-tail EMA values are accidentally cached or reused after
   late real candles arrive.
6. Forager ranking bias if unknown stale candidate tails are converted into zero quote-volume or
   zero log-range instead of carrying forward the latest known ranking EMA within policy.

## Validation

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.
   WEEX validation must cover both 1m and 1h ranges that cross the recent/history boundary and
   assert that every historical request spans no more than 100 aligned candles.
4. Backtest/live parity for live tail-gap EMA projection behavior.
5. Binance archive threshold, publication-lag, checksum, source-order, non-overwrite, and CCXT
   fallback behavior, including public unauthenticated download smokes.
6. Hostile remote-fetch diagnostics are redacted at the manager callback boundary and remain
   redacted after repeated sanitization by direct consumers. Concurrent archive requests preserve
   correlation through URL hashes rather than raw URLs.
7. Hostile local-storage failures retain bounded exception classification without exception values
   while preserving the original cache, migration, lock, and fallback outcomes.
8. Hostile live lifecycle failures retain bounded exception classification without exception values
   or traceback text while preserving completed-close fallback, warmup, cancellation, lock retry,
   refresh scheduling, readiness, and return behavior.
9. Hostile live health and trailing failures retain bounded exception classification without
   exception values while preserving fallback values, affected symbol/position-side unavailability,
   trailing deferral, readiness results, and unaffected consumers.

## Key Code

- `src/candlestick_manager.py`
- `src/binance_ohlcv_archive.py`
- `src/hlcv_preparation.py`
- `src/tools/verify_hlcvs_data.py`
- `exchange_integrations.md`

## Source-resolution reads for HSL

`get_candles(standardize=False)` retains normal source acquisition and caching but
returns sparse source rows without the 1m gap-standardization or outside-range
price seed. The 1m returned array is detached from the mutable cache. Native coarse
cache reads remain native even with no exchange object; they must not relabel 1m
rows as 5m/15m/1h. Existing callers retain standardization by default.

The staged `live.hsl_candles.CandleSourceReader.acquire` reader requests supported
1m/5m/15m/1h sources over the full estimator window. A real 1m close at the inclusive
left edge belongs to the window although its source bucket opened one minute earlier.
Rust rejects earlier closes and coarse buckets straddling the boundary, selects the
finest available source, and applies estimator-local gap carrying. Manager-persisted
verified no-trade observations remain usable under the existing cache contract.

Each source read has a caller-supplied deadline which does not await resistant
cancellation. Expected exchange/transport/read failures retain type-only diagnostics
and attempt a bounded cache-only read; independent resolutions survive. The caller
reuses one `CandleSourceReader` per manager across scopes and cycles. It retains
unfinished reads, refuses another read of the same symbol/timeframe/source kind,
and caps total pending reads (eight by default, configurable on construction).
A timed-out coroutine's return value is never consumed or allowed to extend its
read deadline. Cache fallback is a separate fresh observation: it may include
canonical cache updates completed meanwhile, including an update from the timed-out
fetch. Rows are copied and stamped at that actual cache capture time. There is no
shared atomic cutoff across resolutions; Rust rejects sources unavailable at the
chosen evaluation time, and the eventual runtime caller owns final snapshot
revalidation. Freezing a pre-fetch cache is not required. Pending-read count is
diagnostic; it never supplies a trading decision.

Invalid producer shapes, programming errors and cancellation propagate. Acquisition
wrappers are cancelled and awaited; underlying cancellation-resistant reads remain
tracked within the fixed capacity until completion. Late exceptions are consumed;
unexpected late programming failures are raised on the next acquisition instead of
silently hidden. No source projection is written into factual caches. The caller owns
background scheduling and coherent current-state capture; this staged reader alone
does not activate hsl trading.

HSL source batches may retain a native copy of their immutable scalar candle
rows for repeated evaluations. Replacing a source batch replaces that native copy.
The copy contains no projected window or trading permission: every projection
reapplies its exact lookback bounds and current exchange/UTC observation offset,
and every risk evaluation still consumes current account and mark facts.

### Live EMA preparation pacing

Live EMA preparation keeps serial symbol loading when candle fetch pacing is positive and parallel
loading when it is zero. Position symbols remain first in the ordered list. It does not add a sleep
after preparing each symbol, but yields to other tasks without delay. The candle manager enforces the configured process-local spacing at
actual CCXT OHLCV requests, so cache-only preparation incurs no artificial per-symbol pause.
See [EMA preparation timings](live_events.md#ema-preparation-timings) for structured attribution.

### Parsed disk-row reuse

A manager may retain immutable parsed source rows for up to 16 exact ranges and
8 MiB total. Every reuse rechecks the selected source paths and index using file
identity, size, modification time and change time, including atomic replacements
and in-place corrections whose modification time is restored. Failed/empty shard
reads and changes during parsing are not certified. For a legacy/primary overlay,
every constituent must return nonempty rows before the merged result is certified;
a surviving source cannot mask a failed or empty constituent. Strict audits always
read the source again. Native Windows bypasses parsed-row reuse because its stat
ctime currently reports creation time, which cannot detect same-size in-place
corrections with restored mtime. Ordinary disk reads remain authoritative there.
Clearing the cache or restarting reconstructs identical rows.

Only factual source rows are cached. Mutable consumer arrays and the one-minute
memory cache never alias retained immutable rows. One-minute reads still merge
against the current memory state; identical sorted overlaps may bypass the
concatenate/sort/deduplicate work while preserving detached-array ownership and
last-write-wins behavior. Projections, synthetic gaps, EMA values and trading
permissions are not part of this cache.

### Historical gap range views

JSON-loaded and newly persisted gap lists own a single lazily built interval view.
The view indexes only factual interval bounds, preserves source record order,
and invalidates on list/record mutations or shared index replacement. It lives
with its metadata cohort; no per-window views accumulate. Injected unowned lists
retain conservative full-scan behavior instead of assuming identity/length proves
unchanged content. The persisted JSON schema is unchanged.

Recent-range checks and projection signatures select overlapping records before
normalizing metadata. Legacy defaults are evaluated at read time, returned enhanced
records remain detached, and retry eligibility is always computed from current
clocks. Unknown-gap coverage retains full missing interval widths and excludes
synthetic candles when determining real coverage. A window without unknown gaps
skips copying/sorting the retained candle history. This optimization neither
deletes stale historical gaps nor changes fetch, synthesis, retry or trading policy.
Full metadata reads and index serialization still scale with total history.

Unknown-gap coverage checks build one contiguous timestamp index after copying,
sorting and excluding provisional synthetic candles. All gap slices in that
synchronous cohort share the index, avoiding NumPy's repeated materialization of
the strided structured-array timestamp field. The index is never retained across
calls: corrections, RAM eviction/replacement and synthetic-set changes remain
visible immediately. Full outage widths and candle/EMA policies are unchanged.

### Ordinary market EMA initialization and continuation

Ordinary price, quote/base volume and log-range indicators use floating sample spans. For a new
indicator, normalize exponential weights over at most `ceil(2 * span)` available completed samples:
`numerator = value + decay * numerator`, `denominator = 1 + decay * denominator`, then finalize
`ema = numerator / denominator`. Here `alpha = min(1, 2 / (span + 1))` and `decay = 1 - alpha`.
Prefer one span of history; two spans are a bootstrap cap, never a universal readiness requirement.
Existing strict one-span coverage rules and scoped missing-data policies still apply. A missing
optional older prefix must not become fabricated candles. Bootstrap uses the available contiguous
completed suffix, subject to those policies.

After bootstrap, retain only the finalized value as arithmetic state and advance each new completed
sample once with `ema = decay * previous_ema + alpha * value`. Do not continue the normalized-weight
bootstrap denominator into ordinary operation. Full-series APIs normalize their initial bounded
prefix and then use the same ordinary recurrence. Latest scalar bootstrap helpers fold their
bounded suffix without allocating an output series.

Live cache TTL controls candle refresh, not EMA age or repeated consumption. The same completed
source prefix and end timestamp reuse the finalized value. Appended completed rows advance only
the new suffix; changed/removed processed rows invalidate affected cached values, including
canonical corrections older than the retained bootstrap window. A changed span/timeframe has a
separate initializer. Quote and unknown-tail projections use a temporary copy and cannot advance
committed state. Cache loss or restart uses the available completed bootstrap window; equality to
an unknown infinite past is neither required nor claimed.

CPU backtests initialize each coin at its first usable planning boundary, after the global warmup
and requested-start guards and once the coin is tradeable. Selection and order construction run
after that initializer. Bootstrap includes completed candle `k` and its preceding bounded warmup,
never `k + 1`. Newly eligible coins can use a shorter available prefix. Subsequent candles advance
once. Native GPU replay follows the same boundary. Hourly range indicators include only completed
hour buckets, retain their hourly alpha and advance on hour boundaries rather than every minute.

For example, span `3` and completed values `[2, 4]` initialize to `10/3`; the next completed value
`8` advances to `17/3`. Continuing normalized weights would instead produce `6`, which is a distinct
formula. This policy does not change deliberate HSL zero initialization, realized-loss memory,
risk-history/refold contracts or the existing exposure RMS model.
