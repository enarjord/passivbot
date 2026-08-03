# Candlestick Manager

## Contract

1. Prefer existing local data before remote calls.
2. For backtest preparation, use v2 OHLCV chunks first, legacy raw shards second, and targeted remote fetches last.
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
   This exception is selected explicitly by the consumer: cache-only candidate close EMAs and
   completed-candle forager ranking metrics remain strict, use policy-separated cache entries, and
   cannot reuse a provisional active-strategy result. Synthetic replacement tracking follows the
   manager's live/replay clock so delayed authoritative rows invalidate provisional replay EMAs
   deterministically.
   The live orchestrator likewise requests forager quote-volume and log-range through the strict
   policy even for active symbols. A coincident strategy log-range span may use provisional
   continuity without leaking that value into the separate `forager_m1` ranking bundle.
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
   EMA state. WebSocket shard persistence must be read-verified before the row is exposed to cache
   and EMA readers, including where an immutable legacy shard shadows primary storage. REST remains
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
   cache entries are invalidated, and higher-timeframe EMAs require full requested coverage.

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
   and carried separately from strategy EMA maps. Dynamic-management eligibility is retained in
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
10. Native higher-timeframe EMA windows require full requested coverage on every exchange. WEEX
    additionally requires exact aligned coverage for 1m EMA windows because its recent endpoint
    silently tail-anchors responses. Exchange-independent trailing-extrema and HSL replay-cache
    extension consumers also require exact aligned coverage; incomplete windows become unavailable
    or fall back to authoritative replay.
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
    `live.max_active_candle_tail_gap_minutes`; cache-only forager ranking carry-forward remains
    unavailable across an unresolved internal gap. Complete rows in the supplied EMA window remain
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
    payloads do not prove the gap and start a separate seven-day contextual-proof cooldown. Ordinary
    missing-range retries retain their existing independent schedule.
15. Urgent active-candle refresh records and reports incomplete symbol coverage but does not itself
    gate the whole planner cycle. Canonical EMA consumers determine symbol/order-class readiness;
    unavailable values remain absent, never neutralized. Account surfaces and fresh market
    snapshots retain their separate execution barriers.

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
