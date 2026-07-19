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
   gaps continue to use the normal synthetic gap path with replacement/invalidation tracking.
7. Binance monthly and daily archive requests are parallel within each tier, verify the published
   SHA-256 sidecar before parsing, and write only invalid v2 rows. Monthly archives are attempted
   only after Binance's first-Monday publication window plus a buffer; daily archives exclude the
   current day and two preceding complete UTC days.
8. Live forager planning is cache-only for inactive candidates. Its background refresher must warm
   every consumed native candle surface, including 1m inputs and native 1h log-range inputs with a
   nonzero strategy weight, using the same per-symbol requirements and explicit warmup cap as the
   live EMA bundle.
   Tail-only gaps remain eligible within the configured candidate staleness window; missing basis
   and internal gaps do not. Refresh budgets count symbol/timeframe fetches, health scans are
   bounded and rotated across cycles, interleave each candidate's 1m and native 1h health surfaces,
   keep discovered-but-unfetched stale surfaces pending, charge tokens only for selected fetches,
   and prioritize never-attempted 1m fetches before native 1h backfills. Staleness targets count
   only surfaces handled by this background
   refresher, excluding urgent active symbols. A native 1h range with a fresh tail and only an
   unavailable leading prefix remains nontradable and is retried at most once per 24 hours after a
   successful nonempty fetch which still proves the same requested leading-prefix gap; changed
   requirements, empty results, partial pagination failures, and other failed fetches remain
   eligible for normal retry. A zero OHLCV network budget disables candidate fetches even when
   entry slots are open.
   A forced native higher-timeframe refresh bypasses in-memory range and complete-disk
   short-circuits so a partial cached range cannot consume budget without retrying the exchange.
   Fresh remote rows overwrite matching disk rows, but partial remote results retain any existing
   disk coverage without entering the reusable range or EMA caches. Affected higher-timeframe EMA
   cache entries are invalidated, and higher-timeframe EMAs require full requested coverage.

Cache paths use `to_standard_exchange_name()` rather than raw CCXT identifiers such as
`binanceusdm` or `kucoinfutures`.

## Failure Semantics And Risks

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Stale known-gap metadata should guide retries but must expire; the current default retry horizon is 7 days.
5. Forager ranking drift if projected open-tail EMA values are accidentally cached or reused after
   late real candles arrive.
6. Forager ranking bias if unknown stale candidate tails are converted into zero quote-volume or
   zero log-range instead of carrying forward the latest known ranking EMA within policy.

## Validation

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.
4. Backtest/live parity for live tail-gap EMA projection behavior.
5. Binance archive threshold, publication-lag, checksum, source-order, non-overwrite, and CCXT
   fallback behavior, including public unauthenticated download smokes.

## Key Code

- `src/candlestick_manager.py`
- `src/binance_ohlcv_archive.py`
- `src/hlcv_preparation.py`
- `src/tools/verify_hlcvs_data.py`
- `exchange_integrations.md`
