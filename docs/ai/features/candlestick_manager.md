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
