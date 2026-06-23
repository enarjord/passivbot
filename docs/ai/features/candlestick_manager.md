# Candlestick Manager

## Contract

1. Prefer existing local data before remote calls.
2. For backtest preparation, use v2 OHLCV chunks first, legacy raw shards second, and targeted remote fetches last.
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

## Failure Modes To Watch

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Stale known-gap metadata should guide retries but must expire; the current default retry horizon is 7 days.
5. Forager ranking drift if projected open-tail EMA values are accidentally cached or reused after
   late real candles arrive.
6. Forager ranking bias if unknown stale candidate tails are converted into zero quote-volume or
   zero log-range instead of carrying forward the latest known ranking EMA within policy.

## Test Focus

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.
4. Backtest/live parity for live tail-gap EMA projection behavior.

## Key Code

- `src/candlestick_manager.py`
- `src/tools/verify_hlcvs_data.py`
- `docs/ai/exchange_api_quirks.md`
