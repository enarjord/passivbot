# Candlestick Manager

## Contract

1. Return contiguous minute-series needed for indicator calculations.
2. Prefer cached data, then fetch missing ranges.
3. Synthesize zero-candles for verified no-trade gaps where required.

## Non-Obvious Details

1. Runtime synthetic candles are not always persisted to disk shards.
2. Real candles replacing synthetic candles must trigger EMA cache invalidation.
3. Gap semantics differ within-page vs between-page boundaries.
4. Staged-live open-ended tail gaps use bounded, provisional in-memory EMA projection for close,
   log-range, and quote-volume inputs. Projection computes temporary values as if missing tail
   minutes were no-trade candles, but does not persist open-tail synthetic candles or normal EMA
   cache entries.
5. Projection is stateless per read. Real candles always win on the next read, and bounded internal
   gaps continue to use the normal synthetic gap path with replacement/invalidation tracking.

## Failure Modes To Watch

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Forager ranking drift if projected open-tail EMA values are accidentally cached or reused after
   late real candles arrive.

## Test Focus

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.
4. Backtest/live parity for live tail-gap EMA projection behavior.

## Key Code

- `src/candlestick_manager.py`
- `src/tools/verify_hlcvs_data.py`
- `docs/ai/exchange_api_quirks.md`
