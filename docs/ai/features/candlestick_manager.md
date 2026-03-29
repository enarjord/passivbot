# Candlestick Manager

## Contract

1. Return contiguous minute-series needed for indicator calculations.
2. Prefer cached data, then fetch missing ranges.
3. Synthesize zero-candles for verified no-trade gaps where required.

## Non-Obvious Details

1. Runtime synthetic candles are not always persisted to disk shards.
2. Real candles replacing synthetic candles must trigger EMA cache invalidation.
3. Gap semantics differ within-page vs between-page boundaries.
4. Higher-timeframe fallback (`5m` / `15m` -> `1m`) should be implemented as pure shared OHLCV transforms, not embedded inside one replay or exchange-specific caller.
5. Real fetched higher-timeframe source candles may be persisted to their own timeframe caches while synthesized `1m` candles remain memory-only.

## Failure Modes To Watch

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Exchange retention caps causing missing `1m` history even when higher-timeframe history exists.

## Test Focus

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.

## Key Code

- `src/candlestick_manager.py`
- `src/tools/verify_hlcvs_data.py`
- `docs/ai/exchange_api_quirks.md`
