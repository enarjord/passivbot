# Candlestick Manager

## Contract

1. Return contiguous minute-series needed for indicator calculations.
2. Prefer cached data, then fetch missing ranges.
3. Synthesize zero-candles for verified no-trade gaps where required.

## Non-Obvious Details

1. Runtime synthetic candles are not always persisted to disk shards.
2. Real candles replacing synthetic candles must trigger EMA cache invalidation.
3. Gap semantics differ within-page vs between-page boundaries.
4. Known staged-live parity risk: open-ended live tail gaps currently use bounded carry-forward of
   the latest real candle/EMA state. Backtests have future knowledge and can replay bounded
   no-trade gaps as synthetic zero-candles, so live EMAs for close, log-range, and quote-volume
   may diverge while the latest expected candle is missing. This is a deliberate safety-first
   interim policy, not a parity-complete solution.
5. Preferred future fix for the parity risk is provisional in-memory EMA tail projection: do not
   persist open-tail synthetic candles, but compute temporary EMA values as if missing tail minutes
   were no-trade candles, bounded by the active tail-gap threshold and replaced when real candles
   arrive.

## Failure Modes To Watch

1. Cache path mismatch by exchange naming.
2. Pagination edge behavior causing boundary gaps.
3. Persistent lock or stale data artifacts.
4. Forager ranking drift when live carries forward stale log-range/quote-volume EMAs through an
   open-ended tail gap that the backtest would decay with zero-volume/no-range synthetic candles.

## Test Focus

1. Gap fill behavior and continuity.
2. Replacement/invalidation behavior when real data arrives.
3. Pagination boundary correctness per exchange.
4. Backtest/live parity for live tail-gap EMA fallback or projection behavior.

## Key Code

- `src/candlestick_manager.py`
- `src/tools/verify_hlcvs_data.py`
- `docs/ai/exchange_api_quirks.md`
