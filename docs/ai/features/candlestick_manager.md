# Candlestick Manager

The CandlestickManager handles OHLCV (Open-High-Low-Close-Volume) data fetching, caching, and synthetic candle generation.

## Key Concepts

### Data Flow

1. Request candles for symbol + time range
2. Check local Parquet cache
3. Fetch missing data from exchange or TradFi provider
4. Fill gaps with synthetic candles
5. Return complete, contiguous series

### Synthetic Candles

When exchange data has gaps (illiquid markets, exchange downtime), CandlestickManager synthesizes "zero-candles":
- OHLC = previous close
- Volume = 0

This ensures EMAs can be computed over contiguous data.

Runtime behavior:
- For live/present ranges, finalized minutes with no fills are materialized as synthetic zero-candles in RAM.
- These runtime synthetic candles are not persisted to disk shards.
- If REST later returns a real candle for a synthetic timestamp, the real candle overwrites it and EMA cache is invalidated.

## Implementation Details

### Cache Location

`caches/ohlcv/{exchange}/{symbol}.parquet`

### Concurrency

- Per-symbol file locks prevent concurrent writes
- Stale locks (>10 minutes) auto-cleaned
- Multi-process safe (optimizer workers share cache)

### Data Sources (Priority Order)

1. Local Parquet cache
2. Exchange API (Binance, Bybit, Bitget archives, CCXT fallback)
3. TradFi providers (Yahoo Finance, Finnhub) for stock perps

**Code**: `src/candlestick_manager.py`

## Configuration

Relevant config options in `config.backtest`:
- `exchange` - Primary data source
- `start_date`, `end_date` - Date range for backtesting

## Known Issues / Quirks

### EMA Cache Invalidation

When real data replaces synthetic candles, EMA cache must be invalidated. This is handled automatically but logged at DEBUG level.

### Persistent Gaps

Some symbols have permanent gaps (exchange maintenance, delistings). These are logged once and not re-warned.

### Exchange-Specific Behaviors

See `exchange_api_quirks.md` for exchange-specific data limitations.

## Testing

```bash
# Verify cached data integrity
python3 src/tools/verify_hlcvs_data.py

# Check specific symbol
python3 -c "
from src.candlestick_manager import CandlestickManager
cm = CandlestickManager('binance')
df = cm.get_candles('BTC/USDT:USDT', '2026-01-01', '2026-01-15')
print(f'Candles: {len(df)}, gaps: {df.volume.eq(0).sum()}')
"
```

## Debugging Tips

### Missing candles
1. Check if data exists on exchange for that time range
2. Look for "synthesized X zero-candles" log messages
3. Verify cache file exists and has expected date range

### Stale data
1. Delete cache file: `rm caches/ohlcv/{exchange}/{symbol}.parquet`
2. Re-fetch with longer warm-up period

### Multi-process lock contention
1. Check for orphan lock files: `find caches -name "*.lock"`
2. Delete stale locks (auto-cleaned after 10 minutes normally)
