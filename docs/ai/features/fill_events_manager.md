# Fill Events Manager

The FillEventsManager tracks trade fills and computes realized PnL across all exchanges.

## Key Concepts

### Fill Event

A fill event represents an executed trade:
- Symbol, side (buy/sell), position_side (long/short)
- Quantity, price, fees
- PnL (for closes)
- Timestamp, fill ID

### Event Sources

Different exchanges provide fill data through different endpoints:
- Trade history (`fetch_my_trades`)
- Closed PnL / Position history (`fetch_positions_history`)

Events from multiple sources are merged and deduplicated.

## Implementation Details

### Cache Location

`caches/{exchange}/fill_events.json`

### Exchange-Specific Fetchers

| Exchange | Primary Source | PnL Source |
|----------|---------------|------------|
| Binance | fetch_my_trades | income history |
| Bybit | fetch_my_trades | closed-pnl (hybrid pagination) |
| Bitget | fetch_my_trades | embedded in trade |
| Hyperliquid | fill events | embedded |
| OKX | fetch_my_trades | positions history |
| KuCoin | trades + positions history | positions history |
| Gate.io | fetch_my_trades | embedded |

**Code**: `src/fill_events_manager.py` - see `{Exchange}Fetcher` classes

### Bybit Hybrid Pagination

Bybit's closed-pnl endpoint has two pagination modes with different coverage. The fetcher uses both:
1. Cursor pagination for recent data (no gaps)
2. Time-based sliding window for older data
3. Deduplication by orderId

See `debugging_case_studies.md` for investigation details.

## Configuration

No specific configuration needed. FillEventsManager is initialized automatically by Passivbot.

## Known Issues / Quirks

### Missing PnL on Old Closes

Some exchanges only retain closed-pnl records for 30-60 days. Fills older than that may show `pnl: 0.0`.

### Pagination Limits

Most exchanges limit results per request (typically 100-500). For accounts with many fills, initial load may take multiple API calls.

### KuCoin PnL Discrepancy

KuCoin's trade PnL and position history PnL may differ slightly due to fee calculation timing. Logged at INFO (throttled).

## Testing

```bash
# Debug fill events for an account
python3 -c "
import asyncio
from src.fill_events_manager import FillEventsManager
fem = FillEventsManager('bybit', 'bybit_01')
asyncio.run(fem.ensure_loaded())
print(f'Events: {len(fem.events)}')
"
```

## Debugging Tips

### Missing fills
1. Check if fill exists on exchange (web UI)
2. Run with `--debug-level 2` to see fetch logs
3. Check cache file for raw data

### Wrong PnL
1. Verify closed-pnl record exists on exchange
2. Check if hybrid pagination is working (Bybit)
3. Look for "PnL discrepancy" logs

### Duplicate fills
1. Check for different fill IDs with same trade
2. Verify deduplication keys are correct per exchange
