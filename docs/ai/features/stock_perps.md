# Stock Perpetuals (HIP-3)

Stock perpetuals are perpetual futures contracts that track traditional equity prices. On Hyperliquid, these are deployed via HIP-3 (Hyperliquid Improvement Proposal 3).

## Key Concepts

### Symbol Mapping

| User Input | CCXT Symbol | Exchange Symbol |
|------------|-------------|-----------------|
| `TSLA` | `XYZ-TSLA/USDC:USDC` | `xyz:TSLA` |
| `xyz:TSLA` | `XYZ-TSLA/USDC:USDC` | `xyz:TSLA` |
| `XYZ-TSLA` | `XYZ-TSLA/USDC:USDC` | `xyz:TSLA` |

The `xyz:` prefix indicates TradeXYZ as the HIP-3 builder. Other builders (FLX, KM, etc.) use their own prefixes.

### Margin Mode

**Critical**: Stock perps ONLY support isolated margin.

| Aspect | Stock Perps | Crypto Perps |
|--------|-------------|--------------|
| Margin mode | Isolated only | Cross or Isolated |
| Max leverage | 10x | Up to 50x |
| Balance backing | Per-position | Shared pool |

When isolated margin is used, each position has dedicated margin. Liquidations are contained per-position rather than cascading.

### Balance Implications

Isolated margin locks capital per-position:
```
Total balance: $105
  - Free balance: $12 (available for new positions)
  - TSLA margin: $50 (locked)
  - NVDA margin: $43 (locked)
```

If TSLA gets liquidated, only $50 is lost. Free balance and NVDA position are unaffected.

## Implementation Details

### Auto-Detection

Stock perps are detected by:
1. `xyz:` symbol prefix (or `XYZ-` in CCXT format)
2. `onlyIsolated: true` market flag
3. Membership in known stock ticker list

**Code**: `src/exchanges/hyperliquid.py:is_stock_perp()`

### Margin Mode Setting

Passivbot automatically sets margin mode per-symbol:
- Crypto perps: cross margin (default)
- Stock perps: isolated margin

**Code**: `src/exchanges/hyperliquid.py:set_margin_mode()`

### Leverage Calculation

For isolated margin, minimum leverage is calculated from TWEL:
```python
min_leverage = ceil(max(long_TWEL, short_TWEL))
```

This ensures margin requirements don't exceed balance.

### Symbol Filtering

Stock perps must be routed to Hyperliquid only. In combined/forager mode:
```python
if exchange != "hyperliquid" and is_stock_perp(symbol):
    continue  # Skip - not available on this exchange
```

**Code**: `src/passivbot.py:filter_approved_coins()` (search for stock perp filtering)

## Configuration

### Minimal Config

```json
{
  "live": {
    "user": "hyperliquid_01",
    "approved_coins": ["TSLA"],
    "leverage": 2,
    "filter_by_min_effective_cost": false,
    "minimum_coin_age_days": 0
  }
}
```

Key settings:
- `leverage: 2` - Safe starting point for isolated margin
- `filter_by_min_effective_cost: false` - Allow small balances ($10 min order)
- `minimum_coin_age_days: 0` - Stock perps are new markets

### Mixed Crypto + Stock Perps

```json
{
  "live": {
    "approved_coins": ["BTC", "ETH", "TSLA", "NVDA"]
  }
}
```

Passivbot handles margin mode automatically per-symbol.

## Data Sources

### Live Trading
1. Hyperliquid API (primary)
2. CandlestickManager cache

### Historical Data
1. Local cache (if available)
2. Hyperliquid API (last ~3.5 days)
3. Yahoo Finance (last 7 days, free)
4. Configured TradFi provider (older data)

**Note**: TradFi data is actual stock prices, not perp prices. Does not include funding rates or oracle behavior.

## Known Issues / Quirks

### Builder Registration Required

Before trading XYZ stock perps via API, you must register your wallet at trade.xyz:
1. Connect wallet
2. Sign verification
3. Accept ToS
4. Click "Enable Trading"

Without registration: "Insufficient margin" errors even with sufficient balance.

### Weekend Oracle Behavior

Stock perps use RedStone's HyperStone oracle. During market closure:
- Oracle maintains "stickiness" to closing price
- Large trades can move price within boundaries
- **Risk**: Weekend whale activity can trigger liquidations

### $10 Minimum Order

Stock perps have $10 minimum order value. With small balances, this limits grid depth.

For $100 balance, TSLA at $400:
- Min qty = $10 / $400 = 0.025
- Practical grid depth: ~5-8 entries

## Testing

### Live Test

```bash
# Test with small balance
python3 src/main.py configs/test_stock_perps.json --debug-level 2
```

Watch for:
- Margin mode correctly set to isolated
- Orders placed successfully
- Position tracking accurate

### Verify Symbol Mapping

```python
# In Python REPL with venv activated
import ccxt.pro as ccxt
hl = ccxt.hyperliquid({'options': {'fetchMarkets': {'types': ['swap', 'hip3'], 'hip3': {'dex': ['xyz']}}}})
await hl.load_markets()
print(hl.market('XYZ-TSLA/USDC:USDC'))
```

## Debugging Tips

### "Insufficient margin to place order"
- Wallet not registered with TradeXYZ
- Solution: Complete registration at trade.xyz

### Symbol not found
- Market cache stale
- Solution: Delete `caches/hyperliquid/markets.json`

### Orders rejected during weekends
- Oracle boundary hit or liquidity issues
- Solution: Consider pausing during extended market closures

### Balance discrepancy between UIs
- Hyperliquid UI shows total; TradeXYZ shows breakdown by builder
- Both are correct, just different views

## Files Modified

Key files for stock perps support:
- `src/exchanges/hyperliquid.py` - Margin mode, symbol detection
- `src/passivbot.py` - Symbol filtering, routing
- `src/candlestick_manager.py` - TradFi data source integration
- `src/config_utils.py` - Symbol normalization

See git log for `feature/stock-perps-hyperliquid` branch for full history.
