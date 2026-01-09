# Hedge Algorithm Iteration Log

## Current Benchmarks (USD-only collateral, btc_collateral_cap=0)

| Config | Suite Run |
|--------|-----------|
| No Hedge | `backtests/suite_runs/2026-01-08T21_08_00/suite_summary.json` |
| Hedge 100% | `backtests/suite_runs/2026-01-08T21_10_24/suite_summary.json` |

## Test Command

```bash
python3 src/backtest.py tmp/hedge_testing.json -dp -ht <threshold> --suite -ed 2026-01-06
```

---

## Algorithm Change (Active)

### Mark-to-Market Exposure

Changed `position_exposure()` in `passivbot-rust/src/hedge.rs` to use market price instead of entry price.

```rust
// Before: qty_to_cost(pos.size.abs(), pos.price, ...)
// After:  qty_to_cost(pos.size.abs(), market_price(sym), ...)
```

---

## Threshold Curve (USD-only, MTM)

| Threshold | ADG vs NoHedge | DD vs NoHedge | Hedge P&L | Efficiency |
|-----------|----------------|---------------|-----------|------------|
| No Hedge | 0% | 0% | $0 | - |
| **25%** | **-4.3%** | **-1.5%** | -$142k | **2.8x** (best) |
| 50% | -13.0% | **+1.1%** ⚠️ | -$205k | N/A (no benefit) |
| 75% | -13.0% | -2.5% | -$649k | 5.3x |
| 100% | -19.2% | -2.8% | -$925k | 6.7x |

*Efficiency = ADG cost % per DD benefit %*

### DD Improvement Curve

```
25%   -1.5% ███████████████
50%   +1.1% ░░░░░░░░░░░ (WORSE than no hedge!)
75%   -2.5% ████████████████████████
100%  -2.8% ████████████████████████████
```

---

## Key Findings

### 1. 25% Threshold is Most Efficient
- Best ADG/DD tradeoff: costs 2.8% ADG per 1% DD improvement
- Practical recommendation for users who want some protection at low cost

### 2. 50% Anomaly
- **DD is worse than no hedge** at 50% threshold
- Non-monotonic behavior suggests timing/rebalancing issues at partial coverage
- Needs investigation

### 3. Diminishing Returns Above 25%
- 75% and 100% provide marginally better DD (-2.5%, -2.8%)
- But at much higher ADG cost (-13%, -19%)
- Full hedge (100%) is inefficient

### 4. Hedge is Costly in Upward Markets
- All thresholds lose money on the hedge side (negative P&L)
- Shorting in crypto's upward-biased market is inherently expensive

---

## Recommendations

### For Users
- **Conservative:** Use threshold=0.25 for modest protection at low cost
- **Aggressive:** Use threshold=1.0 if DD reduction is priority over ADG
- **Avoid:** threshold=0.5 (worse than no hedge for DD)

### For Further Development
1. **Investigate 50% anomaly** - test 30%, 40%, 60% to map the failure region
2. **Test pprice vs MTM** - verify MTM is actually helping
3. **EMA gating** - only enter shorts when price > EMA for better timing
4. **Dynamic threshold** - adjust based on market conditions

---

## Files Modified

- `passivbot-rust/src/hedge.rs`: `position_exposure()` uses `market_price(sym)` instead of `pos.price`

---
