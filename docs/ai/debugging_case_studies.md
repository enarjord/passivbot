# Debugging Case Studies

This document captures debugging sessions for complex issues, serving as a reference for future investigations.

## Case Study: Missing Bybit PnL Data (2026-01-25)

### Initial Symptom
User reported that some Bybit close fills had `pnl: 0.0` while others had correct PnL values.

### Investigation Process

#### Step 1: Identify the Pattern
```python
# Load cached fill events and filter to closes
closes = [x for x in events if x['side'] == 'sell' and x['position_side'] == 'long']
zero_pnl = [c for c in closes if abs(c['pnl']) < 0.0001]
```

Found: 1 XMR fill from Dec 30 had zero PnL, while Dec 28 and Jan 12+ fills had PnL.

#### Step 2: Check Raw Data
Examined the `raw` field in the cached fill event:
```python
# Fill with zero PnL only had fetch_my_trades data, no positions_history
event['raw']
# [{'source': 'fetch_my_trades', 'data': {...}}]
# Missing: {'source': 'positions_history', 'data': {...}}
```

#### Step 3: Verify Data Exists on Exchange
Created a test script to query Bybit directly:
```python
# scripts/check_missing_pnl.py
params = {'category': 'linear', 'symbol': 'XMRUSDT', 'limit': 100, 'endTime': end_ms}
result = await api.private_get_v5_position_closed_pnl(params)
# Found the record! It exists on Bybit.
```

**Key finding:** The closed-pnl record existed on Bybit but wasn't being fetched.

#### Step 4: Trace the Fetch Logic
Added debug output to understand pagination:
```python
# Fetch #10: Dec 29 10:45 → Jan 03 14:23 (100 records)
# Missing Dec 30 05:28 record should be in this window!
```

**Key finding:** 128 records existed in the time window, but only 100 were fetched.

#### Step 5: Identify Root Cause
The time-based pagination was skipping records:
1. Fetch returns 100 records, oldest at timestamp T
2. Next fetch uses `endTime = T`
3. Records between T and the previous batch's oldest are skipped

#### Step 6: Test Cursor Pagination
```python
# Use cursor instead of time-based
cursor = response.get('result', {}).get('nextPageCursor')
params['cursor'] = cursor  # Continue with cursor
```

**Key finding:** Cursor pagination only covers ~7 days, then cursor becomes empty.

#### Step 7: Implement Hybrid Solution
Combined both approaches:
1. Use cursor pagination for recent data (no gaps)
2. Fall back to time-based sliding window for older data
3. Deduplicate by orderId

### Verification
```
Before fix: 387 records fetched (cursor-only), missing Dec 30 record
After fix: 1434 records fetched (hybrid), all records including Dec 30
```

### Key Lessons

1. **Don't trust CCXT wrappers blindly** - they may not expose all pagination mechanisms
2. **Check raw API responses** - create test scripts to query directly
3. **Understand pagination limits** - each exchange has different behaviors
4. **Compare counts** - if you expect N records but get fewer, investigate
5. **Check multiple endpoints** - the data might exist via different API calls

### Debug Scripts Created
- `scripts/check_missing_pnl.py` - Query specific orderId directly
- `scripts/debug_positions_history.py` - Trace pagination behavior
- `scripts/verify_pagination_fix.py` - Verify fix works

These can be adapted for similar issues on other exchanges.

---

## Template for New Case Studies

### Initial Symptom
(What the user reported or what was observed)

### Investigation Process
1. Identify the pattern
2. Check raw data
3. Verify data exists at source
4. Trace the code path
5. Identify root cause
6. Implement and verify fix

### Key Lessons
(What was learned that applies to future debugging)
