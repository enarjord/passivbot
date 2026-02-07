# Log Analysis Prompt for Passivbot

Use this prompt to analyze Passivbot logs and identify areas for improvement.

---

## Context

You are analyzing logs from **Passivbot**, a cryptocurrency trading bot for perpetual futures markets. The bot runs continuously, managing positions across multiple exchanges (Binance, Bybit, Bitget, Hyperliquid, OKX, KuCoin, Gate.io) and symbols simultaneously.

### Why Logging Matters for Passivbot

1. **Real-time trading decisions**: Logs must capture enough context to understand why orders were created, modified, or cancelled
2. **Multi-symbol/multi-exchange**: A single bot instance may trade 50+ symbols; logs must clearly identify which symbol/exchange each entry refers to
3. **Debugging production issues**: When something goes wrong with real money, operators need to reconstruct exactly what happened
4. **Performance monitoring**: API rate limits, memory usage, and latency affect trading outcomes
5. **Audit trail**: Regulatory and personal record-keeping require understanding trade history

---

## Log Level Definitions (Target State)

The goal is to make each log level serve a distinct purpose:

### Golden Rules

| Level | Audience | Golden Rule |
|-------|----------|-------------|
| **INFO** | Operators watching logs | Sustainable to tail indefinitely in production |
| **DEBUG** | Developers troubleshooting | Tolerable for debugging sessions (won't generate GB/hour) |
| **TRACE** | Deep debugging specific issues | Full firehose; expect massive logs; enable briefly |

**Key tests:**
- Can someone run `--debug-level 1` (INFO) for days without log bloat? → If not, move spam to DEBUG
- Can someone run `--debug-level 2` (DEBUG) for an hour without logs becoming unmanageable? → If not, move to TRACE
- Is every API payload logged? → That belongs in TRACE, not DEBUG

### INFO (default, human-friendly, essential)
The INFO level should be **clean and readable at a glance**. A human watching the logs should be able to understand what the bot is doing without being overwhelmed.

**Must include (core, essential):**
- Order creations with symbol, side, qty, price, order type
- Order cancellations with symbol and reason
- Fills (trades executed) with symbol, side, qty, price, PnL if close
- Balance changes with before/after values
- Position changes with size, entry price, wallet exposure
- Unexpected errors with full traceback

**Secondary (nice to have, but with spam prevention):**
- Overall health status of bot (periodic summary)
- Mode changes: `normal`, `graceful_stop`, `tp_only`, `panic`, `manual`
- Forager mode: changes in coin ranking by volume/volatility (only when rankings change meaningfully)
- Unstucking allowances and coin rankings for unstuck selection
- EMA gating notifications: when initial entry or unstuck orders are blocked due to EMA distance (users need to know the bot is intentionally waiting, not broken)
- Expected/recoverable errors in short, truncated form

### DEBUG (technical details, still human-readable)
DEBUG should provide **more detail on bot internals** without being overwhelming. Useful for understanding why decisions were made.

**Should include:**
- Cache updates (loading from disk, writing to disk)
- Candle updates and synthetic candle creation
- EMA cache invalidations
- Fill event fetches (every fetch, not just when there are new fills)
- WebSocket connection events (connect, disconnect, reconnect)
- Rate limit backoffs
- Order plan summaries (the detailed breakdown of cancel/create counts)
- API call timing and parameters

**Should NOT include:**
- Full API payloads (save for TRACE)
- Every candle replacement message individually

### TRACE (full debug, firehose mode)
TRACE is for **deep debugging sessions**. Expect hundreds of megabytes of logs per minute. Everything goes here.

**Should include:**
- All DEBUG content
- Full API request/response payloads
- Every candle update individually
- Internal state snapshots
- WebSocket message contents
- Every calculation step in order logic

---

## Current Log Issues (Observed Patterns)

Based on analysis of production logs, here are specific issues to look for:

### 1. INFO Level Spam (Move to DEBUG)

These messages currently appear at INFO but should be DEBUG:

**Fill event fetches with no changes:** ✅ FIXED (Round 1)
```
2026-01-23T20:00:17 INFO     [bybit] BybitFetcher.fetch: done (events=20, trades=99, positions=31)
2026-01-23T20:00:17 INFO     [bybit] [fills] refresh: events=273 (+0) | persisted 3 days
```
*Problem: These log every 30 seconds even when nothing changed. The `(+0)` indicates no new events.*
*Solution: Only log at INFO when there are new fills. Log at DEBUG otherwise.*
*Status: Fixed in `fill_events_manager.py` - now logs at DEBUG when no new fills.*

**Candle replacement messages:** ✅ FIXED (Round 1)
```
2026-01-23T20:00:52 INFO     [bitget] [candle] HYPE/USDT:USDT: real data replaced 2 synthetic candles, EMA cache invalidated
2026-01-23T20:00:52 INFO     [bitget] [candle] SUI/USDT:USDT: real data replaced 200 synthetic candles, EMA cache invalidated
... (dozens more)
```
*Problem: During warmup and data maintenance, these flood the log. Not actionable for operators.*
*Solution: Aggregate into a single summary at INFO (e.g., "replaced synthetic candles for 22 symbols"). Individual messages at DEBUG.*
*Status: Fixed in `candlestick_manager.py` - batch mode during warmup produces aggregated summary at INFO; individual messages at DEBUG.*

**Candle refresh summaries (active/forager):** ✅ FIXED (Round 2)
```
2026-02-04T23:06:40 INFO     [binance] [candle] active refresh symbols=7, refreshed=7 max_stale=99s ...
2026-02-04T23:06:06 INFO     [binance] [candle] forager refresh slots_open=yes candidates=23 stale=23 ...
```
*Problem: These log every ~1–2 minutes and drown INFO logs.*
*Solution: Move to DEBUG and suppress during initial boot.*
*Status: Fixed in `passivbot.py` - now DEBUG with a 5‑minute boot delay.*

**Disk coverage / missing span diagnostics:** ✅ FIXED (Round 2)
```
2026-02-04T23:05:01 INFO     [bitget] [candle] event=disk_coverage_missing ...
2026-02-04T23:06:00 INFO     [binance] [candle] event=historical_missing_spans ...
```
*Problem: Intended for debugging; too noisy at INFO.*
*Solution: Move to DEBUG.*
*Status: Fixed in `passivbot.py` and `candlestick_manager.py`.*

**Volume/volatility EMA rankings (when unchanged):**
```
2026-01-23T20:00:32 INFO     [hyperliquid] volume EMA span 1250: 22 coins elapsed=17s, top8: BTC=846390.13, ...
2026-01-23T20:00:32 INFO     [hyperliquid] log_range EMA span 95: 22 coins elapsed=17s, top8: HYPE=0.002226, ...
```
*Problem: Logged frequently even when rankings haven't changed meaningfully.*
*Solution: Only log at INFO when rankings change enough to affect coin selection. Otherwise DEBUG.*

**Symbol map dumps:** ✅ FIXED (Round 1)
```
2026-01-23T19:59:22 INFO     [bybit] dumping symbol_to_coin_map caches/symbol_to_coin_map.json
2026-01-23T19:59:22 INFO     [bybit] dumping coin_to_symbol_map caches/bybit/coin_to_symbol_map.json
```
*Problem: Internal cache operations, not relevant to normal operation.*
*Solution: Move to DEBUG.*
*Status: Fixed in `utils.py` - now logs at DEBUG.*

**Rate limit backoff notices:** ✅ FIXED (Round 2)
```
2026-02-04T23:02:38 INFO     [hyperliquid] [candle] event=rate_limit_global_set ...
```
*Problem: Backoff is expected behavior; INFO is too chatty.*
*Solution: Move to DEBUG.*
*Status: Fixed in `candlestick_manager.py`.*

### 2. Excessive WARNING for Expected Conditions

**Synthesized zero-candles:** ✅ FIXED (Round 1)
```
2026-01-23T19:59:36 WARNING  [hyperliquid] [candle] ADA/USDC:USDC: synthesized 1 zero-candle at 2026-01-23T19:58 (no trades from exchange) using prev_close=0.361370
... (many more per minute)
```
*Problem: This is normal for illiquid pairs. WARNINGs should indicate problems, not expected behavior.*
*Solution: Aggregate to INFO summary (e.g., "synthesized 17 zero-candles across 15 symbols"). Individual messages at DEBUG. Only use WARNING for large gaps.*
*Status: Fixed in `candlestick_manager.py` - aggregated summary at INFO (WARNING only if >1000 candles); individual messages at DEBUG (WARNING only if >5 candles).*

**Stale lock removal:** ✅ FIXED (Round 1)
```
2026-01-23T19:59:22 WARNING  [bybit] removed stale symbol map lock caches/symbol_to_coin_map.json.lock (age 6599.5s)
```
*Problem: This is self-healing behavior, not a warning condition.*
*Solution: Move to INFO or DEBUG.*
*Status: Fixed in `utils.py` - now logs at INFO.*

**Deprecated API key field names:**
```
2026-01-23T19:59:23 WARNING  [bybit] bybit: 'key' in api-keys.json is deprecated, use 'apiKey' instead
```
*Problem: Logged every startup. Should be INFO or logged once per session.*
*Solution: Use `log_once` pattern or INFO level.*

### 3. Missing INFO-Level Information

**EMA gating for entries:**
When the bot doesn't place initial entry orders because price is too far from EMA, users don't know why the bot isn't entering. This should be logged at INFO.
*Needed: `[entry] BTC/USDT:USDT initial entry blocked: price 89500 is 2.3% above EMA (threshold: 1.5%)`*

**Unstuck gating:**
When unstuck orders are blocked due to EMA distance or other conditions.
*Needed: `[unstuck] SUI/USDT:USDT unstuck entry blocked: EMA distance 3.5% exceeds threshold 2.0%`*

**Unstuck coin selection:**
When forager mode selects which coin to unstuck.
*Needed: `[unstuck] selecting DOGE for unstuck (rank 1/7, WE excess: 15.2%)`*

**Websocket state changes:**
```
2026-01-23T19:59:40 INFO     [bybit] [ws] bybit: starting order watch
```
*Good: This exists. But reconnections should be logged at INFO too.*

### 4. Good Patterns to Preserve

**Position changes with context:**
```
2026-01-23T19:59:30 INFO     [bybit] [pos]     new XMR long  0.0 @ 0.0 -> 0.07 @  519.26 WE: 0.0661 |   4% WEL,   1% WELe,   4% TWEL | PA dist: 0.0025 upnl: -0.0896
```
*Excellent: Shows before/after, wallet exposure percentages, unrealized PnL.*

**Balance changes:**
```
2026-01-23T19:59:30 INFO     [bybit] [balance] 0.0 -> 550.0 equity: 432.7682 source: REST
```
*Good: Shows source of update.*

**Order operations with reasons:**
```
2026-01-23T20:00:23 INFO     [bybit] [order] cancel XMR | buy long 0.07@493.51 entry_grid_normal_long [replace reason=price Δp=0.079%]
2026-01-23T20:00:23 INFO     [bybit] [order]   post XMR | buy long 0.19@449.55 entry_grid_normal_long [replace reason=price Δp=0.216%]
```
*Excellent: Shows order details and reason for action.*

**Health summary:**
```
2026-01-23T19:59:44 INFO     [bybit] [health] uptime=21.0s | loop=4.3s | positions=1 long, 1 short | balance=550.00 USDT | orders=+0/-0 | fills=0 | errors=0/10 | ws_reconnects=0 | rate_limits=0 | rss=0MB
```
*Excellent: Periodic summary with all key metrics.*

**Mode changes:**
```
2026-01-23T21:23:42 INFO     [bybit] [mode] changed long.ETH/USDT:USDT: normal -> graceful_stop
```
*Good: Clear indication of mode transition.*

**Errors with traceback:**
```
2026-01-23T20:00:50 ERROR    [hyperliquid] error with run_execution_loop orchestrator compute_ideal_orders failed: MissingEma { symbol_idx: 5 }
Traceback (most recent call last):
  File "/root/passivbot/src/passivbot.py", line 1263, in run_execution_loop
  ...
```
*Good: Full traceback for unexpected errors.*

**Rate limit detection:**
```
2026-01-23T21:30:06 WARNING  [bybit] event=ccxt_fetch_ohlcv_failed ... error_type=RateLimitExceeded error=bybit {"retCode":10006,"retMsg":"Too many visits..."
2026-01-23T21:30:06 INFO     [bybit] event=rate_limit_global_set ... backoff_seconds=5.0 total_count=1
```
*Good: Clear rate limit handling.*

### 5. Inconsistencies

**Mixed tag formats:** ✅ MOSTLY FIXED (Round 6)
- Some use `[tag]` format: `[boot]`, `[order]`, `[pos]`, `[balance]`
- Some don't: `BybitFetcher.fetch:`, `Memory usage rss=...`
*Solution: Standardize on `[tag]` format for all log categories.*
*Status: Core messages now use `[tag]` format. `BybitFetcher.fetch:` was already moved to DEBUG in Round 2.*

**Mixed f-string vs format strings:**
```python
# F-string (not log-aggregation-friendly)
logging.info(f"removed {len(removed_orders)} orders")
# Better for structured logging
logging.info("removed %d orders", len(removed_orders))
```

---

## Analysis Format

Structure your findings as follows:

```markdown
## Summary

[2-3 sentence overview of log quality and main findings]

## Critical Issues

[Issues that directly impact INFO level readability or miss essential information]

### Issue: [Title]
- **Current**: [Example of current log]
- **Problem**: [What's wrong]
- **Impact**: [Why it matters]
- **Recommendation**: [Specific fix with suggested log message]

## Level Adjustments

[Messages that should move between INFO/DEBUG/TRACE]

### Move to DEBUG: [Category]
- **Current level**: INFO
- **Example**: [Log message]
- **Reason**: [Why it doesn't belong at INFO]

## Missing Information

[Things that should be logged but aren't]

### Missing: [Title]
- **When**: [What situation triggers this]
- **Why needed**: [What operators can't understand without it]
- **Suggested message**: [Example log message]

## Patterns to Preserve

[Good logging patterns observed that should be maintained]

## Questions for Operators

[Clarifying questions about logging requirements or priorities]
```

---

## Specific Areas to Examine

### Startup Sequence
- Is configuration clearly logged?
- Are exchange connections confirmed?
- Is warmup progress visible without spam?
- Is the transition to live trading clearly marked?

### Order Lifecycle
- Can you trace why an order was created (entry grid, unstuck, close)?
- Is it clear why orders are cancelled and recreated?
- Are fills logged with sufficient detail (price, qty, side, PnL)?

### Position Management
- Are position changes (open, add, reduce, close) clearly logged?
- Is wallet exposure logged when approaching limits?
- Is it clear when positions hit unstuck thresholds?

### Forager Mode
- Can you see which coins were selected and why?
- Is it clear when coins rotate in/out of the active set?
- Are ranking changes visible without spam?

### EMA Gating
- Can you tell when entries are blocked due to EMA distance?
- Is it clear what the current EMA vs price distance is?
- Are threshold crossings logged?

### Error Handling
- Do unexpected errors have full tracebacks?
- Are expected/recoverable errors summarized briefly?
- Is error recovery logged (retry attempts, success after retry)?

---

## Example Improved Log Flow (Target State)

```
2026-01-23T20:00:00 INFO     [bybit] [boot] READY - entering main trading loop
2026-01-23T20:00:01 INFO     [bybit] [health] uptime=5s | positions=2L/0S | balance=1500.00 | orders=+0/-0 | fills=0 | errors=0/10
2026-01-23T20:00:15 INFO     [bybit] [fill] BTC long entry_grid_normal_long +0.001 @ 89500.00 id=trade-abc123
2026-01-23T20:00:15 INFO     [bybit] [pos] BTC long 0.001 @ 89500.00 -> 0.002 @ 89525.00 | WE: 12% WEL | upnl: -0.05
2026-01-23T20:00:30 INFO     [bybit] [entry] ETH initial entry blocked: price 2950 is 2.1% above EMA (threshold: 1.5%)
2026-01-23T20:01:00 INFO     [bybit] [mode] changed long.SUI: normal -> graceful_stop (dropped from forager top 7)
2026-01-23T20:01:00 INFO     [bybit] [forager] ranking changed: +DOGE(#7), -SUI | top7: BTC,ETH,SOL,XRP,HYPE,ADA,DOGE
2026-01-23T20:05:00 INFO     [bybit] [unstuck] selecting DOGE for entry (rank 1/7 by excess WE, excess: 15.2%)
2026-01-23T20:05:01 INFO     [bybit] [order] cancel DOGE | 3 orders (entry_grid) [price moved 0.5%]
2026-01-23T20:05:01 INFO     [bybit] [order] create DOGE | buy long 1000@0.124 entry_grid_normal_long
2026-01-23T20:15:00 INFO     [bybit] [health] uptime=15m | positions=3L/0S | balance=1500.00 | orders=+15/-12 | fills=2 | errors=0/10
```

Compare to current (noisy):
```
2026-01-23T20:00:17 INFO     [bybit] BybitFetcher.fetch: done (events=20, trades=99, positions=31)
2026-01-23T20:00:17 INFO     [bybit] [fills] refresh: events=273 (+0) | persisted 3 days
2026-01-23T20:00:18 INFO     [bybit] [candle] BTC/USDT:USDT: real data replaced 1 synthetic candle, EMA cache invalidated
2026-01-23T20:00:18 INFO     [bybit] [candle] ETH/USDT:USDT: real data replaced 1 synthetic candle, EMA cache invalidated
... (15 more candle messages)
2026-01-23T20:00:23 INFO     [bybit] log_range EMA span 95: 22 coins elapsed=2s, top8: HYPE=0.002, XMR=0.0018...
2026-01-23T20:00:30 INFO     [bybit] BybitFetcher.fetch: done (events=20, trades=99, positions=31)
2026-01-23T20:00:30 INFO     [bybit] [fills] refresh: events=273 (+0) | persisted 3 days
```

---

## Questions to Guide Analysis

- If the bot crashed, could you tell what state it was in from INFO logs alone?
- If an order was unexpectedly cancelled, can you find out why?
- If the bot isn't entering positions, can you tell if it's due to EMA gating or other reasons?
- Can you distinguish between "bot is working but waiting" vs "something is wrong"?
- Is the signal-to-noise ratio acceptable for watching logs in real-time?

---

## Implementation Progress

### Round 1 (2026-01-24) ✅ COMPLETED

**Issues addressed:**

1. **Fill event refresh spam** - Changed `[fills] refresh: events=N (+0)` to log at DEBUG when no new fills. Only logs at INFO when there are new fills (`+N` where N > 0).
   - File: `src/fill_events_manager.py`

2. **Candle replacement spam** - Added batch mode for candle replacement messages during warmup:
   - During warmup: Aggregated summary at INFO (e.g., "replaced 5000 synthetic candles across 22 symbols")
   - Outside warmup: Individual messages at DEBUG
   - File: `src/candlestick_manager.py`

3. **Zero-candle synthesis warnings** - Changed from WARNING to appropriate levels:
   - Aggregated summary during warmup: INFO (WARNING only if >1000 candles)
   - Individual messages: DEBUG (WARNING only if >5 candles gap)
   - File: `src/candlestick_manager.py`

4. **Symbol map dumps** - Changed from INFO to DEBUG
   - File: `src/utils.py`

5. **Stale lock removal** - Changed from WARNING to INFO (self-healing is not a warning)
   - File: `src/utils.py`

### Round 2 (2026-01-24) ✅ COMPLETED

**Issues addressed:**

1. **Deprecated API key warnings** - Changed from WARNING to INFO and added log_once pattern to prevent startup spam.
   - File: `src/exchanges/ccxt_bot.py`

2. **Fetcher.fetch spam** - Changed all fetcher status messages from INFO to DEBUG:
   - `BitgetFetcher.fetch: start/done` → DEBUG
   - `BinanceFetcher.fetch: start/done` → DEBUG
   - `BybitFetcher.fetch: done` → DEBUG
   - `HyperliquidFetcher.fetch: fetch #N` → DEBUG
   - `GateioFetcher.fetch: start/done/no trades` → DEBUG
   - `OkxFetcher.fetch: start/done` → DEBUG
   - `KucoinFetcher._fetch_trades: fetch #N` → DEBUG
   - `KucoinFetcher._fetch_positions_history: fetch #N` → DEBUG
   - File: `src/fill_events_manager.py`

3. **Cache oldest event/full refresh spam** - Changed from INFO to DEBUG with once-per-session logging.
   - File: `src/passivbot.py`

4. **Volume/volatility EMA rankings** - Already implemented in Round 1 with ranking change detection. Only logs when top symbols change.
   - File: `src/passivbot.py`

### Round 2b (2026-01-24) ✅ COMPLETED

**Issues addressed:**

1. **CCXT API request/response payloads** - Moved full API payloads from DEBUG to TRACE level.
   - CCXT logs full HTTP request/response data including headers and JSON bodies at DEBUG level
   - These payloads can be 10KB+ per line (e.g., coin list responses) and accounted for ~23% of DEBUG log volume
   - Configured CCXT logger to only output at TRACE level (debug level 3); suppressed at DEBUG (level 2) and below
   - File: `src/logging_setup.py`
   - Impact: Reduces a 2411-line DEBUG log by 558 lines (~23% reduction), and more importantly removes the very long JSON payload lines

### Round 3 (2026-01-24) ✅ COMPLETED

**Issues addressed:**

1. **Zero-candle synthesis warnings repeating** - Changed from rate-limiting (once per minute per symbol) to gap deduplication (once per unique gap origin).
   - Same historical gaps were being warned about every minute (e.g., "DOGE: synthesized 7 zero-candles at 2026-01-23T19:16 to 2026-01-24T13:24")
   - Now uses a set to track which (symbol, first_ts) gaps have been warned about
   - Uses gap start timestamp as key since end timestamp changes as time passes
   - Warns only once per unique gap origin, eliminating repeated warnings
   - File: `src/candlestick_manager.py`

2. **Stale candle lock removal at WARNING** - Changed from WARNING to INFO level.
   - Stale lock removal is self-healing behavior, not an error condition
   - Consistent with symbol map lock removal which is already at INFO level
   - File: `src/candlestick_manager.py`

### Round 4 (2026-01-24) ✅ COMPLETED

**Issues addressed:**

1. **Volume/volatility EMA rankings logging too frequently** - Added 60-second throttle per metric.
   - Rankings were logged every 1-5 seconds during active trading due to small fluctuations
   - Now requires both a ranking change AND 60 seconds since last log
   - Reduces log volume significantly while still showing meaningful ranking changes
   - Files: `src/passivbot.py` (3 locations: `calc_volumes_and_log_ranges`, `calc_log_range`, `calc_volumes`)

2. **Mode changes flip-flopping creates noisy logs** - Added 60-second throttle for mode change logs.
   - Forager mode causes coins to oscillate between `normal` and `graceful_stop` when volatility rankings are close
   - "added" and "removed" logs (startup events) still logged immediately at INFO
   - "changed" logs throttled to at most once per 60 seconds per symbol
   - File: `src/passivbot.py`

### Round 5 (2026-01-25) ✅ COMPLETED

**Issues addressed:**

1. **BinanceFetcher fetch logs with size=0** - Changed to only log at INFO when size > 0.
   - `_fetch_income` and `_fetch_symbol_trades` were logging at INFO for every fetch page even when no data returned (size=0)
   - Now logs at DEBUG when size=0, INFO only when there's actual data
   - File: `src/fill_events_manager.py`

2. **EMA rankings logging too frequently** - Increased throttle from 60 seconds to 5 minutes.
   - Rankings were still logging every 1-2 minutes due to frequent ranking changes in forager mode
   - Now uses 5-minute throttle (300,000ms) which is more sustainable for production tailing
   - Added `[ranking]` tag for consistency
   - Files: `src/passivbot.py` (3 locations)

3. **Zero-candle synthesis warnings repeating** - Improved deduplication and level thresholds.
   - Changed gap key from exact `first_ts` to hour-rounded `first_ts` to better deduplicate gaps detected at different window boundaries
   - Raised WARNING threshold from >5 to >100 candles (startup batch summary already covers normal gaps)
   - File: `src/candlestick_manager.py`

4. **"unexpected step for tf" warnings** - Changed from WARNING to DEBUG.
   - These are expected behavior for exchanges with data gaps (especially illiquid pairs)
   - Not actionable for operators, only useful for debugging data quality
   - Added `[candle]` tag for consistency
   - File: `src/candlestick_manager.py`

5. **Deprecated API key warnings aggregated** - Combined into single message.
   - Was logging one message per deprecated field (key, private_key, passphrase, wallet_address)
   - Now logs single aggregated message: `[config] bybit: deprecated api-keys.json fields remapped: key->apiKey, private_key->privateKey`
   - File: `src/exchanges/ccxt_bot.py`

6. **Fill logs improved** - Added fill ID and consistent PnL for closes.
   - Added truncated fill ID (first 12 chars) to all fill logs for traceability
   - Close orders now always show `pnl=` even when PnL is 0.0
   - Helps distinguish duplicate logs from separate fills of different orders
   - Format: `[fill] SUI long close_unstuck_long -10 @ 1.47, pnl=-0.937 USDT id=trade-abc123`
   - File: `src/passivbot.py`

### Round 6 (2026-01-26) ✅ COMPLETED

**Issues addressed:**

1. **Standardized log tag formats** - Added `[tag]` style to untagged messages for consistency.
   - `Memory usage rss=...` → `[memory] rss=...`
   - `warmup starting: N symbols...` → `[warmup] starting: N symbols...`
   - `warmup candles: N/M...` → `[warmup] candles: N/M...`
   - `Starting hourly_cycle...` → `[hourly] starting maintenance cycle`
   - `Initializing FillEventsManager...` → `[fills] initializing FillEventsManager`
   - `FillEventCache.load:...` → `[fills] cache loaded/load:...`
   - `Symbol/coin mapping fallbacks:...` → `[mapping] fallbacks:...`
   - Files: `src/passivbot.py`, `src/fill_events_manager.py`

2. **Strict mode gaps** - Changed from WARNING to DEBUG level.
   - These are expected for illiquid markets and not actionable for operators
   - File: `src/candlestick_manager.py`

3. **Persistent gaps message** - Changed from WARNING to INFO and added `[candle]` tag.
   - Already had 60-second throttling; message is informational about expected behavior
   - Now uses format string for log aggregation compatibility
   - File: `src/candlestick_manager.py`

### Round 7 (2026-01-26) ✅ COMPLETED

**Issues addressed:**

1. **KucoinFetcher PnL discrepancy throttling** - Enhanced deduplication.
   - Increased throttle from 5 minutes to 1 hour
   - Added delta-based deduplication: only logs again if delta changes by >10%
   - Same discrepancy no longer spams logs every 5 minutes
   - File: `src/fill_events_manager.py`

2. **WebSocket reconnection logging** - Added explicit reconnection messages.
   - `[ws] {exchange}: connection lost (reconnect #N), retrying in 1s: {error_type}` at WARNING
   - `[ws] {exchange}: reconnecting...` at INFO before retry
   - Full traceback moved to DEBUG level to reduce log noise
   - Files: `src/exchanges/ccxt_bot.py`, `src/exchanges/hyperliquid.py`

3. **Mode change throttle** - Extended and improved throttle.
   - Increased throttle from 60s to 120s (2 minutes)
   - Now throttles ALL mode change types (added/removed/changed), not just "changed"
   - Throttle key includes event type to avoid cross-type interference
   - Reduces forager mode oscillation noise significantly
   - File: `src/passivbot.py`

### Round 8 (TODO)

**Log analysis date:** 2026-01-28

**Logs reviewed:**
- `hyperliquid_canon` (5 coins, BTC/ETH/XRP/SOL/ADA long)
- `bitget_01` (forager mode)
- `bybit_01` (forager mode)
- `ebybitsub03` (XMR long/short)
- `hyperliquid_trump` (5 coins long)
- `kucoin_01` (forager mode, 3 positions)
- `gateio_01` (forager mode, 3 positions)
- `okx_faisal` (forager mode, 3 positions)

**Positive observations:**
- Boot sequence is clear and informative
- Health summaries are periodic and useful (~15min intervals)
- Order operations well-tagged with reasons
- Fill logs include relevant details (ID, PnL for closes)
- Position changes show before/after state with WE and upnl
- WebSocket reconnection logging improved (Round 7)
- Unstuck logging now shows allowance, peak, pct_from_peak

**Issues identified (priority ordered):**

1. **HIGH: OkxFetcher.fetch spam at INFO** - Every 30-40 seconds:
   ```
   INFO [okx] OkxFetcher.fetch: /fills #2 after=None size=12
   ```
   This should be DEBUG level. OKX appears to have been missed in Round 2 fix.
   - **Action:** Move to DEBUG in `src/fill_events_manager.py`

2. **HIGH: FillEventsManager messages lack [fills] tag** - Messages like:
   ```
   FillEventsManager.ensure_loaded: loaded 3507 cached events
   FillEventsManager initialized: 3507 cached events loaded
   ```
   Should be `[fills] ensure_loaded:...` and `[fills] initialized:...`
   - **Action:** Add [fills] tag prefix in `src/fill_events_manager.py`

3. **MEDIUM: Persistent gaps messages too frequent** - KuCoin logs show:
   ```
   INFO [kucoin] [candle] persistent gaps: 1 across 1 symbols (TRX/USDT:USDT:1).
   ```
   Logged every ~1 minute. Should be throttled to once per hour per symbol.
   - **Action:** Add throttle in `src/candlestick_manager.py`

4. **MEDIUM: Hyperliquid "Could not fetch server time" WARNING repetition** - Hourly:
   ```
   WARNING Could not fetch server time: hyperliquid fetchTime() is not supported yet
   ```
   This is expected behavior. Should be INFO + log_once or removed entirely.
   - **Action:** Change to DEBUG or log_once at INFO in exchange time sync code

5. **MEDIUM: Mode change oscillation still noisy** - Despite 2-minute throttle:
   - Gate.io SUI mode changes every 2-5 minutes
   - Hyperliquid coins flip frequently
   Consider longer throttle (5min?) or hysteresis (require N minutes stable before switching)
   - **Action:** Increase throttle or add hysteresis logic

6. **LOW: Order plan summary for no-ops** - When nothing happens:
   ```
   order plan summary | cancel 1->0 | create 1->0 | skipped=1 | unchanged_cancel=1 unchanged_create=1
   ```
   Could be DEBUG when all orders are skipped (no actual work done).
   - **Action:** Conditional level based on whether actual work was done

7. **LOW: KuCoin set_margin_mode errors for expected failures**:
   ```
   ERROR [kucoin] HYPE/USDT:USDT: error set_margin_mode 'HYPE/USDT:USDT'
   ```
   If margin mode is already correctly set, this shouldn't be ERROR.
   - **Action:** Check if already set before logging error

8. **DEFERRED: Missing INFO-level information** (from previous rounds):
   - [ ] EMA gating for entries (when initial entry blocked due to EMA distance)
   - [ ] Unstuck coin selection logging improvements

9. **DEFERRED: Zero-candle synthesis improvements** - Low priority, doesn't affect operations

**Plan for Round 8 execution:**

Phase 1 (HIGH priority - fix now): ✅ COMPLETED
- [x] Fix OkxFetcher.fetch INFO spam → moved to DEBUG (`src/fill_events_manager.py:3703`)
- [x] Add [fills] tag to FillEventsManager messages (`src/fill_events_manager.py:1783`, `src/passivbot.py:2517`)

Phase 2 (MEDIUM priority): ✅ COMPLETED
- [x] Throttle persistent gaps messages → increased from 60s to 30min (`src/candlestick_manager.py:746`)
- [x] Fix Hyperliquid fetchTime warning → use log_once to avoid hourly repetition (`src/exchanges/ccxt_bot.py:286`)
- [ ] Consider mode change hysteresis (deferred - requires design discussion)

Phase 3 (Defer to Round 9):
- [x] Order plan summary conditional levels → see Round 9
- [x] KuCoin margin mode error handling → see Round 9
- [ ] EMA gating information
- [ ] Zero-candle deduplication improvements

### Round 9 (2026-01-28) ✅ COMPLETED

**Issues addressed:**

1. **Order plan summary conditional levels** - Use DEBUG when no work done.
   - Changed from always INFO to conditional: INFO when orders created/cancelled, DEBUG when all skipped
   - Reduces noise from no-op order cycles (e.g., `cancel 1->0 | create 1->0 | skipped=1`)
   - File: `src/passivbot.py:4083`

2. **KuCoin exchange config error handling** - Reduce ERROR to WARNING.
   - Changed ERROR to WARNING for set_margin_mode failures (expected on some symbols)
   - Changed ERROR to WARNING for set_leverage failures (expected when symbol not in max_leverage)
   - Fixed copy-paste bug: line 507 said "error set_margin_mode" in set_leverage section
   - File: `src/exchanges/kucoin.py:478,486,507,515`

**Remaining for future rounds:**
- Zero-candle deduplication improvements

### Round 10 (2026-01-28) ✅ COMPLETED

**Issue addressed:**

1. **EMA gating information** - Log when initial entries are blocked due to EMA distance.
   - Added `_log_ema_gating()` method to detect when entries should exist but don't
   - For symbols in "normal" mode with no position and no initial entry order:
     - Calculates EMA bands (min/max of ema_span_0, ema_span_1, sqrt(span0*span1))
     - Calculates EMA entry threshold (ema_lower * (1-dist) for long, ema_upper * (1+dist) for short)
     - Logs when current price is beyond the EMA threshold
   - Throttled to once per 5 minutes per symbol/pside to avoid spam
   - Log format: `[ema] COIN pside entry gated | price=X ema_thresh=Y (+Z% away)`
   - File: `src/passivbot.py` (added `_log_ema_gating` method, called in both orchestrator methods)

### Round 11 (2026-02-04) ✅ COMPLETED

**Issue addressed:**

1. **Mode change logging redesign** - Separate DEBUG and INFO levels for mode changes.
   - **Problem:** Mode change oscillation (normal↔graceful_stop) due to forager ranking fluctuations
     was flooding INFO logs with 10-20+ messages per hour per exchange, making logs hard to tail.
   - **Solution:** Refactored `_log_mode_changes()` method with two-tier logging:
     - **DEBUG level:** All mode changes logged with full detail (no throttling) for debugging
     - **INFO level:** Selective, user-relevant logging only:
       - "added" with "normal" → forager selection (includes slot context like "forager slot 3/5")
       - "added" with "graceful_stop" → only on startup/first run
       - "removed" → always (coin exiting the system)
       - "changed" normal↔graceful_stop → **suppressed** (oscillation noise, available at DEBUG)
       - "changed" to/from tp_only/manual/panic → always (significant mode changes)
   - INFO-level throttle increased from 2 minutes to 5 minutes for remaining logged changes
   - Added slot context to forager selection logs: `[mode] added long.XRP/USDT:USDT: normal (forager slot 3/5)`
   - File: `src/passivbot.py` (refactored mode change logging in `update_PB_modes_and_first_timestamps`)

### Round 12 (2026-02-06) ✅ COMPLETED

**Issues addressed:**

1. **Unsupported markets warning noise** - Demoted to INFO and tagged.
   - Now logs as `[config] skipping unsupported markets...` at INFO (log_once).
   - File: `src/passivbot.py`

2. **Hedge mode success spam** - Moved successful set_position_mode logs to DEBUG with `[config]`.
   - Keeps ERROR/WARNING paths unchanged; removes hourly INFO spam.
   - Files: `src/exchanges/binance.py`, `src/exchanges/bybit.py`, `src/exchanges/bitget.py`, `src/exchanges/okx.py`, `src/exchanges/ccxt_bot.py`

3. **BinanceFetcher unsupported symbols** - Moved to DEBUG with `[fills]` tag.
   - File: `src/fill_events_manager.py`

4. **Bitget OHLCV limit probe** - Moved to DEBUG.
   - File: `src/candlestick_manager.py`

5. **KuCoin PnL discrepancy spam** - Added minimum interval between logs even when delta changes.
   - File: `src/fill_events_manager.py`

6. **Zero-candle warning threshold** - WARNING now only for >1000 candles during live operation.
   - File: `src/candlestick_manager.py`

7. **Order plan tag + MissingEma context** - Added `[order]` tag to plan summaries and emit symbol context on MissingEma errors.
   - File: `src/passivbot.py`

8. **Symbol map lock noise** - Demoted lock acquisition warnings to INFO with `[mapping]`.
   - File: `src/utils.py`
