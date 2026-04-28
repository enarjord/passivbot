# Staged Live Data Reconciliation Plan

## Purpose

Track the work needed to make the Python side of live Passivbot fast, accurate, remote-call
budget aware, and robust before handing required state to Rust for order planning.

Rust remains the source of truth for trading logic. Python owns exchange I/O, state freshness,
market data plumbing, reconciliation, and safety gates before order execution.

## Current Scope

Primary branch: `feat/staged-live-planner`

Initial live smoke accounts:

- Hyperliquid unified: `hyperliquid_01`
- Bybit: `ebybitsub03`

Later validation targets:

- Bitget
- Gate.io
- OKX
- Binance
- KuCoin

## Required Live Inputs

Before order planning/execution, live bot must have coherent state for:

- positions
- fill events
- balance
- open orders
- completed candlesticks needed by indicators and state replay
- fresh market snapshot for bid, ask, and last price

## Working Principles

- Python must fail loudly for trading-critical missing or stale inputs.
- Rust decides ideal orders; Python must not patch trading behavior around stale state.
- Remote-call economy must not weaken active-symbol safety.
- Budgeting may delay inactive/secondary work, not required active account or price state.
- Candlestick manager should be narrowed toward completed candles only.
- Current in-progress minute/hour price truth should come from ticker/orderbook endpoints, not
  incomplete candles.
- Duplicate order prevention is trading-critical and needs explicit tests.

## Checklist

### 1. Freshness Ledger

- [x] Define a formal freshness ledger for live inputs.
- [x] Track freshness per surface: positions, fills, balance, open orders, completed candles, market snapshot.
- [x] Track symbol-level freshness where relevant.
- [x] Add planner precondition checks for required surfaces.
- [x] Add execution precondition checks for disappeared self-order creation safety.
- [x] Add DEBUG/INFO logs explaining disappeared-order freshness blocks.
- [x] Add tests for stale required inputs blocking order creation.

### 2. Market Snapshot Provider

- [x] Introduce a `MarketSnapshot`/price provider contract with bid, ask, last, source, and fetched timestamp.
- [x] Route staged orchestrator last-price reads through market snapshots before candle fallback.
- [x] Require fresh market snapshot before order creation for affected symbols.
- [x] Keep completed candles for indicators/history only.
- [x] Add tests proving incomplete candles are not used as live price truth when ticker data is available.

### 3. Exchange Ticker Capability Probes

- [x] Build a probe tool for `fetch_ticker(symbol)`.
- [x] Build a probe tool for `fetch_tickers(symbols)`.
- [x] Build a probe tool for `fetch_tickers()` without a symbol list.
- [x] Capture bid/ask/last availability and timestamp behavior.
- [x] Capture endpoint failure modes and rate-limit behavior.
- [x] Run and record Bybit probe.
- [x] Run and record Hyperliquid probe.
- [ ] Later: run Bitget, Gate.io, OKX, Binance, KuCoin probes.
- [x] Store durable exchange quirks/config for cheapest safe ticker strategy for tested exchanges.

### 4. Candlestick Manager Completed-Only Scope

- [x] Define the candlestick manager contract as completed candles only.
- [x] Stop using candlestick manager as source of current in-progress minute/hour price truth.
- [x] Ensure incomplete timeframe candles are never persisted.
- [x] Review runtime synthetic candle behavior and keep only completed-candle-safe usage.
- [x] Add tests for completed-minute boundary behavior.
- [ ] Add tests for completed-hour boundary behavior.
- [x] Add tests that current in-progress candles are not persisted.

### 5. Candle Completeness And Health

- [ ] Define required candle windows for HSL replay.
- [ ] Define required candle windows for trailing extrema.
- [x] Define required candle windows for EMA close, volatility, and quote volume.
- [x] Add helper(s) to report whether a symbol has required completed candles.
- [x] Add diagnostics for last completed 1m/15m/1h candle, gaps, synthetic count, cache coverage, and stale age.
- [x] Add INFO logs only for interesting candle health events.
- [x] Keep detailed candle health diagnostics at DEBUG.

### 6. Remote-Call Budgeting

- [x] Keep active position/open-order symbols highest priority.
- [x] Keep active forager slots second priority.
- [x] Rotate eligible inactive symbols by staleness.
- [x] Add configurable maximum inactive eligible candle staleness, e.g. `live.max_forager_candle_staleness_minutes`.
- [x] Ensure no active symbol is budget-starved.
- [ ] Ensure remote-call pacing avoids startup and minute-boundary bursts.
- [x] Add tests for fair stale-symbol rotation.
- [x] Add tests proving active symbols bypass non-critical budgeting.

### 7. Startup/Warmup

- [x] Split minimal trading-ready startup from background broad catch-up.
- [x] Fetch account-critical state first: positions, fills, balance, open orders.
- [x] Fetch active-symbol completed candles before broad eligible-universe candles.
- [ ] Fetch fresh active market snapshots before order creation.
- [x] Pace broad eligible-universe candle catch-up in the background.
- [ ] Add startup timing diagnostics: account-ready, market-ready, active-candle-ready, full-warmup-ready.
- [x] Add tests preventing broad simultaneous OHLCV startup bursts where feasible.

### 8. Duplicate-Order Guardrail

- [ ] Track known bot-emitted orders by exchange/client/order id with symbol, side, position side, qty, and price.
- [ ] Detect when a known bot-created order disappears from open orders.
- [ ] If position for that symbol is unchanged, mark symbol as suspect fill/stale position.
- [ ] Block order creations for suspect symbols until positions, fills, and open orders are refreshed coherently.
- [ ] Prefer blocking creations over risking duplicate entries.
- [ ] Keep cancellation behavior conservative but avoid churn under ambiguous state.
- [ ] Add tests for user cancel, exchange cancel, fill with stale position, fill with fresh position, and restart recovery.

### 9. Websocket-Triggered Reconciliation

- [ ] On order websocket event, mark account-critical state dirty.
- [ ] On missing bot-created order, force account-critical refresh before creations.
- [ ] On position change, force fills refresh.
- [ ] On significant balance change, force planner/order reconciliation.
- [ ] Add DEBUG/INFO reason logs for refresh causes.

### 10. Validation Matrix

- [x] Unit tests for freshness ledger.
- [x] Unit tests for market snapshot provider.
- [x] Unit tests for completed-only candle manager behavior.
- [x] Unit tests for duplicate-order guardrail.
- [ ] Fake-live staged-vs-legacy comparison where practical.
- [x] Bybit DEBUG smoke.
- [x] Hyperliquid unified DEBUG smoke.
- [x] Hyperliquid non-unified vanilla DEBUG smoke.
- [ ] Later: Bitget, Gate.io, OKX, Binance, KuCoin smoke tests.
- [x] Add fake-live support for comparing remote call counts before/after major changes.

## Current Completed Work Relevant To This Plan

- [x] Staged account refresh path exists and logs per-surface timings.
- [x] Bybit staged smoke run completed with DEBUG logging.
- [x] Candlestick manager has process-local OHLCV spacing.
- [x] Initial active/forager candle budget logic exists.
- [x] Fixed TTL bypass that caused secondary symbols to fetch normal one-candle trailing gaps.
- [x] Added regression coverage for single trailing present gap TTL behavior.
- [x] Added read-only `passivbot tool ticker-probe` for exchange ticker/top-of-book capability checks.
- [x] Recorded initial Bybit and Hyperliquid ticker/top-of-book probe findings.
- [x] Added `MarketSnapshotProvider` and routed staged orchestrator price reads through it.
- [x] Bybit DEBUG live smoke on `ebybitsub03` exercised staged market snapshots for 27 symbols:
  20 `fetch_tickers()` cycles, 0 misses, 0 invalid snapshots, 0 candlestick-manager fallback
  lines, mean ticker fetch about 1.2s.
- [x] Added shutdown-aware candle/EMA warmup checks and cancellation of a stuck execution loop
  before CCXT session close.
- [x] Verified Bybit Ctrl-C shutdown smoke on `ebybitsub03`: signal during warmup jitter
  aborted immediately, skipped further warmup/index rebuild work, closed sessions cleanly, and
  exited with code 0 without traceback.
- [x] Added a `FreshnessLedger` for live surfaces and symbol-level blocks; staged refresh now
  stamps account surfaces, market snapshots, and active completed-candle refreshes.
- [x] Added a disappeared-self-order guardrail: when a bot-created order vanishes without a
  known bot cancellation, new creations for that symbol are blocked until the next full account
  freshness cohort confirms balance, positions, open orders, and fills.
- [x] Added deterministic fake-cycle verification that the disappeared-self-order guardrail blocks
  a real order-planner create for the affected symbol, logs the freshness block, and clears after
  a full account freshness cohort.
- [x] Verified Bybit DEBUG smoke on `ebybitsub03` after freshness-ledger changes:
  three staged market-snapshot cycles, 27/27 symbols resolved from bulk `fetch_tickers()`,
  no `ERROR`, `Traceback`, `RecursionError`, or freshness guardrail block in the persisted log.
- [x] Fixed and regression-tested inline execution-loop shutdown: Ctrl-C during live startup no
  longer lets `shutdown_gracefully()` cancel the main `start_bot()` task and recurse through
  asyncio task cancellation.
- [x] Added staged planner precondition checks: before Rust order calculation, staged mode now
  requires current-epoch freshness for balance, positions, open orders, fills, completed candles,
  and market snapshots.
- [x] Tightened CandlestickManager to completed-candle-only behavior: compatibility
  `get_current_close()` now returns the latest completed close, ticker/current-price fetches moved
  to market snapshots, and tests cover no current-minute persistence.
- [x] Added completed-candle health diagnostics: CandlestickManager can now report 1m/15m/1h
  coverage without remote fetches, and live health summaries emit detailed DEBUG diagnostics plus
  INFO only when active-symbol candle health is stale, missing, or synthetic.
- [x] Tightened live candle budgeting: priority position/open-order symbols bypass forager budgets,
  forager/eligible symbols are ranked by latest completed-candle staleness, and
  `live.max_forager_candle_staleness_minutes` caps target staleness for broad eligible refreshes.
- [x] Split live startup candle work: startup now does a synchronous trading-ready warmup for
  symbols with positions/open orders, then schedules broad approved-coin warmup as a cancellable
  background maintainer. `live.defer_broad_candle_warmup=false` keeps legacy blocking warmup.
- [x] Made staged authoritative refresh the default for all exchanges while keeping explicit
  `live.authoritative_refresh_mode=legacy` as a temporary opt-out on this development branch.
- [x] Added `passivbot tool ticker-endpoint-probe` for multi-user CCXT endpoint timing probes.
  The JSON output covers ticker variants, bids/asks where available, order book, 1m OHLCV tail
  current-minute behavior, market metadata, and read-only private account-state endpoint timings.
- [x] Improved market snapshot remote-call economy: bulk ticker responses now populate the cache
  for all valid returned symbols, not only the symbols requested by the immediate caller.
- [x] Added market snapshot fetch coalescing so concurrent cache misses share one in-flight
  `fetch_tickers()` request instead of stampeding multiple bulk ticker calls.
- [x] Added explicit market snapshot ticker strategies: broad `fetch_tickers()` remains the default,
  Hyperliquid keeps its custom `allMids` bulk path, and Bitget defaults to
  `fetch_tickers(symbols)` because the first VPS probe showed Bitget's broad CCXT response missed
  requested USDC perp symbols while the symbol-list endpoint was fast and complete.
- [x] Batched position-change display price lookups so startup or manual-position transitions fetch
  one market snapshot cohort instead of one ticker request per changed position.
- [x] Extended the fake-live validation harness for staged remote-call economy: fake exchange
  request logs now include order writes, scenario assertions can validate request-count paths, and
  staged-vs-legacy comparison reports include remote-call summaries and per-method deltas.
- [x] Tightened staged planning freshness: completed candles are stamped only after exact
  required-symbol coverage is verified, planning universe preparation now happens before market
  refresh, market snapshots are revalidated immediately before order creation, the old
  price-distance order gate was removed in favor of replacement tolerance, and EMA gating
  diagnostics now use the correct PB mode shape.
- [x] Added a formal staged `PlanningSnapshot` contract capturing the exact symbols, ledger
  surface stamps, candle signature, market snapshots, and max ticker age used for a Rust planning
  call. The snapshot is validated immediately before payload construction and attached to
  debug/fake-live return snapshots for review.
- [x] Split planning-universe preparation from derived market-state refresh in the live loop.
  The loop now builds the final symbol universe before refreshing completed candles, and trailing
  data is recomputed only after candle freshness succeeds.

## Initial Ticker Probe Findings

Lightweight single-user probe command shape:

```bash
passivbot tool ticker-probe --user {user} --coins BTC,ETH,XMR --quote USDT --all --order-book --json
```

Multi-user endpoint timing probe command shape for VPS validation:

```bash
passivbot tool ticker-endpoint-probe --users bybit_01,hyperliquid_01 --coins BTC,ETH,SOL --quote USDT --repeats 2 --sleep-between-seconds 5 --out tmp/ticker_endpoint_probe.json
```

Bybit USDT perps with `ebybitsub03`:

- `fetch_ticker(symbol)` works for tested symbols and returns bid, ask, and last.
- `fetch_tickers(symbols)` works for tested symbols and returned the requested symbols in about 1.3s.
- `fetch_tickers()` works and returned the full swap ticker set in about 1.3s.
- Ticker payloads did not expose a top-level timestamp/datetime in the tested CCXT response.
- `fetch_order_book(symbol, limit=5)` works, is per-symbol, and returned timestamped top-of-book data in about 0.5s per tested symbol.

Hyperliquid USDC perps with `hyperliquid_01`:

- `fetch_ticker(symbol)` works for tested symbols and returns bid, ask, and last, but was slow at about 12-13s per symbol in this probe.
- `fetch_tickers(symbols)` timed out once during a possible local network/VPN interruption, then succeeded on retry in about 13.8s for three symbols.
- `fetch_tickers()` works and returned the broad ticker set in about 12s.
- Ticker payloads did not expose a top-level timestamp/datetime in the tested CCXT response.
- `fetch_order_book(symbol, limit=5)` works, is timestamped, and was much faster than ticker calls at about 0.3-1.2s per tested symbol.

## Current Known Gaps

- Ticker endpoint probing is now systematic in tooling, but still needs VPS results for Bitget,
  Gate.io, OKX, Binance, and KuCoin before choosing durable per-exchange ticker strategies.
- Duplicate-order guardrail covers disappeared self-orders; broader stale-state scenarios still
  need fake-live/staged comparison coverage.
- Startup/warmup is split into trading-ready and background work, but account-ready,
  market-ready, active-candle-ready, and full-warmup-ready timing diagnostics are still incomplete.
- Bybit DEBUG smoke still hit OHLCV rate-limit/availability warnings during broad candle/EMA
  work, even though market snapshots were healthy.
- Shutdown responsiveness during candle/EMA warmup is verified for Bybit warmup jitter; further
  exchange smoke tests should still include Ctrl-C during long startup/warmup phases.
- Legacy authoritative refresh still exists as a temporary opt-out on this branch; final merge
  target is staged-only with legacy path removed after broader exchange validation.

## Update Policy

Update this checklist as implementation proceeds. Mark items done only when code and targeted tests
exist, or when a live probe/smoke result has been recorded and reviewed.
