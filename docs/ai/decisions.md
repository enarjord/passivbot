# Architectural Decision Log

This document records significant architectural decisions with context and rationale.

---

## 2025-Q4: Rust as Source of Truth for Order Logic

**Context**: Order calculation logic existed in both Python and Rust, leading to divergence between backtester and live bot behavior.

**Options Considered**:
1. Maintain parallel implementations (Python for prototyping, Rust for production)
2. Python-only implementation (simpler but slower)
3. Rust-only implementation (single source of truth)

**Decision**: Rust is the canonical implementation. All order calculation logic lives in `passivbot-rust/src/`. Python is used only for orchestration, exchange communication, and data management.

**Consequences**:
- Backtester and live bot use identical order logic
- Changes to order behavior must be made in Rust
- Python patches to order calculation are not acceptable
- Requires Rust knowledge to modify core trading logic

**Code Reference**: `passivbot-rust/src/orchestrator.rs`

---

## 2026-01: HIP-3 Isolated Margin Requirement

**Context**: Adding support for Hyperliquid stock perpetuals (HIP-3 markets) which have different margin requirements than standard crypto perps.

**Options Considered**:
1. Treat all Hyperliquid markets the same (cross margin)
2. Auto-detect market type and set appropriate margin mode
3. Require explicit user configuration for margin mode

**Decision**: Auto-detect HIP-3/stock perps and automatically set isolated margin mode. These markets only support isolated margin with 10x max leverage.

**Consequences**:
- Stock perps work out-of-the-box without special user configuration
- Users mixing crypto + stock perps in same config get correct margin modes automatically
- Balance calculations differ (isolated margin locks capital per-position)
- Must filter stock perp symbols to Hyperliquid only (other exchanges don't support them)

**Code Reference**: `src/exchanges/hyperliquid.py:set_margin_mode()`

---

## 2025-Q3: Stateless Design Principle

**Context**: Bot restarts would sometimes cause different behavior due to reliance on in-memory state from previous runs.

**Options Considered**:
1. Persist all runtime state to disk
2. Design for statelessness (rederive everything from exchange state)
3. Hybrid approach with explicit state vs cache distinction

**Decision**: Stateless design - behavior must be identical after restart. Never rely on "what happened earlier" unless it can be rederived from exchange/state snapshot on startup.

**Consequences**:
- No local ad-hoc caches that change behavior (only performance-only caches allowed)
- Bot can be restarted at any time without behavior change
- Easier to reason about and debug
- Cannot implement certain optimizations that require historical context

**Code Reference**: Documented in `principles.yaml`

---

## 2026-01: Hybrid Pagination for Bybit Closed-PnL

**Context**: Bybit's `/v5/position/closed-pnl` endpoint has two pagination mechanisms with different coverage and limitations.

**Options Considered**:
1. Cursor pagination only (only covers ~7 days)
2. Time-based pagination only (misses records when >100 per window)
3. Hybrid approach combining both

**Decision**: Hybrid pagination - use cursor for recent data (no gaps), then time-based sliding window for older data, with deduplication by orderId.

**Consequences**:
- Full historical coverage without gaps
- More complex implementation
- Requires deduplication logic
- Better PnL accuracy for close fills

**Code Reference**: `src/fill_events_manager.py:BybitFetcher._fetch_positions_history()`
**Case Study**: See `debugging_case_studies.md` for full investigation

---

## 2026-01: Exchange Fetch Methods Must Not Catch Exceptions

**Context**: Exchange fetch methods (`fetch_balance`, `fetch_positions`, etc.) were catching exceptions internally, leading to unclear return types and hidden errors.

**Options Considered**:
1. Catch and return sentinel values (e.g., `False`, `None`)
2. Catch and re-raise with context
3. Let exceptions propagate to caller

**Decision**: Fetch methods must NOT catch exceptions. Let them propagate to caller who handles via `restart_bot_on_too_many_errors()`.

**Consequences**:
- Clean return types (no `Union[list, bool]`)
- Preserved exception context and stack traces
- Errors impossible to ignore
- Caller responsible for error handling policy

**Code Reference**: Documented in `principles.yaml:error_handling`

---

## Template for New Decisions

```markdown
## [Date]: [Decision Title]

**Context**: What prompted this decision?

**Options Considered**:
1. Option A
2. Option B
3. Option C

**Decision**: What we chose.

**Consequences**:
- Impact 1
- Impact 2
- Impact 3

**Code Reference**: `file.py:function_name()`
```
