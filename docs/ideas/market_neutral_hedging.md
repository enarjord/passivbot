# Market-Neutral Hedging Overlay (WIP Spec)

This document defines the *specification* for a market-neutral hedging overlay in passivbot.

- Reference WIP implementation: `passivbot-rust/src/hedge.rs`

## Status and Scope

- Status: WIP spec to “nail down” behavior before wiring into orchestrator/backtester.
- Primary scope (v0): **base long-only strategy + reactive hedge shorts**, in **one-way position mode** (no simultaneous long+short on the same symbol).
- Secondary scope (v0): symmetric **base short-only + reactive hedge longs** (same machinery, inverted directions).
- Out of scope (v0): running base longs and base shorts simultaneously *and* adding an additional hedging overlay on top.

## Design Principles (from `passivbot_agent_principles.yaml`)

- Rust is the source of truth; backtester and live must share the same Rust logic.
- Hedge computation is deterministic/pure: same inputs → same outputs; no hidden state; identical behavior after restart.
- Observe exchange constraints:
  - Entries observe effective min qty and effective min cost.
  - Closes observe effective min cost too; only exception: if remaining pos size is below effective min qty, close the full remaining size.
- Fail loudly for invalid config or missing required inputs; do not silently compensate for bad data.

## Terminology and Definitions

### Sides

- **pos_side**: long or short (position direction)
- **order_side**: buy or sell (order direction)

### Prices

Passivbot does not use perpetual futures “mark price” (oracle price). The hedging overlay uses:

- **bid**: current highest bid (in backtest currently: `candle_1m.close`)
- **ask**: current lowest ask (in backtest currently: `candle_1m.close`)
- **market_price**: default mid price `(bid + ask) / 2` (often equal to bid/ask in current passivbot feeds)

Even if the current data feed sets `bid == ask == candle_close`, the hedging logic should keep bid/ask separate to preserve the capability to handle differing values later.

### Exposure

All exposure values are **dimensionless fractions of wallet balance**.

- **Notional**: `notional = abs(qty) * price * c_mult`
- **Gross exposure** for a set of positions: `gross_exposure = Σ(notional) / balance`
- **gross_base**: gross exposure of base positions (e.g. longs in long-only base mode)
- **gross_hedge**: gross exposure of hedge positions (abs; e.g. shorts in long-only base mode)
- **net_exposure**: `gross_base - gross_hedge` (positive = net-long, negative = net-short)

Exposure in passivbot is computed using **position price** (pprice/average entry price), not market price. Exposure therefore does not vary with price action, only with position size changes.

### Underwaterness (for hedge position selection)

“Underwater” means “currently losing” (larger is worse).

- For a **long** position: `underwater = 1.0 - market_price / pprice`
- For a **short** position: `underwater = market_price / pprice - 1.0`

Notes:
- `pprice` is the position’s average entry price.
- `market_price` should use the same definition as `calc_pprice_diff_int` in `passivbot-rust` (`utils.rs`).

## Feature Goal and Non-Goals

### Goal

Maintain market neutrality by keeping gross hedge exposure approximately equal to gross base exposure:

- `gross_hedge ≈ gross_base` within a configurable tolerance band
- Hedge positions are **purely reactive** and **must not interfere** with base strategy objectives beyond the one-way collision rule.

### Non-goals (v0)

- Make money on the hedge leg (expected ≈ break-even after costs).
- Predictive hedging or signal-driven discretionary hedges.
- Cross-exchange or multi-account hedging.

## Invariants (Must Always Hold)

1. One-way constraint: at no time may there be both a long and a short open on the same symbol.
2. Long priority rule (in long-only base mode):
   - If the base wants to open/increase a long on a symbol that currently has a hedge short, the hedge short is closed first and the base order is deferred until the short is closed.
3. Hedge universe restriction:
   - Hedge shorts may only be opened on symbols in `approved_coins.short` (config).
4. Deterministic behavior:
   - Given the same inputs (positions, prices, exchange params, config), the hedge module must produce the same output orders.

## Configuration (Proposed; final naming TBD)

Minimum required configuration for v0:

- `hedge.threshold: float >= 0.0` (0 disables hedging; 1 targets equal hedge/base exposure; 0.5 targets 50% hedge vs base; 1.5 targets 150%, etc.)
- `hedge.mode: "hedge_shorts_for_longs" | "hedge_longs_for_shorts"`
- `hedge.tolerance_pct: float` (see “Tolerance Band”)
- `hedge.hedge_excess_allowance: float` (cap looseness; see “Per-Position Cap”)
- `hedge.max_n_positions: int` (if 0, default to `bot.main_pside.max_n_positions`)
- `hedge.allocation_min_fraction: float` (see “Equalization / Allocation”)
- `live.approved_coins.short: [symbol...]` (eligible hedge shorts)
- `exchange.one_way_mode: bool` (v0 requires `true`; if `false` later, collision policy changes)

Default recommendation:
- `hedge.max_n_positions = bot.main_pside.max_n_positions` in the corresponding base mode.

## Inputs and Outputs (Rust API Level)

The hedge module must be callable from both backtester and live:

### Inputs

- `positions_long`: all open long positions (idx/symbol, signed size, pprice, etc.)
- `positions_short`: all open short positions (idx/symbol, signed size, pprice, etc.)
- `balance`: wallet balance used for exposure normalization
- Market data per symbol used by hedges: `bid`, `ask` (and derived `market_price`)
- Exchange params per symbol: `qty_step`, `min_qty`, `min_cost`, `c_mult`
- `desired_base_orders`: base strategy’s desired orders for this cycle (at minimum: symbol indices involved), so the hedger can:
  - avoid opening new hedge positions on symbols scheduled for base initial entries (one-way mode)
  - close existing hedge positions that collide with base initial entries (long priority)
- `one_way`: whether collisions must be resolved by closing and deferring
- `base_twel`: total wallet exposure limit for the base side (used for tolerance and caps)

### Outputs

- `hedge_orders`: desired hedge orders (idx/symbol, signed qty, price, action, reason code)

## Algorithm (v0)

### 1) Compute current exposures

- In `hedge_shorts_for_longs` mode:
  - `gross_base = gross_exposure(positions_long)`
  - `gross_hedge = gross_exposure(positions_short)`
- In `hedge_longs_for_shorts` mode: symmetric.

Exposure uses position price (pprice/avg entry), as elsewhere in passivbot.

- Define `net_exposure = gross_base - gross_hedge`.

### 2) Tolerance band (to avoid churn)

Define:

- `target_hedge = gross_base * hedge.threshold`
- `tolerance_band = base_twel * hedge.tolerance_pct` (absolute exposure band)

If `abs(gross_hedge - target_hedge) <= tolerance_band`, emit **no hedge rebalance orders**.

If `hedge.threshold == 0.0`, then `target_hedge == 0.0` and the logic reduces existing hedges toward zero (within tolerance).

### 3) One-way collision handling (long priority)

If `one_way == true`:

- For each base desired order that would create/increase a base position on symbol `S`:
  - If there exists an opposite-direction hedge position on `S`:
    - Emit a hedge close order for the full hedge position on `S` (subject to close rounding rules).
    - Do not emit any new hedge open/increase orders on `S` in the same cycle.

Orchestrator responsibility:

- Gate/cancel base initial entry orders on any symbol which currently has an opposite-direction hedge position (one-way constraint).
- There is no need for the hedge module to return a separate “defer base” list; the orchestrator can infer deferral from the presence of a hedge position on the symbol.

### 4) Hedge coin selection

Hedge candidates are restricted to `approved_coins.short` (for hedge shorts) or the symmetric list for hedge longs.

Selection priority:

1. Open hedge positions on **as many distinct eligible symbols as possible** (diversify), constrained by:
   - `hedge.max_n_positions`
   - effective min cost / min qty (per symbol)
2. If all hedge slots are filled (or no additional symbol can be opened due to min constraints), allocate additional hedge notional to existing hedges by “underwaterness”.

Ranking:
- Default: Borda count combining low volatility (prefer lower) + high volume (prefer higher).
- Inputs required: per-symbol `volatility_score` and `volume_score` (definition and computation source must be specified; see “Open Decisions”).

### 5) Increasing hedge exposure (net-long in base long-only mode)

When `gross_hedge < target_hedge - tolerance_band`:

1. **Bootstrap phase**: open minimum-sized hedge positions on eligible symbols (ranked order) until:
   - reaching `hedge.max_n_positions`, or
   - hedge exposure reaches `target_hedge` within tolerance, or
   - remaining hedge allowance cannot satisfy min_cost on any additional symbol.
2. **Allocation phase**: if more hedge exposure is still needed:
   - Allocate additional hedge notional to the **most underwater** hedge position first.
   - When adding to the most underwater hedge, target “equalization” against the second-most underwater hedge, but enforce a minimum per-allocation spend to avoid flip-flopping churn:
     - Define `allocation_chunk = hedge.allocation_min_fraction * remaining_hedge_budget_notional` (clamped by min_cost/min_qty constraints).
     - Spend at least one `allocation_chunk` on the currently selected most-underwater hedge before re-sorting by underwaterness.
   - Repeat until hedge allowance is spent or neutrality is reached.

### 6) Decreasing hedge exposure (net-short in base long-only mode)

When `gross_hedge > target_hedge + tolerance_band`:

- Close hedge positions starting with the **least underwater**.
- Once a hedge position is selected for trimming, **fully close it** (subject to close rounding rules), then move to the next least underwater.

Rationale: fully closing resets `pprice` and allows the selection algorithm to re-pick a better symbol later.

### 7) Per-position cap

Cap each hedge position to:

`cap_exposure = (base_twel * hedge.threshold / max_n_allowed_hedges) * (1 + hedge_excess_allowance)`

This corresponds to “Option B” (cap based on configured maximum exposure) and should be implemented first; later, Option A (cap based on `gross_base`) can be tested for performance/behavior.

## Order Pricing (Backtestable)

Order pricing must be explicit to keep backtesting faithful.

Default (v0) recommendation: place hedge orders as maker limits at best bid/ask:

- Hedge short open/increase: `order_side = sell`, `price = best_ask` (maker)
- Hedge short close: `order_side = buy`, `price = best_bid` (maker)
- Symmetric for hedge longs

## Rounding and Exchange Constraints

Hedge orders must obey the same rounding principles as the rest of passivbot:

- Entry hedge orders:
  - `qty` rounded to `qty_step` and satisfies `min_qty`
  - `notional` satisfies `min_cost`
- Close hedge orders:
  - observe effective min cost, except:
    - if remaining pos size is below effective min qty, close full remaining size

## Backtesting Requirements (High Fidelity)

- The hedge overlay must run inside the same Rust “compute ideal orders” pipeline as live.
- Hedge orders must be handled by the same fill simulation as other orders.
- Collision deferrals must be visible to the orchestrator so base orders are not emitted as executable in the same cycle.
- Each hedge order includes a stable `reason` code (e.g. `rebalance_add`, `rebalance_reduce`, `collision_with_base`).

## Open Decisions / Ambiguities (Need Resolution)

1. **Effective min cost handling in slot bootstrap**
   - Unresolved detail: implement per-symbol feasibility (min_cost/min_qty) in bootstrap vs pre-computing a global slot count.
   - Recommendation: bootstrap in ranked order and check each symbol’s *effective* min cost/qty directly (reusing existing orchestrator logic), rather than deriving slot count from a single symbol.

2. **Graceful stop vs immediate close when a hedge symbol becomes ineligible**
   - Scenario A: exchange delists a hedge symbol with an open hedge position.
   - Scenario B: user removes a symbol from `approved_coins.short` while a hedge is open.
   - Recommendation: treat delist as “force close as soon as possible”; treat user removal as “graceful stop” (no new opens; manage/close naturally), unless user explicitly requests force close.

3. **Default values for `hedge.allocation_min_fraction`**
   - Unresolved: choose a default (example discussed: 0.10) and define clamping behavior for tiny budgets where min_cost dominates.
   - Recommendation: start with default 0.10 and clamp `allocation_chunk` to at least the symbol’s effective min_cost where possible.

4. **Maker vs taker in live**
   - Backtest cannot simulate maker/taker behavior with 1m candles; live has `config.live.market_orders_allowed`.
   - Recommendation: spec v0 as emitting limit orders at bid/ask; live execution policy decides whether to convert some to market orders, but keep hedger output deterministic.

## Notes

You wrote:
- `market_price` should use the same definition as `calc_pprice_diff_int` in `passivbot-rust` (`utils.rs`).

What do you mean that `market_price` should use same definition as `calc_pprice_diff_int`? `market_price` is simply candle.close, or mid == (bid + ask) / 2.


"""
Orchestrator responsibility:

- Gate/cancel base initial entry orders on any symbol which currently has an opposite-direction hedge position (one-way constraint).
- There is no need for the hedge module to return a separate “defer base” list; the orchestrator can infer deferral from the presence of a hedge position on the symbol.
"""

This gating done by orchestrator should be named such that it is clear it belongs to the hedge logic. Just a separation between the generation of the ideal hedge orders and the gating afterwards. If no hedging, none of this gating


1. **Effective min cost handling in slot bootstrap**
follow recommendation.


2. **Graceful stop vs immediate close when a hedge symbol becomes ineligible**
follow recommendation.


3. **Default values for `hedge.allocation_min_fraction`**
follow recommendation.

4. **Maker vs taker in live**
yes, live bot deals with limit/market. backtest assumes limit only.