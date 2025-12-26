# Hedging Overlay (Market-Neutral, WIP)

Passivbot includes a **market-neutral hedging overlay** (work in progress) intended to keep the bot’s **net market exposure near zero** by opening hedge positions that offset the exposure of the base strategy.

This is a *reactive* overlay: it responds to changes in base exposure (position size changes), and does not attempt to predict price direction.

For the current specification, see `ideas/market_neutral_hedging.md`. The source of truth is the Rust implementation in `passivbot-rust/src/hedge.rs` and its orchestrator integration in `passivbot-rust/src/orchestrator.rs`.

## What It Does (High Level)

The overlay supports two symmetric modes:

- `hedge_shorts_for_longs`: base runs long-only; hedges are shorts.
- `hedge_longs_for_shorts`: base runs short-only; hedges are longs.

At each cycle, the hedger:

1. Measures current **base exposure** and **hedge exposure**.
2. Computes a target hedge exposure: `target_hedge = gross_base * hedge.threshold`.
3. If the hedge is outside the tolerance band, it rebalances by:
   - reducing hedges (closing least-underwater hedges first), or
   - increasing hedges (opening minimum-size hedges across eligible symbols, then adding to the most-underwater hedge in chunks).

## Signed Quantity and Position Conventions

Throughout passivbot (including the hedger):

- Buy qty is positive, sell qty is negative.
- Long position size is positive, short position size is negative.

Only the final live order payload (when an exchange requires `abs(qty)`) may use absolute quantities.

## One-Way Constraint (No Long+Short on Same Symbol)

The hedging overlay is designed for **one-way mode** (net position per symbol). That means **you must not hold a long and a short simultaneously on the same symbol**.

To enforce this, the hedger and orchestrator implement collision handling:

- If the base wants to open/increase on a symbol that currently has an opposite-direction hedge, the hedger emits a **hedge close** and signals the orchestrator to **gate** the base entry for that symbol for this cycle.
- On the next cycle (after the close fills), the base entry is allowed again.

This is the “base side priority” rule: base entries win, hedges must get out of the way.

## Exposure Model (Important)

The hedger uses **wallet exposure** (dimensionless fraction of balance).

- Notional: `abs(qty) * pprice * c_mult`
- Exposure: `notional / balance`

Like the rest of passivbot, exposure uses **position price (`pprice`, avg entry price)** rather than the current market price. As a result:

- Pure price movement does not change exposure values.
- Hedge rebalancing is primarily triggered by **position size changes** (entries/closes) rather than mark-to-market swings.

Market price is used for **underwaterness ranking**, not for exposure sizing.

## How the Hedger Chooses Symbols

Hedges may only be opened on an approved universe. In the current Python plumbing this is derived from:

- `live.approved_coins.short` (mapped to the backtest/live symbol universe)

Eligible hedge symbols are ranked with a simple **Borda count** that prefers:

- higher volume, and
- lower volatility

The scores come from the same EMA inputs used for coin selection (see `passivbot-rust/src/orchestrator.rs`).

## Rebalancing Behavior

### Tolerance Band

To avoid churn, no hedge rebalance orders are emitted if the hedge is “close enough”:

`abs(gross_hedge - target_hedge) <= base_twel * hedge.tolerance_pct`

This tolerance band is absolute exposure (it scales with base TWEL).

### Increasing Hedge Exposure (Under-Hedged)

When the hedge is too small:

1. **Bootstrap:** open minimum-size hedge positions on eligible symbols, up to `hedge.max_n_positions` (or base `n_positions` if `0`).
2. **Allocate:** if still under-hedged, add to the **most underwater** hedge position in chunked steps to reduce flip-flopping:
   - each step spends at least `hedge.allocation_min_fraction` of the remaining budget (subject to min cost/min qty and per-position cap).

### Decreasing Hedge Exposure (Over-Hedged)

When the hedge is too large:

- Fully close hedge positions starting from the **least underwater** until within tolerance.

### Per-Position Cap

Each hedge position is capped (in exposure terms) to avoid concentrating the hedge on one symbol:

`cap_exposure = (base_twel * hedge.threshold / max_positions) * (1 + hedge.hedge_excess_allowance_pct)`

## Order Placement

Hedge orders are deterministic maker-style limits:

- Open/increase hedge short: place at best ask.
- Close hedge short: place at best bid.
- (Symmetric for hedge longs.)

In backtesting, bid/ask may be equal (e.g. `candle_1m.close`), but the hedger keeps bid/ask separate.

## Enabling in Backtests (Example)

Example (base long-only + hedge shorts):

```json
"bot": {
  "long": { "n_positions": 10, "total_wallet_exposure_limit": 1.7 },
  "short": { "n_positions": 0, "total_wallet_exposure_limit": 0.0 }
},
"hedge": {
  "threshold": 1.0,
  "tolerance_pct": 0.05,
  "hedge_excess_allowance_pct": 0.2,
  "max_n_positions": 0,
  "allocation_min_fraction": 0.1,
  "mode": "hedge_shorts_for_longs",
  "one_way": true
}
```

Notes:

- Hedging is enabled by setting `hedge.threshold > 0.0`.
- The orchestrator enforces that the opposite base side is disabled for v0 (e.g. base short disabled in `hedge_shorts_for_longs`).

## Current Limitations (WIP)

- This feature is intended for high-fidelity backtesting first; live behavior is not considered production-ready yet.
- The approved hedge universe is currently derived from `live.approved_coins.short` for both modes (subject to change as the spec evolves).
- Exposure uses `pprice` (avg entry price), not current market price; this is consistent with passivbot’s existing exposure model but can surprise users expecting mark-to-market neutrality.
