# Historical risk and unstuck audit notes

These retained non-HSL audit notes record earlier design work; current contracts
and tests take precedence. The superseded HSL sections have been removed. See the
[current HSL guide](../equity_hard_stop_loss.md) and
[HSL contract](../ai/features/equity_hard_stop_loss.md).

### A1.1 - Unstuck Min-Qty Loss-Allowance Overshoot

Plan: retain current behavior and document it as intentional.

Contract: unstuck allowance is a pacing budget, not a hard per-order loss cap
after exchange min qty/cost snapping. If remaining unstuck allowance is small
but exchange min qty/cost forces a larger unstuck close, allow the close. The
self-healing mechanism is that further unstucking is blocked until realized
profits rebuild positive allowance.

Exception: `max_realized_loss_pct` still blocks non-panic lossy closes if the
order would exceed the configured realized-loss limit.

### A1.2 - Risk Gates Fail Open On Invalid Inputs

Plan: code fix.

Required risk inputs should be validated at the Rust/orchestrator boundary and
fail loudly. Do not rely on `log::error!` inside individual calculators as the
safety mechanism.

Implemented: Rust orchestrator core and PyO3 JSON entrypoints now reject
invalid account/risk globals before realized-loss or unstuck gates can
silently skip. Coverage includes non-positive raw balance and realized-PnL
peak/current inconsistencies.

### A2.3 - Bounded `we_excess` Invalid Base/TWEL

Plan: code fix.

Bounded mode should not degrade to raw semantics when base/TWEL is invalid.
Return zero for explicitly disabled/no-budget contexts and fail validation for
active configs with invalid limits.

Implemented: Rust bounded `we_excess` now returns zero allowed exposure for
non-positive/non-finite base WEL and zero excess headroom for
non-positive/non-finite TWEL instead of falling back to the raw excess
percentage. Excess allowance is now bounded in every supported configuration.

### A3.1 - Reducer Stacking

Plan: Rust-side code fix.

Only one panic/auto-reduce/auto-unstuck/other-close reducer should be emitted
per coin+pside per ideal-order batch.

Priority:

1. HSL panic. Panic orders may be market orders and must take absolute
   priority.
2. WEL/TWEL enforcer reducer, using existing deterministic priority if both
   could apply. These are expected to be emitted at bid/ask as limit orders.
3. Auto-unstuck. Despite the EMA gate, emitted unstuck orders should be at
   bid/ask; if the EMA gate prevents a bid/ask-safe candidate, no unstuck order
   should be scheduled.
4. Any other close order, lossy or profitable, trailing/grid/other. If a grid
   wants multiple closes, the existing close-order ordering should choose the
   one more likely to fill first.

Auto-unstuck remains one order per pside, but it should not stack on the same
coin+pside as a WEL/TWEL reducer or panic order in the same batch.

Reasoning: the "only one lossy order" policy is safe only when the competing
orders are all bid/ask-reachable. If one close is farther from market than
another, prioritizing it ahead of a bid/ask unstuck reducer can block the more
useful reduction.

### A3.2 - Snapped Vs Raw Balance

Plan: design/tests before code.

Recommendation: keep both raw and snapped balance, but define the balance basis
per mechanism and add near-boundary parity/oscillation tests. Avoid ad hoc
special cases. Snapped balance is appropriate where hysteresis is intentionally
part of sizing/gating; raw balance is appropriate where exact portfolio
exposure repair is intended.

Do not add a secondary raw-exposure cap for now. Add docs/tests around the
current snapped/raw separation, and emit a warning/preflight note when the
configured hysteresis/snap pct is high, for example above roughly 5%, because
unexpected boundary behavior becomes more likely.

### A3.3 - `reduce_overweight` Dynamic WEL

Plan: docs/test first.

Contract: `reduce_overweight` should use the dynamic currently-tradable slot
count, not only configured `n_positions`. Add tests that cover shrinking and
expanding currently-tradable universes.

### A4 - Entry Cooldown Semantics

Plan: simplify config surface.

Use `entry_cooldown_minutes` as the single entry-ladder/cooldown control.

Contract:

- If `entry_cooldown_minutes == 0.0` and
  `bot.{pside}.entry.retracement_base_pct <= 0.0`, cooldown is disabled and
  full simultaneous entry ladders may be emitted.
- If `entry_cooldown_minutes == 0.0` and
  `bot.{pside}.entry.retracement_base_pct > 0.0`, allow only one entry order on
  the book at a time because the next entry price depends on threshold and
  retracement state.
- If `entry_cooldown_minutes > 0.0`, stage at most one position-adding entry
  order and block any entry for coin+pside whose previous entry fill happened
  less than the configured duration ago, including fractional sub-minute
  durations.
- Backtests evaluate on one-minute steps, so any positive sub-minute cooldown
  prevents same-minute replacement/add and effectively waits until the next
  backtest decision minute. Live trading checks intra-minute and enforces the
  actual millisecond duration.

Keep float logic internally to support possible future sub-minute backtests.

### B1.1 - Live Omits TWEL Enforcer Policy

Plan: code fix.

Add `risk_twel_enforcer_policy` to the live Python -> Rust payload and add
BotParams coverage tests.

### B1.3 - Live-Only Unstuck Admission Logic

Plan: move authority to Rust and simplify Python.

Rust should emit ideal unstuck orders. Python should reconcile them through the
normal create/cancel pipeline and replacement tolerances. Remove bespoke Python
unstuck-specific order juggling where stronger general duplicate-order
guardrails make it unnecessary.

### B1.4 - Balance Hysteresis Parity

Plan: same direction as B1.3.

Move shared hysteresis policy to Rust where feasible. Document any intentional
live/backtest divergence that remains.

### B2.2 - Realized-Loss Gate Inherits Account History

Plan: retain current contract and document clearly.

`max_realized_loss_pct` considers all fill events inside the lookback window,
with no exception except panic orders. Users can reduce lookback or increase the
loss allowance if desired.

### B2.3 - Unstuck Allowance Inheritance

Plan: retain current contract and improve visibility.

Unstuck allowance considers all PnL from the configured lookback window. The
event stream should show the lookback window and allowance basis.

### B2.9 - Trailing Anchor Fallback

Plan: derive from fill events.

The last fill timestamp for coin+pside should be derived from fill events. If
no fill event exists inside available runtime fill history, log a warning and
use the oldest available candle timestamp as trailing extrema anchor. Local
metadata/cache may speed lookup but must not be authoritative.

### B2.10 - Entry Cooldown At Boot

Plan: fill-history based, with explicit fallback.

If no entry fill event is found for coin+pside inside available history, use the
lookback/covered-start timestamp as the cooldown anchor and log it clearly.

Do not add a dedicated entry-cooldown history window for now. The bot may keep
more fill/candle data in memory or local caches than the configured trading
lookback, for example up to seven days, but trading decisions must only use the
allowed lookback contract.

### B3.4 - Residual Neutral Defaults On Risk Inputs

Plan: code hygiene.

Replace trading-risk neutral defaults with explicit errors once tests and stubs
provide required managers. Test-only no-manager paths should be isolated.

### C6 - Entry Cooldown Docs

Plan: fold into A4 docs/redesign.
