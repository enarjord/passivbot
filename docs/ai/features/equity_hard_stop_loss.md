# Equity Hard Stop Loss contract

HSL uses one Rust-owned, best-effort reconstruction and controller across live,
backtest and optimizer paths. Python acquires facts and enforces execution admission;
it does not own a second drawdown formula or retain an earlier panic decision.
The [user guide](../../equity_hard_stop_loss.md) defines configuration and examples.

## Signal authority

- Coin has one controller per symbol/position side; pside has one per side;
  unified has one portfolio controller with an explicit `bot.hsl` policy.
- Reconstruct retained net realized cashflows and UPNL, anchor to current equity
  (balance budget plus current UPNL), and measure drawdown from the episode peak.
  EMA smooths raw drawdown, seeded by the first raw sample. Fractional spans remain floats.
- Current RED means `min(raw_drawdown, ema_drawdown) > red_threshold`. Historical
  breaches alone have no authority. GREEN immediately retires outstanding panic intent.
- A partial close retains the current episode; its updated current signal decides
  whether to continue closing. There are no retained RED commitments or emergency journals.
- Rust's reconciler is the sole authority for estimated history and episode boundaries.
  Missing or ambiguous historical facts produce bounded documented approximations and
  diagnostics, not a correctness-proof barrier to a current decision.
- With no useful history, current size, basis, mark and budget define the single-sample
  loss signal. This is part of the same evaluator, not a separately timed UPNL fallback.
  Missing/unusable current facts remain unavailable; they are not invented.

## Episodes and lifecycle

Only the current or latest completed episode matters. Current exchange positions are
flatness authority, including when the final fill is delayed or absent. Only a RED
terminal accounting sample, including final PnL/fees, starts cooldown; order type is
irrelevant. Use the reconstructed final timestamp, or the latest retained causal fill
for an estimated missing close. Repeated observations do not renew the timestamp.
No retained causal fill means no historical cooldown.

Any renewed exposure clears preceding cooldown. `always` permits restart after cooldown;
`never` restricts restart only while the terminal stop remains within lookback. Retained
history, budget changes or corrected evidence can change terminal RED and thus remaining
cooldown. Everything outside configured lookback is forgotten. No local artifact can
preserve an otherwise unreconstructible trading decision after restart.

## Inputs and execution

Enabled HSL requires explicit supported restart policy and 1–90 days of lookback.
Coin budgets divide current raw balance by applicable configured slots; inactive zero-slot
sides do not invent a divisor. Aggregate budgets use raw balance. TWEL does not scale HSL.

Use minute closes in live and backtest reconstruction. Historical coarser candles use the
shared deterministic OHLC expansion; remaining gaps forward-fill then backfill a missing
prefix. Current held exposure requires fresh usable marks, positions and balance.

Keep account/order/quote freshness, plan receipts, protection-first scheduling, shutdown
checks and connector admission. Position-to-fill settling is shared with all trading
and remains bounded; missing history cannot indefinitely lock that gate. A prior
permission is never a substitute for current facts before a write. Ordinary strategy,
unstucking and PnL consumers retain their own readiness contracts.

## Configuration and removal boundary

HSL has one implementation. Old configurations require migration and revalidation.
Retired tier/intervention/terminal-threshold controls cannot silently acquire a different
meaning; optimization over removed controls is rejected. Unified policy is never hydrated
from a side or hidden template. Legacy optimizer fitness and checkpoints are not evidence
for new signal semantics; reevaluate configurations.

## Code and validation

- `passivbot-rust/src/hsl_*`: factual reconciliation, signal and current controller.
- `passivbot-rust/src/backtest_hsl_*`: simulator integration, disposable caches and reporting.
- `src/live/hsl_*`: immutable observation, current execution admission and diagnostics.
- `src/live/position_fill_sync.py`: bounded shared position-to-fill confirmation.
- `src/config/hsl.py`: canonical policy/configuration validation.

Require shared reference/unit tests, source-verified native caller tests, offline fake-live
cycles, restart and history-repair cases, current RED recovery, terminal cooldown and
HSL-disabled trace parity. GPU values are screening estimates; exact Rust validation owns
retained candidates. Cache loss/rebuild must preserve intent. Performance acceptance compares
trading traces before timings and includes disabled HSL. Live trials require separate approval;
offline tests do not establish exchange execution correctness.
