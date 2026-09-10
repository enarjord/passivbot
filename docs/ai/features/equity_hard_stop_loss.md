# Equity Hard Stop Loss Episode Contract

HSL drawdown state is scoped by `live.hsl_signal_mode`:

| Mode | Episode scope | Episode ends when |
|---|---|---|
| `coin` | one `coin+pside` | that position is fully closed |
| `pside` | all positions on one `pside` | every position on that side is fully closed |
| `unified` | the whole account | every position is fully closed |

## Invariants

1. The drawdown tracker resets after every proven episode end. The next episode begins after the
   flattening fill timestamp.
2. A flattening fill ends the episode regardless of its order type or origin. Panic, take-profit,
   grid-close, manual, and external exchange fills have identical boundary semantics.
3. A RED-seen episode remains entry-blocked until its scope is confirmed flat. Its cooldown begins
   at the flattening fill, not at the RED sample, order submission, bot restart, or observation time.
4. Compact replay derives non-flat/flat transitions from fill events independently of candle or
   unrealized-PnL availability. Multiple boundaries inside one replay minute retain their exact
   fill order, realized PnL, fees, and account balance at each boundary. Missing price replay may
   defer drawdown evaluation, but it must not hide an episode boundary.
   Coin boundary balance reverses all account PnL/fees strictly after the boundary timestamp and
   the proven same-pair fill tail within that timestamp. Other pairs at the same timestamp remain
   included in the account timestamp cohort, matching the incremental live convention.
   Mixed-action fills sharing a millisecond require an unambiguous exchange-provided position
   chain; list order and locally reconstructed position annotations are not ordering evidence.
   Each proven fill boundary evaluates its final risk sample. Distinct boundaries in the same
   minute replace that minute's EMA sample from its prior baseline instead of advancing EMA time
   again. Ordinary polling within the minute remains cached. A RED stop is recorded while flat,
   before a later fill can reopen the scope; a RED-free reset seeds the next episode in that minute.
5. Current flat state is not a timestamp. If the flattening fill is not yet available, live
   finalization and cooldown anchoring defer visibly while protective entry blocking remains active;
   they never substitute the current time. Cooldown re-panic finalization must replay fills from a
   proven non-flat intervention snapshot; an entry or partial-close fill is not flatten evidence.
6. Restart reconstruction uses exchange state, fill/PnL history, candles where required, config, and
   current time. Local latch files are diagnostics, not authority. Restart always reconstructs from
   authoritative exchange-derived inputs; no persisted replay state participates in the decision.
   Live normal interventions and cooldown expiry use that same reconstruction before releasing a
   halt, retaining entry fees and losses before the next observation. A proven RED stop follows the
   same restart rules regardless of closing order type; terminal no-restart takes precedence.
   Same-millisecond normal interventions require the validated fill-chain order and use its
   cumulative PnL prefix so closing losses are excluded while entry fees survive subsequent polls.
7. `bot.{pside}.hsl.panic_close_order_type = "market"` is an explicit protective execution
   override when HSL is enabled. Rust may emit that side's `close_panic_*` as a market order even
   when `live.market_orders_allowed = false`; the live flag gates non-panic market execution and
   must not downgrade an explicitly configured HSL panic close to a limit order. The live producer
   boundary validates this panic execution choice in both directions against the submitted config;
   it must reject either a limit-for-market or market-for-limit mismatch as malformed Rust output.
8. For an open coin scope using `restart_after_red_policy=always`, a fill-proven current episode
   may discard older closed episodes after a flat gap longer than the configured RED cooldown.
   Replay retains preceding episodes while their cooldown horizons overlap the next episode, so a
   chain of possible cooldown interventions remains strict. An ambiguous fill sequence, a position
   size mismatch, `threshold`/`never`, or a missing current-episode boundary preserves full-lookback
   replay. The cumulative realized PnL of discarded episodes becomes the new replay baseline, so
   their gains, losses, and fees cannot affect the retained episode. Unavailable candles before the
   resulting boundary cannot strand an otherwise provable held episode; unavailable required
   candles at or after it still fail closed.
   The same aggregate boundary limits candle fetches, minute-grid allocation, panic markers, and
   replay events returned to the coin initializer. Sparse fills before the boundary remain input
   only long enough to seed exact balance and position state at the boundary; they are not expanded
   into replay rows or reconsidered as retained-episode events.
   Live fill-history readiness uses that same fill-derived boundary as its only held-episode owner
   and also proves every enabled side's flat-scope cooldown horizon. A recent fill for a currently
   flat pair may still own a RED cooldown and therefore preserves the full configured lookback.
   Ambiguous or delayed held evidence also preserves or restores the full requirement before fills
   become authoritative. PnL blockers are evaluated against each held pair's own canonical episode
   boundary; the aggregate earliest boundary exists only to fetch and prove coverage. Coin stop
   finalization consumes pair metrics and must not add an account-wide PnL dependency. Coin mode
   evaluates each configured coin's effective HSL enablement, restart policy, and cooldown.
   `threshold`, `never`, pside, and unified modes remain full-lookback strict.
9. Restart price reconstruction fetches 1m history first. When an exchange cannot provide the
   older leading portion, it may use 5m, then 15m, then 1h candles for that prefix. This is an
   explicitly approximate price path: the finest source wins and its contribution is reported.
   Coarser candles never repair missing rows at or after the first available 1m candle. Fill-based
   episode boundaries, realized PnL, and fees remain exact.

## Failure Semantics

Incomplete fill coverage follows `../error_contract.md`. A required episode boundary is unavailable
until supported by fill evidence. Startup replay validates all enabled scope tapes before replacing
existing protective state. The affected HSL scope remains protective and retries after an
authoritative refresh. Flat scopes pending startup price replay retain the existing per-pair
create gate, leaving unrelated scopes available. Ambiguous required held-episode evidence defers
ordinary shared-account planning: the startup gate runs after portfolio intent construction and
cannot make a plan built from unknown HSL episode state authoritative. Independently ready,
already-latched RED supervision and required panic protection for active cooldown positions still
run during that deferral, using fresh protective account state and the configured execution pacing.
Cancellation-only waves remove entries from terminal no-restart scopes and resting initials from
flat cooldown scopes without constructing new intent or changing terminal state. Manual ownership
begins only after a proven cooldown intervention and persists through
later flat observations; before that intervention, fresh initials remain blocked. If current fill
evidence cannot distinguish those cases, the cancellation wave refreshes the fill tail after its
account observation and preserves manual orders if proof remains unavailable. Graceful-stop
adds to held positions retain their policy semantics.

## Code And Tests

- Replay and live finalization: `src/passivbot_hsl.py`
- Live orchestration bindings: `src/passivbot.py`
- Coin replay and cooldown regressions: `tests/test_hsl_coin_mode.py`
- Pside/unified finalization coverage: `tests/test_unstucking_safeguards.py`

User-facing behavior and configuration are documented in `../../equity_hard_stop_loss.md` and
`../../equity_hard_stop_loss_cooldown_contracts.md`.

### Live balance input recovery

A numeric current raw/sizing balance that is non-finite or non-positive, or an invalid balance
in the required reconstructed HSL history, is explicitly unavailable for live risk evaluation.
Python keeps the process alive and refreshes authoritative account/fill state instead of consuming
the restart budget. Startup remains unready; runtime ordinary planning waits. No balance is
clamped to a positive substitute, no required history is discarded, and HSL remains enabled.
Malformed configuration, payload types/shapes, and unrelated validation errors retain their strict
failure policy. Rust and individual HSL sample consumers keep their positive-balance guards.

Retry delays grow from 5 seconds to 60 seconds for current balances and 300 seconds for history.
Account refresh, shutdown observation, and independently ready protection continue between replay
attempts. Scoped cooldown replays observe the same deadline so protective supervision cannot
bypass the backoff. History balance validation occurs before clearing live protection; existing
latched/cooldown state is retained on this failure. Protection may act only with its own required
inputs; the panic planner also requires positive current balances. Deferral does not fabricate
replacement strategy orders or new HSL state.

`risk.input.status` reports the cause, bounded balance values (non-finite values are `null`), first
invalid historical timestamp/value, invalid-row count, replay window, retry count and delay. It
emits a warning on entry/reason change and at most every five minutes for an unchanged reason,
then a recovery event after successful input validation and HSL initialization/check. Runtime
readiness state is retry scheduling only and is not persisted across process restarts.
