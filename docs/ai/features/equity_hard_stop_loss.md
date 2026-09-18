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
   flat pair may still own a RED cooldown. When the full tape proves that closed episode and
   its cooldown-connected predecessors, their earliest opening remains required. Unproven
   closed-episode evidence preserves the full configured lookback.
   Ambiguous or delayed held evidence also preserves or restores the full requirement before fills
   become authoritative. PnL blockers are evaluated against each held pair's own canonical episode
   boundary; the aggregate earliest boundary exists only to fetch and prove coverage. Coin stop
   finalization consumes pair metrics and must not add an account-wide PnL dependency. Coin mode
   evaluates each configured coin's effective HSL enablement, restart policy, and cooldown.
   `threshold`, `never`, pside, and unified modes remain full-lookback strict.
   Live symbol discovery, PnL sampling, and ordinary-boundary checks use this same
   proven coin window and retain its initialized lower bound until canonical
   reconstruction replaces it. Time passing
   must not slide the boundary past delayed fills, and newly uncertain scope evidence
   restores strict checking. Discarded cache rows cannot reintroduce old episodes.
9. Restart price reconstruction fetches 1m history first. When an exchange cannot provide the
   older leading portion, it may use 5m, then 15m, then 1h candles for that prefix. This is an
   explicitly approximate price path: the finest source wins and its contribution is reported.
   Coarser candles never repair missing rows at or after the first available 1m candle. Fill-based
   episode boundaries, realized PnL, and fees remain exact.

## Failure Semantics

Incomplete fill coverage follows `../error_contract.md`. A required episode boundary is unavailable
until supported by fill evidence. Startup replay validates all enabled scope tapes before replacing
existing protective state. Unavailable HSL evaluation invokes the conservative exit policy below after an
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

## Episode Evidence Ownership

`live/hsl_episode.py` owns immutable quantity reconstruction, exact flatten indices,
PnL prefixes, and cooldown-connected episode ranges. Coverage selection, startup
replay, and live boundary checks consume this evidence. Candle timestamps are a
separate projection: rounding a price row never changes an exact fill/PnL boundary.
A replay window retains the proven opening quantity rather than assuming a clipped
tape starts flat. Python reconstructs exchange facts; Rust still evaluates risk.

Startup captures value-based fill, position, balance, and HSL-config evidence before
history I/O and rejects changed observations before replacing protective state.
The original full tape supplies held and recent closed-episode boundaries even when returned price
history starts with a close whose opening fill was discarded. Live checks validate
current positions again; revisions to already-sampled quantities, PnL, or fees request
canonical reconstruction before ordinary shared-account planning. Evidence is
rederived after restart and never persisted as trading authority.

## Code And Tests

- Replay and live finalization: `src/passivbot_hsl.py`
- Live orchestration bindings: `src/passivbot.py`
- Coin replay and cooldown regressions: `tests/test_hsl_coin_mode.py`
- Pside/unified finalization coverage: `tests/test_unstucking_safeguards.py`

User-facing behavior and configuration are documented in `../../equity_hard_stop_loss.md` and
`../../equity_hard_stop_loss_cooldown_contracts.md`.

### Live risk input recovery

Unavailable current balances, required historical balances, episode evidence, or
required fill/PnL readiness block ordinary planning. With HSL enabled, recovery
must not terminate the protection owner, including after
`live.risk_input_max_attempts` (default 10). That limit escalates diagnostics to
errors; capped retries continue. With HSL disabled, the limit retains its terminal
stop behavior. Malformed configuration, payload shapes/types, and malformed Rust
output retain their strict failure contracts.

A loss of HSL evaluation readiness conservatively closes exposed HSL-enabled scopes
and cancels their resting orders, using a fresh protective account snapshot and
Rust's existing panic planner. This is an explicit live availability policy: it may
close earlier than the configured RED threshold. It does not invent a drawdown,
reset losses, disable HSL, or substitute ordinary strategy intent. Configured panic
execution type still applies. Proven halted scopes retain their existing
cooldown/manual-ownership policy and independent protection. Recovery advances
latched RED supervision one wave per pass so flat confirmations and stop finalization
continue without monopolizing exits in other scopes.

Protective refresh requires positions, orders, and valid current balances; historical
repair and its backoff do not gate the exit. A new position or resting entry observed
during backoff is included. Partial fills and successful submissions do not release
the exit commitment: another fresh account snapshot must show the relevant positions
and orders gone before exact history recovery may release ordinary planning. No
order is sent on stale account state or an invalid current balance. Recovery keeps
refreshing when those inputs are unavailable; this policy cannot execute through an
exchange outage and does not install exchange-native stops. Transient connector
failures and unavailable protective snapshots retain the exit commitment and retry
inside protection rather than entering full-bot restart handling. An incomplete
protective account read keeps the same execution cadence. Retryable connector errors
include network failures and already-gone orders; authentication failures and malformed
requests propagate. Malformed producer output and configuration remain outside that
recovery policy. A failed required fill fetch remains unavailable even without a more
specific pending-PnL, degraded-PnL, or coverage diagnosis.

Retries grow from 5 seconds to 60 seconds for current balances and episode evidence,
and to 300 seconds for history balances. Startup and runtime defer the full historical
cohort until that deadline; a pending exit uses protective-only account refreshes
until confirmed complete. Recovery runs the bounded protective owners first, then
applies one shared execution delay; owners must not each add another delay. Ticker availability is typed across provider and fallback paths; deterministic connector or metadata failures do not enter this recovery. `risk.input.status` records the cause, attempt count,
limit, elapsed time, next delay, and `protective_exit_and_retry` action. First and
limit-reaching failures include bounded tracebacks. Polls within backoff do not
spend attempts; changing reasons does not renew the budget. Successful owning
operations reset recovery. This state is not persisted: restart must encounter the
same unavailable exchange-derived inputs before ordinary trading can resume.

A consumed episode-evidence tape is also the boundary-consumption record. Unchanged
flatten boundaries already represented in that tape must not trigger another replay.
Corrections or late fills in the consumed window still invalidate the tape and
request canonical reconstruction before ordinary planning.
