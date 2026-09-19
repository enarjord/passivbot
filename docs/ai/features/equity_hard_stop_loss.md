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
   chain when a flatten is possible; list order and locally reconstructed position annotations
   are not ordering evidence. Coin mode may use the bounded nonflattening approximation described
   under live risk-input recovery when every ordering provably stays nonflat.
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
   The emergency availability journal below is a separate, explicit continuity exception.
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
create gate, leaving unrelated scopes available. HSL-only held-episode uncertainty follows the scoped grace policy below;
ordinary planning still requires its own current account, fill, and strategy inputs. Independently ready,
already-latched RED supervision and required panic protection for active cooldown positions still
run during that deferral, using fresh protective account state and the configured execution pacing.
An already-authorized close wave runs before balance reads, fresh signal evaluation, and flat-stop
bookkeeping. One protection scheduler owns startup and execution-loop ordering. It restores journal commitments
before inspecting pending work; the scoped health records, not a separate retry-controller flag,
are exit authority. Cold startup loads execution metadata and performs the required read-only connector routing/position-mode
preflight before servicing restored commitments. After metadata loads, config reconciliation
retires disabled or obsolete journal scopes before deciding whether preflight is required. Ordinary exchange-configuration balance gates,
account/history refresh and candle warmup follow protection. Bitget detects UTA/classic routing and verifies hedge mode on held positions; OKX detects account configuration; Binance, KuCoin and
Bitunix verify existing hedge mode; Bybit checks held positions' native position indices.
Unsupported modes retain the commitment and surface the connector error; no mode write is
performed by this preflight. Hourly market refresh
never starts a second commitment-draining loop; runtime execution remains the only order owner. Each wave services
committed exits and existing normal closes, then gives overdue unavailable scopes an evaluation
opportunity even while another scope remains open. Normal RED supervision yields to this scheduler
after its close attempt and before balance/history bookkeeping. Protective account reads retain the connector transport timeout per request (30 seconds by
default, configurable by the client). The scheduler does not impose a whole-cohort cutoff: sequential
and paginated account requests must each be allowed to finish. Transport timeouts yield to the next
protective owner; independently ready close attempts still precede balance-dependent evaluation. Emergency quote reads and optional cooldown fill-tail repair
have five-second deadlines so another due scope cannot wait indefinitely for them. Timed-out readers are cancelled and drained before another wave.
Flat normal stop finalization runs during recovery after ordinary refresh has had an opportunity,
so a flat latch alone cannot monopolize the pre-refresh gate.

The outer execution/startup loop checks existing RED and cooldown close work before
its ordinary account/history refresh, even without an emergency recovery journal entry. Once the
fresh position/order scope has no immediate close work, ordinary repair resumes. These later steps
may defer reopening but cannot prevent that close attempt. Normal
RED still requires `red_active_now` for subsequent panic intent; this ordering change does not alter
its signal-recovery policy. Aggregate mode getters honor the recovered sample's pause on panic closes;
a fresh sample reactivating RED executes a close wave immediately in that same supervisor pass.
A typed quote outage partitions ready symbols before Rust planning;
unavailable symbols retain their existing orders and appear in monitor diagnostics. Missing,
non-finite, non-positive, or crossed quote values are provider-defined availability failures;
no order consumes those values. Structural fetch errors and invalid Rust output still propagate.
Each submitted Rust batch is validated atomically. Singleton recovery probes run concurrently to
avoid accumulating a timeout per failed symbol. Overdue probes are cancelled and drained within
half the fetch-to-hard-TTL headroom so healthy quotes remain fresh for planning. Quote outage
diagnostics survive independent protective waves and clear only on a valid quote observation or
fresh confirmation that every position side of the symbol is flat. Normal-policy
cooldown reopening remains with the ordinary HSL evaluator and its validated current balance;
the reduced protection owner cannot release a halt using an old balance.
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

HSL signal health is scoped by mode, side, and (in coin mode) symbol. A completed
normal evaluation is usable; explicitly bounded approximate evaluation is degraded;
a scope without a usable evaluation is unavailable. A failed fetch alone does not
clear prior evidence. Normal drawdown and EMA formulas are unchanged.

`live.hsl_unavailable_grace_seconds` defaults to 120 seconds. Only continuous
unavailability starts this clock. Changing causes, successful fetches, partial replay,
retry limits, and emergency checks do not reset it. A completed usable/degraded
normal evaluation clears only its own scope's clock. Independently ready coin/pside
scopes continue evaluation while an attributed historical failure retries.

Before grace expires, HSL-specific reconstruction failure does not itself block
ordinary martingale adds when the ordinary planner's own inputs are valid. Its
current balance, fill/PnL, market, and strategy requirements remain enforced. A held
pair pending coin replay may add under these conditions; initial entries in an
unreplayed flat pair remain gated by the existing replay policy. Proven halted
scopes keep their independent cooldown/manual-ownership protection.

Coin evidence can remain usable with an explicitly bounded approximation. If a tied mixed-action
fill cohort has no position-chain metadata, its pre-cohort quantity is known, and all reductions
combined leave it strictly nonflat, every possible ordering belongs to the same episode. This
first approximation also requires monotone realized deltas (all nonpositive or all nonnegative),
so net minute replay cannot hide an intra-cohort PnL peak. Coin HSL uses a deterministic order,
retains the existing EMA, and reports `unordered_nonflattening_fill_cohort` as degraded. It never
invents a fill, PnL value, price, or flat boundary. A possible flatten, contradictory/partial chain
metadata, mixed-sign realized deltas, missing opening quantity, or position mismatch remains
unavailable. Monitoring exposes consecutive degraded evaluation counts, reset after usable
recovery or unavailability. Quality clears when the approximate cohort is trimmed from the active
evidence window or an exact flatten resets the drawdown episode. Consumed fills remain available
for correction detection. Aggregate modes keep
their existing ordering contract. The approximation is reproducible from fills after restart.

After grace, Rust evaluates `max(0, realized_loss - current_upnl) / budget` against the configured
RED threshold, without inventing an EMA. `realized_loss` is normally zero. Coin mode may supply the
verified realized peak minus current realized PnL in the currently held episode, using complete
current-episode coverage, finite non-pending PnL/fees, and a position-matching tape. The current
account cohort must include a successful tail-capable fill request started after its position
observation, with no intervening position observation. Concurrent requests sharing an epoch and
stale cached quantity equality alone are insufficient. If raw loss has not already committed an
exit, the emergency owner may attempt one ordered fill refresh with a five-second timeout and
at most one attempt per ten seconds. Failure leaves raw-UPNL evaluation active; a raw-triggered
close never waits for this optional enrichment. Fetched-fill value errors and fill-cache contract
failures make this optional evidence unavailable; malformed configuration, unrelated programming
errors, and fatal producer failures still propagate. If either the repair or tail phase discovers
new or structurally corrected fills requiring account confirmation, enrichment waits for confirmed positions and balance plus a new ordered tail; equal
net position size does not prove an unchanged cost basis. A proven last flatten excludes
previous closed episodes even before price replay succeeds. This evidence is recomputed each pass;
it is added once, never combined with an already-inclusive equity drawdown. Unavailable optional
evidence leaves raw-UPNL fallback intact. Aggregate emergency formulas remain raw-UPNL based.
New coin emergency decisions use the normal coin signal activity rule: zero configured
`n_positions` or wallet exposure limit makes that side inactive, including residual exposure.
Inactive uncommitted outages are retired, so reactivation starts a fresh grace period if the
signal is still unavailable. Completed emergency provenance remains available for replay.
No replacement budget divisor is inferred. A previously committed exit still owns its remaining
exposure and orders when sizing becomes inactive; explicit HSL disablement or signal-mode changes
retain their documented retirement semantics.

Coin budget is current raw balance divided
by configured `n_positions`; pside/unified use current raw balance. Pside UPNL is
side-scoped, coin UPNL is pair-scoped, and unified UPNL is account-wide. Positive
finite balance and fresh current position/quote inputs are required. An unavailable
quote defers only that emergency evaluation; it does not restart grace. Fresh
exposure with no execution history at all is a severe failure and commits an exit
after grace even if profitable. An empty, flat new account does not satisfy this
condition. Invalid Rust output and malformed producer/configuration values remain
fatal rather than being converted into availability failures.

A threshold-triggered emergency exit is committed until fresh positions and orders
confirm its entire scope flat and order-free. It uses the configured panic order
type and the minimal Rust close API: signed size, current book, tick size, and
execution policy. Existing commitments execute before any balance fetch for other
scopes; a stuck exit cannot starve another scope's emergency check. Partial fills,
submissions, and signal recovery do not release a commitment. The next wave uses
fresh remaining size. Historical repair and its backoff do not gate these closes.
After confirmation, the scope remains entry-blocked until canonical replay can
restore its normal cooldown/no-restart policy. No exchange-native stop is installed.

A narrow local continuity journal persists outage clocks and committed/confirmed
emergency exits across restarts, including the close window needed to attribute an actual
panic flatten to an emergency decision. Replay accepts that observed stop even when
its normal EMA never crossed RED, and still derives the flatten/cooldown timestamp
from fills. The latest emergency close window remains available after recovery. It does not contain reconstructed EMA, equity, or
ordinary strategy intent. This is the explicit exception to normal exchange-derived
restart reconstruction: past local unavailability cannot be recovered from exchange
fills. A restored commitment applies to exposure in that account/scope until fresh
flat/order confirmation, including exposure found after a restart. Deliberately
changing signal mode or disabling a scope retires its old journal scope visibly.
A corrupt/unreadable journal grants no renewed grace when a signal is unavailable;
fresh emergency inputs/thresholds are still required. A write failure is loud and
keeps protection running, but restart continuity is then not guaranteed. Losing the
journal entirely is indistinguishable from a first run and starts new grace.

`live.risk_input_max_attempts` (default 10) escalates diagnostics, never terminates
HSL protection. With HSL disabled, exhaustion remains terminal. Retry backoff grows
from 5 to 60 seconds (300 for historical balance failures), while current account
cohorts continue refreshing. Pending exits use protective refreshes; other emergency
scopes get a current balance cohort after the existing close wave. Network failures,
already-gone orders, and unavailable protective snapshots retain commitments and
retry at execution cadence. Authentication errors and malformed requests propagate.

`risk.input.status` reports reason, attempts, elapsed time and next repair delay;
first and limit-reaching failures include bounded tracebacks. Monitor HSL payloads
include per-scope status, reason, last evaluation time, grace remaining, emergency
budget/loss, commitment, execution blockage, and journal durability. Recovery resets
only after completed signal evaluation, not because another operation succeeded.

A consumed episode-evidence tape is also the boundary-consumption record. Unchanged
flatten boundaries already represented in that tape must not trigger another replay.
Corrections or late fills in the consumed window still invalidate the tape and
request canonical reconstruction before ordinary planning.
