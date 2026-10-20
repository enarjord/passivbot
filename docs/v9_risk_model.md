# V9 risk model

This document specifies the v9 implementation and its numerical reference. Migration and
qualification are required before using a v8 configuration. The implementation and platform
qualification must be reviewed together; this specification alone does not establish readiness.

## Economic inputs and ownership

Rust owns HSL, realized-loss memory, episode transitions and permissions. Python supplies
account, fill, price, coverage and configuration facts and enforces execution admission.
`coin` means a symbol/position-side pair, `position_side` means every position on one side, and
`portfolio` means all positions. Flatness means zero gross inventory, including in a hedged
portfolio. Changing a constituent of an exposed aggregate does not reset its episode.

Maintain the existing exact inventory and economic ledger. The risk boundary converts scoped
currency inputs to finite binary32 values once. A risk observation supplies current scoped
UPNL, current budget, ordered net realized cashflow, gross exposure and an integer timestamp.
Net realized cashflow includes entry/exit fees and signed rebates. Realizing marked PNL must
not debit the economic loss twice. Budget changes and external deposits/withdrawals are not
scoped trading profit or loss. Unattributed funding or collateral changes require an explicit
capability decision; an adapter must not guess their scope.

## Ordered arithmetic and sampling

Each HSL policy has independent `memory_ema_span_minutes` currency drawdown memory and
`drawdown_ratio_ema_span_minutes` ratio smoothing. Local policies live under
`bot.<side>.hsl`; the explicit portfolio policy and global scope live under `bot.global.hsl`.
The account-wide unstuck and loss-gate memories have separate spans under
`bot.global.unstuck.memory_ema_span_minutes` and `bot.global.loss_gate.memory_ema_span_minutes`.
All are floating-point EMA spans in minutes, not half-lives. A half-life is `log(0.5) / log(1 - alpha)` minutes;
span 1 has alpha 1 and half-life 0 as a limiting case.

For either span, cast to binary32, then compute `alpha = 2 / (span + 1)` and
`beta = 1 - alpha`, rounding each operation to binary32. Require finite span >= 1, positive
alpha and beta < 1. Reject spans whose rounded recurrence cannot decay. Disable implicit
multiply/add contraction; the `fma` operations below are explicitly fused and rounded once.

At an economic observation compute, in this order:

```text
loss = (previous_UPNL - current_UPNL) - net_realized_cashflow
first observation of a new minute: D = max(0, fma(beta_slow, D_previous, loss))
later observation in that minute:  D = max(0, D + loss)
E = budget + current_UPNL
P = E + D
R = D / P
M = fma(beta_fast, M_before_this_minute, alpha_fast * R)
RED = min(R, M) > threshold
```

Each non-fused operation rounds to binary32. Clips occur after each ordered economic event;
opposite cashflows must not be netted before clipping. Within a minute the drawdown retains
these causal extrema. Fast smoothing replaces the current minute's sample using the same
prior-minute baseline. It does not recursively smooth every query or every fill.

A pure query consumes no event, reapplies no fee and advances no committed clock.
Completed-minute marks, genuine fill-boundary observations, observed budget changes and
factual account reconciliations commit economic samples.
A current quote is a disposable preview cloned from those committed facts; changed previews
replace one another and never become a later drawdown/EMA baseline. Actual account/position
changes are factual reconciliation events, rather than disposable quote changes. Advancing
to a later minute commits the latest authored sample of the preceding minute. Missing integer
minutes apply the same recurrence once per
minute with the last supported mark/budget; this is not permission to call stale current marks
fresh. Do not substitute a power or adjusted span with a different rounding graph.

The clock is `timestamp_ms.div_euclid(60000)`. A completed candle is observable at its end.
At a shared timestamp, consume that completed mark before subsequent executions at the next
bar open. Executions retain their factual sequence. An adapter without an exchange execution
sequence uses its documented deterministic fact order and discloses ambiguous cohorts. A
fill is valued with the latest causally available scoped marks, including execution facts
when their valuation is supported. No candle high/low or later close may become an earlier
risk observation simply because the simulator knows it.

Finite nonpositive current E or P is an immediate loss branch: permission is RED regardless
of fast smoothing or threshold. Its diagnostic ratio is 1. Missing, nonfinite, overflowing
or otherwise malformed required inputs are errors, rather than neutral risk values.

## Episodes and finite cooldown

A proven flat-to-exposed transition initializes D and M to zero before consuming entry
economics. Partial closes retain the episode. Current GREEN retires unfinished panic intent.
At the exposed-to-flat transition, assess the latest causal pre-final-fill observation and
the final accounting observation, including final realized PNL and fees:

```text
qualified = pre_final_fill_RED OR post_final_accounting_RED
cooldown_until = flatten_timestamp + configured_duration if qualified else no_deadline
in_cooldown = flat AND now < cooldown_until
```

Every close can qualify, including a manual close without an order tag. Qualification does
not mean any historical RED in the episode. The deadline is an integer event result. Convert
a nonnegative duration with `floor(minutes * 60000 + 0.5)`, rejecting integer overflow. Later
GREEN, a new query or a budget change cannot cancel, extend or recreate it. A corrected fact
requires reconstruction of the event, not a local permission journal. Zero duration permits
immediate restart. New actual exposure clears the old terminal record. There is no permanent
`never` policy; an external operator or supervisor owns permanent shutdown.

## Continuous realized-loss memory

Auto-unstuck and the realized-loss gate each own an account-wide net-realized-PNL memory,
with independent enablement and spans. Both consume the same canonical fill facts. Each
excludes UPNL and never resets at an HSL episode boundary. Start from zero at a known origin.
For each enabled memory, advance its own decay once per elapsed minute and process every
fill in economic order:

```text
D_realized = fma(beta_slow, D_realized, 0)        # each elapsed minute
D_realized = max(0, D_realized - net_realized_PNL) # each ordered fill
p = f32(loss_allowance_pct) * f32(applicable_TWEL)
allowance = max(0, fma(p, f32(raw_balance) + D_realized, -D_realized))
```

For the global realized-loss gate use `p = f32(bot.global.loss_gate.max_drawdown_ratio)` in the same
allowance graph, without TWEL multiplication. Its original configured fraction determines
enablement: finite values in [0, 1) activate the gate and values at least 1 disable it.
Zero blocks lossy non-panic closes; negative or nonfinite fractions are invalid.
The gate remains active when HSL and unstuck are
disabled. The canonical producer feeds every actual net fill once to each enabled memory, including
fees and rebates, before any HSL-only early return. The ordinary ledger accounts for the
fill only once. A disabled memory owns no neutral substitute for an enabled consumer.

Retain the existing allowance, selection, exposure, price-EMA and close-sizing policies.
The ordinary balance ledger accounts for each loss or fee exactly once. Profit repays
remembered loss; excess profit creates no negative credit. This changes replenishment and
requires fresh joint calibration of span and loss allowance. Config schema v9.0.0 explicitly
retires the rolling gate window; migrating an active released-v8 gate requires semantic
acknowledgment. Equal numeric windows, zero and all-history policies are not equivalent
exponential spans.

## Bootstrap, correction and restart

Ordinary CPU/GPU replay advances once through facts with constant retained risk state per
scope. It must not rescan historical candles after fills, budget changes or queries. Live
startup and fact correction may rebuild a bounded factual history using the same Rust core.
Rebuilding a derived fold from the same retained boundary and ordered facts reproduces
the decision. Live factual storage is bounded: routine pruning preserves compact currency
drawdown, ratio EMA, event clocks, episode state, RED/flatten timestamps and cooldown deadline,
plus predecessor causal inventory/marks. It never periodically rebases D/M to zero.

The retention horizon is the current bounded acquisition interval, extended to the full
supported cooldown and at least one minute. Once retained history exceeds twice that
horizon, checkpoint immediately before the completed-minute cohort at `now - horizon`.
Retain the later facts and one budget predecessor; discard earlier raw grids, fills, account
observations, duplicate initializer arrays and diagnostic event vectors. Fixed workloads
therefore retain roughly one to two horizons, with a minute of checkpoint alignment slack.
This bounds factual age; event rate still determines storage within the horizon.

Dormant scope owners and their budget maps expire after their applicable horizon without
synthesizing flatness or lifecycle events. A scope omitted by a partial capture is not proven
flat. Known unexpired cooldowns survive ordinary pruning and derived-cache rebuilds.

Restart, owner loss, incompatible policy changes or corrections requiring discarded history
use an explicit bounded approximate reconstruction. Such reconstruction can shift HSL timing
in either direction because the discarded prefix is unknown. It cannot promise exact
historical RED qualification or an unknowable cooldown. Supported corrections and intentional
policy changes may legitimately alter a deadline. There is no persisted panic latch or local
permission journal. Corrections within retained support replay exactly from the compact
boundary. Native continuity diagnostics expose initialization, bootstrap clocks, retained
support, checkpoint time, horizon and reconstruction reason. A successful unavailable result
only owns initialization if native actually retained its ledger.

Current quote previews never replace initialization or committed samples.
Inventory reconciliations use the position read's source clock and retained marks already
supported at that clock. Quote observation clocks remain separate: a later quote cannot
be copied backwards to value an earlier inventory change. Balance components likewise
use their source clock with the policy then observed; changed slot or exposure settings
apply at their own observation clock. Equal source and execution clocks without a proven
phase use a disclosed deterministic phase estimate. Repeated captures do not move a prior
reconciliation or renew cooldown.
Fill revisions and scalar price facts must respect their independent source capture
cutoffs. Quarantine an entire fill revision when any variant exceeds its capture
cutoff or the current snapshot time. A later supported receipt can admit it once.
Reconstruction can consume already-admitted historical corrections without reapplying
the original initializer's earlier cutoff. The original capture clocks remain unchanged.
Identical inventory values at distinct actual source clocks remain factual endpoint
observations; they do not prove that inventory was unchanged between fills.
Compare corrected position observations in source-clock order before detecting no-ops
or fill crossings; capture arrival order cannot erase a later causal reconciliation.
Real live account and event timestamps share UTC. Processing time between clock reads
is not an offset. A synthetic harness supplies its clock projection explicitly as a
receipt-phase estimate, which does not prove historical execution phase. Reject future
projected account clocks instead of clamping them into a valid interval.
The offline harness records each successful account receipt's scenario clock against
its source revision and UTC stamp. A later capture or replacement owner cannot move that
receipt across a fill boundary. Reuse an older matching position read only while its
mapped source remains fresh; otherwise require an independently successful current read.

A newly discovered unknown episode may wait until its first supported valuation and
seed there with an explicit approximation. An already exposed aggregate with an unpriceable
contributor is unavailable until factual correction supplies a causal mark; it cannot
reuse an earlier permission or invent a mark. Held inventory in an enabled coin scope with
zero configured slots is an error; a flat zero-slot scope remains inactive. Retained fact
comparison may scan history; this is distinct from replaying the numerical risk fold on
every query.

A position read preceding a supported partial fill cannot supply a later quote preview's
quantity or basis. If it disagrees with causal terminal inventory while the scope remains
exposed, report unavailable until a fresh account cohort reconciles the facts. Keep the retained
fill facts and boundary, without committing the stale preview.

Target five slow spans of live reconstruction, capped by
`live.risk_bootstrap_max_lookback_minutes` (default 129600, maximum 90 days). Report actual
coverage, unknown prefix, missing marks, ambiguous fills and any shortened target. Keep the
history retention/acquisition horizon distinct from exponential memory.
The effective HSL acquisition horizon is the larger of this bounded slow-span target
and the largest enabled configured cooldown. Live capability validation rejects an
enabled cooldown longer than the configured cap, rather than silently losing its flatten
evidence. This cap bounds startup/acquisition windows; it does not periodically reset
continuous risk state during an uninterrupted run.

From a proven episode origin, replay with D=M=0 before entry economics. Equivalent complete
facts and initialization reproduce the same decision; a bounded restart with an unknown
discarded prefix is an observable approximation. For an unknown already-exposed
prefix, derive an explicitly approximate seed from observed size/basis/mark: D=max(0,-UPNL),
R from the current seed budget, and M=R. This approximation is not a new episode and must
not create a new fast-EMA warmup which temporarily hides an observed loss. If historical
prices are unavailable, the current-basis singleton remains an observable approximation.
Fresh current marks, complete positions and balance remain mandatory.

For an unknown exposed prefix, placing this fresh current-fact seed at the retained left
boundary is an initialization estimate, not evidence those facts existed at that time.
Keep the seed's previous UPNL in the normal economic-delta graph while replaying the
separately estimated history. Do not overwrite it with a guessed opening basis without
the corresponding offset, or debit the final observed current loss twice.

Continuous realized memory starts at zero at the retained coverage boundary with an
`unknown_prefix` diagnostic unless its true origin is known. Report residual attenuation
beta^N; this is a multiplier on unknown initial state, not a bound on economic error.
An arbitrary finite suffix cannot prove equality to an unknown infinite past.
Choose that boundary once per uninterrupted run. A backward clock correction or larger
bootstrap horizon cannot move the retained origin or admit an older zero-initialized prefix.

Historical budgets use supported balance/cashflow/slot facts when available. Otherwise
reconstruction discloses its current-budget estimate; it must not claim those historical
ratios are exact or rewrite already-committed ratios during forward replay. Delayed or
corrected facts invalidate affected reconstruction. Cooldown comes from retained causal flatten
facts and the native compact boundary during an uninterrupted run. A bounded cold restart
reconstructs available facts with declared uncertainty, never an old order label or panic flag.

## Qualification

Check this model against independent fixtures, Rust unit/property tests, source-verified
Python calls and deterministic fake-live planning/admission. Test CPU, Metal and CUDA against
the v9 model. A missing device is missing evidence. V8 is a performance and economic control,
not the correctness oracle for changed risk semantics. Use independent ordinary account
equity metrics for economic comparisons. Freeze economic criteria, workloads, bounds, seeds,
novelty thresholds and timing phases before acceptance runs.
