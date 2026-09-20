# Best-effort HSL reconstruction and shared drawdown design

Status: proposal for architectural review. This PR changes only this plan. It adds no
implementation, tests, configuration, or runtime behavior. Existing canonical contracts
remain authoritative until implementation PRs explicitly update them.

## Problem and intended result

HSL should estimate risk from imperfect history rather than require exact episode proof
before it can evaluate a stop. An old missing opening, ambiguous execution ordering, or
delayed fill should produce an observable approximation, not an indefinite protection
outage. The objective is reliable protective decisions, with particular attention to
missed or materially delayed stops. Premature exits also matter: a short fetch outage
must not discard usable history and defeat a long EMA span.

The target is one Rust-owned reconstruction and drawdown model for `coin`, `pside`, and
`unified`. Modes differ in scope and budget, not in whether they can recognize a past
unrealized equity peak. Coin mode is the first integration step. Temporary comparisons
against existing HSL support validation; a permanent legacy/new configuration switch
is not planned.

Current coin HSL uses a realized-PnL peak minus current realized PnL and UPNL, divided
by a slot budget. Changing it to reconstructed equity-peak drawdown is intentional.
The new calculation is not expected to reproduce old stop timing on all clean tapes.

## Intended policy changes: review against these targets

This is a redesign, not a promise to preserve every current HSL contract. The following
choices are intentional; implementation must update the affected canonical contracts
and test the consequences, rather than restore the old behavior as a review fix:

| Topic | Intended policy |
|---|---|
| Historical uncertainty | Estimate and evaluate; do not require exact opening/coverage proof. |
| Missing historical prices | Resample complete coarse candles, then forward-fill gaps and backfill a missing leading segment within the configured lookback. No separate carry-age veto or automatic switch to another signal. |
| Historical horizon | One configured lookback for all HSL historical influence, including cooldown, no-restart, and unfinished panic state. No extended lifecycle horizon. |
| `restart_after_red_policy=never` | No restart while the imposing stop remains in the lookback; the restriction expires with that event. It is not an indefinite latch. |
| Expired panic commitment | Clear the old commitment and reevaluate current exposure; a new threshold breach may immediately require another panic. |
| Restart authority | Current exchange state, in-window evidence, configuration, and time; no authoritative local journal or flag. Local caches are performance aids. |
| Unified mode | One portfolio signal, one controller, one explicit portfolio configuration, one portfolio-wide panic decision. |

These choices accept approximation error and finite memory. Review should challenge
internal contradictions, missing consequences, or implementation feasibility, but not
assume indefinite lifecycle retention, exact historical prices, or two unified side
controllers are requirements of the new design. Current-input freshness, valid order
sizing, and atomic Rust output validation remain required.

## Authority and minimum inputs

Rust owns historical estimation, drawdown, EMA, and risk decisions. Python owns exchange
I/O, factual fill normalization, caching, scheduling, diagnostics, and execution of Rust
intent. A small independent reference calculation is a test oracle, not a second live
risk engine.

An immutable evaluation snapshot supplies current balance, observed positions and their
bases, mark prices, contract metadata, available fills/candles, observation timestamps,
scope, configuration, and evaluation time. With valid configuration and usable minimum
current inputs, the evaluator always returns a boolean, even if history is empty:

```text
{ should_panic, raw_drawdown, drawdown_ema, quality, reasons, evidence_age }
```

This is one result per decision scope: coin+side in coin mode, side in pside mode,
and the whole portfolio in unified mode. Rust evaluates all applicable scopes; Python
does not derive extra side decisions from a unified result.

Quality explains estimates and repairs; it is not a downstream veto on `should_panic`.
No exact historical coverage or opening-fill certificate is required just to evaluate
protection. Malformed configuration, programming errors, and malformed Rust order output
are not converted into healthy-looking `false` decisions.

If essential current inputs cannot be obtained, suspend the actions requiring them,
report the missing account/market state, and continue recovery. This is outside the
estimator's minimum-input domain. A missing input does not itself cancel an unexpired
panic commitment: independently executable protection still runs with its own required
inputs, including the existing
balance-independent full-position close API. HSL historical degradation alone must not
block otherwise-valid martingale adds; ordinary strategy requirements remain separate.

### Snapshot discipline without a historical-readiness gate

Capture immutable observations with source timestamps and monotonic revisions for
balance, positions/bases, marks, fills, and configuration. Distinguish exchange event
time from fetch time; retain the fill-stream watermark used for the estimate. Use
the existing freshness requirements for essential current inputs, and report their
observed skew instead of claiming the requests were an atomic exchange snapshot.

Before installing reconstructed state, compare captured revisions with current ones.
Do not overwrite newer state with an obsolete reconstruction. On a detected change,
take a new snapshot and perform bounded recomputation; continued changes must lead
to a cheap current-anchored estimate, not an unbounded replay/retry loop. Reconcile or
isolate conflicting historical rows, including fills known to postdate the position
anchor, and refresh in the background. Never count the same transition twice.

The reference suite must specify interleavings and expected approximations. It must
not require an exact post-position fill-tail certificate or perfectly simultaneous
requests for the best-effort decision. Refresh current inputs when they cease to be
usable; otherwise historical mismatch degrades the estimate, not protection itself.
Protective execution independently refreshes/reconciles remaining exposure and orders.

## Reconstruction model

Start with coin/long examples in linear contracts; cover short positions and contract
multipliers before production integration. Preserve existing supported contract types
through their contract-aware PnL arithmetic; unsupported units cannot be approximated
by pretending every quantity is a linear base-asset amount.

1. Normalize/deduplicate fills by execution identity and apply corrections. Gross PnL
   and signed fee balance impact retain the existing accounting contract; net realized
   PnL is their sum. Estimates never overwrite the factual exchange ledger.
2. Anchor current size and average price at their observation time. Walk quantities
   backward: `q_before = q_after - signed_fill_qty`. For the long case, clamp impossible
   negative reconstructed quantities to zero, record the discrepancy, and continue.
3. Walk average entry price forward. For an incomplete initial episode, seed its
   carried quantity using the earliest usable retained fill price. Adds update weighted
   average price; reductions preserve it; an estimated flat resets estimated basis.
   A close's gross PnL may improve a missing basis when it identifies it unambiguously;
   that inference is optional, never another readiness requirement.
4. At the current endpoint use the observed size and basis. Report any reconciliation
   adjustment rather than silently overwriting the endpoint or spreading a correction
   over every old episode. A missing newest fill does not reveal when the discrepancy
   arose; backward allocation is an approximation, not proof of historical position.
5. Retain usable realized PnL and estimate missing components only from available
   evidence. An unknown realized component remains diagnostically unknown even if the
   usable series omits it. Do not invent realized profit or count a loss twice.
6. Rebuild when source observations change. Self-healing means convergence to the
   complete-data reconstruction when the missing evidence arrives. Clamping alone is
   not evidence that historical error has become small.

Same-timestamp rows are handled deterministically per symbol/side before scope
aggregation. Prefer actual sequence information when available. Otherwise use a
documented tie convention or cohort aggregation and report ambiguity; independent
symbols must not require a provable global fill order. Artificial or ambiguous flats
must not erase an in-window stop commitment or release a cooldown; explicit event
expiry is a separate, intended reset rule.

Approximation is local to this risk estimator. It does not certify fills for unrelated
consumers such as trailing-entry state, realized-loss gates, or financial reporting.

## Shared signal

At evaluation time `T`, construct minute observations within the configured lookback
and the agreed episode/reset policy. For scope `s`, aggregate *currency PnL first*:

```text
X_s[t] = sum(net_realized_pnl_cumsum[pair, t] + upnl[pair, t] for pair in scope)
B_s = applicable current balance budget
E_s[t] = B_s + X_s[t] - X_s[last]
P_s[t] = cumulative_max(E_s)[t]
D_s[t] = (P_s[t] - E_s[t]) / P_s[t]
M_s = EMA(D_s, span=ema_span_minutes, adjust=False)
score_s = min(D_s[last], M_s[last])
should_panic = enabled_and_active and score_s > red_threshold
```

Do not sum percentages or independent pair peaks to form an aggregate drawdown.
Current-budget rebasing is deliberate: the final reconstructed equity is `B_s`.
Transfers are not fill PnL. Changes in budget, deposits, and collateral valuation still
need explicit comparison cases; this plan does not claim invariance to changing budget.

| Mode | Signal contributors | Budget | Controller and panic scope |
|---|---|---|---|
| `coin` | One symbol and position side | Current raw balance / applicable side slot count | One controller per pair; close that pair |
| `pside` | All contributors on one position side | Current raw balance | One controller per side; close that side |
| `unified` | Both sides' account strategy PnL | Current raw balance | One portfolio controller; close the portfolio |

Unified panic deliberately covers both sides, including exposure on a side whose
ordinary entries are disabled. Per-side HSL enablement does not exempt a position
from an enabled portfolio controller. Per-coin overrides remain coin-only. TWEL is
an activity/sizing input, not a multiplier of the coin HSL budget. Do not invent a
divisor for an inactive zero-slot coin side; an unexpired existing panic commitment
can still own residual exposure. Portfolio activity must not depend on either side's
slot count when there is portfolio exposure to protect.
The existing dynamic-tradability backtest slot policy must be represented explicitly
in scope inputs, rather than silently changed in a reconstruction rewrite.

### Configuration ownership and migration

Keep the signal-mode selector explicit. Controller parameters use the same field
schema in distinct blocks:

| Mode | Active configuration |
|---|---|
| `unified` | `config.bot.hsl`: portfolio enablement, threshold, EMA span, tiers, panic execution, cooldown, and restart policy |
| `pside` | `config.bot.long.hsl` and `config.bot.short.hsl` |
| `coin` | Side HSL blocks with resolved per-coin overrides |

Unified mode takes parameters from neither side at runtime. For legacy unified
configs without an explicit portfolio block, identical fully resolved long/short
HSL settings may migrate automatically. Differing settings require an explicit
portfolio choice before startup; do not average them, pick one silently, or let a
default-generated portfolio block conceal an unresolved legacy conflict. An explicit
portfolio block is authoritative. This is config migration, not a runtime history gate.

For a fixed snapshot, the batch calculation is the reference. Repeated evaluation
with identical observations/time must give the same result. Any incremental
implementation must match it, including window rolloff, source corrections, current
budget rebasing, and restart. Do not silently mix batch-reseeded EMA with a different
unbounded streaming EMA.

### Price sampling and EMA

Use real 1m prices where available, otherwise the finest complete coarse source in the
1m/5m/15m/1h ladder. Reuse the existing deterministic zigzag expansion: rising or
unchanged candles follow open-low-high-close; falling candles follow open-high-low-close,
with linear segments and turning points near one-third/two-thirds of the interval.
Keep synthetic minute OHLC extrema consistent with the parent candle.

If no resolution covers an interval, forward-fill from the latest known candle; if
there is no earlier candle inside the window, backfill from the first available one.
Apply this to missing intervals, not only the old prefix supported by the current
resolution ladder. Do not look outside the configured window for a seed. If the whole
window has no usable candles, use the minimal-history construction below. The endpoint
always uses the usable current mark and authoritative current position/basis.

Historical price carry has no separate age cutoff inside the lookback and does not
disable evaluation. The finite window bounds it. A single retained candle may therefore
support a largely filled historical path; smoothing and peak errors are accepted
approximation risks, to be exposed in fault comparisons rather than prevented by a new
readiness gate. Old approximation can affect today's peak; it is not claimed harmless.
Record source resolution, filled segments, and age without certifying them as observations.
Never persist estimated rows as factual exchange candles.

Retrospective reconstruction may use a completed candle's full OHLC or a later observed
price to estimate earlier minutes. The source must have been available by evaluation
time `T`; a backtest decision at an earlier `T` cannot read a candle completed later.
This distinguishes approximate historical reconstruction from causal simulation. Do
not silently feed a pre-expanded future-complete candle to earlier backtest decisions.

Seed `M[0] = D[0]` from the first usable raw drawdown sample in the lookback. No extra
warmup history is required. EMA spans stay fractional. Repeated polling within one
minute must not advance the EMA clock multiple times; an updated current sample
replaces that minute's observation.

Known fill boundaries also enter the risk sample stream. Evaluate the final sample
at a scope flatten before its episode reset or later reopening, even within the same
minute. Recompute that minute's EMA contribution from the prior-minute baseline, not
from the previous within-minute update. Record a RED crossing before a later fill can
hide it. This preserves the existing CPU backtest's fill-boundary behavior without
requiring tick prices. Uncertain fill ordering still uses the documented approximation;
absence of exact ordering must not block the current risk estimate.

The numeric formula above describes the desired signal, not complete lifecycle logic.
Nonpositive historical peaks, extreme finite inputs, or inconsistent historical rows
must receive an explicit bounded estimation rule, not NaN propagation, a fabricated
zero drawdown, or a return to indefinite readiness failure. The reference cases must
settle those numeric edges before the new evaluator controls orders.

### No usable history

With usable current inputs and held exposure, create an estimator-local synthetic
opening: `qty=current_size`, `price=current_basis`, realized baseline `0`. The zero
baseline means past realized PnL is unknown, not that no losses occurred. Do not invent
an opening timestamp or write this record into the factual fill ledger.

A one-point equity series alone has zero drawdown after rebasing. Seed a peak reference
at entry value, then take one actual drawdown sample. With current UPNL `U`:

```text
entry_equity_reference = B - U
current_equity = B
peak = max(entry_equity_reference, current_equity)
D_current = (peak - current_equity) / peak
EMA([D_current]) = D_current
```

For loss `L`, the accepted normalization gives `L / (B + L)`, not `L / B`. For example,
budget 1,000 and loss 100 gives 100/1,100, approximately 9.09%. A profitable or flat
position gives zero in this minimal case. The reference seeds the peak only; it is
not an invented earlier zero-drawdown EMA observation or proof of an entry inside the
lookback. Past unobserved realized losses cannot be recovered this way.

This removes the separate timed raw-UPNL emergency evaluator in the eventual migrated
path. It does **not** preserve smoothing delay when no usable history exists: a
singleton EMA has no delay, regardless of span. That is an intentional tradeoff,
including at a cold start. A fetch failure must not discard still-usable retained
history and replace it with a singleton. Test transitions between minimal and richer
history because stop timing may change discontinuously as information arrives.

## Protection, episodes, and restart

Reuse the protective executor and scheduler: existing closes precede historical repair,
one stuck symbol cannot starve other scopes, and sizing uses fresh remaining exposure.
Do not create another Python risk controller around the new Rust evaluator.

### One historical horizon

For finite lookback `W` at evaluation time `T`, historical HSL inputs and their influence
are limited to `[T-W, T]`. No extra lifecycle horizon, out-of-window fill seed, permanent
halt archive, or authoritative local flag is introduced. Current positions, bases,
balances, and exchange orders remain current facts even if exposure originated earlier.
An explicitly supported unlimited lookback has no finite expiry; it is not a hidden
extension of a finite setting.

For a cooldown anchored by a flattening fill at `t_flat`, with duration `C`, it clears
at the earlier of its normal deadline and its anchor leaving the window. With the
inclusive interval above, the predicates are `T >= t_flat + C` or `t_flat < T-W`.
Clearing means removal, not restarting cooldown from now. `never` and threshold-based
no-restart restrictions likewise expire when their imposing event leaves the window.
Document `never` as "no restart while the imposing stop remains in lookback," not an
indefinite promise. Test exact endpoints consistently across live, replay, and backtest.

An unfinished panic commitment also expires when its imposing event leaves the window,
even if a position remains partly open. Clear that historical commitment and immediately
evaluate current exposure. Current losses may create a fresh panic decision. Merely
polling, retrying, or replaying an old event must not renew its timestamp. Expiry does
not erase resting exchange orders: reconcile them with fresh state and current Rust
intent, including cancellation or replacement where required, before new actions.

Flat scopes with in-window cooldowns/no-restart restrictions and scopes with unfinished
in-window exits remain relevant even without exposure. An old halt with no in-window
influence is intentionally forgotten, like other out-of-window PnL history. Both an
uninterrupted bot and a freshly restarted bot must apply this rule; it cannot depend
on whether a cache happened to retain older rows.

### Restart and execution continuity

Retire the existing emergency journal as authority in each migrated path. Reconstruct
from current exchange state and available in-window observations, config, and time.
Local caches may accelerate the same calculation, but must not preserve restrictions
which equivalent exchange-derived evidence would discard. Missing facts may change a
best-effort reconstruction; a decision that left no recoverable evidence is not promised
exact survival on a fresh installation. This deliberately replaces current durable-exit
semantics, rather than accidentally losing them during refactoring.

Within the window, estimated flats alone do not release a known halt. Lifecycle ambiguity
must not prevent estimating risk or servicing a currently justified close. Pending
execution uses current positions/orders and never stale cached order quantities. A
scope that is flat and has no in-window stop influence is not held indefinitely merely
because an ancient stop cannot be disproved.

Reference cases must specify in-window panic latching versus signal recovery and the
observable event used as each expiry anchor. No hidden RAM-only or journal-only decision
authority may be added to make replay appear exact. Actual fill boundaries and cooldown
events are distinct from synthetic position seeds. Warnings must disclose uncertainty
without restoring expired restrictions or introducing a historical-readiness veto.

## Reference and fault-test plan

The next PR builds a small independent reference model and synthetic public fixtures;
it does not switch live execution. Use hand-calculated clean cases, corrupt copies of
those cases, and progressive repair. Production Rust and offline fake-live integration
follow the reviewed reference. No private account data is needed.

| Fault class | Required comparison |
|---|---|
| Missing old opening | Backward quantities, forward basis estimate, current endpoint, and stop timing |
| Missing newest or middle fills | Explicit reconciliation, no readiness veto, convergence after delivery |
| Missing whole round trip | Demonstrate irrecoverable realized-loss uncertainty even when current quantity matches |
| Duplicate/corrected/reordered fills | No double-counting; rebuild on changed source values |
| Same-time mixed actions | Per-pair tie behavior; cross-pair permutation invariance; no invented cooldown reset |
| Over-close, impossible basis, malformed historical fields | Bounded repairs and warnings with valid current inputs |
| Missing/inconsistent PnL or fees | Explicit accounting assumptions; no fabricated profits or double-counted fees |
| Sparse/coarse/stale/absent candles | Zigzag, unbounded-within-window ffill/bfill, causal source availability at evaluation time, and current-mark anchoring |
| Empty history, held/flat/profitable/losing | Explicit synthetic reference, singleton EMA, no endless new-account replay gate |
| Temporally mismatched observations | Immutable timestamps/revisions/watermarks, revalidation, bounded recomputation, and continued current-anchored approximation |
| Invalid minimum inputs, zero slots, extreme numbers | Operational recovery/activity policy; no fake divisor or healthy-looking false |
| Restart, cache/journal loss, partial fills | Equal evidence gives equal decisions without local authority; expiry clears old commitments and reconciles remaining orders |
| Cooldown/no-restart/panic anchors leave lookback | Explicit forgetting in continuous runs and restart replay; no hidden retention or refreshed old timestamp |
| Unified asymmetric legacy settings | One portfolio controller; explicit config choice instead of silent selection, averaging, or two side decisions |
| Manual trades, transfers, budget changes, contract units | Correct attribution, scope, conversion, and denominator behavior |

Acceptance criteria:

- Clean inputs match independently computed signals within justified numeric tolerance,
  with matching trigger decisions/times. Test equality and near-threshold values
  separately from comfortably above/below-threshold cases.
- Historical damage alone never prevents evaluation with valid minimum inputs, and
  quality flags cannot suppress an above-threshold decision.
- Test sustained losses, benign paths, brief excursions under long EMA spans, and
  combinations of faults. Measure earlier/later/missed stops against clean evidence;
  investigate unexplained misses and material delays first. Do not impose one arbitrary
  timing tolerance across fundamentally different information losses.
- Restored evidence converges to clean reconstructed signals under the same window
  and lifecycle policy. Restoration does not revive an expired commitment; in-window
  continuity follows the reviewed latch rule. Permanently missing facts are reported.
- Test sign symmetry, contract multipliers, scope aggregation, per-side parameters,
  inactive sides, portfolio-wide unified execution, configured versus tradability-aware
  budgets, and one-position scope
  equivalence when budgets and parameters are identical.
- Fake-live covers startup, failed refresh with retained evidence, corrections during
  history I/O, partial closes, signal recovery, scheduler fairness, and restart during
  exit. No remote bot or authenticated exchange access is implied.
- Bound CPU, memory, diagnostics, and fetch work by configured lookback and relevant
  scopes, including flat in-window halts. Background repairs cannot delay protective
  waves. Optimization must preserve
  the reference calculation and should not introduce another correctness gate.

EMA and quantity clamping do not guarantee small reconstruction errors. The suite
must expose decision errors rather than declaring every finite result good enough.

## Migration and PR sequence

1. **This PR: design only.** Review architecture, assumptions, intentional behavior
   changes, and the questions below. Merge only with the required review/CI. No runtime
   contract is changed by merging this plan.
2. **Reference model and fault suite.** Settle concrete estimation rules with expected
   traces and decisions. Keep existing trading behavior authoritative.
3. **Shared Rust evaluator.** Validate against the independent oracle and wire bounded
   offline comparisons to the existing path. Retain one execution authority. Rebuild
   and verify the Python extension for affected callers.
4. **Coin integration/replacement.** Change live and backtest/optimizer behavior together;
   retain execution validation and implement the new bounded lifecycle semantics together.
   Update canonical contracts, user docs, schema migration, and changelog in the same
   PR. Review positive/negative results,
   not only crash-freedom. Remove the superseded coin path when integration is ready.
5. **Pside/unified alignment.** Reuse the same evaluator with reviewed scope/budget
   adapters, the explicit portfolio config migration, and parity tests. Until each
   migration lands, that mode keeps its existing
   behavior, rather than exposing an old/new user option.
6. **Final cleanup.** Remove remaining superseded replay/readiness and emergency-signal
   paths, temporary comparison hooks, and obsolete grace configuration once no mode
   consumes them. Preserve supported old-config loading with explicit deprecation or
   migration diagnostics; remove obsolete journal authority rather than keeping hidden
   out-of-window commitments. During staged migration, the legacy journal remains
   authoritative only for modes which have not migrated.

Stages may be combined when independently reviewable, but no commit may activate two
controllers for the same scope. Any live shadow trial requires separate deployment
authorization. Release notes must identify the new coin equity peak, denominator,
minimal-history behavior, finite lifecycle expiry (including `never` and partial exits),
journal retirement, unified portfolio scope/config, and other reviewed signal changes.
Thresholds are not
silently translated or described as numerically equivalent. Rollback is a reviewed
code revert with journal/config compatibility checked, not a permanent legacy switch.

## Architectural review questions and integration gates

The formula and best-effort direction are intended decisions. These remaining seams
need explicit review and then reference cases, not a collection of ad hoc live fixes:

1. **Window versus episode reset:** current HSL resets after confirmed scope flattening.
   Proposed baseline: preserve those resets and apply the shared formula within the
   lookback intersected with the active reset interval; absent proof, evaluate an
   approximate interval without releasing unexpired known halts. Review whether a full rolling
   window spanning confirmed episodes is instead intended. Do not change this silently.
2. **Estimator limits:** settle missing-basis reconciliation, tied fill ordering,
   nonpositive historical peak handling, and the transition
   between one-point and longer history with explicit expected examples. Keep the
   valid-current-input guarantee; estimates must not require proof of exact history.
3. **In-window lifecycle:** specify panic latching versus normal RED pause/reactivation
   and reproducible imposing-event anchors using available in-window evidence. Finite
   expiry of cooldown, `never`, and unfinished panic state, plus retirement of local
   journal authority, are settled policy choices, not open retention questions.
4. **Numeric/config compatibility:** the proposed strict `>` comparison differs from
   current tolerance-inclusive `>=`; review the boundary deliberately. Confirm budgets,
   dynamic slot policy, supported contract types, partial-minute behavior, and which
   existing tier/no-restart consumers must adopt the new signal together. Unified's
   single portfolio controller and dedicated `config.bot.hsl` block are intended;
   review migration correctness, not preservation of two side controllers.
5. **Feasibility:** verify that batch semantics, changing lookback/budget, and approximate
   history can be implemented at live cadence without hidden decision-changing state.
   Review the cost of a simple implementation before adding incremental machinery.

These gates prevent activation of an underspecified trading path; they do not prevent
starting the reference suite after design review. Missing history is a supported input
condition, not a reason to bypass this work indefinitely.

## Current contracts and integration points

- [HSL lifecycle and recovery contract](../ai/features/equity_hard_stop_loss.md)
- [Failure and degradation contract](../ai/error_contract.md)
- [Fill accounting and normalization](../ai/features/fill_events_manager.md)
- [Candle semantics](../ai/features/candlestick_manager.md)
- [Current user-facing HSL behavior](../equity_hard_stop_loss.md)
- [Validation requirements](../ai/validation.md)
- [Rust HSL signals and runtime](../../passivbot-rust/src/equity_hard_stop_loss.rs)
- [Python HSL replay and integration](../../src/passivbot_hsl.py)
- [Protective scheduling and recovery](../../src/live/hsl_protection.py)
- [Episode evidence](../../src/live/hsl_episode.py)

Implementation must replace conflicting historical-readiness rules explicitly in
their canonical contracts. This proposal does not broadly weaken factual fill
integrity, current account freshness, or the atomic Rust order-output contract.
