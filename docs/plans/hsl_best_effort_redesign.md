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

Quality explains estimates and repairs; it is not a downstream veto on `should_panic`.
No exact historical coverage or opening-fill certificate is required just to evaluate
protection. Malformed configuration, programming errors, and malformed Rust order output
are not converted into healthy-looking `false` decisions.

If essential current inputs cannot be obtained, suspend the actions requiring them,
report the missing account/market state, and continue recovery. This is outside the
estimator's minimum-input domain. It must not cancel a committed exit: independently
executable protection still runs with its own required inputs, including the existing
balance-independent full-position close API. HSL historical degradation alone must not
block otherwise-valid martingale adds; ordinary strategy requirements remain separate.

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
must not erase an actual stop commitment or release a cooldown.

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

| Mode | Signal contributors | Budget | Execution authority retained |
|---|---|---|---|
| `coin` | One symbol and position side | Current raw balance / applicable side slot count | That pair's enabled controller |
| `pside` | All contributors on one position side | Current raw balance | That side's enabled controller |
| `unified` | Both sides' account strategy PnL | Current raw balance | Existing enabled side controllers, with their own thresholds/EMA parameters |

Unified signal scope does not authorize closing a disabled opposite side. Per-coin
overrides remain coin-only. TWEL is an activity/sizing input, not a multiplier of the
coin HSL budget. Do not invent a divisor for an inactive zero-slot side; retain the
existing rule that an already-committed exit can still own residual exposure.
The existing dynamic-tradability backtest slot policy must be represented explicitly
in scope inputs, rather than silently changed in a reconstruction rewrite.

For a fixed snapshot, the batch calculation is the reference. Repeated evaluation
with identical observations/time must give the same result. Any incremental
implementation must match it, including window rolloff, source corrections, current
budget rebasing, and restart. Do not silently mix batch-reseeded EMA with a different
unbounded streaming EMA.

### Price sampling and EMA

Use the finest available observed price resolution; forward-fill position state on the
minute grid. Coarse candles and gap filling are explicitly approximate. A candle close
is only available at its close time: spreading an hourly close backward through its
hour would introduce future information. Carry forward known prices across gaps with
visible age; the current endpoint uses the current mark. Do not persist reconstructed
price rows as exchange candles or fabricate a price path before the first observation.

Seed `M[0] = D[0]` from the first usable raw drawdown sample in the lookback. No extra
warmup history is required. EMA spans stay fractional. Repeated polling within one
minute must not advance the EMA clock multiple times; an updated current sample
replaces that minute's observation. Exact fill boundaries remain separately available
for lifecycle handling.

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

Signal estimation and lifecycle authority are separate. Estimated flats can help
reconstruct prices; they cannot prove a cooldown elapsed or erase terminal no-restart
state. Conversely, unresolved lifecycle history must not prevent evaluating a held
position's risk or servicing an existing exit. Initial-entry and reopening permissions
must distinguish an empty new account from a known halted scope.

Preserve durable committed exits across partial fills, signal recovery, restart, and
the migration. The current availability/exit journal is an explicit persistence
exception; do not drop it merely because the old raw-loss evaluator is removed.
New reconstructed equity/EMA remains derived from observations, not an authoritative
cached result. Diagnostics identify approximation, stale evidence, and missing
continuity without mislabeling a failed fetch as restored protection.

Existing normal RED pause/reactivation and emergency commitment semantics differ.
Unifying the estimator must not accidentally choose a new exit-latching policy.
The integration review must map new panic decisions to a single explicit lifecycle
rule while honoring commitments already created by the old version.

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
| Sparse/coarse/stale/absent candles | Observable price approximation, no lookahead, correct minute clock |
| Empty history, held/flat/profitable/losing | Explicit synthetic reference, singleton EMA, no endless new-account replay gate |
| Temporally mismatched observations | Current anchor and deterministic approximation without demanding perfect simultaneity |
| Invalid minimum inputs, zero slots, extreme numbers | Operational recovery/activity policy; no fake divisor or healthy-looking false |
| Restart, cache/journal loss, partial fills | Signal reproducibility with equal evidence; commitment continuity; recovery remains scheduled |
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
- Restored evidence converges to clean reconstructed signals, without undoing an
  already-committed exit. Permanently missing facts are reported as limitations.
- Test sign symmetry, contract multipliers, scope aggregation, per-side parameters,
  inactive sides, configured versus tradability-aware budgets, and one-position scope
  equivalence when budgets and parameters are identical.
- Fake-live covers startup, failed refresh with retained evidence, corrections during
  history I/O, partial closes, signal recovery, scheduler fairness, and restart during
  exit. No remote bot or authenticated exchange access is implied.
- Bound CPU, memory, diagnostics, and fetch work by configured lookback and active
  scopes. Background repairs cannot delay protective waves. Optimization must preserve
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
   preserve execution and lifecycle guarantees. Update canonical contracts, user docs,
   schema migration, and changelog in the same PR. Review positive/negative results,
   not only crash-freedom. Remove the superseded coin path when integration is ready.
5. **Pside/unified alignment.** Reuse the same evaluator with reviewed scope/budget
   adapters and parity tests. Until each migration lands, that mode keeps its existing
   behavior, rather than exposing an old/new user option.
6. **Final cleanup.** Remove remaining superseded replay/readiness and emergency-signal
   paths, temporary comparison hooks, and obsolete grace configuration once no mode
   consumes them. Preserve supported old-config loading with explicit deprecation or
   migration diagnostics; retire obsolete journal fields without losing exit authority.

Stages may be combined when independently reviewable, but no commit may activate two
controllers for the same scope. Any live shadow trial requires separate deployment
authorization. Release notes must identify the new coin equity peak, denominator,
minimal-history behavior, and other reviewed signal changes. Thresholds are not
silently translated or described as numerically equivalent. Rollback is a reviewed
code revert with journal/config compatibility checked, not a permanent legacy switch.

## Architectural review questions and integration gates

The formula and best-effort direction are intended decisions. These remaining seams
need explicit review and then reference cases, not a collection of ad hoc live fixes:

1. **Window versus episode reset:** current HSL resets after confirmed scope flattening.
   Proposed baseline: preserve those resets and apply the shared formula within the
   lookback intersected with the active reset interval; absent proof, evaluate an
   approximate interval without releasing known halts. Review whether a full rolling
   window spanning confirmed episodes is instead intended. Do not change this silently.
2. **Estimator limits:** settle missing-basis reconciliation, tied fill ordering,
   earliest usable candle, nonpositive historical peak handling, and the transition
   between one-point and longer history with explicit expected examples. Keep the
   valid-current-input guarantee; estimates must not require proof of exact history.
3. **Lifecycle migration:** specify whether all new panic decisions commit until flat
   or retain normal RED pause/reactivation, how cooldown/no-restart history is recovered
   when old fills are absent, and what cache/journal loss means. Preserve existing exit
   commitments and decouple uncertain reopening from protective evaluation.
4. **Numeric/config compatibility:** the proposed strict `>` comparison differs from
   current tolerance-inclusive `>=`; review the boundary deliberately. Confirm budgets,
   dynamic slot policy, supported contract types, partial-minute behavior, and which
   existing tier/no-restart consumers must adopt the new signal together.
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
