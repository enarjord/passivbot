# Executable HSL reference cases

This is the first executable specification for the
[best-effort redesign](hsl_best_effort_redesign.md). It changes no production HSL,
config loader, optimizer, or order execution. The standard-library Python oracle
in `tests/hsl_reference.py` imports no production risk code. It is a test instrument,
not a Python trading controller or live fallback. Decimal amounts make it independent
of the future Rust floating-point implementation; integration must compare with
explicit tolerances and exact decision expectations near thresholds.

Run the focused suite with:

```sh
pytest tests/test_hsl_best_effort_reference.py tests/test_hsl_reference_fake_exchange.py
pytest tests/test_hsl_reference_replay.py tests/test_hsl_reference_replay_fake_exchange.py
```

The fake exchange tests use the deterministic, offline `FakeCCXTClient` from the
fake-live harness. They generate real fixture-ledger fills and compare reconstructed
positions, realized PnL, fees, and current UPNL against that client's independently
maintained account state. They do not start a bot or contact an exchange. No revised
HSL has yet been wired into the full live-loop harness.

## Reference choices made concrete

- Rebase currency PnL+UPNL to the current scope budget, then compute equity peaks,
  raw drawdown, and EMA of drawdown. Trigger on strict `min(raw, EMA) > threshold`.
  Seed the first EMA from its first drawdown. Within-minute observations replace
  the EMA contribution from the prior-minute baseline; known flatten boundaries
  must be checked before episode reset/reopening.
- Preserve all supplied boundary rows in the signal output. Episode clipping is
  explicit in the test caller; an inferred flat from ambiguous/clamped quantities
  is not promoted into a proven lifecycle boundary. The old-incomplete/new-clean
  example checks that a supplied independent episode boundary isolates old damage.
- Missing-opening quantity is inferred backward from current signed exposure.
  Basis starts at the first usable retained fill price and walks forward; current
  size/basis always anchor the endpoint. Inverse contracts use harmonic entry basis
  and inverse PnL units; scope budgets must use matching settlement currency.
- Missing gross realized PnL is estimated from the reconstructed reduction and basis.
  Fees are signed balance impacts, added exactly once. Unknown fee impact is omitted
  with a reason; this is uncertainty, not proof that no fee was paid.
  An invalid quantity omits only the unknown position transition with a reason;
  independently usable exchange PnL and fees are still retained.
- Execution identity is deduplicated before reconstruction. Highest supplied revision
  wins before applying timestamp membership; a correction moving a row outside the
  window removes its superseded in-window version. This uses supplied revision
  metadata, not an extended history-fetch horizon or out-of-window PnL. Contradictory
  rows at the same revision are excluded with a reason. Known
  sequences retain their relative order even within partially sequenced cohorts.
  Unsequenced rows follow the known subset, with increases before reductions and
  identity as tie-breaker. Ambiguity is disclosed, not a flat certificate.
- A nonpositive historical running peak has no meaningful ratio. The reference
  assigns raw drawdown 1 to that segment until a positive peak exists. Later normal
  positive-peak arithmetic resumes, including values above 1 if equity is negative.
  This is an explicit numeric reference choice for review, not existing runtime behavior.
- Minute-close sources must be available at evaluation time. Coarse source candles
  must be fully inside the window; real minutes are selected by close timestamp,
  even if their unused opening precedes the window, and override coarse estimates. Zigzag
  close pivots match the documented open/extrema/close path. Gaps forward-fill and
  the missing prefix backfills. The current endpoint is replaced by the fresh mark.
  A real 1m source needs only a usable close; malformed unused wick/open fields
  cannot discard it. Coarse synthesis requires usable OHLC because it consumes them.
- In the explicit minimal-history case, seed an entry-value peak reference, not an
  extra EMA observation. With loss L and budget B, the singleton signal is L/(B+L).
  Adding actual history can legitimately change the decision by restoring smoothing.
- Cooldown expiry uses the actual flatten timestamp. `never` expiry uses the imposing
  stop timestamp. These need not be identical; tests cover exact inclusive-window
  endpoints. Zero cooldown affects waiting only. All lifecycle calls recompute from
  supplied exchange-reconstructible evidence with no prior-decision argument.
  A proven flat plus fresh current exposure is enough to apply intervention policy;
  a delayed or missing reopening fill is not another readiness requirement.

## What these tests establish

Hand-calculated examples assert complete intermediate sequences, not only booleans.
Fault cases remove old, middle, and latest fills; repeat/correct/conflict identities;
damage price, fee, and realized fields; reorder ties; lose entire round trips; and
replace candle coverage with coarse or carried prices. Restoring evidence must restore
the clean series. Missing the latest reduction has an explicit example of a missed
clean-data stop: unknown realized loss and adjusted past sizes lower the estimate.
The suite records that difference instead of equating a finite result with adequate
protection. Opposite pair paths demonstrate why aggregation must precede peaks.
Shorts, flat short endpoints, multipliers, inverse PnL, inactive coin divisors, budget
changes, and extreme finite amounts are exercised separately.

Missing round trips cannot be inferred from an unchanged current size. The fixture
asserts that lost information, rather than manufacturing a loss or claiming full
reconstruction. A finite estimate or an empty reason set is not a completeness proof.

Lifecycle examples cover both restart policies and intervention choices, no-execution
RED followed by recovery, partial panic exposure, re-panic flatten anchors, and window
expiry. `LifecycleEvidence` is an input fixture of exchange-reconstructible event times,
not a persisted latch. It does not solve uncertain event classification by itself.

## Scope-boundary and snapshot experiments

`tests/hsl_reference_replay.py` extends the corpus without replacing production HSL.
It derives scope-flat candidates from in-window fill transitions anchored to observed
positions. Tests compute the boundary's final-risk sample from its consumed fill prefix
and feed the resulting stop/flat times to the lifecycle algebra; timestamps are no
longer merely hand-supplied in those cases. Financial inputs within a scope use the
same settlement currency; no new collateral conversion policy is introduced.

- A partial close does not end an episode. Final flatten rows retain realized PnL and
  fees, with zero UPNL. Actual unique same-pair exchange/simulator sequences preserve distinct boundaries and
  cashflow prefixes even within one millisecond, before a reopening can hide the loss.
  The normalized `Fill.sequence` input is not an arbitrary trade ID, list index, or
  arrival counter. Establishing ordering provenance in connectors is a later integration
  requirement. This follows the redesign's actual-sequence policy; it does not change
  the stricter position-chain requirement of the existing production HSL path.
  Unordered mixed-action cohorts may supply an estimated final-risk row, but its
  `lifecycle_eligible=False` marker forbids using it to release a halt or reset cooldown.
  Such a row is not proof of an internal flatten/reopening.
- Simultaneous activity across independent pairs is applied as a cohort. Local
  sequence numbers do not establish a global ordering. Closing one coin while opening
  another cannot manufacture a portfolio flat; coin/side scopes remain independent.
  Opposite signed positions are exposure, not a net-flat portfolio.
- Malformed quantities, conflicting revisions, or clamping may make a boundary
  unsupported. This is a reason on boundary reconstruction, not a veto on the separate
  best-effort drawdown signal. Missing history is not certified complete merely because
  these local checks pass.
- A contradictory opening followed by a reduction is not sufficient to relabel that
  reduction as a final close. The delayed-final-fill fake-exchange case verifies this.
  An inferred flat can delimit a later independently consistent episode for this
  reference, but is not itself exported as a lifecycle anchor or permission to release
  an existing halt. Older damage must not permanently contaminate a clean suffix.
- Boundaries cannot postdate the relevant position anchors. Fills newer than a captured
  position are isolated and disclosed until refreshed positions catch up, then counted
  once. Current flat state alone never supplies a missing fill timestamp.
- Consumed boundary evidence retains complete canonical fills, including revision and
  content, so a correction invalidates equality even when timestamp and cashflow stay
  unchanged. Repeating unchanged evidence remains idempotent.

Snapshot experiments capture immutable copies of positions, basis, marks, fills,
prices, configuration, source observation times, and producer revisions. They test
balance-only, position, mark, fill/fee correction, price, config, and window-rolloff
changes while a reconstruction is running. An obsolete result cannot replace the
newer snapshot's result, even if a producer accidentally reuses a version number.
Fills include explicit request-start and completion times, distinct from their event
timestamps. Unknown fill-fetch times stay unknown; they are not defaulted to the
position timestamp. Prices have a fetch-completion time and configuration its own
capture time. A fill request begun before the position observation cannot certify a
lifecycle boundary merely because it completed afterward. A tied request-start clock
also needs explicit causal acquisition evidence bound to that exact position and its
revision; otherwise the request must start strictly later. Fixture acquisition order
is recorded explicitly, never inferred from equal clock values. Older fill/price captures remain usable for risk
estimation with reasons, but are not marked revalidated for installing derived state.
Calculation, comparison, and quality checks project the selected coin/side's keys
(`scope_keys`); unified uses the whole snapshot. Balance and configuration remain shared
dependencies; selected pairs carry their own position/mark/fill/price revision tokens.
An unrelated pair's changes, including changes to aggregate producer tokens, must not
prevent installing a coherent coin estimate. Selected-pair revisions still invalidate
the result, and a regression remains unvalidated. Known canonical or conflicting fill
variants after the position anchor keep stable snapshots unvalidated until positions
catch up, without vetoing their risk estimate. Conflicting variants also forbid
lifecycle certification, even though identity normalization excludes their quantities.
Equal millisecond fill/position timestamps do not establish ordering either; they
carry an uncertainty reason until a later position observation resolves the tie.
Risk estimation continues, but a tied snapshot cannot certify a lifecycle reset.
Every explicitly requested scope key must have a current position observation,
including an explicit zero position when flat. An absent pair is unavailable current
state, not an implicit zero-exposure member of an aggregate.
Direct coin boundary selection enforces the same explicit-current-position requirement.
Revalidation also tracks high-water capture times for every selected source and shared
balance/config; repeating an older capture cannot validate it even if producer revisions
and evaluation time are unchanged.
Canonical per-identity fill revisions also participate in the high-water check, so
reusing broader producer tokens cannot validate a superseded fill correction.
Price capture time is explicit. Closes newer than that capture are excluded from the
estimate and leave it unvalidated until a causally possible capture arrives; retained
older prices still supply the ffill/bfill risk estimate.
The bounded attempt also remembers identities previously observed inside the window.
Their unexplained disappearance cannot validate an older tape. This is attempt-local
evidence, not a persisted fill ledger: window expiry or a causally valid correction
outside the window clears that requirement. An impossible future correction cannot
erase the earlier observation. Fills after their capture/evaluation time are excluded
from the estimate and cannot certify a lifecycle boundary.
Impossible revisions are quarantined before canonical selection, as a whole when
conflicting variants share the revision. Earlier causally usable versions therefore
retain their financial history and candidate risk samples. A mixed expired/future
revision cannot clear prior evidence. Missing/regressed-input diagnostics describe
the current candidate against retained high-water evidence, rather than latching:
restored inputs or a valid expiry correction can revalidate within the same attempt.
Revision numbers remain monotonicity evidence even when their financial contents are
quarantined. Conflicted identities retain their possible causal timestamps for detecting
disappearance. Corrected old timestamps are checked only against the current window:
they are ignored while expired, but matter again if a configured expansion includes them.
Retained timestamps are bound to their highest usable identity revision. Lower
revisions cannot overwrite them; same-revision observations retain all possible times.
Only a higher causal revision can supersede that disappearance evidence.
Conflicting timestamps at the same revision remain an in-window diagnostic until a
higher revision repairs them or every possible timestamp expires from the window.
The retained diagnostic record contains complete revision-tagged fill variants,
including quarantined identities and same-time content conflicts. Partial disappearance
cannot silently resolve a quantity/price/PnL/fee conflict; higher causal repair or
window expiry does. Unified evaluation also retains its observed position-key set
through the bounded attempt, requiring explicit zero observations instead of omission.
Future-dated records are quarantined anomalies, not expired history. Their unexplained
omission remains diagnostic until explicit higher causal repair or eventual window expiry.
Monotonic revision regressions (and evaluation-time regressions) stay unvalidated even
if the same older snapshot repeats. The estimate still exposes its current risk result.

Recomputation is bounded. Persistent churn returns a result for the latest captured
snapshot, explicitly marked as not revalidated, while retaining its usable history.
That diagnostic is neither a risk veto nor permission to install stale replay state or
size orders without fresh execution inputs. The test model bounds evaluation count;
it does not claim the eventual full Rust calculation is cheap enough at live cadence.
Freshness limits are fixture inputs, not new live defaults. Unusable essential current
state raises instead of returning an older healthy-looking decision. A failed history
refresh can retain previously usable exchange observations and their EMA smoothing.

The snapshot experiment's `estimate_pair` requires at least one historical minute close.
It fills a sparse grid using the existing reference ffill/bfill convention before
computing EMA, preserving one observation per minute without a historical-readiness veto.
It rejects an absent grid as outside that experiment's domain rather than silently
returning zero drawdown. The separate minimal-history oracle remains unchanged; this
test-helper precondition must not be copied into a production HSL readiness gate.
The candle-free scope experiment below specifies the separate no-price domain. A
mixed-history dispatcher still needs to compose both domains without discarding
retained candle-backed observations.

Offline fake-live exchange scenarios cover both sides, partial closes, delayed final
fills, actual cashflows, and reconstruction from freshly copied exchange evidence.
No revised logic is connected to order execution and no local decision artifact is
used as authority.

## Candle-free scope cashflow reference

`tests/hsl_reference_candle_free.py` composes current contract-aware UPNL with usable
net realized cashflows when no selected pair has historical candles. It is an offline
reference domain, not another live fallback/controller. The existing reconstruction
now exposes cumulative cashflows after each canonical fill independently of its price
sample grid. Known gross PnL and signed fees survive malformed quantity/price fields;
missing components retain the existing estimator rules and diagnostic reasons.

The minimal construction retains one actual EMA observation. Its entry reference is
enriched by known realized cashflows:

```text
R = final cumulative scoped net realized PnL
R_peak = max(0, cumulative scoped net realized PnL)
U = current scoped UPNL
peak_reference = max(B, B + R_peak - R - U)
current_equity = B
```

Thus known realized losses are not erased or counted twice. With budget 1,000, realized
loss/fees 22 and current unrealized loss 20, the drawdown is 42/1,042. Realized profits
followed by losses retain their peak, and current profits offset losses in currency.
Without fills, this is exactly the already specified minimal-history calculation.
There is no reconstructed historical UPNL path in this domain; unknown past unrealized
peaks remain unknown. Cashflow reference points do not invent past equity observations
or EMA delay. Even a very long span has a singleton EMA here. This can stop earlier than
a candle-backed reconstruction and is an explicit approximation requiring comparison.

Aggregate currency cashflows before computing their peak. Actual within-pair sequence
can reveal an intermediate peak. A tied multi-pair cohort is indivisible; independent
pair sequence numbers cannot establish an account-wide order. Ambiguity is diagnostic,
not a veto on the estimate. No cashflow peak certifies a flatten/cooldown boundary.
The later controller supplies the relevant episode interval; this helper does not reset
episodes or emit execution permissions. Zero-slot coin scopes are explicitly inactive;
portfolio scopes do not borrow that inactivity rule.

The helper rejects a supplied historical candle instead of silently discarding it for
a singleton. That is a test-domain guard, not a production readiness condition. A real
dispatcher must use retained candles whenever available, including after a failed fetch,
and must combine candle-backed and candle-free pairs without replacing the entire
portfolio's usable history. That mixed case remains an integration gate.

Tests cover all modes, both sides, linear/inverse contracts, partial realized losses,
fees, profits, simultaneous hedge offsets, tied sequences, invalid historical fields,
deduplication/corrections, exact window boundaries, post-position/future isolation,
zero slots, cache-free repeatability and restored candle smoothing. Offline fake exchange
cases supply actual partial-close cashflows for both long and short positions.

## Remaining coverage before production integration

This PR supplies reference primitives and a first fault corpus. It does not claim
the comprehensive integration gates in the design are already satisfied. In particular:

- Extend boundary-derived lifecycle cases to a complete retrospective controller,
  including normal-intervention resets and repeated cooldown re-panic across episodes;
  compare Rust and cache-free restart against the full traces. The reference permission
  algebra and boundary extraction do not yet implement that whole controller.
- Compose multi-pair episode resets and exact flatten samples with the shared timeline;
  preserve the stated per-scope semantics and same-minute ordering.
- Compose candle-free and candle-backed pairs on the same scope timeline without
  discarding known losses or retained EMA evidence. Compare against the candle-free
  cashflow reference and complete-data paths, including discontinuities on repair.
- Carry the immutable snapshot/interleaving cases into actual orchestration, with config
  migration, optimizer surfaces, and source-correction/candle provenance diagnostics
  tested at their real integration boundaries.
- Feed these fixtures through the rebuilt Rust extension and full fake-live loop,
  including protection scheduling, outstanding-order reconciliation, and partial fills.

Small artificial intervals in the numerical/lifecycle unit cases keep expected values
readable. They are not valid production lookback configurations: enabled revised HSL
still requires the separately specified 1-90 day config validation.
