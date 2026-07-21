# Temporal Rust-Ideal Snapshot Order Churn Gate Plan

## Status

Planning and review specification only. This document intentionally makes no runtime, schema,
configuration, or test implementation changes.

Implementation must not begin until reviewers approve the temporal matching contract, final
execution ordering, risk exemptions, exchange assumptions, and the unresolved restart policy in
this revision.

This revision supersedes both earlier PR #1336 designs:

- it does not infer replacement identity from actual exchange orders or best-effort
  `_context=replace` annotations;
- it does not use `pb_order_type` or any other strategy/order-type name as a behavior-cohort key;
- it does not require stable Rust ladder-slot identity merely to detect ordinary price churn;
- it classifies behavior from a bounded history of normalized Rust ideal-order snapshots using
  deterministic one-to-one temporal multiset association.

Rust remains the source of order intent. Python observes whether Rust's executable ideal prices and
quantities are stable enough to justify leaving distant limit orders on the exchange.

## Decision Summary

1. Capture each complete normalized Rust ideal-order snapshot before reconciliation with actual
   exchange orders.
2. Keep a bounded ten-minute temporal history of valid snapshots. An implementation may compact
   identical observations without changing their logical time coverage.
3. Reconcile actual orders against the current ideal snapshot one-to-one using the existing
   universal `live.order_match_tolerance_pct`.
4. Keep every exact/in-tolerance actual order and cancel every remaining stale actual order.
5. Classify each unmatched current ideal order as `new`, `addition`, `stable_restoration`, or
   `replacement_derived` by associating normalized orders across historical snapshots.
6. Associate orders only within an order-type-agnostic execution cohort:
   `(symbol, position_side, order_side, reduce_only, execution_type)`.
7. Always admit market orders, explicitly Rust-marked risk-critical orders, and limit orders whose
   final fresh signed market distance is within
   `live.order_replacement_churn_gate_market_dist_pct`.
8. Always admit `new`, `addition`, and `stable_restoration` limit orders. They do not consume the
   replacement allowance.
9. Allow up to the available account-wide rolling capacity for far ordinary
   `replacement_derived` create attempts. The initial default is 10 attempts per 10 minutes.
10. When capacity is exhausted, cancel stale predecessors but defer their far replacements. Never
    retain an out-of-tolerance actual order as a substitute.
11. Recheck distance after cancellation and exchange-configuration work, as part of final batch
    admission immediately before the connector call.
12. Prioritize risk-critical creations ahead of ordinary creations before applying the final
    creation batch cap.
13. Treat already-absent/not-found cancellation results as ambiguous state transitions requiring
    authoritative account refresh and a new Rust plan; never create their dependent replacement
    from the old plan.
14. Keep classification and allowance events diagnostic. Trading decisions do not subscribe to
    the event pipeline.

## Problem

Some Rust ideal limit prices move continuously because their inputs move continuously. EMA Anchor
is the current concrete example: both entry and close prices can change with EMA bands, volatility,
and inventory. Small changes beyond `live.order_match_tolerance_pct` can cause hours of low-value
cancel/create traffic while the desired orders remain far from market.

Static grids and fixed distant closes behave differently. Placing them once is useful because they
can catch a sharp move, while leaving them unchanged consumes no further write capacity. A
universal market-distance gate would suppress these useful resting orders.

The desired policy is therefore behavioral:

- observe the executable ideal orders Rust actually emits;
- preserve first placements, additions, and stable restorations;
- identify orders whose normalized price or quantity has moved materially during the recent
  observation window;
- apply distance gating only after the account-wide allowance for such moving replacements is
  exhausted.

The mechanism is an exchange-write economy layer. It does not replace CCXT pacing,
exchange-native rate limiting, retry/backoff, ambiguous-write confirmation, or batch-size limits.

## Goals

- Stop sustained low-value replacement churn caused by moving ideal prices or quantities.
- Remain strategy-agnostic and order-type-agnostic.
- Preserve static distant limit orders and genuinely new ladder levels.
- Share the replacement allowance across the bot instance because exchange capacity is generally
  account-, UID-, subaccount-, or IP-scoped rather than coin-scoped.
- Preserve Rust ideal orders without synthesizing fallback prices, quantities, or strategy intent.
- Never preserve an out-of-tolerance stale actual order because its replacement is deferred.
- Always admit near-market and explicitly risk-critical actions through this economy gate.
- Pace newly available far-replacement capacity one attempt at a time.
- Keep classification, admission, deferral, and reset behavior observable and bounded.

## Non-Goals

- Reimplement exchange-native rate limiters or replace CCXT throttling.
- Coordinate multiple bot processes sharing an account, API key, subaccount, or IP.
- Infer whether an order came from EMA, volatility, trailing, grid, or another strategy input.
- Use `pb_order_type`, client-order IDs, or exchange order IDs as temporal behavior identity.
- Preserve maker queue position for an ideal order Rust materially changed.
- Change Rust prices, quantities, strategy calculations, or backtest order generation.
- Guarantee that a deferred far replacement could not have filled during a sharp market move.
- Suppress same-ideal retries caused solely by exchange rejection or ambiguous state; existing
  confirmation and retry contracts own that separate problem.
- Add exchange-native amend support in the first implementation.
- Treat the proposed 10-per-10-minute values as exchange-native limits.

## Safety Invariants

### Universal reconciliation tolerance remains authoritative

The existing `live.order_match_tolerance_pct` remains the only accepted reason to leave an actual
order at a price or quantity different from the current executable Rust ideal. The churn gate must
not widen exchange-order reconciliation tolerance.

Historical tracking tolerance is identity evidence only. It must never make a stale actual order
count as satisfying the current ideal.

### Cancellation and creation are asymmetric

Every actual order outside exact/universal-tolerance reconciliation remains eligible for
cancellation regardless of behavior classification or allowance state.

If a current replacement is deferred:

- its stale actual predecessor is still cancelled when selected by the normal cancellation batch;
- no stale price remains resting while Python waits for capacity;
- no older pending ideal is created later;
- the next cycle starts from fresh Rust intent and authoritative exchange state.

### Required inputs are never fabricated

Only complete, valid Rust planning results become history snapshots. A failed, partial, unavailable,
or fail-closed planning cycle must not append an empty or partial snapshot because that would
fabricate removals and additions.

Missing or stale market data follows the existing scoped create-deferral contract. Distance never
defaults to zero, infinity, a candle close, or another neutral substitute.

### Risk priority is end-to-end

Market orders, HSL panic orders, the dedicated protective-panic route, and orders Rust explicitly
marks `execution_priority=risk_critical` bypass the churn allowance and distance gate.

`reduce_only=true` alone is not an exemption. Ordinary EMA Anchor closes and take-profit orders can
be churny. Before implementation, Rust must explicitly classify unstuck, WEL/TWEL, auto-reduce,
graceful-stop, cooldown re-panic, and every other exposure-reducing family.

Risk priority must survive every later execution stage. Before `max_n_creations_per_batch` is
applied, risk-critical candidates sort ahead of ordinary candidates. Existing deterministic
market-distance ordering remains the secondary key within each priority class. Hard connector batch
limits remain authoritative; if risk-critical candidates alone exceed the cap, deterministic
priority within that class applies and remaining candidates are retried from a fresh plan.

## Proposed Configuration

Add these canonical `config.live` fields:

| Field | Default | Meaning |
|---|---:|---|
| `order_replacement_churn_gate_activation_count` | `10` | Account-wide rolling capacity for ordinary far replacement-derived create attempts. `0` disables replacement behavior gating. |
| `order_replacement_churn_gate_window_minutes` | `10.0` | Historical behavior and replacement-attempt window. |
| `order_replacement_churn_gate_market_dist_pct` | `0.005` | Final signed market-distance threshold that always admits a limit creation. `0.005` is 0.5%. |
| `order_replacement_churn_gate_tracking_tolerance_pct` | `0.01` | Wider 1% price/quantity association bound used only to connect temporal ideal observations. |

The existing `live.order_match_tolerance_pct`, currently `0.0002` or 0.02%, remains the tight
stability and actual-order equivalence tolerance.

Validation requirements:

- activation count is an integer greater than or equal to zero;
- window is finite and greater than zero when enabled;
- market distance is finite, greater than or equal to zero, and less than one;
- tracking tolerance is finite, greater than `order_match_tolerance_pct`, and less than one;
- defaults are owned by canonical schema/preparation, not repeated in runtime consumers.

### Retirement of the initial-entry setting

`live.initial_entry_exec_max_market_dist_pct` is removed from canonical schema, templates, CLI
aliases, examples, runtime, and current documentation. It must not remain as a silent alias because
the new behavior is temporal, account-wide, and applies beyond flat initial entries.

Recommended compatibility behavior is an actionable configuration error naming the replacement
settings. Reviewers should decide whether a released-version compatibility migration is required.

## Normalized Ideal Snapshot Contract

### Observation point

Capture a snapshot after Rust has applied live market constraints and Python has converted Rust
output into normalized API-ready order dictionaries, but before reconciliation with actual exchange
orders mutates, removes, pairs, or annotates them.

The snapshot contains only fields needed for execution behavior:

```text
symbol
position_side
order_side
reduce_only
execution_type
normalized_price
normalized_qty
execution_priority
```

`pb_order_type`, custom IDs, exchange IDs, and strategy kind may be retained only as bounded
diagnostic metadata. They are not classification keys.

### Logical snapshots and compaction

Conceptually, every complete Rust planning cycle produces an immutable snapshot. Append the current
snapshot after its reconciliation/classification decision, or otherwise ensure it cannot serve as
its own historical evidence.

Deep-copying every full dictionary payload is unnecessary. An implementation may store normalized
immutable tuples and compact consecutive identical snapshots into a time interval, provided the
compacted representation answers exactly the same presence, continuity, and stability questions as
the logical per-cycle history.

Use monotonic time for in-process window pruning. Snapshot cadence and compaction must be tested so
fast and slow planning loops produce the same classifications for the same temporal order sequence.

### Validity boundary

Do not append a snapshot when:

- Rust planning failed or returned an invalid structure;
- required market, account, candle, EMA, trailing, or risk input was unavailable;
- only a subset of required symbols was silently omitted;
- shutdown or authoritative confirmation prevented a complete plan.

A deliberate complete Rust result containing no orders is a valid empty snapshot. It must be
distinguishable from unavailable planning.

## Current Actual-Order Reconciliation

Reconciliation runs one-to-one between authoritative actual orders and a working copy of the
current normalized ideal snapshot:

1. Partition orders by the existing execution-compatible dimensions.
2. Deterministically match exact/universal-tolerance actual and ideal orders.
3. Each actual and ideal order may participate in at most one match.
4. Matched actual orders are kept and their current ideal counterparts are removed from creation
   consideration.
5. Unmatched actual orders are cancellation candidates.
6. Unmatched current ideal orders are creation candidates requiring temporal classification.

Historical matching never changes which actual orders are kept. It only determines whether an
unmatched current creation is new/stable or replacement-derived.

## Temporal Multiset Association

### Execution cohort

Associate normalized ideal observations only within:

```text
(symbol, position_side, order_side, reduce_only, execution_type)
```

This prevents cross-symbol, cross-side, entry/close, reduce-only, and market/limit confusion without
requiring knowledge of Rust order-type names. Market orders are exempt and need no historical
association for admission.

### Deterministic one-to-one assignment

For each pair of consecutive valid snapshots, within each cohort:

1. Find the maximum deterministic set of one-to-one tight matches using
   `order_match_tolerance_pct` for normalized price and quantity.
2. Among the remaining old/current observations, find deterministic one-to-one wider matches using
   `order_replacement_churn_gate_tracking_tolerance_pct`.
3. Among still-unmatched old/current observations in the same cohort, pair deterministically up to
   `min(old_unmatched, current_unmatched)` as discontinuous replacements.
4. Surplus current observations are additions.
5. Surplus old observations are removals.

Stage 3 is deliberate. A gradually or abruptly moving order must not become `new` merely because it
moved more than 1% from an old observation. Conversely, a growing ladder has surplus current
elements, so genuinely additive levels remain additions rather than replacements.

Assignment must minimize a deterministic normalized price/quantity cost with explicit tie-breaking.
Raw list ordering, exchange order proximity, `pb_order_type`, client IDs, and arbitrary dictionary
iteration order are forbidden inputs.

### Temporal tracks

Consecutive assignments form short-lived RAM tracks. A track records normalized observations,
presence intervals, material-change timestamps, and its latest association outcome.

Behavior is evaluated across the whole continuous track window, not only between adjacent samples.
This catches gradual drift: ten 0.01% steps may each be tight to the immediately preceding step but
the track is unstable once its recent observed price/quantity range exceeds tight tolerance.

A track is tight-stable only when every pair of its observations within the active window is
equivalent under the universal tolerance. Equivalent implementation may maintain deterministic
price/quantity extrema or an anchored equivalence envelope instead of an all-pairs scan.

Recently removed tracks remain as bounded tombstones for the history window. Reappearance is
associated against eligible tombstones using the same deterministic rules. Disappearance and
reappearance must not repeatedly manufacture free first placements.

## Creation Classification

For each unmatched current ideal creation:

- `new`: no eligible predecessor track or tombstone exists;
- `addition`: the current cohort has more elements after predecessor association and this element
  is surplus;
- `stable_restoration`: the ideal remained continuously present and tight-stable, but no satisfying
  actual order exists after an authoritative exchange-state transition;
- `replacement_derived`: its temporal track contains a material price/quantity transition, a
  discontinuous replacement, or a bounded disappearance/reappearance transition;
- `removed`: a historical element has no current successor; it produces cancellation only.

Admission consequences:

| Classification | Far ordinary limit creation |
|---|---|
| `new` | Always admitted through this gate. |
| `addition` | Always admitted through this gate. |
| `stable_restoration` | Always admitted through this gate. |
| `replacement_derived` | Uses account-wide rolling capacity once near-market/risk exemptions are evaluated. |
| `removed` | No creation exists; stale cancellation remains eligible. |

A replacement-derived track becomes tight-stable again only after every materially different
observation ages out of the configured history window. Python never keeps an obsolete pending price;
every cycle uses only the latest current Rust ideal.

## Account-Wide Rolling Replacement Allowance

Maintain an account-wide rolling history of ordinary replacement-derived create attempts for one
bot execution authority. The portable accounting unit is one logical order reaching the concrete
create connector-call boundary, not one HTTP request.

Count:

- an admitted far replacement-derived create attempt;
- an admitted near-market replacement-derived create attempt;
- failed, rejected, timed-out, or ambiguous connector-bound attempts;
- each logical order in a batch separately.

Do not count:

- local deferrals that never reach the connector boundary;
- new, additive, or stable-restoration creations;
- market or risk-critical creations;
- cancellations.

Near-market replacement-derived attempts are always admitted but still recorded. The rolling count
may therefore exceed `activation_count`, keeping far replacements gated during sustained useful
near-market activity.

When one timestamp expires, at most one newly far replacement gains capacity before immediately
reserving and consuming that slot. There is no binary release of every deferred creation and no
persistent queue of obsolete ideal orders.

## Cancellation Dependency And Authoritative State

### Batch selection

Replacement creation must not rely on a cancellation merely because that cancellation appeared in
the pre-batch plan. Bind every replacement candidate to the stale actual cancellation actions
selected by `max_n_cancellations_per_batch` for the current wave.

A creation whose dependency was truncated is deferred. Cancellation priority and existing batch
limits remain authoritative.

### Cancellation outcomes

Classify cancellation outcomes:

- positively acknowledged cancellation of the intended open order: dependency may proceed to final
  creation admission in the same wave;
- failed or ambiguous cancellation: dependent creation is deferred;
- already-absent, not-found, already-filled, or already-cancelled response: treat as an ambiguous
  account-state transition, request full authoritative positions, balance, open-orders, and fills
  confirmation, and require a new Rust plan before any dependent creation;
- cancellation truncated from the connector batch: dependent creation is deferred.

An already-absent response never proves that the old ideal remains valid. The order may have filled,
changing position size, balance, risk, close quantities, and trailing state. Existing
`state_change_detected_by_symbol` and authoritative-confirmation barriers must remain effective.

## Final Admission And Connector Ordering

Planning-stage distance and capacity decisions are provisional. Cancellation execution,
authoritative barriers, exchange configuration, retries, and elapsed time may change eligibility.

Immediately before the create connector call, build the final batch deterministically:

1. Remove candidates with unresolved/truncated/ambiguous cancellation dependencies.
2. Apply existing recent-execution, state-change, exchange-configuration, and readiness barriers.
3. Fetch/validate the canonical fresh market snapshot for every remaining symbol.
4. Sort candidates by `execution_priority`, risk-critical first, then by the existing deterministic
   signed market-distance order within each priority class.
5. Walk candidates in priority order while building a batch no larger than
   `max_n_creations_per_batch`.
6. For each ordinary replacement-derived limit candidate at this final boundary:
   - admit without capacity when its current signed distance is within the configured threshold;
   - otherwise require and provisionally reserve one rolling slot;
   - defer it when no slot remains.
7. New, additive, stable-restoration, market, and risk-critical candidates do not require a slot.
8. Convert a provisional reservation into a timestamp only when that logical create reaches the
   connector-call boundary; release reservations for filtered or truncated candidates.

If a provisionally near-market order becomes far before final evaluation, it must consume available
capacity or be deferred. The broader generic create-distance guard does not substitute for this
0.5% churn threshold.

No snapshot can eliminate all movement between a final check and network transmission. Reuse the
existing bounded market-snapshot freshness contract and keep the check as close as practical to the
connector call.

## Restart Reproducibility: Unresolved Implementation Blocker

The temporal snapshot history and replacement-attempt window change whether a far creation is sent.
They are therefore decision-changing state, not a performance-only cache.

The current `AGENTS.md` and `docs/ai/principles.md` require trading decisions to be reproducible
after restart from exchange state and configuration. A RAM-only deque that resets on bot or process
restart does not satisfy that contract. The prior proposed canonical exception cannot override the
current repository instruction.

Implementation is blocked until one option is explicitly selected and reviewed:

1. Persist the minimal normalized temporal tracks and rolling attempt timestamps using restart-safe
   wall-clock time, schema/version validation, bounded retention, corruption fail-closed behavior,
   and deterministic restart tests.
2. Reconstruct equivalent history from authoritative exchange/account data without fabricating
   missing observations. This is likely connector-specific and may be impractical.
3. Replace the temporal gate with a stateless deterministic policy derived solely from current
   exchange state and config, accepting its different missed-fill/churn tradeoff.
4. Explicitly change the repository restart-reproducibility instruction before approving
   RAM-reset implementation.

An in-process main-loop throttle would reduce bursts after automatic bot-object restarts but would
not reproduce behavior after a full process restart. It is not sufficient by itself.

No implementation slice may silently choose persistence, reset behavior, or a weaker throttle.

## Centralized Live-Event Integration

The live-event system is observability-only. Causal snapshot, track, allowance, reservation, and
dependency state is updated directly at canonical planner/executor hooks; no decision subscribes to
an event sink.

Proposed bounded events/reason codes include:

- valid snapshot observed or compacted;
- temporal association outcome: tight, wider, discontinuous, addition, removal, reappearance;
- creation classification: new, addition, stable-restoration, replacement-derived;
- replacement admitted by capacity or near-market distance;
- replacement deferred by account churn gate;
- final distance reclassification;
- risk-critical final-batch priority;
- dependent create deferred for truncated, failed, ambiguous, or absent cancellation;
- restart history restored/reset/unavailable according to the selected restart contract.

Safe fields include bounded cohort hashes, track hashes, classification, symbol, position side,
order side, reduce-only, execution type/priority, rolling count, thresholds, final distance, and
wave correlation. Do not emit raw snapshot arrays, full prices/quantities in summary events,
exchange payloads, arbitrary exception text, or credentials.

Repeated observations and deferrals use bounded periodic summaries. Event failure must not change
classification, cancellation, admission, or execution priority.

## Exchange Rate-Limit Research And Ramifications

Official documentation was reviewed on 2026-07-19 and 2026-07-20. Limits and account tiers change;
implementation reviewers must recheck the exact connector endpoints.

| Exchange/model | Official behavior relevant to the plan | Ramification |
|---|---|---|
| Hyperliquid | REST has a weighted IP pool. Address action allowance is tied to cumulative USDC volume, counts logical actions in batches, grants additional cancellation allowance, and may apply congestion/maker-share limits. [Official docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits) | Count logical actions, not HTTP batches. Preserve cancellation and near-market/fill-producing activity. |
| Bybit | Private limits use rolling per-second UID windows plus IP limits; batch usage is based on order count. [Official docs](https://bybit-exchange.github.io/docs/v5/rate-limit) | The ten-minute economy gate supplements but never replaces burst pacing. |
| Bitget | Futures cancellation has UID-scoped endpoint limits. [Official docs](https://www.bitget.com/api-doc/classic/contract/trade/Batch-Cancel-Orders) | Existing connector pacing stays authoritative. |
| KuCoin | Private quotas use UID resource pools and endpoint weights over short windows. [Rate-limit docs](https://www.kucoin.com/docs-new/rate-limit), [futures order docs](https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order) | Header-adaptive behavior is a separate project. |
| Gate.io | Futures placement/amend and cancellation have separate UID limits. [Official docs](https://www.gate.com/docs/developers/apiv4/en/) | Never infer that high nominal capacity makes low-fill churn harmless. |
| WEEX | Place/cancel and trigger actions use endpoint-specific account/UID or IP limits. [Official docs](https://www.weex.com/api-doc/ai/QuickStart/LIMITS) | The temporal gate targets sustained churn while CCXT owns bursts. |
| Binance USD-M | `/fapi/v1/order` consumes account order-count limits; single cancellation has IP weight; batch placement and cancellation have different weights and maximum element counts. USD-M overload handling explicitly exempts qualifying reduce-only/close and cancellation requests. [General info](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info), [trade endpoints](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#new-order) | Do not use Spot limits. Preserve risk-critical priority and logical action accounting while CCXT handles endpoint weights. |
| OKX | Trading limits are shared across REST/WS, separated by place/amend/cancel operation, generally scoped by user plus instrument; subaccount/fill-ratio rules affect new/amend traffic. [Official docs](https://www.okx.com/docs-v5/) | Cancellation remains independent; native amend is a later optimization. |
| Paradex | Private order APIs use per-account plus IP limits and leaky-bucket refill; batch operations may consume one rate unit for multiple orders. [Rate limits](https://docs.paradex.trade/api/general-information/rate-limits/api), [best practices](https://docs.paradex.trade/api/general-information/api-best-practices) | Logical action counting is conservative; verify JWT mode and native modify behavior. |
| Defx | Official docs expose placement/cancellation and 429 behavior but currently do not publish complete numeric limits. [Official docs](https://docs.defx.com/docs/api-docs/developer-hub/rest-apis-documentation) | Do not invent thresholds; retain CCXT pacing and require evidence before tuning. |
| CCXT/generic fallback | CCXT maintains per-instance throttling; separate instances do not share limiter state. [Official manual](https://github.com/ccxt/ccxt/wiki/manual#rate-limit) | The gate is bot-authority-wide, not a complete account/IP limiter across processes. |

## Live/Backtest Parity Analysis

Rust ideal orders remain unchanged. The universal order-match tolerance remains the only case where
live deliberately leaves a resting actual order at a slightly different current price/quantity.

The churn gate introduces an explicit live execution difference: after removing a stale actual
order, live may temporarily omit a far replacement that Rust currently wants. This may miss a wick
or gap before the order becomes near-market, tight-stable, or receives rolling capacity.

The temporal model bounds that difference:

- first placements and additive ladder levels may rest at any distance;
- tight-stable restoration may rest at any distance;
- the first account-wide replacement-derived attempts inside capacity are admitted;
- near-market and risk-critical actions are always admitted through this economy gate;
- no stale actual order remains as a substitute;
- every cycle uses the latest Rust ideal, never a queued obsolete price.

Fake-live parity evidence must record, for every cycle, the current Rust ideal snapshot, temporal
association/classification, actual order state, cancellation dependency/outcome, allowance state,
final distance, final priority, and connector-bound action.

## Failure And Safety Semantics

- Invalid configuration fails startup validation.
- Incomplete or unavailable planning appends no history snapshot.
- Missing/stale final market data defers affected ordinary creations.
- Missing required Rust execution-priority metadata fails at the schema boundary.
- Failed/ambiguous replacement create attempts consume rolling capacity once connector-bound.
- Failed/ambiguous cancellation dependencies block associated creations.
- Already-absent cancellation results force authoritative account refresh and replanning.
- Every stale cancellation remains eligible regardless of allowance exhaustion.
- Event publication failure is diagnostic only.
- Existing duplicate-write, recent-execution, account-state, HSL, WEL/TWEL, low-balance, mode, and
  readiness barriers retain precedence.
- Exchange 429/backoff may defer more actions than this economy gate.

## Review-Finding Resolution Map

### Current-head findings

1. Stable intent families: `pb_order_type` is removed from classification identity; temporal cohorts
   use only execution semantics and deterministic multiset association.
2. Risk-critical final batching: risk priority is an explicit pre-slice primary sort key.
3. Additive ladder levels: surplus current elements are classified as additions, not replacements.
4. Binance product scope: Spot documentation is replaced by official USD-M general and trade docs.
5. Near-market staleness: distance is revalidated during final connector-bound batch admission.
6. Already-absent cancellation: full authoritative refresh and a new Rust plan are mandatory.
7. Automatic restart reset: surfaced as an unresolved implementation blocker under current
   restart-reproducibility instructions; no silent RAM-reset exception remains.

### Earlier findings retained

- Required cancellations are never capped by the churn gate.
- Released capacity is consumed one-for-one; no bulk release burst exists.
- Ordinary reduce-only is not a blanket exemption; Rust owns explicit risk priority.
- `_context=replace` remains diagnostic and non-causal.
- Every live connector is included or conservatively covered by the generic CCXT contract.
- Creation depends on cancellation actions actually selected by the connector batch.

Review threads should not be resolved merely because this map exists. Reviewers must verify the
revised contracts.

## Edge Cases Requiring Explicit Tests

1. Current snapshot cannot match itself.
2. Invalid/partial planning appends no snapshot; valid empty planning does.
3. Identical snapshot compaction preserves logical time coverage.
4. Tight actual/ideal reconciliation is one-to-one with duplicate prices and quantities.
5. One historical observation cannot satisfy multiple current orders.
6. Static far first placement is new and admitted.
7. A growing static ladder marks only surplus levels as additions.
8. A shrinking ladder marks surplus old levels removed.
9. Equal-cardinality large price jumps are replacements, not new orders.
10. Normal/cropped/inflated `pb_order_type` transitions do not change temporal identity.
11. Raw ideal list reordering changes no association.
12. Equal-cost assignments use deterministic tie-breaking.
13. Gradual sub-tolerance steps whose total range exceeds tolerance become unstable.
14. Material quantity-only changes become replacement-derived.
15. Tight-stable restoration after a confirmed fill/state transition is admitted.
16. Disappearance/reappearance inside the history window does not manufacture repeated new orders.
17. A replacement becomes stable only after its material observations age out.
18. Ten far replacement-derived attempts across symbols consume one account-wide allowance.
19. An eleventh far replacement is cancelled and deferred.
20. One expired timestamp admits at most one newly far replacement.
21. Near-market replacement is admitted and recorded above activation count.
22. Market and risk-critical orders bypass capacity.
23. Risk-critical far create sorts ahead of ordinary near creates before batch slicing.
24. More risk-critical candidates than the hard cap remain deterministic and replan next cycle.
25. Final market movement from near to far consumes capacity or defers.
26. Final market movement from far to near admits without capacity.
27. Missing/non-finite final market data defers without fabricated distance.
28. Cancellation truncated by batch capacity blocks its dependent creation.
29. Positively acknowledged cancellation permits final admission.
30. Failed/ambiguous cancellation blocks dependent creation.
31. Already-absent cancellation requests full account confirmation and blocks same-wave creation.
32. A fill discovered after absent cancellation changes the new Rust ideal before any create.
33. Failed/ambiguous connector-bound create counts once; local deferral counts zero.
34. Current-wave reservations prevent capacity oversubscription.
35. Event failure changes no decision.
36. Persistent observation/deferral keeps console projection bounded.
37. Config reload, forager churn, mode changes, graceful stop, HSL, WEL/TWEL, unstuck,
    auto-reduce, hedge mode, and one-way mode preserve approved semantics.
38. The selected restart policy reproduces classification and allowance after bot-object and full
    process restart.

## Proposed Implementation Slices After Plan Approval

### Slice 0: resolve restart policy

- Select and approve one restart-reproducibility option.
- Update canonical decisions/contracts before runtime implementation.
- Define corruption, clock, schema/version, and reset behavior if persistence is selected.

### Slice 1: Rust priority metadata and canonical configuration

- Add required `execution_priority=ordinary|risk_critical` metadata owned by Rust.
- Audit every current risk/close family.
- Add canonical config fields, validation, templates, CLI aliases, and migration errors.
- Retire `initial_entry_exec_max_market_dist_pct`.
- Rebuild/verify the Python extension and add Rust/Python boundary tests.

### Slice 2: pure temporal tracker

- Implement normalized immutable snapshots, validity boundaries, compaction, cohort partitioning,
  deterministic tight/wider/discontinuous assignment, tracks, tombstones, classifications, rolling
  attempts, and provisional reservations as pure helpers.
- Unit-test cadence invariance, duplicates, additions/removals, gradual drift, and window expiry.

### Slice 3: reconciliation and cancellation dependencies

- Snapshot valid Rust ideal output before actual reconciliation.
- Preserve universal tolerance and unconditional stale cancellation.
- Attach classification/dependency metadata only to creation candidates.
- Carry dependency IDs through cancellation selection/outcome handling.
- Force refresh/replan for already-absent outcomes.

### Slice 4: final admission and executor priority

- Revalidate fresh distance after cancellation/config work.
- Integrate capacity reservation with final batch construction.
- Sort risk-critical candidates before ordinary candidates and then preserve existing ordering.
- Record attempts only at the concrete connector boundary.

### Slice 5: events and fake-live validation

- Add bounded registry events and sink-failure isolation tests.
- Add multi-symbol static, moving EMA entry/close, growing/shrinking ladder, fill restoration,
  cancellation race, batch saturation, restart, and mixed-risk fake-live scenarios.
- Quantify connector-bound calls and missed-order windows with and without the gate.

No authenticated exchange probe or live bot run is authorized by plan approval. Private probes and
live runs require explicit current-task authority.

## Implementation Validation Matrix

At minimum, the implementation PR must run:

- focused Rust order-priority tests;
- Rust suite, rebuilt/verified Python extension, and Python metadata boundary tests;
- config schema/default/template/CLI/roundtrip tests;
- pure temporal association, compaction, track, tombstone, rolling-window, and restart tests;
- reconciliation, cancellation dependency, final distance, priority, and executor tests;
- static/moving/additive/mixed-risk multi-symbol fake-live scenarios;
- event-bus, registry, query, smoke, bounded projection, and sink-failure tests;
- rewritten/removed initial-entry-only distance-gate regressions;
- `PYTHONPATH=src python src/tools/check_ai_docs.py`;
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`;
- `git diff --check`.

Report exact base/head SHAs and a cycle-by-cycle evidence table containing current normalized Rust
ideal, historical association, classification, actual order, cancellation dependency/outcome,
rolling usage, final distance, final priority, and connector-bound action.

## Rollout And Measurement

After offline validation and separate authorization:

1. Run one controlled account with structured DEBUG execution evidence.
2. Measure snapshot/track transitions, additions, stable restorations, replacement admissions and
   deferrals, final distance changes, risk priority, cancellations, fills, 429s, and ambiguity.
3. Prove no stale actual order is retained by the economy gate.
4. Inspect static grids, growing/shrinking ladders, EMA Anchor entries/closes, forager churn, and
   one-for-one release behavior.
5. Inspect Hyperliquid separately because requests-per-volume and cancellation allowance differ.
6. Inspect Binance USD-M risk-priority and batch behavior against its actual endpoints.
7. Revisit defaults only from evidence; do not hide exchange-specific defaults in adapters.

## Reviewer Checklist

Reviewers are explicitly asked to assess:

- Is the normalized snapshot boundary complete and independent of actual-order annotations?
- Can incomplete planning ever look like a valid removal snapshot?
- Does compaction preserve exact logical history semantics?
- Is the order-type-agnostic cohort key narrow enough without reintroducing strategy knowledge?
- Is deterministic one-to-one tight/wider/discontinuous assignment sufficient for equal-price grids,
  changing cardinality, gradual drift, and partial fills?
- Can one historical observation satisfy multiple current orders?
- Can a moving order escape as `new` after crossing the 1% tracking tolerance?
- Are surplus additions distinguished correctly from replacements?
- Can disappearance/reappearance manufacture free first placements?
- Does stable-restoration recreate only current Rust intent?
- Can any superseded ideal price be sent later?
- Is the first-10 account-wide allowance the right balance between forgiveness and churn relief?
- Do near-market attempts intentionally keep far replacements gated when activity remains high?
- Does final distance revalidation occur late enough without bypassing readiness?
- Can any ordinary create displace a risk-critical create from the final batch?
- Can cancellation truncation, failure, ambiguity, or already-absent state ever post a stale-plan
  replacement?
- Does one-for-one reservation consumption eliminate release bursts?
- Which restart-reproducibility option should be selected, and does it satisfy `AGENTS.md`?
- Are all connector assumptions, especially Binance USD-M and Hyperliquid, correctly scoped?
- What happens during fills, config reload, balance changes, forager churn, graceful stop, HSL,
  WEL/TWEL, unstuck, auto-reduce, hedge mode, and one-way mode?

Approval of this plan accepts the temporal classification and execution architecture but does not
resolve the explicitly blocked restart-policy choice, approve implementation details, authorize
authenticated testing, or authorize live trading.
