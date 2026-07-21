# Rust-Ideal Churn Evidence Gate and Cancel-First Execution Plan

## Status

Planning and review specification only. This document intentionally changes no runtime behavior,
configuration schema, or tests.

This revision replaces the earlier identity-tracking designs in PR #1336. It deliberately keeps the
same PR because the problem and architectural objective are unchanged, and the existing review
history explains why this simpler design was selected.

Implementation must not begin until reviewers approve:

- the exact reconciliation identity;
- the quantitative churn-evidence heuristic;
- the cancel-first execution boundary;
- risk-critical exemptions and ordering;
- the narrow RAM-only restart exception; and
- the accepted live/backtest parity tradeoffs.

## Executive Decision

Rust remains the sole source of current strategy intent. Python does not infer whether an order is
conceptually new, a replacement, an addition, or a restoration. Those labels are too ambiguous for
causal trading policy when ladders change cardinality, orders disappear and return, or strategy
order types change.

Python instead asks one narrow, observable question for each unmatched current ideal order:

> During the configured recent window, has Rust emitted an execution-semantically equivalent order
> at a materially different price or quantity?

The proposed behavior is:

1. Keep per-symbol, bounded, RAM-only snapshots of valid normalized Rust ideal orders for ten
   minutes.
2. Reconcile current actual and ideal orders one-to-one under the existing tight tolerance and an
   exact execution-semantic cohort.
3. Cancel every unmatched stale actual order regardless of churn-gate state.
4. For each unmatched current ideal order, look independently in each prior valid snapshot for a
   deterministic one-to-one match in the same cohort:
   - a tight match is evidence of stability in that snapshot;
   - a wider-but-not-tight match is direct price/quantity churn evidence;
   - no wider match is uncertainty, not proof of replacement identity, and therefore fails open.
5. Always admit market orders, explicitly risk-critical orders, and orders within the configured
   final market-distance threshold.
6. Admit the first ten connector-bound, churn-evidenced ordinary create attempts account-wide in a
   rolling ten-minute window. Defer later far creations while the window is full.
7. Never retain a stale actual order while waiting. Cancel it and regenerate current Rust intent on
   a later cycle.
8. Execute ordinary replacements cancel-first by conflict scope: cancel, confirm authoritative
   open-order state, regenerate the Rust plan, and only then create.

This is an operational exchange-write economy layer. It does not alter Rust calculations, widen
the actual-order reconciliation tolerance, replace CCXT/exchange rate limiting, or make event logs
causal inputs.

## Problem

Some ideal limit orders move continuously because their inputs move continuously. EMA Anchor is the
current concrete example: both entries and closes may move with EMA bands, volatility, and
inventory. Changes just outside `live.order_match_tolerance_pct` can cause low-value cancel/create
traffic for hours while the desired orders remain far from market.

Static grids and fixed distant closes are different. Placing them once is useful because they can
catch a sharp move, while leaving them unchanged consumes no further write capacity. A universal
distance gate would suppress these harmless resting orders.

The gate should therefore react to observed Rust output, not to a maintained list of strategy or
`pb_order_type` names believed to be EMA-driven. That makes it work for future strategy kinds
without teaching Python their pricing logic.

## Goals

- Reduce sustained low-value order replacement traffic.
- Remain independent of strategy kind and pricing formula.
- Preserve distant orders for which no recent churn is evidenced.
- Share the allowance account-wide within one bot process because exchange write limits are
  generally broader than one symbol or position side.
- Preserve current Rust intent without synthesizing fallback orders or keeping stale substitutes.
- Keep risk-critical actions and near-market opportunities outside the economy restriction.
- Strengthen duplicate prevention and stale-plan handling around cancellation and creation.
- Make every admission, deferral, cancellation barrier, and RAM reset observable.

## Non-Goals

- Reimplement exchange-native or CCXT throttling.
- Coordinate separate bot processes, API keys, subaccounts, or IP-based limit pools.
- Infer EMA, volatility, grid, trailing, or strategy provenance.
- Establish durable order identities or causal `new`/`replacement`/`restoration` tracks.
- Persist history across restart.
- Preserve maker queue position for an order Rust changed beyond universal tolerance.
- Add exchange-native amend support in the first implementation.
- Guarantee that a deliberately deferred distant limit order cannot miss a sharp move.
- Treat ten attempts per ten minutes as an exchange-native rate limit.

## Normative Invariants

### Rust owns current intent

Only the current valid Rust plan may produce an order creation. Historical snapshots are evidence
for whether the current creation is churny; they are never a queue of orders to execute later.

The heuristic may omit an ordinary distant current ideal order temporarily. It may never:

- create an order absent from the current Rust plan;
- change its price, quantity, side, type, or reduce-only semantics;
- keep an out-of-tolerance actual order as its substitute; or
- weaken readiness, ownership, risk, HSL, WEL/TWEL, or authoritative-state barriers.

### Universal reconciliation tolerance stays universal

`live.order_match_tolerance_pct` remains the only accepted tolerance for deciding that an actual
order satisfies a current ideal order. The wider tracking tolerance is historical evidence only.
It must never widen exchange-order reconciliation.

### Cancellation is not gated

Every unmatched actual order remains eligible for cancellation. If creation is deferred, the
stale actual order is still removed. No replacement allowance or distance decision may leave a
known stale order resting.

### Missing inputs are not fabricated

Invalid, partial, or unavailable planning results append no history. A deliberate valid Rust result
with zero orders is a valid empty snapshot and cancels stale actual orders.

Missing or stale market data follows the existing scoped fail-closed creation contract. Distance
must not default to zero, infinity, a candle close, or another neutral substitute.

### Events remain diagnostic

The centralized live-event stream reports decisions but is not subscribed to by the gate. Causal
state is updated directly at planner/reconciler/executor boundaries.

## Exact Reconciliation Contract

### Cohort key

Actual-to-current-ideal matching and historical ideal association require equality of:

```text
symbol
position_side
order_side
reduce_only
execution_type
time_in_force / post_only execution semantics
pb_order_type
```

Price and quantity must additionally match within `live.order_match_tolerance_pct` for actual
reconciliation or within the explicitly selected historical tolerance for churn evidence.

Matching is deterministic and one-to-one. One actual or historical observation cannot satisfy two
current ideal orders.

### Why `pb_order_type` is required

Exact ordinary `pb_order_type` is mostly diagnostic today, but it is not universally non-causal.
HSL uses panic fill identity for cooldown/reset boundaries, replay-cache safety, latest-panic
selection, and RED-supervisor behavior. Other strategy kinds may acquire similar semantics.

Letting an actual order with the right price and quantity but the wrong type satisfy a current ideal
would preserve stale attribution into fill history and could silently become behaviorally wrong as
strategies evolve. The safer long-term contract is therefore exact normalized `pb_order_type`.

Accepted cost: a rare Rust transition between two order types at effectively the same price and
quantity causes a cancel/recreate. This is preferable to an `ideal_satisfied_by_existing_different_order_type`
escape hatch whose semantics every future strategy would need to understand.

The same exact key applies to historical association. A type transition therefore receives no
cross-type churn evidence and may get one fail-open creation. That is intentional: suppressing a
semantically distinct current order based on an older type would make `pb_order_type` causal in the
opposite and riskier direction. If repeated type alternation becomes a material write source, Rust
should expose a separately reviewed stable intent-family field rather than Python collapsing names.

If an exchange open-order payload lacks trustworthy type attribution, the connector must recover it
from the bot's client-order metadata or authoritative local execution record. Unknown does not equal
a known ideal type, and Python must not fabricate one. The implementation audit must cover every
connector before enabling the gate.

### Why authoritative `reduce_only` is required

`reduce_only` is an execution guarantee, not a diagnostic label. A side/position-side inference is
insufficient: multiple individually plausible close orders can over-close after another order fills
or after a manual/external position change. The exchange's reduce-only protection prevents a flip
or unintended exposure increase.

Therefore actual order normalization must use the authoritative exchange flag. Unknown does not
match either `true` or `false`; it blocks normal planning for the affected symbol until refreshed or
otherwise resolved under the error contract. Existing inference from side and position side must
not be carried into this implementation.

This is a prerequisite correction to the current reconciler, whose tight matching tuple omits
`reduce_only` and whose open-order snapshot derives it from side and position side. The churn gate
must not be layered on top of that inference.

### Why execution semantics stay exact

Market versus limit and time-in-force/post-only behavior change fill guarantees and taker risk.
They remain exact keys even when price and quantity match. This avoids treating a resting maker
order as equivalent to an immediately executable or cancellable order.

## Snapshot Boundary and Scope

Capture normalized immutable ideal-order tuples after Rust has produced executable live orders and
Python has applied deterministic exchange precision/normalization, but before reconciliation with
actual orders removes or annotates candidates.

Maintain an independent bounded deque per symbol using monotonic process time. A logical snapshot
contains only normalized fields needed by the matching contract plus diagnostic metadata. Deep
copies of full payload dictionaries are unnecessary.

The symbol processing universe is the union of:

- symbols with a valid current plan;
- symbols present in authoritative actual open orders;
- symbols with a held position; and
- symbols still present in recent history.

This union prevents a partial planner result from making actual orders or positions disappear from
consideration.

Planning validity is scoped per symbol. A failure for one symbol appends no snapshot and performs no
ordinary reconciliation for that symbol; healthy symbols continue. A valid empty plan for one
symbol appends an empty snapshot and removes stale orders for that symbol.

Append the current snapshot only after its decision, or otherwise exclude it from its own history.
Prune by the configured window. Consecutive identical snapshots may be compacted only if tests prove
the same logical presence answers at every relevant timestamp.

### Why history is per symbol but allowance is account-wide

Per-symbol history prevents one unhealthy or omitted symbol from poisoning all classification and
keeps matching bounded. The allowance remains account-wide because the scarce resource—private
order-write capacity—is normally shared across symbols for the account, UID, subaccount, or IP.

This bot-process scope is intentionally not a complete multi-process account limiter. Cross-process
coordination is a separate feature.

## Quantitative Churn-Evidence Heuristic

For each unmatched current ideal order outside the unconditional exemptions:

1. Partition it by the exact cohort key above.
2. For each prior valid snapshot in the active per-symbol window, newest first, perform a
   deterministic one-to-one association within that snapshot.
3. Prefer tight matches using `order_match_tolerance_pct` for both normalized price and quantity.
4. Among remaining observations, find wider matches using
   `order_replacement_churn_gate_tracking_tolerance_pct`.
5. If the current candidate has any wider-but-not-tight match, it has recent churn evidence.
6. If it has tight matches only, or no wider match, it has no proven churn evidence and is admitted.

The implementation may stop after finding wider-but-not-tight evidence. It need not build tracks,
tombstones, predecessor identities, or causal classifications.

### Deterministic matching

Within each snapshot and cohort, matching must maximize one-to-one matches, minimize a normalized
price/quantity distance cost, and apply explicit deterministic tie-breaking. Raw Rust list order,
dictionary iteration order, exchange IDs, and actual-order proximity are forbidden tie-breakers.

This matters for duplicate-price ladders: one historical observation must never be reused to mark
several current orders stable or churny.

### Why uncertainty fails open

No wider match may mean a genuinely new order, a ladder cardinality change, a very large move, a
history gap, or a restart. Python cannot distinguish those cases without inventing identity. It
therefore admits the current Rust order.

This knowingly misses some very large or discontinuous churn. The feature targets sustained small
drift, which is the observed long-term write-cost problem. A forgiving false negative is safer for
fills and simpler than a false-positive identity model.

### Why the wider tolerance defaults to 0.2%

The proposed default is `0.002` (0.2%), ten times the existing 0.02% reconciliation tolerance and
substantially narrower than the earlier 1% proposal. It is wide enough to associate ordinary EMA
drift while reducing accidental pairing between neighboring grid levels. This is a heuristic,
configurable, and must be tuned from fake-live and authorized live evidence rather than presented
as a universal market property.

### Equal-cardinality ladder rolls

A wholesale ladder roll may create wider associations and be treated as churn even when human
narrative would call the levels new. That is acceptable: the gate claims only recent quantitative
price/quantity instability, not durable order identity. Near-market admission, the first-ten
allowance, and fail-open unmatched orders bound the consequence.

## Admission Policy

### Unconditional exemptions

The churn gate always admits:

- market orders;
- HSL panic and the dedicated protective-panic path;
- orders Rust explicitly marks `execution_priority=risk_critical`; and
- limit orders whose final fresh market distance is at or inside
  `order_replacement_churn_gate_market_dist_pct`.

`reduce_only=true` alone is not an exemption. EMA-gated closes can churn too. Rust must explicitly
own risk priority for unstuck, WEL/TWEL, auto-reduce, graceful-stop, cooldown re-panic, and other
exposure-reducing families. The implementation audit must classify every current family rather than
letting Python infer urgency from names.

### Account-wide rolling allowance

The default allowance is ten churn-evidenced ordinary create attempts in a rolling ten-minute
window for one bot execution authority.

Count one logical order when it reaches the concrete connector create-call boundary, including
failed, rejected, timed-out, and ambiguous attempts and each logical member of a batch. Do not count
local deferrals or cancellations.

Near-market churn-evidenced ordinary attempts are always admitted but still count because they use
the same exchange write capacity. Market and explicitly risk-critical actions neither consume nor
wait for this economy allowance.

When one timestamp expires, at most one newly far attempt gains capacity before consuming that
slot. There is no bulk release and no persistent queue of deferred orders.

### Final market-distance check

Distance decisions made during planning are provisional. Recheck fresh signed distance as close as
practical to the connector call, after cancellations, configuration work, and authoritative
barriers. A candidate that moved near is admitted; one that moved far must have capacity or be
deferred. Missing/non-finite/stale market data defers an affected ordinary create.

### Final batch priority

Exemption from the churn gate does not bypass readiness or authoritative-state safety barriers.
After those barriers and the final distance check, sort risk-critical candidates before ordinary
candidates and only then apply `max_n_creations_per_batch`. Preserve the existing deterministic
distance/order key as the secondary ordering inside each priority class. An ordinary near-market
order must not displace a far risk-critical order merely because it sorts closer to market.

If risk-critical candidates alone exceed the connector cap, use deterministic ordering within that
class and regenerate remaining work from a fresh plan. Never silently expand a connector hard cap.

## Cancel-First Execution Contract

### Existing behavior to replace

The current flow in `src/live/executor.py` calls cancellations and then creations from one
precomputed plan. Cancellation selection may be truncated by
`max_n_cancellations_per_batch` while the creation list remains unchanged. That permits a new
ordinary order to be posted while its stale predecessor was not even sent for cancellation.
Existing recent-create suppression and state-change barriers help with ambiguous writes, but they
do not establish a general cancel-confirm-replan-create contract.

### Execution conflict scope

Use a broader scope for write sequencing than for order satisfaction:

```text
(symbol, position_side, order_side, reduce_only, execution_type)
```

`pb_order_type` and time-in-force are deliberately absent. A stale grid order and a new trailing
order, or a stale maker order and a changed execution style, can still compete for the same
position-side exposure and create duplicates during transition.

This broader scope is only a sequencing barrier. It does not make unlike orders equivalent during
reconciliation.

### Ordinary two-phase flow

For each planning wave:

1. Reconcile authoritative actual orders with the current valid Rust ideal.
2. If any unmatched stale actual exists in a conflict scope, suppress every ordinary creation in
   that scope for the entire wave—whether its cancellation is selected, truncated, succeeds,
   fails, or is reported absent.
3. Send the selected cancellation batch under existing limits and pacing.
4. Request authoritative open-orders confirmation for affected scopes and end those scopes' work
   for the wave.
5. On the next loop, refresh state, regenerate the Rust plan, reconcile again, and create only in
   scopes now confirmed clean.

Independent scopes without stale actual orders may create in the same wave. This avoids a global
one-cycle stall while preventing stale-predecessor duplicates.

Failed, ambiguous, not-found, already-filled, or already-cancelled results do not prove a scope is
clean. They require the existing authoritative confirmation behavior; terminal ambiguity that may
represent a fill refreshes balance, positions, open orders, and fills before replanning.

### Risk-critical exception

Dedicated market panic may bypass the ordinary dependency path entirely. Other Rust-marked
risk-critical limit orders remain ahead of ordinary orders in final batching and may create in the
same wave only if every stale cancellation in their conflict scope was selected and positively
acknowledged, with no truncation or ambiguity. Otherwise they replan through the fastest existing
authoritative path.

This exception avoids delaying emergency exposure reduction while still preventing an assumed
cancellation from producing duplicate risk orders.

### Why one-cycle latency is accepted

For ordinary orders, a fresh Rust plan after confirmed cancellation is simpler and safer than
carrying per-create dependency objects through batch truncation, partial success, fills, and
position changes. The affected scope pays one loop of latency; independent scopes and emergency
orders continue. This is an explicit reliability tradeoff, not an accidental delay.

## RAM-Only State and Restart Contract

The snapshot deque and rolling attempt timestamps live only in the bot instance and reset on bot or
process restart. Reset intentionally fails open: until new churn evidence accumulates, current Rust
orders may be placed normally.

That means restart temporarily reduces exchange-write savings. It does not authorize a non-current
order, retain a stale order, bypass risk/readiness checks, or queue old intent. A few replacements
after restart are acceptable because the target is hours of continuous churn, not restart-perfect
rate accounting.

Current `AGENTS.md` and `docs/ai/principles.md` broadly require decision-changing behavior to be
reproducible after restart. This plan cannot silently override them. Before runtime implementation,
Slice 0 must add and review this narrow canonical exception:

> An approved operational economy gate may keep RAM-only state when losing that state can only
> remove throttling and return execution toward the current Rust-ideal baseline. Reset must not
> create non-ideal orders, relax readiness/risk/ownership checks, preserve stale orders, or queue
> obsolete intent. Reset must be observable and covered by tests.

Plan approval selects that intended exception; runtime implementation remains blocked until the
canonical instruction and principle are updated together.

## Proposed Configuration

Add canonical `config.live` fields:

| Field | Default | Meaning |
|---|---:|---|
| `order_replacement_churn_gate_activation_count` | `10` | Rolling account-wide capacity for churn-evidenced ordinary create attempts. `0` disables churn gating. |
| `order_replacement_churn_gate_window_minutes` | `10.0` | Per-symbol evidence and account-wide attempt window. |
| `order_replacement_churn_gate_market_dist_pct` | `0.005` | Final market-distance threshold that always admits a limit creation; 0.5%. |
| `order_replacement_churn_gate_tracking_tolerance_pct` | `0.002` | Wider price/quantity tolerance used only for historical churn evidence; 0.2%. |

The existing `live.order_match_tolerance_pct`, normally `0.0002` (0.02%), remains the actual-order
equivalence and tight historical tolerance.

Validation:

- activation count is an integer greater than or equal to zero;
- window is finite and greater than zero when enabled;
- distance is finite, greater than or equal to zero, and less than one;
- tracking tolerance is finite, greater than `order_match_tolerance_pct`, and less than one; and
- canonical schema/preparation owns defaults.

### Retire the old initial-entry setting

Remove `live.initial_entry_exec_max_market_dist_pct` from canonical schema, templates, aliases,
examples, runtime, and current docs. It must not remain a silent alias: the new policy is temporal,
account-wide, and applies to entries and closes.

Preferred migration behavior is an actionable configuration error naming the replacement fields.
If compatibility requirements demand a one-release migration, reviewers must define it explicitly;
runtime consumers must not interpret the old value as the new policy silently.

## Exchange Rate-Limit Research and Ramifications

Official documentation was reviewed on 2026-07-19 and 2026-07-20. Limits and tiers change; the
implementation review must recheck the exact endpoints used by each connector.

| Exchange/model | Official behavior relevant to this plan | Ramification |
|---|---|---|
| Hyperliquid | REST has weighted IP limits. Address action allowance is tied to cumulative USDC volume, counts logical actions within batches, grants extra cancellation allowance, and may apply congestion/maker-share limits. [Official docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits) | Count logical actions, not HTTP requests. Do not suppress cancellations or fill-producing near-market/risk actions. |
| Bybit | Private limits use rolling per-second UID windows plus IP limits; batch consumption is based on order count. [Official docs](https://bybit-exchange.github.io/docs/v5/rate-limit) | The ten-minute economy policy supplements short-window pacing. |
| Bitget | Futures cancellation has UID-scoped endpoint limits. [Official docs](https://www.bitget.com/api-doc/classic/contract/trade/Batch-Cancel-Orders) | Existing connector pacing remains authoritative. |
| KuCoin | Private quotas use UID resource pools and endpoint weights over short windows. [Rate-limit docs](https://www.kucoin.com/docs-new/rate-limit), [futures order docs](https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order) | Header-adaptive pacing is separate work. |
| Gate.io | Futures placement/amend and cancellation have separate UID limits. [Official docs](https://www.gate.com/docs/developers/apiv4/en/) | High nominal capacity does not make indefinite low-fill churn useful. |
| WEEX | Place/cancel and trigger actions have endpoint-specific account/UID or IP limits. [Official docs](https://www.weex.com/api-doc/ai/QuickStart/LIMITS) | The gate targets sustained churn; CCXT remains responsible for bursts. |
| Binance USD-M | New orders consume account order-count limits; placement and cancellation have different request weights and batch bounds. Qualifying reduce-only/close and cancellation requests receive special overload treatment. [General info](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info), [trade endpoints](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#new-order) | Use USD-M, not Spot, assumptions. Preserve risk/cancel priority and count logical actions. |
| OKX | Trading limits are shared across REST/WS, separated by place/amend/cancel operation, and generally scoped by user plus instrument; subaccount fill-ratio rules affect new/amend traffic. [Official docs](https://www.okx.com/docs-v5/) | Native amend and adaptive fill-ratio optimization are later projects. |
| Paradex | Private APIs use account plus IP limits with leaky-bucket refill; batch operations may consume one unit for multiple orders. [Rate limits](https://docs.paradex.trade/api/general-information/rate-limits/api), [best practices](https://docs.paradex.trade/api/general-information/api-best-practices) | Logical action counting is deliberately conservative. |
| Defx | Official docs expose placement/cancellation and 429 behavior but do not publish a complete numeric model. [Official docs](https://docs.defx.com/docs/api-docs/developer-hub/rest-apis-documentation) | Do not invent exchange thresholds. |
| CCXT/generic fallback | CCXT rate limiting is per exchange instance; separate instances do not share limiter state. [Official manual](https://github.com/ccxt/ccxt/wiki/manual#rate-limit) | This per-bot gate cannot guarantee account/IP coordination across processes. |

Hyperliquid is the clearest reason not to call this a generic rate-limit budget. Its cancellation
allowance and requests-per-volume economics differ materially from rolling endpoint quotas. The
policy still helps by reducing low-fill far creations, but cancellations and risk/fill-producing
orders must retain priority.

## Live/Backtest Parity

Rust ideal generation is unchanged. The existing tiny universal tolerance remains the only reason
live may leave a slightly different actual order resting.

The explicit live-only difference is omission: after stale cancellation, live may temporarily omit
a far current ideal order with proven recent churn when the allowance is full. It never substitutes
an older price. This can miss a sharp move before the order becomes near-market or capacity returns.

The consequence is bounded by:

- fail-open behavior for no/uncertain history;
- ten initial churn-evidenced attempts per rolling window;
- unconditional final near-market admission;
- risk-critical and market exemptions;
- one-for-one capacity release; and
- immediate use of fresh current Rust intent on every later cycle.

This tradeoff must be measured in fake-live and, only with separate authority, live operation.

## Failure Semantics

- Invalid config fails startup.
- Invalid or unavailable symbol planning appends no snapshot and performs no ordinary actions for
  that symbol; valid symbols continue.
- A valid empty symbol plan is authoritative and may cancel all stale orders in that symbol.
- Unknown required `pb_order_type`, `reduce_only`, or execution semantics do not match a known ideal
  and fail closed for affected normal planning.
- Missing/stale final market data defers ordinary creation.
- Ambiguous create attempts count once when connector-bound and use existing confirmation guards.
- Failed, ambiguous, absent, or truncated cancellations block ordinary creation in their conflict
  scope and require confirmation/replanning.
- Event publication failure changes no trading decision.
- Existing exchange pacing, 429/backoff, readiness, ownership, risk, HSL, WEL/TWEL, mode, and
  authoritative-state barriers retain precedence.

## Observability

Add bounded diagnostic events/reason codes for:

- valid per-symbol snapshot observed, compacted, skipped, or reset;
- tight, wider-but-not-tight, absent, and ambiguous historical association;
- churn evidence present/absent;
- admission by exemption, distance, or rolling allowance;
- deferral by allowance or unavailable final distance;
- conflict-scope create suppression;
- cancellation acknowledgement, ambiguity, truncation, confirmation, and replan;
- final risk priority and connector-bound attempt accounting.

Safe payloads include symbol, position side, order side, normalized semantic keys, bounded cohort
hashes, rolling count, thresholds, distance, wave ID, and reason code. Do not emit credentials,
arbitrary exchange payloads, unbounded snapshot arrays, or raw exception text. Repeated steady-state
events use bounded periodic summaries.

## Accepted Tradeoffs

- Exact `pb_order_type` may cause rare type-transition churn; this preserves attribution and future
  strategy safety.
- Wider matching can flag an unrelated equal-cardinality ladder roll; the heuristic claims churn,
  not identity, and exemptions bound missed-fill risk.
- A move beyond wider tolerance may be admitted as uncertain; this favors fills and targets the
  observed repetitive small-drift problem.
- Ordinary cancel-first sequencing adds one loop of latency in an affected conflict scope; it
  removes stale-plan and truncated-cancel duplicate hazards.
- RAM reset temporarily restores baseline write churn; it does not weaken the current Rust plan or
  safety barriers.
- Account-wide means one bot execution authority, not multiple processes sharing exchange limits.

## Resolution of Review Findings

### Latest design questions

1. **Scoped planning outage:** history and validity are per symbol; the processing universe includes
   plans, actual orders, positions, and history. Healthy symbols continue. Recovery without history
   intentionally receives fail-open placement.
2. **Ambiguous order identity and equal-cardinality ladders:** authoritative narrative
   classifications and temporal tracks are removed. Policy uses only wider-but-not-tight
   quantitative evidence and explicitly accepts bounded false positives/negatives.
3. **`pb_order_type`:** it is restored as an exact semantic key after auditing current fill-history
   use. The rationale and unknown-attribution behavior are explicit.
4. **`reduce_only`:** it is an exact authoritative key; side/position-side inference is forbidden.
5. **Cancel/create races:** all ordinary creates in a conflict scope wait for cancel confirmation,
   fresh state, and a fresh Rust plan. Batch truncation cannot strand a dependent creation.

### Earlier findings retained

- The allowance governs churn-evidenced creations, never cancellations.
- Capacity releases one attempt at a time; no deferred-order burst or stale-intent queue exists.
- Final distance is checked close to the connector call.
- Risk-critical candidates sort ahead of ordinary candidates before final batch slicing.
- Already-absent cancellation triggers authoritative confirmation and replanning.
- `_context=replace`, order names, client IDs, and exchange IDs are non-causal for churn evidence.
- Every supported connector is researched directly or covered by the conservative CCXT contract.
- Binance assumptions are explicitly USD-M, not Spot.
- Restart behavior is an intentional bounded exception requiring a canonical contract change first.

## Required Tests and Evidence

### Matching and normalization

- deterministic one-to-one tight actual reconciliation, including duplicate prices/quantities;
- exact mismatch for `pb_order_type`, panic versus non-panic, `reduce_only`, execution type, and
  time-in-force/post-only semantics;
- unknown actual type/reduce-only fail-closed behavior for every connector adapter;
- no side/position-side inference of reduce-only;
- one historical observation cannot satisfy multiple current candidates;
- raw list reordering and dictionary order do not change outcomes.

### Per-symbol history and heuristic

- current snapshot never matches itself;
- scoped outage skips one symbol while healthy symbols continue;
- valid empty is distinct from unavailable planning;
- processing-universe union retains actual/position/history-only symbols;
- first observation and restart reset fail open;
- tight-only history admits; wider-but-not-tight history marks churn;
- no wider association admits as uncertain;
- price-only and quantity-only churn;
- neighboring grids, duplicate levels, growing/shrinking ladders, equal-cardinality rolls;
- evidence expires exactly with the rolling window;
- compaction, if implemented, preserves logical results across fast/slow planning cadence.

### Allowance and final admission

- ten attempts across multiple symbols share one process-wide allowance;
- eleventh far churn-evidenced ordinary create defers;
- one expired timestamp releases one slot;
- near-market churn attempts always admit and count;
- market and risk-critical orders always bypass and do not count;
- failed/rejected/timed-out/ambiguous connector-bound creates count exactly once;
- local deferrals count zero;
- final price movement near-to-far consumes capacity or defers and far-to-near admits;
- missing/stale/non-finite final market data defers without fabrication;
- concurrent candidates cannot oversubscribe current capacity.

### Cancel-first executor

- stale actual suppresses every ordinary create in its conflict scope;
- selected, truncated, failed, ambiguous, and absent cancellation outcomes;
- positive acknowledgement still requires next-cycle authoritative confirmation for ordinary work;
- fills and manual position changes regenerate different Rust intent before create;
- independent conflict scopes may create in the cancellation wave;
- dedicated market panic bypasses ordinary sequencing;
- risk-critical limit same-wave create requires every scoped cancel positively acknowledged;
- no ordinary create is sent from a pre-cancellation plan;
- existing recent-create/ambiguous-write guards remain effective without duplicate responsibilities.

### Integration

- static grid, moving EMA Anchor entry and close, trailing, HSL, WEL/TWEL, unstuck, auto-reduce,
  graceful stop, forager changes, hedge mode, and one-way mode fake-live scenarios;
- batch saturation and connector-specific create/cancel bounds;
- RAM reset event and post-restart fail-open behavior;
- bounded event output and event-sink failure isolation;
- connector-bound action counts and omitted-order windows with gate enabled/disabled.

## Implementation Slices After Approval

### Slice 0: canonical RAM-only exception

- Update `AGENTS.md` and `docs/ai/principles.md` with the reviewed narrow economy-gate exception.
- Specify reset observability and tests.
- Do not begin decision-changing RAM implementation before this lands.

### Slice 1: exact semantic normalization

- Add canonical config fields, validation, templates, migration behavior, and changelog entry.
- Audit every connector for authoritative `reduce_only`, normalized `pb_order_type`, execution type,
  and time-in-force/post-only data.
- Audit Rust order families for explicit ordinary/risk-critical priority.
- Retire `initial_entry_exec_max_market_dist_pct`.

### Slice 2: pure per-symbol evidence helper

- Implement immutable snapshots, validity boundaries, pruning/optional compaction, deterministic
  one-to-one tight/wider association, and account-wide rolling attempt accounting.
- Keep helpers independent of live events and exchange I/O.

### Slice 3: cancel-first reconciler/executor overhaul

- Separate exact satisfaction cohorts from broader execution conflict scopes.
- Suppress ordinary scoped creations whenever stale actual orders exist.
- Cancel, request authoritative confirmation, end affected scopes, and replan next cycle.
- Preserve independent-scope concurrency and explicit emergency paths.
- Consolidate overlapping recent-create, cancellation, and state-change delay machinery where tests
  prove simplification is safe.

### Slice 4: final admission and priority

- Recheck final fresh distance.
- Apply allowance reservations atomically at connector boundary.
- Sort risk-critical before ordinary candidates, then preserve deterministic existing secondary
  ordering and native batch caps.

### Slice 5: events and fake-live validation

- Register bounded diagnostic events and prove sink isolation.
- Run the complete multi-strategy, multi-symbol, connector, restart, ambiguity, and batch matrix.
- Quantify write reduction and parity cost before proposing any default change.

No authenticated exchange probe or live bot run is authorized by plan approval. Those require
explicit authority in the implementation task.

## Implementation Validation

At minimum, the implementation PR must run:

- focused Rust risk-priority tests;
- Rust tests plus rebuilt and verified Python extension where Rust metadata changes;
- config schema/default/template/CLI/roundtrip and retired-setting migration tests;
- pure normalization, matching, per-symbol history, rolling-window, and reset tests;
- reconciler/executor cancellation, ambiguity, conflict-scope, final-distance, and batch tests;
- static/moving/mixed-risk multi-symbol fake-live scenarios across every connector harness;
- live-event registry/query/sink-failure/bounded-projection tests;
- `PYTHONPATH=src python src/tools/check_ai_docs.py`;
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`;
- `git diff --check`.

Report exact base/head SHAs and cycle-level evidence containing current normalized Rust ideal,
actual order state, historical tight/wider result, churn evidence, cancellation outcome, conflict
scope state, rolling usage, final distance/priority, and connector-bound action.

## Rollout and Measurement

After offline validation and separate live authorization:

1. Run one controlled account with structured DEBUG execution evidence.
2. Measure ideal stability/churn, admissions, deferrals, cancellations, fills, 429s, ambiguity, and
   conflict-scope latency.
3. Prove the gate never leaves an out-of-tolerance stale actual order resting.
4. Inspect static grids, changing ladders, EMA Anchor entries/closes, trailing, forager changes, and
   one-for-one capacity release.
5. Inspect Hyperliquid separately because its requests-per-volume and cancellation economics differ.
6. Recheck Binance USD-M overload/risk behavior on the actual endpoints.
7. Tune defaults only from evidence; do not hide exchange-specific policy in adapters.

## Reviewer Checklist

Reviewers are explicitly asked to challenge intent, architecture, edge cases, unintended
consequences, and connector assumptions—not merely wording:

- Is the exact reconciliation key complete, especially `pb_order_type`, authoritative
  `reduce_only`, and time-in-force/post-only semantics?
- Can every connector recover those fields without fabrication, and is fail-closed scope correct?
- Is the broader execution conflict scope sufficiently conservative without stalling unrelated
  work?
- Can any cancellation truncation, ambiguity, fill, manual action, or race still send an ordinary
  order from a stale Rust plan?
- Are the market-panic and risk-critical-limit exceptions narrow enough and prioritized before
  batching?
- Does per-symbol validity isolate outages while the union processing universe preserves actual
  orders and positions?
- Can one historical order satisfy multiple current candidates?
- Are deterministic wider associations sensible for duplicate grids, cardinality changes, and
  equal-cardinality rolls?
- Is failing open on no wider match preferable to trying to infer identity for large moves?
- Is 0.2% an appropriate initial evidence tolerance, and what fake-live cases could falsify it?
- Does the account-wide attempt counter model each connector reasonably, especially Hyperliquid?
- Are near-market attempts correctly always admitted yet still counted?
- Can a stale order ever remain while a replacement is gated?
- Is the one-cycle ordinary latency acceptable for parity and missed-fill risk?
- Is the proposed RAM-only canonical exception truly bounded to operational economy and incapable
  of weakening safety after reset?
- What interactions remain with config reload, forager symbol churn, HSL, WEL/TWEL, unstuck,
  auto-reduce, graceful stop, hedge mode, and one-way mode?

Approval accepts the design and its documented tradeoffs. It does not authorize runtime code,
authenticated probes, live trading, or unreviewed changes to canonical safety contracts.
