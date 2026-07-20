# Rust-Ideal-Aware Order Replacement Churn Gate Plan

## Status

Planning and review specification only. This document intentionally makes no runtime, schema,
configuration, or test implementation changes.

The implementation must not begin until reviewers have challenged the intent, cohort and generation
model, safety exemptions, cancellation/create dependency contract, exchange assumptions,
live/backtest implications, edge cases, and operational consequences in this revision.

This revision supersedes the original closest-price replacement-pair design. It incorporates the
architecture feedback on PR #1336 and the subsequent design discussion.

## Decision Summary

The proposed implementation is a hybrid of an account-wide rolling replacement budget and dynamic
order-behavior detection:

1. Observe changes between consecutive normalized Rust ideal-order generations.
2. Classify current desired creations as fresh/static, restoration, or replacement-derived without
   pairing them to individual exchange orders.
3. Allow first-time and unchanged/static ideal orders to be placed and rest at any distance.
4. Allow at most the available account-wide rolling capacity for far, non-exempt,
   replacement-derived limit creations.
5. Always allow a creation whose current desired price is within
   `live.order_replacement_churn_gate_market_dist_pct` of fresh market price.
6. Always allow market, panic, and explicitly Rust-classified risk-critical creations.
7. Never retain an actual order outside `live.order_match_tolerance_pct` because replacement
   creation is deferred. The churn gate never filters required cancellation.
8. Keep a deferred generation replacement-derived after its former order has disappeared, so it
   cannot escape the gate merely because ordinary reconciliation now labels it `new`.
9. Consume rolling capacity at the concrete connector-call boundary. Re-admitted far creations
   consume newly freed capacity one for one, preventing a release burst.
10. Keep all behavior state in RAM for one bot instance and deliberately reset it on restart under
    the narrow exception proposed below.

## Problem

Some ideal limit prices move frequently because their inputs move frequently. EMA Anchor is the
current example: both entry and close prices can change with EMA bands, volatility, and inventory.
A one-tick desired-price change outside `live.order_match_tolerance_pct` can therefore produce a
cancel/create cycle even when the desired order remains far from market and unlikely to fill.

Static grids and distant fixed take-profit orders have different behavior. Their first placement is
useful because they may catch a sharp market move, while leaving them resting creates no sustained
write traffic. A universal distance gate would unnecessarily remove that benefit.

The executor should not need an allowlist of known EMA-derived order types. It should observe
whether Rust's executable ideal intent changes materially. New strategy kinds and order families
then participate without changes to the churn gate.

This mechanism is an exchange-write economy layer. It does not replace CCXT pacing, exchange-native
rate limiting, retry/backoff, ambiguous-write confirmation, or batch-size controls.

## Alternatives Considered

### Universal distance gating

Apply the market-distance threshold to every limit creation at all times.

- Advantage: smallest and easiest implementation.
- Rejected as the default because it prevents harmless static orders from resting and increases
  the chance of missing a sharp move through a distant ideal price.

### Known churny order-type allowlist

Apply distance gating only to preclassified order types.

- Advantage: static families can remain on the book.
- Rejected because Python would need advance knowledge of strategy internals, EMA switches, and
  every future order family. This is brittle and does not self-correct from observed behavior.

### Undifferentiated account-wide replacement budget

Allow a number of replacements per window, then distance-gate all far creations.

- Advantage: strategy-agnostic and simple.
- Rejected alone because unrelated account churn could defer a first-time static order that would
  otherwise rest without further writes.

### Individual exchange-order behavior learning

Infer which actual order replaced which prior actual order and learn churn per order identity.

- Advantage: potentially precise.
- Rejected because current `_context=replace` metadata is explicitly best-effort, exchange client
  IDs are not stable ideal identities, and grids, partial fills, and changing ladder counts make
  causal individual pairing ambiguous.

### Selected hybrid

Use Rust-ideal cohort generations to determine whether a desired creation is replacement-derived,
then apply an account-wide rolling admission budget only to those creations. This preserves static
resting orders while remaining strategy-agnostic and self-correcting.

## Goals

- Stop hours of low-value cancel/create traffic caused by moving ideal limit prices.
- Preserve first-time and unchanged static limit orders at any market distance.
- Protect the write budget shared by symbols in one exchange account/bot instance.
- Work without interpreting strategy kind, EMA use, EMA speed, or known order-family names.
- Keep Rust as the source of ideal orders, order-family identity, and risk priority.
- Never keep an out-of-tolerance actual order merely because replacement creation is deferred.
- Always admit near-market and explicitly risk-critical actions.
- Avoid causal dependence on best-effort logging annotations or diagnostic event delivery.
- Pace release from the gate without a persistent queue or Python-side strategy priority.
- Make classification, budget use, bypasses, deferrals, and reset visible in bounded live events.

## Non-Goals

- Reimplement an exchange's native rate limiter.
- Replace CCXT request pacing, endpoint weights, retries, or backoff.
- Coordinate multiple Passivbot processes that share an API key, account, subaccount, or IP.
- Persist, restore, or reconstruct churn state after process restart.
- Change Rust prices, quantities, strategy calculations, or backtest order generation.
- Preserve maker queue position for a materially changed Rust ideal order.
- Guarantee that a deferred far order could not have filled during a market gap.
- Suppress same-ideal creation retries caused by exchange rejection or ambiguous state; existing
  recent-execution, confirmation, and retry contracts own that separate failure mode.
- Implement exchange-native amend support in the first slice.
- Treat the initial 10-per-10-minute values as exchange-native limits.

## Safety Invariants

### Universal tolerance remains authoritative

Rust remains the source of the current ideal order. An actual order may satisfy that ideal only
when exact reconciliation or `live.order_match_tolerance_pct` says it is equivalent. The current
default tolerance is `0.0002`, or 0.02%.

The churn gate runs after normal executable-order normalization and equivalence matching. It must
not widen this tolerance or keep a stale actual order.

### Cancellation and creation are asymmetric

The churn gate only filters creation:

- every out-of-tolerance actual order remains in the cancellation plan;
- symbol retirement, forager deselection, mode changes, and graceful cancellation remain allowed;
- existing `max_n_cancellations_per_batch` and connector pacing may delay a cancellation, but the
  churn gate itself never does;
- a deferred replacement leaves no stale order resting once its cancellation is executed;
- a replacement creation must not be sent before the current wave's required stale cancellation
  dependencies reach the accepted connector outcome boundary.

### Near-market creation is always admitted

For every non-market limit creation, calculate the existing side-aware signed market distance from
a fresh market snapshot. A creation is always admitted when:

```text
order_market_diff(order.side, order.price, market_price)
    <= live.order_replacement_churn_gate_market_dist_pct
```

This includes marketable negative distances. Distance is measured for the current desired creation,
not the stale actual order.

The near-market bypass is intentionally allowed to exceed the rolling activation count. Sustained
near-market churn can therefore keep far replacement-derived orders gated. CCXT and exchange-native
limits remain responsible for hard request pacing.

### Required market data is never fabricated

The gate reuses the canonical pre-create market snapshot. Missing, stale, non-finite, or invalid
market data follows existing scoped create deferral. It must not default distance to zero or
infinity. Risk-critical paths use their existing readiness contracts.

## Proposed Configuration

Add these canonical `config.live` fields:

| Field | Default | Meaning |
|---|---:|---|
| `order_replacement_churn_gate_activation_count` | `10` | Rolling count at which far, non-exempt, replacement-derived limit creations require a free slot. `0` disables the churn gate. |
| `order_replacement_churn_gate_window_minutes` | `10.0` | Rolling RAM window for admitted replacement-derived connector-call attempts. |
| `order_replacement_churn_gate_market_dist_pct` | `0.005` | Side-aware market-distance threshold that always admits a desired limit creation. `0.005` is 0.5%. |

`activation_count` is deliberately not called `max_replacements`. Near-market bypasses can make the
observed deque exceed the threshold, and required cancellations are never capped by this mechanism.

Validation requirements:

- activation count is an integer greater than or equal to zero;
- window is finite and greater than zero when the gate is enabled;
- market distance is finite, greater than or equal to zero, and less than one;
- defaults are owned by canonical schema/preparation, not repeated in runtime consumers.

### Retirement of the initial-entry setting

`live.initial_entry_exec_max_market_dist_pct` is removed from the canonical schema, template, CLI
aliases, examples, runtime, and current user documentation. It must not remain as a silent alias:
the old field was unconditional and initial-entry-only, while the new field is one part of a
dynamic strategy-agnostic system.

Recommended compatibility behavior is an actionable configuration error naming the three new
settings. Any migration tool must add the count/window defaults explicitly. Reviewers should decide
whether a released-version compatibility window is required.

Historical release notes and historical live events remain historical.

## Rust-Ideal Behavior Model

### Observation point

Observe executable ideal orders after Rust has applied its live market constraints, Python has
converted the output to API-ready order dictionaries, and invalid/duplicate zero-quantity intents
have been removed, but before comparison with actual exchange orders mutates or annotates them.

This observes the orders that would cause live writes, rather than raw sub-tick Rust movement that
cannot change an exchange order.

### Behavior cohort

Group normalized ideal orders by:

```text
(symbol, position_side, order_side, reduce_only, opaque_rust_order_family)
```

`opaque_rust_order_family` may initially use the existing normalized `pb_order_type`. Python uses
only exact equality and never interprets family names or maintains an allowlist. A future Rust
`intent_group_id` may replace it if review shows that existing family identity is too broad.

Execution type is part of the generation contents rather than the cohort key. Market orders are
exempt regardless.

### Ideal generation

An ideal generation is the cohort's deterministic multiset of normalized desired orders. Compare
successive generations using one-to-one deterministic multiset matching and the existing universal
price/quantity tolerance.

The comparison must be order-independent. Do not hash raw list ordering or pair by closest actual
exchange price. Suggested implementation is deterministic minimum-cost or price-sorted multiset
matching within one cohort, with explicit tie breaking and tests for equal prices, grids, quantity
changes, and changing order counts.

### Classification

For a cohort:

- `fresh`: first observation with no retained tombstone; creations do not consume the budget;
- `unchanged`: current generation is tolerance-equivalent to the prior generation;
- `restoration`: the ideal generation is unchanged but a desired actual order is absent after a
  fill or authoritative exchange-state transition; creation does not consume the budget;
- `replacement_derived`: current generation differs materially from the prior generation;
- `removed`: the cohort disappeared; cancellation proceeds, but no replacement creation exists.

A removed cohort retains a bounded tombstone for the rolling window. Reappearance before tombstone
expiry is replacement-derived, preventing rapid forager/mode disappearance and reappearance from
being treated repeatedly as first placement. Reappearance after expiry is fresh.

### Pending replacement generation

When a replacement-derived generation is not fully admitted, retain only the latest generation in
RAM. It stays replacement-derived until its required creation attempts reach the connector boundary
or Rust supersedes it with another generation.

This state prevents the next cycle from reclassifying the desired order as a free `new` creation
after the prior actual order has been cancelled. Superseded prices and quantities are discarded;
Python never creates a stale pending ideal.

## Risk-Critical Exemptions

The following creations always bypass the economy gate:

- market orders;
- HSL panic orders and the dedicated protective-panic reconciliation route;
- any order Rust explicitly marks `execution_priority=risk_critical`;
- any future action whose canonical risk contract explicitly requires immediate admission.

`reduce_only=true` alone is not an exemption. Ordinary EMA Anchor closes and take-profit orders can
be churny and are part of the problem being solved.

Before implementation, audit and explicitly classify unstuck, WEL/TWEL, auto-reduce, graceful-stop
close creation, cooldown re-panic, and other exposure-reducing families. Rust owns the classification;
Python must not infer it from `pb_order_type` strings. The safe default for a missing required
priority field is a startup/schema failure, not silently ordinary or silently exempt.

Risk-critical bypasses do not consume the economy ledger. Their connector calls remain subject to
exchange/CCXT rate controls and remain observable separately.

## Account-Wide Rolling Admission Ledger

Maintain in RAM:

```text
replacement_create_attempt_timestamps: deque[monotonic_time]
cohort_previous_generations: map[cohort_key, generation]
cohort_pending_generations: map[cohort_key, latest_pending_generation]
cohort_tombstones: map[cohort_key, removed_at_monotonic]
current_wave_reservations: temporary integer
```

The account scope is one live bot instance. There is no per-symbol or per-position-side budget.

Count one ledger item when a non-exempt replacement-derived limit creation reaches the concrete
connector-call boundary. Count failed, rejected, timed-out, and ambiguous calls because they may
consume exchange resources. A local deferral consumes nothing. A batch of `n` order actions counts
as `n` ledger items even when the exchange charges one HTTP request.

Near-market replacement-derived creations are always admitted and are still recorded. The deque
may therefore exceed `activation_count`; this is expected and keeps far actions gated while the
account is experiencing sustained useful/near-market churn.

Fresh, unchanged, restoration, market, and risk-critical creations do not consume this ledger.

## Decision And Execution Flow

For each normal cycle:

1. Obtain normalized executable Rust ideal orders.
2. Build behavior cohorts and compare them with prior ideal generations.
3. Record replacement-derived generations and bounded removal tombstones.
4. Snapshot authoritative actual open orders and perform exact reconciliation.
5. Apply `live.order_match_tolerance_pct`.
6. Keep every remaining required cancellation in the plan.
7. Attach cohort/generation classification to desired creations. Do not use diagnostic
   `_context=replace` as causal input.
8. Prune ledger timestamps and tombstones outside the rolling window.
9. Apply explicit market/risk-critical exemptions.
10. Fetch and validate the existing pre-create market snapshot.
11. For each non-exempt replacement-derived limit creation in deterministic existing priority:
    - admit it when desired signed distance is within the market threshold;
    - otherwise admit it only while historical count plus current-wave reservations is below
      `activation_count`;
    - otherwise defer the latest cohort generation.
12. Fresh, unchanged, and restoration creations follow existing gates without consuming the
    replacement ledger.
13. Apply cancellation batch selection before final replacement creation selection.
14. Bind every same-wave replacement creation to the required stale cancellation actions actually
    selected for that connector batch.
15. Execute selected cancellations first.
16. Admit a dependent replacement creation only when its cancellation dependencies reached the
    accepted successful/confirmed-absent outcome boundary. Ambiguous or failed cancellation
    outcomes defer creation pending authoritative state.
17. At each admitted replacement create connector call, convert its reservation into a monotonic
    ledger timestamp.
18. Discard unused reservations and retain only the latest still-pending Rust generation.

Conceptually:

```text
desired creation
    |
    +-- market or risk-critical ------------------------------> allow
    |
    +-- fresh / unchanged restoration ------------------------> normal gates; allow
    |
    +-- replacement-derived
          |
          +-- desired price within market distance -----------> allow; record attempt
          |
          +-- rolling capacity available ----------------------> allow; record attempt
          |
          +-- rolling capacity exhausted ----------------------> cancel stale dependency;
                                                                 defer latest creation
```

## Cancellation/Create Batch Dependency Contract

Current cancellation and creation caps are applied independently in different executor functions.
The new implementation must not assume a planned cancellation was sent merely because it existed in
the pre-batch plan.

Use a cohort-generation dependency rather than inferred individual order replacement identity:

- each replacement-derived generation lists the stale actual cancellation action IDs required
  before its creations may be sent;
- cancellation batch selection retains current protective priority and existing limits;
- a creation whose dependency was truncated is deferred;
- a dependency that failed or is ambiguous defers creation until authoritative state resolves it;
- when a prior cycle already removed every stale actual order, the pending latest generation has no
  current cancellation dependency and is controlled only by admission capacity and other gates;
- fresh independent creations have no cancellation dependency.

This may defer a whole multi-order cohort when only part of its stale set was selected. That
conservative cohort barrier is preferable to posting a new ladder alongside an out-of-tolerance
old ladder. Reviewers should decide whether a more granular Rust intent identity is worth the added
complexity.

## Release, Fairness, And Ordering

There is no binary release that admits every deferred far order when one timestamp expires.

Each far replacement-derived creation requires a free rolling slot and immediately reserves it.
If one timestamp expires, at most one far creation is newly admitted. If several expire together,
up to that many may be admitted, still bounded by existing creation batch limits.

Candidates use existing deterministic order sorting and batch priority. Python must not add a new
strategy priority. Risk-critical and near-market bypasses remain independent.

There is no persistent queue. Every cycle recomputes the latest Rust ideal, so obsolete deferred
generations disappear and static behavior self-corrects naturally.

## RAM Reset And Canonical Exception

All behavior generations, tombstones, pending state, and ledger timestamps reset on bot restart.
The first observed post-restart generation is fresh. A few initial replacement cycles after restart
are an accepted consequence because the objective is long-duration churn relief, not durable rate
limit accounting.

This is decision-changing live execution state and conflicts with the repository's general
restart-reproducibility rule unless a narrow exception is approved. The implementation must add a
canonical decision with this boundary:

> Transient exchange-write economy state may reset on restart only when it never changes Rust ideal
> orders, never retains an out-of-tolerance actual order, never gates cancellation or risk-critical
> creation, only defers ordinary non-near-market creation, has bounded startup behavior, and is
> observable and explicitly tested.

Approval of this plan explicitly approves that narrow exception in intent. The implementation PR
must still record it in the canonical decisions/contracts before enabling the runtime behavior; if
reviewers do not approve the exception, the RAM-only design is blocked rather than silently changed
to persistent or exchange-state-derived accounting.

Required restart tests:

- state is empty after restart;
- first generations are treated as fresh;
- existing actual orders still undergo exact/tolerance reconciliation;
- stale actual orders are still cancelled;
- risk-critical and near-market orders remain admitted;
- startup creation batch limits bound the reset burst;
- a reset event documents that prior churn history was discarded.

Repeated restarts can intentionally or accidentally bypass the long-duration economy gate. This is
an operational limitation, not a supported way to increase exchange throughput.

## Centralized Live-Event Integration

The live-event pipeline is observability-only. Causal state updates occur at the same canonical
planner/executor hooks that emit events; no trading decision subscribes to an event sink.

Proposed bounded events or reason codes:

- cohort generation transition: fresh, unchanged, replacement-derived, removed, superseded;
- replacement creation admitted, with admission source `capacity` or `near_market`;
- replacement creation deferred, reason `account_order_replacement_churn_gate`;
- gate threshold transition and rolling usage;
- restart reset;
- dependent create deferred because cancellation was truncated, failed, or ambiguous.

Safe fields include cohort hash, generation number/hash, symbol, position side, order side,
reduce-only, opaque family, execution priority, rolling count, threshold, window, desired distance,
distance threshold, action/wave correlation, and bounded counts. Do not expose raw order arrays,
prices/quantities in summary events, exchange payloads, arbitrary exception text, or credentials.

Repeated cohort deferrals use bounded periodic summaries. Structured detail remains queryable even
when console/text projection is suppressed. Event failure must not alter classification, state,
cancellation, creation, or release.

## Exchange Rate-Limit Research And Ramifications

Official documentation was reviewed on 2026-07-19. Limits and account tiers change;
implementation reviewers must recheck the exact endpoints used by each connector.

| Exchange/model | Official behavior relevant to the plan | Ramification |
|---|---|---|
| Hyperliquid | REST has a weighted IP pool. Address action allowance is tied to cumulative USDC volume, begins with a buffer, counts each action in a batch, grants additional cancellation allowance, and may apply maker-share congestion limits. [Official docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits) | Count logical connector actions, not HTTP batches. Near-market/fill-producing activity is economically different from unfilled churn, supporting the universal near-market bypass. Never withhold stale cancellation. |
| Bybit | Private limits use rolling per-second UID windows plus IP limits; batch usage is based on order count. [Official docs](https://bybit-exchange.github.io/docs/v5/rate-limit) | The 10-minute gate reduces sustained waste but does not replace short-window pacing. Count batch elements. |
| Bitget | Futures batch cancellation is UID-scoped with endpoint-specific limits. [Official docs](https://www.bitget.com/api-doc/classic/contract/trade/Batch-Cancel-Orders) | Existing connector batching/pacing remains authoritative; the gate reduces desired writes. |
| KuCoin | Private quotas use UID resource pools and endpoint weights over short windows; headers expose remaining/reset capacity. [Rate-limit docs](https://www.kucoin.com/docs-new/rate-limit), [futures order weight](https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order) | The generic rolling gate is supplementary. Header-adaptive behavior remains a separate future project. |
| Gate.io | Futures placement/amend and cancellation have separate UID limits, and Gate documents behavior/fill-efficiency controls in parts of its API. [Official docs](https://www.gate.com/docs/developers/apiv4/en/) | High nominal request capacity does not make low-fill churn harmless. Verify exact futures behavior controls. |
| WEEX | Place/cancel and trigger actions use endpoint-specific account/UID or IP limits; current contract documentation states 10 requests/s for relevant actions. [Official docs](https://www.weex.com/api-doc/ai/QuickStart/LIMITS) | The gate targets sustained churn while CCXT owns the burst ceiling observed in live WEEX testing. |
| Binance | Binance uses interval `REQUEST_WEIGHT` and `ORDERS` limiters, IP weights, account order counts, and fill-sensitive unfilled-order behavior. Product/endpoint values vary. [Official docs](https://developers.binance.com/en/docs/products/spot/rest-api) | Recheck USD-M endpoints; do not infer one universal futures number. Fill-producing restoration should not be treated like moving-price replacement churn. |
| OKX | Trading limits are shared across REST/WS, separated by place/amend/cancel operation, generally scoped by user plus instrument. Batch limits count orders, while subaccount and fill-ratio rules additionally affect new/amend traffic. [Official docs](https://www.okx.com/docs-v5/) | Per-order accounting is conservative. Cancellation must remain independent. Fill ratio reinforces preserving static/restoration orders and reducing low-fill replacements. Native amend is a possible later optimization. |
| Paradex | Standard private order APIs use per-account limits plus an IP-wide limit and leaky-bucket refill. Official best practices state batch operations can consume one rate unit for up to 50 orders; accounts also have open-order limits per market. [Rate limits](https://docs.paradex.trade/api/general-information/rate-limits/api), [best practices](https://docs.paradex.trade/api/general-information/api-best-practices), [open-order limits](https://docs.paradex.trade/api/general-information/rate-limits/open-orders-per-account) | Logical action count is deliberately more conservative than batch request units. Verify standard versus interactive JWT behavior. Batch efficiency and native modify support are future connector-specific opportunities. |
| Defx | Official REST documentation supports private placement/cancellation and 429 responses but currently labels numeric rate limits `Coming Soon`. [Official docs](https://docs.defx.com/docs/api-docs/developer-hub/rest-apis-documentation) | Do not invent native thresholds. Keep CCXT pacing/backoff enabled, treat defaults as an economy policy only, and require live observation before exchange-specific tuning. |
| CCXT/generic fallback | CCXT enables a per-instance limiter and documents leaky-bucket and optional rolling algorithms; separate exchange instances do not share limiter state. [Official manual](https://github.com/ccxt/ccxt/wiki/manual#rate-limit) | Keep the limiter enabled. The RAM gate is bot-instance-wide, not truly account/IP-wide across processes. Generic fallback connectors receive only the portable economy contract, not exchange-specific guarantees. |

## Live/Backtest Parity Analysis

Rust ideal orders remain unchanged. The universal order-match tolerance remains the only case where
live leaves an actual order at a slightly different price from the current ideal.

The churn gate introduces an explicit live execution exception: after removing a stale actual
order, live may temporarily have no corresponding far replacement on the exchange. This can miss a
gap or wick through the ideal before the next fresh snapshot and create cycle.

The revised design reduces that exception relative to a universal gate:

- first-time static orders may rest at any distance;
- unchanged restoration after a fill may be recreated at any distance;
- only materially changed replacement-derived generations are budgeted;
- near-market and risk-critical creations are always admitted;
- no stale old order remains as a substitute for the missing current ideal.

Required parity analysis must quantify missed-fill windows in deterministic fake-live scenarios and
compare current Rust ideal, actual exchange state, cancellation outcome, cohort generation, budget
state, and admitted creation for every cycle. Python must never synthesize a fallback order or price.

## Failure And Safety Semantics

- Invalid configuration fails startup validation.
- Missing required Rust order-family or execution-priority metadata fails at the schema boundary.
- Missing required market data uses existing scoped create deferral.
- Event publication failure is diagnostic only.
- Connector-bound failed or ambiguous replacement creates consume the economy ledger.
- Failed or ambiguous cancellation dependencies block the associated creation pending authoritative
  state; they never count as proof that the stale order disappeared.
- Every cancellation outside tolerance remains eligible regardless of ledger exhaustion.
- Existing duplicate-cancel, recent-execution, state-change, and confirmation protections remain.
- Exchange 429/backoff policy may defer more actions than this gate.
- Low balance, HSL, risk, modes, and readiness retain their current precedence.

## Review-Finding Resolution Map

This revision addresses all unique findings on the original PR head:

1. Restart reproducibility: proposes an explicit narrow canonical exception and restart tests.
2. Risk-reducing closes: adds Rust-owned `risk_critical` priority; does not blanket-exempt every
   reduce-only order.
3. Best-effort replacement pairing: removes `_context=replace` from causal classification and uses
   consecutive Rust ideal generations.
4. Misleading `max_replacements`: renames the setting to an activation count and states why the
   deque may exceed it.
5. Release burst: every re-admitted far creation consumes a newly free rolling slot.
6. Missing connector research: adds OKX, Paradex, Defx, and generic CCXT scope.
7. Cancel/create batch split: adds explicit cohort-generation cancellation dependencies after batch
   selection and before creation.

Review threads should not be resolved merely because this text exists. Reviewers must verify that
the new contracts actually answer the safety concern.

## Edge Cases Requiring Explicit Tests

1. First observation of a static far grid is fresh and allowed.
2. An unchanged static grid remains exact/tolerance matched and consumes no ledger capacity.
3. Recreating the same unchanged ideal after a confirmed fill is restoration and allowed.
4. A materially moved price becomes replacement-derived.
5. A quantity-only material change becomes replacement-derived.
6. Sub-tick or within-tolerance movement does not create a new generation.
7. Multi-order cohorts compare independent of list order.
8. Equal-price and equal-quantity ties resolve deterministically.
9. Grid expansion, contraction, and partial fills do not mutate unrelated cohorts.
10. Removed cohorts cancel actual orders without consuming create capacity.
11. Reappearance inside tombstone lifetime is replacement-derived; after expiry it is fresh.
12. A deferred generation stays replacement-derived after the old actual order disappears.
13. A superseding Rust generation replaces the pending one; no stale desired price is created.
14. Ten admitted far replacements across symbols exhaust one account-wide threshold.
15. An eleventh far replacement is cancelled and deferred.
16. A near-market replacement is admitted while exhausted and recorded even above threshold.
17. One expired timestamp admits at most one newly far replacement before recording a new one.
18. Market and risk-critical creations bypass the gate.
19. Ordinary reduce-only EMA Anchor closes remain eligible for dynamic gating.
20. HSL panic and every reviewed risk-critical order family remain ungated.
21. Missing/stale/non-finite market data uses existing fail-closed creation behavior.
22. Signed distance is correct for buys and sells on long and short position sides.
23. A cancellation truncated by batch capacity blocks its dependent creation.
24. A failed or ambiguous cancellation blocks dependent creation until authoritative resolution.
25. A successful selected cancellation allows its dependent creation when other gates pass.
26. Fresh independent creations do not require unrelated cancellation dependencies.
27. Current-wave reservations prevent oversubscription by a large batch.
28. Batch creates count logical order actions, not only HTTP requests.
29. Failed/ambiguous connector-bound creates count once; locally filtered plans count zero.
30. Event sink failure changes no gate decision.
31. Console/text projection remains bounded under persistent deferral.
32. Restart empties RAM state, emits reset visibility, and retains safety invariants.
33. Multiple bot processes remain independent and documentation does not claim otherwise.
34. Config reload, forager churn, mode changes, graceful stop, auto-reduce, WEL/TWEL, unstuck, and
    HSL transitions exercise their approved priority and tombstone semantics.

## Proposed Implementation Slices After Plan Approval

### Slice 1: Rust metadata and canonical contracts

- Add or confirm opaque stable order-family identity on every executable Rust ideal order.
- Add required `execution_priority=ordinary|risk_critical` metadata owned by Rust.
- Audit every current risk/close family and document its classification.
- Add the approved narrow restart exception to canonical decisions/contracts.
- Rebuild and verify the Python extension and add Rust/Python metadata contract tests.

### Slice 2: configuration and pure behavior state

- Add canonical fields, defaults, validation, templates, CLI aliases, and migration error.
- Retire `initial_entry_exec_max_market_dist_pct`.
- Implement pure cohort generation, tombstone, pending-generation, monotonic deque, and reservation
  helpers.
- Unit-test classification, rolling boundaries, reset, and deterministic multiset comparison.

### Slice 3: reconciliation and dependency-aware execution

- Observe normalized ideal generations before actual-order reconciliation annotations.
- Keep universal tolerance first and every stale cancellation eligible.
- Apply exemption, market-distance, and rolling-capacity admission to creations.
- Carry cohort/generation/dependency IDs through batch selection.
- Execute cancellation dependencies first and defer creations on truncated/failed/ambiguous results.
- Record replacement ledger timestamps only at the concrete create connector boundary.

### Slice 4: events and operator visibility

- Add stable registry values for generation, admission, deferral, dependency, and reset outcomes.
- Add bounded periodic summaries and smoke-report visibility.
- Prove diagnostic sink failure isolation.

### Slice 5: regression and fake-live validation

- Add multi-symbol fake-live static-grid, moving-EMA entry/close, fill restoration, and mixed-risk
  scenarios.
- Quantify connector-bound calls with and without the gate.
- Demonstrate that static orders rest, moving far orders stop churning, near orders remain admitted,
  risk-critical orders bypass, and no stale actual order is preserved.
- Demonstrate restart behavior and release pacing.

No authenticated exchange probe or live bot run is authorized by plan approval. Any later private
probe or live run requires explicit current-task authority.

## Implementation Validation Matrix

At minimum, the implementation PR must run:

- focused Rust order metadata and risk-priority tests;
- Rust suite, rebuilt/verified Python extension, and Python boundary tests;
- config schema/default/template/CLI/roundtrip tests;
- pure cohort/generation/tombstone/deque tests;
- reconciliation, orchestration, cancellation dependency, and executor tests;
- static/moving/mixed-risk multi-symbol fake-live scenarios;
- event-bus, registry, query, smoke, bounded projection, and sink-failure tests;
- rewritten/removed initial-entry-only distance-gate regressions;
- `PYTHONPATH=src python src/tools/check_ai_docs.py`;
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`;
- `git diff --check`.

Report exact base/head SHAs and a cycle-by-cycle evidence table showing Rust ideal generation,
classification, actual order, cancellation dependency/outcome, rolling count, distance decision,
and connector-bound action.

## Rollout And Measurement

After offline validation and separate authorization:

1. Run one controlled account with structured DEBUG execution evidence.
2. Measure generation transitions, admitted/deferred replacement creates, near-market bypasses,
   risk-critical bypasses, cancellations, fills, 429s, and ambiguous outcomes.
3. Prove no out-of-tolerance actual order is retained by the economy gate.
4. Inspect static grids, EMA Anchor entries/closes, forager churn, and release behavior.
5. Inspect Hyperliquid separately because requests-per-volume and cancellation allowance differ.
6. Inspect Defx conservatively because official numeric limits are not yet published.
7. Revisit defaults only from evidence; do not hide exchange-specific defaults in adapters.

## Reviewer Checklist

Reviewers are explicitly asked to assess:

- Is the observation point late enough to reflect actual executable order changes but early enough
  to remain independent of exchange-order pairing?
- Is the cohort key stable and sufficiently narrow for grids and future strategy kinds?
- Does opaque family equality avoid both allowlist debt and accidental cross-family coupling?
- Is deterministic multiset comparison authoritative enough, or is a Rust intent-group identity
  required?
- Are fresh, restoration, replacement-derived, removed, and tombstoned semantics correct?
- Can any deferred generation escape gating after its actual predecessor disappears?
- Can any superseded Rust ideal be created later?
- Are all true risk-critical paths explicitly marked without exempting ordinary take-profit churn?
- Does near-market always-allow behavior create unacceptable account-wide starvation or request
  consumption?
- Is counting connector-bound replacement creates the correct portable economy unit?
- Does activation-count terminology accurately describe bypass behavior?
- Can cancellation batch truncation, failure, or ambiguity ever leave old and new cohorts resting
  together?
- Is cohort-level cancellation dependency too conservative for large grids?
- Does one-for-one slot consumption eliminate release bursts without adding strategy priority?
- Is the RAM-only restart exception sufficiently narrow, observable, and safe?
- Are OKX, Paradex, Defx, generic CCXT, and all other live connector implications adequately covered?
- What behavior appears during fills, config reload, position changes, forager churn, graceful stop,
  HSL, WEL/TWEL, unstuck, auto-reduce, hedge mode, and one-way mode?
- Could exchange-native amend support later preserve queue position and reduce calls without
  weakening this portable contract?

Approval of this plan means the intent and high-level behavioral contracts are accepted. It does
not approve unreviewed implementation details, exchange-specific tuning, or authenticated testing.
