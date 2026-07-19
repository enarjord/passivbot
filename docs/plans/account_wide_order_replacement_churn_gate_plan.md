# Account-Wide Order Replacement Churn Gate Plan

## Status

Planning and review specification only. This document intentionally makes no runtime, schema, or
configuration changes.

The implementation must not begin until reviewers have challenged the intent, architecture,
exchange assumptions, edge cases, live/backtest implications, and operational consequences in
this plan.

## Problem

Some ideal limit prices move frequently because their inputs move frequently. EMA Anchor is the
current concrete example: both its entry and close quotes depend on EMA bands, volatility, and
inventory. A one-tick ideal-price change outside `live.order_match_tolerance_pct` can therefore
produce a cancel/create cycle even when the desired order remains far from market and unlikely to
fill.

Classifying known EMA-gated order types is not a durable solution. It requires the Python executor
to understand each strategy's price-generation internals, configuration switches, and future
order types. The remote-call problem is observable directly: repeated replacement of resting
orders. The proposed gate therefore reacts to measured replacement churn without inspecting the
strategy kind or Passivbot order-type family.

The objective is long-duration exchange-write economy. This is not intended to reproduce each
exchange's native rate limiter or replace CCXT throttling.

## Review Decisions Already Requested

The following requirements are deliberate inputs to the review:

1. The churn budget is account-wide within one live bot instance, not per symbol or position side.
2. The initial default is at most 10 replacements in a rolling 10-minute window.
3. Churn state exists only in RAM and resets when the bot process restarts. A few replacements
   after restart are acceptable.
4. The gate is strategy-agnostic and applies to known and future limit-order types.
5. `live.initial_entry_exec_max_market_dist_pct` is retired in favor of
   `live.order_replacement_churn_gate_market_dist_pct`.
6. The existing universal `live.order_match_tolerance_pct` remains the only accepted reason to
   leave a resting order at a price different from Rust's current ideal.
7. Once an actual order is outside the universal replacement tolerance, it must be cancelled.
   Exhausted replacement budget must never preserve that stale order.
8. When the account gate is active, a far replacement is cancelled and not recreated until the
   desired order is admitted by the distance gate or the rolling churn gate releases.

Reviewers should focus on whether the proposed mechanics satisfy these requirements safely, not
replace them with strategy-specific EMA classification or persistent state.

## Goals

- Stop hours of low-value cancel/create traffic caused by moving ideal limit prices.
- Protect the write budget shared by all symbols in one exchange account/bot instance.
- Work without advance knowledge of strategy kind, EMA use, EMA speed, or order-type naming.
- Preserve Rust as the source of ideal orders and Python as the live reconciliation/execution
  gatekeeper.
- Never keep an out-of-tolerance actual order merely because replacement creation is deferred.
- Permit near-market orders despite churn pressure so the gate has a bounded effect on likely
  fills.
- Make activation, deferral, release, and connector-bound replacement attempts visible in the
  centralized live-event stream.
- Keep the first implementation small enough to review and validate independently.

## Non-Goals

- Reimplement any exchange's complete native rate limiter.
- Replace CCXT's built-in request pacing, endpoint weights, retry policy, or exchange error handling.
- Coordinate budgets across multiple Passivbot processes that share an API key or IP address.
- Persist or reconstruct churn state across restarts.
- Predict whether an order was derived from EMA, volatility, trailing logic, or another input.
- Change Rust order prices, quantities, strategy behavior, or backtest calculations.
- Guarantee that a deferred far order could not have filled during a market gap.
- Treat the initial 10-per-10-minute values as exchange-native limits.

## Existing Contracts To Preserve

### Rust ideal and universal tolerance

Rust remains the source of the current ideal order. Reconciliation may treat an actual order as
satisfied only when it is an exact match or is within `live.order_match_tolerance_pct`. The current
default is `0.0002`, or 0.02% as a fractional relative tolerance.

The churn gate runs only after this equivalence check. An in-tolerance order is preserved and is
not replacement churn. An out-of-tolerance order is no longer the current live ideal and must not
remain open due to this gate.

### Cancellation and creation are asymmetric

When the account gate is active:

- cancellation of an out-of-tolerance order still proceeds;
- creation of the desired replacement may be deferred;
- an unpaired retirement cancellation still proceeds and is not classified as replacement churn;
- a market order is never held by a limit-order distance gate;
- the existing dedicated protective-panic reconciliation path remains outside this economy gate.

This asymmetry is intentional. It favors removing an invalid resting order over keeping a stale
order that Rust no longer wants.

### Fresh market data remains mandatory

The gate uses the existing pre-create market snapshot and `order_market_diff` semantics. It must not
invent a neutral market price or bypass current freshness/readiness checks. If the required market
snapshot is unavailable, the existing broader create deferral owns the decision.

## Proposed Configuration

Add these canonical `config.live` fields:

| Field | Default | Meaning |
|---|---:|---|
| `order_replacement_churn_gate_max_replacements` | `10` | Account-wide replacement count that exhausts the rolling budget. `0` disables the churn gate. |
| `order_replacement_churn_gate_window_minutes` | `10.0` | Rolling RAM window for connector-bound replacement attempts. |
| `order_replacement_churn_gate_market_dist_pct` | `0.005` | Fractional signed market-distance threshold used only while the account gate is active. `0.005` is 0.5%. |

Validation requirements:

- maximum replacements is an integer greater than or equal to zero;
- the window is finite and greater than zero when the gate is enabled;
- market distance is finite, greater than or equal to zero, and less than one;
- defaults are owned by the canonical schema/preparation path, not repeated in runtime consumers.

### Retirement of the initial-entry setting

`live.initial_entry_exec_max_market_dist_pct` should be removed from the canonical schema, template,
CLI aliases, examples, runtime, and current user documentation. It must not remain as a silent
alias because the old and new settings have materially different scope.

Recommended compatibility behavior is an actionable configuration error that names the three new
settings and explains that the gate is now adaptive and account-wide. If a migration tool handles
the old field, migration must be explicit and must add the new count/window defaults rather than
silently pretending the old threshold alone preserves behavior. Reviewers should decide whether a
released-version compatibility window is required.

Historical release notes and historical live events remain historical; they need not be rewritten.

## Definitions

### Replacement pair

A logical replacement is a cancel/create pair that remains after exact reconciliation and universal
order-match tolerance. The current reconciler annotates such pairs with `context=replace` by
matching symbol, order side, position side, and closest price.

This plan does not make `pb_order_type` part of replacement eligibility. Review must assess whether
closest-price pairing is sufficiently unambiguous for multi-order ladders and same-side grid orders.

### Counted replacement attempt

Count one logical replacement when its cancellation reaches the concrete connector-call boundary.
Do not wait for a successful response: failed, rejected, timed-out, or ambiguous calls may still
consume exchange resources. Do not count a plan that is filtered locally and never reaches that
boundary.

A batch containing `n` replacement cancellations counts as `n` logical replacements for this
policy. This is conservative across exchanges and matches Hyperliquid's address-based treatment of
batched actions even where one HTTP request has a lower IP weight.

Counting the cancellation rather than both writes makes the configured unit understandable as one
replacement cycle. Metrics should separately expose connector calls so operators can estimate the
actual create-plus-cancel write cost.

### Account scope

One bot instance has one deque of replacement timestamps. There is no symbol, position-side, or
order-side budget partition. A noisy coin can activate distance gating for another coin because the
protected resource is shared account/API capacity.

This scope does not protect an account or IP shared by multiple Passivbot processes. That limitation
must be visible in documentation and should not be obscured by calling the mechanism a complete
exchange rate limiter.

## Proposed State And Decision Flow

Maintain in RAM:

```text
replacement_attempt_timestamps: deque[monotonic_time]
gate_active: derived from the rolling deque plus current-plan reservations
current_plan_reservations: temporary count, never persisted
```

The deque resets empty at bot startup.

For each normal reconciliation cycle:

1. Obtain Rust ideal orders and authoritative actual open orders through existing paths.
2. Reconcile exact matches.
3. Apply `live.order_match_tolerance_pct`.
4. Annotate the remaining cancel/create replacement pairs.
5. Prune replacement timestamps older than the configured rolling window.
6. Determine the account's used replacement budget.
7. Include projected replacement cancellations in the current plan so one large batch cannot
   bypass a nearly exhausted budget.
8. Keep every required out-of-tolerance cancellation in the cancel plan.
9. If the account budget is not exhausted, allow the desired creation through existing gates.
10. If the account budget is exhausted:
    - allow market orders;
    - preserve dedicated protective-panic bypass behavior;
    - compute each limit creation's signed distance from fresh market price;
    - allow the creation when distance is less than or equal to
      `order_replacement_churn_gate_market_dist_pct`;
    - otherwise defer the creation, including a creation that appears as `context=new` in a later
      cycle after its previous order was cancelled.
11. Commit a timestamp only when the replacement cancellation reaches the connector-call boundary;
    discard unused current-plan reservations.
12. Release account-wide distance gating naturally as timestamps leave the rolling window and the
    used count falls below the configured maximum.

Conceptually:

```text
actual order outside universal tolerance?
    no  -> keep it; no churn event
    yes -> cancel it
           |
           +-- account replacement budget available -> desired create follows normal gates
           |
           +-- budget exhausted
                 |
                 +-- market/protective bypass -> allow
                 +-- desired limit within 0.5% -> allow
                 +-- desired limit farther away -> defer; leave no stale order resting
```

## Same-Cycle Budget Reservations

Historical events alone are insufficient when one plan contains many replacement pairs. For
example, with nine recent replacements and twenty replacements in the current plan, all twenty
must not be treated as if the account still has one free slot.

The planner should use deterministic temporary reservations while evaluating the batch. Stale
orders are still all cancelled, but only creations admitted before exhaustion or allowed by the
near-market bypass remain eligible. Reservations become real deque entries only for cancellations
that reach the connector-call boundary.

Reviewers should inspect ordering carefully. Candidate ordering may affect which far creations are
admitted immediately. The least surprising first implementation is to reuse existing order sorting
and batch-priority behavior rather than create a new strategy priority in Python. Near-market and
dedicated protective actions bypass exhaustion independently.

## Release Burst And Fairness Caveat

An account-wide binary gate can stage far orders across many symbols. When an old timestamp leaves
the rolling window, many creations may become eligible in the same cycle even though only one
replacement slot technically became free. Existing `max_n_creations_per_batch` bounds the immediate
connector batch, but may not eliminate a multi-cycle release burst.

This is an explicit review question before implementation:

- Is normal batch limiting sufficient because these are one-time re-creations rather than
  continuous replacement churn?
- Or should previously deferred far creations be re-admitted through a paced account-wide queue?

The first implementation should not add a complex fairness scheduler without review. If paced
re-admission is required, its ordering must be explicit and must not give Python authority to
reinterpret Rust strategy priority.

## Centralized Live-Event Integration

Existing execution events already carry symbol, position side, order side, `context=replace`,
price/quantity deltas, connector-call evidence, and terminal outcomes. They are suitable for
auditing and measuring the gate.

The event pipeline is observability-only by contract. The churn gate must therefore update its RAM
deque at the same canonical executor hook that emits connector-call events; it must not subscribe
to a diagnostic sink or make event publication success a trading dependency.

Proposed event changes:

- Add replacement `context` to connector-call-started payloads if it is not already retained.
- Emit a bounded gate transition event when the account changes between available and exhausted.
- Emit `execution.create_deferred` with reason code
  `account_order_replacement_churn_gate` for a withheld creation.
- Include safe bounded fields: scope=`account`, used count, maximum count, window seconds, desired
  distance, distance threshold, symbol, position side, order side, and cycle/wave correlation.
- Aggregate repeated deferrals into periodic summaries; do not produce one operator-visible line
  per order per cycle.
- Preserve structured detail even when console/text projection is suppressed.
- Ensure sink failure cannot change deque updates, cancellation, creation, or release behavior.

## Exchange Rate-Limit Research And Ramifications

This proposal is deliberately a conservative policy layer above heterogeneous native systems.
Official documentation was reviewed on 2026-07-19; implementation reviewers must recheck it because
limits and account tiers change.

| Exchange/model | Officially documented behavior relevant to this plan | Ramification |
|---|---|---|
| Hyperliquid | REST has a weighted IP pool of 1200/minute. Address-based action allowance is tied to cumulative USDC volume, starts with a 10,000-request buffer, permits one request per 10 seconds when exhausted, and gives cancels the cumulative allowance `min(limit + 100000, limit * 2)`. A batch of `n` orders/cancels is one request for IP weight but `n` requests for the address limit. Congestion may additionally allocate block space from maker share. [Official docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits) | A fixed rolling gate does not model the address's actual remaining allowance. Nevertheless, reducing cancel/recreate traffic improves requests-per-fill and is especially valuable for low-volume accounts after the initial buffer. Count batch elements, not only HTTP requests. Continue allowing stale-order cancellation because Hyperliquid intentionally provides extra cancel capacity. Avoid duplicate cancel resends already confirmed by the API; this gate does not replace confirmation logic. |
| Bybit | Private API limits use a rolling per-second, per-UID window with endpoint-specific headers. Standard create and cancel endpoints are commonly 10/s, while an IP limit also applies. Batch consumption is based on the number of orders, not merely one request. [Official docs](https://bybit-exchange.github.io/docs/v5/rate-limit) | Ten replacements per ten minutes is not a burst limiter for Bybit. CCXT/existing batching still owns short-window compliance. Per-order logical counting is appropriate for batch operations. |
| Bitget | Futures batch cancellation is documented at 10 requests/s for average users and scoped by user ID; endpoint limits differ and batch cancellation accepts multiple order IDs. [Official docs](https://www.bitget.com/api-doc/classic/contract/trade/Batch-Cancel-Orders) | The account-wide churn gate reduces sustained low-value traffic but must not be treated as Bitget endpoint pacing. Batch composition and partial outcomes remain connector concerns. |
| KuCoin | Private quotas are UID resource pools, with futures capacity expressed over a 30-second pool and endpoint weights; order placement currently has futures-pool weight 2. Response headers expose remaining quota/reset, and overload limiting can occur separately. [Rate-limit docs](https://www.kucoin.com/docs-new/rate-limit), [futures add-order weight](https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order) | The proposed gate is much slower than native windows and addresses waste rather than peak compliance. Future adaptive use of response headers is possible but is outside the first implementation. |
| Gate.io | Futures placement/amendment and cancellation have separate UID limits, and Gate also documents behavior-based restrictions for frequent low-fill trading in parts of its API. [Official docs](https://www.gate.com/docs/developers/apiv4/en/) | High nominal cancellation capacity does not make churn harmless: low fill efficiency and behavior-based controls matter. Reviewers should verify which behavior-based restrictions apply to the exact futures endpoints used by Passivbot. |
| WEEX | Official limits are account/UID or IP scoped by endpoint; current contract/AI API documentation states place, cancel, trigger-place, and trigger-cancel interfaces are capped at 10 requests/s. [Official docs](https://www.weex.com/api-doc/ai/QuickStart/LIMITS) | The account-wide gate addresses the observed continuous replacement pattern, while existing CCXT pacing remains responsible for the 10/s burst ceiling. |
| Binance | Binance documents multiple interval-based `REQUEST_WEIGHT`/`ORDERS` limiters, IP-weight headers, account order counts, and fill-sensitive unfilled-order behavior. Product and endpoint values vary. [Official docs](https://developers.binance.com/en/docs/products/spot/rest-api) | Do not infer one universal Binance futures number from this plan. The implementation review must check the current USD-M endpoint contract and preserve header/backoff handling. The churn gate remains supplementary. |
| CCXT | CCXT enables a per-exchange-instance limiter by default and documents leaky-bucket and optional rolling-window algorithms. It warns that separate exchange instances have independent limiters. [Official manual](https://github.com/ccxt/ccxt/wiki/manual#rate-limit) | Keep CCXT rate limiting enabled. The churn gate reduces unnecessary desired writes; CCXT controls when admitted calls are sent. Multiple bot processes or exchange instances remain outside the RAM gate's account-wide claim. |

### Hyperliquid-specific questions for review

1. Is the universal 10-per-10-minute default conservative enough for low-volume addresses whose
   initial request buffer is depleted?
2. Should a future optional Hyperliquid adapter read `userRateLimit` and expose diagnostics without
   changing the first generic gate?
3. Does Passivbot's Hyperliquid batching mean the executor hook sees each address-counted action
   clearly enough to record `n` logical replacements?
4. Are any confirmed cancel results currently resent during congestion, independently of ideal
   price churn?
5. Could account-wide staging improve maker fill ratio but harm maker queue position enough to
   require different defaults for market-making strategies?

## Live/Backtest Parity Analysis

The universal order-match tolerance is the only accepted case where live intentionally leaves an
actual order at a slightly different price from Rust's current ideal.

The new gate creates a different explicit live execution exception: after cancelling an invalid
order, live may temporarily have no corresponding far order on the exchange. This can miss a gap or
fast wick through the Rust ideal price. The 0.5% admission threshold bounds normal approach risk but
does not eliminate gap risk.

This exception is preferable to retaining an out-of-tolerance stale order because the stale order
can fill at a price Rust no longer requests. Reviewers should nevertheless assess:

- whether 0.5% is appropriate for both entries and closes;
- whether signed distance works correctly for long/short entries and closes;
- whether reduce-only ordinary closes should follow the same distance rule;
- whether any protective limit-order path lacks the dedicated panic bypass;
- whether market movement can jump from outside the threshold through the ideal price between
  snapshots;
- whether current market-snapshot cadence is adequate for staged near-market admission;
- whether order queue-position loss materially changes expected fills for any supported strategy.

No Python fallback price or order may be created to compensate for a deferred Rust ideal.

## Failure And Safety Semantics

- Invalid churn-gate configuration fails startup validation.
- Missing required market data follows existing create deferral; it does not default distance to
  zero or infinity.
- Event publication failure is diagnostic only and must not affect the gate.
- Ambiguous connector calls count as attempted replacement churn once the call boundary was
  reached, while existing authoritative confirmation policy still determines order state.
- A cancellation outside tolerance is never withheld by replacement-budget exhaustion.
- Existing duplicate-cancel and recent-execution protections remain active.
- Exchange 429/backoff handling remains authoritative and may defer more than this gate.
- Low balance, stale account state, mode changes, HSL, risk, and planning readiness retain their
  current precedence. This plan must not broaden action permission.

## Edge Cases Requiring Explicit Tests

1. Exact and within-tolerance orders remain open and do not consume churn budget.
2. Out-of-tolerance orders are cancelled even while the account gate is active.
3. A far desired replacement is not created while gated.
4. A near desired replacement is created while gated.
5. A far create remains gated on the cycle after its old order disappeared and reconciliation now
   labels it `new`.
6. Ten replacements across ten different symbols exhaust one shared account budget.
7. Entry churn on one symbol gates a distant close or entry on another symbol.
8. Current-plan reservations prevent a large replacement batch from overshooting before the deque
   is updated.
9. Failed and ambiguous connector-bound replacement attempts count once; locally filtered plans
   count zero.
10. Batch cancellations count logical order actions consistently with the chosen policy.
11. Unpaired retirements are cancelled but do not count as replacements.
12. Market orders and dedicated protective-panic reconciliation bypass distance gating.
13. Signed market distance is correct for buy and sell orders on long and short position sides.
14. A missing/stale market snapshot uses existing fail-closed creation behavior.
15. Rolling-window expiry releases the gate without wall-clock-jump errors; use monotonic time.
16. Restart begins with an empty RAM budget and allows initial replacements as specified.
17. Multiple bot instances remain independent and documentation does not claim otherwise.
18. Event-sink failure leaves decisions unchanged.
19. Operator projection is bounded under dozens of continuously deferred symbols.
20. Gate release with many staged far orders respects creation batch limits and does not create an
   uncontrolled burst.
21. Forager deselection, graceful stop, mode changes, and symbol retirement still cancel obsolete
   orders immediately.
22. Partial fills that change desired price/quantity interact correctly with replacement pairing.
23. Multiple same-side grid orders pair deterministically without misclassifying a retirement as a
   replacement.
24. Quantity-only replacements consume the same budget as price replacements unless review decides
   otherwise.
25. Exchange acknowledgement followed by websocket/snapshot lag does not double-count one logical
   replacement.

## Proposed Implementation Slices After Plan Approval

### Slice 1: configuration and pure state helper

- Add canonical fields/defaults/validation.
- Retire the initial-entry-only field and CLI aliases with the approved migration behavior.
- Add a small pure rolling-window helper with monotonic timestamps and current-plan reservations.
- Unit-test boundary timing, disable semantics, and reset behavior.

### Slice 2: reconciliation and execution integration

- Preserve exact/tolerance reconciliation first.
- Identify replacement pairs without strategy-specific order lists.
- Cancel every out-of-tolerance actual order.
- Apply account-wide distance filtering to creates only after budget exhaustion.
- Update the deque at the connector-call boundary.
- Preserve dedicated protective and market-order bypasses.

### Slice 3: events and operator visibility

- Add stable reason/event registry values.
- Expose account gate transitions, attempts, counts, and deferred creates.
- Add bounded periodic summaries and smoke-report visibility.
- Prove event failure isolation.

### Slice 4: regression and fake-live validation

- Extend reconciliation/executor tests across multiple symbols and both order sides.
- Run deterministic fake-live scenarios with fast-moving ideal orders for entries and closes.
- Demonstrate that stale orders are cancelled, far creates are withheld after exhaustion, near
  creates are admitted, and long-duration connector calls fall materially.
- Compare event-derived replacement counts with actual fake connector calls.
- Run broader live orchestration, event registry, config, and docs suites.

No authenticated exchange probe or live bot run is part of plan approval. Any later authenticated
validation requires separate current-task authorization.

## Implementation Validation Matrix

At minimum, the implementation PR should run:

- focused config schema/default/CLI tests;
- focused reconciliation, order orchestration, and executor tests;
- event-bus, event-registry, query, smoke-report, and sink-failure-isolation tests;
- multi-symbol fake-live churn scenarios;
- existing initial-entry distance-gate tests rewritten or removed according to the retired field;
- `PYTHONPATH=src python src/tools/check_ai_docs.py`;
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`;
- `git diff --check`.

The implementation PR must report exact base/head SHAs and quantify connector-bound cancel/create
calls with and without the gate in the deterministic scenario.

## Rollout And Measurement

After offline validation and separate authorization:

1. Run one controlled live account with structured DEBUG execution events enabled.
2. Measure replacement attempts, cancellations, creations, deferred creates, near-market bypasses,
   fills, 429s, and ambiguous outcomes before and after activation.
3. Confirm no out-of-tolerance order is retained due to the gate.
4. Inspect gate activation across many symbols and release behavior after ten minutes.
5. Inspect Hyperliquid separately because requests-per-volume, cancel allowance, batching, and
   congestion behavior differ from ordinary endpoint windows.
6. Revisit defaults only from evidence; do not silently add exchange-specific defaults in adapter
   code.

## Reviewer Checklist

Reviewers are explicitly asked to look beyond code style and assess:

- Is account-wide scope correct for every supported connector and bot topology?
- Is counting connector-bound replacement cancellations the right portable churn unit?
- Should failed/ambiguous attempts count, and can any outcome be double-counted?
- Does batch handling correspond to actual exchange resource consumption?
- Can any out-of-tolerance order remain open because creation is gated?
- Can a far order be recreated accidentally on the next cycle after cancellation?
- Can a noisy symbol starve important orders elsewhere, and is the 0.5% bypass adequate?
- Are all truly protective paths outside the gate without brittle order-type allowlists?
- Is release susceptible to a thundering herd, starvation, or unfair ordering?
- Does the proposal preserve current authoritative-state and ambiguous-write safety?
- Is the explicit live/backtest exception bounded and acceptable for entries and closes?
- Does retiring the old config field need a compatibility or migration window?
- Are central events complete enough to reconstruct every gate decision without becoming causal?
- How should Hyperliquid's cumulative volume-linked address allowance, enlarged cancel allowance,
  batch accounting, and congestion maker-share policy affect design or defaults?
- Do Binance, Bybit, Bitget, Gate.io, KuCoin, WEEX, Hyperliquid, or other supported exchanges have
  endpoint, subaccount, IP, fill-ratio, or daily-order policies missing from the analysis?
- Could exchange-native amend endpoints reduce churn more safely than cancel/create on any
  connector, and should that remain a separate future project?
- What unintended behavior appears with partial fills, grids, quantity-only changes, one-way mode,
  hedge mode, graceful stop, HSL, unstuck, auto-reduce, or forager churn?

Approval of this plan should mean the intent and high-level contract are accepted. It should not be
interpreted as approval of unreviewed implementation details or authenticated live testing.
