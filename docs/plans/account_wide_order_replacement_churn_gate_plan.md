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
3. Cancel every unmatched stale actual order in bot-managed semantic scope regardless of
   churn-gate state. Orders in intentionally unmanaged scope do not arm the stale-cancellation
   barrier; orders whose scope cannot be proven make the account-critical open-orders surface
   unavailable and arm an account-wide no-write/no-cancel readiness barrier.
4. For every cohort, associate the complete current and historical groups one-to-one within each
   prior valid snapshot before classifying individual unmatched current ideals:
   - a tight match is evidence of stability in that snapshot;
   - a wider-but-not-tight match is direct price/quantity churn evidence;
   - no wider match is uncertainty, not proof of replacement identity, and therefore fails open.
5. Exempt market orders, explicitly risk-critical orders, and orders within the configured final
   market-distance threshold from allowance waiting. Only dedicated protective market panic may
   bypass the account-wide stale-cancellation barrier.
6. Admit far churn-evidenced ordinary creates when their create-plus-required-configuration
   reservation leaves at most ten connector-bound action attempts in the account-wide rolling
   ten-minute window. The eleventh action defers; cancellations do not count. Exempt creates always
   proceed but count for subsequent ordinary admission.
7. Never retain a stale actual order while waiting. Cancel it and regenerate current Rust intent on
   a later cycle.
8. Execute all non-panic creations behind an account-wide cancel-first barrier. If any stale actual
   exists, cancel, confirm authoritative balance, positions, open orders, and fills, regenerate the
   complete Rust plan, and only then create in a later clean wave.

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
- weaken readiness, mode scope, semantic attribution, risk, HSL, WEL/TWEL, or
  authoritative-state barriers.

### Universal reconciliation tolerance stays universal

`live.order_match_tolerance_pct` remains the only accepted tolerance for deciding that an actual
order satisfies a current ideal order. The wider tracking tolerance is historical evidence only.
It must never widen exchange-order reconciliation.

### Cancellation is not gated

Every unmatched actual order in a bot-managed semantic scope remains eligible for cancellation.
If creation is deferred, that stale actual order is still removed. No replacement allowance or
distance decision may leave a stale managed order resting.

Management is mode- and intent-scoped, not ownership-scoped:

- `normal` and other fully bot-managed modes cancel every unmatched actual order in scope,
  including user-created orders, and recreate every missing current Rust ideal after the ordinary
  sequencing barriers are clear;
- `panic` is fully bot-managed: unmatched entries and obsolete closes are cancelled while only the
  current Rust panic intent is created, with the separately reviewed dedicated market-panic bypass;
- `graceful_stop` is fully bot-managed: unmatched orders are cancelled and only the current Rust
  graceful-stop ideal is recreated, so a flat side cannot revive an obsolete entry;
- `manual` leaves both entries and closes unmanaged: no creation and no cancellation;
- `tp_only` leaves entries unmanaged but manages closes normally;
- `tp_only_with_active_entry_cancellation` creates no entries, actively removes stale entry orders,
  and manages closes normally, preserving its existing distinction from `tp_only`.

Throughout this plan, **dedicated protective market panic** means a current Rust-emitted market
order whose payload is authoritatively close-only/reduce-only for the affected position. A panic
name or priority marker without that close-only proof never receives the bypass.

Accordingly, a user cancellation in a bot-managed scope causes the still-current Rust ideal to be
recreated, and a user-created unmatched order in that scope is cancelled. Client-order ownership,
client-ID presence, and Passivbot-looking markers do not change that decision. Unknown coin,
position side, or entry-versus-close semantics fail closed because the management scope cannot be
proven; missing `pb_order_type` or client ownership metadata alone is not a no-cancel condition once
the management scope is authoritative. The dedicated protective market-panic path retains its
existing separately reviewed authority.

This mode-scoped management rule does not retire the existing competing-writer safety detector.
Open-order snapshots run that detector before reconciliation; repeated evidence that another
Passivbot process is writing to the account still stops this bot. Passivbot-looking ownership
metadata is not a keep/cancel criterion only after that independent safety check has allowed the
cycle to continue.

### Missing inputs are not fabricated

Invalid, partial, or unavailable planning results append no history. A deliberate valid Rust result
with zero orders is a valid empty snapshot and cancels stale actual orders in bot-managed scopes.

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
pb_order_type
```

Price and quantity must additionally match within `live.order_match_tolerance_pct` for actual
reconciliation or within the explicitly selected historical tolerance for churn evidence. Actual
order quantity means authoritative remaining open quantity, never the original submitted amount
after a partial fill. Prefer an exchange-provided `remaining`; derive `amount - filled` only when
both fields are authoritative, finite, non-negative, and internally consistent. Every actual order
also requires an authoritative exchange order ID, symbol, order side, position side, positive finite
price, and close-only effect. A missing, non-finite, contradictory, or otherwise malformed required
field makes the account-critical open-orders surface unavailable and blocks every exchange write
until a fresh authoritative snapshot resolves it. Unknown `pb_order_type` or resting execution type
alone is not malformed because the order can still be scoped and cancelled, but it cannot satisfy a
known current ideal. Do not cancel an uncertain order, create elsewhere, or let a panic/risk
exemption bypass this readiness failure.

Matching is deterministic and one-to-one. One actual or historical observation cannot satisfy two
current ideal orders.

### One-way `position_side` attribution

The exact cohort retains `position_side` even when the venue has one net position ledger. For any
one-way actual order, normalize it deterministically in this order:

1. prefer durable client-order/local execution metadata whose exact `pb_order_type` or explicit
   field identifies long versus short;
2. otherwise use the reviewed one-way intent tuple:
   - non-close-only buy = long entry;
   - non-close-only sell = short entry;
   - close-only sell = long close;
   - close-only buy = short close.

Do not infer from the current held position: it can change after order placement and would re-label
the same resting order across fills or restart. Unknown close-only effect, side, or ambiguous
position-side attribution prevents proving the applicable management scope and therefore makes the
account-critical open-orders surface unavailable, blocking every exchange write account-wide
until a clean authoritative refresh resolves it. Missing client ownership metadata does not.
Hyperliquid's current position-dependent helper must be replaced or bypassed for this canonical
snapshot path before enabling the gate. Hyperliquid fill normalization already produces
`position_side` from authoritative direction/close semantics; that does not make the current
open-order helper safe. Open-order reconciliation must use durable client/local intent metadata or
the same reviewed side-plus-close-only tuple, never the currently held position.

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

### Why authoritative exchange-native close-only effect is required

The exact cohort field remains named `reduce_only` in the current Rust/Python payload contract, but
its canonical meaning here is **authoritative exchange-native close-only effect**, not necessarily a
literal REST `reduceOnly=true` response. It is an execution guarantee, not a diagnostic label.
Multiple individually plausible close orders can over-close after another order fills or after a
manual/external position change; the connector must prove the venue-native action cannot increase
or flip the scoped position.

For venues whose contract uses a literal reduce-only flag, actual normalization must use that
authoritative flag. For venues whose hedge action is authoritatively encoded by an action tuple,
the adapter may normalize that tuple to the same canonical close-only boolean only under a
documented connector contract and focused fixtures. Unknown does not match either `true` or
`false`. Because that ambiguity prevents proving both mode scope and the account-wide cancellation
dependency, it blocks every exchange write account-wide until authoritative refresh resolves it
under the error contract. Generic side/position-side inference remains forbidden.

WEEX V3 omits a placement `reduceOnly` field but documents it in open-order and order-info responses
while identifying hedge actions by `side` plus `positionSide`. The WEEX adapter must prefer and
verify the authoritative response field. If live payloads omit or misreport it, its V3 action tuple
may be normalized only after offline contract tests and separately authorized live evidence prove
the combination cannot increase or flip the scoped position.

Bitget UTA/Elite hedge mode is the second explicit connector-native case. Its contract requires
`side` plus `posSide`, rejects `reduceOnly` with `posSide`, and may report a close with
`reduceOnly=NO`. The existing canonical exchange contract therefore overrides that literal flag:
the adapter must normalize the documented UTA action tuple to close-only effect. Classic Bitget
v2/mix and one-way modes retain their separate authoritative `tradeSide`/`reduceOnly` handling.

OKX long/short mode is the third explicit connector-native case. CCXT emulates reduce-only by
selecting `posSide` and may expose a top-level `reduceOnly=false` for a valid close. The adapter must
therefore normalize the documented `side` plus `posSide` action tuple in effective long/short mode,
while effective one-way/net mode requires the authoritative raw `reduceOnly` response. A parser
default may not override an omitted raw exchange field.

This is a prerequisite correction to the current reconciler, whose tight matching tuple omits
`reduce_only` and whose generic open-order snapshot derives it from side and position side. The
churn gate must not be layered on top of that generic inference.

### Why execution type stays exact but time-in-force does not

Market versus limit changes execution guarantees and remains exact even when price and quantity
match. Time-in-force and post-only do not participate in actual-order satisfaction or historical
churn cohorts.

Post-only is a placement-time constraint: once an order is accepted and resting, post-only and GTC
orders at the same price and quantity have equivalent ongoing execution behavior. IOC and FOK
orders do not remain open. Connectors must still apply the configured time-in-force to every new
creation and preserve returned semantics as diagnostic metadata. If Passivbot later supports a
persistent lifetime behavior such as GTD expiry, add a narrower authoritative
`resting_lifetime_semantics` key rather than persisting placement-only metadata.

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

This union prevents downstream symbol-scoped normalization or availability handling from making
actual orders or positions disappear from consideration.

Rust ideal planning is currently one account-wide call because TWEL, entry gates, risk ordering,
and portfolio allocation cross symbol boundaries. A Rust planning failure returns no safe partial
plan: append no snapshots and perform no ordinary reconciliation for any symbol. Do not retry the
same decision per symbol or salvage partial output in Python. Symbol-scoped validity begins only
after a complete Rust result exists; a downstream normalization or required-input failure then
blocks only that symbol while healthy symbols continue **only when authoritative state proves the
blocked symbol has neither actual open orders nor a held position**. If it has actual orders or an
authoritatively nonzero position, suppress all account-wide non-panic creates for the wave because
a fill or unknown stale order there can invalidate the shared Rust plan; a dedicated protective
market panic remains the sole bypass. If either position side is absent, malformed, non-finite, or
otherwise unproven, block every exchange write because even panic side and quantity are unsafe. A
valid empty symbol plan appends an empty snapshot and removes stale orders for that symbol. The
account-critical open-orders exception is
stricter: a missing, non-finite, contradictory, or otherwise malformed required exchange order ID,
symbol, order side, position side, positive price, positive authoritative remaining quantity, or
close-only effect makes that surface unavailable and blocks every exchange write until a clean
refresh. A future partial-result Rust API must preserve account-wide decisions explicitly before
this boundary may be narrowed.

Append the current snapshot only after its decision, or otherwise exclude it from its own history.
Prune by the configured window. Consecutive identical snapshots may be compacted only if tests prove
the same logical presence answers at every relevant timestamp.

### Planning-policy compatibility

History compatibility has both account-wide and scoped epochs because the Rust plan has
account-wide dependencies. The account epoch covers hysteresis-snapped wallet balance,
per-symbol/position-side signed position size and average entry price,
authoritative fill identity, the exact `realized_pnl_cumsum_max` and
`realized_pnl_cumsum_last` values passed to Rust, Rust-reported active risk pairs, global
strategy/live configuration, and effective hedge/one-way mode. An active risk pair is any
symbol/position-side for which Rust emits a risk-critical order or a realized-loss-gate block. It
excludes equity, available margin,
unrealized PnL, raw quote-valued balance, mark/liquidation price, mark-to-market notional, and other
price-derived exchange fields. On multi-collateral venues, raw balance may change solely because a
collateral asset's quote value changes; including it would continuously erase churn evidence.
Rust continues to consume exact raw balance for risk decisions. Entry into or exit from a
Rust-reported risk phase resets the account epoch immediately, and every ideal order sharing an
active risk symbol/position-side bypasses economy-only churn deferral. This covers raw-balance
changes that alter realized-loss, TWEL, or unstuck behavior without treating valuation-only raw
balance ticks as policy transitions. Authoritative fills, realized-PnL cumulative values, and
position transitions also reset the account epoch immediately, while a material wallet change
advances the hysteresis-snapped balance.
A change clears all ideal history before classifying the next complete Rust plan. A scoped
epoch covers each symbol's effective `PB_modes`, approved/ignored membership, forager membership,
and active-universe membership. A scoped change clears only that symbol's history. This prevents a
rotating empty forager slot from continuously erasing evidence for unrelated resting orders. It is
safe despite cross-symbol allocation effects because Rust's current ideal remains authoritative:
the reconciliation path still cancels stale actuals immediately, and any resulting price/quantity
change is observed directly by the history classifier. Global config changes remain account-wide,
including operator changes to configured modes or lists. Neither reset clears account-wide attempt
timestamps: exchange writes already consumed remain consumed for the rolling window.

Because snapshots are captured after exchange precision and sizing normalization, each symbol's
scoped compatibility epoch also includes its effective price step, quantity step, minimum quantity,
minimum cost, contract multiplier, active flag, and any other order-validity constraint used by the
normalizer. `init_markets()` may refresh these values while the process remains alive. A change
clears that symbol's history before classifying its next current ideal; it does not require an
account-wide reset unless the changed metadata also alters an account-wide Rust input.

Continuously changing prices, EMAs, volatility, trailing extrema, raw quote-valued balance, and
other price-derived account fields are excluded; including them would continuously reset the
evidence gate and defeat its purpose. A new authoritative fill identity, realized-PnL
cumulative-value change, Rust-reported risk-phase transition, hysteresis-snapped wallet-balance
change, or signed position-size/average-entry-price transition advances the account epoch because
TWEL, entry gates, realized-loss/unstuck policy, risk ordering, and portfolio sizing may change
ideals on other symbols. This also preserves
the existing contract that every fill resets trailing extrema and prevents pre-fill evidence from
suppressing the first entry, close, HSL, or re-entry order of the new account phase. Ordinary market
price movement does not advance the epoch. If later Rust metadata exposes a proven dependency graph,
cross-symbol invalidation may be narrowed under separate review; Python must not infer it. Current
coin overrides are normally startup-parsed, so restart already resets RAM history; explicit epochs
keep the contract correct for list refreshes, mode transitions, and any future hot config reload.

### Why history is per symbol but allowance is account-wide

Per-symbol history prevents one unhealthy or omitted symbol from poisoning all classification and
keeps matching bounded. The allowance remains account-wide because the scarce resource—private
order-write capacity—is normally shared across symbols for the account, UID, subaccount, or IP.

This bot-process scope is intentionally not a complete multi-process account limiter. Cross-process
coordination is a separate feature.

## Quantitative Churn-Evidence Heuristic

For each exact cohort, first associate the complete current-ideal group with the complete historical
group. This includes current ideals already satisfied by an actual order; they reserve their
historical mates even though only unmatched ideals are later eligible for creation. Market,
risk-critical, and near-market ideals likewise participate in this one-to-one assignment even though
they are exempt later during admission. Omitting any of these peers could let its historical mate be
reused to misclassify an ordinary unmatched create. After group matching, classify each unmatched
non-exempt current ideal as follows:

1. For each prior valid snapshot in the active per-symbol window, newest first, perform a
   deterministic one-to-one association within that snapshot.
2. Prefer tight matches using `order_match_tolerance_pct` for both normalized price and quantity.
3. Among remaining observations, find wider matches using
   `order_replacement_churn_gate_tracking_tolerance_pct`.
4. A newest contiguous run of tight matches spanning
   `order_replacement_churn_gate_stability_minutes` proves that the current price and quantity have
   settled; older wider evidence is ignored.
5. Before that stability horizon is reached, any wider-but-not-tight match is current churn
   evidence. A missing association is uncertainty/new intent and fails open rather than inheriting
   older evidence across the gap.
6. Tight-only history shorter than the stability horizon, no history, or no wider association has
   no proven churn evidence and is admitted.

The implementation may stop after proving either continuous recent stability or current
wider-but-not-tight evidence. It need not build tracks, tombstones, predecessor identities, or
causal classifications.

### Why newer stability clears older evidence

The target is sustained replacement churn, not a permanent penalty for one earlier move. Without a
stability horizon, a rejected create or manual cancellation after the order settles could remain
deferred until an unrelated old roll ages out of the full window. The newest tight run therefore
supersedes older wider evidence once it spans the configured stabilization duration.

The run compares the current candidate to every snapshot in the contiguous prefix, not merely to
the immediately preceding snapshot. Slow cumulative drift can therefore break tight stability even
when each adjacent step is individually small. A gap fails open because Python cannot prove the
same intent continued across it.

“Contiguous” is explicit, not inferred from two timestamps. Increment a planning-generation counter
for every attempted account-wide Rust planning cycle, including failures. A stability run may use
only immediately consecutive successful generations; any unavailable/failed/invalid generation
breaks it. Also require the current decision and every adjacent snapshot in the run to be separated
by no more than three nominal quiet execution cadences, with a 10-second floor. One nominal quiet
cadence is `live.execution_delay_seconds` plus the execution loop's maximum 30-second scheduled
wait for a websocket trigger. The scheduled wait is ordinary runtime behavior and therefore is not
itself an evidence discontinuity. A larger wall-clock gap breaks the run even if no attempt was
recorded. The tight prefix must span the full configured stability duration and contain at least
two prior snapshots. After any generation or time gap, the candidate fails open until new
contiguous evidence accumulates; two tight snapshots several minutes apart cannot clear older
evidence.

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
price/quantity instability, not durable order identity. Near-market admission, the initial rolling
allowance, and fail-open unmatched orders bound the consequence.

## Admission Policy

### Allowance exemptions

The rolling churn allowance never delays:

- market orders;
- HSL panic and the dedicated protective-panic path;
- orders Rust explicitly marks `execution_priority=risk_critical`; and
- limit orders whose final fresh market distance is at or inside
  `order_replacement_churn_gate_market_dist_pct`.

These exemptions apply only to churn-allowance waiting. They do not bypass readiness,
authoritative-state, batch, or cancel-first sequencing. A non-panic Rust-promoted market order and a
risk-critical limit still wait for cancellation, full confirmation, and a fresh plan whenever any
stale actual exists account-wide. Dedicated protective market panic is the sole same-wave
cancel-first bypass.

`reduce_only=true` alone is not an exemption. EMA-gated closes can churn too. Rust must explicitly
own risk priority for unstuck, WEL/TWEL, auto-reduce, graceful-stop, cooldown re-panic, and other
exposure-reducing families. The implementation audit must classify every current family rather than
letting Python infer urgency from names.

### Account-wide rolling allowance

The default ordinary allowance is evaluated against all connector-bound create attempts and any
connector-bound configuration attempts reserved by those creates in a rolling ten-minute window
for one bot execution authority. With the default of ten, a far churn-evidenced ordinary create is
admitted when its create-plus-required-configuration reservation leaves **at most ten** active
attempts; defer only when the reservation would produce the eleventh or later attempt.

Count one logical order when it reaches the concrete connector create-call boundary, including
failed, rejected, timed-out, and ambiguous attempts and each logical member of a batch. Count each
required connector-bound configuration action exactly once when it reaches its connector boundary,
including failed, rejected, timed-out, pending, and ambiguous outcomes. A proven-unsent reservation
is released. Do not count local deferrals or cancellations.

Near-market, market, and explicitly risk-critical actions never wait for allowance, but every one
counts for subsequent ordinary admission because it consumes the same connector and account action
capacity. Allowance exemption means “never wait for economy capacity,” not “bypass sequencing” or
“invisible to accounting.” A burst of exempt actions may therefore hold later far churn-evidenced
ordinary creates until timestamps expire; it may never delay the exempt action itself.

When one timestamp expires, at most one newly far attempt gains capacity before consuming that
slot. There is no bulk release and no persistent queue of deferred orders.

### Hyperliquid action-headroom overlay

Hyperliquid does not use time-window address capacity. Its adapter therefore adds a second,
connector-specific admission predicate for far churn-evidenced ordinary creates. It periodically
queries the official `userRateLimit` info surface, caches `nRequestsUsed`, `nRequestsCap`, and
`nRequestsSurplus`, and debits every observed local signed address action since that snapshot.
Create/cancel batch members count individually; any connector action not routed through the central
counter invalidates the cached estimate and requires refresh before an ordinary far create.
Cancellations and allowance-exempt creates are never delayed by this overlay but still debit the
cached estimate.

Linearize each refresh around explicit local action-lifecycle records, not only a request-start
timestamp. Record an action token before the connector call and mark it terminal only after an
authoritative, non-ambiguous response. A raised exception, timeout, empty/partial response, or other
ambiguous outcome remains unresolved and locally debited for the rest of the bot process unless a
separately reviewed authoritative terminal-confirmation path proves its disposition. A successful
refresh may retire only terminal actions completed before its request started. It must carry and
subtract every unresolved action, plus every action attempted or terminally completed while the
info request was in flight, because the returned server snapshot cannot prove that any of those
actions was observed. A later successful refresh may retire a carried terminal debit once that
action was completed before the later request began. This intentionally understates headroom after
ambiguous actions but never overstates it. Failed refreshes preserve lifecycle records; they do not
fabricate a new inclusion boundary.

Compute snapshot headroom as
`max(0, nRequestsCap - nRequestsUsed + nRequestsSurplus)`, then subtract every logical local address
action attempted after that snapshot. The official response defines `nRequestsUsed` as
`max(0, cumulative_used - reserved)` and `nRequestsSurplus` as
`max(0, reserved - cumulative_used)`. Surplus is therefore unused prepaid request weight which
extends headroom once used has bottomed out at zero; omitting it discards real reserved capacity.
Reject non-finite, negative, or contradictory fields rather than clamping invalid payloads into
availability. Fills may increase server capacity between polls, so local debiting can only
understate headroom; refresh recovers it.

If the authoritative response plus local debits shows no normal address-action headroom, defer far
churn-evidenced ordinary creates even if the generic ten-minute window has capacity. The generic
window remains a conservative write-economy ceiling when headroom exists; it is not treated as
capacity refill. A reservation is admissible when it exactly consumes the reported remaining
headroom; defer only when projected signed actions plus the reservation exceed headroom. A stale or
unavailable `userRateLimit` snapshot may not be replaced by invented
headroom: defer only the far churn-evidenced ordinary class, refresh the info surface under existing
pacing, and continue cancellations, near-market, market, and risk-critical actions. The exact
server-field arithmetic above and any future safety reserve must be rechecked against current
official response semantics during implementation review; do not reverse-engineer refill from HTTP
timing.

### Final market-distance check

Distance decisions made during planning are provisional. After cancellations and authoritative
barriers, recheck fresh signed distance before any candidate-specific exchange configuration or
margin write. A candidate that moved near is admitted. A candidate that moved far is deferred for
capacity only when its retained Rust-history classification proves churn evidence; stable or
uncertain candidates keep their fail-open admission and are not capacity-gated. Once any admitted
candidate or its required configuration write reaches the connector, however, every attempt is
debited for subsequent churn-evidenced admission. Missing, non-finite, or stale market data defers
an affected ordinary create. Sort by priority and apply
`max_n_creations_per_batch` before reserving or mutating configuration, so a batch-trimmed candidate
cannot consume configuration capacity. Only admitted, selected candidates may cause exchange
configuration work. For connectors whose configuration mutates the same account action budget, the
first admission reserves the required configuration actions and the create together; configuration
costs shared by multiple creates on one symbol count once.

After configuration, retain the existing final market-snapshot safety guard and then rerun churn
distance from a forced-fresh, no-candle-fallback market read plus allowance/headroom admission
immediately before the connector create call. Every connector-bound configuration attempt is
debited exactly once before awaiting its result, including rejected, timed-out, pending, or
ambiguous outcomes; only a reservation proven unsent is released. A dependent create proceeds only
after the configuration is authoritatively applied or proven already correct. Rejected, timed-out,
pending, or ambiguous configuration leaves that symbol pending and removes its creates from the
wave. For Hyperliquid, the expected CCXT success response must contain `status=ok` and the native
`response.type=default`; empty or partial payloads are not terminal success and retain their local
action debit. Attempted configuration actions are not reserved a second time. A candidate that moved from
near to far during configuration is
re-evaluated with its existing Rust-history churn classification: ordinary capacity applies only
when it remains churn-evidenced, while stable or uncertain far candidates still fail open. A
deferred candidate may therefore have configuration work only when it passed the earlier
reservation-backed admission and was invalidated by a later market move or capacity change.

Use the existing side-aware convention: `1 - order_price / market_price` for buys and
`order_price / market_price - 1` for sells. Admit when the result is less than or equal to the
threshold, so a marketable negative distance is also inside the gate. Validate the market price
before calling the helper; its legacy zero fallback for an invalid market price must not become a
false near-market admission.

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

### Account-wide create barrier

Write sequencing is account-wide, not merely order-cohort-, position-side-, or symbol-scoped. Any
stale actual order may partially fill before or during cancellation and change wallet balance,
realized PnL, TWEL, entry gates, risk ordering, and portfolio allocation for other symbols in the
same Rust plan. Without a Rust-owned dependency graph, no non-panic creation from that pre-cancel
plan is proven fresh.

Effective hedge/one-way mode still matters for exact actual-order attribution and diagnostics, but
does not narrow this barrier. This conservative account-wide stall is the same dependency boundary
used for planning failure and compatibility epochs. A future narrower barrier requires explicit
Rust dependency metadata and separate review.

### Ordinary two-phase flow

For each planning wave:

1. Reconcile authoritative actual orders with the current valid Rust ideal.
2. If any actual order has a malformed required field or ambiguous position-side/entry-close
   semantics, treat the account-critical open-orders surface as unavailable: send no cancellation
   or creation and request a clean authoritative refresh. An intentionally unmanaged actual in
   `manual` or the entry portion of `tp_only` is not stale managed work and does not arm the
   cancellation barrier below, but it must still be fully well formed; a missing or contradictory
   required field on an unmanaged order arms this account-wide no-write readiness barrier.
3. Otherwise, if any unmatched stale actual exists in proven bot-managed semantic scope, suppress
   every non-panic creation account-wide for the entire wave—whether its cancellation is selected,
   truncated, succeeds, fails, or is reported absent—and send only the selected cancellation batch
   under existing limits and pacing.
4. Request authoritative balance, positions, open-orders, and fills confirmation and end all
   non-panic creation work for the wave. A positive cancellation acknowledgement does not prove
   that no partial fill changed account state.
5. On the next loop, refresh state, regenerate the Rust plan, reconcile again, and create only in
   a wave with no stale actual order.

Every ordinary market creation and every risk-critical limit obeys this barrier; market execution
is not itself an emergency exemption. The account pays at least one full planning loop of latency
after any stale cancellation, and possibly more when cancellation batching leaves stale actuals.

Every stale cancellation outcome requires full balance, positions, open-orders, and fills
confirmation before replanning. Failed, ambiguous, not-found, already-filled, already-cancelled,
and positive acknowledgements all may follow a partial or concurrent fill. A future connector fast
path may narrow this only when its response contract proves no fill and no account-state change.

### Risk-critical sequencing

Dedicated market panic may bypass the ordinary dependency path entirely only when its current order
payload is authoritatively close-only/reduce-only for the affected position. Every Rust-marked
risk-critical limit order remains ahead of ordinary orders in final batching, but if any stale
actual exists account-wide it must wait for cancellation, authoritative positions, balance,
open-orders, and fills confirmation, and a fresh Rust plan. A malformed or non-close-only panic
never bypasses readiness or sequencing. A positive cancel acknowledgement is insufficient: the
cancelled order or another order may have filled before the acknowledgement and changed position
size, PnL, risk, or close quantity.

Once admitted through that dedicated path, the same reduce-only market panic is exempt from the
generic post-cancellation `state_change_detected_by_symbol` create-delay filter; ambiguous or
already-gone cancellation outcomes must not accidentally remove the sole emergency bypass. This
does not weaken the earlier readiness rule: unproven position state blocks every exchange write
before a panic can reach this path.

This keeps the only same-wave bypass narrow and explicit. If later measurements show that one-loop
risk-critical limit latency is unacceptable, design a separate fast path that proves no fill or
account-state change and recomputes current Rust intent; do not infer safety from cancellation
success alone.

### Why one-cycle latency is accepted

For ordinary orders, a fresh Rust plan after confirmed cancellation is simpler and safer than
carrying per-create dependency objects through batch truncation, partial success, fills, and
position changes. The account's non-panic creations pay one loop of latency; dedicated emergency
market panic continues. This is an explicit reliability tradeoff, not an accidental delay.

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
> create non-ideal orders, relax readiness/risk/mode-scope checks, preserve stale orders, or queue
> obsolete intent. Reset must be observable and covered by tests.

Plan approval selects that intended exception; runtime implementation remains blocked until the
canonical instruction and principle are updated together.

## Proposed Configuration

Add canonical `config.live` fields:

| Field | Default | Meaning |
|---|---:|---|
| `order_replacement_churn_gate_activation_count` | `10` | Rolling account-wide connector-action count below which far churn-evidenced ordinary creates may proceed. Logical creates and their required configuration writes count; cancellations do not. Exempt creates always proceed but count. `0` disables only far churn-evidenced economy deferral; exact reconciliation, stale cancellation, and cancel-confirm-refresh-replan safety remain active. |
| `order_replacement_churn_gate_window_minutes` | `10.0` | Per-symbol evidence and account-wide attempt window. |
| `order_replacement_churn_gate_stability_minutes` | `2.0` | Newest contiguous tight-match duration that clears older wider churn evidence. |
| `order_replacement_churn_gate_market_dist_pct` | `0.005` | Final market-distance threshold that always exempts a limit creation from allowance waiting; 0.5%. |
| `order_replacement_churn_gate_tracking_tolerance_pct` | `0.002` | Wider price/quantity tolerance used only for historical churn evidence; 0.2%. |

`order_replacement_churn_gate_activation_count = 0` is an explicit special case: skip the
far-churn economy-capacity predicate entirely rather than comparing a reservation against zero.
Priority, batch caps, reconciliation, readiness, and cancel-first safety remain active.

The existing `live.order_match_tolerance_pct`, normally `0.0002` (0.02%), remains the actual-order
equivalence and tight historical tolerance.

Validation:

- activation count is an integer greater than or equal to zero;
- window is finite and greater than zero when enabled;
- stability duration is finite, greater than zero, and no greater than the window;
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

Official documentation was reviewed on 2026-07-19 through 2026-07-21. Limits and tiers change; the
implementation review must recheck the exact endpoints used by each connector.

| Exchange/model | Official behavior relevant to this plan | Ramification |
|---|---|---|
| Hyperliquid | REST has weighted IP limits. Address action allowance is tied to cumulative USDC volume, counts logical actions within batches, grants extra cancellation allowance, exposes `userRateLimit`, and falls back to one request per ten seconds once limited. [Rate-limit docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits), [`userRateLimit` info](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint) | Count logical actions, not HTTP requests. Add the authoritative action-headroom overlay; elapsed ten-minute time never fabricates refill. Do not suppress cancellations or fill-producing near-market/risk actions. |
| Bybit | Private limits use rolling per-second UID windows plus IP limits; batch consumption is based on order count. [Official docs](https://bybit-exchange.github.io/docs/v5/rate-limit) | The ten-minute economy policy supplements short-window pacing. |
| Bitget | Futures cancellation has UID-scoped endpoint limits. [Official docs](https://www.bitget.com/api-doc/classic/contract/trade/Batch-Cancel-Orders) | Existing connector pacing remains authoritative. |
| KuCoin | Private quotas use UID resource pools and endpoint weights over short windows. [Rate-limit docs](https://www.kucoin.com/docs-new/rate-limit), [futures order docs](https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order) | Header-adaptive pacing is separate work. |
| Gate.io | Futures placement/amend and cancellation have separate UID limits. [Official docs](https://www.gate.com/docs/developers/apiv4/en/) | High nominal capacity does not make indefinite low-fill churn useful. |
| WEEX | Place/cancel and trigger actions have endpoint-specific account/UID or IP limits. [Official docs](https://www.weex.com/api-doc/ai/QuickStart/LIMITS) | The gate targets sustained churn; CCXT remains responsible for bursts. |
| Binance USD-M | New orders consume account order-count limits; placement and cancellation have different request weights and batch bounds. Qualifying reduce-only/close and cancellation requests receive special overload treatment. [General info](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info), [trade endpoints](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#new-order) | Use USD-M, not Spot, assumptions. Preserve risk/cancel priority and count logical actions. |
| OKX | Trading limits are shared across REST/WS, separated by place/amend/cancel operation, and generally scoped by user plus instrument; subaccount fill-ratio rules affect new/amend traffic. [Official docs](https://www.okx.com/docs-v5/) | Native amend and adaptive fill-ratio optimization are later projects. |
| Paradex | Private APIs use account plus IP limits with leaky-bucket refill; batch operations may consume one unit for multiple orders. [Rate limits](https://docs.paradex.trade/api/general-information/rate-limits/api), [best practices](https://docs.paradex.trade/api/general-information/api-best-practices) | Logical action counting is deliberately conservative. |
| CCXT/generic fallback | CCXT rate limiting is per exchange instance; separate instances do not share limiter state. [Official manual](https://github.com/ccxt/ccxt/wiki/manual#rate-limit) | Comparative context only. Arbitrary generic-fallback exchanges are outside implementation and rollout scope until their order attribution and remaining-quantity contracts receive a connector audit. |

Defx is excluded. It is a deliberately unsupported legacy placeholder under
`docs/ai/features/exchange_integrations.md`; stale adapter and `setup_bot()` code do not make it part
of this feature's implementation, validation, research, or rollout scope.

Paradex is also outside the supported production and rollout boundary. Its matrix row is retained
only because its account/IP leaky-bucket and batch semantics are useful comparative research; it is
not an implementation or live-validation target for this feature.

The generic `CCXTBot` fallback is likewise excluded from this feature. `setup_bot()` must mark the
churn reconciliation, evidence, cancel-first barrier, admission, and accounting policy disabled for
any exchange outside the explicit supported production allowlist. The local `fake` connector is an
explicit non-production exception: it enables the complete feature so deterministic fake-live can
exercise the same reconciliation, history, sequencing, and accounting paths. This does not remove
or reject the generic adapter; it preserves that adapter's prior reconciliation/execution behavior
for basic matching, tolerance, batching, and sequencing until the named connector receives the same
authoritative metadata audit as a supported connector. The old shared initial-entry-only distance
gate is still retired atomically for every connector; its removal does not opt the generic adapter
into temporal churn evidence, strict reconciliation, cancel-first, or action-attempt accounting.

Hyperliquid is the clearest reason not to call the generic window a rate-limit budget. Its
cancellation allowance and requests-per-volume economics differ materially from rolling endpoint
quotas. The connector-specific authoritative headroom overlay prevents the generic window from
pretending capacity refills with elapsed time. Cancellations and risk/fill-producing orders retain
priority and are debited, not delayed, by that overlay.

## Live/Backtest Parity

Rust ideal generation is unchanged. The existing tiny universal tolerance remains the only reason
live may leave a slightly different actual order resting.

The explicit live-only difference is omission: after stale cancellation, live may temporarily omit
a far current ideal order with proven recent churn when the allowance is full. It never substitutes
an older price. This can miss a sharp move before the order becomes near-market or capacity returns.

The consequence is bounded by:

- fail-open behavior for no/uncertain history;
- a ten-attempt rolling threshold counting every connector-bound create;
- unconditional final near-market allowance exemption;
- risk-critical and market no-wait exemptions which still count;
- one-for-one capacity release; and
- immediate use of fresh current Rust intent on every later cycle.

This tradeoff must be measured in fake-live and, only with separate authority, live operation.

## Failure Semantics

- Invalid config fails startup.
- Account-wide Rust planning failure appends no snapshots and performs no ordinary actions. A
  symbol-scoped failure after a complete Rust result permits healthy-symbol actions only when the
  affected symbol is authoritatively flat and has no actual open orders. If its position is
  authoritatively nonzero or any actual order remains, block every account-wide non-panic creation;
  dedicated close-only/reduce-only market panic remains the sole create bypass. If either position
  side is missing, malformed, non-finite, or otherwise unproven, block every exchange write because
  panic side and quantity are not trustworthy. Any other unavailable account-critical surface such
  as open orders likewise blocks every exchange write.
- A valid empty symbol plan is authoritative and may cancel all stale orders in bot-managed scopes
  for that symbol.
- Unknown `pb_order_type` or execution type cannot satisfy a current ideal but does not by itself
  exempt an actual order from cancellation in a proven bot-managed scope. Unknown or contradictory
  position-side/close-only semantics prevent proving whether the actual order is entry or close and
  whether `manual`/`tp_only` applies; do not cancel it and block every exchange write account-wide
  until authoritative semantics resolve the scope.
- Missing/stale final market data defers ordinary creation.
- Missing/stale Hyperliquid action-headroom state defers only far churn-evidenced ordinary creates;
  it never delays cancellations or allowance-exempt actions.
- Ambiguous create attempts count once when connector-bound and use existing confirmation guards.
- Required configuration attempts count once when connector-bound, including ambiguous failures;
  proven-unsent reservations are released.
- Every stale cancellation outcome blocks all non-panic creation account-wide and requires full
  balance/positions/open-orders/fills confirmation plus complete replanning.
- Event publication failure changes no trading decision.
- Existing exchange pacing, 429/backoff, readiness, mode scope, semantic attribution, risk, HSL,
  WEL/TWEL, and
  authoritative-state barriers retain precedence.

## Observability

Add bounded diagnostic events/reason codes for:

- valid per-symbol snapshot observed, compacted, skipped, or reset;
- tight, wider-but-not-tight, absent, and ambiguous historical association;
- churn evidence present/absent;
- admission by exemption, distance, or rolling allowance;
- deferral by allowance or unavailable final distance;
- account-wide post-cancellation create suppression;
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
- Cancel-first sequencing adds at least one loop of non-panic creation latency account-wide; it
  removes stale-plan and truncated-cancel duplicate hazards across Rust's global dependencies.
- RAM reset temporarily restores baseline write churn; it does not weaken the current Rust plan or
  safety barriers.
- Account-wide means one bot execution authority, not multiple processes sharing exchange limits.
- Risk-critical limit orders incur refresh/replan latency whenever any stale actual exists; only
  dedicated market panic bypasses that sequencing barrier.

## Resolution of Review Findings

### Latest design questions

1. **Scoped planning outage:** history and validity are per symbol; the processing universe includes
   plans, actual orders, positions, and history. Healthy symbols continue only when authoritative
   state proves the affected symbol flat with no actual orders. Actual orders or an authoritative
   nonzero position block account-wide non-panic creates, while unproven position state blocks every
   exchange write. Recovery without history intentionally receives fail-open placement only after
   those readiness conditions hold.
2. **Ambiguous order identity and equal-cardinality ladders:** authoritative narrative
   classifications and temporal tracks are removed. Policy uses only wider-but-not-tight
   quantitative evidence and explicitly accepts bounded false positives/negatives.
3. **`pb_order_type`:** it is restored as an exact semantic key after auditing current fill-history
   use. The rationale and unknown-attribution behavior are explicit.
4. **Close-only effect:** canonical `reduce_only` remains an exact key, but means authoritative
   venue-native close-only effect; generic side/position-side inference is forbidden.
5. **Cancel/create races:** all non-panic creates account-wide wait for cancel confirmation, fresh
   state, and a fresh Rust plan. Batch truncation cannot strand a dependent creation.
6. **Ordinary market promotion:** execution type is removed from the sequencing scope, so a
   Rust-promoted market creation cannot bypass its stale limit predecessor.
7. **WEEX close semantics:** WEEX `COMBINED` mode must normalize close-only effect from the
   authoritative V3 `side` plus `positionSide` action tuple. The response's `reduceOnly` literal
   is not authoritative because regular V3 placement cannot set it and valid closes may report
   `false`.
8. **Defx metadata:** Defx is explicitly outside the supported live-exchange boundary and is not an
   implementation prerequisite.
9. **Cancel acknowledgement after fills:** every limit order, including risk-critical, refreshes and
   replans after a stale cancellation; positive acknowledgement alone is insufficient.
10. **Time-in-force provenance:** placement-only TIF/post-only is removed from reconciliation and
    history identity while remaining enforced on new creations.
11. **Operator policy changes:** compatible account/scoped histories reset on planning-policy epoch
    changes without refunding account-wide attempt timestamps.
12. **Partial-fill quantity:** actual matching uses authoritative remaining open quantity, never
    original submitted amount after a partial fill.
13. **Paradex scope:** Paradex is explicitly experimental and excluded from production rollout;
    its rate-limit row is comparative research only.
14. **Cross-side cancellation races:** any stale actual activates the account-wide non-panic create
    barrier, so no entry/close or cross-symbol stale-plan create can race its cancellation.
15. **Exempt action accounting:** market, risk-critical, and near-market creates always proceed but
    count against later ordinary capacity.
16. **Account-wide Rust failures:** no per-symbol salvage is attempted from an all-or-nothing Rust
    planning failure; symbol-scoped validity begins after a complete account plan.
17. **Fill phase changes:** authoritative fills and account-state transitions reset compatible
    account-wide history without refunding action timestamps.
18. **Cancellation confirmation:** every stale cancellation requests full account-state and fill
    confirmation unless a future connector proves a no-fill outcome.
19. **One-way conflicts:** effective one-way mode is explicit in attribution and epoch state; the
    final cancellation barrier is now account-wide for hedge and one-way modes alike.
20. **Bitget UTA close semantics:** its documented `side` plus `posSide` hedge action overrides a
    literal `reduceOnly=NO`; classic/one-way Bitget handling stays separate.
21. **Cross-symbol risk state:** fill identity, snapped balance/position inputs, Rust-reported risk
    phases, global configuration changes, and uncertain dependencies advance an account epoch and
    clear all ideal history; price-derived exchange state does not.
22. **Runtime eligibility:** effective `PB_modes`, approved/ignored sets, forager membership, and
    active-universe membership are symbol-scoped compatibility inputs; hedge capability remains
    account-wide.
23. **Market wording:** market/risk/near exemptions apply only to allowance waiting; dedicated
    protective market panic is the sole account-wide stale-barrier same-wave bypass.
24. **Recovered stability:** a newest contiguous tight run spanning the stability horizon clears
    older wider evidence; gaps fail open.
25. **Hyperliquid capacity:** authoritative `userRateLimit` headroom plus local logical-action
    debits overlays the generic economy window; elapsed time never fabricates address refill.
26. **Hyperliquid surplus arithmetic:** headroom is `cap - used + surplus`; the mutually exclusive
    used/surplus fields are validated before unused prepaid weight is admitted.
27. **Stability gaps:** failed generations and explicit derived cadence gaps break the tight prefix;
    sparse snapshots cannot prove stability.
28. **Price-driven epoch fields:** only snapped balance, signed position size/average entry, fill
    identity, Rust-reported risk phases, and static/runtime policy enter epochs; mark-to-market
    fields are excluded. Exact raw balance remains a Rust risk input, not an epoch scalar.
29. **Cross-symbol cancel dependency:** any stale cancellation blocks every non-panic create until a
    full authoritative refresh and complete Rust replan.
30. **One-way position attribution:** prefer durable PB metadata, otherwise use the reviewed
    order-side plus close-only tuple; current position inference is forbidden.
31. **Realized-PnL enrichment:** account epochs include the exact cumulative max/last values passed
    to Rust, so fee/PnL enrichment cannot retain pre-enrichment churn evidence.
32. **Management scope, not ownership:** normal bot-managed scopes reconcile and cancel every
    actual order regardless of who submitted it; `manual` manages none; `tp_only` manages closes
    only; `tp_only_with_active_entry_cancellation` manages closes plus stale-entry cancellation;
    missing ownership metadata is irrelevant, while unknown entry/close or position-side semantics
    fail closed because mode scope cannot be proven.
33. **OKX and KuCoin attribution:** OKX long/short mode uses its native side-plus-`posSide` action
    tuple instead of CCXT's emulated reduce-only default. KuCoin open orders never use current
    positions as `position_side` evidence.

### Earlier findings retained

- The allowance governs churn-evidenced creations, never cancellations.
- Capacity releases one attempt at a time; no deferred-order burst or stale-intent queue exists.
- Final distance is checked close to the connector call.
- Risk-critical candidates sort ahead of ordinary candidates before final batch slicing.
- Already-absent cancellation triggers authoritative confirmation and replanning.
- `_context=replace`, order names, client IDs, and exchange IDs are non-causal for churn evidence.
- Every connector enabled for this feature is a directly audited supported connector with focused
  normalization fixtures; generic CCXT normalization and unsupported/experimental connectors fail
  closed and remain outside the gate.
- Binance assumptions are explicitly USD-M, not Spot.
- Restart behavior is an intentional bounded exception requiring a canonical contract change first.

## Required Tests and Evidence

### Matching and normalization

- deterministic one-to-one tight actual reconciliation, including duplicate prices/quantities;
- partially filled actual orders match by authoritative remaining quantity; original amount,
  missing remaining, and contradictory amount/filled/remaining fixtures cannot silently match;
- exact mismatch for `pb_order_type`, panic versus non-panic, canonical venue-native close-only
  effect, and execution type;
- TIF/post-only differences do not prevent an otherwise exact resting-order match, while every new
  creation still receives the configured placement semantics;
- unknown `pb_order_type` or resting execution type is cancel-safe but cannot satisfy a known ideal;
  missing or contradictory close-only effect or `position_side` fails closed account-wide for every
  supported connector adapter;
- no generic side/position-side inference of close-only; focused WEEX V3, Bitget UTA, KuCoin
  hedge/one-way, and OKX long/short-mode action-tuple fixtures prove their explicit
  connector-native mappings;
- one-way pside attribution prefers durable PB metadata, then covers all four side/close-only tuples;
  current-position changes cannot relabel a resting order and unknown attribution fails closed;
- normal-mode user-created orders and PB-created orders reconcile by the same exact semantics;
  unmatched user orders are cancelled and user-cancelled current ideals are recreated;
- `manual`, `tp_only`, and `tp_only_with_active_entry_cancellation` fixtures prove respectively no
  management, close-only management, and close management plus stale-entry cancellation;
- missing client IDs and foreign Passivbot-looking markers do not exempt orders in managed scopes;
  ambiguous position-side or entry/close attribution fails closed without cancellation;
- one historical observation cannot satisfy multiple current candidates;
- a current ideal already satisfied by an actual order still reserves its historical mate before an
  unmatched peer is classified;
- raw list reordering and dictionary order do not change outcomes;
- WEEX balance normalization subtracts `unrealizePnl` from V3 account equity so position
  mark-to-market movement does not alter Rust raw balance. On any venue, a raw-balance risk change
  advances the churn epoch when it changes Rust's active risk pairs; a material wallet change also
  advances the snapped-balance epoch.

### Per-symbol history and heuristic

- current snapshot never matches itself;
- account-wide Rust planning failure appends no history and performs no ordinary reconciliation;
- downstream symbol-scoped normalization outage skips one symbol while healthy symbols continue
  only when that symbol is authoritatively flat with no actual orders; actual orders or a proven
  nonzero position block every non-panic create account-wide while dedicated protective market
  panic remains available, whereas unproven position state blocks every exchange write;
- valid empty is distinct from unavailable planning;
- processing-universe union retains actual/position/history-only symbols;
- first observation and restart reset fail open;
- tight-only history admits; wider-but-not-tight history marks churn;
- a newest contiguous tight run spanning the stability horizon clears older wider evidence;
- adjacent tiny steps whose cumulative current-to-snapshot drift breaks tight tolerance remain
  churn-evidenced until genuinely stable;
- failed planning generations, a stale current decision, and adjacent gaps above
  `max(10 seconds, 3 * execution_delay_seconds)` break stability and fail open;
- sparse tight snapshots cannot satisfy the stability duration or minimum-snapshot requirement;
- no wider association admits as uncertain;
- price-only and quantity-only churn;
- neighboring grids, duplicate levels, growing/shrinking ladders, equal-cardinality rolls;
- evidence expires exactly with the rolling window;
- a scoped coin-override revision resets compatible history; runtime approved/ignored membership,
  forager membership, effective mode, and active-universe changes reset only the affected symbol;
  global configuration and hedge-capability revisions reset account-wide history;
- an hourly `init_markets()` refresh that changes price/quantity steps, minimums, contract
  multiplier, active status, or another normalization constraint resets only the affected symbol's
  history before its next classification; unchanged metadata and volatile raw exchange `info` do
  not reset it;
- price, EMA, volatility, trailing extrema, equity, available margin, unrealized PnL, mark/liquidation
  price, and mark-to-market notional changes do not reset history;
- authoritative fill identity, realized-PnL cumulative max/last, Rust-reported active risk pairs,
  hysteresis-snapped wallet balance, signed position size, and average-entry-price transitions reset
  account-wide history
  before the first new-phase plan;
- every ideal order for a Rust-reported active risk pair fails open past churn evidence and cannot
  be distance- or allowance-deferred by the churn gate;
- a policy-history reset does not clear account-wide attempt timestamps;
- compaction, if implemented, preserves logical results across fast/slow planning cadence.

### Allowance and final admission

- ten attempts across multiple symbols share one process-wide allowance;
- eleventh far churn-evidenced ordinary create defers;
- one expired timestamp releases one slot;
- near-market churn attempts always bypass allowance waiting and count;
- market and risk-critical orders always bypass allowance waiting but count against subsequent
  ordinary admission; non-panic orders still obey dirty-scope sequencing;
- failed/rejected/timed-out/ambiguous connector-bound creates count exactly once;
- local deferrals count zero;
- final price movement near-to-far consumes capacity or defers only for an already churn-evidenced
  ordinary candidate; stable/uncertain far candidates still fail open and far-to-near admits;
- missing/stale/non-finite final market data defers without fabrication;
- concurrent candidates cannot oversubscribe current capacity;
- Hyperliquid `userRateLimit` headroom, local logical signed-action debits, unobserved-action cache
  invalidation, stale/unavailable snapshots, batch-member accounting, and exempt-action no-wait
  behavior; positive unused reserved surplus extends `cap - used` headroom exactly once;
- a Hyperliquid create-plus-config reservation equal to remaining headroom is admitted, while a
  reservation exceeding it is deferred;
- Hyperliquid headroom refresh includes actions attempted while the info request is in flight;
  configuration attempts are debited before awaiting success, and batch-trimmed candidates perform
  no configuration writes;
- ambiguous create, cancel, and configuration responses retain their signed-action tokens across
  later successful `userRateLimit` refreshes, while acknowledged or explicitly terminal responses
  retire only after a safe refresh boundary.

### Cancel-first executor

- any stale actual suppresses every non-panic create account-wide in hedge and one-way modes;
- a malformed required open-order field or unknown position-side/entry-close attribution suppresses
  every exchange write account-wide because the account-critical open-orders surface is unavailable;
- selected, truncated, failed, ambiguous, and absent cancellation outcomes;
- every stale cancellation outcome, including positive acknowledgement, requires next-cycle full
  balance/positions/open-orders/fills confirmation for ordinary work;
- fills and manual position changes regenerate different Rust intent before create;
- no independent symbol or position-side may create in the stale-cancellation wave;
- only authoritatively close-only/reduce-only dedicated market panic bypasses ordinary sequencing;
- an already-gone or ambiguous cancellation that marks `state_change_detected_by_symbol` still
  permits only that dedicated reduce-only market panic through the generic state-change filter;
  unproven position readiness blocks before this exception and permits no write;
- ordinary Rust-promoted market creation cannot escape a stale limit conflict;
- risk-critical limits replan after every stale cancellation, including positive acknowledgements
  whose response reports or may conceal a partial fill;
- no ordinary create is sent from a pre-cancellation plan;
- existing recent-create/ambiguous-write guards remain effective without duplicate responsibilities.

### Integration

- static grid, moving EMA Anchor entry and close, trailing, HSL, WEL/TWEL, unstuck, auto-reduce,
  graceful stop, forager changes, hedge mode, and one-way mode fake-live scenarios;
- batch saturation and connector-specific create/cancel bounds;
- RAM reset event and post-restart fail-open behavior;
- bounded event output and event-sink failure isolation;
- connector-bound action counts and omitted-order windows with gate enabled/disabled, including an
  explicit `activation_count=0` case that skips only churn-specific distance/headroom economy
  predicates. The universal forced-fresh market-snapshot guard, reconciliation readiness, priority,
  and batch caps still apply.

## Implementation Slices After Approval

### Slice 0: canonical RAM-only exception

- Update `AGENTS.md` and `docs/ai/principles.md` with the reviewed narrow economy-gate exception.
- Specify reset observability and tests.
- Do not begin decision-changing RAM implementation before this lands.

### Slice 1: exact semantic normalization

- Add canonical config fields, validation, templates, migration behavior, and changelog entry.
- Audit every connector for authoritative venue-native close-only effect, normalized
  `position_side` in both hedge and one-way modes, normalized `pb_order_type`, execution type, and
  authoritative remaining open quantity. Add focused Hyperliquid one-way, WEEX V3, Bitget UTA,
  KuCoin hedge/one-way, and OKX long/short action-tuple mappings without using current-position
  inference. Defx and Paradex are outside the supported production boundary.
- Make reconciliation eligibility explicit from effective mode plus entry/close semantics, not
  ownership: normal scopes manage all orders, `manual` none, `tp_only` closes only, and
  `tp_only_with_active_entry_cancellation` closes plus stale-entry cancellation. Unknown semantic
  scope is an explicit no-cancel fail-closed state; missing client ownership metadata is not.
- Keep TIF/post-only enforcement at creation and diagnostic normalization, not reconciliation.
- Audit Rust order families for explicit ordinary/risk-critical priority.
- Retire `initial_entry_exec_max_market_dist_pct`.

### Slice 2: pure per-symbol evidence helper

- Implement immutable snapshots, exact realized-PnL/snapped-balance/position and Rust risk-phase
  compatibility epochs,
  pruning/optional compaction, deterministic
  one-to-one tight/wider association, newest stability clearing, and account-wide rolling attempt
  accounting.
- Keep helpers independent of live events and exchange I/O.

### Slice 3: cancel-first reconciler/executor overhaul

- Separate exact satisfaction cohorts from the account-wide cancellation/create barrier.
- Suppress all non-panic creations whenever any stale actual order exists.
- Cancel, request full authoritative confirmation, end account-wide non-panic creation work, and
  replan next cycle.
- Preserve only the explicit dedicated market-panic emergency path.
- Consolidate overlapping recent-create, cancellation, and state-change delay machinery where tests
  prove simplification is safe.

### Slice 4: final admission and priority

- Recheck final fresh distance.
- Apply churn admission before candidate-specific exchange configuration or margin writes; deferred
  candidates must cause neither configuration mutations nor create attempts.
- Apply priority sorting and `max_n_creations_per_batch` before configuration reservation or writes.
- Reserve connector-reported signed configuration costs together with creates, then rerun churn
  admission with a forced-fresh market read after configuration and the existing market-snapshot
  guard immediately before create.
- Debit every connector-bound config attempt exactly once in both the generic action window and any
  connector-specific overlay. Linearize Hyperliquid headroom refreshes against unresolved actions
  and actions attempted while the info request is in flight.
- Apply allowance reservations atomically at connector boundary.
- Add Hyperliquid authoritative action-headroom refresh and local logical-action debits without
  delaying cancellations or exempt actions.
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
- reconciler/executor cancellation, ambiguity, account-wide barrier, final-distance, and batch tests;
- account-wide no-action readiness tests for missing or contradictory actual remaining quantity;
- static/moving/mixed-risk multi-symbol fake-live scenarios across every connector harness;
- live-event registry/query/sink-failure/bounded-projection tests;
- `PYTHONPATH=src python src/tools/check_ai_docs.py`;
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`;
- `git diff --check`.

Report exact base/head SHAs and cycle-level evidence containing current normalized Rust ideal,
actual order state, historical tight/wider result, churn evidence, cancellation outcome,
account-wide barrier state, rolling usage, final distance/priority, and connector-bound action.

## Rollout and Measurement

After offline validation and separate live authorization:

1. Run one controlled account with structured DEBUG execution evidence.
2. Measure ideal stability/churn, admissions, deferrals, cancellations, fills, 429s, ambiguity, and
   account-wide post-cancellation latency.
3. Prove the gate never leaves an out-of-tolerance stale actual order resting.
4. Inspect static grids, changing ladders, EMA Anchor entries/closes, trailing, forager changes, and
   one-for-one capacity release.
5. Inspect Hyperliquid separately: compare cached/debited `userRateLimit` headroom with observed
   action acceptance, requests-per-volume growth, and cancellation behavior.
6. Recheck Binance USD-M overload/risk behavior on the actual endpoints.
7. Tune defaults only from evidence; do not hide exchange-specific policy in adapters.

## Reviewer Checklist

Reviewers are explicitly asked to challenge intent, architecture, edge cases, unintended
consequences, and connector assumptions—not merely wording:

- Is the exact reconciliation key complete with `pb_order_type`, authoritative venue-native
  close-only effect, and execution type, while correctly excluding placement-only TIF/post-only?
- Are the WEEX V3 and Bitget UTA action-tuple mappings narrow, documented, and incapable of
  increasing/flipping the scoped position?
- Is one-way `position_side` attribution deterministic from durable PB metadata or the four reviewed
  side/close-only tuples, never current position state?
- Can every connector recover those fields without fabrication, and is fail-closed scope correct?
- Do normal, `manual`, `tp_only`, and `tp_only_with_active_entry_cancellation` apply the intended
  create/cancel scope to PB-created and user-created orders alike, without treating missing client
  IDs or foreign-looking markers as cancellation exemptions?
- Is unknown position-side or entry/close attribution guaranteed no-cancel until the effective mode
  scope can be proven?
- Is Defx's explicit unsupported boundary clear enough that stale adapter code cannot expand scope?
- Is Paradex's experimental, comparative-only boundary equally clear?
- Does any stale actual block every non-panic creation account-wide until full confirmation and a
  complete fresh Rust plan, with no cross-symbol same-wave escape?
- Can any cancellation truncation, ambiguity, fill, manual action, or race still send an ordinary
  order from a stale Rust plan?
- Does every non-panic market creation obey the stale-limit barrier?
- Is dedicated market panic the correct sole same-wave bypass, and are risk-critical limits still
  prioritized adequately after authoritative refresh/replan?
- Does per-symbol validity isolate outages while the union processing universe preserves actual
  orders and positions?
- Can one historical order satisfy multiple current candidates?
- Are deterministic wider associations sensible for duplicate grids, cardinality changes, and
  equal-cardinality rolls?
- Does the contiguous tight stability horizon clear stale evidence without allowing slowly drifting
  orders to evade detection?
- Do failed generations, explicit cadence gaps, and sparse snapshots break stability deterministically?
- Is failing open on no wider match preferable to trying to infer identity for large moves?
- Is 0.2% an appropriate initial evidence tolerance, and what fake-live cases could falsify it?
- Does the generic attempt counter model rolling-window connectors reasonably, and does the
  separate Hyperliquid `userRateLimit` overlay use `cap - used + surplus` without omitting prepaid
  capacity, inventing refill, or delaying exempt actions?
- Are near-market, market, and risk-critical attempts correctly exempt from allowance waiting yet
  still counted and still subject to all non-allowance safety barriers?
- Can a stale order ever remain while a replacement is gated?
- Is the one-cycle ordinary latency acceptable for parity and missed-fill risk?
- Is the proposed RAM-only canonical exception truly bounded to operational economy and incapable
  of weakening safety after reset?
- Do account/scoped compatibility epochs reset snapped balance, position, fill identity, exact
  realized-PnL cumulative max/last inputs, and Rust-reported risk-phase transitions account-wide,
  runtime modes/lists/forager membership only
  for affected symbols, and configured operator changes account-wide, while excluding equity,
  margin, unrealized PnL, mark/liquidation prices, other moving fields, and attempt-ledger refunds?
- What interactions remain with config reload, forager symbol churn, HSL, WEL/TWEL, unstuck,
  auto-reduce, graceful stop, hedge mode, and one-way mode?

Approval accepts the design and its documented tradeoffs. It does not authorize runtime code,
authenticated probes, live trading, or unreviewed changes to canonical safety contracts.
