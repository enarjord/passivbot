# Order Replacement Churn Gate

Status: implemented; simplified after architectural audit

This document records the current contract and the reasons for it. The earlier
account-wide design is retained in Git history, but it should not be used as an
implementation specification.

## Purpose

The gate is an economy mechanism. It reduces repeated creation of far-from-market
limit orders when recent Rust ideal-order snapshots prove that the same ladder
has been continuously drifting in price or quantity.

It is not:

- an order-validity or safety authority;
- a substitute for exchange reconciliation;
- a reason to leave a stale actual order resting;
- an exchange rate-limit model;
- a proof that configuration, coin lists, balances, or account state are
  unchanged.

Rust remains authoritative for the ideal order set. Reconciliation remains
authoritative for cancelling actual orders which do not match a current ideal.

## Non-negotiable safety boundary

Actual-to-ideal matching uses `live.order_match_tolerance_pct`, whose default is
`0.0002` (0.02%) and whose accepted range is 0% through 1% inclusive. If an
actual order does not match a current Rust ideal within that tolerance, it is
cancelled. Churn evidence never widens the tolerance and never preserves the
stale actual order.

The gate may defer only the replacement create. This distinction preserves
live/backtest intent during sharp market movement: an obsolete resting order is
removed even when the economy gate decides not to place its drifting
replacement yet.

Market orders, risk-critical orders, and all near-market limit orders bypass
churn deferral.

## Evidence model

For each normalized ideal limit order, retain recent Rust ideal snapshots for
the configured window (default: ten minutes).

Orders are grouped by a stable semantic cohort:

- symbol;
- position side;
- buy/sell side;
- reduce-only status;
- execution type;
- Passivbot order type.

Within a cohort, the sorted ladder rank associates successive snapshots. A
continuous price/quantity track is usable only while:

- the cohort exists in every inspected snapshot;
- cohort cardinality is unchanged;
- adjacent samples are separated by no more than the bounded planning cadence
  gap.

The order is churn-evidenced only when at least two consecutive price or
quantity moves have the same direction and the current directional run, measured
from its first changed observation, covers the configured stability interval. A
one-time jump, price oscillation, missing history, cadence gap, cohort change,
or cardinality change is insufficient for directional-drift evidence.

Repeated exclusive cohort switching is separate churn evidence. When every
snapshot in a contiguous sequence contains exactly one semantic cohort, a
current cohort which appears in at least three distinct runs while every
recurring cohort retains its ladder cardinality is churn-evidenced when the
fixed interval from the preceding appearance to the current run's start covers
the configured stability interval.
Time spent unchanged in the current run cannot turn a short switching episode
into evidence. This catches recurring long/short or order-type replacement
without merging their identities. The first appearance and first reappearance
remain fail-open; empty snapshots, coexisting cohorts, cadence gaps, and
ladder-shape changes break the proof.

Recent stability clears older drift. If the current ideal has remained within
the universal order-match tolerance for the stability interval, it is not
churn-evidenced. When every order in the sole current cohort completes that
tight stable interval, snapshots preceding the shared stable prefix are
discarded for the symbol so a later isolated switch cannot resurrect an older
switching episode. This decision-changing history boundary increments and emits
the existing churn-history reset telemetry.

Uncertain evidence fails open: reconciliation and creation proceed normally.

## Admission model

One churn-admission pass runs after exchange-configuration writes and the final
fresh market-snapshot guard, immediately before the connector create call. This
keeps both the generic maximum-distance guard and the near-market churn exemption
anchored to a snapshot which has not aged during configuration network calls.
Stable risk-first create-batch capacity is then applied only to admitted
candidates, so an order rejected by market freshness or churn policy cannot
consume capacity needed by a later admissible order.

The rolling create-attempt count uses the configured churn window and activation
count. Once the allowance is exhausted, only ordinary, churn-evidenced limit
orders farther from market than
`live.order_replacement_churn_gate_market_dist_pct` are deferred.

The market distance is calculated by that final generic pre-create guard. The
churn gate does not perform its own market fetch.

Connector calls remain responsible for their own exchange rate limiting.

## Deliberately removed machinery

The simplified implementation does not maintain:

- account-wide configuration or realized-state epochs;
- symbol epochs for approved/ignored lists or market metadata;
- a wider churn-only tracking tolerance;
- min-cost max-flow with composite integer costs;
- connector-specific request-budget headroom;
- reservations for exchange-configuration action costs;
- separate pre- and post-configuration admission passes;
- signed-action reservation and completion ledgers;
- reconciliation barriers caused only by churn-history normalization.

Those mechanisms tried to make an economy heuristic prove too much. Normal
runtime changes already disrupt the recent behavioral track naturally, and
uncertainty can safely disable churn evidence without disabling reconciliation.

The retired
`live.order_replacement_churn_gate_tracking_tolerance_pct` key is removed during
configuration migration. History now uses `live.order_match_tolerance_pct`.

## Validation contract

Regression coverage must prove:

1. no history and one-time movement fail open;
2. sustained monotonic price or quantity drift becomes evidence;
3. price oscillation, gaps, isolated cohort changes, and ladder cardinality
   changes fail open;
4. repeated exclusive cohort reappearance becomes evidence without merging
   position sides or other cohort identities;
5. recent stability clears old drift and exclusive-switching evidence;
6. near-market, market, risk-critical, and non-churn orders remain admissible;
7. missing market distance affects only a churn-evidenced candidate;
8. actual orders outside 0.02% are still cancelled even when their replacement
   carries churn evidence;
9. matching remains deterministic, one-to-one, and maximum-cardinality;
10. diagnostics cannot change admission or reconciliation.

## Operational review

When trialling this change on a live checkout, inspect rather than tune first:

- churn evidence reason counts;
- deferred-create reason counts;
- cancel/create attempt rates by symbol and position side;
- the share of candidates classified near market;
- evidence resets caused by natural gaps or ladder-shape changes;
- any repeated stale-order cancellation without a later valid create.

Do not infer success merely from fewer creates. The expected result is fewer
far drifting replacements while stale-order cancellation, risk actions, and
near-market execution remain unchanged.
