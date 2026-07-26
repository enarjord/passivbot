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
`0.0002` (0.02%). If an actual order does not match a current Rust ideal within
that tolerance, it is cancelled. Churn evidence never widens the tolerance and
never preserves the stale actual order.

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
track is usable only while:

- the cohort exists in every inspected snapshot;
- cohort cardinality is unchanged;
- adjacent samples are separated by no more than the bounded planning cadence
  gap.

The order is churn-evidenced only when at least two consecutive price or
quantity moves have the same direction and the observed span covers the
configured stability interval. A one-time jump, oscillation, missing history,
cadence gap, cohort change, or cardinality change is insufficient.

Recent stability clears older drift. If the current ideal has remained within
the universal order-match tolerance for the stability interval, it is not
churn-evidenced.

Uncertain evidence fails open: reconciliation and creation proceed normally.

## Admission model

Stable risk-first create-batch capacity is applied before exchange-configuration
writes, so capacity-deferred ordinary symbols cannot delay configuration of an
admitted risk-critical order.

One churn-admission pass runs after exchange-configuration writes and the final
fresh market-snapshot guard, immediately before the connector create call. This
keeps both the generic maximum-distance guard and the near-market churn exemption
anchored to a snapshot which has not aged during configuration network calls.

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
3. oscillation, gaps, cohort changes, and ladder cardinality changes fail open;
4. recent stability clears old drift;
5. near-market, market, risk-critical, and non-churn orders remain admissible;
6. missing market distance affects only a churn-evidenced candidate;
7. actual orders outside 0.02% are still cancelled even when their replacement
   carries churn evidence;
8. matching remains deterministic, one-to-one, and maximum-cardinality;
9. diagnostics cannot change admission or reconciliation.

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
