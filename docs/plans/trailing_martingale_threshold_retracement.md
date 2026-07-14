# Trailing Martingale Threshold/Retracement Redesign

Design record for the v8 replacement for the old `trailing_grid` strategy.

Canonical strategy kind: `trailing_martingale`.

## Goal

Model the strategy as recursive martingale/DCA order generation with optional trailing
confirmation and volatility/WE-scaled thresholds:

- `threshold` defines the distance from the reference price where an entry or close becomes
  eligible.
- `retracement` defines the confirmation move after the threshold excursion.
- `retracement <= 0.0` means passive recursive limit-order behavior.
- `retracement > 0.0` means trailing behavior.

This removes the separate `*_grid_ratio` controls and avoids the current overlap between
`grid_spacing_pct`, `trailing_threshold_pct`, and grid/trailing interpolation logic.

The old "grid" behavior remains available as the `retracement <= 0.0` mode, but the strategy is
not fundamentally a grid bot. Its defining traits are recursive martingale sizing, optional
trailing confirmation, and volatility/WE-scaled distances.

## Proposed Schema

Under `bot.<side>.strategy.trailing_martingale`:

```json
{
  "ema_span_0": 770,
  "ema_span_1": 210,
  "entry": {
    "double_down_factor": 0.0,
    "initial_qty_pct": 0.0,
    "initial_ema_dist": 0.0,

    "threshold_base_pct": 0.0,
    "threshold_we_weight": 0.0,
    "threshold_volatility_1m_weight": 0.0,
    "threshold_volatility_1h_weight": 0.0,

    "retracement_base_pct": 0.0,
    "retracement_we_weight": 0.0,
    "retracement_volatility_1m_weight": 0.0,
    "retracement_volatility_1h_weight": 0.0
  },
  "close": {
    "qty_pct": 0.0,

    "threshold_base_pct": 0.0,
    "threshold_we_weight": 0.0,
    "threshold_volatility_1m_weight": 0.0,
    "threshold_volatility_1h_weight": 0.0,

    "retracement_base_pct": 0.0,
    "retracement_volatility_1m_weight": 0.0,
    "retracement_volatility_1h_weight": 0.0
  }
}
```

Naming note: use `we_weight` rather than `wallet_exposure_weight` if the rest of the v8 schema
continues using `WE`, `WEL`, and `TWEL` terminology.

`ema_span_0` and `ema_span_1` sit at the strategy root because the resulting EMA bands are shared
by initial entry logic and auto unstuck. Keep them shared for now to avoid growing the optimizer
search space. If later evidence shows auto unstuck needs independent EMA bands, add explicit
unstuck EMA spans as a separate change.

## Entry Behavior

These strategy/entry fields keep their current meaning:

- `ema_span_0`
- `ema_span_1`
- `double_down_factor`
- `initial_qty_pct`
- `initial_ema_dist`

`threshold_base_pct` replaces and merges the current entry grid spacing and entry trailing
threshold concepts.

For long entries:

1. If no position exists, initial entry behavior remains EMA-band based.
2. For reentries, threshold is measured below position price.
3. If `retracement_base_pct <= 0.0`, the strategy behaves like a recursive limit-order DCA:
   `entry_price = position_price * (1 - effective_threshold)`.
4. If `retracement_base_pct > 0.0`, the first condition is:
   `min_since_open <= position_price * (1 - effective_threshold)`.
5. The second condition is:
   `max_since_min >= min_since_open * (1 + effective_retracement)`.

Short entries mirror the same logic above position price.

Entry threshold and retracement should be multiplicative distances. Positive volatility or
wallet-exposure weights widen the distance:

```text
we_ratio = abs(wallet_exposure) / effective_wallet_exposure_limit

entry_threshold_multiplier =
    max(1.0, 1.0
        + volatility_1m * threshold_volatility_1m_weight
        + volatility_1h * threshold_volatility_1h_weight
        + we_ratio * threshold_we_weight)

entry_retracement_multiplier =
    max(1.0, 1.0
        + volatility_1m * retracement_volatility_1m_weight
        + volatility_1h * retracement_volatility_1h_weight
        + we_ratio * retracement_we_weight)

effective_entry_threshold =
    threshold_base_pct * entry_threshold_multiplier

effective_entry_retracement =
    max(0.0, retracement_base_pct) * entry_retracement_multiplier
```

For entries, canonicalize negative `threshold_base_pct` and `retracement_base_pct` to `0.0`.
Runtime logic should still treat `effective_entry_threshold <= 0.0` as "threshold condition already
satisfied" and `effective_entry_retracement <= 0.0` as "trailing disabled." Optimizer bounds should
normally be non-negative for entry base distances.

For long entries, a larger effective threshold means a lower entry price. For short entries, it
means a higher entry price.

## Close Behavior

`close.qty_pct` keeps its current recursive close-grid meaning.

For long closes:

1. If `retracement_base_pct <= 0.0`, `threshold_base_pct` works like a limit/grid markup from
   position price.
2. If `retracement_base_pct > 0.0`, the first condition is:
   `max_since_open >= position_price * (1 + effective_threshold)`.
3. The second condition is:
   `min_since_max <= max_since_open * (1 - effective_retracement)`.

Short closes mirror the same logic below position price.

Close threshold should be additive, not multiplicative, because it may intentionally cross zero.
This supports break-even or negative-markup closes when position exposure is high:

```text
effective_close_threshold =
    threshold_base_pct
    + we_ratio * threshold_we_weight
    + volatility_1m * threshold_volatility_1m_weight
    + volatility_1h * threshold_volatility_1h_weight
```

Typical close threshold setup:

- `threshold_base_pct > 0.0`
- `threshold_we_weight < 0.0`
- volatility weights `>= 0.0`

This means:

- small position: close farther from position price
- large position: close closer to break-even, or even at a negative markup
- higher volatility: shift the close threshold farther away again

Close retracement should remain multiplicative and volatility-only for now:

```text
close_retracement_multiplier =
    max(1.0, 1.0
        + volatility_1m * retracement_volatility_1m_weight
        + volatility_1h * retracement_volatility_1h_weight)

effective_close_retracement =
    retracement_base_pct * close_retracement_multiplier
```

There is no close retracement WE modifier in this draft. The position-size risk pressure should
act on close threshold/markup, not on the pullback confirmation requirement.

## Recursive Close Grid

Close orders are computed recursively, as they are today.

For a full long position:

1. Compute the first close using the current full-position `we_ratio`.
2. Size it up to `close.qty_pct` of the current full position.
3. Simulate that order as filled.
4. Recompute `we_ratio` from the reduced synthetic position.
5. Compute the next close.
6. Repeat until the recursive close grid is complete or the position is exhausted.

This preserves the current behavior where larger positions produce more close orders and deeper
orders become more eager to reduce risk.

Example intent:

```text
threshold_base_pct = 0.01
threshold_we_weight = -0.03
close.qty_pct = 0.10
```

At high exposure, early close orders may be near break-even or negative markup. As recursive fills
reduce the synthetic position, later close orders use lower `we_ratio`, so their thresholds drift
back toward `threshold_base_pct`.

Volatility is added to each recursive threshold calculation after the WE term, so high volatility
can shift even negative WE-adjusted markups upward.

### Fill/Recompute Contract

Every fill for a coin+pside resets that coin+pside's trailing extrema tracker. After a close fill,
the next close calculation starts from the reduced position and fresh trailing extrema.

Backtest should continue to model recursive fills as recompute-after-fill. This matters for close
thresholds that depend on WE ratio, because the next synthetic close may move after each simulated
fill. This is the same class of behavior as the legacy `markup_start` / `markup_end` recursive
close grid.

### Clamping And Collapsing Fillable Closes

When trailing is disabled and close threshold depends on WE ratio, a positive
`threshold_we_weight` creates a descending close ladder for long positions as exposure falls.

Example long:

```text
pprice = 100
psize = 80
qty_pct = 0.10
starting we_ratio = 0.80
threshold_base_pct = 0.01
threshold_we_weight = 0.005
retracement_base_pct = 0.0
```

Recursive close prices:

```text
10 @ 101.40
10 @ 101.35
10 @ 101.30
10 @ 101.25
10 @ 101.20
10 @ 101.15
10 @ 101.10
10 @ 101.05
```

If the lowest close fills first, live recomputation from the reduced position recreates the lower
ladder. This is expected. A fast upward move can fill more of the resting ladder before the live
bot recomputes; slower oscillation near the lowest level tends to fill repeated lower-level closes
after each recomputation.

If current market price already crosses several computed close prices, clamp those close prices to
the current touch and collapse identical clamped prices:

```text
long close price = max(order_book_bid, computed_close_price)
short close price = min(order_book_ask, computed_close_price)
```

Example with long `bid = 101.27`:

```text
computed:
10 @ 101.40
10 @ 101.35
10 @ 101.30
10 @ 101.25
10 @ 101.20
10 @ 101.15
10 @ 101.10
10 @ 101.05

ideal after clamp/collapse:
10 @ 101.40
10 @ 101.35
10 @ 101.30
50 @ 101.27
```

The Rust order pipeline already has related order collapsing behavior; this plan should preserve
that contract explicitly.

### Single-Order Shortcut

If close trailing is disabled and close threshold does not change across recursive fills, recursive
close generation collapses to one price. In that case `close.qty_pct` is effectively moot, and the
strategy should emit one full-position close.

Practical condition:

```text
close.retracement_base_pct <= 0.0
close.threshold_we_weight == 0.0
```

Volatility terms are constant across recursive close simulation, so they do not require multiple
orders by themselves. WE-dependent threshold is the recursive dependency that requires a ladder.

## Removed Concepts

This redesign removes:

- `entry_trailing_grid_ratio`
- `close_trailing_grid_ratio`
- separate entry grid spacing versus entry trailing threshold
- separate close grid markup start/end versus close trailing threshold

The replacement is:

- `entry.threshold_*`
- `entry.retracement_*`
- `close.threshold_*`
- `close.retracement_*`
- recursive close-grid sizing through `close.qty_pct`

## Implementation Notes

- Rust remains the source of truth for order behavior.
- The strategy config should live in `bot.<side>.strategy.trailing_martingale`.
- The backtester and live bot should pass explicit per-side strategy params to the Rust
  orchestrator.
- Required volatility EMA inputs should fail loudly when weights require them.
- EMA spans remain floats.
- Keep the formulas centralized in a small helper module so entry, close, and future strategies use
  the same dynamic multiplier/additive adjustment conventions.

## Test Plan

- Recursive limit-order entries when `entry.retracement_base_pct <= 0.0`.
- Trailing entries when `entry.retracement_base_pct > 0.0`.
- Entry negative base threshold/retracement values canonicalize to zero.
- Entry `threshold_base_pct <= 0.0` treats the threshold condition as already satisfied.
- Entry threshold widens with positive WE and volatility weights.
- Entry retracement widens with positive WE and volatility weights.
- Recursive limit-order closes when `close.retracement_base_pct <= 0.0`.
- Trailing closes when `close.retracement_base_pct > 0.0`.
- Close threshold can cross from positive to zero to negative as WE ratio rises.
- Close negative base threshold is allowed and covered by unit tests.
- Positive close volatility weights shift threshold upward even when WE makes it negative.
- Recursive close-grid orders recompute WE ratio after each synthetic fill.
- Close-grid clamp/collapse combines fillable duplicate close prices at current bid/ask.
- With close trailing disabled and `close.threshold_we_weight == 0.0`, emit one full-position close
  instead of redundant same-price recursive closes.
- Close retracement widens with volatility and is unaffected by WE ratio.
- Long and short behavior mirror correctly.
- Missing required volatility EMA input hard-fails.

## Open Questions

- Should `threshold_base_pct` and `retracement_base_pct` allow negative values for entries, or
  should entries clamp/validate base distances to `>= 0.0`?

  Updated recommendation: allow negative input at the config boundary but canonicalize it to
  `0.0` for entry base threshold and retracement. This preserves the existing useful semantics:
  `threshold <= 0.0` means the threshold condition is already triggered, and
  `retracement <= 0.0` means trailing is disabled. It also keeps canonical configs clean and avoids
  exposing negative entry distances as meaningful optimizer targets.

  Human comment: the meaning of a threshold less than or equal to zero is "first condition always
  triggered". I'm thinking it's acceptable to allow negative values for threshold, and simply direct
  the flow to its own subroutine if threshold <= 0.0. That's how it is currently. Negative
  retracement has a related meaning: retracement <= 0.0 means trailing disabled. However, like this,
  clamping to zero when negative is given is also fine, and changes nothing, and maybe better for
  config hygiene.

- Should close threshold have explicit min/max clamps, or should optimizer bounds be the only
  guardrail?

  Updated recommendation: do not add runtime clamps beyond finite-value validation. Negative close
  thresholds are an intended feature of this design, including configs where volatility terms shift
  an otherwise negative base threshold back to a positive effective markup. Use schema and optimizer
  bounds as the guardrails. Add explicit unit tests for negative close base thresholds and for
  volatility-weighted thresholds crossing back above zero.

  Human comment: close.threshold given as negative can be acceptable, especially if volatility
  modifications push close price higher (for long, lower for short) in an additive way. For example,
  user sets base_threshold at -0.001, but uses high volatility weights, so that effective markup is,
  say, 0.002 for low volatility like BTC, and 0.008 for high volatility like HYPE. So I agree that
  close.threshold is clamped by optimize bounds, not hard coded to always be positive. Probably,
  unit tests around negative values would be useful.

- Should close threshold volatility terms be applied before or after WE adjustment? This draft uses
  additive commutative terms, so order does not matter numerically.

  Updated recommendation: keep the additive formula. Treat WE pressure and volatility pressure as
  independent basis-point adjustments to threshold. This makes the formula order-independent,
  preserves negative-threshold behavior, and is easier to reason about than multiplying a threshold
  that may cross zero.

  Human comment: Agreed.

- Should `entry.ema_span_*` stay under `entry`, or should EMA band config move to a shared
  strategy-level namespace?

  Updated recommendation: move `ema_span_0` and `ema_span_1` to the `trailing_martingale` strategy
  root. They are not only entry params because auto unstuck uses the same lower/upper EMA bands.
  Do not give auto unstuck separate EMA spans in this pass; that adds flexibility but also expands
  the search space. Revisit separate unstuck EMA spans only if optimization results or live behavior
  show the shared bands are a real constraint.

  Human comment: Might be better to move them out of trailing.entry, because auto unstuck uses the
  same lower/upper EMA bands computed from ema_span_0 and ema_span_1. Another alternative is to
  decouple them, given auto unstuck its own EMA spans. Tradeoff: more flexibility for bigger search
  space. If we retain shared EMA spans for initial entry and auto unstuck, then they should be moved
  out of trailing.entry.
