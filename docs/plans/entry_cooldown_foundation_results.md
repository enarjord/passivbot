# Entry cooldown foundation and formula experiments

## Delivered foundation

Public config now uses `bot.<side>.entry_cooldown.base_duration_minutes` with the
same numeric defaults (long 24.1, short 0.0). Config schema v8.5.0 migrates the old
`risk.entry_cooldown_minutes` leaf, optimizer bounds, and per-coin overrides.
Legacy CLI flags and leaf/group selectors continue to resolve. Independently
authored override files and inline patches migrate before merging; ordinary
inline precedence is retained. Conflicting old/new grouped leaves in one source
warn and retain the explicit new leaf.

Rust's `entry_cooldown.rs` owns duration evaluation, millisecond rounding,
elapsed-time enforcement, and the cooldown part of ladder staging. Orchestration
still owns filtering orders. The internal `risk_entry_cooldown_minutes` key and
optimizer gene names remain stable, including the GPU adapters.

This stage adds no adaptive behavior, score weights, duration bounds, or new
market-data requirements. It is independently usable before choosing modifiers.

## Additive versus multiplicative

Compare these definitions, using identical scores and one final clamp:

```text
additive       = clamp(base + sum(score_i * minutes_i), floor, ceiling)
multiplicative = clamp(base * (1 + sum(score_i * weight_i)), floor, ceiling)
```

For every fixed positive base, setting `minutes_i = base * weight_i` makes them
equivalent. A seeded 10,000-case comparison, including negative weights and floor/
ceiling saturation, had maximum error below 6e-14 minutes. This is floating-point
roundoff, not evidence of a trading advantage for either representation.

At base zero, additive weights can still introduce a cooldown; multiplicative
weights cannot. With base 5, adverse score 0.75, and additive weight 20 minutes,
the result is 20 minutes. Multiplicative weight 4 gives the same result. With base
zero, those settings yield 15 and zero minutes respectively.

**Recommendation:** additive weights expressed in minutes, with one weight per
score and zero disabling that input. This avoids a mode switch and lets base zero
mean "wait only when a modifier warrants it." Independently exposing both kinds
of weights would be redundant at fixed base and make optimization less identifiable.
Multiplicative scaling is useful if changing the base should proportionally scale
all modifiers; that is a configuration convenience, not a distinct policy family
under the formulas above. A product of individual factors is a different model
with interactions; it is outside this recommendation.

Proposed next-stage shape (not implemented or defaulted in this foundation):

```json
{
  "entry_cooldown": {
    "base_duration_minutes": 5.0,
    "min_duration_minutes": 0.0,
    "max_duration_minutes": 60.0,
    "weights_minutes": {
      "exposure_ratio": 0.0,
      "adverse_directionality": 20.0
    }
  }
}
```

The configured base is also the quiet-market baseline: score zero returns base,
not an extra reduction below base. Prefer nonnegative initial weights and scores;
keep exposure ratio's behavior above 1 explicit rather than silently clipping it.
Clamp once after combining contributions. All disabled weights must avoid reading
their inputs. To preserve existing arbitrary fixed durations, a missing maximum
must not clip the base. A finite maximum should be required when modifiers are
enabled, so live fill-history coverage and restart reconstruction have a bounded
horizon. Validate `0 <= min <= max` when max is present.

Before enabling adaptive durations, update live history lookback, pair activation,
fill anchors, per-coin overrides, and restart coverage together. A zero base must
not suppress fill anchors when a modifier can produce a positive duration. Define
whether the effective duration is recomputed from current inputs (recommended for
restart reproducibility) and separately define ladder staging when it reaches zero.

## Unilateralness formula comparison

Let `r = log(close_t / close_(t-1))`, using completed one-minute bars. Preserve a
floating-point span `s` and use `alpha = 2 / (s + 1)` for both accumulator EMAs.
`U` is the penalty: 0 is no directionality and 1 is maximal directionality.
`1 - U` is the corresponding higher-is-better score contribution. It is not a
penalty increasing with unilateralness.

| Formula | What it measures | Main tradeoff |
|---|---|---|
| `abs(EMA(r)) / EMA(abs(r))` | Directional purity of the remaining exponentially weighted movement | Scores uneven one-way motion and a single shock as fully unilateral; flat tails preserve the ratio |
| `abs(EMA(r)) / sqrt(EMA(r*r))` | Consistent directional movement relative to RMS returns | Decays in flat tails, but downweights isolated shocks and uneven step sizes |
| `abs(EMA(sign(r)))` | Imbalance in up/down bar counts, with flat bars contributing zero | Ignores move sizes; many small rises can outweigh a large fall |
| `abs(log(EMA_fast(close) / EMA_slow(close)))` | Separation between two price horizons | Unbounded, magnitude-dependent, two horizons; arithmetic price EMAs are not exactly reciprocal-symmetric |
| `abs(sum(r)) / sum(abs(r))` over a window | Finite-window directional purity | Abrupt removal of old moves and rolling-window state |

Synthetic results at span 60, zero-seeded return accumulators following flat prices:

| Path at evaluation | EW efficiency | EW RMS | EW sign | Window efficiency |
|---|---:|---:|---:|---:|
| Flat throughout | 0 | 0 | 0 | 0 |
| Sustained equal up steps (or down steps) | 1.000 | 1.000 | 1.000 | 1.000 |
| Alternating equal up/down steps | 0.0167 | 0.0167 | 0.0167 | 0 |
| Alternating small/large steps, all up | 1.000 | 0.748 | 1.000 | 1.000 |
| One shock now | 1.000 | 0.181 | 0.0328 | 1.000 |
| Shock, then 60 flat bars | 1.000 | 0.0666 | 0.00444 | 0 |
| Shock, then 300 flat bars | 1.000 | 0.00122 | approximately 0 | 0 |

**Recommendation:** EW efficiency best matches directional purity and treats a
large one-way shock as unilateral. EW RMS is preferable if fading the score through
flat periods is part of the intended definition. Neither is universally better;
the flat-tail behavior needs an explicit choice. Do not conceal that choice behind
an arbitrary denominator epsilon. Exact zero movement can return U=0; unavailable
data must remain unavailable. Both return-normalized candidates are invariant to
scaling all log returns: neither measures absolute adverse-move magnitude. Add a
separate volatility/magnitude input if that is the desired cooldown behavior.

An EMA span controls decay, not a hard lookback. Initialization, warmup, handling of
gaps, and restart replay must be defined consistently before production use.
Direction reversal cancels recent signed returns and can briefly look non-unilateral
even while volatility is high; these metrics do not replace volatility controls.

The existing forager normalizes components across candidates and gives higher
combined scores preference. A new nonnegative `score_weights.unilateralness` leaf
should have default zero, use a lower-is-better U component, and reuse that weighting
pattern. Decide explicitly whether to keep cross-candidate normalization or use
bounded `1-U` directly; the former preserves the current component convention but
can amplify tiny differences between otherwise similar candidates.

## CPU cost and reproducibility

Run `python src/tools/research_entry_timing.py --span 60`. The script contains
seeded synthetic cases, composition checks, and an isolated two-EMA arithmetic
benchmark. One run took about 0.057 seconds for one million Python updates. This
is an arithmetic microbenchmark, not an end-to-end bot benchmark or a Rust timing.
EW efficiency needs two accumulator values per symbol/span and constant work per
bar. Data preparation, readiness, replay, and integration dominate the design cost.

Regression coverage checks score bounds, long/short symmetry, return scaling,
fractional spans, flat-tail behavior, composition equivalence, and zero base.
Refactor validation also compared full fill/equity/analysis outputs before and
after against base commit `d214ea370`: three strategies, cooldowns 0, 0.05, 5, and
24.1 minutes, two synthetic coins, and 720 one-minute bars. All 12 outputs matched
exactly. Canonical default values, flattened optimizer bounds, and runtime bot
parameters also matched after accounting for the public grouping/schema version.
These experiments establish semantics and refactor parity, not profitability.

Trading evaluation should follow with fixed configurations across disjoint periods,
coins, and regimes, comparing exposure, drawdown, underwater time, realized losses,
turnover/fees, missed recoveries, and returns. Compare one modifier at a time before
combinations; rescale weights before comparing additive and multiplicative policies.

## Partial fills: separate research scope

Current initial-entry top-up logic in `entries.rs` recognizes positions below 80%
of the calculated initial quantity. It is a position-size heuristic, not general
logical-order completion tracking. Cooldown enforcement still applies to
position-increasing orders, including partial-entry top-ups.

Live cooldown anchors use the latest position-increasing fill per symbol/side,
with a position-delta guard for delayed fill observations. They do not distinguish
fragments of one logical re-entry from separate re-entries. An active cooldown
removes all adds from Rust's current intent, potentially including the remainder
of a partially filled add during the next reconciliation. CPU candle backtests
process an eligible entry at its modeled order quantity; that does not establish
realistic exchange-fragment behavior.

A follow-up should reconstruct order identity, original quantity, incremental vs
cumulative fills, remaining quantity, cancellation/replacement, restart history,
and exchange minimums. Then compare starting cooldown at first fill, latest fill,
or completion. Allowing a remainder during cooldown needs an explicit bounded
completion rule that cannot grow into an unrestricted new DCA. Do not bypass the
current gate based solely on a small fill or current position size.
