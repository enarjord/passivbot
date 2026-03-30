# Forager Weighted Ranking Plan

This document is a handoff spec for redesigning forager coin selection for initial-entry slot filling.

## Goal

Replace the current mostly sequential volume-then-volatility forager heuristic with a cleaner two-stage model:

1. hard eligibility plus coarse pruning
2. weighted multi-factor ranking

The redesign should:

1. reduce dead slot occupancy from coins that are far from their initial-entry EMA gate
2. preserve liquidity sanity checks
3. remain extensible for future ranking factors
4. use as few parameters as practical
5. stay easy to reason about, test, and optimize

## Non-Negotiable Principles

### Config correctness is centralized

All config shape hydration and migration must happen in `format_config()`.

Downstream consumers should assume required params exist and are valid.

Do not add silent fallback patterns like:

1. `dict.get("required_key", default)`
2. broad `try/except` around ranking config
3. missing-weight compensation in Rust/Python ranking code

If a required ranking field is missing after `format_config()`, fail loudly.

This is especially important for this task because the ranking logic will become more configurable and silent defaults would make debugging difficult.

### Rust is source of truth

The forager selection logic belongs in Rust:

1. `passivbot-rust/src/orchestrator.rs`
2. `passivbot-rust/src/coin_selection.rs`

Python should only handle:

1. config/template/formatting
2. payload plumbing
3. tests/docs/changelog

### Terminology

Use `forager` consistently for this subsystem.

User-facing config should prefer:

1. `forager_*`

Avoid mixing `filter_*` names for coin-selection behavior that is specifically about initial-entry slot selection.

## Existing High-Level Behavior

Current rough behavior:

1. build eligible coin set
2. sort by relative volume and drop the low-volume tail
3. sort the survivors by volatility
4. fill remaining initial-entry slots from that list

Problem:

Initial entries are EMA-gated:

1. `initial_entry_price_long = lower_ema_band * (1 - entry_initial_ema_dist)`
2. `initial_entry_price_short = upper_ema_band * (1 + entry_initial_ema_dist)`

So a forager slot can be occupied by a coin that is attractive on volume/volatility but far from actual initial-entry readiness, leaving the slot idle for a long time.

## Proposed New Model

### Stage 1: Hard Eligibility

A coin is eligible only if all hard requirements pass.

Examples include:

1. tradable
2. not delisted
3. not trading halted
4. in approved coins
5. not in ignored coins
6. coin age above minimum
7. effective min cost low enough
8. one-way mode allows this `pside`
9. mode allows initial entry on this `pside`
10. any existing current hard eligibility gate already used by the orchestrator

This stage is binary:

1. eligible
2. ineligible

Only eligible coins proceed.

### Stage 2: Coarse Volume Pruning

From the eligible set, apply a privileged liquidity sanity filter:

1. sort eligible coins by relative volume descending
2. drop the lowest `forager_volume_drop_pct`
3. but do not drop so aggressively that `n_remaining_candidates + n_already_filled_slots < n_positions`

Equivalent practical rule:

1. always retain enough candidate coins to still make it possible to fill the remaining open slots

This stage is a coarse gate, not the final ranking.

### Stage 3: Weighted Multi-Factor Ranking

For survivors of stage 2:

1. compute raw scores for each enabled ranking factor
2. normalize each factor score to `[0.0, 1.0]`
3. compute weighted final forager score
4. rank descending by final score
5. select top candidates needed to fill the remaining open slots

Initial ranking factors:

1. `volume`
2. `ema_readiness`
3. `volatility`

The model must be extensible to future factors later.

## Config Shape

Per `pside`:

```json
"bot": {
  "long": {
    "forager_volume_drop_pct": 0.2,
    "forager_score_weights": {
      "volume": 0.4,
      "ema_readiness": 0.4,
      "volatility": 0.2
    }
  }
}
```

Same under `bot.short`.

### Parameter semantics

`forager_volume_drop_pct`

1. coarse pre-ranking low-volume pruning fraction
2. range `[0.0, 1.0]`
3. still constrained by the “retain enough candidates to fill slots” rule

`forager_score_weights`

1. required dict
2. required keys in v1:
   - `volume`
   - `ema_readiness`
   - `volatility`
3. missing keys should be treated as config/schema errors, not silently defaulted
4. all-zero weights should be treated as invalid config and fail loudly

## Ranking Factors

### 1. Volume

Purpose:

1. prefer more liquid coins among survivors

Raw input:

1. the current relative-volume metric already used by the forager
2. currently sourced from EMA of `m1.volume`

Ordering:

1. higher is better

Normalization:

1. continuous normalization over the current survivor set to `[0.0, 1.0]`
2. do not use crude ordinal/Borda ranks
3. if all values are equal, assign all coins the same normalized score

Volume remains both:

1. a privileged pre-ranking filter
2. a ranking factor in the final weighted score

### 2. EMA Readiness

Purpose:

1. prefer coins that are already at or close to actual initial-entry eligibility
2. reduce slots sitting idle on far-away coins

This is `pside`-specific.

#### Long

Compute:

1. `initial_entry_price_long = lower_ema_band * (1 - entry_initial_ema_dist)`
2. `ema_dist_long = market_price / initial_entry_price_long - 1`

Interpretation:

1. `ema_dist_long <= 0.0` means immediate long initial-entry readiness or already crossed
2. small positive values mean near-ready
3. large positive values mean far from entry readiness

#### Short

Compute:

1. `initial_entry_price_short = upper_ema_band * (1 + entry_initial_ema_dist)`
2. `ema_dist_short = 1 - market_price / initial_entry_price_short`

Interpretation:

1. `ema_dist_short <= 0.0` means immediate short initial-entry readiness or already crossed
2. small positive values mean near-ready
3. large positive values mean far from entry readiness

#### Ordering and normalization

1. smaller distance is better
2. crossed/ready coins should score best
3. normalize continuously across the survivor set to `[0.0, 1.0]`
4. do not reduce this to coarse ordinal ranks only

Important:

1. EMA readiness must use distance to the actual offset entry threshold, not merely distance to raw EMA bands

### 3. Volatility

Purpose:

1. prefer coins with stronger recent movement opportunity

Raw input:

1. the existing volatility metric already used by the forager
2. currently derived from EMA of `ln(high / low)` or equivalent `m1.log_range`

Ordering:

1. higher is better

Normalization:

1. continuous normalization over the survivor set to `[0.0, 1.0]`
2. if all equal, assign same score to all coins

Volatility should remain in v1.

Do not drop volatility entirely in favor of EMA readiness. They are correlated sometimes, but not equivalent.

## Final Score

For each surviving coin:

1. compute normalized score for each factor
2. compute weighted final score:

`final_score = w_volume * s_volume + w_ema_readiness * s_ema_readiness + w_volatility * s_volatility`

Weights do not need to sum to 1.0 in config; implementation may normalize internally if desired.

Then:

1. sort by `final_score` descending
2. break ties deterministically by symbol index / stable input order
3. select the top coins needed to fill remaining open initial-entry slots

## Why Continuous Normalization Instead of Borda Count

Do not use pure ordinal Borda aggregation.

Reason:

1. two nearly equal raw scores should remain nearly equal after normalization
2. a whole-rank jump for tiny raw-score changes is arbitrary
3. retaining proportionality makes weights more meaningful and rankings smoother

Example motivating this:

1. raw values could produce normalized scores like `[0.01, 0.44, 0.441, 0.99]`
2. treating `0.441` as a whole discrete rank above `0.44` loses useful information

## Migration / Backward Compatibility

This branch does not need compatibility with intermediate dev-branch schemas, but it must maintain compatibility with `master`.

So:

1. directly rename user-facing config from `filter_volume_drop_pct` to `forager_volume_drop_pct`
2. add a `format_config()` migration from legacy:
   - `bot.{pside}.filter_volume_drop_pct -> bot.{pside}.forager_volume_drop_pct`
3. add missing `forager_score_weights` in `get_template_config()` and `configs/template.json`
4. downstream code should only use the new `forager_*` names

## Recommended Defaults

Defaults should preserve current behavior as closely as practical.

Suggested initial defaults:

1. `forager_volume_drop_pct = legacy filter_volume_drop_pct`
2. `forager_score_weights = {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}`

This gives compatibility-like behavior by default while enabling users to opt into readiness-aware ranking.

Alternative future defaults can be revisited after experimentation, but v1 should be stable and unsurprising.

## Suggested Rust Shape

Likely touched files:

1. `passivbot-rust/src/orchestrator.rs`
2. `passivbot-rust/src/coin_selection.rs`
3. `passivbot-rust/src/types.rs` if new serde-exposed bot params are needed

Suggested structural changes:

1. expand `CoinFeature` to carry raw factor values needed for final scoring
2. keep eligibility construction in `build_forager_features_into(...)`
3. keep coarse volume pruning as an explicit stage
4. add normalized scoring helpers in `coin_selection.rs`
5. compute final weighted score there
6. preserve deterministic ordering and tie-breaking

Possible helper functions:

1. `compute_ema_readiness_raw(...)`
2. `normalize_feature_scores(...)`
3. `score_forager_candidates(...)`
4. `prune_low_volume_tail(...)`

## Testing Requirements

Add targeted Rust/Python regression tests for:

1. legacy config migration:
   - `filter_volume_drop_pct -> forager_volume_drop_pct`
2. missing new config is added by `format_config()`
3. all-zero weights fail loudly
4. EMA readiness ranking:
   - ready/crossed coin preferred over far-away coin when other factors equal
5. volume pruning still never removes so many coins that remaining open slots cannot be filled
6. weighted ranking behaves deterministically on ties
7. legacy-like defaults preserve current selection behavior
8. one-way and mode-based ineligibility still work exactly as before

## Documentation / Changelog

Update:

1. `CHANGELOG.md`
2. user-facing forager/backtest/optimize docs if applicable

Document:

1. new `forager_*` parameter names
2. the three factor weights
3. that volume is both a pre-ranking prune gate and a final ranking factor
4. that EMA readiness uses the actual offset initial-entry threshold

## Implementation Recommendation

Proceed in this order:

1. config/template/format migration
2. Rust factor and scoring model
3. targeted tests
4. user docs/changelog

Avoid:

1. arbitrary user-defined ranking pipelines in v1
2. silent config fallbacks
3. Python-side reimplementation of the selection logic

The correct v1 is:

1. hard eligibility
2. coarse volume pruning
3. weighted continuous normalized multi-factor ranking in Rust
