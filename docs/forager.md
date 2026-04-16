# Forager Mode

This document explains Passivbot's dynamic coin-selection system, referred to in code and config as `forager`.

## Motivation

Forager mode exists to answer one question:

Which symbols should be allowed to occupy the limited entry slots right now?

Forager answers that question in a simple sequence:

1. prune obviously weak candidates by volume
2. rank the survivors by a weighted score
3. fill the available shortlist slots from that ranking

This lets you control the tradeoff between liquidity, volatility, and entry readiness directly in config.

## Inputs

For each side (`bot.long` and `bot.short`), forager uses:

- `forager_volume_drop_pct`
- `forager_volume_ema_span`
- `forager_volatility_ema_span`
- `forager_score_weights`
- `ema_span_0`
- `ema_span_1`
- `entry_initial_ema_dist`

It also depends on live/backtest market state:

- latest market price
- EMA bands derived from `ema_span_0`, `ema_span_1`, and `sqrt(span_0 * span_1)`
- 1m EMA quote volume
- 1m EMA log range
- tradability / enabled flags / slot availability

## Selection Flow

At a high level, forager selection is:

1. Build candidate inputs for all approved symbols that are old enough.
2. Disable candidates that fail trading constraints such as effective min cost.
3. Apply coarse low-volume pruning using `forager_volume_drop_pct`.
4. Normalize the remaining feature values.
5. Compute a weighted score using `forager_score_weights`.
6. Return the best candidates needed to fill the available shortlist slots.

The canonical shortlist logic now lives in Rust and is shared by live selection and orchestrator/backtest selection.

## Feature Definitions

### Volume

`forager_volume_ema_span` controls the 1m EMA of quote volume used in ranking and pruning.

- Higher is better.
- If `forager_volume_drop_pct > 0`, volume is required even if the volume score weight is zero.

### Volatility

`forager_volatility_ema_span` controls the 1m EMA of log range:

`log_range = ln(high / low)`

- Higher is better.
- This is a ranking input, not an entry trigger by itself.

### EMA Readiness

EMA readiness measures distance to the actual initial-entry threshold, not just distance to raw EMA bands.

For long:

`readiness = bid / (ema_lower * (1 - entry_initial_ema_dist)) - 1`

For short:

`readiness = 1 - ask / (ema_upper * (1 + entry_initial_ema_dist))`

Lower is better.

Interpretation:

- negative: already beyond the initial-entry threshold
- near zero: close to triggering
- larger positive: farther away from entry

## `forager_score_weights`

`forager_score_weights` is a dict with required keys:

- `volume`
- `ema_readiness`
- `volatility`

Rules:

- each value must be finite and non-negative
- positive weights are relative; only their proportions matter
- positive weights are normalized to unit sum before scoring
- if all three weights are zero, Passivbot normalizes them to EMA-readiness-only ranking

Examples:

- `{"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}`
  - prioritize the most volatile candidates
- `{"volume": 1.0, "ema_readiness": 0.0, "volatility": 0.0}`
  - maximize liquidity after coarse pruning
- `{"volume": 0.0, "ema_readiness": 1.0, "volatility": 0.0}`
  - prioritize symbols closest to a real initial entry
- `{"volume": 0.2, "ema_readiness": 0.6, "volatility": 0.2}`
  - bias toward actual setups while still preferring liquid, active symbols
- `{"volume": 0.0, "ema_readiness": 0.0, "volatility": 0.0}`
  - normalize to EMA-readiness-only ranking

These sub-weights are also available under `optimize.bounds` as:

- `long_forager_score_weights_volume`
- `long_forager_score_weights_ema_readiness`
- `long_forager_score_weights_volatility`
- `short_forager_score_weights_volume`
- `short_forager_score_weights_ema_readiness`
- `short_forager_score_weights_volatility`

## Failure Policy

Forager inputs are trading-critical.

That means:

- missing required forager data should raise
- non-finite required forager data should raise
- fetch-budget exhaustion should raise if it prevents building required shortlist inputs
- neutral defaults like `0.0`, `(0.0, 0.0)`, or `inf` are not acceptable for required shortlist inputs

The only fallback philosophy accepted in Passivbot for critical indicator paths is the explicitly reviewed and bounded EMA fallback used in approved EMA paths. Forager should not invent local substitute values.

## Caveats

- Live currently uses the latest available market price as both bid and ask in the Python-collected payload. The shortlist math itself is side-aware in Rust, so moving to real order-book bid/ask later does not require changing the ranking definition.
- If a feature weight is zero and no other part of the selection requires that feature, the feature may be skipped entirely.

## Practical Tuning Notes

- Start with volatility-only if you want the shortlist to favor active, moving symbols.
- Add `ema_readiness` when you want empty slots to favor symbols closer to producing a real initial entry.
- Keep some volume pruning or volume weight if your candidate universe contains many illiquid tails.
- If rankings look noisy, first widen the forager EMA spans before changing entry parameters.

## Implementation Notes

- Canonical shortlist scoring and selection live in Rust.
- Python is responsible for gathering market data, building payloads, and calling Rust.
- The canonical config names are `forager_volume_ema_span` and `forager_volatility_ema_span`.
