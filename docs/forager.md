# Forager Mode

Forager ranks eligible symbols to fill available initial-entry slots. Rust owns the shared
selection logic used by live trading, backtesting and optimization.

## Configuration And Selection

Shared defaults belong under `bot.global.forager`; `bot.long.forager` and
`bot.short.forager` override them independently. Relevant settings are:

- `volume_ema_span_minutes`
- `log_range_ema_span_minutes`
- `unilateralness_ema_span_minutes` (a floating-point span, default `60.0`)
- `score_weights.volume`, `score_weights.ema_readiness`, `score_weights.log_range`
  and `score_weights.unilateralness`
- `score_hysteresis_margin`

The universe's approved/ignored lists, side-specific minimum coin age and effective-cost
policy determine eligibility. Directional activation and available position slots also apply.
Forager normalizes the required features, computes the weighted ranking and fills available
slots. Volume is a ranking feature; there is no authored low-volume pruning control.

The active strategy supplies its entry threshold and price EMA bands. Trailing Martingale
price spans are `strategy.trailing_martingale.entry.ema_span_0_minutes` and
`entry.ema_span_1_minutes`. EMA Anchor uses `strategy.ema_anchor.ema_span_0_minutes`
and `ema_span_1_minutes`. The geometric-mean span is derived without rounding.

## Ranking Features

### Quote Volume

`forager.volume_ema_span_minutes` controls the EMA of one-minute quote volume.
Higher values rank better. This raw quantity is distinct from its normalized score component.

### Log Range

`forager.log_range_ema_span_minutes` controls the EMA of `ln(high / low)` from
one-minute candles. Higher values rank better. This logarithmic high/low range does not
estimate the standard deviation of returns and is not itself an entry trigger.

### EMA Entry Readiness

The raw `ema_entry_price_gap_fraction` measures market price relative to the strategy's
initial-entry threshold:

- Long: `bid / entry_threshold - 1`
- Short: `1 - ask / entry_threshold`

Lower values rank better. Negative means price has passed the threshold, zero means it
is at the threshold, and positive means it remains away from entry. Rust derives the threshold
from the active strategy; the gap is a signed scale-one fraction. The normalized
`ema_readiness_component` reverses that preference so a higher component ranks better.

### Unilateralness

`forager.score_weights.unilateralness` optionally penalizes sustained one-way movement
in either direction. It uses the absolute signed RMS score from completed one-minute close
returns. Lower values rank better; normalization reverses this penalty for the combined score.
The weight defaults to zero. See [adaptive entry cooldown and RMS](adaptive_entry_cooldown.md)
for the formula, normalization, completed-candle readiness and decay through flat prices.

## Weights And Hysteresis

`forager.score_weights` contains four finite, nonnegative weights. Their relative proportions
matter: positive weights are normalized to unit sum before scoring. If all four are zero,
canonical loading selects EMA-readiness-only ranking.

| Weight choice | Preference |
| --- | --- |
| `log_range: 1.0`, others zero | Larger candle log ranges |
| `volume: 1.0`, others zero | Higher quote volume |
| `ema_readiness: 1.0`, others zero | Prices closer to their initial-entry threshold |
| `unilateralness: 1.0`, others zero | Less sustained one-way movement |
| `volume: 0.2`, `ema_readiness: 0.6`, `log_range: 0.2`, `unilateralness: 0.0` | Entry readiness with liquidity and range contributions |

Raw weights are distinct from normalized feature components and the combined score.
`score_hysteresis_margin` is an absolute margin between normalized candidate scores.
For example, `0.02` is a score margin, rather than a price percentage or relative score change.
Hysteresis can retain an incumbent when the challenger does not exceed that margin.

These weights rank candidates for available initial-entry slots. They do not force entries,
bypass strategy conditions or bypass risk and minimum-size gates.

## Optimization Bounds

Authored bounds use `optimize.bounds.<side>.forager.score_weights.<feature>` and
`optimize.bounds.<side>.forager.unilateralness_ema_span_minutes`.

Missing unilateralness weight and span bounds are hydrated as
`[configured_value, configured_value]` using each side's normalized weight and floating-point
span. New configs default to `[0.0, 0.0]` for the weight and `[60.0, 60.0]` for the span.
Templates omit these bounds until loading derives them from bot values. Widen them explicitly
to search unilateralness; explicit ranges are preserved.

## Live Candle Readiness

Broad candidate refreshes are best-effort and budgeted. The live bot refreshes the stalest
eligible candidates first and spreads remote candle calls over time. Candidates within
`live.candles.max_forager_candle_staleness_minutes` remain rankable even if their latest
completed candle is stale. Selection must not favor whichever symbols refreshed first.

- When the cap is `null`, the budget-derived window is never shorter than
  `live.candles.max_active_candle_tail_gap_minutes` (10 minutes by default). An explicit
  positive cap may shorten that window.
- Close EMA readiness may use bounded flat-close/no-trade projection for the stale tail.
- Quote-volume and log-range inputs carry their latest known EMA values without appending
  synthetic zero-volume or zero-range tail candles.
- Synthetic zero volume/log range is valid for verified no-trade continuity gaps. Unknown
  stale tails caused by REST delay or refresh budgeting do not establish such gaps.
- A candidate with no prior feature basis, nonfinite carried values or feature age beyond
  policy is unavailable for new forager entries until refreshed.

Unavailable candidates remain visible in diagnostics. Missing volume/log range must not be
silently replaced with neutral values. Materially narrowed ranking universes should be visible,
and new entries may be deferred instead of filling slots from whichever subset refreshed first.

Before placing or retaining an initial-entry order, live execution also requires fresh
account-critical surfaces, a sufficiently fresh market snapshot and the finite strategy inputs
that the action consumes. Bounded reuse of prior indicators follows the explicit readiness
contract; Forager does not invent substitute observations.

## Diagnostics And Tuning

The monitor's ranking total comes from the latest Rust scoring diagnostic, including all
four components and tie ordering. Its source timestamp identifies the observation. Without
a current scoring diagnostic, the monitor does not fetch RMS data or reconstruct a total.

Raw native fields `volume_ema`, `log_range_ema`, `ema_entry_price_gap_fraction` and
`unilateralness_score` are distinct from their normalized score components.

Live currently supplies its latest market price as both bid and ask in the Python-collected
payload; Rust's readiness calculation remains position-side-aware. Features whose weights
are zero may be skipped when no other selection requirement consumes them.

Start with log-range weighting to favor active symbols, add EMA readiness to favor nearer
entries, and add volume weight when liquidity matters. If rankings look noisy, first consider
longer Forager EMA spans before changing strategy entry parameters.
