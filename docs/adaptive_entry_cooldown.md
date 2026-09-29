# RMS unilateralness and adaptive entry cooldown

Both features are opt-in. New weights default to zero; existing cooldown durations,
Forager weights, CLI aliases and optimizer bounds retain their defaults. CPU backtests,
CPU optimization and live planning share the Rust calculations. GPU optimization rejects
enabled modifiers, duration bounds, or unilateralness scoring rather than ignoring them.

## One indicator, two consumers

For completed one-minute closes, let `r = log(close_t) - log(close_previous)` and
`a = 2 / (span + 1)`. Update two zero-seeded accumulators:

```
mean   += a * (r - mean)
square += a * (r*r - square)
signed_unilateralness = mean / sqrt(square)
U = abs(signed_unilateralness)
```

An entirely flat window has U = 0. U ranges from 0 to 1 and measures sustained,
consistent one-way movement. Upward and downward trends receive the same Forager
penalty. Small two-way moves cancel; isolated shocks receive less penalty than
consistent trends. This is directionality, not absolute price-move magnitude.

`bot.<side>.forager.unilateralness_ema_span_1m` is the shared floating-point span,
default 60. Both consumers on that side use it. The supported range is 1–100000.
For deterministic restart reconstruction, both runtimes use exactly
`ceil(20 * span)` returns, requiring one more completed close, with exponential
weights throughout. Live reconstructs the window; CPU backtests maintain its
weighted moments incrementally. Their scores agree within floating-point roundoff. Older information is discarded only after twenty spans;
this bounded initialization is separate from a simple moving average. A flat
tail makes RMS decay approximately by `(1-a)^(minutes/2)` until the prior movement
leaves the replay history. At span 60 its flat-tail half-life is about 42 minutes.

`bot.<side>.forager.score_weights.unilateralness` defaults to zero. The existing
cross-candidate lower-is-better normalization converts U into a higher-is-better
component, then combines it with the other relative weights. An absolute `1-U`
score is not substituted for that existing normalization. With insufficient
competition to require ranking, this input is not required for selection. CPU backtests
request its history without postponing unrelated trading. An incomplete scoring window
defers the entire ranking decision while a compared candidate lacks its required score;
ready candidates are not ranked as a smaller substitute universe.

Live ranking may carry the latest complete, contiguous RMS observation within the
same per-symbol age allowance used by other Forager ranking metrics. Debug logs
identify the source, span and age. Cooldown requires the current completed window.
Unknown gaps and stale tails are never turned into zero returns to make RMS fade.
An unavailable enabled input defers the affected entry/ranking consumer while
independent closes and protective actions continue. Invalid prices are errors.

## Additive minutes

Under `bot.<side>.entry_cooldown`:

```json
{
  "base_duration_minutes": 5.0,
  "min_duration_minutes": 0.0,
  "max_duration_minutes": 60.0,
  "weights_minutes": {
    "exposure_ratio": 10.0,
    "adverse_directionality": 20.0
  }
}
```

These are illustrative settings, not tuned defaults. Duration is:

```
clamp(base + exposure_ratio * exposure_weight
           + adverse_score * adverse_weight, min, max)
```

Exposure uses the existing position wallet exposure divided by its effective wallet
exposure limit; it is not capped at one. A flat position contributes zero. If a
held position has a zero effective limit, the exposure modifier saturates the
configured ceiling. Long adverse score is
`max(0, -signed_unilateralness)`; short adverse score is
`max(0, signed_unilateralness)`. Favorable movement adds no directionality delay.
With base 5, exposure ratio 0.8, and adverse score 0.75, the example gives
`5 + 0.8*10 + 0.75*20 = 28` minutes. A zero base and adverse weight 20 alone
would give 15 minutes. At fixed positive base, additive minutes can express the
same policies as `base * (1 + score * weight)` after rescaling the weight.

All values and weights are finite and nonnegative. The default floor is zero;
the default ceiling is null (unbounded fixed duration). Enabling either modifier
requires a finite ceiling, with `floor <= ceiling`, to bound fill-history coverage.
Each cooldown leaf can be overridden per coin and side. A partial weight override
inherits the other weight. Zero weights do not require their input.

The effective duration is recalculated at each decision and compared with elapsed
time since the latest position-increasing fill. It is not a newly started timer
at each decision, or a deadline latched at the fill. Improving conditions can
shorten the wait. Fill history and restart anchors cover the maximum possible
configured duration, including when base is zero. Quiet conditions return toward
the base plus the exposure contribution, subject to the bounds.

Positive effective duration stages at most one add. At zero effective duration,
the existing strategy's simultaneous-ladder/retracement rules apply. Partial
fills keep the current latest-increasing-fill semantics; this feature does not
introduce logical-order completion tracking or exempt re-entry fragments.

## Optimization and evaluation

New dimensions are opt-in: add bounds under the matching nested config paths,
for example `optimize.bounds.long.entry_cooldown.weights_minutes.exposure_ratio`
or `optimize.bounds.long.forager.score_weights.unilateralness`. The existing
Forager bound shorthand also supports `score_weights_unilateralness`. Keep a
finite ceiling in the base config, or supply numeric ceiling bounds, when a searched
modifier can become positive. A ceiling search can start from the default null ceiling;
an unbounded starting-config ceiling
projects to the upper search bound. History loading covers positive consumer weights and
maximum float spans reachable through optimizer bounds, even when fixed weights are zero.
This history budget does not impose a delay on candidates that do not consume RMS.
Ordinary backtest, optimizer and suite metadata stamp only the shared non-RMS activation budget;
each candidate retains its own Rust RMS entry/ranking readiness.
Search ranges for floor and ceiling must satisfy `highest floor <= lowest ceiling`,
including a fixed bound when only the other is optimized and every effective coin override.
Coin-pinned cooldown leaves supersede their global search dimensions, including zero
modifier pins with a null ceiling. Nested adaptive bounds may be mixed with legacy flat
bounds. The CLI accepts `null` to clear a ceiling; omitting the option keeps its current value.
Dormant RMS weights on a statically disabled side do not require history or one-minute
candles. After selecting a dataset, optimizer preflight checks reachable RMS consumers
against finalized per-coin eligibility and overrides. Entry-ineligible sides neither
restrict the candle interval nor extend RMS trade activation.
Live cooldown fill-history coverage includes only globally enabled sides.
RMS requires one-minute backtest candles and its full replay window for
each consuming decision. An incomplete window defers only the entry side that uses
adverse cooldown, leaving closes and unrelated sides/coins available. N returns need
N+1 closes and become ready at index first_valid + N. Score-only RMS warmup is scoped
to required ranking. Compare one modifier at a time before combinations across distinct
periods and markets.

Lower fill counts alone do not establish an improvement. Assess drawdown, exposure,
underwater time, missed recoveries, fees and returns. Changing the span or weights
can change all of these. The implementation's regression tests establish semantics
and disabled-feature parity, not profitability.

CPU backtests maintain a rolling window with two aggregate stacks. Each return is
processed on arrival and transferred at most once, giving amortized constant work
per candle and memory proportional to the window, per enabled coin/span. Both sides
share a tracker when their spans match. This preserves finite-window expiry and
exact zero for an entirely flat window without subtracting nearly equal moments.
The live reconstruction remains a full replay; CPU cache state is not restart state.

A reproducible synthetic comparison against the replay reference is available:

```sh
cargo test --release --no-default-features --manifest-path passivbot-rust/Cargo.toml \
  unilateralness::tests::benchmark_rolling_against_replay -- --ignored --nocapture
```

The test reports both timings and maximum score error for the same fixed input.
Record CPU, compiler, OS, system load and repeated runs when comparing performance;
this indicator benchmark does not establish whole-optimizer throughput or profitability.

CPU backtests require one-minute candles only when RMS can be consumed. A
score-only side with one eligible coin, or a fixed position-slot budget covering
its eligible universe, leaves RMS inactive and can use aggregated candles.
Multi-coin dynamic-tradability budgets retain the requirement: held positions can
occupy slots while new coins become tradable. Adverse cooldown requires
one-minute candles on an entry-eligible, enabled side unless the clamp makes the
duration constant. Equal floor and ceiling, or a base already at/above the ceiling,
needs no modifier inputs. Optimizer preflight also checks corners where a lower
base/floor or higher ceiling makes those inputs necessary again.
