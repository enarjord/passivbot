# Directional-efficiency experiment

This opt-in experiment adds two independently configurable controls to the exact Rust planner:
a coin-ranking penalty and adverse-trend DCA pacing. Both default to zero. It does not implement
Hawkbot's strategy, support/resistance grids, inventory recycling, or target-average sizing.
Profitability is unproven; compare on unseen data at the same exposure limits and costs.

## Calculation and timing

For a lookback of N minutes, use N+1 completed 1-minute closes:

```text
r_i = log(close_i) - log(close_(i-1))
signed_efficiency = sum(r_i) / sum(abs(r_i))
```

A flat window returns zero; monotonic rises/falls return +1/-1. The calculation lives in Rust
and is also exposed to the live candle loader through the Python extension. Invalid prices fail;
missing rows are not converted into a neutral signal or a shorter lookback. Backtests use the
completed bar at index k and preceding history, never k+1. Live excludes the unfinished minute.
The existing backtest data preparation/gap policy still applies; set
`backtest.gap_tolerance_ohlcvs_minutes: 0` when requiring gap-free historical input.

## Parameters

The following keys exist separately under `bot.long` and `bot.short`:

| Path relative to the side | Default | Meaning |
|---|---:|---|
| `forager.directional_efficiency_lookback_minutes` | 60 | Integer window, 1–10080 minutes |
| `forager.directional_efficiency_penalty` | 0 | Ranking penalty strength, 0–1 |
| `risk.directional_efficiency_lookback_minutes` | 60 | Integer window, 1–10080 minutes |
| `risk.directional_efficiency_cooldown_minutes` | 0 | Maximum additional DCA delay, 0–10080 minutes |

The two consumers can use different windows. Coin overrides follow normal per-side merging.

### Ranking

After normal volume pruning and feature normalization:

```text
score = original_forager_score * (1 - penalty * abs(signed_efficiency))
```

This penalizes both sustained rises and sustained falls. Existing volume, volatility and EMA
readiness weights, tie-breaking, slots, exposure limits and score hysteresis remain in force.
Forced-normal and already-held symbols do not require this ranking feature. If all eligible
candidates fit the available slots, no ranking signal is required.

Live ranking may carry a completed real-candle window within the existing Forager candle
staleness budget. It never projects an unfinished price or invents zero returns for an unknown
tail. Missing/out-of-budget history excludes only candidates that actually require ranking.

### DCA pacing

For an already-held position:

```text
adverse = max(0, -signed_efficiency)  # long
adverse = max(0,  signed_efficiency)  # short
effective_cooldown = risk.entry_cooldown_minutes
                   + adverse * risk.directional_efficiency_cooldown_minutes
```

For example, a base delay of 5 minutes, additional delay of 30 minutes and long efficiency of
-0.8 require 29 minutes since the last position-increasing fill. The current completed signal
is reevaluated each cycle; the delay can shrink when direction changes. Existing fill-history
reconstruction supplies the last-increase timestamp, so restarting does not reset a timer.

Enabling pacing stages at most one adding order per batch, including the initial batch. This
prevents a simultaneously resting initial ladder from bypassing the fill-based delay. The initial
entry does not itself wait for an adverse-direction cooldown; additions to an existing position,
including partial-initial replenishment, do. Favourable movement adds zero delay, but sequential
staging remains enabled. This staging effect is part of the experiment and should be compared
with an ordinary fixed-cooldown baseline too.

Current completed history is required for pacing. A stale ranking window cannot authorize DCA.
When live history is missing, Rust omits only additions that consume it; ordinary closes and
independent protective reducers continue with their own required inputs. Normal reconciliation
cancels omitted entry orders. Exchange fills racing cancellation remain possible.

## Backtest and optimize

Only **1-minute backtests and the exact CPU optimizer** are supported. Enabled features or
nonzero feature bounds are rejected by GPU proxy execution rather than silently ignored.
Lookback bounds must use integer endpoints and an integer step. Defaults fix the new optimizer
dimensions at their disabled settings. To explore, explicitly set bounds such as:

```json
{
  "forager": {
    "directional_efficiency_lookback_minutes": [30, 240, 30],
    "directional_efficiency_penalty": [0, 0.75, 0.05]
  },
  "risk": {
    "directional_efficiency_lookback_minutes": [30, 240, 30],
    "directional_efficiency_cooldown_minutes": [0, 60, 5]
  }
}
```

These are example fields under `optimize.bounds.long`, not recommended trading parameters.
Use `optimize.backend: "pymoo"`. Warmup includes the complete finite window even if an EMA
warmup ratio or cap would otherwise shorten it. Newly listed coins cannot trade before that
window exists. Long windows increase calculation work; start with a modest search.

Run the included long-side ablation suite against your reviewed base configuration:

```bash
passivbot backtest path/to/config.json --suite y \
  --suite-config configs/examples/suites/directional_efficiency.json
```

The suite runs baseline, ranking-only, pacing-only and combined variants. It explicitly disables
both experimental short controls, and holds every other base setting unchanged. The standard
suite runner requires symmetric approved-coin lists even when short exposure is zero. Select
more eligible coins than position slots to exercise ranking. Use fixed dates, identical warmup,
exposure limits, fees and market slippage. With a prepared cache, add `--offline y`.

Evaluate equity drawdown, net equity return, exposure, position duration, underwater time and
costs, not just realized balance gains. Test several market regimes and an untouched holdout.
The normal one-minute fill simulation cannot resolve every intrabar path; no successful smoke
test establishes an execution edge or profitability.

## Validation

`tests/test_directional_efficiency.py` exercises the real extension, independent arithmetic,
long/short pacing, unchanged closes, missing/invalid inputs, deterministic restart, ranking,
future-candle invariance, strict windows, bounded live carry-forward, config round trips,
optimizer constraints and actual Rust backtest fills. Existing orchestrator, config, warmup and
live candle-budget suites remain part of the regression checks.
