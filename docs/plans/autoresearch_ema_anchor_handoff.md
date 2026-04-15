# ema_anchor Autoresearch Handoff

## Status

This line of work is shelved for now on branch `exp/autoresearch-ema-anchor`.

The useful pieces on the branch are:

- constrained scorer:
  - `src/tools/score_ema_anchor_autoresearch.py`
- warm-start round runner:
  - `src/tools/run_ema_anchor_autoresearch_round.py`
- baseline manifest writer:
  - `src/tools/write_ema_anchor_autoresearch_baseline.py`
- protocol/instructions:
  - `docs/plans/autoresearch_ema_anchor.md`
  - `experiments/autoresearch/ema_anchor/program.md`

## Main Conclusion

For Passivbot, a Karpathy-style loop should not judge strategy code from one fixed config backtest.

The meaningful loop is:

1. build a strong baseline optimize run for the current strategy
2. after each strategy-code change, warm-start from that baseline Pareto front
3. run a smaller optimize with `--fine_tune_params`
4. compare tuned candidate vs tuned baseline with a constrained scorer

This is required because strategy code and parameter tuning interact strongly. A better strategy can
look bad under a stale config, and a worse strategy can look good if the comparison is unfair.

## Fixed Research Window

The branch scaffold uses:

- coin: `XMR`
- candle interval: `1m`
- date window: `2024-10-01` to `2026-04-01`
- CPUs: `3`

This was chosen as a first narrow loop suitable for a laptop.

## Current Constraint Set

The scorer currently requires:

- `adg_strategy_pnl_rebased >= 0`
- `drawdown_worst_hsl <= 0.35`
- `peak_recovery_hours_hsl <= 336`
- `fills_per_day >= 0.25`
- `hours_no_fills_max <= 168`
- `hours_no_fills_mean <= 72`
- `hours_no_fills_median <= 48`

These are conservative starting gates intended to block obvious overfitting or “dead bot” solutions.

## Baseline Manifest

`write_ema_anchor_autoresearch_baseline.py` scans an optimize run, scores every Pareto member with
the constrained scorer, and writes `baseline.json`.

That manifest is meant to be the stable reference for:

- best baseline score
- best baseline Pareto member
- best constrained metrics
- a ready-made candidate command template

The agent should compare candidates against `baseline.json`, not against an arbitrary config.

## Important Review Insights

### 1. `configs/examples/ema_anchor.json` drifted into experiment territory

It is no longer acting like a general example config. It is effectively an XMR-specific experiment
config with a fixed window and tuned bot state.

If this work resumes, it would be cleaner to move the research config into a dedicated experiment
path instead of continuing to overload `configs/examples/ema_anchor.json`.

### 2. Optimize bounds and optimize limits need alignment

When last reviewed, the `ema_anchor` config had these structural issues:

- mirrored long/short bot state, but inconsistent short-side bounds
- optimize limits not aligned with the outer constrained scorer
- `volume_pct_per_day_avg` still present as a scoring objective even though fill-gap metrics are a
  better fit for the stated research goal

If the loop resumes, align the optimize-space and optimize limits with the actual acceptance gate
before treating a run as the baseline to beat.

### 3. The tuned XMR config was promising but not baseline-worthy yet

Backtest artifact:

- `backtests/combined/2026-04-13T23_40_27/`

Notable metrics from that artifact:

- `adg_strategy_pnl_rebased = 0.0025678723`
- `drawdown_worst_hsl = 0.2872848203`
- `peak_recovery_hours_hsl = 531.95`
- `fills_per_day = 10.1627314715`
- `hours_no_fills_max = 90.95`
- `liquidated = false`

Interpretation:

- good raw tune
- healthy fill cadence
- no liquidation
- acceptable drawdown under the current cap
- failed the current autoresearch gate on `peak_recovery_hours_hsl`

So it looked promising as a return-seeking XMR tune, but not yet acceptable as the branch’s
constrained autoresearch baseline.

## Recommended Resume Order

If this work is resumed later:

1. cleanly separate the experiment config from the example config
2. align optimize bounds and optimize limits with the constrained scorer
3. run a serious baseline optimize
4. write `baseline.json`
5. only then start small strategy-code experiments in `passivbot-rust/src/strategies/ema_anchor.rs`

## Scope Guidance

The most disciplined first pass remains:

- editable file:
  - `passivbot-rust/src/strategies/ema_anchor.rs`
- optional only if strictly required:
  - `passivbot-rust/src/strategies/mod.rs`

Avoid letting the loop mutate:

- backtest harness
- optimizer engine
- exchange code
- config normalization/migrations
- unrelated strategies
