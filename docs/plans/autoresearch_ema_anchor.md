# ema_anchor Autoresearch Plan

## Goal

Run a tightly constrained agent loop that improves `ema_anchor` strategy behavior without letting the
agent mutate backtest plumbing, exchange code, or config architecture.

## Why This Branch

Use `refactor/rust-strategy-runtime-plan` as the base because:

- `ema_anchor` is actively evolving here.
- Rust is already the source of truth for strategy behavior.
- backtest/live parity is stronger here than on older branches.

Do not run the experiment directly on `refactor/rust-strategy-runtime-plan`. Use a child branch such
as `exp/autoresearch-ema-anchor`.

## Scope

First pass scope:

- strategy under test: `ema_anchor`
- preferred editable file: `passivbot-rust/src/strategies/ema_anchor.rs`
- optional secondary file only when strictly required:
  - `passivbot-rust/src/strategies/mod.rs`

Out of scope for the first loop:

- exchange code
- backtest harness
- optimizer engine
- config migrations
- Python orchestration
- other strategies

## Evaluation Contract

The loop should optimize `adg_strategy_pnl_rebased` subject to hard constraints:

- not liquidated
- drawdown capped
- peak recovery capped
- minimum fills/day
- maximum no-fill gap

Use [score_ema_anchor_autoresearch.py](/Users/eiriknarjord/repos/passivbot-4/src/tools/score_ema_anchor_autoresearch.py)
to score backtest `analysis.json` files or artifact dirs.

Default constraints in the scorer:

- `adg_strategy_pnl_rebased >= 0`
- `drawdown_worst_hsl <= 0.35`
- `peak_recovery_hours_hsl <= 336`
- `fills_per_day >= 0.25`
- `hours_no_fills_max <= 168`
- `hours_no_fills_mean <= 72`
- `hours_no_fills_median <= 48`

These are deliberately conservative starting gates. Tighten them only after the loop is stable.

## Concrete Protocol

Use a two-stage loop:

1. build a strong XMR-only baseline with a serious optimize run
2. after each strategy-code change, warm-start from the baseline Pareto front and run a much smaller
   fine-tune optimize budget

This is required for Passivbot because code changes and parameter tuning interact strongly.

### Fixed Research Window

- symbol: `XMR`
- `backtest.candle_interval_minutes = 1`
- start date: `2024-10-01`
- end date: `2026-04-01`
- CPUs: `3`

This is roughly an 18-month window and keeps the benchmark stable across candidates.

### Step 1: Baseline Generation

Run a serious baseline optimize budget on the unmodified strategy:

```bash
PYTHONPATH=src python3 src/tools/run_ema_anchor_autoresearch_round.py baseline
```

Defaults:

- config: `configs/examples/ema_anchor.json`
- symbol: `XMR`
- iterations: `100000`
- CPUs: `3`

This prints the exact optimize command. Add `--run` to execute it.

After the baseline run finishes, write a stable manifest for agents and follow-up tools:

```bash
PYTHONPATH=src python3 src/tools/write_ema_anchor_autoresearch_baseline.py \
  optimize_results/autoresearch_baseline_<timestamp>_XMR
```

This writes `baseline.json` beside the optimize run and records:

- baseline run dir
- baseline pareto dir
- constrained scorer settings
- best Pareto member path
- best score
- best metrics
- candidate command template

### Step 2: Candidate Evaluation

After a code edit, evaluate the candidate against the same fixed XMR window using the baseline Pareto
front as seed input and a smaller fine-tune budget:

```bash
PYTHONPATH=src python3 src/tools/run_ema_anchor_autoresearch_round.py candidate \
  --baseline-pareto optimize_results/autoresearch_baseline_<timestamp>_XMR/pareto
```

Defaults:

- iterations: `3000`
- warm-start seeds: baseline Pareto front
- tunable params: a fixed `ema_anchor`-centric subset including exposure, unstuck EMA dist, and
  strategy-local params such as offset, EMA spans, volatility weights, and double-down factor

This is a bounded local search around a known strong region, not a fresh blind optimize run.

### Promotion Rule

Promote a code change only if:

- the candidate optimize run passes the scorer constraints
- the best scored Pareto member from the candidate run beats the `best_score` recorded in
  `baseline.json`
- the code diff remains localized to the allowed strategy surface

Do not promote based on one arbitrary config backtest. Always compare tuned-vs-tuned.

## Promotion Rule

Do not keep a candidate only because one run improved. Promote only when:

- all scored candidate results pass constraints
- the best candidate score improves over the baseline best score
- diff is reviewable and localized

For now, keep the loop human-supervised at the commit boundary.
