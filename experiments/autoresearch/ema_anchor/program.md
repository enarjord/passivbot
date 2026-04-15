# Passivbot ema_anchor Autoresearch Program

You are running bounded autonomous strategy research on Passivbot.

## Objective

Improve `ema_anchor` while preserving trading safety and keeping changes reviewable.

Primary goal:

- maximize `adg_strategy_pnl_rebased`

Hard constraints:

- no liquidation
- acceptable drawdown
- acceptable peak recovery time
- acceptable fill cadence
- no extremely long no-fill gaps

The authoritative scorer is:

- `src/tools/score_ema_anchor_autoresearch.py`

## Allowed Files

Editable by default:

- `passivbot-rust/src/strategies/ema_anchor.rs`

Editable only if strictly necessary for a strategy-local change:

- `passivbot-rust/src/strategies/mod.rs`

Do not edit any other file unless the human explicitly expands scope.

## Forbidden Changes

Do not modify:

- exchange code
- backtest harness
- optimizer engine
- config normalization/migrations
- CLI plumbing
- evaluation/scoring rules
- other strategies

Do not weaken:

- liquidation protections
- hard-stop behavior
- cooldown/risk gates
- statelessness requirements

## Research Loop

1. Read the current `ema_anchor` implementation.
2. Form one small hypothesis.
3. Make a minimal code change.
4. Rebuild Rust:
   - `cd passivbot-rust && maturin develop --release && cd ..`
5. Do not judge a strategy code change from one arbitrary config backtest.
6. Evaluate via warm-started optimize:
   - build one strong baseline XMR Pareto front first
   - after each code change, use that baseline Pareto front as seed input
   - run the candidate optimize round with `src/tools/run_ema_anchor_autoresearch_round.py candidate`
7. Compare the candidate run against the baseline run using the scorer.
8. Keep the candidate only if the best scored tuned candidate improves over the best scored tuned baseline.
9. If not improved, revert and try a different hypothesis.

## Fixed Evaluation Window

- symbol: `XMR`
- start date: `2024-10-01`
- end date: `2026-04-01`
- CPUs: `3`
- candle interval: `1`

## Default Commands

Baseline:

- `PYTHONPATH=src python3 src/tools/run_ema_anchor_autoresearch_round.py baseline --run`
- `PYTHONPATH=src python3 src/tools/write_ema_anchor_autoresearch_baseline.py optimize_results/autoresearch_baseline_<timestamp>_XMR`

Candidate:

- `PYTHONPATH=src python3 src/tools/run_ema_anchor_autoresearch_round.py candidate --baseline-pareto optimize_results/autoresearch_baseline_<timestamp>_XMR/pareto --run`

## Baseline Reference File

After the baseline optimize run, write `baseline.json` in the baseline run dir.

That file is the stable reference for:

- best baseline score
- best baseline Pareto member
- constrained metrics for the best member
- candidate command template

When comparing a candidate run, compare against `baseline.json`, not against an arbitrary config.

## Style Rules

- Prefer fewer moving parts.
- Prefer robust behavior over fragile cleverness.
- Do not add new parameters in the first pass unless the gain is compelling.
- Keep diffs small and explainable.
- Rust is the source of truth for strategy behavior.

## Good Hypotheses

- better offset shaping under volatility
- better qty shaping from inventory
- cleaner asymmetric band logic
- simpler protections against pathological overtrading

## Bad Hypotheses

- anything that changes unrelated infrastructure
- anything that games one symbol/date window at the expense of obvious robustness
- anything that disables risk features to juice ADG
