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

## Suggested Inner-Loop Suite

Start with a short fixed suite and disable plotting:

1. `passivbot backtest configs/examples/ema_anchor.json -s XMR -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`
2. `passivbot backtest configs/examples/ema_anchor.json -s DOGE -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`
3. `passivbot backtest configs/examples/ema_anchor.json -s SOL -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`

Then score the newest three artifacts with the scorer.

## Promotion Rule

Do not keep a candidate only because one run improved. Promote only when:

- all suite members pass constraints
- aggregate suite score improves
- diff is reviewable and localized

For now, keep the loop human-supervised at the commit boundary.
