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
5. Run the fixed inner-loop suite:
   - `passivbot backtest configs/examples/ema_anchor.json -s XMR -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`
   - `passivbot backtest configs/examples/ema_anchor.json -s DOGE -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`
   - `passivbot backtest configs/examples/ema_anchor.json -s SOL -sd 2024-01-01 -ed 2026-04-01 --disable_plotting all`
6. Score the newest artifacts:
   - `python3 src/tools/score_ema_anchor_autoresearch.py backtests/combined/<artifact1> backtests/combined/<artifact2> backtests/combined/<artifact3> --require-pass`
7. Keep the candidate only if the full suite improves.
8. If not improved, revert and try a different hypothesis.

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
