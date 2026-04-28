# Rust Strategy Runtime Refactor Execution Checklist

This is the implementation checklist for the current branch. It supersedes the older execution
plan that still assumed transitional naming and the top-level `strategy` config shape.

It is written as an execution document for this branch, not as a historical design note.

## Current Status

Completed on this branch:

- Pass 0: contract and compatibility policy locked in docs
- Pass 1: canonical strategy names are `trailing_grid` and `ema_anchor`
- Pass 2: `ema_anchor` uses strategy-specific order tags
- Pass 3: one-way `ema_anchor` emission rule is enforced via shared post-processing
- Pass 4/5: canonical strategy config moved under `bot.<side>.strategy.<kind>` with active-only
  user/artifact dumps
- Pass 6: backtest artifacts dump canonical config instead of transitional runtime shape
- Pass 7: optimize bounds are nested/grouped and strategy-aware
- New shared-bot grouping slice:
  - canonical shared bot config now lives under `bot.<side>.risk`
  - canonical shared bot config now lives under `bot.<side>.forager`
  - canonical shared bot config now lives under `bot.<side>.hsl`
  - canonical shared bot config now lives under `bot.<side>.unstuck`
  - runtime compilation still injects the flat shared keys the engine uses internally
  - shipped `refactor_test`, `ema_anchor`, and `ema_anchor_smoke_test` configs now use the
    grouped canonical shape
- Optimize-bounds cleanup slice:
  - nested optimize bounds now use subgroup-local keys instead of repeating legacy-style prefixes
  - flat optimizer keys remain only as an adapter/runtime boundary format
- Sparse user-facing config cleanup slice:
  - `clean_config()` now prunes inactive strategy subtrees and inactive strategy bounds
  - shipped canonical configs and artifact configs stay active-only by default

Current health-gate task:

- keep the Rust/Python boundary and config-pipeline tests green after the 2026-04-28
  `origin/master` merge
- use the post-merge `refactor_test` artifact as the current comparison target; the older frozen
  artifact remains a legacy pre-v7.10.0 reference only

## Branch

- `refactor/rust-strategy-runtime-plan`

## Baseline

Required parity command:

```bash
VIRTUAL_ENV=/Users/eiriknarjord/repos/passivbot-4/venv \
PATH=/Users/eiriknarjord/repos/passivbot-4/venv/bin:$PATH \
passivbot backtest configs/refactor_test.json
```

Legacy frozen baseline artifact:

- `backtests/combined/2026-04-03T19_45_50`

Legacy frozen baseline metrics for pure-refactor phases before the v7.10.0 order-behavior merge:

- `gain_usd = 3.6817791411966043`
- `adg_usd = 0.001099589010446378`
- `adg_strategy_pnl_rebased = 0.001099589010446378`
- `drawdown_worst_hsl = 0.5093468201133804`
- `drawdown_worst_mean_1pct_hsl = 0.3698590596944554`
- `loss_profit_ratio = 0.7172103105112119`
- `sharpe_ratio_usd = 0.03870218143031628`
- `sortino_ratio_usd = 0.03411395981422809`
- `hard_stop_triggers = 110`
- `hard_stop_restarts = 109`
- `total_wallet_exposure_max = 1.7589753109100572`
- `fills = 30539`

Current accepted post-merge baseline artifact:

- `backtests/combined/2026-04-28T18_33_35`

Current accepted post-merge baseline metrics:

- `gain_usd = 1.7003610119451211`
- `adg_usd = 0.00044768921199223044`
- `adg_strategy_eq = 0.00044768921199223044`
- `drawdown_worst_strategy_eq = 0.4606813979729244`
- `drawdown_worst_mean_1pct_strategy_eq = 0.38208417077733114`
- `loss_profit_ratio = 0.7972439602460609`
- `hard_stop_triggers = 66`
- `hard_stop_restarts = 66`
- `total_wallet_exposure_max = 1.7006920478587886`
- `fills.csv rows including header = 39451`

Baseline acceptance note:

- The post-merge artifact is expected to diverge from the legacy frozen artifact because `master`
  removed inflated grid re-entry behavior for v7.10.0. The first fill-level divergence is fill row
  `83` at `2023-01-04 20:10:00` for `SOL`: the legacy artifact emits
  `entry_grid_inflated_long` with `qty = 1628.31`, while the post-merge artifact emits
  `entry_grid_normal_long` with `qty = 1504.27`. The relevant trailing-grid config parameters are
  identical across the two artifacts; the difference is the intentional runtime behavior change.
  Legacy artifacts still decode historical `entry_grid_inflated_*` ids, but current runtime output
  should be normal-or-cropped.

Comparison rule after every behavior-affecting pass:

1. Run targeted tests for the touched area.
2. Rebuild the Rust extension if Rust changed.
3. Re-run `passivbot backtest configs/refactor_test.json`.
4. Compare `analysis.json` and `fills.csv` to the current accepted post-merge baseline.
5. If any discrepancy appears, stop and identify the first behavioral divergence before proceeding.
6. If the divergence reveals a pre-existing bug, decide explicitly which behavior is correct, fix
   it, and then update the expected comparison target.

## Locked Decisions

These are not open questions for this branch anymore.

### Strategy names

- Canonical strategy kinds are:
  - `trailing_grid`
  - `ema_anchor`
- These names are used directly with no branch-local alias layer.

### Compatibility policy

- Backward compatibility is only for official `master` release transitions.
- Do not add compatibility aliases, duplicate schema support, migrations, or shims between
  iterations of this dev branch unless explicitly requested.

### Order type identity

- `ema_anchor` must not reuse trailing-grid order type labels.
- Canonical `ema_anchor` order types:
  - `entry_ema_anchor_long`
  - `close_ema_anchor_long`
  - `entry_ema_anchor_short`
  - `close_ema_anchor_short`

### One-way position rule for `ema_anchor`

- This rule belongs at ideal-order emission / shared post-processing level, not fill simulation.
- Required emitted-order behavior:
  - if current long position exists, do not emit `entry_ema_anchor_short`
  - if current short position exists, do not emit `entry_ema_anchor_long`
  - closes remain clamped to the existing position size
- Prefer reusing or tightening the shared one-way / `hedge_mode: false` gate if it can express this
  cleanly. Do not implement a backtest-only fill-stage safeguard.

### Canonical config model

- `live.strategy_kind` selects the active strategy kind.
- Canonical config groups per-side data under `bot.<side>`.
- Shared subsystems live in fixed namespaces:
  - `bot.<side>.risk`
  - `bot.<side>.forager`
  - `bot.<side>.hsl`
  - `bot.<side>.unstuck`
- Strategy params live in fixed namespaces:
  - `bot.<side>.strategy.trailing_grid`
  - `bot.<side>.strategy.ema_anchor`
- The full reference schema contains all supported strategy subtrees.
- User configs may omit inactive strategy subtrees.
- Artifact configs should default to showing only the active strategy subtree.
- Runtime compatibility flattening must not leak into saved canonical configs.

### Optimize bounds model

- Human-facing optimize bounds are nested, not flat `long_foo_bar` keys.
- Bounds are grouped by side and subsystem.
- Active strategy bounds should be visibly separated from shared subsystem bounds.
- Artifact configs should not expand irrelevant optimize bounds by default.

## Target Canonical Shape

The fixed reference schema should be structurally deterministic even though only one strategy is
active at runtime.

Reference shape:

```json
{
  "live": {
    "strategy_kind": "trailing_grid"
  },
  "bot": {
    "long": {
      "risk": {},
      "forager": {},
      "hsl": {},
      "unstuck": {},
      "strategy": {
        "trailing_grid": {},
        "ema_anchor": {}
      }
    },
    "short": {
      "risk": {},
      "forager": {},
      "hsl": {},
      "unstuck": {},
      "strategy": {
        "trailing_grid": {},
        "ema_anchor": {}
      }
    }
  },
  "optimize": {
    "bounds": {
      "long": {
        "risk": {},
        "forager": {},
        "hsl": {},
        "unstuck": {},
        "strategy": {
          "trailing_grid": {},
          "ema_anchor": {}
        }
      },
      "short": {
        "risk": {},
        "forager": {},
        "hsl": {},
        "unstuck": {},
        "strategy": {
          "trailing_grid": {},
          "ema_anchor": {}
        }
      }
    }
  }
}
```

User-facing config instances may be sparse, for example keeping only the active strategy subtree.

## Pass Breakdown

### Pass 0: Lock The Contract In Docs

Commit intent:

- document the compatibility rule and the agreed strategy/config contract before more code changes

Tasks:

1. Add the branch compatibility rule to agent-facing docs.
2. Update this execution checklist to reflect the final agreed direction.
3. Update the main refactor doc as needed so it no longer recommends the old top-level
   `strategy.long` / `strategy.short` shape.

Phase gate:

- documentation is aligned with current decisions

### Pass 1: Rename Strategy Kinds

Commit intent:

- replace old branch-local strategy names with final canonical names

Tasks:

1. Rename strategy kinds everywhere to the final canonical names:
   - `trailing_grid`
   - `ema_anchor`
2. Remove old constants, spec identifiers, tests, config examples, and docs references.
3. Update Rust strategy registry and Python adapters to use only the new names.
4. Update smoke/example configs and tests accordingly.

Verification:

- config pipeline tests
- strategy spec tests
- orchestrator JSON API tests
- parity backtest against `configs/refactor_test.json`

### Pass 2: Fix Strategy-Specific Order Tags

Commit intent:

- make order identity match the active strategy

Tasks:

1. Replace reused trailing-grid order labels in `ema_anchor`.
2. Update any downstream analytics/tests that classify orders by type.
3. Consider making trailing-grid tags strategy-explicit too if that improves clarity.

Verification:

- orchestrator JSON API tests
- `ema_anchor` smoke backtest with fill inspection
- trailing-grid parity backtest

### Pass 3: Enforce No Opposite Entry Emission For `ema_anchor`

Commit intent:

- prevent same-candle flip behavior at the emitted-order level

Tasks:

1. Reuse or tighten the shared one-way / non-hedge gate if it already models the desired rule.
2. Ensure `ema_anchor` does not emit opposite-side entries while an opposing position exists.
3. Keep closes clamped to current position size.
4. Keep behavior identical in live and backtest by implementing this before fills exist.

Verification:

- new orchestrator regressions for blocked opposite entries
- `ema_anchor` smoke backtest
- trailing-grid parity backtest

### Pass 4: Reshape Canonical Config Under `bot.<side>`

Commit intent:

- move from the transitional top-level `strategy` section to the fixed grouped per-side schema

Tasks:

1. Redesign schema/reference defaults around:
   - `bot.<side>.risk`
   - `bot.<side>.forager`
   - `bot.<side>.hsl`
   - `bot.<side>.unstuck`
   - `bot.<side>.strategy.trailing_grid`
   - `bot.<side>.strategy.ema_anchor`
2. Remove the top-level canonical `strategy` section.
3. Update normalization, hydration, projection, validation, and runtime compilation.
4. Update runtime extraction so only `bot.<side>.strategy[active_kind]` is passed into Rust.

Verification:

- config pipeline tests
- config cleaning tests
- config utils tests
- trailing-grid parity backtest

### Pass 5: Support Sparse Active-Only User Configs

Commit intent:

- keep the fixed internal schema while allowing clean user configs

Tasks:

1. Allow user configs to provide only the active strategy subtree.
2. Hydrate missing active strategy params from defaults for the selected kind.
3. Do not auto-expand inactive strategy subtrees into user-facing dumped configs.
4. Validate the active subtree strictly.

Verification:

- config pipeline regressions for sparse active-only configs
- `ema_anchor` smoke backtest
- trailing-grid parity backtest

### Pass 6: Fix Backtest Artifact Config Dumping

Commit intent:

- ensure artifact configs are canonical and human-facing

Tasks:

1. Dump canonical config, not runtime-expanded compatibility config.
2. Show only the active strategy subtree by default.
3. Ensure `ema_anchor` artifacts do not include trailing-grid-only params.

Verification:

- artifact config regression tests
- `ema_anchor` smoke backtest and artifact inspection
- trailing-grid parity backtest

### Pass 7: Reshape Optimize Bounds

Commit intent:

- make optimize bounds canonical, grouped, and strategy-aware

Tasks:

1. Replace flat human-facing bound keys with nested grouped bounds.
2. Keep any flattening only at the optimizer adapter boundary if still needed internally.
3. Separate shared subsystem bounds from active strategy bounds.
4. Avoid dumping irrelevant optimize defaults into artifact configs.

Verification:

- optimization adapter tests
- optimize conversion tests
- trailing-grid parity backtest

### Pass 8: Refresh Reference Schema And Examples

Commit intent:

- make the reference/default surface match the final shape

Tasks:

1. Refresh examples for `trailing_grid` and `ema_anchor`.
2. Make smoke/example configs reusable and current.
3. Remove old naming and old schema examples from docs.

Verification:

- example config load tests
- `ema_anchor` smoke backtest
- trailing-grid parity backtest

### Pass 9: Final Cleanup

Commit intent:

- remove transitional remnants and dead paths

Tasks:

1. Remove obsolete old-name tests/comments/helpers.
2. Sweep docs for outdated terminology.
3. Keep only final canonical names and schema references.

Verification:

- targeted test suite for all touched areas
- final trailing-grid parity backtest
- final `ema_anchor` smoke backtest
