# Code Review Checklist (Lean)

Use this checklist when reviewing Passivbot changes.

## Critical Checks (Block Merge)

1. Rust boundary respected for behavior changes.
2. Stateless requirement preserved.
3. Error handling follows `error_contract.md`.
4. `position_side` vs `side` usage is correct.
5. Signed qty conventions preserved.
6. EMA span derivations remain float.
7. No security regressions or secret leakage.

## Important Checks

1. Config placement follows hierarchy (`config.live` default).
2. Exchange-specific pagination/quirks handled where relevant.
3. Logging level/tag usage is sane for operators.
4. Tests cover normal path, edge cases, and regression behavior.
5. User-facing changes have `CHANGELOG.md` entry.

## Test Execution Minimum

1. Run existing relevant tests.
2. Add/run targeted tests for changed paths.
3. If Rust changed, rebuild extension before Python tests.

```bash
cd passivbot-rust && maturin develop --release && cd ..
pytest
```

## Report Format

1. Findings first, ordered by severity, with `file:line`.
2. Open questions/assumptions.
3. Short change summary.
4. Tests run and outcomes.
