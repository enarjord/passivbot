# AGENTS.md

Instructions for AI coding assistants working on Passivbot.

## Always Read First

Read these files for every task:

1. `AGENTS.md`
2. `docs/ai/principles.yaml`
3. `docs/ai/error_contract.md`

Then use `docs/ai/README.md` to load task-specific docs only when relevant.

## Non-Negotiables

1. Rust is source of truth for order behavior.
- Behavior changes in entries/closes/risk/unstuck belong in `passivbot-rust/src/`, not Python patches.
2. Stateless behavior is required.
- Bot behavior must be reproducible after restart from exchange state + config.
3. Fail loudly in trading-critical paths.
- Default is hard-fail for exchange data, EMA inputs, risk gates, and order construction.
- Fallbacks are exceptions, not defaults, and must follow `docs/ai/error_contract.md`.
4. Keep terminology and signed-qty conventions exact.
- `position_side` = long/short.
- `side` / `order_side` = buy/sell.
- `qty` and `pos_size` are signed in internal logic.
5. EMA spans are floats.
- Do not round derived spans like `sqrt(span0 * span1)`.
6. Avoid scope creep.
- Make only requested or strictly necessary changes.

## Before Coding

1. Read `docs/ai/README.md` and open only docs relevant to the task.
2. If touching exchange code, read `docs/ai/exchange_api_quirks.md`.
3. If touching a documented feature, read the corresponding file in `docs/ai/features/`.
4. Run a silent-handling self-audit for touched areas:

```bash
rg -n "except Exception|return_exceptions=True|\.get\([^\n]*,\s*(0|0\.0|None|False|\{\}|\[\])\)" src tests
```

5. Remove unsafe patterns or document explicit, approved fallback behavior with tests.

## Testing Expectations

1. Run targeted tests for changed paths.
2. Add regression tests for bug fixes and fallback behavior.
3. If Rust changed, rebuild extension before Python tests:

```bash
cd passivbot-rust && maturin develop --release && cd ..
```

See `docs/ai/code_review_prompt.md` for the review/test checklist.

## Commands

Use `docs/ai/commands.md` for setup, test, backtest, optimizer, and Rust build commands.

## Documentation Hygiene

1. Keep AI docs lean and task-oriented.
2. Put durable rules in `principles.yaml` or `error_contract.md`, not in many files.
3. Put deep investigations in case-study docs, not core instruction docs.
4. Update `CHANGELOG.md` for user-facing behavior changes.
