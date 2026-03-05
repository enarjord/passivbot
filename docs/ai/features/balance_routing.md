# Balance Routing (Raw vs Snapped)

## Contract

1. `balance` is hysteresis-snapped for sizing stability.
2. `balance_raw` is true wallet balance for risk/accounting-sensitive paths.
3. Paths must not silently swap these semantics.

## Non-Obvious Details

1. Risk gates and peak-sensitive logic should use `balance_raw`.
2. Sizing/order-shaping logic may use snapped `balance` by design.
3. Legacy test stubs may only provide `balance`; treat compatibility fallback as transitional.

## Test Focus

1. Correct routing of raw vs snapped balance fields.
2. Regression coverage for peak/risk drift scenarios.
3. Explicit handling when `balance_raw` is missing/non-finite.

## Key Code

- `src/passivbot.py`
- `passivbot-rust/src/orchestrator.rs`
- `tests/test_passivbot_balance_split.py`
- `tests/test_orchestrator_json_api.py`
