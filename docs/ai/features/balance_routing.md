# Balance Routing (Raw vs Snapped)

## Contract

1. `balance` is hysteresis-snapped for sizing stability.
2. `balance_raw` is true wallet balance for risk/accounting-sensitive paths.
3. Paths must not silently swap these semantics.

## Failure Semantics

Missing or non-finite `balance_raw` makes a risk/accounting consumer unavailable; it must not
silently substitute snapped balance. Test-only stubs may provide an explicit compatibility value,
but that fallback must not enter production runtime behavior.

## Non-Obvious Details

1. Risk gates and peak-sensitive logic should use `balance_raw`.
2. Sizing/order-shaping logic may use snapped `balance` by design.
3. Some legacy test stubs provide only `balance`; migrate them or keep any compatibility adapter
   explicitly test-scoped.

## Validation

1. Correct routing of raw vs snapped balance fields.
2. Regression coverage for peak/risk drift scenarios.
3. Explicit handling when `balance_raw` is missing/non-finite.

## Key Code

- `src/passivbot.py`
- `passivbot-rust/src/orchestrator.rs`
- `tests/test_passivbot_balance_split.py`
- `tests/test_orchestrator_json_api.py`
