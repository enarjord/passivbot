# Balance Routing (Raw vs Snapped)

Passivbot carries two balance values with distinct responsibilities:

- `balance`: hysteresis-snapped effective balance for sizing/order-shaping stability.
- `balance_raw`: true/raw wallet balance for risk/accounting accuracy.

## API Contract

### Rust orchestrator input (`OrchestratorInput`)

- `balance` is required and used for sizing logic.
- `balance_raw` is used for peak/PnL-sensitive risk gates.
- Legacy alias `balance_true` is accepted by Rust JSON input for backward compatibility with older callers.

### Python runtime (`src/passivbot.py`)

- `self.balance` stores the snapped/effective value.
- `self.balance_raw` stores the true/raw value.
- Helper methods:
  - `get_effective_balance()` -> snapped `balance`
  - `get_true_balance()` -> raw `balance_raw` (fallback-compatible for older test stubs)

## Routing Rules

- Use snapped/effective balance for sizing-like paths:
  - min-effective-cost eligibility
  - order-shaping paths that are intentionally hysteresis-stabilized
- Use raw/true balance for risk/accounting paths:
  - peak reconstruction (`balance_peak`)
  - realized-loss gate floor checks
  - auto-unstuck allowance budget calculations
  - equity/accounting displays

## Migration Guidance

For any external caller or test fixture constructing orchestrator JSON:

1. Provide both `balance` and `balance_raw`.
2. If you still send `balance_true`, Rust accepts it as an alias, but migrate to `balance_raw`.
3. Do not treat `balance` and `balance_raw` as interchangeable:
   - replacing `balance_raw` with snapped `balance` can reintroduce peak drift bugs.

For lightweight Python test doubles:

- Prefer defining both `balance` and `balance_raw`.
- If a legacy stub only has `balance`, `get_true_balance()` fallback keeps tests functional, but this should be transitional.

## Regression Coverage

See:

- `tests/test_passivbot_balance_split.py`
- `tests/test_orchestrator_json_api.py`

These tests verify split routing, legacy alias acceptance, and the peak-drift regression scenario.
