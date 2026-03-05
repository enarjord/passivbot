# Fill Events Manager

## Contract

1. Build a deduplicated fill-event stream per exchange/account.
2. Preserve source data needed for realized PnL reconstruction.
3. Keep fetch behavior explicit and observable during investigations.

## Non-Obvious Details

1. Exchanges split fill/PnL data across different endpoints.
2. Bybit requires hybrid pagination for better closed-PnL completeness.
3. Historical retention limits can make old PnL records unavailable.

## Failure Modes To Watch

1. Missing records from pagination assumptions.
2. Duplicate events from multi-source merge logic.
3. PnL mismatch between trade feed and positions-history feed.

## Test Focus

1. Deduplication correctness.
2. Pagination completeness for high-activity windows.
3. PnL attachment behavior when auxiliary endpoints fail.

## Key Code

- `src/fill_events_manager.py`
- `docs/ai/exchange_api_quirks.md`
- `docs/ai/debugging_case_studies.md`
