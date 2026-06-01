# Fill Events Manager

## Contract

1. Build a deduplicated fill-event stream per exchange/account.
2. Preserve source data needed for realized PnL reconstruction.
3. Keep fetch behavior explicit and observable during investigations.
4. Canonical event accounting uses `pnl_contract = gross_pnl_quote_fee_best_effort_v2`:
   `pnl` is gross price PnL excluding fees, `fee_paid` is signed quote-currency
   balance impact (paid fees negative, rebates positive), and net realized PnL
   is derived as `pnl + fee_paid`.
5. Fee resolution is best-effort but observable: reported quote fee, reported
   non-quote fee converted by a fresh ticker, reported fee rate, then
   `live.fee_pct_fallback`. Every fill is sanity-checked by fee/notional ratio
   against `live.fee_pct_sanity_abs_max`; outliers use the fallback percentage.
6. Do not mix legacy/missing-contract cache rows with current rows. Repair or
   rebuild legacy fill-event caches before using trading-critical accounting.

## Exchange Endpoint Map (Quick Lookup)

| Exchange | Primary Fills Source | PnL/Close Source |
|----------|----------------------|------------------|
| Binance | `fetch_my_trades` | income history |
| Bybit | `fetch_my_trades` | closed-pnl (hybrid pagination) |
| Bitget | `fetch_my_trades` | embedded in trade payload |
| Hyperliquid | fill events | embedded |
| OKX | `fetch_my_trades` | positions history |
| KuCoin | trades + positions history | positions history |
| Gate.io | `fetch_my_trades` | embedded |

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
