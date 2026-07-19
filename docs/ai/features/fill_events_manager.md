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
7. Newly discovered fills may carry immutable `provenance` with attribution
   `first_ingested_by_runtime`. This identifies the Passivbot runtime that first
   persisted the fill locally; it does not claim that runtime created the order
   or caused the exchange fill. Refresh and deduplication preserve an existing
   provenance record, including the absence of provenance on legacy cache rows.
   Historical rows are never retroactively attributed.

## Runtime Provenance

The optional fill provenance record contains the runtime run id, Passivbot
version, Python Git commit and tracked-dirty state, canonical config hash, Rust
crate version, embedded Rust source fingerprint, loaded extension artifact hash,
runtime start timestamp, and the local first-ingestion timestamp. It contains
hashes rather than raw config, paths, commands, API payloads, or credentials.

The runtime identity proves which local runtime first ingested a fill after this
contract was introduced. Determining which historical runtime submitted an
order remains a separate attribution exercise using client-order identifiers,
logs, runtime windows, and immutable manifests.

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
| WEEX | `fetch_my_trades` in seven-day windows | embedded `realizedPnl` |

## Non-Obvious Details

1. Exchanges split fill/PnL data across different endpoints.
2. Bybit requires hybrid pagination for better closed-PnL completeness.
3. Historical retention limits can make old PnL records unavailable.
4. WEEX trade-detail queries are limited to 100 rows and seven days per request,
   with up to 365 days of retention; its client order id may require an order-detail lookup.
   Full responses are recursively split into disjoint time windows because the endpoint does not
   guarantee row ordering or expose a stable cursor. Saturation within one millisecond is unavailable
   rather than silently treated as complete.

## Failure Semantics And Risks

1. Missing records from pagination assumptions.
2. Duplicate events from multi-source merge logic.
3. PnL mismatch between trade feed and positions-history feed.

Exchange fetch methods propagate endpoint failures. The manager or caller may repair, retry,
quarantine, rebuild, or defer according to `../error_contract.md`; it must not attach neutral PnL
merely because an auxiliary endpoint failed.

## Validation

1. Deduplication correctness.
2. Pagination completeness for high-activity windows.
3. PnL attachment behavior when auxiliary endpoints fail.
4. Provenance round-trip, preservation during refresh/deduplication, and legacy
   rows remaining unattributed.

## Key Code

- `src/fill_events_manager.py`
- `exchange_integrations.md`
- `../case_studies/debugging.md`
