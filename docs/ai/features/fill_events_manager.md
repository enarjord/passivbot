# Fill Events Manager

## Contract

1. Build a deduplicated fill-event stream per exchange/account.
2. Preserve source data needed for realized PnL reconstruction.
3. Keep fetch behavior explicit and observable during investigations.
4. When reconstructing historical position state from fills, current live exchange positions are authoritative over synthetic replay residuals.
5. Position reconstruction for restart-critical consumers should use two passes:
   - replay backwards from authoritative current position sizes to infer initial position size
   - replay forwards from that seed to annotate `psize`/`pprice`
6. If reconciled replay still disagrees with live position truth, log explicit warnings; do not silently treat synthetic replay as authoritative in HSL-critical paths.

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
4. Initial `pprice` cannot be recovered reliably from terminal state alone. Two-pass reconciliation can infer initial size, but may still leave initial `pprice` unknown.
5. HSL replay must recognize historical panic-finalization from fill-event truth, not just coarse minute-bucket state. Same-minute re-entry and incomplete history can otherwise leave stale pre-panic peaks alive across restart.

## Failure Modes To Watch

1. Missing records from pagination assumptions.
2. Duplicate events from multi-source merge logic.
3. PnL mismatch between trade feed and positions-history feed.
4. Synthetic forward-only replay can drift into phantom residual positions that do not match current exchange truth.
5. A residual replay mismatch after `close_panic_*` can incorrectly suppress historical panic-finalization unless restart logic is bounded by live flatness checks and explicit warnings.

## Test Focus

1. Deduplication correctness.
2. Pagination completeness for high-activity windows.
3. PnL attachment behavior when auxiliary endpoints fail.
4. Two-pass size reconciliation from authoritative current positions.
5. Warning visibility when reconstructed positions still disagree with exchange truth.
6. HSL restart regressions where a recent panic should reconstruct active cooldown instead of re-entering RED from a stale peak.

## Key Code

- `src/fill_events_manager.py`
- `docs/ai/exchange_api_quirks.md`
- `docs/ai/debugging_case_studies.md`
