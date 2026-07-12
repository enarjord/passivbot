# Active Decision Index

This index records why selected current contracts exist. The linked contract, not this summary, is
normative. Superseded discussion remains in git history and `case_studies/`.

| Decision | Rationale | Current contract |
|---|---|---|
| Rust owns trading behavior | Prevent live/backtest divergence and Python-side strategy patches | `principles.md`, `architecture.md` |
| Decisions are restart-reproducible | Local history must not silently change post-restart trading behavior | `principles.md` |
| Exchange fetch methods propagate failures | Caller policy needs complete error context for retry, defer, or restart | `error_contract.md` |
| Bybit closed-PnL uses hybrid pagination | Cursor-only and broad time windows can each lose coverage | `features/exchange_integrations.md`, `features/fill_events_manager.md` |
| Rolling PnL lookback matches the naive live contract | Optimized peak/current bookkeeping previously drifted after the window moved | `case_studies/debugging.md`, `features/strategy_runtime.md` |
| Canonical strategy schema does not silently accept removed legacy fields | Development-branch aliases hide migration errors and split behavior | `features/strategy_runtime.md` |

Add an entry only when the rationale materially prevents a likely future reversal. Do not duplicate
ordinary feature documentation here.
