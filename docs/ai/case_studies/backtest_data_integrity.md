# Stock-Perp Backtest Data Integrity Decision

The stock-perp data-integrity audit accepted tradable synthetic flat candles during traditional
market closures for Hyperliquid `xyz:` backtests using `ohlcv_source_dir`.

The model reflects a venue that trades continuously while source equity markets close. Flat
synthetic candles can dilute volume/range features and create a regime seam with native data, but
unmarking them would make the period categorically untradable. Strict backtest fill inequalities
already prevent fills for orders resting exactly at the unchanged flat price.

The current contract and validation targets live in `../features/stock_perps.md`; the original deep
audit remains in `docs/plans/backtest_data_integrity_audit_20260709.md`.
