# Passivbot v7.11.0

Released 2026-05-13.

## Highlights

- Live authoritative state now always uses the staged account-state pipeline. The legacy live refresh selector was removed; use the `v7.10` branch if you need to compare old behavior.
- Staged live planning now requires explicit freshness for account state, completed candles, and market snapshots before Rust order calculation.
- Live market prices now come from a dedicated market snapshot provider instead of incomplete candle paths, with strict retry/fail behavior for missing ticker data.
- Live fill events now distinguish detected fills from realized-PnL enrichment. Close fills can log `pnl=pending`, PnL-dependent logic waits for enrichment, and an enrichment log is emitted when authoritative PnL arrives.
- Rust order orchestration now emits only the next most-likely flat entry order for live symbols without a position, while preserving full grid expansion once a position exists.
- Live initial-entry posting has an optional executor-side distance gate to reduce far-from-market EMA-driven order churn.
- Forager selection now supports score hysteresis and improved INFO/DEBUG diagnostics.
- Candle refreshes now prioritize active symbols, defer broad background warmup by default, cap forager refresh wall time, and tolerate bounded open-ended 1m tail gaps.
- Hyperliquid startup now detects account abstraction mode, treats `portfolioMargin` as unified-compatible for HIP-3/non-standard perps, and limits non-unified accounts to vanilla perps.
- Live logs were tuned to keep normal operation readable while preserving slow-path, order-wave, freshness, and exchange-error diagnostics.

## New Tools

- `passivbot tool ticker-probe`
- `passivbot tool ticker-endpoint-probe`
- `passivbot tool hyperliquid-abstraction-probe`

## Configuration Notes

- `live.authoritative_refresh_mode` was removed.
- `live.price_distance_threshold` was removed.
- `live.initial_entry_exec_max_market_dist_pct` controls the narrower live-only initial-entry posting economy gate.
- `live.forager_score_hysteresis_pct` defaults to `0.02`.
- `live.max_active_candle_tail_gap_minutes` defaults to `10`.
- `live.max_forager_candle_refresh_seconds` defaults to `45`.
- `live.max_ohlcv_fetches_per_minute` defaults to `24`.
- `live.recv_window_ms` defaults to `10000`.
- `live.max_n_cancellations_per_batch` must be greater than `live.max_n_creations_per_batch`.

## Upgrade Notes

- Reinstall after pulling so the Python package and Rust extension are rebuilt for `7.11.0`.
- Operators running live bots should expect the first startup after update to rebuild the Rust extension if stale.
- Review configs for removed `live.authoritative_refresh_mode` and `live.price_distance_threshold` keys; config normalization strips stale `price_distance_threshold`, but the legacy refresh selector is no longer supported.
- Watch early live logs after upgrade for exchange-specific ticker, candle, and account-state diagnostics, especially on KuCoin and Hyperliquid.

See `CHANGELOG.md` for the full change list.
