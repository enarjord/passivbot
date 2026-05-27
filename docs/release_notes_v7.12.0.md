# Passivbot v7.12.0

Released 2026-05-27.

## Highlights

- Backtest and optimizer HLCV preparation now treat normal market availability limits as coverage metadata instead of aborting whole runs for late coin starts or unavailable tails.
- Final `caches/hlcvs_data/` datasets now require valid manifests and artifact checksums; manifest-less final caches rebuild by default.
- Explicit `backtest.hlcvs_data_dir` override datasets now require valid manifests, with `intersection` and `dataset` modes for config-clipped replay or exact artifact replay.
- Materialized HLCV datasets now include per-coin coverage metadata for valid ranges, missing leading/trailing spans, internal gaps, and synthetic fill provenance.
- Strict v2 HLCV repair now handles authoritative pre-inception boundaries, stale persistent gap records, sparse valid windows, and direct writes into `caches/ohlcvs` without repopulating deprecated `caches/ohlcv` shards.
- Backtest analysis and optimizer metrics gained additional recovery-duration, entry-interval, completion-ratio, and finite omega-ratio behavior needed for reliable suite/optimizer aggregation.
- Several live-data correctness fixes landed for Bybit UTA balance parsing, KuCoin realized-PnL enrichment, staged live tail-gap EMA projection, and pending-PnL refresh backoff.

## Configuration Notes

- New configs should use `config_version = "v7.12.0"`.
- `backtest.hlcvs_data_dir` can point to a prepared final HLCV dataset for replay.
- `backtest.hlcvs_data_override_mode` accepts `intersection` (default) or `dataset`.
- Old manifest-less final HLCV caches are no longer authoritative in the deterministic default path and will be rebuilt.

## Upgrade Notes

- Reinstall after pulling so the Python package and Rust extension are rebuilt for `7.12.0`.
- Expect the first backtest/optimizer run after upgrade to rebuild old final `caches/hlcvs_data/` entries that do not have a valid manifest.
- If using `--hlcvs-data-dir`, verify the target dataset includes `manifest.json`, `timestamps`, `hlcvs`, `btc_usd_prices`, `coins`, and `market_specific_settings`.

See `CHANGELOG.md` for the full change list.
