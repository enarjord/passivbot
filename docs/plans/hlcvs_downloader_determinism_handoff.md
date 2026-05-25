# HLCV Downloader Determinism Handoff

Status: superseded by the current HLCV coverage contract documented in
`docs/backtesting.md`, `docs/configuration.md`, `docs/tools.md`, and
`docs/ai/features/candlestick_manager.md`.

This historical handoff described an earlier strict/full-range design that no
longer matches the implementation. Do not use this file as implementation
guidance.

Current implementation guidance:

- Final `caches/hlcvs_data/` datasets require valid manifests. Manifest-less
  final caches are rebuilt.
- Explicit `--hlcvs-data-dir` override datasets require valid manifests and
  checksums.
- HLCV repair order is local v2 store, legacy raw shards, then targeted remote
  fetches for the remaining missing windows.
- Market availability gaps are coverage metadata, not automatic process
  failures. This includes leading pre-listing or unavailable prefixes, trailing
  unavailable suffixes, and verified internal exchange-side gaps.
- Hard failures are reserved for cases where the backtest cannot proceed
  meaningfully or safely: malformed/corrupt local data that cannot be repaired,
  no data inside the requested range, no tradable candles after warmup, or no
  BTC benchmark data.
- Known exchange-side gaps are retried after the catalog retry interval instead
  of being treated as permanent forever.
