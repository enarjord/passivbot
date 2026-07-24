# Crash Discovery And Stress-Suite Runbook

`passivbot tool crash-finder` finds difficult historical crashes and pumps in local OHLCV data and generates
backtest/optimizer scenario suites. Keep output under
`crash_finder_results/<date>_crash_scenarios/`; it is local research evidence, not PR content.

## Modes And Semantics

- Full discovery reads cached 1m candles, builds epoch-aligned discovery candles, finds ordered
  high-to-later-low and/or low-to-later-high events, clusters them by direction, and generates
  scenarios.
- `--clusters-csv` regenerates suites from existing clusters without reading candles. It cannot
  discover events added by a later data download. It preserves every direction present in the CSV
  unless `--direction down|up` is explicitly supplied as a regeneration filter.
- `--timeframe` defaults to `1h`; wider candles may combine unrelated moves and shift cluster time.
- `--threshold` is log return (`-0.10` is about a 9.5% decline).
- `--pump-threshold` is a positive log return (`0.10` is about a 10.5% rally).
- `--min-valid-minutes` counts valid source rows per discovery candle.

Do not replace the ordered metric with plain high/low range; range cannot prove that the high
preceded the low. Preserve the linear grouping contract: derive valid indices and contiguous
bucket boundaries once, then evaluate compact slices. Full discovery verifies cache checksums
before use.

## Refresh And Verify Source Data

Downloads contact public exchange endpoints. Run Binance and Bybit separately so failures and
catalog differences remain attributable:

```bash
passivbot download --symbols <coins> --exchanges binance \
  --start-date <start-date> --end-date <end-date>
passivbot download --symbols <coins> --exchanges bybit \
  --start-date <start-date> --end-date <end-date>
```

Use repository symbol normalization or catalog records for aliases such as `1000PEPE` and
`SHIB1000`. Verify actual catalog coverage rather than inferring it from the requested end date.

## Full Discovery

```bash
passivbot tool crash-finder \
  --root caches/ohlcvs \
  --exchange binance --exchange bybit \
  --source-timeframe 1m --timeframe 1h \
  --direction both --threshold -0.10 --pump-threshold 0.10 --min-valid-minutes 2 \
  --top-per-coin 20 --top-clusters 80 \
  --pre-days 14 --post-days 60 \
  --scenario-coin-mode affected --scenario-kind all \
  --scenario-force-normal adverse --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios
```

Do not call a long first-symbol scan stuck without checking progress and measuring a bounded
single-symbol scan. Checksum verification and dense range materialization have real cost even when
event grouping is linear.

## Fast Regeneration

```bash
passivbot tool crash-finder \
  --clusters-csv crash_finder_results/<source>/crash_clusters.csv \
  --top-clusters 80 --pre-days 14 --post-days 60 \
  --scenario-coin-mode affected --scenario-kind all \
  --scenario-force-normal both --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios
```

Keep `scanned_ranges.csv` beside the cluster CSV. It removes coins without overlapping source
coverage and preserves scan metadata.

## Scenario Contract

- Market-wide scenarios include affected coins but never force individual coins to normal mode.
- Idiosyncratic `adverse` scenarios isolate long exposure for crashes and short exposure for
  pumps by setting the opposite side to manual mode.
- Market-wide `adverse` scenarios isolate the whole portfolio side by setting the non-adverse
  side's `total_wallet_exposure_limit` to zero without forcing individual coins.
- Forced-normal overrides apply only to idiosyncratic events and at most two coins per scenario.
- Split overlapping force targets into repeated scenarios when more than two are required.
- Every scenario coin has source data overlapping its date range on at least one scenario exchange.
- If filtering removes all targets, omit the scenario; never remove `coins` and broaden it silently.
- Merge overlapping lead-up/fallout windows while preserving eligible coins, exchanges, outer dates,
  and worst severity.
- UTC bucket labels may differ from the event's remembered local calendar date.

## Validation

1. Use full discovery after the latest download when new events are expected.
2. Review `scan_errors.csv`; skipped symbols are not successful scans.
3. Check scenario coverage against `scanned_ranges.csv` or the current catalog.
4. Verify market-wide and forced-normal constraints.
5. Confirm expected regression events when their source data is in scope.
6. Keep `crash_finder_results/` out of the PR diff.
7. Run `pytest tests/test_crash_finder_tool.py`; include a bounded real-cache single-symbol smoke
   when discovery performance changes.
