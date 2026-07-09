# Crash Discovery And Stress-Suite Workflow

Use this guide when finding difficult historical periods in the local OHLCV cache or generating
optimizer/backtest crash suites. The generated artifacts are local research output; keep them under
`crash_finder_results/<YYYY-MM-DD>_crash_scenarios/` and do not commit them.

## What The Tool Does

`passivbot tool crash-finder` has two distinct modes:

1. Full discovery reads cached 1m candles, builds larger discovery candles, finds crash events,
   clusters related events, and generates scenario suites.
2. The `--clusters-csv` fast path regenerates suites from an existing cluster file. It does not read
   candles and cannot discover a crash added by a recent data download.

After updating candle data, run full discovery if the goal is to find new events. Use the CSV fast
path only when changing scenario-window, filtering, merging, or override settings while keeping the
previously discovered events.

## Discovery Semantics

- The cached source is currently required to be `1m` (`--source-timeframe 1m`).
- `--timeframe` controls the epoch-aligned discovery candle and defaults to `1h`. Durations such as
  `4h` and `12h` are supported.
- `1h` is the standard crash-suite horizon. It catches fast idiosyncratic failures such as M while
  still finding liquidation cascades. Wider candles can find slower moves, but may combine unrelated
  price action and change cluster timestamps.
- The default `ordered` metric is the worst high-to-later-low log return inside each discovery
  candle. Do not replace it with plain candle high/low range: the range cannot tell whether the low
  happened before the high and creates false crash events.
- `--threshold` is a log return. For example, `-0.10` is approximately a 9.5% price decline.
- `--min-valid-minutes` counts valid 1m source rows in a discovery candle. Reconsider this minimum
  when intentionally using sparse data or much wider candles.

The efficient grouping contract matters. Extract valid source-row indices once, derive contiguous
bucket boundaries once, then evaluate each compact bucket slice. Do not restore a per-bucket mask
over the full multi-year minute array; that turns a linear scan into an hours-times-minutes scan.

Full discovery verifies every cache chunk checksum before using it. This is deliberate. A future
optimization may stream verified monthly chunks instead of materializing one dense symbol range,
but it must preserve event results, gap handling, source order, and fail-loud integrity checks.

## Refresh Candle Data First

Build the requested coin universe from all scenario files, then update Binance and Bybit separately.
The downloader's combined-exchange path and each exchange's current market catalog can differ, so
separate commands make skipped symbols and failures attributable.

```shell
passivbot download \
  --symbols BTC,ETH,OM,M \
  --exchanges binance \
  --start-date 2021-01-01 \
  --end-date 2026-07-07

passivbot download \
  --symbols BTC,ETH,OM,M \
  --exchanges bybit \
  --start-date 2021-01-01 \
  --end-date 2026-07-07
```

Treat a current-market lookup failure separately from cached-data absence. Exchange symbols may use
contract aliases such as `1000PEPE`, `1000SHIB`, or `SHIB1000`; use the repository's coin/symbol
normalization helpers or catalog records instead of guessing `COIN/USDT:USDT`.

After downloading, verify the catalog's actual last timestamp per exchange and normalized coin.
The requested end date and the downloader's end-exclusive/materialized timestamp are not always
described identically in logs.

## Standard Full Discovery

Use a dated local output directory. The standard scenario has 14 days of lead-up and 60 days of
fallout, merges overlapping date windows, and writes filtered suites.

```shell
passivbot tool crash-finder \
  --root caches/ohlcvs \
  --exchange binance \
  --exchange bybit \
  --source-timeframe 1m \
  --timeframe 1h \
  --threshold -0.10 \
  --min-valid-minutes 2 \
  --top-per-coin 20 \
  --top-clusters 80 \
  --pre-days 14 \
  --post-days 60 \
  --scenario-coin-mode affected \
  --scenario-kind all \
  --scenario-force-normal both \
  --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios
```

Do not infer that a long first symbol means the run is stuck. Check the log's symbol progress and
measure a representative single-symbol scan first. The grouping implementation should make the
event scan linear, but checksum verification and dense range loading still have real cost.

## Fast Suite Regeneration

Use this only when the cluster CSV remains the event source of truth:

```shell
passivbot tool crash-finder \
  --clusters-csv crash_finder_results/<source>/crash_clusters.csv \
  --top-clusters 80 \
  --pre-days 14 \
  --post-days 60 \
  --scenario-coin-mode affected \
  --scenario-kind all \
  --scenario-force-normal both \
  --scenario-merge-overlaps \
  --write-filtered-suites \
  --output-dir crash_finder_results/$(date +%F)_crash_scenarios
```

Keep `scanned_ranges.csv` beside `crash_clusters.csv`. The fast path uses it to remove coins whose
data does not overlap a scenario window and copies the other scan artifacts when present. Older
`scanned_ranges.csv` files with `hours_scanned` are accepted as 1h discovery metadata.

## Scenario Requirements

- Market-wide clusters include the affected market coins but never force individual coins to normal
  mode. The forager should choose naturally during broad market stress.
- Forced-normal overrides are only for idiosyncratic, non-market-wide crash coins.
- No scenario may force more than two coins. If overlapping idiosyncratic events would exceed two,
  emit repeated scenarios for the same window with force targets split into groups of at most two.
- A scenario coin must have cached data overlapping that scenario's date range on at least one of
  the scenario's exchanges.
- Merge scenarios when their 14-day lead-up and 60-day fallout ranges overlap. Preserve the union of
  eligible coins and exchanges, the outer date range, and the worst-severity label.
- Keep separate all, market-wide, coin-focused, and strict single-coin suite files when
  `--write-filtered-suites` is requested.

Event labels use UTC bucket starts. A crash remembered by local calendar date may appear on the next
UTC date. The M event discussed during initial development is labeled around
`2026-06-25T00:00:00Z`, although it may be referred to as the June 24 crash locally.

## Validation Checklist

Before reporting a generated suite as complete, verify:

1. Full discovery was run after the latest candle download when new events were expected.
2. `scan_errors.csv` is reviewed; skipped symbols are not silently treated as successfully scanned.
3. Every scenario coin overlaps its date range according to `scanned_ranges.csv` or the current
   catalog.
4. Market-wide scenarios have no forced-normal overrides.
5. Every other scenario has at most two forced-normal coins.
6. The expected known events are present, including M and OM when their data is in scope.
7. `crash_finder_results/` remains untracked and is absent from the PR diff.

Run `pytest tests/test_crash_finder_tool.py` after changing scanner, clustering, suite, CSV, or CLI
behavior. Include a bounded real-cache single-symbol smoke scan when changing discovery performance.
