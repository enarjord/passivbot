# Offline GPU/CPU parity checks

`passivbot tool gpu-parity` runs a real Rust CPU backtest and a GPU replay through
the asynchronous execution service, then compares their metrics and canonical limit
feasibility. It does not run optimization or download market data. It requires a
full backtest installation, an optional supported GPU runtime, and a Rust extension
whose embedded source fingerprint matches the current source. Normal optimization
never calls this tool.

The existing GPU simulator still has deliberate screening approximations. A passing
comparison covers the selected inputs, metrics and policies; it does not certify all
configurations. The development [contract](plans/gpu_optimizer_contract.md) tracks
the work required before authoritative GPU optimization replaces screening/validation.

## Reproducible fixtures

```sh
passivbot tool gpu-parity --fixture trailing_martingale --coins 2 --sides long
passivbot tool gpu-parity --fixture trailing_martingale --coins 2 --sides short --diagnostics
passivbot tool gpu-parity --fixture ema_anchor --coins 2 --sides both --hsl unified
passivbot tool gpu-parity --fixture trailing_martingale --sides both --market-orders
```

Defaults are 5,760 one-minute candles and seed 7, with two synthetic markets,
USD cash and no BTC collateral. Side activation, HSL mode, unstuck, market execution
and minimum-effective-cost filtering are explicit options. These toggles exercise
configuration paths; enabling a controller does not prove that a particular fixture
triggers its transitions. Add targeted stress fixtures for behavioral coverage.

The fixture's requested dates and ordered market metadata match its candle arrays.
The tool strips optimizer gene bounds when preparing standalone comparisons; it
does not generate candidates or apply a search policy.

## Prepared inputs

```sh
passivbot tool gpu-parity --config candidate.json --dataset prepared.npz \
  --markets markets.json --exchange binance --report comparison.json
```

The NPZ must contain `hlcvs` with shape `(bars, coins, 4)` (high, low, close, volume),
one-dimensional `timestamps` in milliseconds, aligned `btc` prices, and an ordered
string array `coins`. Object/pickle arrays are rejected. If the raw config declares
`backtest.coins[exchange]`, its order must exactly match the dataset. The markets JSON
is the normal prepared market-settings mapping, including quantity/price steps,
minima, fees, valid indices, warmup and `__meta__.requested_start_ts`.

Preserve effective candidate/scenario settings and dataset preparation metadata.
Fixture switches (`--sides`, `--coins`, `--bars`, `--seed`, `--hsl`, `--unstuck`,
`--market-orders` and `--filter-by-min-effective-cost`) are rejected with `--config`,
including explicit default values. Change the prepared config to compare another
simulation setting; these switches only construct synthetic fixtures.
Fixed runtime overrides are materialized through the optimizer's canonical helper
before either simulation. Optimizer `enable_overrides` must already be materialized
and cleared in prepared inputs; unresolved policies are rejected.
The tool compares one prepared scenario; suite-scoped limits are reported as requiring
a suite rather than treated as satisfied. Keep private inputs and reports out of commits
and PRs. Reports omit config contents and full histories; optional diagnostics still
contain coin identities and a small sample of fills.

## Policies and reports

The default requested metrics are `adg_strategy_eq`, `drawdown_worst_strategy_eq`
and `fills_per_day`. Configured limit metrics are also requested. Defaults include
provisional tolerances for these metrics and `backtest_completion_ratio`. They are
measurement gates, not release-wide accepted discrepancies. Supply a JSON policy to
override or extend them:

```json
{
  "adg_strategy_eq": {"absolute": 0.0000001, "relative": 0.0001},
  "fills_gap_longest_days": {"absolute": 0.0007, "relative": 0},
  "omega_ratio_strategy_eq": {
    "absolute": 0.00001, "relative": 0.0001, "matching_infinity": true
  }
}
```

Use `--metrics NAME ... --tolerances policy.json`. Allowed error is
`absolute + relative * abs(CPU value)`. Relative error in the report uses the larger
observed magnitude, so it remains bounded for normal finite values. Missing metrics
or policies cannot pass. NaN never matches; equal infinities match only when explicitly
enabled for that metric. Unsupported metrics and failed simulation are distinct from
successful comparisons with mismatches.

Canonical limit checks run independently of the numerical tolerance. Even a metric
within tolerance can flip feasibility. Scalar/single-scenario reducers use the singleton
observation; missing or non-finite observations cannot become apparently feasible.
Suite-scoped checks remain unassessed.

Standard JSON goes to stdout; simulator chatter goes to stderr. `--compact` removes
indentation and `--report PATH` also saves the report. Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | All requested metrics and assessed limits match |
| 1 | A mismatch or incomplete comparison |
| 2 | Unsupported contract, invalid inputs or execution failure |

An optional report-save failure also returns 2 and writes a diagnostic to stderr;
the completed comparison is preserved on stdout.

Reports include an input/config digest, Python and Rust source fingerprints, per-metric
values/errors/policies, feasibility checks and separate CPU, GPU preparation and cold
execution timings. These timings describe a diagnostic single-request run, not optimizer
throughput. `--diagnostics` additionally requests CPU fills and reports a bounded CPU
fill/state summary and raw GPU scalar summaries; it changes the CPU collection path
and is not a warm performance benchmark.

GPU replay construction and resource cleanup run on the service's owning worker via a
registered factory. Preparation timing is measured during that construction and excluded
from cold execution timing. Diagnostic hooks are restored before replay disposal.

## Measured development parity

The initial 21-case seed-7 matrix covers both strategies, three side modes, one/two
coins and selected feature toggles at 5,760 bars. These are historical observations
before the multicoin admission and passive-recursion fixes:

- Matching completion coverage and two passing long-only Trailing Martingale cases.
- Small EMA Anchor fill differences and ADG relative differences up to about 0.50%
  in these fixtures. They remain observations, not an approved blanket tolerance.
- Material multicoin short/dual-side Trailing Martingale passive-order differences:
  roughly 60% fewer GPU fills and 95–97% lower ADG. Market-order variants reduce this
  to much smaller differences, narrowing the order-path investigation.
- Minimum-effective-cost filtering suppresses all GPU trades in the tested dual-side
  fixtures while CPU trades, consistent with the existing conservative admission rule.

Both large gaps are now corrected in these fixtures. Repeating the 21-case matrix after
the fixes preserves matching completion and the two passing long-only TM cases. Strict
policies still expose small residual differences: multicoin passive TM short/dual ADG
relative errors are approximately 0.0412%/0.0549%, and fill errors 0.0234%/0.0210%.
EMA differences remain at the earlier scale. These observations do not widen the tool's
policies or establish a blanket acceptance tolerance.

Minimum-cost boundary coverage checks affordable equality and cases 0.01% below/above
the threshold for both strategies. Inputs below float32 resolution can still straddle
the CPU float64 boundary; the tool reports their potentially large path discontinuities
as mismatches. See [GPU admission precision](optimizing.md#gpu-backend-experimental).

The first fixture revision failed to account for canonical UTC-day end-date normalization;
the recorded matrix uses matching midnight endpoints. Its earlier completion discrepancy
was a fixture-preparation finding and is not included as a simulator defect.

These are simulator measurements, not execution-service regressions. The matrix
does not establish general HSL/unstuck transition coverage, long-run rolling-history
parity, or suite/ranking acceptance. Follow the [inventory](plans/gpu_optimizer_inventory.md)
and development checklist for those remaining gates.
