# Offline GPU/CPU parity checks

`passivbot tool gpu-parity` runs a real Rust CPU backtest and a GPU replay through
the asynchronous execution service, then compares their metrics and canonical limit
feasibility. It does not run optimization or download market data. It requires a
full backtest installation, an optional supported GPU runtime, and a Rust extension
whose embedded source fingerprint matches the current source. Normal optimization
never calls this tool.

For repeated cohort throughput, caller-observed completion latency and Pareto
effects, use the standalone [GPU cohort benchmark](gpu_cohort_benchmark.md).

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
passivbot tool gpu-parity --fixture trailing_martingale --coins 1 --gpu-engine native
```

Defaults are 5,760 one-minute candles and seed 7, with two synthetic markets,
USD cash and no BTC collateral. Side activation, HSL mode, unstuck, market execution
and minimum-effective-cost filtering are explicit options. These toggles exercise
configuration paths; enabling a controller does not prove that a particular fixture
triggers its transitions. Add targeted stress fixtures for behavioral coverage.

`--gpu-engine native` uses the actual `CudaBacktestService` selected by native GPU
optimization, including its shared-account replay for a single coin. It requires NVIDIA
CUDA and never falls back to the legacy path. The default `legacy` retains the existing
single-coin/multicoin replay selection, including supported Apple Metal installations.
Reports label both the chosen engine and replay family; a one-coin legacy comparison does
not establish parity of the native shared-account implementation. Neither mode changes
metric tolerances or calls this comparator during optimization.

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
string array `coins` in sorted order, matching the canonical backtest payload layout.
Unsorted inputs are rejected; sort coin identities and reorder candle columns together
when preparing the NPZ. Object/pickle arrays are rejected. If the raw config declares
`backtest.coins[exchange]`, its order must exactly match the dataset. The same checks
apply to a supported `{"config": {...}}` candidate wrapper. Exchange aliases are
normalized consistently; conflicting coin lists under equivalent aliases are rejected.
The markets JSON
is the normal prepared market-settings mapping, including quantity/price steps,
minima, fees, valid indices, warmup, explicit per-coin `exchange` and
`__meta__.requested_start_ts`. `--exchange` must match the effective config's sole
exchange, or be `combined` for multiple exchanges. Candle venues use `ohlcv_source`,
falling back to the market `exchange` when absent, and must match the configured data
sources and explicit combined `backtest.coin_sources`. Market settings may independently
use `backtest.market_settings_sources`. Combined preparation's documented fallback to
the candle venue is accepted as already resolved metadata; the tool does not fetch or
substitute settings. The Binance default does not infer or relabel a prepared exchange.

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

For native service comparisons, `gpu_cold` includes worker preparation and simulation;
`gpu_prepare` is null because that timing is not exposed by the service boundary. CPU
preparation of shared arrays precedes that timer. Native diagnostics retain the bounded CPU
fill summary and actual GPU liquidation status; they do not expose worker buffers or
invent unavailable final positions. Legacy diagnostics retain their raw scalar summaries.

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

Running the same 21-case seed-7, 5,760-bar matrix explicitly through `--gpu-engine native`
preserves matching completion and the two passing long-only TM cases. Native one-coin
shared-account results also expose small residual differences: EMA ADG relative error
reaches 0.4911% and fill error 0.1967%; passive short/dual-side TM ADG/fill relative errors
stay below 0.055%/0.035% in these fixtures. These measurements retain the same strict
policies and do not certify controller-transition coverage or ranking acceptance.

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
