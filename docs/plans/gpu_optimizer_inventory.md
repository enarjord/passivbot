# GPU optimizer cutover inventory

Pinned starting implementation: master `00ce7d0ddf123071ae67db149753d6b3619f52ac`.
This is a development inventory, not a new runtime contract. Follow the acceptance
gates in [the development contract](gpu_optimizer_contract.md).

## Existing execution and ownership

- [`gpu_backend.py`](../../src/optimization/backends/gpu_backend.py) currently mixes
  request preparation, suite scheduling, proxy fitness, evolution, exact-worker pools,
  drift gates, checkpoints and publication of CPU-evaluated results. NSGA-II receives
  proxy fitness, while the archive receives CPU-validated records.
- [`service.py`](../../src/optimization/gpu/service.py) packs prepared backtest payloads,
  selects replay kernels and reduces compact output to scalar metrics. Its synchronous
  `evaluate(candidates)` methods are the transitional asynchronous-service adapters.
- [`mps_kernel.py`](../../src/optimization/gpu/mps_kernel.py) owns shader specialization,
  dispatch, mutable replay buffers and temporal chunking. Rust-owned shader sources
  run on CUDA through [`cuda_kernel.py`](../../src/optimization/gpu/cuda_kernel.py).
- CPU scoring/limits and scenario reduction remain with the existing evaluator helpers.
  Reusing their metric-processing methods must not call their CPU simulation methods.
- CPU backend dispatch lazily imports the GPU backend. Preserve that dependency boundary.

## Topologies and explicit exclusions

| Surface | Starting behavior | Cutover requirement |
| --- | --- | --- |
| EMA Anchor / Trailing Martingale | Single coin; directional multicoin; shared-account fused long/short multicoin | Keep effective config semantics across these paths |
| Hedge / one-way | Supported by the existing shared-account paths | Test shared cash, side arbitration and exposure, not sums of independent directional runs |
| HSL | Coin, pside and unified controllers subject to topology checks; bounded history | Preserve controller semantics and disabled-feature specialization |
| Unstuck | All listed topologies; static overrides; rolling TM multicoin history | Audit history semantics in every other topology |
| Recursive TM entries/closes | Supported, with market-order and exposure interactions | Cover ladder expansion, duplicate groups, minima and shared loss reservations |
| Scenarios / overrides | Supported paths validated before preparation; compatible grouping | Preserve canonical scenario reduction and distinguish screening from complete results |
| Other strategies / BTC collateral | `trailing_grid_v7` and positive collateral cap rejected | Keep explicit exclusions unless deliberately implemented |
| Coin count | At most 64 coins per prepared scenario | Keep explicit capacity errors; do not silently truncate |
| Metrics | Supported names in `metrics.SUPPORTED_METRICS`; exact-only exclusions in `metric_registry.GPU_EXACT_ONLY_METRICS` | Audit reducers and sentinels; do not export omitted metrics as neutral defaults |

The complete name lists remain code-owned in
[`metrics.py`](../../src/optimization/gpu/metrics.py) and
[`metric_registry.py`](../../src/optimization/gpu/metric_registry.py). Requested objective
and limit names must remain explicit. Existing CPU validation emits a broader surface;
the replacement must not claim to have computed those additional metrics.

## Deliberate differences requiring a decision

| Difference | Existing evidence / owner | Initial treatment |
| --- | --- | --- |
| Float32 paths versus CPU float64 | GPU parameter packing, shader state and CUDA lowering | Measure material effects; permit justified numerical differences |
| Conservative single-coin minimum-effective-cost filtering | Directional Rust shader filters; documented liquidation-floor and all-history-minimum bound | Multicoin now uses simulated cash/current-price minima; replace the remaining single-coin restrictions |
| Conservative realized-loss allowance | Shader loss gates and topology-specific histories; documented all-history/zero-loss envelopes | Compare rolling-window expiry and shared reservations; do not treat this as decimal noise |
| Rolling unstuck history only in some topologies | TM multicoin finite-history path and `test_gpu_unstuck_lookback.py` | Inventory consumers before reusing history storage elsewhere |
| Bounded logarithmic histogram tails | Fill-gap and drawdown reducers in `metrics.py` | Quantify bin error and optimizer feasibility/ranking effects before accepting |
| Hourly recovery sampling | Recovery distribution buffers and metric reducer | Evaluate sample-resolution error independently of simulation correctness |
| Weighted partial UTC-day exclusion | Weighted volume/daily reducers | Compare exact time boundaries and sample sufficiency |
| Independent hedged summary reducer | Retained helper `_combine_hedged_multicoin_outputs`; normal dual-side constructors select fused shared-account kernels | Do not accidentally revive this ranking-only fallback during service extraction |

See [current GPU behavior and limitations](../optimizing.md#gpu-backend-experimental)
for the existing public specification. Classification is provisional until standalone
parity evidence is collected. An asynchronous transport test certifies transport/replay
equivalence, not CPU parity or authority of a particular reducer.

## Starting regression corpus

- `tests/optimization/test_gpu_cuda.py`: dispatch ABI, current stream, capacity and
  disabled-HSL specialization, temporal chunking, production replay/service equivalence.
- `test_gpu_entry_sizing_parity.py`: raw-touch sizing, executable-price minima and
  forager readiness compared with real CPU fills.
- `test_gpu_unstuck_lookback.py`: finite-history expiry, shared accounting, reuse,
  scratch overflow and interruption.
- `test_gpu_hsl_multicoin.py` and related GPU HSL/market/loss tests: controller and
  generation-time execution behavior.
- `test_gpu_metrics.py`: individual reducer definitions and boundary handling.

Existing tests are useful starting evidence, not a claim that all feature combinations
or long histories are covered. Extend the corpus when the standalone parity tool exposes
material gaps. Distinguish baseline failures from asynchronous-service regressions.

## Baseline findings to resolve before cutover

- Multicoin TM passive execution originally expanded recursive entry and close
  suffixes only when market orders were enabled. Separating expansion from execution
  policy recovers the missing trajectory in the public parity fixtures. Small remaining
  fill/ADG differences stay visible under the tool's unchanged strict measurement gates;
  broader controller, reducer and loss-gate combinations still require assessment.
- The disabled-HSL CUDA regression expects compact state, but the current multicoin
  runner unconditionally sets `dispatch_hsl_disabled=False`. The earlier compact path
  was restricted to the removed legacy HSL engine. Restore useful specialization for
  the current controller ABI rather than blindly re-enabling the old condition.
- The synthetic dual-side unstuck regression reports 71 GPU fills versus 72 CPU fills
  with both finite and all-history lookbacks. Single-side cases pass. This test calls
  the synchronous replay directly, without the new execution service. Quantify account,
  position and metric effects before deciding whether to fix or accept the discrepancy.
- Two source-specialization tests leaked a fake Torch module into later device tests;
  import the real optional runtime for these source-only assertions. Coin-capacity tests
  also assumed capacity was the final shader-cache argument, although history-layout
  arguments now follow it; bind the arguments to the loader signature instead.
