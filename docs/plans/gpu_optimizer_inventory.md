# GPU optimizer cutover inventory

Pinned starting implementation: master `00ce7d0ddf123071ae67db149753d6b3619f52ac`.
This is a development inventory, not a new runtime contract. Follow the acceptance
gates in [the development contract](gpu_optimizer_contract.md).

## Existing execution and ownership

- Retained legacy [`gpu_backend.py`](../../src/optimization/backends/gpu_backend.py) mixes
  request preparation, suite scheduling, proxy fitness, evolution, exact-worker pools,
  drift gates, checkpoints and publication of CPU-evaluated results. NSGA-II receives
  proxy fitness, while the archive receives CPU-validated records.
- [`service.py`](../../src/optimization/gpu/service.py) packs prepared backtest payloads,
  selects replay kernels and reduces compact output to scalar metrics. Its synchronous
  `evaluate(candidates)` methods are the transitional asynchronous-service adapters.
- [`mps_kernel.py`](../../src/optimization/gpu/mps_kernel.py) owns shader specialization,
  dispatch, mutable replay buffers and temporal chunking. Rust-owned shader sources
  run on CUDA through [`cuda_kernel.py`](../../src/optimization/gpu/cuda_kernel.py).
- Native [`gpu_native_backend.py`](../../src/optimization/backends/gpu_native_backend.py)
  owns search and persistence through `NativeCandidatePlanner`, `NativeEvaluationSession`
  and `CanonicalResultScorer`. CPU preparation/scoring does not invoke simulation.
  [`native.py`](../../src/optimization/gpu/native.py) owns resident replay behind prepared
  descriptors; [`executor.py`](../../src/optimization/gpu/executor.py) receives requests
  and completes their futures. Device implementation and tuning remain service-owned.
- CPU scoring/limits and scenario reduction remain with the existing evaluator helpers.
  Reusing their metric-processing methods must not call their CPU simulation methods.
- CPU backend dispatch lazily imports the GPU backend. Preserve that dependency boundary.

## Topologies and explicit exclusions

| Surface | Starting behavior | Cutover requirement |
| --- | --- | --- |
| EMA Anchor / Trailing Martingale | Single coin; directional multicoin; shared-account fused long/short multicoin | Keep effective config semantics across these paths |
| Hedge / one-way | Supported by the existing shared-account paths | Test shared cash, side arbitration and exposure, not sums of independent directional runs |
| HSL | Coin, pside and unified controllers subject to topology checks; bounded history; enabled HSL requires minute candles | Preserve controller semantics and disabled-feature specialization |
| Unstuck | All listed topologies; static overrides; bounded rolling EMA/TM shared-account history | Native single-coin uses shared-account replay; retained legacy single-coin history remains separate |
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
| EMA flat-candidate ranking cadence | Original shared EMA replay retained flat selections until a fill or eligibility change; the replacement refreshes every bar and uses outstanding entry orders for hysteresis, like CPU/TM | Twelve native side/scope fixtures, 191 replay/selection cases and 36 native optimizer CLI cases pass; PR #1918 passed current-head review and CI and is integrated into development. The two-coin short-only material gap closes with HSL enabled and disabled. |
| HSL time-in-red | Shared-account replay now integrates elapsed steps using preceding portfolio RED state, including temporal carry; retained legacy directional replay still counts samples | Initial/terminal/repeated-time probes and controller-active native comparisons cover the reporting contract. Keep trading decisions separate from observational time accounting. |
| TM short/shared shock fixtures | On six two-coin, 3,000-bar, seed-43 cases, original/corrected shaders have identical assessed non-time metrics; CPU ADG differs by at most 0.00004505 and fill rate by 0.0978% | Accept 0.00005 absolute ADG and 0.1% relative fill-rate error only on those fixtures. Drawdown and lifecycle gates stay unchanged, and corrected time-in-red matches CPU exactly. These bounds do not widen general tool policy or certify other trajectories. |
| Disabled-HSL EMA long ADG | Original and corrected shaders return identical assessed metrics on the two-coin, 3,000-bar, seed-43 fixture; fills match CPU and ADG differs by 0.000000888 | Accept 0.000001 absolute ADG error only for this fixture; keep default standalone-tool gates unchanged and assess other numerical differences separately. |
| Conservative single-coin minimum-effective-cost filtering | Retained legacy directional shaders use liquidation-floor/all-history-minimum bounds | Native requests use simulated cash/current-price minima through the shared-account engine even for one coin. Keep legacy-only restrictions separate from native acceptance. |
| Realized-loss allowance | EMA/TM shared-account admission uses effective finite/all fill history and shared generation-time reservations; retained legacy directional single-coin engines remain separate | Shared-account envelopes have been replaced; preserve finalized quantities, expiry and unfilled reservations. Do not attribute legacy-only exclusions to native single-coin requests, which use the shared-account engine. |
| Rolling fill-PnL history | EMA/TM shared-account auto-unstuck and loss admission; `test_gpu_unstuck_lookback.py`, `test_gpu_realized_loss_lookback.py`, `test_gpu_tm_loss_admission.py` | Both consumers share bounded scratch and preserve expiry/intrabar peaks, including loss-only history. HSL history remains separate. |
| Portfolio HSL EMA tail | Rust observes each bar's maximum enabled signal scope, then reduces its worst 1%; shared GPU replay now captures the same joint observation | The former max of side tails lost joint timing. A controlled 200-bar case gives 0.9 versus 0.6; bounded histogram and f32 replay materiality remain separate acceptance work. |
| Bounded logarithmic histogram tails | Fill-gap and drawdown reducers in `metrics.py`; fill-gap percentiles restore same-candle zeros and use 512 bins, independent of 128-bin initial-entry intervals | Short-gap cohorts improve with fixed storage; assess long-gap/overflow and drawdown-tail materiality separately before accepting |
| Recovery distribution sampling | Requested metrics retain every simulation step and reduce strict time-to-exceed durations on the GPU | Shared EMA/TM recovery observations use raw realized PnL plus UPNL, including terminal losses below account clamping, independently of weighted capture. Strict comparisons remain sensitive to float32 plateaus and small trading differences. Service dispatch budgets include sample and reduction scratch. |
| Traded-volume normalization and suffixes | Shared EMA/TM replay normalizes actual fill quantities and reduces requested weighted suffixes from GPU per-step contributions | Retained legacy directional single-coin/daily-only helpers still approximate partial days. Assess residual CPU/GPU fill-trajectory differences independently of the volume reducer. |
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
  scratch overflow and interruption; both strategies and native single-coin/shared
  EMA service replays, without CPU simulation in the service.
- `test_gpu_hsl_multicoin.py` and related GPU HSL/market/loss tests: controller and
  generation-time execution behavior.
- `test_gpu_metrics.py`: individual reducer definitions and boundary handling.

Existing tests are useful starting evidence, not a claim that all feature combinations
or long histories are covered. Extend the corpus when the standalone parity tool exposes
material gaps. Distinguish baseline failures from asynchronous-service regressions.

The [acceptance evidence map](gpu_optimizer_acceptance.md) separates verified service,
CPU search/storage and real-device CLI cases from the remaining simulator/metric and
resource gates. The lifecycle foundation does not itself certify metric authority.

## Baseline findings to resolve before cutover

- Multicoin TM passive execution originally expanded recursive entry and close
  suffixes only when market orders were enabled. Separating expansion from execution
  policy recovers the missing trajectory in the public parity fixtures. Small remaining
  fill/ADG differences stay visible under the tool's unchanged strict measurement gates;
  broader controller, reducer and loss-gate combinations still require assessment.
- Disabled-HSL specialization is restored for the current controller ABI in eligible
  single-side EMA pside/unified replay. `_use_disabled_hsl_specialization` verifies the
  effective matrix and excludes coin overrides that may enable HSL. Coin mode retains
  per-coin forced-delisting loss telemetry; fused/TM layouts remain separate. Do not
  treat the original unconditional `dispatch_hsl_disabled=False` as current behavior.
- The original high-churn dual-side unstuck fixture reports 71/72 GPU/CPU fills, but
  its retired cooldown alias never changed the canonical zero cooldown. The corrected
  allowance-exhaustion experiment sets `entry_cooldown.base_duration_minutes` directly;
  CPU/GPU agree on 21 finite-lookback fills and 10 all-history fills, and its 26-case
  suite passes. This resolves the intended fixture, not every high-churn trajectory;
  retain the old measurement as historical evidence, not a general parity claim.
- The source-specialization Torch leakage and positional cache-argument assertions
  are corrected: device modules import the real optional runtime and capacity checks
  bind arguments to the loader signature. Keep these as regression coverage rather
  than unresolved simulator defects.
- The all-157-metric audit produces finite output for six long shock cases but is not
  full parity acceptance. Normalized HSL loss now uses the existing produced panic-loss
  sum. Lifecycle replacement covers open RED, GREEN restart, retriggers and censored
  durations. The review correction passes 58 lifecycle/endpoint cases, 57 broader
  reporting cases, 181 replay/ablation controls and 36 CPU-forbidden optimizer cases.
  Unified portfolio events no longer populate directional restart counters, and retained
  forced-delisting endpoints are distinguished from ordinary-fill liquidation.
  Four additional retained ordinary-liquidation comparisons still differ by one minute:
  identical published/current shader metrics and positions expose existing extra-entry
  or missing market-panic behavior. All eight native counterparts pass. Do not confuse
  retained-engine trading differences with reporting-clock acceptance. The earlier seven
  assessed HSL metrics match in six long comparisons; full cutover remains unaccepted.
- Ordinary side-equity reporting formerly depended on HSL signal eligibility.
  Shared replay now records factual raw cashflows and UPNL on the account-equity
  clock independently of protection. Constant inactive curves retain their complete
  recovery horizon. The side-risk matrix, including HSL-disabled and unified cases,
  covers that correction; two TM long drawdowns retain documented fixture-local
  one-basis-point bounds. Arbitrary trading trajectories remain unaccepted.
- Requested weighted raw/account capture now follows the corresponding Rust curves
  and suffix definitions; unweighted raw growth uses compact factual daily summaries.
  Controlled and same-curve references establish those reductions, while ordinary
  replay differences retain explicit fixture-local limits. Preserve that distinction
  when assessing shape, histogram-tail and long-trajectory differences below.

## Current account-shape evidence

The shared controlled Rust fixture now covers the six account-equity shape fields
at both input precisions and three fill variants. CPU/CUDA reductions and fixed-price
BTC routing pass all 2,160 comparisons. The twenty-day TM short diagnostic in the
[acceptance record](gpu_optimizer_acceptance.md#controlled-account-equity-shape-references)
finds matching clocks and daily capture; actual Rust reductions of identical curves
agree within 1e-12. Its 5.205% weighted-jerkiness replay difference amplifies an
equity-curve deviation below .059%. Keep that replay/materiality debt separate from
the now-tested shape definitions; no general tolerance policy is widened.

The related four-cohort assessment preserves ADG/weighted-jerkiness fronts and
pair ordering, but re-ranking all shape axes changes two EMA fit fronts and nine
pair relations. A choppiness gap reaches 29.524% symmetric relative error. Strong
reference calculations do not guarantee stable thresholds near a singular ratio
or close objective ties. Keep these observations visible in practical acceptance.
