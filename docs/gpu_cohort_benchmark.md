# Offline GPU cohort benchmark

`passivbot tool gpu-cohort-benchmark` measures deterministic synthetic candidate
cohorts through serial CPU backtests, direct shared-account GPU replay and the
native CUDA service. It never downloads data, reads accounts or runs optimization.
It requires a full backtest installation, NVIDIA CUDA and a source-verified Rust
extension. CPU backtests belong to this explicit development measurement only.

```sh
passivbot tool gpu-cohort-benchmark --seeds 7 43 --report cohort.json
passivbot tool gpu-cohort-benchmark --strategies ema_anchor --coins 2 --bars 512 \
  --candidates 3 --warm-runs 1 --widths 1 2 auto
passivbot tool gpu-cohort-benchmark --adg-floor 0 --drawdown-ceiling 0.01
passivbot tool gpu-cohort-benchmark --metrics strategy_eq_recovery_days_p95 \
  volume_pct_per_day_avg_w --limit fills_gap_p95_hours greater_than 0.05
```

Defaults use both strategies, four coins, both sides, 10,080 one-minute candles,
16 candidates, seed 7 and three warm runs after each first run. Candidate `i` uses
quantity fraction `0.01 + i * 0.001` and first strategy EMA span `5 + i * 1.25`
on each active side. Other settings come from the public parity fixture. HSL and
unstuck default off; explicit toggles prepare additional workloads but do not
guarantee controller transitions. No production configuration is accepted.

The JSON report includes fixture/candidate and implementation fingerprints,
per-candidate strict CPU/GPU metric comparisons, two-objective Pareto membership,
pair-order disagreements including ties, and CPU ADG regret of the GPU-selected
maximum. Optional diagnostic limits use canonical feasibility calculations; these
are explicit measurement thresholds, not the constraints of a production search.
Missing or non-finite objectives leave ranking unassessed.

`--metrics` adds requested GPU work to the three core metrics; it does not replace
ADG/drawdown ranking or fills/day comparisons. This enables measurements of opt-in
histories/reductions and their resource cost. Repeat `--limit METRIC MODE VALUE` to
add canonical scalar diagnostic checks; modes are `less_than` and `greater_than`.
Limit metrics are automatically requested from both simulators. The convenience
`--adg-floor` and `--drawdown-ceiling` options remain available. These are measurement
thresholds, not a production search configuration or optimizer policy.

`--tolerances policy.json` uses the same explicit per-metric format as
[`gpu-parity`](gpu_parity.md#policies-and-reports). The resolved policies appear in
the report; the policy file path is omitted. Added metrics have no tolerance unless
one is provided or already has a provisional parity policy. Undefined policies remain
`unassessed`; they never become matches. Strict numerical comparisons and canonical
feasibility are separate, so an allowed error can still produce a limit flip.
Additional metric comparisons/limits do not extend the two-objective ranking test.

Native service results must retain the direct GPU metric keys and liquidation status
exactly. Finite metric differences may be at most eight float64 rounding units, using
the larger `math.ulp` of the two values. The fixed `native_reference_policy` and each
accepted difference's count, maximum absolute error and maximum rounding units appear
in the report. `matches_direct_gpu_exactly` remains false when rounding is observed;
`matches_direct_gpu` records agreement under this machine-scale policy. A float32 ULP
of replay drift exceeds this bound. CPU/GPU tolerance policies remain separate.
Requests are consumed individually through their futures and checked against the
submitted identity. Successful batch sizes
and controller evidence distinguish configured width from actual demand; a final
automatic width alone is not evidence that tuning improved performance.
The tuning report preserves cumulative eligible samples/seconds and completed
windows, including rejected trials; pending-window fields describe only the
unconsumed remainder. Cold shapes and underfilled batches remain ineligible.

Interpret timings by their scope:

- CPU samples include serial payload preparation and simulation. They do not
  represent multicore CPU optimization or its evolutionary/scenario overhead.
- Direct GPU preparation is reported separately. Its first replay may reuse an
  existing compiler cache; warm replay times follow it.
- Native first-use samples include worker preparation after direct GPU warmup.
  Compiler and packing caches are retained and may be reused; these samples are
  not cold machine/compiler times. Both paths use the same dispatch work budget.
- Latency is observed by the caller after submission and includes queueing and
  caller collection overhead. Internal microbatch completions may share a timestamp.
- Memory fields measure Torch allocated/reserved bytes during each service run,
  including its starting allocator cache. They omit allocations outside Torch and
  do not establish total device VRAM or process-memory peaks.

A completed measurement returns exit code `0`, even if strict CPU/GPU comparisons
disagree. Inspect those comparisons and ranking/feasibility evidence separately;
execution failures return `2` with structured error JSON. Use
[`gpu-parity`](gpu_parity.md) for per-candidate comparison exit status and detailed
parity diagnostics. Cohort throughput and Pareto agreement do not certify general
simulator parity or repeated-seed evolutionary search quality.

## Initial synthetic measurements

The default seed-7 recipe gives these median warm cohort times (seconds) on an
RTX 3070 Ti Laptop GPU, with Torch 2.13/CUDA 13.0 and CuPy 13.6. Each sample
contains 16 candidates; CPU is serial preparation plus simulation.

| Strategy | CPU serial | Direct GPU | Native width 1 | Width 4 | Width 16 | Automatic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EMA Anchor | 1.087 | 0.638 | 9.884 | 2.523 | 0.640 | 0.644 |
| Trailing Martingale | 8.898 | 2.474 | 35.369 | 9.185 | 2.493 | 2.489 |

All direct/native results match exactly across widths and repetitions. Automatic
mode retains configured width 64, observes at most 16 candidates per batch and
accumulates zero eligible controller samples. This workload demonstrates transport
equivalence and useful microbatching, not successful automatic tuning. Width 1 also
does not materially improve first-completion latency here: warm EMA first completion
is about 0.63 seconds versus 0.64 at width 16; TM is about 2.35 versus 2.49 seconds.

Seed 43, measured with widths 16/automatic, retains matching CPU/GPU front membership
and zero CPU ADG regret of the GPU-selected maximum, as does seed 7. EMA seed 43
has three ADG and two drawdown pair-order disagreements out of 120 pairs each;
the other three strategy/seed cases have none. Diagnostic ADG floor 0 and drawdown
ceiling 0.01 produce no feasibility flips: EMA CPU/GPU feasible counts are 8/16
for seed 7 and 4/16 for seed 43, while TM is 16/16 in both.

Strict metric comparisons still disagree: only 2/16 and 6/16 EMA candidates pass
all requested gates for seeds 7/43; no TM candidate does. For seed 7, maximum
absolute ADG/drawdown/fills-per-day errors are approximately `1.53e-6`, `1.47e-5`,
`0.288` for EMA and `8.77e-5`, `7.61e-7`, `8.338` for TM. Preserve these observations
and existing tolerances; front agreement in four fixed cohorts is limited evidence.

To isolate first-use CuPy compilation from an existing on-disk compiler cache:

```sh
CUPY_CACHE_DIR="$(mktemp -d)" passivbot tool gpu-cohort-benchmark \
  --seeds 43 --widths 16 auto --adg-floor 0 --drawdown-ceiling 0.01
```

In a fresh process with that empty CuPy cache, seed-43 direct first replay takes
34.68 seconds for EMA and 45.25 for TM, versus warm medians of 0.638/2.478.
This is a controlled CuPy-cache comparison; CUDA context initialization and prepared
data caches are not cleared by the tool. Native first-use samples still follow direct
warmup. Torch peaks are roughly 1.8 MiB in these small fixtures and do not establish
process RAM, total VRAM or larger-workload bounds.

## Refresh after shared-account semantic repairs

The following recipe repeats both seeds after the loss-history, recovery-resolution
and weighted-volume changes. It requests the tool's three default metrics, so the
optional recovery/volume capture buffers are compiled out; it does not measure their
enabled cost. Inputs, candidate fingerprints, strict metric comparisons and ranking
results match the earlier cohorts exactly.

```sh
passivbot tool gpu-cohort-benchmark --seeds 7 43 --widths 16 auto \
  --warm-runs 3 --adg-floor 0 --drawdown-ceiling 0.01 --report cohort.json
```

Median warm cohort seconds on the same GPU/runtime:

| Strategy | Seed | CPU serial | Direct GPU | Native width 16 | Automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA Anchor | 7 | 1.097 | 0.634 | 0.644 | 0.647 |
| EMA Anchor | 43 | 1.101 | 0.634 | 0.635 | 0.650 |
| Trailing Martingale | 7 | 8.894 | 2.468 | 2.458 | 2.469 |
| Trailing Martingale | 43 | 9.067 | 2.452 | 2.461 | 2.463 |

These measurements show comparable warm cost; they do not establish a speedup over
the earlier implementation. All native/direct outputs match exactly. CPU/GPU front
membership still agrees, GPU-selected CPU ADG regret remains zero and the two
diagnostic limits have no feasibility flips in the four fixed cohorts. The earlier
strict comparison failures and EMA seed-43 pair-order disagreements remain.

Automatic execution again has no eligible width samples: sixteen candidates cannot
fill its initial width of 64. It preserves that width without having measured an
optimal choice. Larger completed-work and suite measurements are still required.

An external observer sampled the benchmark process tree's RSS every 0.2 seconds and
global GPU memory via `nvidia-smi` every second. Sampled peak RSS was 1,039 MiB; global
GPU usage ranged from 344 to 963 MiB. This includes runtime/compiler allocations and
existing device usage; it is not a per-service VRAM maximum or a bound. Sampling can
miss transient peaks. The earlier Torch-only figures substantially understate total
resource use, and representative larger suites remain a separate acceptance gate.

## Refresh after bounded allowance and HSL integration

The same seven-day, four-coin, both-side, sixteen-candidate recipe runs on the integrated
bounded-allowance/HSL implementation with HSL disabled. The full strict metric rows,
ranking and diagnostic-limit results are unchanged from the prior refresh. Evaluation
and candidate-parameter digests change with the policy/schema update; these are fresh
measurements, not old fitness reused under a new implementation identity.

| Strategy | Seed | CPU serial | Direct GPU | Native width 16 | Automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA Anchor | 7 | 1.129 | 0.678 | 0.674 | 0.681 |
| EMA Anchor | 43 | 1.129 | 0.677 | 0.681 | 0.683 |
| Trailing Martingale | 7 | 9.040 | 2.508 | 2.516 | 2.513 |
| Trailing Martingale | 43 | 9.239 | 2.500 | 2.517 | 2.507 |

Values are median warm cohort seconds on the same GPU/runtime. This remains a
comparable-cost foundation rather than evidence of a speedup. Native/direct outputs
match exactly, fronts agree, the selected maximum has zero CPU ADG regret, and the
ADG/drawdown diagnostic checks have no feasibility flips. Strict passes remain 2/16
and 6/16 for EMA and 0/16 for TM; EMA seed 43 retains three ADG and two drawdown
pair-order disagreements. Automatic width again has zero eligible tuning samples.

The run starts with an empty CuPy cache using the earlier recipe. The first strategy
case takes 34.40 seconds for EMA and 46.98 for TM; second-seed cases reuse that cache.
Native first-use runs still follow direct warmup. These timings do not measure a cold
CUDA context or native-first compiler path, enabled HSL, or optional-history resource cost.

A requested-metric TM seed-43 cohort exposed weighted ADG values
`0.021492166751842533` and `0.021492166751842512` at batch sizes 16 versus 1+15.
The difference is `2.08e-17`, or six float64 rounding units. A controlled replay
verifies identical daily equity summaries, timestamps, fill counts and drawdowns; only
the float64 reduction result differs. The CUDA regression preserves those exact raw
inputs and bounds metric rounding. Material changes, missing metrics, changed
liquidation, non-finite disagreements and errors outside the stated bound still fail
with bounded per-metric diagnostics. This observation does not widen CPU parity gates.


## Optional metric cohort measurements

The integrated implementation also runs eleven requested metrics, including per-step
recovery and weighted-volume captures. Reproduce the four strategy/seed cohorts with:

```sh
passivbot tool gpu-cohort-benchmark --seeds 7 43 --widths 16 auto --warm-runs 3 \
  --adg-floor 0 --drawdown-ceiling 0.01 --metrics \
  drawdown_worst_mean_1pct_strategy_eq drawdown_worst_mean_1pct_ema_strategy_eq \
  fills_gap_p95_hours adg_strategy_eq_w strategy_eq_recovery_days_mean \
  strategy_eq_recovery_days_p95 strategy_eq_recovery_days_mean_worst_1pct \
  volume_pct_per_day_avg_w --limit fills_gap_p95_hours greater_than 0.05 \
  --limit strategy_eq_recovery_days_p95 greater_than 0.05 \
  --limit drawdown_worst_mean_1pct_ema_strategy_eq greater_than 0.01 \
  --report optional-cohort.json
```

Median warm cohort seconds on the same GPU/runtime:

| Strategy | Seed | CPU serial | Direct GPU | Native width 16 | Automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA Anchor | 7 | 1.116 | 0.704 | 0.699 | 0.709 |
| EMA Anchor | 43 | 1.122 | 0.702 | 0.713 | 0.719 |
| Trailing Martingale | 7 | 8.986 | 2.531 | 2.539 | 2.547 |
| Trailing Martingale | 43 | 9.203 | 2.515 | 2.525 | 2.533 |

All native/direct comparisons satisfy the stated machine-scale policy. TM seed 43
reports one six-unit weighted-ADG difference in each native mode, so those modes
explicitly report nonexact agreement. Other cases are exact. Core CPU/GPU comparison
rows and the ADG/drawdown ranking match the preceding refresh exactly. These five
diagnostic limits produce no feasibility flips in the tested cohorts; that does not
prove decisions near other thresholds or ranking with recovery/volume objectives.
Automatic width still has zero eligible tuning samples.

Added CPU/GPU metrics remain unassessed without explicit policies. EMA seed 7/43
maximum fill-gap p95 errors are three/two minutes; maximum worst-one-percent mean
recovery errors are about 93/58 minutes (CPU tail means about 3.38/5.55 days in those
candidates). Maximum absolute weighted-volume errors are approximately 0.00012/0.00624
for EMA and 0.319/0.153 for TM. These residuals need materiality assessment; they are
not resolved by allowing tiny float64 GPU-reference rounding. HSL is disabled here,
so its EMA tail is trivially zero and provides no HSL-controller or tail acceptance.

Per-service Torch peak allocated memory is roughly 4.86 MiB with requested histories.
A separate observer samples the whole benchmark process tree every 0.2 seconds and
global GPU memory every second: sampled peak RSS is 1,040 MiB, and global GPU usage
ranges from 344 to 1,029 MiB. Runtime/compiler and existing device use are included;
these sampled whole-run values are not per-service bounds and can miss transients.
The compiler cache is already warm, native runs still follow direct warmup, and this
small fixed-cohort experiment does not close larger-suite resource, cold-native,
default tuning or repeated-seed search-quality acceptance.
