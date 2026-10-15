# Native CUDA service suite benchmark

`passivbot tool gpu-service-benchmark` measures three synthetic scenarios through
identified native-service requests. It runs no CPU simulations, downloads or
optimizer search. Use [GPU parity](gpu_parity.md) for CPU/GPU correctness and the
[cohort benchmark](gpu_cohort_benchmark.md) for ranking and diagnostic limits.

```bash
passivbot tool gpu-service-benchmark --strategy ema_anchor --prepare-only --report prepared.json
passivbot tool gpu-service-benchmark --strategy ema_anchor --report service.json
passivbot tool gpu-service-benchmark --strategy trailing_martingale --candidates 128 \
  --tuning-windows 2 --max-rounds 256 --report tuning.json
```

The preparation-only mode creates requests without importing Torch, CuPy or the
replay kernel. Execution requires a full CUDA installation and a source-verified
Rust extension. Skipped verification, absent stamps and mismatched source
fingerprints are rejected before fixture or device preparation. All data and candidate configurations come from the public
seed-seven parity fixture. Both sides run; `--hsl unified` enables factual HSL
with a 0.99 threshold and one-day lookback. This tool does not claim every risk
transition occurs in that fixture.

The base scenario uses all selected coins and rows. Early and late scenarios
use disjoint halves of the same candle arrays and the first/last third of coins
(at least two). Requests interleave these scenarios. Scenario preparation,
packing, residency switches and async completion use the production service.
Accumulation defaults to the production adaptive controller. Set
`--accumulation-delay 0` to isolate width tuning, or supply a bounded fixed delay.
The recipe records this choice; width evidence windows remain unchanged.
One device dataset must remain resident, and shared source arrays must remain
unchanged. Packed spill files must be removed after each service closes.

Width one supplies isolated GPU metric references in two rounds. Fixed width
eight and automatic execution run at least `--rounds` rounds each. Every result
must retain its request/scenario identity and liquidation flag; metrics must
agree with the width-one reference within eight float64 reduction ULPs. This
permits machine-scale reduction differences, not float32 replay drift. Required
result, ownership and integrity checks remain enabled under optimized Python
execution. Recovery
and weighted metrics are requested to include their history/reduction costs.

The report separates first-use latency, warm median throughput, first completion
and 95th-percentile request latency. Compiler/cache state is retained between
phases; first use of a later phase is not a fresh cold compile. Automatic phases
can have more rounds than fixed phases, so their medians are not matched-duration
comparisons. Workload and observed batch counts remain in the report.

`--tuning-windows N` extends automatic execution until **every scenario** completes
N production tuner evidence windows, or until `--max-rounds` is reached. It does
not shorten the default 24-sample/30-second evidence windows. Exit status two
means the requested evidence was insufficient; the completed measurements are
still saved. Enough windows prove that decisions were exercised, not that the
chosen settings are globally optimal. Small candidate counts can underfill the
nominal width or prevent growth; use a sufficient backlog to study larger widths.
Fast scenarios can exhaust the round bound while still awaiting their 30 seconds
of evidence; do not lower that threshold merely to make a benchmark finish.

Resource observations include Torch allocator peaks, owner snapshots and
one-second samples of Linux process-tree RSS, global device memory/utilization
and packing disk bytes. Global device memory sums all GPUs, including driver,
display and unrelated processes; it is not exclusive service VRAM. Sampling may
miss short peaks. Linux RSS traverses children created by every process thread,
including service-worker compiler children; whole-process CPU time
includes its workers, preparation, in-process compilation and monitoring; it
excludes compiler-child CPU time and is not isolated orchestrator cost. Linux
procfs must expose child lists for each sampled process; unavailable
child lists produce a null RSS observation and disable process-tree availability
in that phase rather than reporting parent-only memory. Unsupported or failed
resource observations are explicit. Global-device availability requires at least
one usable memory/utilization sample; finding the executable is insufficient.
Sampler errors are retained. The tool measures service execution only: it does not
establish Pareto quality, CPU optimizer throughput or whole-search speedup.
