# Optimizing

For a staged search procedure, monitoring and completion criteria, see the
[agent campaign runbook](ai/runbooks/optimization_campaign.md) and its
[long-only template recipe](ai/case_studies/long_only_template_optimization.md).

Optimizer resume reuses checkpoint fitness only when its fixed evaluation contract matches the
current run. This includes backtest execution and data policies, backtest-consumed live settings,
fixed bot parameters, resolved coin overrides, and all fixed fine-tune anchor values. Candidate
vector values, `optimize.n_cpus`, and output paths are excluded. The shared resume check still
compares the complete `optimize.gpu` mapping, including GPU worker and hardware-sizing controls;
changing those controls currently requires a fresh run. Dataset selectors
`backtest.market_settings_sources`, `backtest.ohlcv_source_dir`, and `backtest.hlcvs_data_dir`
are compared as configured. Moving input directories therefore requires a fresh run even when
contents match; input paths are data selectors, not excluded output paths.
New result records include an `optimizer_evaluation_contract` snapshot. It binds fitness to
content hashes of the prepared candles, BTC prices, timestamps, selected scenario slices, and
market settings, plus the evaluator's transitive local Python imports, Python version, imported
package/dependency versions, and the verified loaded Rust source and artifact.
Changing a prepared input in place therefore requires a fresh run even when its directory is unchanged.
Rust artifact identity is deliberately strict: rebuilding the binary can require a fresh run even
from unchanged sources. The source identity follows imports conservatively, including imports
inside optional branches; an unresolvable dynamic import includes all local Python sources. Docs,
tests, and unrelated installed packages are excluded. Changes to an imported dependency can require
a fresh run even when the affected branch was not used. Hashing happens once during preparation
with bounded memory use.
Legacy results without historical evaluator and prepared-data evidence require a fresh run;
current files or current code cannot establish how their scores were produced. Every reconstructed
result in a compressed stream is validated, and GPU checkpoints carry the same evidence before
reusing native GPU metrics. Importing starting candidate configs for fresh evaluation remains supported.
Moving an override file without changing its resolved values does not change this contract.
Override files are resolved before CPU candidate evaluation and snapshotting. Every backend records
the effective external suite and scenario filter, with prepared concrete scenario dates and resolved
per-scenario coin patches. Prepared candidates use those same frozen patches, so later file edits
cannot change a running evaluation silently. Dynamic end-date tokens (`now`, `today`, empty, or null) resolve during config preparation; a later resolved
cutoff requires a fresh run instead of reusing scores from the earlier window.

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

Optional [configuration prefilters](config_prefilter.md) return graded constraints
before CPU or native GPU replay. All checks are disabled by default; each must be
explicitly enabled under `optimize.prefilter.checks`.

The canonical defaults live in `src/config/schema.py`. The example config
`configs/examples/default_trailing_martingale_long.json` mirrors those defaults exactly. For the
recommended config workflow, see [Config Workflow](config_workflow.md).

Optimization requires the full install profile:

```bash
pip install -e ".[full]"
```

## Session identity and output names

For exponential recency weighting, select `cagr_fraction_strategy_equity_ew`.
`backtest.ew_half_life_days` resolves to 180 days when omitted or null; an explicit value must be
positive and finite. CPU and native GPU use the same calendar-weighted minute log-growth
[definition](metrics.md). One half-life applies across the campaign. Half-life and metric definition
version participate in evaluation identity; changing either requires fresh evaluation. Old
candidate configurations may still seed that fresh run.

Optimizer sessions use the [shared session naming format](backtesting.md#backtest-results)
under `optimize_results/`. Small suites include their actual coin names and the number of
prepared scenarios. The deterministic setup fingerprint includes effective settings after CLI
overrides, fixed evaluation policy, optimizer bounds/objectives/backend settings, resolved dates,
actual prepared data and evaluator implementation, and selected starting-config contents in
consumption order. Starting configs are frozen in a bounded-memory disk snapshot before hashing
and execution; filenames, old metrics and output paths are not seed identity.

An unspecified `optimize.seed` is resolved to a concrete random seed before a fresh search.
It is recorded in `session.json` and saved candidates. New sessions with different actual seeds
have different setup fingerprints. An explicit seed can be used to compare otherwise equivalent
setups. `--resume` retains the original output directory and restores its recorded seed when the
input seed is unspecified; existing checkpoint/evaluation compatibility checks still apply.
Legacy sessions without `session.json` keep their original seed policy. Fresh runs always get a
new random run ID, even when their setup fingerprints match. Full SHA-256 setup fingerprints,
run IDs and canonical setup inputs are available in `session.json`.

Fresh GPU runs without `-t/--start` initialize every population member randomly, matching the
CPU backends. The input config supplies fixed and non-optimized values but is not implicitly
inserted as a candidate. Explicit starting configs remain supported, and `--resume` restores
the saved population rather than starting a new one.

## Running Optimization

An authored v9 config searches only its explicitly listed domains. Missing bounds fix the
corresponding normalized numeric genes; they do not inherit ranges from the schema template.
Explicit `enable_overrides` relationships can still derive effective target values from searched
sources, even when a target's own gene is omitted or fixed. Review these relationships with bounds;
a fixed independent gene does not imply a fixed derived result.
Use `passivbot tool optimizer-bounds input.json search.json` to write recommended ranges explicitly.
Running `passivbot optimize` without a config selects the complete starting profile, whose visible
bound ranges remain searched. Existing fixed/runtime/scenario and anchor precedence still applies;
anchors and seeds are clamped to the effective domains, including omitted fixed coordinates.

Normalized Forager weights must be searched together when they have positive participating
coordinates. A partial weight fine-tune, scenario pin, or fixed positive sibling fails with an actionable
group diagnostic; either select the entire group or freeze it. A missing zero coordinate is
permitted only when it remains zero through normalization, including the all-zero fallback.

```bash
passivbot optimize [path/to/config.json]
```

- Defaults to the in-code schema in `src/config/schema.py` if no config is specified
- Use existing configs as starting points: `--start path/to/config(s)`
- Enable suite scenarios defined in `backtest.scenarios` with `--suite [y/n]` (omit value to enable)
- Layer additional scenario definitions via `--suite-config path/to/file.json`

The canonical default profile keeps `backtest.suite_enabled = false`, so optimize runs are
single-scenario by default unless you explicitly enable suite mode.

Example:
```bash
passivbot optimize configs/examples/default_trailing_martingale_long.json --start configs/starting_pool/
```

Wallet exposure shortcuts `-ltwel` and `-stwel` set
`optimize.bounds.long.risk.total_wallet_exposure_limit_ratio` and the short equivalent.
Use a single value to fix exposure (for example `-ltwel 0`), or `LOW,HIGH[,STEP]`
to set its search range. In live and backtest commands these shortcuts still set bot values.

Most config parameters can be modified via CLI. `passivbot optimize -h` for more info.

### Scoring Objectives

`optimize.scoring` defines the metrics the optimizer tries to improve. The canonical form is a
list of objective objects:

```json
"scoring": [
  {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
  {"metric": "drawdown_max_fraction_strategy_equity", "goal": "min"}
]
```

- `metric` is a canonical backtest metric name.
- `goal: "max"` means higher raw metric values are better.
- `goal: "min"` means lower raw metric values are better.

The older shorthand form is still accepted for built-in metrics with known default goals:

```json
"scoring": ["cagr_fraction_strategy_equity", "drawdown_max_fraction_strategy_equity"]
```

Passivbot normalizes that shorthand to the canonical object form during config formatting. Use the explicit object form to choose a direction. Selectors must belong to the core catalog;
new metrics require an implementation change.

### Backend Selection

Passivbot supports these optimizer backends:

- `optimize.backend: pymoo`
  - Uses pymoo. This is now the default optimizer backend.
  - The default pymoo algorithm mode is `auto`: Passivbot uses `nsga2` when optimizing `3` or fewer objectives, and `nsga3` when optimizing more than `3`.
- `optimize.backend: gpu`
  - Uses native NVIDIA/CUDA or Apple Metal simulations with CPU-owned pymoo ask/tell evolution,
    scenario selection, scoring, limits and result processing. This backend remains
    experimental; see [the service contract](gpu_backtest_service.md#gpu-optimizer).

Example:

```json
{
  "optimize": {
    "backend": "pymoo",
    "pymoo": {
      "algorithm": "auto"
    }
  }
}
```

CPU and GPU search both retain NSGA-II and NSGA-III. The `deap` backend and its separate operator
controls are retired. Convert v8 configurations explicitly before starting a v9 search.

CPU checkpoints contain the search algorithm, population and random generator state together with
the critical search configuration and source/runtime/data evaluation identity. Resume attaches a
fresh evaluator and worker pool and permits a changed evaluation budget or worker count. It rejects
old bare pymoo or DEAP checkpoints and changed fitness inputs; use migrated saved configurations as
fresh seeds instead. A partial checkpoint write leaves the previous checkpoint intact.

### GPU Backend (Experimental)

<a id="apple-mps-gpu-backend-experimental"></a>

GPU optimization supports NVIDIA CUDA on Linux/WSL2 or Apple Silicon Metal on macOS.
Install the matching optional runtime. For NVIDIA:

```bash
python3 -m pip install -e ".[full,gpu-cuda]"
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
passivbot optimize configs/examples/ema_anchor.json --optimizer-backend gpu
```

A compatible NVIDIA driver is required. In WSL2 install that driver on Windows;
a system-wide CUDA toolkit is unnecessary. CPU optimization (`pymoo`) and
standalone CPU backtests/plots remain available without GPU dependencies.

For Apple Silicon on macOS:

```bash
python3 -m pip install -e ".[full,gpu-mps]"
python3 -c "import torch; print(torch.backends.mps.is_available())"
passivbot optimize configs/examples/ema_anchor.json --optimizer-backend gpu
```

The service selects Metal automatically on supported Macs. Both EMA Anchor and
Trailing Martingale use the native shared-account replay, including long/short,
suites and HSL. Metal uses float32 replay/EW arithmetic, integer UTC clocks and
the shared Rust finalizer for compact sufficient-statistic packets; metric observation state is
bounded per candidate. Saved fitness records a distinct Metal precision identity, so
CUDA and Metal checkpoints are not interchangeable. This backend is experimental;
fixture parity does not certify every long-history trading path.

The CPU prepares canonical configs, immutable scenarios and compact candidate requests.
One asynchronous service owns GPU packing, compilation, residency, scheduling, simulation
and requested metric reductions. Full GPU results directly supply canonical CPU scoring,
limits, selection, Pareto and result files. There is no CPU backtest validation or fallback
inside GPU optimization, including starting configs and resume. Independent
[GPU/CPU parity checks](gpu_parity.md) provide the development comparison tools.

EMA Anchor and Trailing Martingale use the shared-account multicoin replay, including
single-coin runs, long/short and hedged/one-way portfolios. The current ceiling is 64 coins
per prepared scenario. Other strategies, BTC collateral and continuous changes to
scenario-owned coin patches are explicit errors. Enabled HSL requires one-minute candles.
Requested metrics must belong to the code-owned GPU surface; unsupported names fail
rather than returning fabricated values. Float32 replay and accumulation can differ from CPU float64 results. See the
[feature and approximation inventory](plans/gpu_optimizer_inventory.md) and
[acceptance evidence](plans/gpu_optimizer_acceptance.md) for measured limits.

GPU metric eligibility is a policy, not a list of known calculation defects.
Exclusions can reflect approximation or cost, or favor normalized growth, activity
and downside-risk signals. Gain shares ADG's smoothed endpoint; ADG adds daily
normalization. Positive equity-balance divergence has no universally desirable
optimization direction; raw counts and absolute panic losses favor rates and loss
percentages, while realized-only growth can reward leaving losses unrealized.
Excluded metrics remain available to CPU optimization and standalone backtests.
Enabling one requires supported-strategy/topology parity, alias/scoring/limit
coverage, explicit tolerances and ranking/constraint evidence. Profile requested
outputs and reductions separately; eligibility alone does not establish a speedup.

CPU orchestration reuses `optimize.pymoo`, `optimize.population_size` and the existing
cohort meaning of `optimize.iters`. All deduplicated starting configs and initial parents
receive complete GPU evaluations. Within each cohort, requests are replenished and full
results/Pareto members are written as completions arrive. Evolution advances when the
required complete offspring are collected. Fine-tune anchors and supported optimizer
runtime overrides use canonical CPU preparation; fixed policies remove ineffective genes.

For GPU-selected configs, the legacy `optimize.gpu.population_size` migrates to
`optimize.population_size` when the latter is omitted, null or an auto alias. A concrete
new value wins, with a warning when the legacy field is discarded. Invalid legacy values
are rejected; positive legacy values retain the former minimum of eight parents. CPU
backend configs do not use this migration. Save the normalized config to remove the
obsolete field. NSGA-III can still raise the actual population to fit its reference directions.

#### Execution settings

```json
"gpu": {
  "batch_size": null,
  "checkpoint_interval_seconds": 5.0,
  "max_dispatch_candidate_bars": null,
  "screening": {
    "scenarios": [],
    "survival_fraction": 0.1,
    "min_survivors": 64
  },
  "tuning_mode": "auto"
}
```

An omitted, null or `"auto"` batch width lets the service tune from completed production
work. A positive explicit width fixes its ceiling. `tuning_mode: "off"` disables batch-width
and request-accumulation tuning and uses width 64 if none is supplied. `"refresh"` currently
starts fresh like `"auto"`; advisory measurements are run-local. Prepared work/history
budgets can reduce physical widths further. Native CUDA history chunks still adapt in
`"off"` mode, targeting bounded command durations without a preemption guarantee.
CPU consumption cadence also adapts independently. These controls change execution only;
they never tune simulation precision or search policy.

Automatic native CUDA sessions size replay scratch when a dataset's replay owner is
created, after its market tensors are resident. The budget includes per-candidate
history and shared metric workspace, and reserves the
larger of 1 GiB or 20% of total VRAM when feasible. Under pressure it uses 35% of free
VRAM, with room for one candidate if that candidate fits. Directional owners share
the allowance. Before a physical batch changes width or factual storage shape,
automatic CUDA owners release all completed scratch so replacement histories do
not overlap retained histories. Unchanged shapes reuse their buffers. Switching
datasets releases the old owner's scratch and samples free VRAM again; active owners are never resized. Explicit batch widths, tuning mode
`"off"`, and Metal retain their existing scratch budgets. Increased capacity does
not eagerly allocate scratch or force a wider batch: dispatch limits and completed
throughput evidence still govern actual widths. The allowance leaves room for
factual histories to grow during a resident owner's lifetime.

Legacy CPU exact-worker, validation-count, drift-probe and screened-seed controls have
been removed. Population size belongs to `optimize.population_size`. Old screening/
validation checkpoints and earlier experimental `gpu_native` checkpoints cannot resume;
start a fresh run with saved configs as starting inputs. Current checkpoints retain CPU
algorithm/cohort state, complete fitness, anchors and separate partial screening evidence.
SIGINT stops admission, drains completed work and saves a usable checkpoint. Some
unfinished GPU work may repeat after resume; already durable Pareto members are preserved.

#### GPU scenario screening

`screening.scenarios` selects labels from the prepared backtest suite. Later offspring
may first receive GPU evaluations on that subset, then CPU feasibility/Pareto-diversity
selection promotes survivors for the remaining scenarios. Retain the larger of
`ceil(cohort_size * survival_fraction)` and `min_survivors`, capped at cohort size.
Explicit objective/limit scenarios must be included. Only full-suite survivors receive
fitness, result records and Pareto admission. Completed scenario rows can be reused
at promotion without reusing partial scores as full fitness.
Suite-aggregated EW objectives and limits reduce the selected scenarios during screening;
full evaluation still requires all scenarios and available EW samples from every source.

Empty screening labels disable the partial stage. Selecting all scenarios or retaining
all offspring bypasses it. Screening reduces full evaluations within `iters`; it does
not extend the generation budget. Single-scenario runs cannot use suite screening.

The [prepared service guide](gpu_backtest_service.md) documents the request boundary,
residency, tuning and interruption contract. [Service benchmarks](gpu_service_benchmark.md)
measure execution/resource behavior, while [cohort benchmarks](gpu_cohort_benchmark.md)
compare numerical, feasibility and ranking effects outside optimization. These tools
must distinguish first-use preparation/compilation from warm replay.

### Pymoo Configuration

Pymoo-specific settings live under `optimize.pymoo`:

```json
{
  "optimize": {
    "backend": "pymoo",
    "population_size": null,
    "pymoo": {
      "algorithm": "auto",
      "shared": {
        "crossover_distribution_index": 20.0,
        "crossover_probability_per_variable": 0.5,
        "mutation_distribution_index": 20.0,
        "mutation_probability": "auto",
        "mutation_probability_per_variable": "auto",
        "eliminate_duplicates": true
      },
      "algorithms": {
        "nsga2": {},
        "nsga3": {
          "ref_dirs": {
            "method": "das_dennis",
            "n_partitions": "auto"
          }
        }
      }
    }
  }
}
```

#### NSGA-III, Reference Directions, and `das_dennis`

`nsga3` is a many-objective evolutionary algorithm. Unlike NSGA-II, it does not rely only on
crowding distance to spread candidates across the Pareto front. Instead, it uses a set of
reference directions in objective space and tries to keep the population distributed across them.

Passivbot uses the `das_dennis` method to generate those reference directions. This is the
standard simplex-partition method for NSGA-III and is a sensible default for Passivbot optimize
runs.

The main NSGA-III-specific knob is:

- `optimize.pymoo.algorithms.nsga3.ref_dirs.n_partitions`
  - Controls how fine the reference-direction grid is.
  - Higher values generate more reference directions, which increases diversity resolution but also
    makes each generation heavier.
  - With an 11-objective scoring set, common reference-direction counts are:
    - `n_partitions = 2` -> `66`
    - `n_partitions = 3` -> `286`
    - `n_partitions = 4` -> `1001`
  - Default is `"auto"`. For an 11-objective setup, Passivbot currently resolves that to
    `n_partitions = 3`, which gives `286` reference directions.
  - In auto mode, Passivbot chooses the largest Das-Dennis grid whose reference-direction count
    fits within the resolved NSGA-III population budget. This preserves the NSGA-III invariant that
    `population_size >= reference directions`.

- `optimize.population_size`
  - For `pymoo` + `nsga3`, `null` means “auto”.
  - In that case Passivbot uses the default NSGA-III population budget of `500`, then resolves
    auto reference directions to the finest Das-Dennis grid that fits inside that budget.
  - For an 11-objective setup, that means `population_size = 500` with `286` reference
    directions.
  - For a 10-objective setup, auto keeps `population_size = 500` and uses `220` reference
    directions (`n_partitions = 3`) because the next grid would require `715` reference directions.
  - For `pymoo` + `nsga2`, `null` means “auto” and currently resolves to `250`.
  - Set an explicit integer when you want to change the per-generation evaluation budget and the
    maximum auto reference-direction grid size. For example, `population_size = 1000` allows the
    10-objective auto resolver to use `715` reference directions (`n_partitions = 4`).

#### Shared Pymoo Hyperparameters

The `shared` block controls the SBX crossover and polynomial mutation operators used by both
`nsga2` and `nsga3`.

Current meaning of the main pymoo knobs:

- `optimize.pymoo.algorithm`
  - `auto`, `nsga2`, or `nsga3`.
  - Default is `auto`.
  - `auto` chooses `nsga2` when `len(optimize.scoring) <= 3`, otherwise `nsga3`.
  - Use explicit `nsga2` or `nsga3` only when you want to override that default selection.
- `optimize.pymoo.shared.crossover_probability_per_variable`
  - Per-variable SBX crossover probability.
  - Higher values mix more parameters between parents on each crossover.
  - Default `0.5` is a conservative middle ground for Passivbot's parameter space.
- `optimize.pymoo.shared.crossover_distribution_index`
  - SBX distribution index.
  - Higher values keep offspring closer to the parents; lower values explore more aggressively.
  - Default `20` is a standard conservative setting and is usually a good starting point.
- `optimize.pymoo.shared.mutation_probability`
  - Per-individual polynomial-mutation probability; `"auto"` means `1 / n_params`.
  - The offline v8 converter reports any replacement of the retired `mutation_prob_var` key.
    Normal v9 loading accepts only the canonical control.
- `optimize.pymoo.shared.mutation_probability_per_variable`
  - Conditional per-variable mutation probability; `"auto"` means `min(0.5, 1 / n_params)`,
    matching pymoo's historical implicit default.
  - The probability of selecting a coordinate for mutation is the product of the two gates. To experiment
    with approximately one coordinate selected for mutation per offspring when `n_params >= 2`,
    set `mutation_probability: 1.0` and
    leave this control on `"auto"`. This is a stronger mutation regime, not the default.
  - `n_params` is the CPU problem's vector length or the GPU problem's active vector length.
- `optimize.pymoo.shared.mutation_distribution_index`
  - Polynomial-mutation distribution index.
  - Higher values make smaller, more local mutations.
  - Default `20` keeps mutation fairly local, which is usually appropriate for expensive
    backtests.
- `optimize.pymoo.shared.eliminate_duplicates`
  - Skip duplicate candidates before wasting a full backtest on them.
  - Default `true`.
  - Recommended for Passivbot because each evaluation is relatively expensive.
- `optimize.pymoo.algorithms.nsga3.ref_dirs.method`
  - Reference-direction generator for NSGA-III.
  - Currently `das_dennis`.

Recommended defaults for typical Passivbot runs:

- Use `optimize.backend: pymoo` with `optimize.pymoo.algorithm: auto`.
- Keep `mutation_probability: "auto"`.
- Keep `crossover_distribution_index: 20` and `mutation_distribution_index: 20` unless you have a specific reason to make
  variation much more local or much more aggressive.
- Keep `crossover_probability_per_variable: 0.5` unless you have evidence that crossover is either too timid or
  too disruptive for your runs.
- Leave `population_size: null` and `ref_dirs.n_partitions: "auto"` for default pymoo behavior:
  NSGA-II resolves null population size to `250`, while NSGA-III uses a default population budget
  of `500` and chooses the finest compatible reference-direction grid.
- Keep `pareto_max_size: 1000` unless archived front updates become a measured bottleneck for your
  machine or workflow.
- If you need more or less exploration pressure, change `population_size` first. It is the main
  budget/coarseness knob for NSGA-III auto reference directions. Use explicit `n_partitions` only
  when you specifically want to force a grid resolution.

Practical interpretation for the default shared block:

```json
"shared": {
  "crossover_distribution_index": 20,
  "crossover_probability_per_variable": 0.5,
  "eliminate_duplicates": true,
  "mutation_distribution_index": 20,
  "mutation_probability": "auto",
  "mutation_probability_per_variable": "auto"
}
```

- `crossover_distribution_index: 20`
  - conservative crossover; offspring stay fairly close to parents
- `crossover_probability_per_variable: 0.5`
  - within parent pairs selected for crossover, each parameter has a 50% chance of participating in crossover
- `mutation_distribution_index: 20`
  - conservative mutation; most mutations are relatively local
- `mutation_probability: "auto"`
  - select each offspring for mutation with probability `1 / n_params`
- `mutation_probability_per_variable: "auto"`
  - within selected offspring, mutate each coordinate with probability `min(0.5, 1 / n_params)`
- `eliminate_duplicates: true`
  - do not spend backtests on duplicate candidates

These defaults are intentionally conservative. For most Passivbot optimize runs, scoring choice,
suite design, and evaluation budget matter more than fine-tuning these operator settings.

Algorithm selection under the default `auto` mode:

- `1` to `3` objectives -> `nsga2`
- `4+` objectives -> `nsga3`

That means the default 11-objective Passivbot template uses `nsga3`, while small custom scoring
lists automatically fall back to `nsga2`.

### Candle Interval

For faster optimization runs, you can aggregate 1-minute data into coarser candles before the
backtest loop runs. This reduces the number of bars processed per iteration.

Set `backtest.candle_interval_minutes` to a value greater than 1:

```json
{
  "backtest": {
    "candle_interval_minutes": 5
  }
}
```

Trade-offs:

- Intra-interval fill ordering is lost (fills occur only at the aggregated bar boundaries).
- Metrics are still time-correct because analysis uses timestamps rather than bar indices.
- Native GPU optimization supports aggregated candles with HSL disabled. Enabled HSL requires
  one-minute candles. Minute-denominated strategy/Forager EMA spans and elapsed-time cooldowns
  retain their time meaning. GPU results supply fitness directly; compare interval-sensitive
  metrics separately with the [standalone parity tool](gpu_parity.md).

### Fine-Tuning Specific Parameters

When you only want to adjust a handful of parameters and keep everything else fixed, use
`--fine_tune_params` (short: `-ft`). Provide a comma-separated list of dotted config-path
selectors to keep tunable; all other bounds are locked to their current config values
before the run starts. Selectors match full path segments by prefix or suffix, not partial
substrings.
The leading `bot.` may be omitted for side-local paths, so `long.risk` is equivalent to
`bot.long.risk`. A `*` segment may be used as a one-segment wildcard, for example
`*.strategy.close` matches both long and short active-strategy close params. A leaf selector
such as `we_excess_allowance_fraction` matches every bound whose config path ends with that
parameter name.

```bash
passivbot optimize configs/examples/default_trailing_martingale_long.json \
  --fine_tune_params long.risk,long.forager,long.unstuck
```

Behind the scenes the optimizer sets every unlisted bound to `[value, value]`, so the GA
can mutate only the parameters you specified. Bounds for the listed parameters remain as
configured. The optimizer logs each selector expansion on separate sorted lines before
the run starts.

`optimize.fixed_params` provides the config-file equivalent for selectors that should always
be fixed to their current config values. It uses the same dotted path matching as
`--fine_tune_params`.

Useful examples:

```json
"optimize": {
  "fixed_params": ["long.strategy"]
}
```

This fixes every optimizer bound under `bot.long.strategy.<active_strategy>`, such as all
`trailing_martingale` entry/close thresholds, retracements, EMA spans, and volatility spans.
It does not fix `bot.long.risk`, `bot.long.forager`, or `bot.long.unstuck`.

```json
"optimize": {
  "fixed_params": ["long.strategy.close", "long.hsl"]
}
```

This fixes only the active long strategy's close subtree plus long HSL bounds.

Internally, `--fine_tune_params` and `optimize.fixed_params` are merged into one effective
fixed-parameter set before bounds are collapsed.

### Polishing Around A Selected Config

Use `--polish-bounds-fraction` to narrow every configured optimizer bound around the current config
value before the run starts:

```bash
passivbot optimize path/to/config.json --polish-bounds-fraction 0.25
```

By default this keeps the polished bounds inside the original `optimize.bounds` domain and
leaves fixed bounds fixed. `--polish-bounds-mode` changes that policy:

- `clamp`: default behavior; intersect polished bounds with the original bounds.
- `override-tunable`: allow tunable bounds to escape the original bounds; fixed bounds stay fixed.
- `override-all`: allow tunable bounds to escape the original bounds and expand fixed bounds too.

Polish uses a scale-one relative half-width: `0.25` means ±25%, producing
`[value * (1 - half_width_fraction), value * (1 + half_width_fraction)]`. A current
value of `0.0` therefore remains fixed at `[0.0, 0.0]`.

`optimize.fixed_params` is applied after polish. That makes it the right way to polish all
bounds while pinning selected parameters to the config value:

```bash
passivbot optimize path/to/config.json \
  --polish-bounds-fraction 0.25 \
  --polish-bounds-mode override-all \
  --optimize.fixed_params long.risk.max_positions,long.risk.total_wallet_exposure_limit_ratio
```

When `--fine_tune_params` is combined with `--start`, the base optimizer config remains the run
policy. Anchor configs provide values only for optimizer-bound bot parameters that are fixed by the
anchor plan; boolean toggles and other non-bound runtime policy fields such as
`bot.long.hsl.enabled` continue to come from the base config or explicit runtime overrides. Seed and
anchor values outside `optimize.bounds` are clamped into bounds during seed loading and logged in
aggregate with counts, source examples, key/path, original value samples, bound, and clamped value.

`optimize.fixed_runtime_overrides` is different: it overrides runtime config values only during
optimize evaluations, without changing the stored/live config value. This is useful for
operator-risk settings such as:

```json
"optimize": {
  "fixed_runtime_overrides": {
    "bot.long.hsl.cooldown_after_red_minutes": 60.0,
    "bot.short.hsl.cooldown_after_red_minutes": 60.0
  }
}
```

These overrides set the finite cooldown after a qualifying flatten during optimizer
evaluations. Portfolio mode uses `bot.portfolio.hsl.cooldown_after_red_minutes` instead of side
paths. Inspect strategy drawdown and recovery with `drawdown_max_fraction_strategy_equity`,
`ulcer_index_fraction_strategy_equity` and `peak_recovery_max_days_strategy_equity`. HSL loss
limits use `hsl_panic_episode_loss_max_fraction` and `hsl_panic_episode_loss_mean_fraction`;
`hsl_red_time_fraction` includes reported RED cooldown. Detailed HSL reports remain optional
backtest artifacts. Use exported account equity and fills for independent economic comparisons.

Migrate v8 starting configurations explicitly before using them with v9. The main optimizer
config owns HSL activation, memory and cooldown policy; numeric starting suggestions do not
restore old policy or authorize old fitness. Every seed is freshly evaluated, and old
checkpoints cannot resume under the v9 risk contract. The main config requires migration
even when its HSL policies are disabled. See the [migration guide](v8_to_v9_migration.md).

When you provide many starting configs to a CPU optimizer, seed evaluations use a bounded queue
with at most one pending candidate per configured CPU worker. All provided starting configs are
freshly evaluated before pymoo trims them to its initial population. This bounds memory use for
large seed pools, including suites with larger per-candidate metrics payloads.

### Optimizer Suites

The optimizer reuses the backtest suite configuration and allows every candidate to
be evaluated across multiple scenarios before scoring. Each scenario can override coins,
date ranges, exchanges, `coin_sources`, and bot parameters via `overrides`. The optimizer
prepares a single shared dataset that covers the union of the requested data so additional
scenarios add minimal overhead.

Key fields (directly under `backtest`):

- `backtest.suite_enabled`: master toggle for suite mode, can also be set with `--suite [y/n]`
- `backtest.scenarios`: list of scenario dictionaries (same schema as backtest scenarios)
- `backtest.reducer`: default per-metric reducers for combining scenario results (default:
  `{"default": "mean"}`)

`optimize.objective_scenario` sets the default scoring basis. Set it to a unique scenario label
(commonly `base`) to read objectives from that scenario by default, or leave it `null` to use suite
aggregation by default. The equivalent global CLI override is `--objective-scenario LABEL`.
Each `optimize.limits` entry independently uses suite aggregation by default or may select one
named scenario.

Each object-form `optimize.scoring` entry may override the default:

```json
{
  "backtest": {
    "reducer": {"default": "mean"}
  },
  "optimize": {
    "objective_scenario": "base",
    "scoring": [
      {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
      {
        "metric": "pain_index_fraction_strategy_equity",
        "goal": "min",
        "scenario": null
      },
      {
        "metric": "peak_recovery_max_days_strategy_equity",
        "goal": "min",
        "scenario": null,
        "reducer": "max"
      }
    ]
  }
}
```

An omitted `scenario` inherits `optimize.objective_scenario`, a named value selects that scenario,
and explicit `null` selects suite reduction. A reduced objective without its own `reducer`
uses the metric-specific `backtest.reducer` rule and then its `default`; an explicit `reducer`
may be `mean`, `min`, `max`, `std`, or `median`. A named scenario and `reducer` are mutually
exclusive. This permits representative base-period performance objectives alongside mean or
worst-case stress objectives while limits retain their independent suite-wide contract.
Each metric may still appear only once in `optimize.scoring`; scoring the same metric from multiple
bases is not supported in this first version.

Limits may combine a scenario-specific threshold with a separate suite-wide threshold for the same
metric:

```json
"limits": [
  {
    "metric": "drawdown_max_fraction_strategy_equity",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_max_fraction_strategy_equity",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  }
]
```

The first limit reads only `base`; the second reads the maximum across scenarios. Their violations
are evaluated independently and their penalties accumulate. A named `scenario` and `reducer` are
mutually exclusive. An omitted or null `scenario` uses suite aggregation; with no explicit `reducer`,
the limit falls back through the metric-specific and default `backtest.reducer` rules.

Suite mode is opt-in. The default schema/example config does not enable it automatically.

During evaluation the optimizer records:

- Per-scenario combined metrics (the same mean/min/max/std set produced by standalone
  backtests). These are exposed on each individual as `<label>__{metric}`.
- Aggregated metrics computed with the `backtest.reducer` rules (default `mean`).
  These values feed `optimize.limits` and suite-reduced `optimize.scoring` entries.

`reducer` is canonical across suite configuration, scoring entries, limits, CLI entries, and
serialized configs. Normal v9 loading requires that name. The explicit released-v8 converter
handles older `aggregate`, `stat`, and `scenario_stat` fields (`field` for limits),
rejecting conflicting values before writing the canonical config.
Historical Pareto artifact payloads keep their existing `suite_metrics.aggregate`, `aggregated`,
and `stats` keys and remain readable.

See [Suite Examples](suite_examples.md) for practical scenario configurations including exchange
comparisons, date range testing, and parameter sensitivity analysis.

Result directories stay under `optimize_results/`, but the coin portion of the folder
name switches to `suite_{n}_coins` to make suite runs easy to locate.

Each evaluation written to disk now includes a compact `suite_metrics` payload:

```json
"suite_metrics": {
  "aggregate": {
    "aggregated": {"cagr_fraction_strategy_equity": 0.0012, "...": "..."},
    "stats": {"cagr_fraction_strategy_equity": {"mean": 0.0011, "min": 0.0008, "max": 0.0014, "std": 1.5e-4}}
  },
  "scenarios": {
    "scenario_a": {"stats": {"cagr_fraction_strategy_equity": {"mean": 0.0012, "min": 0.0011, ...}}},
    "scenario_b": {"stats": {"cagr_fraction_strategy_equity": {"mean": 0.0009, ...}}}
  }
}
```

Pareto members store a compact metrics payload under `metrics.stats` (and `suite_metrics` when suite
mode is enabled) instead of the older `analyses_combined` / per-exchange analysis blocks.

## Optimization Process

- Uses a multi-objective evolutionary backend (`pymoo` or experimental `gpu`)
- `pymoo` defaults to NSGA-III for many-objective runs, with NSGA-II still available explicitly
- Backtests across historical OHLCV data
- Uses multiprocessing with shared memory for reduced RAM load
- Maintains Pareto front of best-performing configurations
- Enforces constraints via `optimize.limits`
- Optimizes for multiple metrics via `optimize.scoring`
- Avoids duplicates through hash tracking and perturbation
- Logs starting-config dedup statistics at startup, including how many raw configs collapsed after quantization and how many extra TWEL-scaled variants survived

Per-coin warmup inside an optimizer run is sized from `optimize.bounds`, not from the `bot.*` template values. The optimizer treats each optimized field as if it were at its upper bound when computing how much history each coin needs, so every individual evaluated in the same run trades on an identical window. Standalone `passivbot backtest <config.json>` is unaffected and still sizes warmup from whatever bot values the config you hand it contains.

## Output Structure

Each optimization run creates a directory:
```
optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{analysis_duration_days}days_{coin_label}_{hash}/
```

Contents:
- `all_results.bin`: Binary log of all evaluated configs (msgpack format)
- `pareto/`: JSON files for Pareto-optimal configurations
  - Named `{hash}.json`
  - Files are added/removed over time as the Pareto front updates and is pruned to `optimize.pareto_max_size`

## Analyzing Results

Full analysis is included in each member of the Pareto front. Two helper tools are available:

```bash
# Single-candidate selector
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto

# Interactive dashboard (recommended)
passivbot tool pareto-dash --data-root optimize_results

# Static matplotlib plotter
python3 src/pareto_store.py optimize_results/.../pareto/
```

`passivbot tool pareto` is the quickest way to promote one config from a large Pareto set. It
loads the JSON artifacts, optionally filters them with `--limit` / `--limits`, then chooses one
candidate using a named decision rule. It accepts either a `pareto/` directory, an optimize run
directory, or no path at all, in which case it falls back to the newest local
`optimize_results/.../pareto` by lexicographic run-directory name, considering only runs whose
`pareto/` subdirectory contains at least one `*.json` candidate. It also shows the retained
front's ideal point for the active objectives. Recommended workflow:

1. apply hard filters with `--limit`
2. use `-m reference` if you already know your target ADG / drawdown / recovery regime
3. otherwise start with the default `ideal` selector
4. switch to `-m knee` if you specifically want a balanced-compromise heuristic
5. use `--show-top N` to inspect the shortlist before promoting one config
6. use `--json` if you want to script the selection

Available methods:

- `knee`: approximate balanced compromise point
- `reference`: closest to user-specified targets via `--target metric=value`
- `ideal`: closest to the observed ideal point; default
- `utility`: weighted scalarization via `--weight metric=value`
- `lexicographic`: strict objective priority via `--priority metric_a,metric_b,...`
- `outranking`: simplified PROMETHEE-style pairwise ranking

These are practical selection heuristics for large Passivbot Pareto fronts rather than fully
formal MCDM implementations. For most real runs, `knee`, `reference`, and `utility` are the most
useful methods.

`-o` / `--objectives` can also reference stored metrics outside the original `optimize.scoring`
list, for example `sharpe_ratio_strategy_equity`, as long as that metric is present in the
saved Pareto JSON and Passivbot has a known default min/max direction for it.

Example:

```bash
passivbot tool pareto \
  -o sharpe_ratio_strategy_equity,cagr_fraction_strategy_equity,peak_recovery_max_days_strategy_equity \
  -m ideal
```

`pareto_dash.py` scans one or more optimization runs and launches a Plotly Dash app with:

- Scatter/histogram views for any metrics or objectives
- Defaults to the metrics listed in `config.optimize.scoring`, so the scatter/histogram
  immediately highlight your optimization objectives when the app loads
- Scenario-aware box plots (per-metric distributions broken down by suite scenario)
- Correlation heat maps and parameter-vs-metric scatter plots for quick diagnostics
- Streaming history chart sourced from `all_results.bin`
- CSV export of the current run's dataset for offline analysis

Use the full install profile (`pip install -e ".[full]"`) if the dashboard dependencies are not already present.
The legacy `pareto_store.py` script still supports quick 2D/3D matplotlib plots if a GUI
isn't needed.

## Optimization Limits

To enforce constraints during optimization, populate `optimize.limits` with a list of limit
objects. Each object describes when to penalize a result:

- `metric`: canonical metric name (e.g. `drawdown_max_fraction_strategy_equity`, `loss_profit_ratio`, `cagr_fraction_strategy_equity`).
- `penalize_if`: comparison operator. Use `<`, `<=`, `>`, `>=`, `==` (or aliases like `less_than`
  / `greater_than`), `outside_range`
  to keep a metric within `[low, high]`, or `inside_range` to forbid a band.
- `value`: numeric threshold for `<`/`>` limits.
- `range`: `[low, high]` for the range-based operators.
- Optional `scenario`: select one named suite scenario. Omitted or explicit `null` uses suite
  aggregation. Named scenarios cannot be combined with `reducer`.
- Optional `reducer`: for suite aggregation, override the statistic to compare against (`min`, `max`,
  `mean`, `std`, or `median`).
  Without `reducer`, Passivbot resolves the metric through `backtest.reducer`: first a
  metric-specific reducer rule, then `backtest.reducer.default`, then `mean`.

Example:

```json
"limits": [
  {
    "metric": "drawdown_max_fraction_strategy_equity",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_max_fraction_strategy_equity",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  },
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "cagr_fraction_strategy_equity", "penalize_if": "<", "value": 0.2, "reducer": "mean"}
]
```

CLI overrides can replace the full limit set with the same JSON/HJSON payload:

```bash
passivbot optimize --limits '[{"metric":"drawdown_max_fraction_strategy_equity","penalize_if":">","value":0.35}]'
```

For quicker one-off edits, use repeatable `--limit` entries. The symbolic scalar operators
in `--limit` are written as keep conditions, matching `pareto_store.py` filtering:
- `--limit 'cagr_fraction_strategy_equity > 0.2'` means keep only results with `cagr_fraction_strategy_equity > 0.2`
- `--limit 'drawdown_max_fraction_strategy_equity <= 0.35'` means keep only results with `drawdown_max_fraction_strategy_equity <= 0.35`

```bash
passivbot optimize \
  --clear-limits \
  --limit 'drawdown_max_fraction_strategy_equity <= 0.35' \
  --limit 'drawdown_max_fraction_strategy_equity <= 0.5 scenario=base' \
  --limit 'backtest_completion_fraction>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'cagr_fraction_strategy_equity > 0.2 reducer=mean'
```

You can also combine both forms. `--limits` loads a whole list first, and each `--limit`
appends one more canonical entry:

```bash
passivbot optimize \
  --limits '[{"metric":"drawdown_max_fraction_strategy_equity","penalize_if":">","value":0.35}]' \
  --limit 'peak_recovery_max_days_strategy_equity <= 21'
```

Semantics:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` is repeatable and appends one parsed entry to that replacement set.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

Limit violations produce penalties in optimizer engine space. A candidate satisfying all limits
ranks ahead of a candidate with violations. Every numeric selector accepts lower and upper
penalties independently or together. Selected unavailable values make the candidate infeasible;
they cannot pass a limit through an empty or missing value.

## Performance Metrics

Rust owns the [core metric catalog and formulas](metrics.md): 24 quantities, with separate long
and short selectors for position utilization. Objectives and enabled limits determine which
accumulators run for each optimizer candidate. Regular backtests report the core catalog.
Retired metrics, alternate units, automatic side copies and USD/BTC presentation variants are
not emitted. Strategy-equity metrics exclude direct collateral revaluation and conversion costs.

Use CAGR for endpoint growth, or CAGR EW for recency-weighted growth. Sharpe, Sortino, Omega and
volatility use complete UTC daily returns. Drawdown, Ulcer, Pain, exposure and recovery retain
minute observations, including flat periods. Passivbot uses exactly 365.25 days per year and
subtracts no risk-free return. Maximum drawdown is primarily a limit; Ulcer and Pain are useful
continuous risk objectives when paired with a growth or activity condition.

Authored selectors use the exact catalog name. Put `reducer` in the objective or limit object,
or in `backtest.reducer`; do not append a reducer, unit or denomination suffix. Only requested
outer reductions are computed, using every required exchange and scenario. Raw metric values
and explicit unavailable/infinite statuses are persisted with their definition version.

After raw reductions and limit comparisons, a minimization engine receives
`(2 / pi) * atan(sign * value)`, with `sign = -1` for maximization and `+1` for minimization.
Genuine infinite objectives map to finite endpoints. This monotonic adapter preserves ordering
and Pareto dominance but changes distances and crowding. Saved fitness from an earlier metric
or evaluator definition requires fresh evaluation.

## Utilities

Loading results programmatically:
```python
from opt_utils import load_results

for config in load_results("optimize_results/.../all_results.bin"):
    # Work with config
```

### Monitoring Optimizer Memory Usage

The script `src/tools/capture_optimize_memory.py` samples an active optimizer process tree and
host memory state into a JSON report. This is useful when validating that shared-memory datasets
are behaving as expected on a given machine.

```bash
PYTHONPATH=src python3 src/tools/capture_optimize_memory.py --wait --output tmp/optimize_memory.json
```

Use `--pid <pid>` to monitor a specific optimizer process instead of waiting for the newest
matching process.

### Profiling Suite Optimizer Evaluation

Set `PASSIVBOT_OPTIMIZE_PROFILE=1` to log per-candidate suite evaluator timings:

```bash
PASSIVBOT_OPTIMIZE_PROFILE=1 passivbot optimize path/to/config.json
```

The profile is opt-in and emits `[opt-profile]` INFO lines for suite evaluation phases such as
scenario config assembly, runtime compilation, payload construction, Rust backtest execution,
metric combining, aggregate metric building, and Pareto result recording. When enabled, candidate
metrics also include a `profile` block so retained Pareto configs can be inspected after the run.
