# Optimizing

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

The canonical defaults live in `src/config/schema.py`. The example config
`configs/examples/default_trailing_martingale_long.json` mirrors those defaults exactly. For the
recommended config workflow, see [Config Workflow](config_workflow.md).

Optimization requires the full install profile:

```bash
pip install -e ".[full]"
```

## Running Optimization

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

Most config parameters can be modified via CLI. `passivbot optimize -h` for more info.

### Scoring Objectives

`optimize.scoring` defines the metrics the optimizer tries to improve. The canonical form is a
list of objective objects:

```json
"scoring": [
  {"metric": "adg_strategy_eq", "goal": "max"},
  {"metric": "drawdown_worst_strategy_eq", "goal": "min"}
]
```

- `metric` is a canonical backtest metric name.
- `goal: "max"` means higher raw metric values are better.
- `goal: "min"` means lower raw metric values are better.

The older shorthand form is still accepted for built-in metrics with known default goals:

```json
"scoring": ["adg_strategy_eq", "drawdown_worst_strategy_eq"]
```

Passivbot normalizes that shorthand to the canonical object form during config formatting. For
custom or newly added metrics, use the explicit object form so the optimizer knows whether to
minimize or maximize the metric.

### Backend Selection

Passivbot supports three optimizer backends:

- `optimize.backend: deap`
  - Uses the existing DEAP evolutionary backend.
- `optimize.backend: pymoo`
  - Uses pymoo. This is now the default optimizer backend.
  - The default pymoo algorithm mode is `auto`: Passivbot uses `nsga2` when optimizing `3` or fewer objectives, and `nsga3` when optimizing more than `3`.
- `optimize.backend: gpu`
  - Uses an Apple Metal screening proxy and exact Rust validation. This backend is experimental and
    deliberately limited to the scope documented below.

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

### Apple MPS GPU Backend (Experimental)

The GPU backend is an additive research backend for Apple Silicon. Install the normal optimizer
dependencies plus its optional PyTorch runtime:

```bash
python3 -m pip install -e ".[full,gpu-mps]"
```

Select it with `--backend gpu` or `optimize.backend: "gpu"`. Normal live operation, backtesting,
and the DEAP/pymoo CPU optimizers do not import or require PyTorch.

The supported slice is intentionally narrow:

- Apple Silicon with `torch.backends.mps.is_available()`
- one prepared dataset per independent run or suite scenario, using one-minute candles; the
  dataset may be an individual exchange or the canonical combined multi-exchange dataset
- `strategy_kind: ema_anchor` or `trailing_martingale`, with long-only, short-only, or
  long+short enabled for one coin
- long-only, short-only, or dual-side hedge-mode multi-coin EMA-anchor and trailing-martingale runs
  for up to 64 coins, with dynamic wallet-exposure allocation and independent per-side Forager
  selection; dual-side runs require matching long/short approved and ignored coin sets, and all
  multi-coin runs require `backtest.dynamic_wel_by_tradability: true`;
  `live.forager_score_hysteresis_pct` preserves flat incumbent candidates when a challenger's
  normalized Forager-score lead is within the configured gap
- each enabled side's `n_positions` pinned to `1` and wallet-exposure limit kept positive
  for single-coin runs; supported multi-coin bounds may vary `n_positions` between `1` and the
  prepared coin count
- hedge mode and one-way mode; one-way flat-side arbitration uses the active strategy's Rust rule
- suite mode with exactly one prepared dataset per scenario, including individual-exchange
  comparisons and combined multi-exchange scenarios;
  single-coin EMA-anchor and trailing-martingale scenarios keep their existing directional
  support, while both strategies' suites may also use different multi-coin subsets of up to 64
  coins when every effective scenario shares the same supported side topology;
  scenario date, coin, ignored-coin, exchange selection, and fail-closed `bot.long`/`bot.short`
  config overrides are supported; scenario-local `coin_overrides` for supported multi-coin
  EMA-anchor and trailing-martingale runs, starting balance, maker fee, liquidation threshold,
  Forager hysteresis, and hedge mode are also supported, while other
  non-bot override paths remain unsupported; combined scenarios may use canonical per-coin source
  assignments, while an individual-exchange scenario fails closed if an effective assignment
  for one of its prepared coins selects another exchange
- static `coin_overrides` for each enabled side of multi-coin EMA-anchor and trailing-martingale
  runs: active-strategy parameters, `risk.entry_cooldown_minutes`, and explicit
  `wallet_exposure_limit` are supported; trailing-martingale `entry.ema_gate_mode`, disabled sides,
  and other override leaves fail closed
- HSL and auto-unstuck disabled
- BTC collateral, realized-loss gating, and exposure enforcers disabled
- `backtest.filter_by_min_effective_cost` may be enabled or disabled. When enabled, Metal uses the
  projected initial-entry cost test with dynamic or static per-coin wallet-exposure limits. The
  screening proxy compares against the highest executable minimum observed for that coin in the
  prepared window, rounds that threshold upward, and discounts the projected float32 product so
  boundary rounding cannot turn a just-below-threshold proxy projection into an admission. A
  conservative balance-error allowance grows on every proxy fill and is subtracted before that
  comparison, covering accumulated float32 fee and realized-PnL rounding. Exact validation may
  admit a conservative proxy false negative. A failing flat side is excluded from new-entry
  selection while an open position remains managed. Multi-coin runs still require this option to
  be disabled because their approximate selection and execution path cannot conservatively bound
  exact Rust's portfolio balance
- `live.market_orders_allowed: false`
- no invalid candle tail after the selected coin's final valid candle

Unsupported combinations fail before optimization begins. Dual-side multi-coin EMA Anchor and
Trailing Martingale use one long and one short Metal dispatch per candidate in hedge mode. Their
directional surfaces form a
conservative portfolio screening proxy; every accepted metric still comes from the unchanged exact
Rust portfolio backtest, and classification, rank, and drift gates halt material disagreement.
Dual-side one-way arbitration, unmodeled non-bot suite scenario overrides, HSL, and auto-unstuck
are not silently approximated by this release. Dual-side
multi-coin screening also rejects `fills_gap_longest_days`,
`strategy_eq_recovery_days_max`, and `volume_pct_per_day_avg`: the independent directional
summaries cannot reconstruct cross-side-only fill gaps, alternating portfolio recovery periods,
or fill volume normalized by the shared balance safely.

For a supported suite, each Metal candidate is dispatched across every prepared scenario. The GPU
path then calls the same canonical suite reducer and scenario-selection logic as the CPU optimizer
for aggregate reducers, named-scenario objectives, and named-scenario or suite-reduced limits.
Every exact validation still runs the unchanged Rust backtest for every scenario; only those exact
suite metrics enter `all_results.bin` and the Pareto front. A scenario may select a different coin
subset, date window, individual exchange, or canonical combined multi-exchange dataset, but each
scenario must resolve to exactly one prepared dataset. Scenario
`overrides` require
explicit candidate-shadowing behavior in the proxy. This slice models `bot.long` and `bot.short`
overrides: the canonical exact
suite evaluator still applies them last, after candidate materialization, while each scenario's
Metal proxy shadows the corresponding candidate parameters with the same effective values. Every
overridden scenario is rechecked against the directional GPU scope, so an override cannot silently
enable HSL, auto-unstuck, an exposure enforcer, an invalid position count, or another unsupported
behavior. In a multicoin suite, a scenario with fewer coins must keep the effective `n_positions`
range within that subset, either through common bounds or an explicit scenario override. Metal
uses the strategy-specific single-coin or multicoin kernel independently for each scenario, then
feeds all results to the same suite reducer. Scenario-local `coin_overrides` for supported
multi-coin EMA-anchor runs, `backtest.starting_balance`,
`backtest.maker_fee_override`, `backtest.liquidation_threshold`,
`live.forager_score_hysteresis_pct`, and `live.hedge_mode` are accepted because every scenario
proxy consumes them through the canonical backtest payload and then passes the same fail-closed
scope checks. Combined scenario `coin_sources` use the same prepared per-coin candles and market
settings as exact Rust; their resolved OHLCV and market-settings exchanges are part of checkpoint
identity. Other non-bot paths remain rejected until their proxy semantics are modeled. The
effective external suite definition and any `--scenarios` filter are
stored in the run contract and checkpoint identity, with dynamic scenario dates resolved to the
prepared concrete dates. The checkpoint signature also records each scenario's ordered effective
coins, side topology, and prepared candle window, so resume fails closed if preparation changes.

Ordinary `-t/--start` seeding and fine-tuning with `-ft/--fine-tune-params` use the same optimizer
shape as the CPU backends. When `-t` and `-ft` are combined, the GPU population includes the
discrete anchor id alongside the selected tunable bounds. Anchor-fixed values are supplied to the
screening proxy before each candidate's tunable values, while the unchanged exact Rust path
materializes the same anchor and tunable vector. The complete range across all anchors is checked
against the GPU scope: an anchor cannot change enabled sides or introduce unsupported risk
behavior. Base-config runtime policy fields still win over anchor configs as described in
[Fine-Tuning Specific Parameters](#fine-tuning-specific-parameters).

The V8 `optimize.enable_overrides` values `mirror_short_from_long` and
`lossless_close_trailing` are applied to Metal candidates in the same order as exact candidate
materialization. Mirroring may be used with the supported single-coin directional scopes; short
genes that exact materialization overwrites are omitted from the proxy search dimensions.
`lossless_close_trailing` is available only with `strategy_kind: trailing_martingale`. The legacy
`forward_tp_grid` and `backward_tp_grid` values remain unsupported because the GPU backend does not
support `trailing_grid_v7`.

`optimize.fixed_runtime_overrides` is also applied in exact candidate-materialization order: after
anchor and tunable values, and before `optimize.enable_overrides`. A fixed override that shadows a
supported GPU optimizer bound removes that gene from the Metal search and supplies the fixed value
to both proxy screening and exact Rust validation. The effective overridden config is checked
against the same fail-closed GPU scope, so an override cannot silently enable unsupported behavior.
Exact finalized boundary configs are validated before either optimizer backend starts. Config
normalization preserves user-defined dotted leaf paths, rejects malformed or unknown paths,
rejects aliases that resolve to the same setting, and rejects mapping-level replacements. When a
fixed value disables dependent trailing-martingale parameters, the Metal search removes and
hash-canonicalizes those dead genes using the same rule as exact candidate materialization.

Proxy scoring and limits are likewise fail-closed. This slice supports `adg_strategy_eq`,
`adg_strategy_eq_w`, `mdg_strategy_eq`, `sharpe_ratio_strategy_eq`,
`sortino_ratio_strategy_eq`, `volume_pct_per_day_avg`, `strategy_eq_recovery_days_max`,
`position_held_days_max`, `strategy_eq_underwater_pct_mean`, `drawdown_worst_strategy_eq`,
`drawdown_worst_mean_1pct_strategy_eq`, `fills_gap_longest_days`, and
`backtest_completion_ratio`. Metrics such as `fills_gap_p95_hours` that require exact per-fill
interpolation are rejected before a run starts.

The backend is hybrid rather than a replacement backtester:

1. pymoo NSGA-II proposes large normalized candidate batches.
2. A Rust-owned Metal screening program evaluates every candidate against candle data resident on
   MPS; Python only prepares buffers and dispatches the program. EMA-anchor and
   trailing-martingale use separate single-coin and multi-coin kernels. Directional runs keep
   separate long/short indicator,
   trailing, and position state with one shared balance and the exact Rust fill ordering. Python
   also precomputes strict high/low crossing boundaries as integer price ticks so float32 Metal
   comparisons preserve Rust's decimal-tick fill decisions. Candle-derived touches are classified
   from the original float64 data. EMA uses Rust-compatible directional ticks. Trailing-martingale
   uses those ticks to choose the controlling raw/target value before float32 can collapse nearby
   prices, then mirrors Rust's directional entry finalization and nearest-tick close finalization.
   The multi-coin trailing-martingale screening kernel retains per-coin EMA, volatility, trailing,
   position, cooldown, and pending-order state plus shared portfolio allocation. It stages one
   entry and close per coin per candle; exact Rust validation remains responsible for authoritative
   recursive same-candle ladders, and the normal constraint/rank/drift gates halt if that screening
   approximation stops ordering candidates reliably.
   Raw-touch close minimum quantities are computed from the original float64 price before close-
   price finalization, and their ordering relative to aligned quantity steps is retained across
   float32 transport. Tick-aligned computed targets remain on their exchange tick; residual
   float32 arithmetic drift is measured by exact validation.
3. In suite mode, the same candidate batch is screened once per scenario and reduced with the
   canonical suite scoring and limit contract.
4. Diverse proxy-front candidates and broad drift probes are sent to the unchanged Rust backtester.
5. Only exact Rust results enter `all_results.bin` and the persisted Pareto front.
6. Rolling rank and constraint-agreement gates independently stop the run if proxy/exact agreement
   falls below `drift_halt` after sufficient evidence. Constraint classification is monitored over
   all validations and independently for proxy-front candidates and broad probes. An isolated
   disagreement is retained as drift evidence rather than aborting immediately; the exact Rust
   result remains authoritative and an exact-infeasible candidate cannot enter the Pareto front.
   Feasibility disagreements are not rank-comparable and already count against the constraint
   gates, so rank correlation uses only classification-agreeing samples and requires at least eight
   comparable broad probes. Window and exact-budget validation reserve enough total probes to retain
   those eight whenever the configured probe constraint-agreement gate has not already failed.

`optimize.iters` remains the number of exact Rust validations. GPU screening counts and throughput
are reported separately in the log. `n_cpus` controls the exact-validation worker pool; MPS device
scheduling is managed by Metal.

GPU-specific settings live under `optimize.gpu`:

```json
{
  "optimize": {
    "backend": "gpu",
    "gpu": {
      "batch_size": 4096,
      "checkpoint_interval_seconds": 5.0,
      "drift_halt": 0.6,
      "drift_min_samples": 32,
      "drift_probes": 4,
      "drift_window": 128,
      "exact_workers": 0,
      "max_pending_exact": 0,
      "population_size": 4096,
      "validate_per_generation": 8
    }
  }
}
```

The CPU-side NSGA-II proposal stage uses the same `optimize.pymoo.shared` crossover, mutation, and
duplicate-elimination controls as the ordinary pymoo optimizer.

- `population_size` is the NSGA-II proxy population.
- `batch_size` caps candidates per MPS dispatch.
- `validate_per_generation` caps exact candidates selected from each proxy generation.
- `drift_probes` reserves at least part of that validation budget for candidates away from the
  proxy front.
  If the complete feasible proxy front occupies nearly the whole population, the optimizer uses
  every genuinely off-front candidate available and fills the remaining exact quota with diverse
  true-front candidates. Front members are never relabeled as broad probes. The separate broad-
  probe gates activate only after enough truthful off-front evidence has accumulated; allocation
  shortfalls and recovery are logged.
- `drift_window`, `drift_min_samples`, and `drift_halt` configure the rolling rank and optimizer-
  limit classification safety gates. Broad-probe Spearman correlation plus aggregate,
  proxy-front, and broad-probe constraint agreement must each remain at or above `drift_halt`.
  `drift_halt` must be greater than zero and at most one.
  At least eight samples of a validation class are required before its independent low agreement
  can halt a run, so `drift_window` and `optimize.iters` must be large enough to retain and reach
  eight true proxy-front validations even when the complete feasible proxy front contributes only
  one novel candidate per generation. Off-front validations stay truthfully classified as broad
  probes rather than being relabeled as front evidence, and scarce or duplicate probes give their
  unused exact slots back to true-front candidates. Front membership is carried independently
  through exact-result persistence and resume recovery. A generation still fails closed if
  duplicate filtering leaves no novel proxy-front candidate, so broad probes cannot silently
  consume the exact budget needed to activate the front gate. `drift_probes` must remain below
  `validate_per_generation` so each generation requests proxy-front safety evidence. A partial
  final validation batch scales its reserved probe count down proportionally.
- `exact_workers: 0` inherits `optimize.n_cpus`; a positive value overrides it for this backend.
- `max_pending_exact: 0` defaults to twice the exact-worker count.
  It must be at least `validate_per_generation` so throttling cannot change the configured
  proxy-front/broad-probe evidence allocation; the backend waits for that capacity before
  screening another generation.
- `checkpoint_interval_seconds` bounds generation-level optimizer-state checkpoint writes. Exact
   result batches are checkpointed immediately, and each durable result carries the proxy/exact
   safety evidence needed to recover if its flush outruns the companion checkpoint. A final
   evidence-budget check applies to fresh and resumed runs and includes recovered class membership,
   the rolling-window suffix, discarded pending work, and all full or partial validation batches.
   Exact worker results are consumed in submission order even if workers finish out of order,
   preserving the modeled batch sequence. Resume fails closed if the mandatory proxy-front gate can
   no longer reach its minimum sample count. Broad probes remain opportunistic across restart, just
   as in an uninterrupted run: recovered truthful probes are retained, and their independent gates
   activate only after enough off-front evidence exists. A final checkpoint is always written on
   successful completion.

The proxy is a float32 ranking model, not an authoritative simulator. Exact Rust metrics and configs
remain the only stored optimization results. The screening source is owned and exported by the
Rust extension; it does not replace or modify the exact Rust backtester. Credit: the Torch
metric-reduction work was adapted from RustyCZ's Passivbot GPU branch at commit `7c529bc73`; the
MPS Metal integration and hybrid validation gates are specific to this implementation.

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
        "crossover_eta": 20.0,
        "crossover_prob_var": 0.5,
        "mutation_eta": 20.0,
        "mutation_prob_var": "auto",
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
  - With the default 11-objective Passivbot scoring set, common reference-direction counts are:
    - `n_partitions = 2` -> `66`
    - `n_partitions = 3` -> `286`
    - `n_partitions = 4` -> `1001`
  - Default is `"auto"`. For the default 11-objective setup, Passivbot currently resolves that to
    `n_partitions = 3`, which gives `286` reference directions.
  - In auto mode, Passivbot chooses the largest Das-Dennis grid whose reference-direction count
    fits within the resolved NSGA-III population budget. This preserves the NSGA-III invariant that
    `population_size >= reference directions`.

- `optimize.population_size`
  - For `pymoo` + `nsga3`, `null` means “auto”.
  - In that case Passivbot uses the default NSGA-III population budget of `500`, then resolves
    auto reference directions to the finest Das-Dennis grid that fits inside that budget.
  - For the default 11-objective setup, that means `population_size = 500` with `286` reference
    directions.
  - For a 10-objective setup, auto keeps `population_size = 500` and uses `220` reference
    directions (`n_partitions = 3`) because the next grid would require `715` reference directions.
  - For `pymoo` + `nsga2`, `null` means “auto” and currently resolves to `250`.
  - Set an explicit integer when you want to change the per-generation evaluation budget and the
    maximum auto reference-direction grid size. For example, `population_size = 1000` allows the
    10-objective auto resolver to use `715` reference directions (`n_partitions = 4`).
  - For `deap`, Passivbot currently falls back to its legacy fixed default when `null` is left in
    place.

#### Shared Pymoo Hyperparameters

The `shared` block controls the SBX crossover and polynomial mutation operators used by both
`nsga2` and `nsga3`.

Current meaning of the main pymoo knobs:

- `optimize.pymoo.algorithm`
  - `auto`, `nsga2`, or `nsga3`.
  - Default is `auto`.
  - `auto` chooses `nsga2` when `len(optimize.scoring) <= 3`, otherwise `nsga3`.
  - Use explicit `nsga2` or `nsga3` only when you want to override that default selection.
- `optimize.pymoo.shared.crossover_prob_var`
  - Per-variable SBX crossover probability.
  - Higher values mix more parameters between parents on each crossover.
  - Default `0.5` is a conservative middle ground for Passivbot's parameter space.
- `optimize.pymoo.shared.crossover_eta`
  - SBX distribution index.
  - Higher values keep offspring closer to the parents; lower values explore more aggressively.
  - Default `20` is a standard conservative setting and is usually a good starting point.
- `optimize.pymoo.shared.mutation_prob_var`
  - Per-variable polynomial-mutation probability.
  - `"auto"` means `1 / n_params`.
  - This is the default and is usually the right choice for Passivbot's parameter counts because
    it scales automatically with the number of tunable parameters.
- `optimize.pymoo.shared.mutation_eta`
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
- Keep `mutation_prob_var: "auto"`.
- Keep `crossover_eta: 20` and `mutation_eta: 20` unless you have a specific reason to make
  variation much more local or much more aggressive.
- Keep `crossover_prob_var: 0.5` unless you have evidence that crossover is either too timid or
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
  "crossover_eta": 20,
  "crossover_prob_var": 0.5,
  "eliminate_duplicates": true,
  "mutation_eta": 20,
  "mutation_prob_var": "auto"
}
```

- `crossover_eta: 20`
  - conservative crossover; offspring stay fairly close to parents
- `crossover_prob_var: 0.5`
  - each parameter has a 50% chance of participating in crossover
- `mutation_eta: 20`
  - conservative mutation; most mutations are relatively local
- `mutation_prob_var: "auto"`
  - mutate each parameter with probability `1 / n_params`
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

### Fine-Tuning Specific Parameters

When you only want to adjust a handful of parameters and keep everything else fixed, use
`--fine_tune_params` (short: `-ft`). Provide a comma-separated list of dotted config-path
selectors to keep tunable; all other bounds are locked to their current config values
before the run starts. Selectors match full path segments by prefix or suffix, not partial
substrings.
The leading `bot.` may be omitted for side-local paths, so `long.risk` is equivalent to
`bot.long.risk`. A `*` segment may be used as a one-segment wildcard, for example
`*.strategy.close` matches both long and short active-strategy close params. A leaf selector
such as `we_excess_allowance_pct` matches every bound whose config path ends with that
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

Use `--polish-pct` to narrow every configured optimizer bound around the current config
value before the run starts:

```bash
passivbot optimize path/to/config.json --polish-pct 0.25
```

By default this keeps the polished bounds inside the original `optimize.bounds` domain and
leaves fixed bounds fixed. `--polish-bounds-mode` changes that policy:

- `clamp`: default behavior; intersect polished bounds with the original bounds.
- `override-tunable`: allow tunable bounds to escape the original bounds; fixed bounds stay fixed.
- `override-all`: allow tunable bounds to escape the original bounds and expand fixed bounds too.

Polish still uses relative bounds: `[value * (1 - pct), value * (1 + pct)]`. A current
value of `0.0` therefore remains fixed at `[0.0, 0.0]`.

`optimize.fixed_params` is applied after polish. That makes it the right way to polish all
bounds while pinning selected parameters to the config value:

```bash
passivbot optimize path/to/config.json \
  --polish-pct 0.25 \
  --polish-bounds-mode override-all \
  --optimize.fixed_params long.risk.n_positions,long.risk.total_wallet_exposure_limit
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
    "bot.long.hsl.no_restart_drawdown_threshold": 1.0,
    "bot.short.hsl.no_restart_drawdown_threshold": 1.0
  }
}
```

That default override disables terminal no-restart during optimizer evaluations so candidates can
be constrained through `drawdown_worst_strategy_eq`, `drawdown_worst_ema_strategy_eq`,
`drawdown_worst_mean_1pct_strategy_eq`, `drawdown_worst_mean_1pct_ema_strategy_eq`, and
`strategy_eq_recovery_days_max` instead of being prematurely truncated.

When you provide many starting configs, optimizer bounds how many seed evaluations may be in flight
at once. For the DEAP backend, the same cap also applies to generation offspring evaluations:

```json
"optimize": {
  "max_pending_starting_evals_per_cpu": 1
}
```

Effective cap:

- `max_pending = n_cpus * max_pending_starting_evals_per_cpu`
- All provided starting configs are still evaluated before the optimizer trims them down to the
  backend's initial population.

This is mainly a memory-control knob for large seed pools and DEAP generation batches, especially
in suite mode where each candidate returns a larger metrics payload. Lower it first if the VPS
spikes RAM during seed or offspring evaluation.

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
      {"metric": "adg_strategy_eq", "goal": "max"},
      {
        "metric": "strategy_eq_underwater_pct_mean",
        "goal": "min",
        "scenario": null
      },
      {
        "metric": "strategy_eq_recovery_days_max",
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
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_worst_strategy_eq",
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
serialized configs. For backward compatibility, the input aliases `aggregate`, `stat`, and
`scenario_stat` are accepted at the same schema positions (`field` is additionally accepted for
legacy limits). Same-valued aliases collapse to `reducer`; conflicting aliases fail validation.
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
    "aggregated": {"adg_btc_w": 0.0012, "...": "..."},
    "stats": {"adg_btc_w": {"mean": 0.0011, "min": 0.0008, "max": 0.0014, "std": 1.5e-4}}
  },
  "scenarios": {
    "scenario_a": {"stats": {"adg_btc_w": {"mean": 0.0012, "min": 0.0011, ...}}},
    "scenario_b": {"stats": {"adg_btc_w": {"mean": 0.0009, ...}}}
  }
}
```

Pareto members store a compact metrics payload under `metrics.stats` (and `suite_metrics` when suite
mode is enabled) instead of the older `analyses_combined` / per-exchange analysis blocks.

## Optimization Process

- Uses a multi-objective evolutionary backend (`deap`, `pymoo`, or experimental `gpu`)
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
optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{n_days}days_{coin_label}_{hash}/
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
list, for example `sharpe_ratio_strategy_eq`, as long as that metric is present in the
saved Pareto JSON and Passivbot has a known default min/max direction for it.

Example:

```bash
passivbot tool pareto \
  -o sharpe_ratio_strategy_eq,adg_strategy_eq,strategy_eq_recovery_days_max \
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

- `metric`: canonical metric name (e.g. `drawdown_worst_btc`, `loss_profit_ratio`, `adg`).
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
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  },
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg", "penalize_if": "<", "value": 0.0008, "reducer": "mean"}
]
```

CLI overrides can replace the full limit set with the same JSON/HJSON payload:

```bash
passivbot optimize --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]'
```

For quicker one-off edits, use repeatable `--limit` entries. The symbolic scalar operators
in `--limit` are written as keep conditions, matching `pareto_store.py` filtering:
- `--limit 'adg > 0.0008'` means keep only results with `adg > 0.0008`
- `--limit 'drawdown_worst <= 0.35'` means keep only results with `drawdown_worst <= 0.35`

```bash
passivbot optimize \
  --clear-limits \
  --limit 'drawdown_worst <= 0.35' \
  --limit 'drawdown_worst_strategy_eq <= 0.5 scenario=base' \
  --limit 'backtest_completion_ratio>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'adg > 0.0008 reducer=mean'
```

You can also combine both forms. `--limits` loads a whole list first, and each `--limit`
appends one more canonical entry:

```bash
passivbot optimize \
  --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]' \
  --limit 'strategy_eq_recovery_days_max <= 21'
```

Semantics:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` is repeatable and appends one parsed entry to that replacement set.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

Limit violations do not disqualify a config. They produce positive penalty scores in optimizer
engine space: if a limit metric matches a configured scoring metric, the penalty replaces that
objective's engine score for the candidate; unmatched/global penalties affect every objective.
Metric names may include `_usd` / `_btc` suffixes to lock a denomination; when omitted, USD is
assumed.

## Performance Metrics

Backtest statistics originate in the Rust engine (`passivbot-rust/src/analysis.rs`) and are
augmented in Python (`src/backtest.py`). The optimizer aggregates them per exchange and then
over all exchanges before scoring.

### How Metrics Feed Scoring

- `optimize.scoring` lists the objective metrics. Each entry becomes a fitness component in
  sorted order.
- Each scoring entry is normalized to `{metric, goal}`. `goal: "max"` means higher raw metric
  values are better; `goal: "min"` means lower raw metric values are better.
- For every metric, `Evaluator.combine_analyses` computes mean/min/max/std across all
  exchanges in the run. The scoring logic uses the mean (`{metric}_mean`).
- Internally, Passivbot converts all objectives into optimizer engine space where lower is better,
  so both the `deap` and `pymoo` backends receive consistent minimization-style values.
- Penalties from `optimize.limits` replace affected engine objective scores when a bound is
  violated, turning constraint breaches into very poor scores while preserving the raw metrics
  stored for inspection.
- Metrics are emitted with both USD and BTC suffixes (for example, `adg_usd` and `adg_btc`).
- `_btc` metrics use BTC-denominated balance/equity as the numeraire even when
  `backtest.btc_collateral_cap = 0`, so they can be used to compare strategy performance against
  passive BTC exposure.
- The tables below reference the base metric names for brevity; append `_usd` or `_btc` to select the denomination you want to use.
- Exposure-normalized variants (e.g., `adg_per_exposure_long`) divide the base metric by
  that side’s configured `total_wallet_exposure_limit`, letting you compare bots that use
  different leverage budgets.

### Returns & Growth
| Metric | Description |
|--------|-------------|
| `adg`, `adg_w` | Average Daily Gain (smoothed geometric) and its recency-biased counterpart |
| `mdg`, `mdg_w` | Median Daily Gain and its recency-biased counterpart |
| `gain` | Final balance gain (end/start ratio) |
| `adg_strategy_eq`, `adg_strategy_eq_w` | Collateral-agnostic geometric growth on the synthetic strategy-equity curve |
| `mdg_strategy_eq`, `mdg_strategy_eq_w` | Median-day version of the same strategy-equity growth family |
| `*_per_exposure_{long,short}` | Above metrics divided by the configured exposure limit per side |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `drawdown_worst` | Maximum peak-to-trough drawdown |
| `drawdown_worst_mean_1pct` | Mean of worst 1% daily worst drawdowns, computed from full-resolution drawdowns before daily reduction |
| `drawdown_worst_strategy_eq` | Worst drawdown on collateral-agnostic strategy equity |
| `drawdown_worst_ema_strategy_eq` | Worst EMA-smoothed strategy-equity drawdown, shared as `max(long, short)` |
| `drawdown_worst_mean_1pct_strategy_eq` | Mean of worst 1% daily worst strategy-equity drawdowns, computed from full-resolution strategy-equity drawdowns before daily reduction |
| `drawdown_worst_mean_1pct_ema_strategy_eq` | Mean of worst 1% EMA-smoothed strategy-equity drawdown samples, shared as `max(long, short)` |
| `expected_shortfall_1pct` | Mean of worst 1% daily losses (CVaR) |
| `equity_balance_diff_neg_max` / `pos_max` | Largest divergence between equity and account balance (negative side tracks only drawdowns below balance; positive side tracks only run-ups above balance) |
| `equity_balance_diff_neg_mean` / `pos_mean` | Average divergence between equity and balance (split by sign as above) |

### Ratios & Efficiency
| Metric | Description |
|--------|-------------|
| `sharpe_ratio`, `sharpe_ratio_w` | Return-to-volatility ratio and its recency-biased variant |
| `sortino_ratio`, `sortino_ratio_w` | Return-to-downside-volatility ratio |
| `calmar_ratio`, `calmar_ratio_w` | Return divided by maximum drawdown |
| `sterling_ratio`, `sterling_ratio_w` | Return divided by the average of the worst 1% drawdowns |
| `omega_ratio`, `omega_ratio_w` | Sum of positive returns / sum of absolute negative returns |
| `*_strategy_eq`, `*_strategy_eq_w` ratios | Collateral-agnostic ratio family using the strategy-equity curve |

### Position & Execution Metrics
| Metric | Description |
|--------|-------------|
| `positions_held_per_day` | Average number of unique positions opened per day |
| `position_held_hours_{mean,median,max}`, `position_held_days_{mean,median,max}` | Holding-time statistics in hours and equivalent days |
| `position_unchanged_hours_max`, `position_unchanged_days_max` | Longest span without modifying an existing position, in hours and equivalent days |
| `volume_pct_per_day_avg`, `volume_pct_per_day_avg_w` | Average traded volume as % of account per day, with recency bias |
| `peak_recovery_hours_equity_usd`, `_btc`; `peak_recovery_days_equity_usd`, `_btc` | Longest time the equity curve stayed below its prior peak before recovering, per denomination, in hours and equivalent days. Available for scoring and limit checks (e.g. `{"metric": "peak_recovery_days_equity_usd", "penalize_if": ">", "value": 7}`). |
| `peak_recovery_hours_pnl`, `peak_recovery_days_pnl` | Longest recovery time of cumulative realised PnL (USD), in hours and equivalent days. Useful for monitoring realised drawdown recovery latency. |
| `strategy_eq_recovery_days_mean`, `strategy_eq_recovery_days_median`, `strategy_eq_recovery_days_p95`, `strategy_eq_recovery_days_p99`, `strategy_eq_recovery_days_mean_worst_5pct`, `strategy_eq_recovery_days_mean_worst_1pct`, `strategy_eq_recovery_days_max` | Per-sample strategy-equity time-to-exceed distribution in days. Each sample measures how long until a later strategy-equity sample strictly exceeds it; unrecovered samples use the open tail to the backtest end. |
| `peak_recovery_hours_strategy_eq`, `peak_recovery_days_strategy_eq` | Legacy max-recovery metrics. `peak_recovery_days_strategy_eq` is an alias for `strategy_eq_recovery_days_max`; the hours variant remains available for older configs. |
| `high_exposure_hours_{mean,max}_long`, `high_exposure_days_{mean,max}_long` | Mean / maximum duration of continuous periods where total long wallet exposure exceeded the daily-resampled average long TWE, in hours and equivalent days |
| `high_exposure_hours_{mean,max}_short`, `high_exposure_days_{mean,max}_short` | Mean / maximum duration of continuous periods where total short wallet exposure exceeded the daily-resampled average short TWE, in hours and equivalent days |

### Equity Curve Quality
| Metric | Description |
|--------|-------------|
| `equity_choppiness`, `equity_choppiness_w` | Normalized total variation (lower is smoother) |
| `equity_jerkiness`, `equity_jerkiness_w` | Normalized mean absolute second derivative |
| `exponential_fit_error`, `exponential_fit_error_w` | MSE from a log-linear equity fit |

> Metrics with the `*_w` suffix use recency-weighted means: the metric is evaluated on ten
> overlapping slices of the equity curve (full history, last 1/2, last 1/3, …, last 1/10)
> and averaged. This biases the score toward recent behavior without ignoring the past.

The equity-balance difference metrics are derived by computing `(equity - balance) / balance`
minute-by-minute. Positive deviations contribute exclusively to the `*_pos_*` metrics, while
negative deviations contribute exclusively to the `*_neg_*` metrics; no cross-contamination
occurs. This mirrors the separation implemented in `passivbot-rust/src/analysis.rs` and helps
highlight asymmetric behavior in bots whose equity routinely sits above or below the
account’s wallet exposure limit baseline.

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
