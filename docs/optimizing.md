# Optimizing

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

The canonical defaults live in `src/config/schema.py`. The example config
`configs/examples/default_trailing_grid_long_npos10.json` mirrors those defaults exactly. For the
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
passivbot optimize configs/examples/default_trailing_grid_long_npos10.json --start configs/starting_pool/
```

Most config parameters can be modified via CLI. `passivbot optimize -h` for more info.

### Backend Selection

Passivbot supports two optimizer backends:

- `optimize.backend: deap`
  - Uses the existing DEAP evolutionary backend.
- `optimize.backend: pymoo`
  - Uses pymoo. This is now the default optimizer backend.
  - The default pymoo algorithm mode is `auto`: Passivbot uses `nsga2` when optimizing `3` or fewer objectives, and `nsga3` when optimizing more than `3`.

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
  - With the default 8-objective Passivbot scoring set, common reference-direction counts are:
    - `n_partitions = 3` -> `120`
    - `n_partitions = 4` -> `330`
    - `n_partitions = 5` -> `792`
  - Default is `"auto"`. For the default 8-objective setup, Passivbot currently resolves that to
    `n_partitions = 4`, which gives `330` reference directions.

- `optimize.population_size`
  - For `pymoo` + `nsga3`, `null` means “auto”.
  - In that case Passivbot resolves the NSGA-III reference directions first and then uses the
    number of reference directions as the population size.
  - For the default 8-objective setup, that means `population_size = 330`.
  - For `pymoo` + `nsga2`, set an explicit integer.
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
- Leave `population_size: null` and `ref_dirs.n_partitions: "auto"` for the default Passivbot NSGA-III behavior.
- Keep `pareto_max_size: 1000` unless archived front updates become a measured bottleneck for your
  machine or workflow.
- If you need more or less exploration pressure, change `n_partitions` or override
  `population_size` explicitly before you start tuning crossover/mutation hyperparameters.

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

That means the default 8-objective Passivbot template uses `nsga3`, while small custom scoring
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
`--fine_tune_params` (short: `-ft`). Provide a comma-separated list of `optimize.bounds`
keys to keep tunable; all other bounds are locked to their current config values before
the run starts.

```bash
passivbot optimize configs/examples/default_trailing_grid_long_npos10.json \
  --fine_tune_params long_entry_grid_spacing_pct,long_entry_initial_qty_pct
```

Behind the scenes the optimizer sets every unlisted bound to `[value, value]`, so the GA
can mutate only the parameters you specified. Bounds for the listed parameters remain as
configured.

`optimize.fixed_params` provides the config-file equivalent: list `optimize.bounds` keys that
should always be fixed to their current config values. Internally, `--fine_tune_params` and
`optimize.fixed_params` are merged into one effective fixed-parameter set before bounds are
collapsed.

`optimize.fixed_runtime_overrides` is different: it overrides runtime config values only during
optimize evaluations, without changing the stored/live config value. This is useful for
operator-risk settings such as:

```json
"optimize": {
  "fixed_runtime_overrides": {
    "bot.long.hsl_no_restart_drawdown_threshold": 1.0,
    "bot.short.hsl_no_restart_drawdown_threshold": 1.0
  }
}
```

That default override disables terminal no-restart during optimizer evaluations so candidates can
be constrained through `drawdown_worst_hsl`, `drawdown_worst_ema_hsl`,
`drawdown_worst_mean_1pct_hsl`, `drawdown_worst_mean_1pct_ema_hsl`, and
`peak_recovery_hours_hsl` instead of being prematurely truncated.

When you provide many starting configs, optimizer now also bounds how many seed evaluations may be
in flight at once:

```json
"optimize": {
  "max_pending_starting_evals_per_cpu": 1
}
```

Effective cap:

- `max_pending = n_cpus * max_pending_starting_evals_per_cpu`
- All provided starting configs are still evaluated before the optimizer trims them down to the
  backend's initial population.

This is mainly a memory-control knob for large seed pools, especially in suite mode where each
candidate returns a larger metrics payload. Lower it first if the VPS spikes RAM during initial
seed evaluation.

### Optimizer Suites

The optimizer reuses the backtest suite configuration and allows every candidate to
be evaluated across multiple scenarios before scoring. Each scenario can override coins,
date ranges, exchanges, `coin_sources`, and bot parameters via `overrides`. The optimizer
prepares a single shared dataset that covers the union of the requested data so additional
scenarios add minimal overhead.

Key fields (directly under `backtest`):

- `backtest.suite_enabled`: master toggle for suite mode, can also be set with `--suite [y/n]`
- `backtest.scenarios`: list of scenario dictionaries (same schema as backtest scenarios)
- `backtest.aggregate`: how to combine per-scenario metrics (default: `{"default": "mean"}`)

Suite mode is opt-in. The default schema/example config does not enable it automatically.

During evaluation the optimizer records:

- Per-scenario combined metrics (the same mean/min/max/std set produced by standalone
  backtests). These are exposed on each individual as `<label>__{metric}`.
- Aggregated metrics computed with the `backtest.aggregate` rules (default `mean`).
  These aggregated values feed directly into `optimize.scoring` and `optimize.limits`.

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

- Uses a multi-objective evolutionary backend (`deap` or `pymoo`)
- `pymoo` defaults to NSGA-III for many-objective runs, with NSGA-II still available explicitly
- Backtests across historical OHLCV data
- Uses multiprocessing with shared memory for reduced RAM load
- Maintains Pareto front of best-performing configurations
- Enforces constraints via `optimize.limits`
- Optimizes for multiple metrics via `optimize.scoring`
- Avoids duplicates through hash tracking and perturbation
- Logs starting-config dedup statistics at startup, including how many raw configs collapsed after quantization and how many extra TWEL-scaled variants survived

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
- `index.json`: List of Pareto member hashes

Each recorded result now also includes runtime provenance so later replay mismatches can be
diagnosed directly from the artifact.

## Analyzing Results

Full analysis is included in each member of the Pareto front. Two helper tools are available:

```bash
# Single-candidate selector
passivbot tool pareto optimize_results/.../pareto -m knee
passivbot tool pareto -m knee

# Interactive dashboard (recommended)
passivbot tool pareto-dash --data-root optimize_results

# Static matplotlib plotter
python3 src/pareto_store.py optimize_results/.../pareto/
```

`passivbot tool pareto` is the quickest way to promote one config from a large Pareto set. It
loads the JSON artifacts, optionally filters them with `--limit` / `--limits`, then chooses one
candidate using a named decision rule. It accepts either a `pareto/` directory, an optimize run
directory, or no path at all, in which case it falls back to the newest local
`optimize_results/.../pareto`. It also shows the retained front's ideal point for the active
objectives. Recommended workflow:

1. apply hard filters with `--limit`
2. use `-m reference` if you already know your target ADG / drawdown / recovery regime
3. otherwise use `-m knee` for a balanced compromise
4. use `--show-top N` to inspect the shortlist before promoting one config
5. use `--json` if you want to script the selection

Available methods:

- `knee`: approximate balanced compromise point
- `reference`: closest to user-specified targets via `--target metric=value`
- `ideal`: closest to the observed ideal point
- `utility`: weighted scalarization via `--weight metric=value`
- `lexicographic`: strict objective priority via `--priority metric_a,metric_b,...`
- `outranking`: simplified PROMETHEE-style pairwise ranking

These are practical selection heuristics for large Passivbot Pareto fronts rather than fully
formal MCDM implementations. For most real runs, `knee`, `reference`, and `utility` are the most
useful methods.

`-o` / `--objectives` can also reference stored metrics outside the original `optimize.scoring`
list, for example `sharpe_ratio_strategy_pnl_rebased`, as long as that metric is present in the
saved Pareto JSON and Passivbot has a known default min/max direction for it.

Example:

```bash
passivbot tool pareto \
  -o sharpe_ratio_strategy_pnl_rebased,adg_strategy_pnl_rebased,peak_recovery_hours_hsl \
  -m knee
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
- Optional `stat`: override the statistic to compare against (`min`, `max`, `mean`, `std`).
  The default is `_max` for `>` checks, `_min` for `<` checks, and `_mean` for range checks.

Example:

```json
"limits": [
  {"metric": "drawdown_worst_btc", "penalize_if": ">", "value": 0.3},
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg", "penalize_if": "<", "value": 0.0008, "stat": "mean"}
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
  --limit 'backtest_completion_ratio>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'adg > 0.0008 stat=mean'
```

You can also combine both forms. `--limits` loads a whole list first, and each `--limit`
appends one more canonical entry:

```bash
passivbot optimize \
  --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]' \
  --limit 'peak_recovery_hours_hsl <= 500'
```

Semantics:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` is repeatable and appends one parsed entry to that replacement set.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

Penalties are added to every objective as a positive modifier; they do not disqualify a config but will push it far from the Pareto front when violated. Metric names may include `_usd` / `_btc` suffixes to lock a denomination; when omitted, USD is assumed.

Pareto logging also includes the top violated constraints and their penalties so you can see which limits are driving a bad candidate.

## Performance Metrics

Backtest statistics originate in the Rust engine (`passivbot-rust/src/analysis.rs`) and are
augmented in Python (`src/backtest.py`). The optimizer aggregates them per exchange and then
over all exchanges before scoring.

### How Metrics Feed Scoring

- `optimize.scoring` lists the objective metrics. Each entry becomes a fitness component in
  sorted order.
- For every metric, `Evaluator.combine_analyses` computes mean/min/max/std across all
  exchanges in the run. The scoring logic uses the mean (`{metric}_mean`).
- `Evaluator.scoring_weights` (`src/optimize.py`) assigns the optimization direction: a
  negative weight means “maximize” (value is multiplied by -1 before minimization), while a
  positive weight means “minimize.”
- Penalties from `optimize.limits` are added to every objective when a bound is violated,
  turning constraint breaches into very poor scores.
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
| `adg_strategy_pnl_rebased`, `adg_strategy_pnl_rebased_w` | Collateral-agnostic geometric growth on the strategy-PnL rebased equity curve |
| `mdg_strategy_pnl_rebased`, `mdg_strategy_pnl_rebased_w` | Median-day version of the same rebased growth family |
| `*_per_exposure_{long,short}` | Above metrics divided by the configured exposure limit per side |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `drawdown_worst` | Maximum peak-to-trough drawdown |
| `drawdown_worst_mean_1pct` | Mean of worst 1% drawdowns (daily) |
| `drawdown_worst_hsl` | Worst account-level HSL drawdown |
| `drawdown_worst_ema_hsl` | Worst EMA-smoothed HSL drawdown, shared as `max(long, short)` |
| `drawdown_worst_mean_1pct_hsl` | Mean of worst 1% HSL drawdown samples |
| `drawdown_worst_mean_1pct_ema_hsl` | Mean of worst 1% EMA-smoothed HSL drawdown samples, shared as `max(long, short)` |
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
| `*_strategy_pnl_rebased`, `*_strategy_pnl_rebased_w` ratios | Collateral-agnostic ratio family using the strategy-PnL rebased equity curve |

### Position & Execution Metrics
| Metric | Description |
|--------|-------------|
| `positions_held_per_day` | Average number of unique positions opened per day |
| `position_held_hours_{mean,median,max}` | Holding-time statistics in hours |
| `position_unchanged_hours_max` | Longest span without modifying an existing position |
| `volume_pct_per_day_avg`, `volume_pct_per_day_avg_w` | Average traded volume as % of account per day, with recency bias |
| `peak_recovery_hours_equity_usd`, `_btc` | Longest time (in hours) the equity curve stayed below its prior peak before recovering, per denomination. Available for scoring and limit checks (e.g. `{"metric": "peak_recovery_hours_equity_usd", "penalize_if": ">", "value": 168}`). |
| `peak_recovery_hours_pnl` | Longest recovery time (hours) of cumulative realised PnL (USD). Useful for monitoring realised drawdown recovery latency. |
| `peak_recovery_hours_hsl` | Longest time below the all-time rebased HSL peak before recovery. Intended for optimizer risk limits. |
| `high_exposure_hours_{mean,max}_long` | Mean / maximum duration (hours) of continuous periods where total long wallet exposure exceeded the daily-resampled average long TWE |
| `high_exposure_hours_{mean,max}_short` | Mean / maximum duration (hours) of continuous periods where total short wallet exposure exceeded the daily-resampled average short TWE |

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

The script `tools/profile_optimizer_memory.py` (requires `psutil`) can be used to launch
two optimizer runs with different CPU counts and record both process RSS and system-wide
memory pressure. This is useful when validating that shared-memory datasets are behaving
as expected on a given machine.

```bash
python tools/profile_optimizer_memory.py \
  --coins BTC ETH XRP SOL \
  --iters 20 \
  --population-size 12 \
  --cpus 2 6
```

The script writes raw samples and a summary to `tmp/optimizer_mem_profiles/`.
