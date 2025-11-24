# Optimizing

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

## Running Optimization

```bash
python3 src/optimize.py [path/to/config.json]
```

- Defaults to `configs/template.json` if no config is specified
- Use existing configs as starting points: `--start path/to/config(s)`
- Enable suite scenarios defined in the config with `--suite [y/n]` (omit value to enable)
- Layer an external suite definition via `--suite-config path/to/file.json`

Example:
```bash
python3 src/optimize.py configs/template.json --start configs/starting_pool/
```

Most config parameters can be modified via CLI. `python3 src/optimize.py -h` for more info.

### Fine-Tuning Specific Parameters

When you only want to adjust a handful of parameters and keep everything else fixed, use
`--fine_tune_params` (short: `-ft`). Provide a comma-separated list of `optimize.bounds`
keys to keep tunable; all other bounds are locked to their current config values before
the run starts.

```bash
python3 src/optimize.py configs/template.json \
  --fine_tune_params long_entry_grid_spacing_pct,long_entry_initial_qty_pct
```

Behind the scenes the optimizer sets every unlisted bound to `[value, value]`, so the GA
can mutate only the parameters you specified. Bounds for the listed parameters remain as
configured.

### Optimizer Suites

`optimize.suite` mirrors the structure of `backtest.suite` and allows every candidate to
be evaluated across multiple scenarios before scoring. Each scenario can override coins,
date ranges, exchanges, and `coin_sources`. The optimizer prepares a single shared
dataset that covers the union of the requested data so additional scenarios add minimal
overhead.

Key fields:

- `optimize.suite.enabled`: can also be toggled with `--suite [y/n]`
- `optimize.suite.include_base_scenario` / `base_label`
- `optimize.suite.scenarios`: same schema as backtest scenarios

During evaluation the optimizer records:

- Per-scenario combined metrics (the same mean/min/max/std set produced by standalone
  backtests). These are exposed on each individual as `<label>__{metric}`.
- Aggregated metrics computed with the `optimize.suite.aggregate` rules (default `mean`).
  These aggregated values feed directly into `optimize.scoring` and `optimize.limits`.

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

Only aggregated statistics remain in `analyses_combined`; the verbose per-scenario
flattened keys have been removed to keep Pareto members and `all_results.bin` lean.

## Optimization Process

- Uses NSGA-II genetic algorithm to evolve configurations
- Backtests across historical OHLCV data
- Uses multiprocessing with shared memory for reduced RAM load
- Maintains Pareto front of best-performing configurations
- Enforces constraints via `optimize.limits`
- Optimizes for multiple metrics via `optimize.scoring`
- Avoids duplicates through hash tracking and perturbation

## Output Structure

Each optimization run creates a directory:
```
optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{n_days}days_{coin_label}_{hash}/
```

Contents:
- `all_results.bin`: Binary log of all evaluated configs (msgpack format)
- `pareto/`: JSON files for Pareto-optimal configurations
  - Named `{distance}_{hash}.json` where `distance` is normalized distance to ideal point
- `index.json`: List of Pareto member hashes

## Analyzing Results

Full analysis is included in each member of the Pareto front. Two helper tools are available:

```bash
# Interactive dashboard (recommended)
python3 src/tools/pareto_dash.py --data-root optimize_results

# Static matplotlib plotter
python3 src/pareto_store.py optimize_results/.../pareto/
```

`pareto_dash.py` scans one or more optimization runs and launches a Plotly Dash app with:

- Scatter/histogram views for any metrics or objectives
- Defaults to the metrics listed in `config.optimize.scoring`, so the scatter/histogram
  immediately highlight your optimization objectives when the app loads
- Scenario-aware box plots (per-metric distributions broken down by suite scenario)
- Correlation heat maps and parameter-vs-metric scatter plots for quick diagnostics
- Streaming history chart sourced from `all_results.bin`
- CSV export of the current run's dataset for offline analysis

Install the dependencies via `pip install dash plotly` if they are not already present.
The legacy `pareto_store.py` script still supports quick 2D/3D matplotlib plots if a GUI
isn't needed.

## Optimization Limits

To enforce constraints during optimization, populate `optimize.limits` with a list of limit
objects. Each object describes when to penalize a result:

- `metric`: canonical metric name (e.g. `drawdown_worst_btc`, `loss_profit_ratio`, `adg`).
- `penalize_if`: comparison operator. Use `<` / `>` (or `less_than` / `greater_than`), `outside_range`
  to keep a metric within `[low, high]`, or `inside_range` to forbid a band.
- `value`: numeric threshold for `<`/`>` limits.
- `range`: `[low, high]` for the range-based operators.
- Optional `stat`: override the statistic to compare against (`min`, `max`, `mean`, `std`).
  Defaults mirror the legacy behaviour (`>` checks use `_max`, `<` checks use `_min`, range checks use `_mean`).

Example:

```json
"limits": [
  {"metric": "drawdown_worst_btc", "penalize_if": ">", "value": 0.3},
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg", "penalize_if": "<", "value": 0.0008, "stat": "mean"}
]
```

CLI overrides accept the same JSON/HJSON payload:

```bash
python3 src/optimize.py --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]'
```

For quick-and-dirty tweaks, the legacy format (`--penalize_if_greater_than_drawdown_worst 0.3`) is still recognized and converted to the new schema at runtime.

Penalties are added to every objective as a positive modifier; they do not disqualify a config but will push it far from the Pareto front when violated. Metric names may include `_usd` / `_btc` suffixes to lock a denomination; when omitted, USD is assumed.

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
| `*_per_exposure_{long,short}` | Above metrics divided by the configured exposure limit per side |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `drawdown_worst` | Maximum peak-to-trough drawdown |
| `drawdown_worst_mean_1pct` | Mean of worst 1% drawdowns (daily) |
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

### Position & Execution Metrics
| Metric | Description |
|--------|-------------|
| `positions_held_per_day` | Average number of unique positions opened per day |
| `position_held_hours_{mean,median,max}` | Holding-time statistics in hours |
| `position_unchanged_hours_max` | Longest span without modifying an existing position |
| `volume_pct_per_day_avg`, `volume_pct_per_day_avg_w` | Average traded volume as % of account per day, with recency bias |
| `peak_recovery_hours_equity_usd`, `_btc` | Longest time (in hours) the equity curve stayed below its prior peak before recovering, per denomination. Available for scoring and limit checks (e.g. `{"metric": "peak_recovery_hours_equity_usd", "penalize_if": ">", "value": 168}`). |
| `peak_recovery_hours_pnl` | Longest recovery time (hours) of cumulative realised PnL (USD). Useful for monitoring realised drawdown recovery latency. |

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
