# Optuna Optimizer

The Optuna optimizer is a modern alternative to the legacy DEAP-based optimizer, offering native multi-objective optimization, multiple sampling algorithms, resumable studies, and real-time monitoring via optuna-dashboard.

## Why Optuna?

| Feature | Legacy (DEAP) | Optuna |
|---------|---------------|--------|
| Multi-objective | Custom NSGA-II implementation | Native support with multiple algorithms |
| Samplers | NSGA-II only | TPE, NSGA-II, NSGA-III, GP, Random |
| Constraints | Penalty-based limits | Explicit min/max constraints with hard/soft modes |
| Resumability | Start from seed configs only | Full study persistence—resume exactly where you left off |
| Monitoring | Post-hoc with pareto_dash.py | Real-time with optuna-dashboard |
| Fine-tuning | `--fine_tune_params` | `--fine-tune` (same capability) |

The Optuna optimizer uses the same backtest engine and shared-memory architecture as the legacy optimizer, so performance characteristics are similar.

## Quick Start

### Minimal Config Changes

Add these three sections to your `config.optimize`:

```json
{
  "optimize": {
    "objectives": [
      {"metric": "mdg_w", "direction": "maximize"},
      {"metric": "sterling_ratio", "direction": "maximize"}
    ],
    "constraints": [
      {"metric": "drawdown_worst_usd", "max": 0.5},
      {"metric": "loss_profit_ratio", "max": 0.6}
    ],
    "optuna": {
      "n_trials": 10000,
      "n_cpus": 8
    }
  }
}
```

Your existing `bounds` section works unchanged.

### Running

```bash
# Start a new optimization
python3 src/optuna_optimize.py configs/template.json

# Resume an interrupted study
python3 src/optuna_optimize.py optimize_results/2024-01-15T12_34_56_abcd1234/
```

### Monitoring with optuna-dashboard

While the optimizer runs, launch the dashboard in another terminal:

```bash
pip install optuna-dashboard  # if not installed
optuna-dashboard optimize_results/YOUR_STUDY_DIR/study.log
```

Open `http://localhost:8080` to see real-time trial progress, Pareto fronts, and parameter importance.

## Config Reference

### Objectives

Define what metrics to optimize. Unlike the legacy `scoring` list, each objective explicitly declares its direction.

```json
"objectives": [
  {"metric": "mdg_w", "direction": "maximize"},
  {"metric": "sterling_ratio", "direction": "maximize"}
]
```

| Field | Description |
|-------|-------------|
| `metric` | Any metric from the backtest analysis (see [Performance Metrics](optimizing.md#performance-metrics)) |
| `direction` | `"maximize"` or `"minimize"` |

**Metric suffixes:** Append `_mean`, `_min`, `_max`, or `_std` to target specific statistics across exchanges/scenarios. Without a suffix, `_mean` is used. You can also use `_usd` or `_btc` for denomination.

### Constraints

Replace the legacy `limits` with explicit bounds. Constraints define acceptable ranges for metrics.

```json
"constraints": [
  {"metric": "drawdown_worst_usd", "max": 0.5},
  {"metric": "loss_profit_ratio", "max": 0.6},
  {"metric": "positions_held_per_day_w", "min": 3},
  {"metric": "sharpe_ratio", "min": 0.5, "max": 3.0}
]
```

| Field | Description |
|-------|-------------|
| `metric` | Metric to constrain |
| `min` | Minimum acceptable value (optional) |
| `max` | Maximum acceptable value (optional) |

At least one of `min` or `max` is required. You can specify both for range constraints.

**Constraint modes** are controlled by `optuna.penalty_weight`:

| `penalty_weight` | Mode | Behavior |
|------------------|------|----------|
| `-1` | Hard | Violations are passed to the sampler as infeasible; trial is marked failed |
| `0` | Disabled | Constraints are ignored |
| `> 0` (e.g., `1000`) | Soft | Violation amounts are multiplied by this weight and added to objective scores |

### Optuna Section

```json
"optuna": {
  "n_trials": 10000,
  "n_cpus": 8,
  "penalty_weight": 1000,
  "max_best_trials": 200,
  "sampler": {
    "name": "nsgaii",
    "population_size": 250
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `n_trials` | 250000 | Total optimization trials to run |
| `n_cpus` | 8 | Worker processes |
| `penalty_weight` | 1000 | Constraint penalty mode (see above) |
| `max_best_trials` | 200 | Maximum Pareto configs to export |
| `sampler` | NSGA-II | Sampling algorithm configuration |

### Samplers

#### TPE (Tree-structured Parzen Estimator)
```json
"sampler": {
  "name": "tpe",
  "n_startup_trials": 50,
  "multivariate": true,
  "seed": null
}
```
Good for: General-purpose optimization, especially with many parameters.

#### NSGA-II
```json
"sampler": {
  "name": "nsgaii",
  "population_size": 250,
  "mutation_prob": null,
  "crossover_prob": 0.9,
  "seed": null
}
```
Good for: Multi-objective optimization with 2-3 objectives. The default choice.

#### NSGA-III
```json
"sampler": {
  "name": "nsgaiii",
  "population_size": 250,
  "mutation_prob": null,
  "crossover_prob": 0.9,
  "seed": null
}
```
Good for: Many-objective optimization (4+ objectives).

#### GP (Gaussian Process)
```json
"sampler": {
  "name": "gp",
  "n_startup_trials": 10,
  "deterministic_objective": false,
  "seed": null
}
```
Good for: Expensive objectives with few parameters. Not recommended for Passivbot (backtests are fast enough that TPE/NSGA-II are more efficient).

#### Random
```json
"sampler": {
  "name": "random",
  "seed": null
}
```
Good for: Baseline comparisons, or when you want pure exploration.

## CLI Reference

```bash
python3 src/optuna_optimize.py <path> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `path` | Config file (starts new optimization) or study directory (resumes existing) |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--n-trials` | `-n` | Override number of trials |
| `--n-cpus` | `-c` | Override number of workers |
| `--study-name` | `-s` | Custom study name (new only, default: timestamp_hash) |
| `--sampler` | | Sampler override: `tpe`, `nsgaii`, `nsgaiii`, `gp`, `random` (new only) |
| `--fine-tune` | `-ft` | Comma-separated params to tune; all others fixed (new only) |
| `--start` | `-t` | Seed configs file or directory (new only) |
| `--debug-level` | `-d` | Log verbosity: 0=warn, 1=info, 2=debug, 3=trace |

### Examples

```bash
# Basic optimization
python3 src/optuna_optimize.py configs/template.json

# More trials, more workers
python3 src/optuna_optimize.py configs/template.json -n 50000 -c 16

# Use NSGA-III sampler
python3 src/optuna_optimize.py configs/template.json --sampler nsgaiii

# Fine-tune only specific parameters
python3 src/optuna_optimize.py configs/template.json \
  --fine-tune long_ema_span_0,long_ema_span_1,long_entry_grid_spacing_pct

# Seed with existing good configs
python3 src/optuna_optimize.py configs/template.json \
  --start configs/starting_pool/

# Resume interrupted study with more trials
python3 src/optuna_optimize.py optimize_results/2024-01-15T12_34_56_abcd1234/ -n 5000

# Verbose debugging
python3 src/optuna_optimize.py configs/template.json -d 2
```

## Monitoring with optuna-dashboard

The Optuna optimizer stores its study in a journal file (`study.log`) that optuna-dashboard can read in real-time.

### Setup

```bash
pip install optuna-dashboard
```

### Launching

While the optimizer is running (or after it completes):

```bash
optuna-dashboard optimize_results/YOUR_STUDY_DIR/study.log
```

Then open `http://localhost:8080` in your browser.

### Dashboard Features

- **Trial history**: See all completed trials with their objective values
- **Pareto front visualization**: Interactive plots of the multi-objective trade-offs
- **Parameter importance**: Which parameters most affect each objective
- **Hyperparameter relationships**: Slice plots showing parameter vs objective correlations
- **Timeline**: Trial completion rate over time

The dashboard updates live as new trials complete, making it useful for monitoring long optimization runs.

## Output Structure

Each optimization creates a directory under `optimize_results/`:

```
optimize_results/2024-01-15T12_34_56_abcd1234/
├── study.log           # Optuna journal (can be used with optuna-dashboard)
├── config.json         # Config snapshot used for this run
└── pareto/
    ├── members.json    # Pareto front configurations
    └── metrics.json    # Pareto metrics summary
```

### Pareto Output

When optimization completes (or is interrupted), the Pareto front is extracted automatically. `pareto/members.json` contains the non-dominated configurations, limited to `max_best_trials` entries.

You can also use the existing `pareto_dash.py` tool for analysis:

```bash
python3 src/tools/pareto_dash.py --data-root optimize_results
```

## Migration Guide

### Config Changes

#### Objectives (replaces `scoring`)

**Legacy:**
```json
"scoring": ["adg_pnl", "mdg_pnl_w", "sharpe_ratio"]
```

**Optuna:**
```json
"objectives": [
  {"metric": "adg_pnl", "direction": "maximize"},
  {"metric": "mdg_pnl_w", "direction": "maximize"},
  {"metric": "sharpe_ratio", "direction": "maximize"}
]
```

The legacy optimizer inferred direction from hardcoded weights. Optuna requires explicit direction.

#### Constraints (replaces `limits`)

**Legacy:**
```json
"limits": [
  {"metric": "drawdown_worst_usd", "penalize_if": ">", "value": 0.5},
  {"metric": "loss_profit_ratio", "penalize_if": "greater_than", "value": 0.6},
  {"metric": "adg", "penalize_if": "<", "value": 0.001}
]
```

**Optuna:**
```json
"constraints": [
  {"metric": "drawdown_worst_usd", "max": 0.5},
  {"metric": "loss_profit_ratio", "max": 0.6},
  {"metric": "adg", "min": 0.001}
]
```

Simpler syntax: `penalize_if: ">"` becomes `max`, `penalize_if: "<"` becomes `min`.

#### Optimizer Settings

**Legacy:**
```json
"population_size": 250,
"n_cpus": 8,
"mutation_probability": 0.34,
"crossover_probability": 0.65
```

**Optuna:**
```json
"optuna": {
  "n_trials": 10000,
  "n_cpus": 8,
  "sampler": {
    "name": "nsgaii",
    "population_size": 250,
    "crossover_prob": 0.9
  }
}
```

Key difference: Optuna uses `n_trials` (total evaluations) instead of generations × population.

### CLI Changes

| Legacy | Optuna |
|--------|--------|
| `python3 src/optimize.py` | `python3 src/optuna_optimize.py` |
| `--fine_tune_params` | `--fine-tune` |
| `--start` | `--start` (same) |
| `--population_size 250` | Config only, or `--sampler nsgaii` + config |
| N/A | `--sampler tpe/nsgaii/nsgaiii/gp/random` |
| N/A | Resume by passing study directory |

### What Stays the Same

- `bounds` section format unchanged
- Backtest configuration (`backtest.*`) unchanged
- Output goes to `optimize_results/`
- `pareto_dash.py` works with both optimizers

### Not Yet Supported

- **Suite mode**: Multi-scenario evaluation (`backtest.suite`) is not yet implemented. Use the legacy optimizer if you need suite scenarios.
