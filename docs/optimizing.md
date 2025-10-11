# Optimizing

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

## Running Optimization

```bash
python3 src/optimize.py [path/to/config.json]
```

- Defaults to `configs/template.json` if no config is specified
- Use existing configs as starting points: `--start path/to/config(s)`

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

Full analysis is included in each member of the Pareto front. Use
```bash
python3 src/pareto_store.py optimize_results/.../pareto/
```
to produce a visualization. Supports plotting for 2 or 3 metrics.

## Optimization Limits

To enforce constraints during optimization, use the `optimize.limits` key. Each limit defines a threshold beyond which the configuration will be penalized. Penalty grows with severity of violation. CLI and config file formats are supported.

### CLI Format:
Example:
```bash
--limits "--penalize_if_greater_than_drawdown_worst 0.3 --penalize_if_lower_than_adg 0.001"
```

This will:
- Penalize any config where `drawdown_worst > 0.3`
- Penalize any config where `adg < 0.001`

### Config Format:
```json
"limits": {
  "penalize_if_greater_than_drawdown_worst": 0.3,
  "penalize_if_lower_than_adg": 0.001
}
```

### Notes:
- If the limit key is just `metric_name`, the direction will be inferred from its scoring weight.
- Metric names may be either plain (e.g., `adg` for USD collateralized backtest) or prefixed with "btc\_" (e.g., `btc_adg` for BTC collateralized backtest).
- Penalties are applied to the objective score; they do not disqualify a config.

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
- Metrics are available in USD collateral form by default. If
  `backtest.use_btc_collateral` is true, BTC-denominated variants are exported with the
  `btc_` prefix.
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
| `flat_btc_balance_hours` | Hours spent with the BTC collateral balance flat while USD debt is being worked down (BTC collateral mode pays off USD borrow first, so long plateaus here highlight stretches where losses took time to recover before fresh BTC could be accumulated) |

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
