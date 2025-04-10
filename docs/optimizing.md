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

## Optimization Process

- Uses NSGA-II genetic algorithm to evolve configurations
- Backtests across historical OHLCV data using shared memory for performance
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

```bash
python3 src/pareto_store.py optimize_results/.../pareto/
```

Features:
- 2D/3D metric visualization
- Identifies ideal point and closest configuration
- Optional JSON output with `--json` flag

## Performance Metrics

### Returns & Growth
| Metric | Description |
|--------|-------------|
| `adg` / `adg_w` | Average Daily Gain (smoothed geometric) |
| `mdg` / `mdg_w` | Median Daily Gain |
| `gain` | Final Balance Gain (ratio of end/start balance) |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `drawdown_worst` | Maximum peak-to-trough drawdown |
| `drawdown_worst_mean_1pct` | Mean of worst 1% drawdowns |
| `expected_shortfall_1pct` | Mean of worst 1% daily losses (CVaR) |
| `equity_balance_diff_neg_max` | Maximum negative equity-balance difference |
| `equity_balance_diff_neg_mean` | Mean negative equity-balance difference |
| `equity_balance_diff_pos_max` | Maximum positive equity-balance difference |
| `equity_balance_diff_pos_mean` | Mean positive equity-balance difference |

### Ratios & Efficiency
| Metric | Description |
|--------|-------------|
| `sharpe_ratio` / `sharpe_ratio_w` | Return-to-Volatility |
| `sortino_ratio` / `sortino_ratio_w` | Return-to-Downside Volatility |
| `calmar_ratio` / `calmar_ratio_w` | Return-to-Max Drawdown |
| `sterling_ratio` / `sterling_ratio_w` | Return-to-Average Worst Drawdowns |
| `omega_ratio` / `omega_ratio_w` | Probability-weighted ratio of gains to losses |
| `loss_profit_ratio` / `loss_profit_ratio_w` | Total Loss / Total Profit |

### Position Metrics
| Metric | Description |
|--------|-------------|
| `positions_held_per_day` | Average number of positions held daily |
| `position_held_hours_max` | Maximum duration of any position (hours) |
| `position_held_hours_mean` | Average position duration (hours) |
| `position_held_hours_median` | Median position duration (hours) |
| `position_unchanged_hours_max` | Maximum time between position adjustments (hours) |

### Equity Curve Quality
| Metric | Description |
|--------|-------------|
| `equity_choppiness` / `equity_choppiness_w` | Normalized total variation (lower is smoother) |
| `equity_curve_smoothness` / `equity_curve_smoothness_w` | Normalized mean absolute second derivative |
| `exponential_fit_error` / `exponential_fit_error_w` | MSE from log-linear fit (lower = more consistent growth) |

> Metrics with `_w` suffix use recency-weighted means across time slices

## Utilities

Loading results programmatically:
```python
from opt_utils import load_results

for config in load_results("optimize_results/.../all_results.bin"):
    # Work with config
```
