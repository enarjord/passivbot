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
- Metric names may be either plain (e.g., `adg` for USD collateralized backtest) or prefixed with "\_btc" (e.g., `btc_adg` for BTC collateralized backtest).
- Penalties are applied to the objective score; they do not disqualify a config.
- Penalty magnitudes are exponentially scaled but capped to maintain stability.

## Performance Metrics

### Returns & Growth
| Metric | Description |
|--------|-------------|
| `adg` | Average Daily Gain (smoothed geometric) |
| `mdg` | Median Daily Gain |
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
| `sharpe_ratio` | Return-to-Volatility |
| `sortino_ratio` | Return-to-Downside Volatility |
| `calmar_ratio` | Return-to-Max Drawdown |
| `sterling_ratio` | Return-to-Average 1% Worst Drawdowns |
| `omega_ratio` | `sum(pos_returns) / sum(abs(neg_returns))` where returns are daily equity pct changes |

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
| `equity_choppiness` | Normalized total variation (lower is smoother) |
| `equity_jerkiness` | Normalized mean absolute second derivative |
| `exponential_fit_error` | MSE from log-linear fit (lower = more consistent growth) |

> Metrics with the \_w suffix use recency-weighted means across multiple time slices.
Specifically, each \_w metric is computed as the average of the metric evaluated over 10 overlapping subsets of the equity curve: the entire period, last 1/2, last 1/3, ..., down to the last 1/10. This emphasizes recent performance while still accounting for longer-term behavior.

## Utilities

Loading results programmatically:
```python
from opt_utils import load_results

for config in load_results("optimize_results/.../all_results.bin"):
    # Work with config
```

