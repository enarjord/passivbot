# Optimizing

Passivbot's configuration can be automatically optimized through iterative backtesting to find optimal parameters.

## Usage

```shell
python3 src/optimize.py [path/to/config.json]
```
Defaults to `configs/template.json` if no config specified.

## Results Storage

Optimization results are stored in `optimize_results/`` with filenames containing date, exchanges, number of coins, and unique identifier. Each result is appended as a single-line JSON string containing analysis and configuration.

## Analysis
The script automatically runs `src/tools/extract_best_config.py` after optimization to identify the best performing configuration, saving the best candidate and the pareto front to `optimize_results_analysis/`.


Manual analysis:

```shell
python3 src/tools/extract_best_config.py path/to/results_file.txt
```

## Performance Metrics

Based on daily equity changes: `daily_eqs = equity.groupby(day).pct_change()`

### Key Metrics:

- adg: Average daily gain (`daily_eqs.mean()`)
- mdg: Median daily gain
- gain: Final gain (`balance[-1] / balance[0]`)
- drawdown_worst: Maximum peak-to-trough equity decline
- drawdown_worst_mean_1pct: Mean of the 1% worst drawdowns on daily equity samples
- expected_shortfall_1pct: Average of worst 1% losses (CVaR)

### Risk Ratios:

- sharpe_ratio: Risk-adjusted return (`adg / daily_eqs.std()`)
- sortino_ratio: Downside risk-adjusted return (`adg / downside_eqs.std()`)
- calmar_ratio: Return to max drawdown ratio (`adg / drawdown_worst`)
- sterling_ratio: Return to average worst 1% drawdowns ratio (`adg / drawdown_worst_mean_1pct`)
- omega_ratio: Ratio of gains to losses
- loss_profit_ratio: Absolute loss sum to profit sum ratio
- equity_balance_diff_neg_max: greatest distance between balance and equity when equity is less than balance
- equity_balance_diff_neg_mean: mean distance between balance and equity when equity is less than balance
- equity_balance_diff_pos_max: greatest distance between balance and equity when equity is greater than balance
- equity_balance_diff_pos_mean: mean distance between balance and equity when equity is greater than balance

Suffix `_w` indicates weighted mean across 10 temporal subsets (whole, last_half, last_third, ... last_tenth).
