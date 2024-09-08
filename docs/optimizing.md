# Optimizing

Passivbot's config parameters may be automatically optimized by iterating many backtests and extracting the optimal config.

## Usage

```shell
python3 src/optimize.py
```
Or
```shell
python3 src/optimize.py path/to/config.json
```
If no config is specified, it will default to `configs/template.json`

## Optimizing Results

All backtest results produced by the optimizer are stored in `optimize_results/`

## Analyzing Results

After optimization is complete, the script `src/tools/extract_best_config.py` will be run, analyzing all the backtest results and dumping the best one to `optimize_results_analysis/`
