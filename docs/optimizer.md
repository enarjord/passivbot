# Optimizer

To optimize a configuration by iterating multiple backtests,

1. modify `configs/backtest/default.hjson` and `configs/optimize/default.hjson` as desired
2. run with `python3 optimize.py`
3. optional args: `-b or --backtest-config`: use different backtest config
4. optional args: `-o or --optimize-config`: use different optimize config
5. optionally make optimizer start from given candidate(s) by adding kwarg `--start {path_to_starting_candidate.json}`
   if pointing to a directory, will use all .json files in that directory as starting candidates

See [wiki](https://github.com/enarjord/passivbot/wiki) for more info on backtesting and optimizing