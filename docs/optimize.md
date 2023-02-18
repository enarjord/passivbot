# Optimize

Configs may be optimized with harmony search or particle swarm optimization algorithms.  
Tens, hundreds or thousands of backtests with different configs are performed,  
new candidate configs determined based on the backtest result of the previous iterations.  More iterations lead to more optimal configs.

Too little historical data to optimize on may lead to overfitting.  
Set bounds for config parameters in `configs/optimize/default.hjson`.  

Once the necessary price data has been downloaded, the optimizer will begin with optional starting candidate(s), 
test against the price history, and continue iterating through the ranges for each variable.  

!!! Warning
    The optimizer is resource greedy and may take hours and days to converge, depending on number of symbols and time span.


Due to the heuristic nature of these optimization algorithms, repeated optimize runs do not necessarily return the exact same result.

## Running an optimize session

To execute an optimize, execute the following command from the root folder:

```shell
python3 optimize.py
```

Note: the default market is Futures. Use one of the keys to define spot market if you want that. 

!!! Info
    This page assumes you have already performed configuration to execute a backtest (like setting up your account).
    If you have not done so, you can read the instructions [here](backtesting.md).

Besides the `configs/backtest/default.hjson` file as input, the optimize process sets up the search space using
the parameters defined in `configs/optimize/default.hjson`  

The search parameters that can be specified for the optimize process are as follows.

| Parameter     | Description
| ----------    | -----------
| `algorithm`       | Particle Swarm Optimization or Harmony Search
| `iters`       | The number of iterations to perform during optimize
| `n_cpus`    | The number of cores used to perform the optimize. Using more cores will speed up the optimize
| `do_long` | Indicates if the optimize should perform long positions
| `do_short` | Indicates if the optimize should perform short positions

...see more parameters and descriptions in config file

Other than the parameters specified in the table above, the parameters found in the live config file are also specified
as a range. For a description of each of those individual parameters, please see [Running live](live.md) 

!!! Info
    If you find that an optimize execution is taking longer than you expected, you can kill it using `ctrl+c` from the command-line.
    After doing so, you can still find the best result the optimize achieved so far by looking in  
    `results_harmony_search_{recursive/static/neat/clock}` or `results_particle_swarm_optimization_{recursive/static/neat/clock}`.

### Command-line arguments

Other than modifying the `configs/backtest/default.hjson` and `configs/optimize/default.hjson` files, it is also possible
to specify a number of configuration options to use via the commandline.
One or more arguments are provided to the optimizer using the following syntax on the command line:

```shell
python3 optimize.py <key> <value>  
```

The following options can be provided to the backtester. Note that any argument provided will override a value specified in the backtest configuration file.

| Key | Description
| --- | -----------
| -t / --start | Specifies one specific live config (.json) file, a directory with multiple config files to use as starting point for optimizing or a harmony memory (hm_xxxxxx.json)
| --nojit | Disables the use of numba's just in time compiler during backtests
| -b / --backtest_config | The backtest config hjson file to use<br/>**Default value:** configs/backtest/default.hjson
| -o / --optimize_config | The optimize config hjson file to use<br/>**Default value:** configs/optimize/default.hjson
| -d / --download-only | Instructs the backtest to only download the data, but not dump the ticks caches to disk
| -s / --symbol | A CSV specifying the symbol(s) to optimize
| -u / --user | The name of the account used to download trade data
| --start_date | The starting date of the backtest<br/>**Syntax:** YYYY-MM-DD
| --end_date | The end date of the backtest<br/>**Syntax:** YYYY-MM-DD
| -bd / --base_dir | The base directory to place the output files
| -m spot / --market_type spot | Sets the market to spot instead of the default Futures
| -pm / --passivbot_mode | choices [r/recursive, s/static], static or recursive passivbot mode
| -oh / --ohlcv | if given, will use 1m ohlcv instead of 1s sampled ticks


Run
```shell
python3 optimize.py --help
```
for more info.
