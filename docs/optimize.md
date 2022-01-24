# Optimize

Generally, setting your variables manually can lead to the bot going awry over time, so exercise caution when
using hand-crafted configurations. PassivBot comes with an optimize functionality, a script designed to assist you in
finding the best configuration for your use case.

Utilizing the optimizer is the best way to come up with new configurations, but requires that you have a
basic understanding of the configuration file. The optimizer's job is to look at a coin pair's price
history (ETH/USDT for the last 30 days for example), examine your provided conditions (leverage, percent balance,
grid spacing and so on...), and test those settings over the timeframe you selected. The bot will iterate
through every trade as if it were doing it live, and return the best found results to the user. In order
to find the best result possible across many iterations, Passivbot relies on [particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
There is a default optimize configuration with the bot in every version, so examine it and set the test
parameters to suit your desired results.

Once the necessary price data has been downloaded, the optimizer will begin with the starting candidate, 
test against the price history, and continue iterating through the ranges for each variable. 
Once the history for a given asset is downloaded,
additional price history can simply be tacked on to the end of the cache (done automatically),
cutting down testing times. 

!!! Warning
    The optimizer process is computed by the CPU, and can be time consuming depending on the testing period, 
    despite optimization. Also the number of trades in the specified time range can have an impact on the
    amount of memory needed to run an optimize. For example, running an optimize on a full year of BTC trading data
    with 10.000 cycles on a 12-core CPU & 32GB ram takes a couple of days. Running a full year of BTC trading data on 
    the same machine for BTS takes a couple of hours.

The key to finding new, profitable configurations is using the optimizer often and familiarizing
yourself with the settings and ranges. Adjusting the ranges narrows the proverbial 'area' the
PassivBot needs to search for good configurations, reducing the test time while potentially
cutting more or less profitable settings out of the search range.

Due to the nature of particlee swarm optimization, repeated optimize runs do not necessarily return the exact same result.

## Running an optimize

To execute an optimize, you can execute the following command from the root folder:

```shell
python3 optimize.py
```
Note: the default market is Futures. Use one of the keys to define spot market if you want that. 

!!! Info
    This page assumes you have already performed configuration to execute a backtest (like setting up your account).
    If you have not done so, you can read the instructions [here](backtesting.md).

Besides the `configs/backtest/default.hjson` file as input, the optimize process sets up the search space using
the parameters defined in `configs/optimize/default.hjson`.

The search parameters that can be specified for the optimize process are as follows.

| Parameter     | Description
| ----------    | -----------
| `iters`       | The number of iterations to perform during optimize
| `num_cpus`    | The number of cores used to perform the optimize. Using more cores will speed up the optimize
| `options`     | The parameters W, c1 and c2 are the inertia weight, the cognitive coefficient and the social coefficient used in particle swarm optimization
| `n_particles` | The number of particles used in the swarm optimization
| `break_early_factor` | Set to 0.0 to disable breaking early
| `minimum_bankruptcy_distance` | The minimum backruptcy distance achieved in an optimize cycle before it is discarded
| `minimum_equity_balance_ratio` | The minimum equity/balance ratio achieved in an optimize cycle before it is discarded
| `minimum_slice_adg` | The minimum average daily gain in a slice before it is discarded
| `maximum_hrs_no_fills` | The maximum hours for no filles to occur. If an optimize cycle exceeds this threshold, it is discarded
| `maximum_hrs_no_fills_same_side` | The maximum hours for no filles to occur on the same side. If an optimize cycle exceeds this threshold, it is discarded
| `sliding_window_days` | The number of days take make up a sliding window. Set to 0.0 to disable sliding windows
| `reward_multiplier_base` | For each completed slice, objective is multiplied by reward_multiplier_base**(z + 1) where z is enumerator of slices
| `metric` | The metric used to measure the objective on an individual optimize cycle
| `do_long` | Indicates if the optimize should perform long positions
| `do_short` | Indicates if the optimize should perform short positions

Other than the parameters specified in the table above, the parameters found in the live config file are also specified
as a range. For a description of each of those individual parameters, please see [Running live](live.md) 

!!! Info
    If you find that an optimize execution is taking longer than you expected, you can kill it using `ctrl+c` from the command-line.
    After doing so, you can still find the best result the optimize achieved so far by looking at the `backtests/{exchange}/{symbol}/optimize/{date}/intermediate_best_result.json`.

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
| -t / --start | Specifies one specific live config (.json) file or a directory with multiple config files to use as starting point for optimizing
| --nojit | Disables the use of numba's just in time compiler during backtests
| -b / --backtest_config | The backtest config hjson file to use<br/>**Default value:** configs/backtest/default.hjson
| -o / --optimize_config | The optimize config hjson file to use<br/>**Default value:** configs/optimize/default.hjson
| -d / --download-only | Instructs the backtest to only download the data, but not dump the ticks caches to disk
| -s / --symbol | A CSV specifying the symbol(s) to run the backtest on
| -u / --user | The name of the account used to download trade data
| --start_date | The starting date of the backtest<br/>**Syntax:** YYYY-MM-DD
| --end_date | The end date of the backtest<br/>**Syntax:** YYYY-MM-DD
| -bd / --base_dir | The base directory to place the output files
| -m spot / --market_type spot | Sets the market to spot instead of the default Futures

### Running batch optimize

You can run the optimze for multiple coins in a row, so you don't have to manually start an optimize for each coin. To do this, you can simply specify multiple coins in the backtest-config used (`symbol: BTCUSDT,ETHUSDT,BNBUSDT`), or specify the symbols to be used via the command-line argument to override the config file (`python3 optimize.py -s BTCUSDT,ETHUSDT,BNBUSDT`).
