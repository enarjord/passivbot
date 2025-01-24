# Tools

## Extract Pareto Frontier and best config from optimize output

The pareto front and best config extracted will be dumped in `optimize_results_analysis/`. Results from an optimize session are usually dumped in `optimize_results/`.

```shell
python3 src/tools/extract_best_config.py path/to/all_results.txt
```

## Copy ohlcv data from old location to new location

In Passivbot v7.2.13 the location of ohlcv data changed. Run this script to copy data already downloaded in earlier versions.

```shell
python3 src/tools/copy_ohlcvs_from_v7.2.12.py
```

## Generate list of approved coins based on market cap

```shell
python3 src/tools/generate_mcap_list.py
```

Output from `python3 src/tools/generate_mcap_list.py -h`:
```
  --n_coins N_COINS, -n N_COINS
                        Maxiumum number of top market cap coins. Default=100
  --minimum_market_cap_dollars MINIMUM_MARKET_CAP_MILLIONS, -m MINIMUM_MARKET_CAP_MILLIONS
                        Minimum market cap in millions of USD. Default=300.0
  --exchange EXCHANGE, -e EXCHANGE
                        Optional: filter by coins available on exchange. Comma separated values. Default=None
  --output OUTPUT, -o OUTPUT
                        Optional: Output path. Default=configs/approved_coins_{n_coins}_{min_mcap}.json
```
