import os
import pandas as pd

from plotting import dump_plots

symbols = [
            'DOGEUSDT', 'XLMUSDT', 'VETUSDT', 'CHZUSDT', 'SHIB1000USDT', 'IOSTUSDT', 'BITUSDT', 'ONEUSDT', 'SLPUSDT', 'SPELLUSDT', 'PEOPLEUSDT'
           ]
import time

config_path = 'configs/live/auto_unstuck_enabled.example.json'
tedy_config_backtest_config_path = 'configs/backtest/default.hjson'
tedy_config_user = 'bybit_tedy'
tedy_config_start_date = '2021-01-01'
tedy_config_end_date= '2022-01-21'
tedy_config_starting_balance = 100.0

run_id = int(round(time.time() * 1000))


def save_to_db(Args, analysis, config_to_test, hand_tuned):
    import sqlite3
    import zlib
    import copy
    database = "./passivbot.results.db"
    print('Saving to db...', database)

 
    start_date = Args().start_date
    end_date = Args().end_date

    # connect to db, if not there, create it.
    try:
        con = sqlite3.connect(f"{database}")
    except Error as e:
        print(e)

    # set cursor
    cur = con.cursor()

    # create table, if it exists do nothing
    # column 'config' is marked UNIQUE to avoid duplicates
    cur.execute(""" CREATE TABLE IF NOT EXISTS results530 (
                                            symbol text,
                                            adg_we_pad_mean_long real,
                                            adg_we_pad_std_long real,
                                            wallet_exposure_long,
                                            wallet_exposure_short,
                                            adg_long real,
                                            adg_per_exposure_long real,
                                            adg_per_exposure_short real,
                                            adg_short real,
                                            adg_DGstd_ratio_long real,
                                            adg_DGstd_ratio_short real,
                                            average_daily_gain real,
                                            avg_fills_per_day real,
                                            avg_fills_per_day_long real,
                                            avg_fills_per_day_short real,
                                            biggest_psize real,
                                            biggest_psize_long real,
                                            biggest_psize_quote real,
                                            biggest_psize_quote_long real,
                                            biggest_psize_quote_short real,
                                            biggest_psize_short real,
                                            closest_bkr real,
                                            DGstd_long real,
                                            DGstd_short real,
                                            eqbal_ratio_mean real,
                                            eqbal_ratio_min real,
                                            exchange real,
                                            fee_sum real,
                                            fee_sum_long real,
                                            fee_sum_short real,
                                            final_balance real,
                                            final_equity real,
                                            gain real,
                                            gain_long real,
                                            gain_short real,
                                            hrs_stuck_avg real,
                                            hrs_stuck_avg_long real,
                                            hrs_stuck_avg_short real,
                                            hrs_stuck_max real,
                                            hrs_stuck_max_long real,
                                            hrs_stuck_max_short real,
                                            loss_sum real,
                                            loss_sum_long real,
                                            loss_sum_short real,
                                            n_closes real,
                                            n_days real,
                                            start_date real,
                                            end_date_real,
                                            n_entries real,
                                            n_fills real,
                                            n_ientries real,
                                            n_ientries_long real,
                                            n_ientries_short real,
                                            n_normal_closes real,
                                            n_normal_closes_long real,
                                            n_normal_closes_short real,
                                            n_rentries real,
                                            n_rentries_long real,
                                            n_rentries_short real,
                                            n_unstuck_closes real,
                                            n_unstuck_closes_long real,
                                            n_unstuck_closes_short real,
                                            n_unstuck_entries real,
                                            n_unstuck_entries_long real,
                                            n_unstuck_entries_short real,
                                            net_pnl_plus_fees real,
                                            net_pnl_plus_fees_long real,
                                            net_pnl_plus_fees_short real,
                                            pa_distance_max_long real,
                                            pa_distance_max_short real,
                                            pa_distance_mean_long real,
                                            pa_distance_mean_short real,
                                            pa_distance_std_long real,
                                            pa_distance_std_short real,
                                            pnl_sum real,
                                            pnl_sum_long real,
                                            pnl_sum_short real,
                                            profit_sum real,
                                            profit_sum_long real,
                                            profit_sum_short real,
                                            starting_balance real,
                                            volume_quote real,
                                            volume_quote_long real,
                                            volume_quote_short real,
                                            config text type,
                                            config_crc32 text,
                                            run_id real
                                        ); """
                )

    # set variables
    AdgWePadMeanLong = (analysis["adg_long"] / config_to_test["long"]["wallet_exposure_limit"] / analysis[
        "pa_distance_mean_long"])
    AdgWePadStdLong = (analysis["adg_long"] / config_to_test["long"]["wallet_exposure_limit"] / analysis[
        "pa_distance_std_long"])
    config = str(hand_tuned).replace("'", '"')  # need to replace ' with "

    # overwrite config WE with 0 to make crc equal with different WE tests.
    config2 = copy.deepcopy(hand_tuned)
    config2['long']['wallet_exposure_limit'] = 0
    config2['short']['wallet_exposure_limit'] = 0
    configcrc = hex(zlib.crc32((str(config2).encode())))[
                2:]  # use-case for this: same config crc for different WE tests. 

    # create result tuple
    results = (
        analysis["symbol"],
        AdgWePadMeanLong,
        AdgWePadStdLong,
        hand_tuned["long"]["wallet_exposure_limit"],
        hand_tuned["short"]["wallet_exposure_limit"],
        analysis["adg_long"],
        analysis["adg_per_exposure_long"],
        analysis["adg_per_exposure_short"],
        analysis["adg_short"],
        analysis["adg_DGstd_ratio_long"],
        analysis["adg_DGstd_ratio_short"],
        analysis["average_daily_gain"],
        analysis["avg_fills_per_day"],
        analysis["avg_fills_per_day_long"],
        analysis["avg_fills_per_day_short"],
        analysis["biggest_psize"],
        analysis["biggest_psize_long"],
        analysis["biggest_psize_quote"],
        analysis["biggest_psize_quote_long"],
        analysis["biggest_psize_quote_short"],
        analysis["biggest_psize_short"],
        analysis["closest_bkr"],
        analysis["DGstd_long"],
        analysis["DGstd_short"],
        analysis["eqbal_ratio_mean"],
        analysis["eqbal_ratio_min"],
        analysis["exchange"],
        analysis["fee_sum"],
        analysis["fee_sum_long"],
        analysis["fee_sum_short"],
        analysis["final_balance"],
        analysis["final_equity"],
        analysis["gain"],
        analysis["gain_long"],
        analysis["gain_short"],
        analysis["hrs_stuck_avg"],
        analysis["hrs_stuck_avg_long"],
        analysis["hrs_stuck_avg_short"],
        analysis["hrs_stuck_max"],
        analysis["hrs_stuck_max_long"],
        analysis["hrs_stuck_max_short"],
        analysis["loss_sum"],
        analysis["loss_sum_long"],
        analysis["loss_sum_short"],
        analysis["n_closes"],
        analysis["n_days"],
        start_date,
        end_date,
        analysis["n_entries"],
        analysis["n_fills"],
        analysis["n_ientries"],
        analysis["n_ientries_long"],
        analysis["n_ientries_short"],
        analysis["n_normal_closes"],
        analysis["n_normal_closes_long"],
        analysis["n_normal_closes_short"],
        analysis["n_rentries"],
        analysis["n_rentries_long"],
        analysis["n_rentries_short"],
        analysis["n_unstuck_closes"],
        analysis["n_unstuck_closes_long"],
        analysis["n_unstuck_closes_short"],
        analysis["n_unstuck_entries"],
        analysis["n_unstuck_entries_long"],
        analysis["n_unstuck_entries_short"],
        analysis["net_pnl_plus_fees"],
        analysis["net_pnl_plus_fees_long"],
        analysis["net_pnl_plus_fees_short"],
        analysis["pa_distance_max_long"],
        analysis["pa_distance_max_short"],
        analysis["pa_distance_mean_long"],
        analysis["pa_distance_mean_short"],
        analysis["pa_distance_std_long"],
        analysis["pa_distance_std_short"],
        analysis["pnl_sum"],
        analysis["pnl_sum_long"],
        analysis["pnl_sum_short"],
        analysis["profit_sum"],
        analysis["profit_sum_long"],
        analysis["profit_sum_short"],
        analysis["starting_balance"],
        analysis["volume_quote"],
        analysis["volume_quote_long"],
        analysis["volume_quote_short"],
        config,
        configcrc,
        run_id
    )

    # insert query
    sqlite_insert = """INSERT OR IGNORE INTO results530
                    VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ;"""

    # execute insert query with result variables
    cur.execute(sqlite_insert, results)

    # save to db.
    con.commit()

    print("results stored in db")  # if its a duplicate, it wont be stored.


async def backtest_symbol(symbol):
    class Args:
        def __init__(self):
            self.backtest_config_path = tedy_config_backtest_config_path
            self.exchange = 'binance'
            self.symbol = symbol
            self.market_type = 'futures'
            self.user = tedy_config_user
            self.start_date = tedy_config_start_date
            self.end_date = tedy_config_end_date
            self.starting_balance = tedy_config_starting_balance
            self.starting_configs = ''
            self.base_dir = 'backtests'

    config = await prepare_backtest_config(Args())
    dl = Downloader(config)
    sts = time()
    data = await dl.get_sampled_ticks()
    timestamps = data[:, 0]
    qtys = data[:, 1]
    prices = data[:, 2]
    config['n_days'] = (timestamps[-1] - timestamps[0]) / (1000 * 60 * 60 * 24)

    # config for bybit
    config['min_cost'] = 0.0
    config['min_qty'] = 1.0
    config['maker_fee'] = -0.00025  # bybit maker rebate
    config['taker_fee'] = 0.00075  # bybit maker rebate

    print(f'millis to load {len(prices)} ticks {(time() - sts) * 1000:.0f}ms')

    # choose a slice on which to test
    wsize_days = 365
    ts = int(data[-1][0] - 60 * 60 * 24 * 1000 * wsize_days)
    idx = np.argmax(data[:, 0] >= ts)
    dataslice = data[idx:]

    hand_tuned = load_live_config(config_path)
    config["starting_balance"] = Args().starting_balance
    config["latency_simulation_ms"] = 1000
    config.update(hand_tuned)
    config_to_test = {**config, **numpyize(hand_tuned)}

    print('Starting backtesting...')
    sts = time()
    fills, stats = backtest(config_to_test, dataslice)
    elapsed = time() - sts
    print(f"seconds elapsed {elapsed:.4f}")
    fdf, sdf, analysis = analyze_fills(fills, stats, config_to_test)
    # pprint.pprint(analysis)
    print(f"ADG: {analysis['average_daily_gain'] * 100}%")

    save_to_db(Args, analysis, config_to_test, hand_tuned)

    # config["result"] = analysis
    # config["plots_dirpath"] = make_get_filepath(
    #     os.path.join(config["plots_dirpath"], f"{ts_to_date(time())[:19].replace(':', '')}", "")
    # )
    # fdf.to_csv(config["plots_dirpath"] + "fills.csv")
    # sdf.to_csv(config["plots_dirpath"] + "stats.csv")
    # df = pd.DataFrame({**{"timestamp": data[:, 0], "qty": data[:, 1], "price": data[:, 2]}, **{}})
    # print("dumping plots...")
    # dump_plots(config, fdf, sdf, df)


def wrapper(coro):
    return asyncio.run(coro)


def main():
    print(symbols)
    results = None
    
    coros = [backtest_symbol(symbol) for symbol in symbols]
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = executor.map(wrapper, coros)

    # pprint.pprint(results)


if __name__ == '__main__':
    import asyncio
    import pprint
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from time import time

    import numpy as np

    from backtest import backtest
    from downloader import Downloader
    from procedures import load_live_config, prepare_backtest_config, make_get_filepath
    from pure_funcs import numpyize, analyze_fills, ts_to_date

    main()
