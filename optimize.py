import os

os.environ["NOJIT"] = "false"

from downloader import Downloader, load_hlc_cache
import argparse
import asyncio
import json
import numpy as np
import traceback
from copy import deepcopy
from backtest import backtest
from multiprocessing import Pool, shared_memory
from njit_funcs import round_dynamic
from pure_funcs import (
    analyze_fills_slim,
    denumpyize,
    numpyize,
    make_compatible,
    get_template_live_config,
    ts_to_date,
    ts_to_date_utc,
    date_to_ts,
    tuplify,
    sort_dict_keys,
    determine_passivbot_mode,
    get_empty_analysis,
    calc_scores,
    analyze_fills,
)
from procedures import (
    add_argparse_args,
    prepare_optimize_config,
    load_live_config,
    make_get_filepath,
    load_exchange_key_secret_passphrase,
    prepare_backtest_config,
    dump_live_config,
    utc_ms,
)
from time import sleep, time
import logging
import logging.config

logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})


def calc_metrics_mean(analyses: dict):
    """
    take list of analyses and return either min, first, max or mean for each item
    """
    mins = [
        "closest_bkr_long",
        "closest_bkr_short",
        "eqbal_ratio_mean_of_10_worst_long",
        "eqbal_ratio_mean_of_10_worst_short",
        "eqbal_ratio_min_long",
        "eqbal_ratio_min_short",
    ]
    firsts = [
        "n_days",
        "exchange",
        "adg_long",
        "adg_per_exposure_long",
        "adg_weighted_long",
        "adg_weighted_per_exposure_long",
        "adg_short",
        "adg_per_exposure_short",
        "adg_weighted_short",
        "adg_weighted_per_exposure_short",
        "fee_sum_long",
        "fee_sum_short",
        "final_balance_long",
        "final_balance_short",
        "final_equity_long",
        "final_equity_short",
        "gain_long",
        "gain_short",
        "loss_sum_long",
        "loss_sum_short",
        "n_closes_long",
        "n_closes_short",
        "n_days",
        "n_entries_long",
        "n_entries_short",
        "n_fills_long",
        "n_fills_short",
        "n_ientries_long",
        "n_ientries_short",
        "n_normal_closes_long",
        "n_normal_closes_short",
        "n_rentries_long",
        "n_rentries_short",
        "n_unstuck_closes_long",
        "n_unstuck_closes_short",
        "n_unstuck_entries_long",
        "n_unstuck_entries_short",
        "net_pnl_plus_fees_long",
        "net_pnl_plus_fees_short",
        "pnl_sum_long",
        "pnl_sum_short",
        "profit_sum_long",
        "profit_sum_short",
        "starting_balance",
        "pa_distance_1pct_worst_mean_long",
        "pa_distance_1pct_worst_mean_short",
        "symbol",
        "volume_quote_long",
        "volume_quote_short",
        "drawdown_max_long",
        "drawdown_max_short",
        "drawdown_1pct_worst_mean_long",
        "drawdown_1pct_worst_mean_short",
        "sharpe_ratio_long",
        "sharpe_ratio_short",
    ]
    maxs = [
        "hrs_stuck_max_long",
        "hrs_stuck_max_short",
    ]
    analysis_combined = {}
    for key in mins:
        if key in analyses[0]:
            analysis_combined[key] = min([a[key] for a in analyses])
    for key in firsts:
        if key in analyses[0]:
            analysis_combined[key] = analyses[0][key]
    for key in maxs:
        if key in analyses[0]:
            analysis_combined[key] = max([a[key] for a in analyses])
    for key in analyses[0]:
        if key not in analysis_combined:
            try:
                analysis_combined[key] = np.mean([a[key] for a in analyses])
            except:
                analysis_combined[key] = analyses[0][key]
    return analysis_combined


def backtest_wrap(config_: dict, ticks_caches: dict):
    """
    loads historical data from disk, runs backtest and returns relevant metrics
    """
    config = {
        **{"long": deepcopy(config_["long"]), "short": deepcopy(config_["short"])},
        **{
            k: config_[k]
            for k in [
                "starting_balance",
                "latency_simulation_ms",
                "symbol",
                "market_type",
                "config_no",
                "adg_n_subdivisions",
                "n_backtest_slices",
                "slim_analysis",
            ]
        },
        **{k: v for k, v in config_["market_specific_settings"].items()},
    }
    if config["symbol"] in ticks_caches:
        ticks = ticks_caches[config["symbol"]]
    else:
        ticks = np.load(config_["ticks_cache_fname"])
    try:
        assert "adg_n_subdivisions" in config
        analyses = []
        n_slices = max(1, config["n_backtest_slices"])
        slices = [(0, len(ticks))]
        if n_slices > 2:
            slices += [
                (
                    int(len(ticks) * (i / n_slices)),
                    min(len(ticks), int(len(ticks) * ((i + 2) / n_slices))),
                )
                for i in range(max(1, n_slices - 1))
            ]
        for ia, ib in slices:
            data = ticks[ia:ib]
            fills_long, fills_short, stats = backtest(config, data)
            if config["slim_analysis"]:
                analysis = analyze_fills_slim(fills_long, fills_short, stats, config)
            else:
                longs, shorts, sdf, analysis = analyze_fills(fills_long, fills_short, stats, config)
            analyses.append(analysis.copy())
        analysis = calc_metrics_mean(analyses)
    except Exception as e:
        analysis = get_empty_analysis()
        logging.error(f'error with {config["symbol"]} {e}')
        logging.error("config")
        traceback.print_exc()
        with open(make_get_filepath("tmp/optimize_errors.txt"), "a") as f:
            f.write(json.dumps([time(), "error", str(e), denumpyize(config)]) + "\n")
    return analysis


async def main(algorithm=None):
    logging.basicConfig(format="", level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        prog="Optimize multi symbol", description="Optimize passivbot config multi symbol"
    )
    parser.add_argument(
        "-oc",
        "--optimize_config",
        type=str,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/default.hjson",
        help="optimize config hjson file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        dest="optimize_output_path",
        default=None,
        help="optimize results directory. Defaults to 'results_{algorithm}_{passivbot_mode}/",
    )
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="start with given live configs.  single json file or dir with multiple json files",
    )
    parser.add_argument(
        "-i", "--iters", type=int, required=False, dest="iters", default=None, help="n optimize iters"
    )
    parser.add_argument(
        "-c", "--n_cpus", type=int, required=False, dest="n_cpus", default=None, help="n cpus"
    )
    parser.add_argument(
        "-le",
        "--long",
        type=str,
        required=False,
        dest="long_enabled",
        default=None,
        help="long enabled: [y/n]",
    )
    parser.add_argument(
        "-se",
        "--short",
        type=str,
        required=False,
        dest="short_enabled",
        default=None,
        help="short enabled: [y/n]",
    )
    parser.add_argument(
        "-pm",
        "--passivbot_mode",
        "--passivbot-mode",
        type=str,
        required=False,
        dest="passivbot_mode",
        default=None,
        help="passivbot mode options: [r/recursive_grid, n/neat_grid, c/clock]",
    )
    parser.add_argument(
        "-a",
        "--algo",
        "--algorithm",
        type=str,
        required=False,
        dest="algorithm",
        default=None,
        help="optimization algorithm options: [p/pso/particle_swarm_optimization, h/hs/harmony_search]",
    )
    parser.add_argument(
        "-ct",
        "--clip-threshold",
        "--clip_threshold",
        type=float,
        required=False,
        dest="clip_threshold",
        default=None,
        help="clip_threshold (see opt config for details)",
    )
    parser.add_argument(
        "-ser", "--serial", help="optimize symbols singly, not multi opt", action="store_true"
    )
    parser.add_argument(
        "-sm",
        "--skip_multicoin",
        "--skip-multicoin",
        type=str,
        required=False,
        dest="skip_multicoin",
        default=None,
        help="y/n when using --start dir/, skip multicoin configs (see opt config for details)",
    )
    parser.add_argument(
        "-ss",
        "--skip_singlecoin",
        "--skip-singlecoin",
        type=str,
        required=False,
        dest="skip_singlecoin",
        default=None,
        help="y/n when using --start dir/, skip single coin configs (see opt config for details)",
    )
    parser.add_argument(
        "-sns",
        "--skip_non_matching_single_coin",
        "--skip-non-matching-single-coin",
        type=str,
        required=False,
        dest="skip_non_matching_single_coin",
        default=None,
        help="y/n when using --start dir/, skip configs of other symbols (see opt config for details)",
    )
    parser.add_argument(
        "-sms",
        "--skip_matching_single_coin",
        "--skip-matching-single-coin",
        type=str,
        required=False,
        dest="skip_matching_single_coin",
        default=None,
        help="y/n when using --start dir/, skip configs of same symbol (see opt config for details)",
    )
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    config = prepare_optimize_config(args)
    args = parser.parse_args()
    pool = Pool(processes=config["n_cpus"])
    if algorithm is not None:
        args.algorithm = algorithm
    if args.serial:
        all_symbols = config["symbols"].copy()
        print(f"running single coin optimizations serially for symbols {all_symbols}")
        for symbol in all_symbols:
            args.symbols = symbol
            config = prepare_optimize_config(args)
            config["pool"] = pool
            await run_opt(args, config)
    else:
        config["pool"] = pool
        await run_opt(args, config)


async def run_opt(args, config):
    try:
        config.update(get_template_live_config(config["passivbot_mode"]))
        config["long"]["backwards_tp"] = config["backwards_tp_long"]
        config["short"]["backwards_tp"] = config["backwards_tp_short"]
        config["exchange"] = load_exchange_key_secret_passphrase(config["user"])[0]
        if args.long_enabled is None:
            config["long"]["enabled"] = config["do_long"]
        else:
            if "y" in args.long_enabled.lower():
                config["long"]["enabled"] = config["do_long"] = True
            elif "n" in args.long_enabled.lower():
                config["long"]["enabled"] = config["do_long"] = False
            else:
                raise Exception("please specify y/n with kwarg -le/--long")
        if args.short_enabled is None:
            config["short"]["enabled"] = config["do_short"]
        else:
            if "y" in args.short_enabled.lower():
                config["short"]["enabled"] = config["do_short"] = True
            elif "n" in args.short_enabled.lower():
                config["short"]["enabled"] = config["do_short"] = False
            else:
                raise Exception("please specify y/n with kwarg -le/--short")
        if args.symbols is not None:
            config["symbols"] = args.symbols.split(",")
        if args.n_cpus is not None:
            config["n_cpus"] = args.n_cpus
        if args.base_dir is not None:
            config["base_dir"] = args.base_dir
        if config["passivbot_mode"] == "clock":
            config["ohlcv"] = True
        print()
        lines = [
            (k, config[k])
            for k in config
            if any(isinstance(config[k], type_) for type_ in [str, float, int])
        ]
        for line in lines:
            logging.info(f"{line[0]: <{max([len(x[0]) for x in lines]) + 2}} {line[1]}")
        print()

        # download ticks .npy file if missing
        if config["ohlcv"]:
            cache_fname = f"{config['start_date']}_{config['end_date']}_ohlcv_cache.npy"
        else:
            cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
        exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")
        config["symbols"] = sorted(config["symbols"])
        config["ticks_caches"] = {}
        config["shared_memories"] = {}
        for symbol in config["symbols"]:
            cache_dirpath = os.path.join(config["base_dir"], exchange_name, symbol, "caches", "")
            # if config["ohlcv"] or (
            if not os.path.exists(cache_dirpath + cache_fname) or not os.path.exists(
                cache_dirpath + "market_specific_settings.json"
            ):
                logging.info(f"fetching data {symbol}")
                args.symbols = symbol
                tmp_cfg = prepare_backtest_config(args)
                if config["ohlcv"]:
                    data = await load_hlc_cache(
                        symbol,
                        tmp_cfg["inverse"],
                        tmp_cfg["start_date"],
                        config["end_date"],
                        base_dir=config["base_dir"],
                        spot=tmp_cfg["spot"],
                        exchange=tmp_cfg["exchange"],
                    )
                    """
                    config["shared_memories"][symbol] = shared_memory.SharedMemory(
                        create=True, size=data.nbytes
                    )
                    config["ticks_caches"][symbol] = np.ndarray(
                        data.shape, dtype=data.dtype, buffer=config["shared_memories"][symbol].buf
                    )
                    config["ticks_caches"][symbol][:] = data[:]
                    """
                else:
                    downloader = Downloader({**config, **tmp_cfg})
                    await downloader.get_sampled_ticks()

        # prepare starting configs
        cfgs = []
        if args.starting_configs is not None:
            if os.path.exists(args.starting_configs):
                if os.path.isdir(args.starting_configs):
                    # a directory was passed as starting config
                    fnames = [f for f in os.listdir(args.starting_configs) if f.endswith(".json")]

                    if "skip_multicoin" in config["starting_configs_filtering_conditions"]:
                        fnames = [f for f in fnames if "symbols" not in f]
                    if "skip_singlecoin" in config["starting_configs_filtering_conditions"]:
                        fnames = [f for f in fnames if "symbols" in f]
                    if (
                        "skip_non_matching_single_coin"
                        in config["starting_configs_filtering_conditions"]
                    ):
                        fnames = [f for f in fnames if "symbols" in f or config["symbols"][0] in f]
                    if "skip_matching_single_coin" in config["starting_configs_filtering_conditions"]:
                        fnames = [
                            f for f in fnames if "symbols" in f or config["symbols"][0] not in f
                        ]

                    for fname in fnames:
                        try:
                            cfg = load_live_config(os.path.join(args.starting_configs, fname))
                            assert (
                                determine_passivbot_mode(cfg) == config["passivbot_mode"]
                            ), "wrong passivbot mode"
                            cfgs.append(cfg)
                            logging.info(f"successfully loaded config {fname}")

                        except Exception as e:
                            logging.error(f"error loading config {fname}: {e}")
                elif args.starting_configs.endswith(".json"):
                    hm_load_failed = True
                    if "hm_" in args.starting_configs:
                        try:
                            hm = json.load(open(args.starting_configs))
                            for k in hm:
                                cfg = {
                                    "long": hm[k]["long"]["config"],
                                    "short": hm[k]["short"]["config"],
                                }
                                cfg = sort_dict_keys(numpyize(make_compatible(cfg)))
                                assert (
                                    determine_passivbot_mode(cfg) == config["passivbot_mode"]
                                ), "wrong passivbot mode in harmony memory"
                                cfgs.append(cfg)
                            logging.info(f"loaded harmony memory {args.starting_configs}")
                            hm_load_failed = False
                        except Exception as e:
                            logging.error(
                                f"error loading harmony memory {args.starting_configs}: {e}"
                            )
                    if hm_load_failed:
                        try:
                            cfg = load_live_config(args.starting_configs)
                            assert (
                                determine_passivbot_mode(cfg) == config["passivbot_mode"]
                            ), "wrong passivbot mode"
                            cfgs.append(cfg)
                            logging.info(f"successfully loaded config {args.starting_configs}")
                        except Exception as e:
                            logging.error(f"error loading config {args.starting_configs}: {e}")

        config["starting_configs"] = cfgs
        config["keys_to_include"] = [
            "starting_balance",
            "latency_simulation_ms",
            "market_type",
            "adg_n_subdivisions",
            "n_backtest_slices",
            "slim_analysis",
        ]

        if config["algorithm"] == "particle_swarm_optimization":
            from particle_swarm_optimization import ParticleSwarmOptimization

            particle_swarm_optimization = ParticleSwarmOptimization(
                config, backtest_wrap, config["pool"]
            )
            particle_swarm_optimization.run()
        elif config["algorithm"] == "harmony_search":
            from harmony_search import HarmonySearch

            harmony_search = HarmonySearch(config, backtest_wrap, config["pool"])
            harmony_search.run()
    finally:
        if "shared_memories" in config:
            for symbol in config["shared_memories"]:
                config["shared_memories"][symbol].close()
                config["shared_memories"][symbol].unlink()


if __name__ == "__main__":
    asyncio.run(main())
