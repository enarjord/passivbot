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
    analyze_fills,
    denumpyize,
    get_template_live_config,
    ts_to_date,
    ts_to_date_utc,
    date_to_ts,
    tuplify,
    sort_dict_keys,
    determine_passivbot_mode,
    get_empty_analysis,
    calc_scores,
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
        fills_long, fills_short, stats = backtest(config, ticks)
        longs, shorts, sdf, analysis = analyze_fills(fills_long, fills_short, stats, config)
        logging.debug(
            f"backtested {config['symbol']: <12} pa distance long {analysis['pa_distance_mean_long']:.6f} "
            + f"pa distance short {analysis['pa_distance_mean_short']:.6f} adg long {analysis['adg_long']:.6f} "
            + f"adg short {analysis['adg_short']:.6f} std long {analysis['pa_distance_std_long']:.5f} "
            + f"std short {analysis['pa_distance_std_short']:.5f}"
        )
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
        "-o",
        "--optimize_config",
        type=str,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/default.hjson",
        help="optimize config hjson file",
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
        help="passivbot mode options: [s/static_grid, r/recursive_grid, n/neat_grid, c/clock]",
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
        "-ser", "--serial", help="optimize symbols singly, not multi opt", action="store_true"
    )
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    if args.symbol is None or "," in args.symbol:
        args.symbol = "BTCUSDT"  # dummy symbol
    config = await prepare_optimize_config(args)
    args = parser.parse_args()
    if algorithm is not None:
        args.algorithm = algorithm
    if args.symbol is not None:
        config["symbols"] = args.symbol.split(",")
    if args.serial:
        all_symbols = config["symbols"].copy()
        print(f"running single coin optimizations serially for symbols {all_symbols}")
        for symbol in all_symbols:
            args.symbol = symbol
            config = await prepare_optimize_config(args)
            await run_opt(args, config)
    else:
        await run_opt(args, config)


async def run_opt(args, config):
    try:
        if args.passivbot_mode is not None:
            if args.passivbot_mode in ["s", "static_grid", "static"]:
                config["passivbot_mode"] = "static_grid"
            elif args.passivbot_mode in ["r", "recursive_grid", "recursive"]:
                config["passivbot_mode"] = "recursive_grid"
            elif args.passivbot_mode in ["n", "neat_grid", "neat"]:
                config["passivbot_mode"] = "neat_grid"
            elif args.passivbot_mode in ["c", "clock"]:
                config["passivbot_mode"] = "clock"
            else:
                raise Exception(f"unknown passivbot mode {args.passivbot_mode}")
        algorithm = config["algorithm"] if args.algorithm is None else args.algorithm
        if algorithm in [
            "p",
            "pso",
            "particle_swarm_optimization",
            "particle-swarm-optimization",
        ]:
            config["algorithm"] = "particle_swarm_optimization"
        elif algorithm in ["h", "hs", "harmony_search", "harmony-search"]:
            config["algorithm"] = "harmony_search"
        else:
            raise Exception(f"unknown optimization algorithm {algorithm}")
        passivbot_mode = config["passivbot_mode"]
        assert passivbot_mode in [
            "recursive_grid",
            "static_grid",
            "neat_grid",
            "clock",
        ], f"unknown passivbot mode {passivbot_mode}"
        config.update(get_template_live_config(passivbot_mode))
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
        if args.symbol is not None:
            config["symbols"] = args.symbol.split(",")
        if args.n_cpus is not None:
            config["n_cpus"] = args.n_cpus
        if args.base_dir is not None:
            config["base_dir"] = args.base_dir
        if passivbot_mode == "clock":
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
                args.symbol = symbol
                tmp_cfg = await prepare_backtest_config(args)
                if config["ohlcv"]:
                    data = load_hlc_cache(
                        symbol,
                        config["start_date"],
                        config["end_date"],
                        base_dir=config["base_dir"],
                        spot=config["spot"],
                        exchange=config["exchange"],
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
        if config["algorithm"] == "particle_swarm_optimization":
            from particle_swarm_optimization import ParticleSwarmOptimization

            # prepare starting configs
            cfgs = []
            if args.starting_configs is not None:
                logging.info("preparing starting configs...")
                if os.path.isdir(args.starting_configs):
                    for fname in os.listdir(args.starting_configs):
                        try:
                            """
                            if config["symbols"][0] not in os.path.join(args.starting_configs, fname):
                                print("skipping", os.path.join(args.starting_configs, fname))
                                continue
                            """
                            cfg = load_live_config(os.path.join(args.starting_configs, fname))
                            assert (
                                determine_passivbot_mode(cfg) == passivbot_mode
                            ), "wrong passivbot mode"
                            cfgs.append(cfg)
                            logging.info(f"successfully loaded config {fname}")

                        except Exception as e:
                            logging.error(f"error loading config {fname}: {e}")
                elif os.path.exists(args.starting_configs):
                    try:
                        cfg = load_live_config(args.starting_configs)
                        assert determine_passivbot_mode(cfg) == passivbot_mode, "wrong passivbot mode"
                        cfgs.append(cfg)
                        logging.info(f"successfully loaded config {args.starting_configs}")
                    except Exception as e:
                        logging.error(f"error loading config {args.starting_configs}: {e}")
            config["starting_configs"] = cfgs
            particle_swarm_optimization = ParticleSwarmOptimization(config, backtest_wrap)
            particle_swarm_optimization.run()
        elif config["algorithm"] == "harmony_search":
            from harmony_search import HarmonySearch

            # prepare starting configs
            cfgs = []
            if args.starting_configs is not None:
                logging.info("preparing starting configs...")
                if os.path.isdir(args.starting_configs):
                    for fname in os.listdir(args.starting_configs):
                        try:
                            cfg = load_live_config(os.path.join(args.starting_configs, fname))
                            assert (
                                determine_passivbot_mode(cfg) == passivbot_mode
                            ), "wrong passivbot mode"
                            cfgs.append(cfg)
                            logging.info(f"successfully loaded config {fname}")

                        except Exception as e:
                            logging.error(f"error loading config {fname}: {e}")
                elif os.path.exists(args.starting_configs):
                    hm_load_failed = True
                    if "hm_" in args.starting_configs:
                        try:
                            hm = json.load(open(args.starting_configs))
                            for k in hm:
                                cfg = {
                                    "long": hm[k]["long"]["config"],
                                    "short": hm[k]["short"]["config"],
                                }
                                assert (
                                    determine_passivbot_mode(cfg) == passivbot_mode
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
                                determine_passivbot_mode(cfg) == passivbot_mode
                            ), "wrong passivbot mode"
                            cfgs.append(cfg)
                        except Exception as e:
                            logging.error(f"error loading config {args.starting_configs}: {e}")
            config["starting_configs"] = cfgs
            harmony_search = HarmonySearch(config, backtest_wrap)
            harmony_search.run()
    finally:
        for symbol in config["shared_memories"]:
            config["shared_memories"][symbol].close()
            config["shared_memories"][symbol].unlink()


if __name__ == "__main__":
    asyncio.run(main())
