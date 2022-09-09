import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import json
import pprint
import numpy as np
import argparse
import hjson
from procedures import load_live_config, dump_live_config, make_get_filepath
from pure_funcs import config_pretty_str, candidate_to_live_config


def main():

    parser = argparse.ArgumentParser(prog="view conf", description="inspect conf")
    parser.add_argument("results_fpath", type=str, help="path to results file")
    parser.add_argument(
        "-p",
        "--PAD",
        "--pad",
        dest="PAD_max",
        type=float,
        required=False,
        default=None,
        help="max pa dist",
    )
    parser.add_argument(
        "-l",
        "--profit_loss_ratio",
        dest="profit_loss_ratio_max",
        type=float,
        required=False,
        default=None,
        help="max profit loss ratio",
    )
    parser.add_argument(
        "-i", "--index", dest="index", type=int, required=False, default=1, help="best conf index"
    )
    parser.add_argument(
        "-sf",
        dest="score_formula",
        type=str,
        required=False,
        default="adgrealizedPADstd",
        help="choices: [adgPADstd, adg_mean, adg_min, adgPADmean, adgrealizedPADmean, adgrealizedPADstd]",
    )
    parser.add_argument(
        "-d",
        "--dump_live_config",
        action="store_true",
        help="dump config",
    )

    args = parser.parse_args()

    opt_config_path = (
        "configs/optimize/harmony_search.hjson"
        if "harmony" in args.results_fpath
        else "configs/optimize/particle_swarm_optimization.hjson"
    )

    if args.PAD_max is None:
        opt_config = hjson.load(open(opt_config_path))
        PAD_max_long = opt_config["maximum_pa_distance_std_long"]
        PAD_max_short = opt_config["maximum_pa_distance_std_short"]
    else:
        PAD_max_long = args.PAD_max
        PAD_max_short = args.PAD_max

    if args.profit_loss_ratio_max is None:
        opt_config = hjson.load(open(opt_config_path))
        maximum_loss_profit_ratio_long = opt_config["maximum_loss_profit_ratio_long"]
        maximum_loss_profit_ratio_short = opt_config["maximum_loss_profit_ratio_short"]
    else:
        maximum_loss_profit_ratio_long = args.profit_loss_ratio_max
        maximum_loss_profit_ratio_short = args.profit_loss_ratio_max

    keys_ = [
        "adg",
        "adg_realized_per_exposure",
        "pa_distance_mean",
        "pa_distance_std",
        "loss_profit_ratio",
        "profit_sum",
        "loss_sum",
    ]
    keys = [k + "_long" for k in keys_] + [k + "_short" for k in keys_]
    with open(args.results_fpath) as f:
        """
        results = []
        for x in f.readlines():
            j = json.loads(x)
            rsl = {
                "results": {
                    s: {k: j["results"][s][k] for k in keys if k in j["results"][s]}
                    for s in j["results"]
                    if s != "config_no"
                }
            }
            rsl["results"]["config_no"] = j["results"]["config_no"]
            rsl["config"] = j["config"]
            results.append(rsl)
        """
        results = [json.loads(x) for x in f.readlines()]

    print(
        "n results",
        len(results),
        f"score formula: {args.score_formula}, PAD max:",
        [PAD_max_long, PAD_max_short],
        f"loss_profit_ratio max:",
        [maximum_loss_profit_ratio_long, maximum_loss_profit_ratio_short],
    )
    best_config = {}
    adgs_sides = {"long": {}, "short": {}}
    PAD_max = {"long": PAD_max_long, "short": PAD_max_short}
    maximum_loss_profit_ratio = {
        "long": maximum_loss_profit_ratio_long,
        "short": maximum_loss_profit_ratio_short,
    }
    sides = ["long", "short"]
    for side in sides:
        stats = []
        for r in results:
            adgs, adgs_realized, PAD_stds, PAD_means, loss_profit_ratios = [], [], [], [], []
            for s in (rs := r["results"]):
                try:
                    adgs.append(rs[s][f"adg_{side}"])
                    adgs_realized.append(rs[s][f"adg_realized_per_exposure_{side}"])
                    PAD_stds.append(rs[s][f"pa_distance_std_{side}"])
                    PAD_means.append(rs[s][f"pa_distance_mean_{side}"])
                    if f"loss_profit_ratio_{side}" in rs[s]:
                        loss_profit_ratios.append(rs[s][f"loss_profit_ratio_{side}"])
                    else:
                        # for backwards compatibility
                        lpr = (
                            (abs(rs[s][f"loss_sum_{side}"]) / rs[s][f"profit_sum_{side}"])
                            if rs[s][f"profit_sum_{side}"]
                            else 1.0
                        )
                        rs[s][f"loss_profit_ratio_{side}"] = lpr
                        loss_profit_ratios.append(lpr)
                except Exception as e:
                    pass
            adg_mean = np.mean(adgs)
            adg_realized_mean = np.mean(adgs_realized)
            PAD_std_mean_raw = np.mean(PAD_stds)
            PAD_std_mean = np.mean([max(PAD_max[side], x) for x in PAD_stds])
            PAD_mean_mean_raw = np.mean(PAD_means)
            PAD_mean_mean = np.mean([max(PAD_max[side], x) for x in PAD_means])
            loss_profit_ratio_mean = np.mean(
                [max(maximum_loss_profit_ratio[side], x) for x in loss_profit_ratios]
            )
            loss_profit_ratio_mean_raw = np.mean(loss_profit_ratios)
            if args.score_formula.lower() == "adgpadstd":
                score = adg_mean / max(PAD_max[side], PAD_std_mean)
            elif args.score_formula.lower() == "adg_mean":
                score = adg_mean
            elif args.score_formula.lower() == "adg_min":
                score = min(adgs)
            elif args.score_formula.lower() == "adgpadmean":
                score = adg_mean * min(1, PAD_max[side] / PAD_mean_mean)
            elif args.score_formula.lower() == "adgrealizedpadmean":
                score = adg_realized_mean / max(PAD_max[side], PAD_mean_mean)
            elif args.score_formula.lower() == "adgrealizedpadstd":
                score = adg_realized_mean / max(PAD_max[side], PAD_std_mean)
            else:
                raise Exception("unknown score formula")

            score *= maximum_loss_profit_ratio[side] / max(
                loss_profit_ratio_mean, maximum_loss_profit_ratio[side]
            )

            stats.append(
                {
                    "config": r["config"],
                    "adg_mean": adg_mean,
                    "adg_realized_mean": adg_realized_mean,
                    "PAD_std_mean": PAD_std_mean,
                    "PAD_std_mean_raw": PAD_std_mean_raw,
                    "PAD_mean_mean": PAD_mean_mean,
                    "PAD_mean_mean_raw": PAD_mean_mean_raw,
                    "loss_profit_ratio_mean_raw": loss_profit_ratio_mean_raw,
                    "score": score,
                    "config_no": r["results"]["config_no"],
                }
            )
        ss = sorted(stats, key=lambda x: x["score"])
        bc = ss[-args.index]
        best_config[side] = bc["config"][side]
        for r in results:
            if r["results"]["config_no"] == bc["config_no"]:
                rs = r["results"]
                syms = [s for s in rs if "config" not in s]
                print(f"results {side} best config no {bc['config_no']}")
                print("symbol               adg      PADmean  PADstd   lp_ratio score")
                for s in sorted(syms, key=lambda x: rs[x][f"adg_realized_per_exposure_{side}"]):
                    adgs_sides[side][s] = rs[s][f"adg_realized_per_exposure_{side}"]
                    print(
                        f"{s: <20} {rs[s][f'adg_realized_per_exposure_{side}']:.6f} "
                        + f"{rs[s][f'pa_distance_std_{side}']:.6f} {rs[s][f'pa_distance_mean_{side}']:.6f} "
                        + f"{rs[s][f'loss_profit_ratio_{side}']:.6f} "
                        + f"{bc['score']:.6f}"
                    )
                print(
                    f"{'means': <20} {bc['adg_realized_mean']:.6f} "
                    + f"{bc['PAD_std_mean_raw']:.6f} "
                    + f"{bc['PAD_mean_mean_raw']:.6f} "
                    + f"{bc['loss_profit_ratio_mean_raw']:.6f} "
                )
                bcstats = r
        other_side = [x for x in sides if x != side][0]
        for symbol in r["results"]:
            if type(r["results"][symbol]) == dict:
                pprint.pprint({k: v for k, v in r["results"][symbol].items() if other_side not in k})
    live_config = candidate_to_live_config(best_config)
    if args.dump_live_config:
        lc_fpath = make_get_filepath(f"{args.results_fpath.replace('.txt', '_best_config.json')}")
        print(f"dump_live_config {lc_fpath}")
        dump_live_config(live_config, lc_fpath)
    adgs_sums = {s: adgs_sides["long"][s] + adgs_sides["short"][s] for s in adgs_sides["long"]}
    print("\nsum adgs")
    for k, v in sorted(adgs_sums.items(), key=lambda x: x[1]):
        print(f"{k: <12} {v:.6f}")
    print(config_pretty_str(live_config))


if __name__ == "__main__":
    main()
