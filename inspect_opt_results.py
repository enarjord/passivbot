import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import json
import pprint
import numpy as np
import argparse
from procedures import load_live_config
from pure_funcs import config_pretty_str, candidate_to_live_config


def main():

    parser = argparse.ArgumentParser(prog="view conf", description="inspect conf")
    parser.add_argument("results_dir", type=str, help="path to results dir")
    parser.add_argument(
        "-s", "--side", dest="side", type=str, required=False, default="long", help="long/short"
    )
    parser.add_argument(
        "-p", "--pad", dest="pad_max", type=float, required=False, default=0.035, help="max pa dist"
    )
    parser.add_argument(
        "-i", "--index", dest="index", type=int, required=False, default=1, help="best conf index"
    )

    parser.add_argument(
        "-f",
        dest="filename",
        type=str,
        required=False,
        default="all_results.txt",
        help="all_results fpath",
    )
    parser.add_argument(
        "-sf",
        dest="score_formula",
        type=str,
        required=False,
        default="adgpadstd",
        help="choices: [adgpadstd, adg, adgpadmean]",
    )

    args = parser.parse_args()

    side = args.side
    pad_max = args.pad_max

    with open(args.results_dir + args.filename) as f:
        results = [json.loads(x) for x in f.readlines()]

    stats = []
    print("n results", len(results), "score formula: adg / PADstd, PAD max:", pad_max)
    for r in results:
        adgs, pad_stds, pad_means, adg_DGstd_ratios = [], [], [], []
        for s in (rs := r["results"]):
            try:
                adgs.append(rs[s][f"adg_{side}"])
                pad_stds.append(max(pad_max, rs[s][f"pa_distance_std_{side}"]))
                pad_means.append(max(pad_max, rs[s][f"pa_distance_mean_{side}"]))
                adg_DGstd_ratios.append(rs[s][f"adg_DGstd_ratio_{side}"])
            except Exception as e:
                pass
        adg_mean = np.mean(adgs)
        pad_std_mean = np.mean(pad_stds)
        pad_mean_mean = np.mean(pad_means)
        adg_DGstd_ratios_mean = np.mean(adg_DGstd_ratios)
        adg_DGstd_ratios_std = np.std(adg_DGstd_ratios)
        if args.score_formula == 'adgpadstd':
            score = adg_mean / max(pad_max, pad_std_mean)
        elif args.score_formula == 'adg':
            score = adg_mean
        elif args.score_formula == 'adgpadmean':
            score = adg_mean * min(1, pad_max / pad_mean_mean)
        else:
            raise Exception('unknown score formula')
        stats.append(
            {
                "config": r["config"],
                "adg_mean": adg_mean,
                "pad_std_mean": pad_std_mean,
                "pad_mean_mean": pad_mean_mean,
                "score": score,
                "adg_DGstd_ratios_mean": adg_DGstd_ratios_mean,
                "adg_DGstd_ratios_std": adg_DGstd_ratios_std,
                "config_no": r["results"]["config_no"],
            }
        )
    ss = sorted(stats, key=lambda x: x["score"])
    bc = ss[-args.index]
    print(config_pretty_str(candidate_to_live_config(bc["config"])))
    pprint.pprint({k: v for k, v in bc.items() if k != "config"})
    for r in results:
        if r["results"]["config_no"] == bc["config_no"]:
            rs = r["results"]
            syms = [s for s in rs if "config" not in s]
            print("symbol               adg      PADmean  PADstd   adg/DGstd")
            for s in sorted(syms, key=lambda x: rs[x][f"adg_{side}"]):
                print(
                    f"{s: <20} {rs[s][f'adg_{side}'] / bc['config'][side]['wallet_exposure_limit']:.6f} "
                    + f"{rs[s][f'pa_distance_std_{side}']:.6f} {rs[s][f'pa_distance_mean_{side}']:.6f} "
                    + f"{rs[s][f'adg_DGstd_ratio_{side}']:.6f} "
                )
            print(
                f"{'means': <20} {bc['adg_mean'] / bc['config'][side]['wallet_exposure_limit']:.6f} "
                + f"{bc['pad_std_mean']:.6f} "
                + f"{bc['pad_mean_mean']:.6f} {bc['adg_DGstd_ratios_mean']:.6f}"
            )


if __name__ == "__main__":
    main()
