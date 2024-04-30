import os
import json
import hjson
import pandas as pd
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from procedures import utc_ms, make_get_filepath
from pure_funcs import (
    flatten_dict,
    ts_to_date_utc,
    backtested_multiconfig2live_multiconfig,
)


# Function definitions
def calc_dist(p0, p1):
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5


def dominates_d(x, y, higher_is_better):
    """Check if point x dominates point y."""
    better_in_one = False
    for xi, yi, hib in zip(x, y, higher_is_better):
        if hib:
            if xi > yi:
                better_in_one = True
            elif xi < yi:
                return False
        else:
            if xi < yi:
                better_in_one = True
            elif xi > yi:
                return False
    return better_in_one


def calc_pareto_front_d(objectives: dict, higher_is_better: [bool]):
    sorted_keys = sorted(
        objectives,
        key=lambda k: [
            -objectives[k][i] if higher_is_better[i] else objectives[k][i]
            for i in range(len(higher_is_better))
        ],
    )
    pareto_front = []
    for kcandidate in sorted_keys:
        is_dominated = False
        for kmember in pareto_front:
            if dominates_d(objectives[kmember], objectives[kcandidate], higher_is_better):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front = [
                kmember
                for kmember in pareto_front
                if not dominates_d(objectives[kcandidate], objectives[kmember], higher_is_better)
            ]
            pareto_front.append(kcandidate)
    return pareto_front


def gprint(verbose):
    if verbose:
        return print
    else:
        return lambda *args, **kwargs: None


def process_single(file_location, verbose=False):
    print_ = gprint(verbose)
    try:
        result = json.load(open(file_location))
        print_(json.dumps(result, indent=4))
        return result
    except:
        pass
    with open(file_location) as f:
        lines = [x.strip() for x in f.readlines()]
    print_(f"n backtests: {len(lines)}")
    xs = [json.loads(x) for x in lines if x]
    res = pd.DataFrame([flatten_dict(x) for x in xs])

    worst_drawdown_lower_bound = res.iloc[0].args_worst_drawdown_lower_bound
    print_("worst_drawdown_lower_bound", worst_drawdown_lower_bound)

    keys, higher_is_better = ["w_adg_weighted", "w_sharpe_ratio"], [False, False]
    keys = ["analysis_" + key for key in keys]
    candidates = res[res.analysis_worst_drawdown <= worst_drawdown_lower_bound][keys]
    print_("n candidates", len(candidates))
    pareto = candidates.loc[
        calc_pareto_front_d(
            {i: x for i, x in zip(candidates.index, candidates.values)}, higher_is_better
        )
    ]

    cands_norm = (candidates - candidates.min()) / (candidates.max() - candidates.min())
    pareto_norm = (pareto - candidates.min()) / (candidates.max() - candidates.min())
    dists = [calc_dist(p, [float(x) for x in higher_is_better]) for p in pareto_norm.values]
    pareto_w_dists = pareto_norm.join(pd.Series(dists, name="dists", index=pareto_norm.index))
    closest_to_ideal = pareto_w_dists.sort_values("dists")
    best = closest_to_ideal.dists.idxmin()
    print_("best")
    print_(candidates.loc[best])
    print_("pareto front:")
    print_(pareto.loc[closest_to_ideal.index])

    # Processing the best result for configuration
    best_d = xs[best]
    best_d["analysis"]["n_iters"] = len(lines)
    cfg = best_d["live_config"]
    cfg["long"]["wallet_exposure_limit"] = cfg["global"]["TWE_long"] / len(best_d["args"]["symbols"])
    cfg["short"]["wallet_exposure_limit"] = cfg["global"]["TWE_short"] / len(
        best_d["args"]["symbols"]
    )
    cfg["long"]["enabled"] = best_d["args"]["long_enabled"]
    cfg["short"]["enabled"] = best_d["args"]["short_enabled"]
    fjson = json.dumps(best_d, indent=4, sort_keys=True)
    print_(fjson)
    coins = "".join([s.replace("USDT", "") for s in best_d["args"]["symbols"]])
    coins = [s.replace("USDT", "") for s in best_d["args"]["symbols"]]
    print_(file_location)
    fname = os.path.split(file_location)[-1].replace("_all_results.txt", "")
    coins_str = "_".join(coins) if len(coins) <= 5 else f"{len(coins)}_coins"
    if coins_str not in fname:
        fname += "_" + coins_str
    fname += ".json"
    full_path = make_get_filepath(os.path.join("results_multi_analysis", fname))
    json.dump(best_d, open(full_path, "w"), indent=4, sort_keys=True)
    return best_d


def main(args):
    if os.path.isdir(args.file_location):
        for fname in sorted(os.listdir(args.file_location), reverse=True):
            fpath = os.path.join(args.file_location, fname)
            try:
                process_single(fpath)
                print(f"successfully processed {fpath}")
            except Exception as e:
                print(f"error with {fpath} {e}")
    else:
        try:
            result = process_single(args.file_location, args.verbose)
            print(f"successfully processed {args.file_location}")
            if args.user is not None:
                live_config = backtested_multiconfig2live_multiconfig(result)
                live_config["user"] = args.user
                now = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
                fpath = f"configs/live/{now}_{args.user}.hjson"
                hjson.dump(live_config, open(fpath, "w"))
                print(f"successfully dumped live config {fpath}")
        except Exception as e:
            print(f"error with {args.file_location} {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results.")
    parser.add_argument("file_location", type=str, help="Location of the results file or directory")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbosity",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=False,
        dest="user",
        default=None,
        help="if user is passed, generate live config",
    )
    args = parser.parse_args()

    main(args)
