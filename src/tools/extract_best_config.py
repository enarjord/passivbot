import os
import json
import hjson
import pandas as pd
import argparse
import sys
import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pure_funcs import config_pretty_str
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from procedures import utc_ms, make_get_filepath, dump_config, format_config
from pure_funcs import (
    flatten_dict,
    ts_to_date_utc,
    backtested_multiconfig2live_multiconfig,
    sort_dict_keys,
    config_pretty_str,
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
        print_(config_pretty_str(sort_dict_keys(result)))
        return result
    except:
        pass
    with open(file_location) as f:
        lines = [x.strip() for x in f.readlines()]
    print_(f"n backtests: {len(lines)}")
    xs = [json.loads(x) for x in lines if x]
    res = pd.DataFrame([flatten_dict(x) for x in xs])

    keys, higher_is_better = ["w_0", "w_1"], [False, False]
    keys = ["analysis_" + key for key in keys]
    candidates = res[(res.analysis_w_0 <= 0.0) & (res.analysis_w_1 <= 0.0)][keys]
    print_("n candidates", len(candidates))
    if len(candidates) == 1:
        best = candidates.iloc[0].name
    else:
        pareto = candidates.loc[
            calc_pareto_front_d(
                {i: x for i, x in zip(candidates.index, candidates.values)}, higher_is_better
            )
        ]

        cands_norm = (candidates - candidates.min()) / (candidates.max() - candidates.min())
        pareto_norm = (pareto - candidates.min()) / (candidates.max() - candidates.min())
        dists = [calc_dist(p, [float(x) for x in higher_is_better]) for p in pareto_norm.values]
        pareto_w_dists = pareto_norm.join(
            pd.Series(dists, name="dist_to_ideal", index=pareto_norm.index)
        )
        closest_to_ideal = pareto_w_dists.sort_values("dist_to_ideal")
        best = closest_to_ideal.dist_to_ideal.idxmin()
        print_("best")
        print_(candidates.loc[best])
        print_("pareto front:")
        res_to_print = res[[x for x in res.columns if "analysis" in x]].loc[closest_to_ideal.index]
        res_to_print.columns = [x.replace("analysis_", "") for x in res_to_print.columns]
        print_(res_to_print)

        # Processing the best result for configuration
    best_d = xs[best]
    best_d["analysis"]["n_iters"] = len(lines)
    best_d.update(deepcopy(best_d["config"]))
    del best_d["config"]
    fjson = config_pretty_str(best_d)
    print_(fjson)
    coins = [s.replace("USDT", "") for s in best_d["backtest"]["symbols"]]
    print_(file_location)
    full_path = file_location.replace("_all_results.txt", "") + ".json"
    base_path = os.path.split(full_path)[0]
    full_path = make_get_filepath(full_path.replace(base_path, base_path + "_analysis/"))
    dump_config(format_config(best_d), full_path)
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
    args = parser.parse_args()

    main(args)
