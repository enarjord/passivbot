import os
import json
import hjson
import pandas as pd
import argparse
import sys
import pprint
import dictdiffer
from tqdm import tqdm
import traceback

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


def data_generator(all_results_filename, verbose=False):
    """
    Generator function that iterates over an all_results.txt file written with dictdiffer.
    It yields the full data at each step by reconstructing it using diffs.

    Args:
        all_results_filename (str): Path to the all_results.txt file.
        verbose (bool): If True, disable all printing and progress tracking.

    Yields:
        dict: The full data dictionary at each step.
    """
    prev_data = None
    # Get the total file size in bytes
    file_size = os.path.getsize(all_results_filename)
    # Disable progress bar and printing if verbose is True
    with open(all_results_filename, "r") as f:
        with tqdm(
            total=file_size,
            desc="Loading content",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not verbose,
        ) as pbar:
            for line in f:
                if verbose:
                    pbar.update(len(line.encode("utf-8")))
                try:
                    data = json.loads(line)
                    if "diff" not in data:
                        # This is the first entry; full data is provided
                        prev_data = data
                        yield deepcopy(prev_data)
                    else:
                        # Apply the diff to the previous data to get the current data
                        diff = data["diff"]
                        for i in range(len(diff)):
                            if len(diff[i]) == 2:
                                diff[i] = ("change", diff[i][0], (0.0, diff[i][1]))
                        prev_data = dictdiffer.patch(diff, prev_data)
                        yield deepcopy(prev_data)
                except Exception as e:
                    if verbose:
                        print(
                            f"Error in data_generator: {e} Filename: {all_results_filename} line: {line}"
                        )
                    yield {}
            if not verbose:
                pbar.close()


# Function definitions remain unchanged
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
    xs = []
    for x in data_generator(file_location, verbose=verbose):
        if x:
            xs.append(x)
    if not xs:
        print_(f"No valid data found in {file_location}")
        return None
    print_("Processing...")
    res = pd.DataFrame([flatten_dict(x) for x in xs])

    # Determine the prefix based on the data
    if "analyses_combined" in xs[0]:
        analysis_prefix = "analyses_combined_"
        analysis_key = "analyses_combined"
    elif "analysis" in xs[0]:
        analysis_prefix = "analysis_"
        analysis_key = "analysis"
    else:
        raise Exception("Neither 'analyses_combined' nor 'analysis' found in data")

    keys, higher_is_better = ["w_0", "w_1"], [False, False]
    keys = [analysis_prefix + key for key in keys]
    print_("n backtests", len(res))

    # Adjust the filtering condition based on the prefix
    res_keys_w_0 = res[analysis_prefix + "w_0"]
    res_keys_w_1 = res[analysis_prefix + "w_1"]
    candidates = res[(res_keys_w_0 <= 0.0) & (res_keys_w_1 <= 0.0)][keys]
    if len(candidates) == 0:
        candidates = res[keys]
    print_("n candidates", len(candidates))
    if len(candidates) == 1:
        best = candidates.iloc[0].name
        pareto = candidates
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
        res_to_print = res[[x for x in res.columns if analysis_prefix[:-1] in x]].loc[
            closest_to_ideal.index
        ]
        res_to_print.columns = [x.replace(analysis_prefix, "") for x in res_to_print.columns]
        print_(res_to_print)

    # Processing the best result for configuration
    best_d = xs[best]
    # Adjust for 'analysis' or 'analyses_combined'
    best_d[analysis_key]["n_iters"] = len(xs)
    if "config" in best_d:
        best_d.update(deepcopy(best_d["config"]))
        del best_d["config"]
    fjson = config_pretty_str(best_d)
    print_(fjson)
    coins = [s.replace("USDT", "") for s in best_d["backtest"]["symbols"]]
    print_(file_location)
    full_path = file_location.replace("_all_results.txt", "") + ".json"
    base_path = os.path.split(full_path)[0]
    full_path = make_get_filepath(full_path.replace(base_path, base_path + "_analysis/"))
    pareto_to_dump = [x for i, x in enumerate(xs) if i in pareto.index]
    for i in range(len(pareto_to_dump)):
        if "config" in pareto_to_dump[i]:
            pareto_to_dump[i].update(deepcopy(pareto_to_dump[i]["config"]))
            del pareto_to_dump[i]["config"]
    with open(full_path.replace(".json", "_pareto.txt"), "w") as f:
        for x in pareto_to_dump:
            f.write(json.dumps(x) + "\n")
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
                traceback.print_exc()
    else:
        try:
            result = process_single(args.file_location, args.verbose)
            print(f"successfully processed {args.file_location}")
        except Exception as e:
            print(f"error with {args.file_location} {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results.")
    parser.add_argument("file_location", type=str, help="Location of the results file or directory")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Disable printing and progress tracking",
    )
    args = parser.parse_args()

    main(args)
