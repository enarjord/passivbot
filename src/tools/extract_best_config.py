import os
import json
import sys
import argparse
import traceback
from copy import deepcopy

import pandas as pd
import dictdiffer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pure_funcs import config_pretty_str, flatten_dict
from procedures import make_get_filepath, dump_config, format_config
from opt_utils import calc_dist, dominates_d, update_pareto_front, gprint
import matplotlib.pyplot as plt


def plot_pareto_front(
    objectives, pareto_indices, best_idx=None, ideal_point=None, title="Pareto Front"
):
    """
    Plots the Pareto front.

    Args:
        objectives: dict mapping index -> (w0, w1)
        pareto_indices: list of indices in the Pareto front
        best_idx: optional index of "best" candidate to highlight
        ideal_point: optional tuple of ideal point in original units
        title: plot title
    """
    pareto_w0 = [objectives[i][0] for i in pareto_indices]
    pareto_w1 = [objectives[i][1] for i in pareto_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_w0, pareto_w1, label="Pareto front", color="blue")

    if best_idx is not None:
        best_w0, best_w1 = objectives[best_idx]
        plt.scatter(
            best_w0,
            best_w1,
            label="Selected best candidate",
            color="red",
            marker="*",
            s=150,
        )

    if ideal_point is not None:
        plt.scatter(
            ideal_point[0],
            ideal_point[1],
            label=f"Ideal point ({ideal_point[0]:.4g}, {ideal_point[1]:.4g})",
            color="green",
            marker="x",
            s=100,
        )

    # Expand axis limits to include ideal point
    w0_all = pareto_w0 + ([ideal_point[0]] if ideal_point else [])
    w1_all = pareto_w1 + ([ideal_point[1]] if ideal_point else [])

    w0_min, w0_max = min(w0_all), max(w0_all)
    w1_min, w1_max = min(w1_all), max(w1_all)

    w0_padding = (w0_max - w0_min) * 0.1
    w1_padding = (w1_max - w1_min) * 0.1

    plt.xlim(w0_min - w0_padding, w0_max + w0_padding)
    plt.ylim(w1_min - w1_padding, w1_max + w1_padding)

    plt.xlabel("Objective w₀")
    plt.ylabel("Objective w₁")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def data_generator_all_results(filename: str, verbose: bool = False):
    prev_data = None
    file_size = os.path.getsize(filename)
    with open(filename, "r") as f:
        with tqdm(
            total=file_size, desc="Loading content", unit="B", unit_scale=True, disable=not verbose
        ) as pbar:
            for line in f:
                if verbose:
                    pbar.update(len(line.encode("utf-8")))
                try:
                    data = json.loads(line)
                    if "diff" not in data:
                        prev_data = data
                        yield deepcopy(prev_data)
                    else:
                        diff = data["diff"]
                        for i in range(len(diff)):
                            if len(diff[i]) == 2:
                                diff[i] = ("change", diff[i][0], (0.0, diff[i][1]))
                        prev_data = dictdiffer.patch(diff, prev_data)
                        yield deepcopy(prev_data)
                except Exception as e:
                    if verbose:
                        print(f"Error in data_generator: {e} line: {line}")
                    yield {}


def data_generator_pareto(filename: str, verbose: bool = False):
    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        with tqdm(
            total=file_size, desc="Loading pareto", unit="B", unit_scale=True, disable=not verbose
        ) as pbar:
            for line in f:
                if verbose:
                    pbar.update(len(line))
                try:
                    data = json.loads(line.decode("utf-8"))
                    yield data
                except Exception as e:
                    if verbose:
                        print(f"Error parsing line in pareto file: {e} line: {line}")


def process_single(file_location, verbose=False, plot=False):
    print_ = gprint(verbose)
    analysis_prefix = None
    analysis_key = None
    objectives = {}
    pareto = []
    index_to_entry = {}
    index = 0

    # Determine reader type
    if file_location.endswith("_all_results.txt"):
        generator = data_generator_all_results
    elif file_location.endswith("_pareto.jsonl"):
        generator = data_generator_pareto
    else:
        print_("Unknown file type")
        return None

    higher_is_better = [False, False]

    for x in generator(file_location, verbose=verbose):
        if not x:
            continue

        if analysis_prefix is None:
            if "analyses_combined" in x:
                analysis_prefix, analysis_key = "analyses_combined_", "analyses_combined"
            elif "analysis" in x:
                analysis_prefix, analysis_key = "analysis_", "analysis"
            else:
                continue

        flat_x = flatten_dict(x)
        w0_key, w1_key = analysis_prefix + "w_0", analysis_prefix + "w_1"
        if w0_key not in flat_x or w1_key not in flat_x:
            continue

        try:
            w0, w1 = float(flat_x[w0_key]), float(flat_x[w1_key])
        except:
            continue

        objectives[index] = (w0, w1)
        old_pareto = pareto.copy()
        pareto = update_pareto_front(index, (w0, w1), pareto, objectives, higher_is_better)
        if index not in old_pareto and index in pareto:
            index_to_entry[index] = deepcopy(x)

        index += 1

    print_(f"n backtests: {index}")

    if not pareto:
        print_("No Pareto candidates found.")
        return None

    # Compute min/max over Pareto front only
    pareto_w0 = [objectives[idx][0] for idx in pareto]
    pareto_w1 = [objectives[idx][1] for idx in pareto]
    mm = [min(pareto_w0), max(pareto_w0), min(pareto_w1), max(pareto_w1)]

    range_w0 = mm[1] - mm[0] if mm[1] != mm[0] else 1.0
    range_w1 = mm[3] - mm[2] if mm[3] != mm[2] else 1.0

    distances = []
    for idx in pareto:
        w0, w1 = objectives[idx]
        norm_w0 = (w0 - mm[0]) / range_w0
        norm_w1 = (w1 - mm[2]) / range_w1
        dist = calc_dist((norm_w0, norm_w1), (0.0, 0.0))
        distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])
    best_idx = distances[0][0]
    best_entry = index_to_entry.get(best_idx)

    if best_entry is None:
        print_("Best entry not found.")
        return None

    pareto_entries = [index_to_entry.get(idx[0]) for idx in distances if index_to_entry.get(idx[0])]

    pdf = pd.DataFrame([x[analysis_key] for x in pareto_entries])
    selected_columns = [x for x in pdf.columns if x.endswith("_mean")]
    pdf = pdf[selected_columns]
    pdf.columns = [
        x[:-5].replace("equity_balance", "eqbal").replace("position", "pos") for x in selected_columns
    ]

    print_(f"n pareto members: {len(pdf)}")
    for i in range(0, len(pdf.columns), 10):
        print_(pdf[pdf.columns[i : i + 10]])
        print_()

    best_entry[analysis_key]["n_iters"] = index
    if "config" in best_entry:
        best_entry.update(deepcopy(best_entry["config"]))
        del best_entry["config"]

    fjson = config_pretty_str(
        {
            "analyses": best_entry["analyses"],
            "backtest": {
                k: best_entry["backtest"][k] for k in best_entry["backtest"] if k != "coins"
            },
            "bot": best_entry["bot"],
            "optimize": best_entry["optimize"],
        }
    )
    print_("selected candidate:")
    print_(fjson)
    print_(file_location)

    full_path = file_location.replace("_all_results.txt", "").replace("_pareto.bin", "") + ".json"
    base_path = os.path.split(full_path)[0]
    full_path = make_get_filepath(full_path.replace(base_path, base_path + "_analysis/"))

    for entry in pareto_entries:
        if "config" in entry:
            entry.update(deepcopy(entry["config"]))
            del entry["config"]

    with open(full_path.replace(".json", "_pareto.txt"), "w") as f:
        for x in pareto_entries:
            f.write(json.dumps(x) + "\n")

    dump_config(format_config(best_entry), full_path)

    if plot:
        ideal_point = (mm[0], mm[2])
        plot_pareto_front(objectives, pareto, best_idx, ideal_point)

    return best_entry


def main(args):
    if os.path.isdir(args.file_location):
        for fname in sorted(os.listdir(args.file_location), reverse=True):
            fpath = os.path.join(args.file_location, fname)
            try:
                process_single(fpath, args.verbose)
                print(f"successfully processed {fpath}")
            except Exception as e:
                print(f"error with {fpath} {e}")
                traceback.print_exc()
    else:
        try:
            process_single(args.file_location, args.verbose, args.plot)
            print(f"successfully processed {args.file_location}")
        except Exception as e:
            print(f"error with {args.file_location} {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results.")
    parser.add_argument("file_location", type=str, help="Location of the results file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--plot", action="store_true", help="Show Pareto front plot")
    args = parser.parse_args()

    main(args)
