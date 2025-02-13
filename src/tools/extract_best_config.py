import os
import json
import sys
import argparse
import traceback
from copy import deepcopy

import pandas as pd
import dictdiffer
from tqdm import tqdm

# Ensure modules from the parent directory are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project-specific imports
from pure_funcs import (
    config_pretty_str,
    flatten_dict,
)
from procedures import make_get_filepath, dump_config, format_config


def data_generator(all_results_filename: str, verbose: bool = False):
    """
    Generate data entries from a file line-by-line. If a line contains a 'diff',
    apply it (via dictdiffer.patch) to the previously yielded data.

    :param all_results_filename: Path to the file containing results data.
    :param verbose: Whether to show tqdm progress and error prints.
    :yield: A dictionary representing the cumulative state after each line.
    """
    prev_data = None
    file_size = os.path.getsize(all_results_filename)

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
                    # If there's no diff, just yield the data as is
                    if "diff" not in data:
                        prev_data = data
                        yield deepcopy(prev_data)
                    else:
                        # Apply the diff to the previous state
                        diff = data["diff"]
                        # Ensure the 'change' tuple for dictdiffer
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


def calc_dist(p0: tuple, p1: tuple) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    :param p0: The first point (x0, y0).
    :param p1: The second point (x1, y1).
    :return: The Euclidean distance between p0 and p1.
    """
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5


def dominates_d(x: tuple, y: tuple, higher_is_better: list) -> bool:
    """
    Determine if x dominates y in a Pareto sense, given a list indicating
    whether higher or lower is better for each objective.

    :param x: Tuple of objective values (e.g. (w0, w1)) for candidate x.
    :param y: Tuple of objective values (e.g. (w0, w1)) for candidate y.
    :param higher_is_better: List of booleans. If True at index i, a higher value
                             of x[i] is better. If False, a lower value is better.
    :return: True if x strictly dominates y in at least one objective and is not
             worse in any objective.
    """
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


def update_pareto_front(
    new_index: int, new_obj: tuple, current_front: list, objectives_dict: dict, higher_is_better: list
) -> list:
    """
    Given a new candidate with certain objective values, update the current Pareto
    front indices accordingly.

    :param new_index: Index of the new candidate.
    :param new_obj: Tuple of objective values for the new candidate.
    :param current_front: List of indices representing the current Pareto front.
    :param objectives_dict: Dictionary mapping index -> (objective tuple).
    :param higher_is_better: List of booleans indicating if higher is better.
    :return: The updated Pareto front list of indices.
    """
    # If the new candidate is dominated by any on the front, return as is
    for idx in current_front:
        existing_obj = objectives_dict[idx]
        if dominates_d(existing_obj, new_obj, higher_is_better):
            return current_front

    # Otherwise, remove those that are dominated by the new candidate
    new_front = []
    for idx in current_front:
        existing_obj = objectives_dict[idx]
        if not dominates_d(new_obj, existing_obj, higher_is_better):
            new_front.append(idx)
    new_front.append(new_index)
    return new_front


def gprint(verbose: bool):
    """
    Provide a 'conditional' print function based on verbosity.

    :param verbose: If True, return the built-in print. Else, return a no-op.
    :return: A function that prints only if verbose is True.
    """
    return print if verbose else (lambda *args, **kwargs: None)


def process_single(file_location: str, verbose: bool = False):
    """
    Process a single file of results. Collect and track Pareto-optimal objectives,
    then select the 'best' entry by minimizing Euclidean distance to (0,0) in
    normalized objective space.

    :param file_location: Path to a single results file.
    :param verbose: Whether to print additional info for debugging.
    :return: The best candidate dictionary (or None if no candidates found).
    """
    print_ = gprint(verbose)
    analysis_prefix = None
    analysis_key = None

    # Dictionaries keyed by incremental index: index -> (w0, w1)
    all_objectives = {}
    filtered_objectives = {}

    # Indices of the current Pareto front
    all_pareto = []
    filtered_pareto = []

    # Map from index -> full entry for retrieval
    index_to_entry = {}
    index = 0

    # Track min/max across w0, w1 for normalization
    all_min_w0 = all_max_w0 = None
    all_min_w1 = all_max_w1 = None
    filtered_min_w0 = filtered_max_w0 = None
    filtered_min_w1 = filtered_max_w1 = None

    higher_is_better = [False, False]  # We want to minimize w0, w1

    # Read the file line-by-line (with incremental diffs if present)
    for x in data_generator(file_location, verbose=verbose):
        if not x:
            continue

        # Determine the analysis key/prefix if not yet known
        if analysis_prefix is None:
            if "analyses_combined" in x:
                analysis_prefix = "analyses_combined_"
                analysis_key = "analyses_combined"
            elif "analysis" in x:
                analysis_prefix = "analysis_"
                analysis_key = "analysis"
            else:
                # No recognized analysis data found
                continue

        flat_x = flatten_dict(x)
        w0_key = analysis_prefix + "w_0"
        w1_key = analysis_prefix + "w_1"

        # Skip if we don't find the w0/w1 keys
        if w0_key not in flat_x or w1_key not in flat_x:
            continue

        # Convert to float, skip entry if it fails
        try:
            w0 = float(flat_x[w0_key])
            w1 = float(flat_x[w1_key])
        except:
            continue

        # Update overall min/max
        if all_min_w0 is None or w0 < all_min_w0:
            all_min_w0 = w0
        if all_max_w0 is None or w0 > all_max_w0:
            all_max_w0 = w0
        if all_min_w1 is None or w1 < all_min_w1:
            all_min_w1 = w1
        if all_max_w1 is None or w1 > all_max_w1:
            all_max_w1 = w1

        # Determine if it meets the "filtered" condition (both <= 0)
        is_filtered = w0 <= 0.0 and w1 <= 0.0
        if is_filtered:
            # Update filtered min/max
            if filtered_min_w0 is None or w0 < filtered_min_w0:
                filtered_min_w0 = w0
            if filtered_max_w0 is None or w0 > filtered_max_w0:
                filtered_max_w0 = w0
            if filtered_min_w1 is None or w1 < filtered_min_w1:
                filtered_min_w1 = w1
            if filtered_max_w1 is None or w1 > filtered_max_w1:
                filtered_max_w1 = w1

        # Update all_objectives and Pareto front
        all_objectives[index] = (w0, w1)
        old_all_pareto = all_pareto.copy()
        all_pareto = update_pareto_front(
            index, (w0, w1), all_pareto, all_objectives, higher_is_better
        )
        # If it's newly added to the Pareto front, store a deep copy
        if index not in old_all_pareto and index in all_pareto:
            index_to_entry[index] = deepcopy(x)

        # Update filtered_objectives and filtered Pareto front
        if is_filtered:
            filtered_objectives[index] = (w0, w1)
            old_filtered_pareto = filtered_pareto.copy()
            filtered_pareto = update_pareto_front(
                index, (w0, w1), filtered_pareto, filtered_objectives, higher_is_better
            )
            if index not in old_filtered_pareto and index in filtered_pareto:
                index_to_entry[index] = deepcopy(x)

        index += 1

    print_("Processing...")

    # Decide which Pareto front to pick from
    if len(filtered_pareto) > 0:
        candidates_indices = filtered_pareto
        min_w0, max_w0 = filtered_min_w0, filtered_max_w0
        min_w1, max_w1 = filtered_min_w1, filtered_max_w1
        candidates_objectives = filtered_objectives
    else:
        candidates_indices = all_pareto
        min_w0, max_w0 = all_min_w0, all_max_w0
        min_w1, max_w1 = all_min_w1, all_max_w1
        candidates_objectives = all_objectives

    if not candidates_indices:
        print_("No candidates found.")
        return None

    # Normalize distances and find the point closest to (0, 0)
    range_w0 = max_w0 - min_w0 if max_w0 != min_w0 else 1.0
    range_w1 = max_w1 - min_w1 if max_w1 != min_w1 else 1.0

    distances = []
    for idx in candidates_indices:
        w0, w1 = candidates_objectives[idx]
        norm_w0 = (w0 - min_w0) / range_w0
        norm_w1 = (w1 - min_w1) / range_w1
        dist = calc_dist((norm_w0, norm_w1), (0.0, 0.0))
        distances.append((idx, dist))

    # Sort by distance ascending
    distances.sort(key=lambda x: x[1])
    best_idx = distances[0][0]

    # Retrieve the best entry from index_to_entry
    best_entry = index_to_entry.get(best_idx)
    if best_entry is None:
        print_("Best entry not found.")
        return None

    # Build a list of Pareto entries sorted by distance
    pareto_entries = [index_to_entry.get(idx[0]) for idx in distances]
    pareto_entries = [e for e in pareto_entries if e is not None]

    # Create a DataFrame to show relevant 'analyses_combined' columns
    pdf = pd.DataFrame([x["analyses_combined"] for x in pareto_entries])
    selected_columns = [x for x in pdf.columns if x.endswith("_mean")]
    pdf = pdf[selected_columns]
    pdf.columns = [
        x[:-5].replace("equity_balance", "eqbal").replace("position", "pos") for x in selected_columns
    ]

    n_cols = 10
    print_("n pareto members", len(pdf))
    # Print the DataFrame in chunks for readability
    for i in range(0, len(pdf.columns), n_cols):
        print_(pdf[pdf.columns[i : i + n_cols]])
        print_()

    # Set the best entry's "n_iters" for clarity
    best_d = best_entry
    best_d[analysis_key]["n_iters"] = index

    # If there's a "config" key, flatten it into the main dictionary
    if "config" in best_d:
        best_d.update(deepcopy(best_d["config"]))
        del best_d["config"]

    # Print selected candidate info
    fjson = config_pretty_str(
        {
            "analysis": {k: best_d["analyses_combined"][k] for k in selected_columns},
            "backtest": {k: best_d["backtest"][k] for k in best_d["backtest"] if k != "coins"},
            "bot": best_d["bot"],
        }
    )
    print_("selected candidate:")
    print_(fjson)
    print_(file_location)

    # Determine output paths
    full_path = file_location.replace("_all_results.txt", "") + ".json"
    base_path = os.path.split(full_path)[0]
    full_path = make_get_filepath(full_path.replace(base_path, base_path + "_analysis/"))

    # Flatten out "config" in each Pareto entry if present
    for entry in pareto_entries:
        if "config" in entry:
            entry.update(deepcopy(entry["config"]))
            del entry["config"]

    # Write all Pareto entries to a file
    with open(full_path.replace(".json", "_pareto.txt"), "w") as f:
        for x in pareto_entries:
            f.write(json.dumps(x) + "\n")

    # Dump the best config to disk
    dump_config(format_config(best_d), full_path)
    return best_d


def main(args):
    """
    Main entry point of the script. Processes either a single file or
    an entire directory of files.

    :param args: Parsed command-line arguments containing:
        - file_location: Path to file or directory.
        - verbose: Boolean indicating verbosity.
    """
    if os.path.isdir(args.file_location):
        # Process every file in the directory in reverse-sorted order
        for fname in sorted(os.listdir(args.file_location), reverse=True):
            fpath = os.path.join(args.file_location, fname)
            try:
                process_single(fpath)
                print(f"successfully processed {fpath}")
            except Exception as e:
                print(f"error with {fpath} {e}")
                traceback.print_exc()
    else:
        # Process a single file
        try:
            result = process_single(args.file_location, args.verbose)
            print(f"successfully processed {args.file_location}")
        except Exception as e:
            print(f"error with {args.file_location} {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results.")
    parser.add_argument("file_location", type=str, help="Location of the results file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    main(args)
