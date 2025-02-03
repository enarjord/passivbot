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
from copy import deepcopy

# Append parent directories to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pure_funcs import (
    config_pretty_str,
    flatten_dict,
    ts_to_date_utc,
    backtested_multiconfig2live_multiconfig,
    sort_dict_keys,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from procedures import utc_ms, make_get_filepath, dump_config, format_config


def data_generator(all_results_filename, verbose=False):
    """
    Generator function that iterates over an all_results.txt file written with dictdiffer.
    It yields the full data at each step by reconstructing it using diffs.

    Args:
        all_results_filename (str): Path to the all_results.txt file.
        verbose (bool): If True, enables printing and progress tracking.

    Yields:
        dict: The full data dictionary at each step.
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
                    if "diff" not in data:
                        # First entry; full data provided.
                        prev_data = data
                        yield deepcopy(prev_data)
                    else:
                        # Apply the diff to the previous data.
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
            if dominates_d(
                objectives[kmember], objectives[kcandidate], higher_is_better
            ):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front = [
                kmember
                for kmember in pareto_front
                if not dominates_d(
                    objectives[kcandidate], objectives[kmember], higher_is_better
                )
            ]
            pareto_front.append(kcandidate)
    return pareto_front


def gprint(verbose):
    if verbose:
        return print
    else:
        return lambda *args, **kwargs: None


def process_batch(batch, analysis_prefix, verbose):
    """
    Process a batch of records to select its best candidate.

    This function:
      1. Flattens each record in the batch.
      2. Converts the flattened records into a DataFrame.
      3. Filters for candidate rows (using keys 'w_0' and 'w_1' with the appropriate prefix).
      4. Computes the Pareto front and selects the candidate closest to the ideal point.

    Args:
        batch (list): List of records (dicts).
        analysis_prefix (str): Prefix for analysis keys (e.g., "analysis_" or "analyses_combined_").
        verbose (bool): Controls printing.

    Returns:
        dict: The best candidate from the batch.
    """
    print_ = gprint(verbose)
    # Flatten all records in the batch.
    flat_records = [flatten_dict(x) for x in batch]
    df = pd.DataFrame(flat_records)
    keys = [analysis_prefix + "w_0", analysis_prefix + "w_1"]
    higher_is_better = [False, False]

    if keys[0] in df.columns and keys[1] in df.columns:
        candidates = df[(df[keys[0]] <= 0.0) & (df[keys[1]] <= 0.0)][keys]
        if candidates.empty:
            candidates = df[keys]
    else:
        candidates = df

    if len(candidates) == 1:
        best_index = candidates.index[0]
    else:
        obj_dict = {i: candidates.loc[i, keys].tolist() for i in candidates.index}
        pareto_indices = calc_pareto_front_d(obj_dict, higher_is_better)
        if len(pareto_indices) == 1:
            best_index = pareto_indices[0]
        else:
            norm_candidates = (candidates - candidates.min()) / (
                candidates.max() - candidates.min()
            )
            norm_pareto = norm_candidates.loc[pareto_indices]
            distances = [calc_dist(row, [0.0, 0.0]) for row in norm_pareto.values]
            best_index = norm_pareto.index[distances.index(min(distances))]
            print_("Best candidate in batch:")
            print_(candidates.loc[best_index])
            print_("Pareto front:")
            res_to_print = df[
                [col for col in df.columns if analysis_prefix[:-1] in col]
            ].loc[norm_pareto.index]
            res_to_print.columns = [
                col.replace(analysis_prefix, "") for col in res_to_print.columns
            ]
            print_(res_to_print)
    return batch[best_index]


def process_single(file_location, verbose=False, batch_size=1000):
    """
    Process a single results file in batches.

    Instead of loading all records into memory, records are read sequentially
    and processed in batches. For each batch, the best candidate is selected.
    Finally, the list of batch winners is reduced to determine the overall best candidate.

    Args:
        file_location (str): Path to the results file.
        verbose (bool): Enable verbose output.
        batch_size (int): Number of records to process per batch.

    Returns:
        dict: The final best candidate.
    """
    print_ = gprint(verbose)

    # First, try to load the file as a full JSON (shortcut if possible)
    try:
        result = json.load(open(file_location))
        print_(config_pretty_str(sort_dict_keys(result)))
        return result
    except:
        pass

    # Use the generator to read the file sequentially.
    gen = data_generator(file_location, verbose=verbose)
    first = None
    for record in gen:
        if record:
            first = record
            break
    if not first:
        print_(f"No valid data found in {file_location}")
        return None

    # Determine analysis prefix based on the first record.
    if "analyses_combined" in first:
        analysis_prefix = "analyses_combined_"
        analysis_key = "analyses_combined"
    elif "analysis" in first:
        analysis_prefix = "analysis_"
        analysis_key = "analysis"
    else:
        raise Exception("Neither 'analyses_combined' nor 'analysis' found in data")

    total_records = 1  # first record already read
    batch_candidates = []
    current_batch = [first]

    # Read remaining records in batches.
    for record in gen:
        if record:
            current_batch.append(record)
            total_records += 1
        if len(current_batch) >= batch_size:
            candidate = process_batch(current_batch, analysis_prefix, verbose)
            batch_candidates.append(candidate)
            current_batch = []
    if current_batch:
        candidate = process_batch(current_batch, analysis_prefix, verbose)
        batch_candidates.append(candidate)

    print_(
        f"Processed {total_records} records in batches; found {len(batch_candidates)} batch winners."
    )

    # Final reduction: process the list of batch winners to get the overall best candidate.
    final_candidate = process_batch(batch_candidates, analysis_prefix, verbose)

    # Update configuration information.
    final_candidate[analysis_key]["n_iters"] = total_records
    if "config" in final_candidate:
        final_candidate.update(deepcopy(final_candidate["config"]))
        del final_candidate["config"]

    fjson = config_pretty_str(final_candidate)
    print_(fjson)
    print_(file_location)

    full_path = file_location.replace("_all_results.txt", "") + ".json"
    base_path = os.path.split(full_path)[0]
    full_path = make_get_filepath(
        full_path.replace(base_path, base_path + "_analysis/")
    )

    # Dump the batch winners (for reference) and the final configuration.
    with open(full_path.replace(".json", "_pareto.txt"), "w") as f:
        for candidate in batch_candidates:
            f.write(json.dumps(candidate) + "\n")
    dump_config(format_config(final_candidate), full_path)
    return final_candidate


def main(args):
    if os.path.isdir(args.file_location):
        for fname in sorted(os.listdir(args.file_location), reverse=True):
            fpath = os.path.join(args.file_location, fname)
            try:
                process_single(fpath, args.verbose, args.batch_size)
                print(f"successfully processed {fpath}")
            except Exception as e:
                print(f"error with {fpath}: {e}")
                traceback.print_exc()
    else:
        try:
            result = process_single(args.file_location, args.verbose, args.batch_size)
            print(f"successfully processed {args.file_location}")
        except Exception as e:
            print(f"error with {args.file_location}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results in batches.")
    parser.add_argument(
        "file_location", type=str, help="Location of the results file or directory"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose printing and progress tracking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of records to process per batch (default: 1000)",
    )
    args = parser.parse_args()
    main(args)
