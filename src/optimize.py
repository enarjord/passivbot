import os
import sys

if sys.platform.startswith("win"):
    # ==== BEGIN fcntl stub for Windows ====
    try:
        import fcntl
    except ImportError:
        # create a fake module so later `import fcntl` works without error
        class _FcntlStub:
            LOCK_EX = None
            LOCK_SH = None
            LOCK_UN = None

            def lockf(self, *args, **kwargs):
                pass

            def ioctl(self, *args, **kwargs):
                pass

        sys.modules["fcntl"] = _FcntlStub()
        fcntl = sys.modules["fcntl"]
    # ==== END fcntl stub for Windows ====
import shutil
import passivbot_rust as pbr
import asyncio
import argparse
import multiprocessing
import mmap
from multiprocessing import Queue, Process
from collections import defaultdict
from contextlib import nullcontext
from backtest import (
    prepare_hlcvs_mss,
    prep_backtest_args,
    expand_analysis,
)
from pure_funcs import (
    get_template_live_config,
    symbol_to_coin,
    ts_to_date_utc,
    denumpyize,
    sort_dict_keys,
    calc_hash,
    flatten,
    date_to_ts,
)
from procedures import (
    make_get_filepath,
    utc_ms,
    load_hjson_config,
    load_config,
    format_config,
    add_arguments_recursively,
    update_config_with_args,
)
from downloader import add_all_eligible_coins_to_config
from copy import deepcopy
from main import manage_rust_compilation
import numpy as np
from uuid import uuid4
import logging
import traceback
import json
import pprint
from deap import base, creator, tools, algorithms
from contextlib import contextmanager
import tempfile
import time
import math
import fcntl
from tqdm import tqdm
from optimizer_overrides import optimizer_overrides
from opt_utils import make_json_serializable, generate_incremental_diff, round_floats
from pareto_store import ParetoStore
import msgpack
from typing import Sequence, Tuple, List

logging.basicConfig(
    format="%(asctime)s %(processName)-12s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


TEMPLATE_CONFIG_MODE = "v7"

# === bounds helpers =========================================================

Bound = Tuple[float, float]  # (low, high)


def enforce_bounds(
    values: Sequence[float], bounds: Sequence[Bound], sig_digits: int = None
) -> List[float]:
    """
    Clamp each value to its corresponding [low, high] interval.
    Also round to significant digits (optional).

    Args:
        values : iterable of floats (length == len(bounds))
        bounds : iterable of (low, high) pairs
        sig_digits: int

    Returns
        List[float]  – clamped copy (original is *not* modified)
    """
    assert len(values) == len(bounds), "values/bounds length mismatch"
    rounded = values if sig_digits is None else round_floats(values, sig_digits)
    return [high if v > high else low if v < low else v for v, (low, high) in zip(rounded, bounds)]


def extract_bounds_tuple_list_from_config(config) -> [Bound]:
    """
    extracts list of tuples (low, high) which are lower and upper bounds for bot parameters.
    also sets all bounds to (low, low) if pside is not enabled.
    """

    def extract_bound_vals(key, val) -> tuple:
        if isinstance(val, (float, int)):
            return (val, val)
        elif isinstance(val, (tuple, list)):
            if len(val) == 1:
                return (val[0], val[0])
            elif len(val) == 2:
                return tuple(sorted([val[0], val[1]]))
        raise Exception(f"malformed bound {key}: {val}")

    template_config = get_template_live_config(
        TEMPLATE_CONFIG_MODE
    )  # single source of truth for key names
    keys_ignored = get_bound_keys_ignored()
    bounds = []
    for pside in sorted(template_config["bot"]):
        is_enabled = all(
            [
                extract_bound_vals(k, config["optimize"]["bounds"][k])[1] > 0.0
                for k in [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
            ]
        )
        for key in sorted(template_config["bot"][pside]):
            if key in keys_ignored:
                continue
            bound_key = f"{pside}_{key}"
            assert (
                bound_key in config["optimize"]["bounds"]
            ), f"bound {bound_key} missing from optimize.bounds"
            bound_vals = extract_bound_vals(bound_key, config["optimize"]["bounds"][bound_key])
            bounds.append(bound_vals if is_enabled else (bound_vals[0], bound_vals[0]))
    return bounds


def get_bound_keys_ignored():
    return ["enforce_exposure_limit"]


# ============================================================================


def results_writer_process(
    queue,
    results_dir,
    sig_digits,
    flush_interval,
    *,
    compress: bool = True,
    write_all_results: bool = True,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)-12s %(levelname)-8s %(message)s",
    )
    log = logging.getLogger("optimizer.pareto")
    store = ParetoStore(
        directory=results_dir,
        sig_digits=sig_digits,
        flush_interval=flush_interval,
        log_name="optimizer.pareto",
    )

    pareto_front = []
    objectives_dict = {}
    index_to_entry = {}
    iteration = 0
    n_objectives = None
    scoring_keys = None

    results_filename = os.path.join(results_dir, "all_results.bin")

    try:
        with open(results_filename, "ab") if write_all_results else nullcontext() as f:
            packer = msgpack.Packer(use_bin_type=True) if write_all_results else None
            prev_data = None
            counter = 0
            while True:
                data = queue.get()
                if data == "DONE":
                    store.flush_now()
                    break
                if write_all_results:
                    try:
                        # Write raw results (diffed if compress enabled)
                        if compress:
                            if prev_data is None or counter % 100 == 0:
                                output_data = make_json_serializable(data)
                            else:
                                diff = generate_incremental_diff(prev_data, data)
                                output_data = make_json_serializable(diff)
                            counter += 1
                            prev_data = data
                        else:
                            output_data = data

                        if scoring_keys is None:
                            scoring_keys = data["optimize"]["scoring"]
                            n_objectives = len(scoring_keys)

                        # --- Write to all_results.bin ---
                        f.write(packer.pack(output_data))
                        f.flush()
                    except Exception as e:
                        logging.error(f"Error writing results: {e}")
                try:
                    store.add_entry(data)
                except Exception as e:
                    logging.error(f"ParetoStore error: {e}")

    except Exception as e:
        logging.error(f"Results writer process error: {e}")
    finally:
        # ------------------------------------------------------------------
        # Make *absolutely* sure the Pareto directory has fresh distance
        # prefixes before we quit (even after Ctrl-C or an uncaught error).
        # ------------------------------------------------------------------
        try:
            store.flush_interval = 0.0
            store.flush_now()
            logging.info("Final Pareto-front update completed.")
        except Exception as e1:
            logging.error(f"Unable to flush Pareto front on shutdown: {e1}")
            traceback.print_exc()


def create_shared_memory_file(hlcvs):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    logging.info(f"Creating shared memory file: {temp_file.name}...")
    shared_memory_file = temp_file.name
    temp_file.close()

    try:
        total_size = hlcvs.nbytes
        chunk_size = 1024 * 1024  # 1 MB chunks
        hlcvs_bytes = hlcvs.tobytes()

        with open(shared_memory_file, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Writing to shared memory"
            ) as pbar:
                for i in range(0, len(hlcvs_bytes), chunk_size):
                    chunk = hlcvs_bytes[i : i + chunk_size]
                    f.write(chunk)
                    pbar.update(len(chunk))

    except IOError as e:
        logging.error(f"Error writing to shared memory file: {e}")
        raise
    logging.info(f"Done creating shared memory file")
    return shared_memory_file


def check_disk_space(path, required_space):
    total, used, free = shutil.disk_usage(path)
    logging.info(
        f"Disk space - Total: {total/(1024**3):.2f} GB, Used: {used/(1024**3):.2f} GB, Free: {free/(1024**3):.2f} GB"
    )
    if free < required_space:
        raise IOError(
            f"Not enough disk space. Required: {required_space/(1024**3):.2f} GB, Available: {free/(1024**3):.2f} GB"
        )


def mutPolynomialBoundedWrapper(individual, eta, low, up, indpb):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        indpb: Independent probability for each attribute to be mutated.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions
    # This adjustment is arbitrary and won't affect the outcome since the mutation
    # won't be effective in these dimensions
    temp_low = np.where(equal_bounds_mask, low_array - 1e-6, low_array)
    temp_up = np.where(equal_bounds_mask, up_array + 1e-6, up_array)

    # Call the original mutPolynomialBounded function with the temporarily adjusted bounds
    tools.mutPolynomialBounded(individual, eta, list(temp_low), list(temp_up), indpb)

    # Reset values in dimensions with originally equal bounds to ensure they remain unchanged
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            individual[i] = low[i]

    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, low, up):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.

    Returns:
        A tuple of two individuals after crossover operation.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions to prevent division by zero
    # This adjustment is arbitrary and won't affect the outcome since the crossover
    # won't modify these dimensions
    low_array[equal_bounds_mask] -= 1e-6
    up_array[equal_bounds_mask] += 1e-6

    # Call the original cxSimulatedBinaryBounded function with adjusted bounds
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta, list(low_array), list(up_array))

    # Ensure that values in dimensions with originally equal bounds are reset
    # to the bound value (since they should not be modified)
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            ind1[i] = low[i]
            ind2[i] = low[i]

    return ind1, ind2


def individual_to_config(individual, optimizer_overrides, overrides_list, template):
    """
    assume individual is already bound enforced (or will be after)
    """
    keys_ignored = get_bound_keys_ignored()
    config = deepcopy(template)
    i = 0
    for pside in sorted(config["bot"]):
        for key in sorted(config["bot"][pside]):
            if key in keys_ignored:
                continue
            config["bot"][pside][key] = individual[i]
            i += 1
        config = optimizer_overrides(overrides_list, config, pside)

    return config


def config_to_individual(config, bounds, sig_digits=None):
    keys_ignored = get_bound_keys_ignored()
    return enforce_bounds(
        [
            config["bot"][pside][key]
            for pside in sorted(config["bot"])
            for key in sorted(config["bot"][pside])
            if key not in keys_ignored
        ],
        bounds,
        sig_digits,
    )


@contextmanager
def managed_mmap(filename, dtype, shape):
    mmap = None
    try:
        mmap = np.memmap(filename, dtype=dtype, mode="r", shape=shape)
        yield mmap
    except FileNotFoundError:
        if shutdown_event.is_set():
            yield None
        else:
            raise
    finally:
        if mmap is not None:
            del mmap


def validate_array(arr, name):
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains inf values")


class Evaluator:
    def __init__(
        self,
        shared_memory_files,
        hlcvs_shapes,
        hlcvs_dtypes,
        btc_usd_shared_memory_files,
        btc_usd_dtypes,
        msss,
        config,
        results_queue,
        seen_hashes=None,
        duplicate_counter=None,
    ):
        logging.info("Initializing Evaluator...")
        self.shared_memory_files = shared_memory_files
        self.hlcvs_shapes = hlcvs_shapes
        self.hlcvs_dtypes = hlcvs_dtypes
        self.btc_usd_shared_memory_files = btc_usd_shared_memory_files
        self.btc_usd_dtypes = btc_usd_dtypes
        self.msss = msss
        self.exchanges = list(shared_memory_files.keys())

        self.mmap_contexts = {}
        self.shared_hlcvs_np = {}
        self.exchange_params = {}
        self.backtest_params = {}
        for exchange in self.exchanges:
            logging.info(f"Setting up managed_mmap for {exchange}...")
            self.mmap_contexts[exchange] = managed_mmap(
                self.shared_memory_files[exchange],
                self.hlcvs_dtypes[exchange],
                self.hlcvs_shapes[exchange],
            )
            self.shared_hlcvs_np[exchange] = self.mmap_contexts[exchange].__enter__()
            _, self.exchange_params[exchange], self.backtest_params[exchange] = prep_backtest_args(
                config, self.msss[exchange], exchange
            )
            logging.info(f"mmap_context entered successfully for {exchange}.")

        self.config = config
        logging.info("Evaluator initialization complete.")
        self.results_queue = results_queue
        self.seen_hashes = seen_hashes if seen_hashes is not None else {}
        self.duplicate_counter = duplicate_counter
        self.bounds = extract_bounds_tuple_list_from_config(self.config)
        self.sig_digits = config.get("optimize", {}).get("round_to_n_significant_digits", 6)
        self.scoring_weights = {
            "adg": -1.0,
            "adg_per_exposure_long": -1.0,
            "adg_per_exposure_short": -1.0,
            "adg_w": -1.0,
            "adg_w_per_exposure_long": -1.0,
            "adg_w_per_exposure_short": -1.0,
            "btc_adg": -1.0,
            "btc_adg_per_exposure_long": -1.0,
            "btc_adg_per_exposure_short": -1.0,
            "btc_adg_w": -1.0,
            "btc_adg_w_per_exposure_long": -1.0,
            "btc_adg_w_per_exposure_short": -1.0,
            "btc_calmar_ratio": -1.0,
            "btc_calmar_ratio_w": -1.0,
            "btc_drawdown_worst": 1.0,
            "btc_drawdown_worst_mean_1pct": 1.0,
            "btc_equity_balance_diff_neg_max": 1.0,
            "btc_equity_balance_diff_neg_mean": 1.0,
            "btc_equity_balance_diff_pos_max": 1.0,
            "btc_equity_balance_diff_pos_mean": 1.0,
            "btc_equity_choppiness": 1.0,
            "btc_equity_choppiness_w": 1.0,
            "btc_equity_jerkiness": 1.0,
            "btc_equity_jerkiness_w": 1.0,
            "btc_expected_shortfall_1pct": 1.0,
            "btc_exponential_fit_error": 1.0,
            "btc_exponential_fit_error_w": 1.0,
            "btc_gain": -1.0,
            "btc_gain_per_exposure_long": -1.0,
            "btc_gain_per_exposure_short": -1.0,
            "btc_loss_profit_ratio": 1.0,
            "btc_loss_profit_ratio_w": 1.0,
            "btc_mdg": -1.0,
            "btc_mdg_per_exposure_long": -1.0,
            "btc_mdg_per_exposure_short": -1.0,
            "btc_mdg_w": -1.0,
            "btc_mdg_w_per_exposure_long": -1.0,
            "btc_mdg_w_per_exposure_short": -1.0,
            "btc_omega_ratio": -1.0,
            "btc_omega_ratio_w": -1.0,
            "btc_sharpe_ratio": -1.0,
            "btc_sharpe_ratio_w": -1.0,
            "btc_sortino_ratio": -1.0,
            "btc_sortino_ratio_w": -1.0,
            "btc_sterling_ratio": -1.0,
            "btc_sterling_ratio_w": -1.0,
            "calmar_ratio": -1.0,
            "calmar_ratio_w": -1.0,
            "drawdown_worst": 1.0,
            "drawdown_worst_mean_1pct": 1.0,
            "equity_balance_diff_neg_max": 1.0,
            "equity_balance_diff_neg_mean": 1.0,
            "equity_balance_diff_pos_max": 1.0,
            "equity_balance_diff_pos_mean": 1.0,
            "equity_choppiness": 1.0,
            "equity_choppiness_w": 1.0,
            "equity_jerkiness": 1.0,
            "equity_jerkiness_w": 1.0,
            "expected_shortfall_1pct": 1.0,
            "exponential_fit_error": 1.0,
            "exponential_fit_error_w": 1.0,
            "gain": -1.0,
            "gain_per_exposure_long": -1.0,
            "gain_per_exposure_short": -1.0,
            "loss_profit_ratio": 1.0,
            "loss_profit_ratio_w": 1.0,
            "mdg": -1.0,
            "mdg_per_exposure_long": -1.0,
            "mdg_per_exposure_short": -1.0,
            "mdg_w": -1.0,
            "mdg_w_per_exposure_long": -1.0,
            "mdg_w_per_exposure_short": -1.0,
            "omega_ratio": -1.0,
            "omega_ratio_w": -1.0,
            "position_held_hours_max": 1.0,
            "position_held_hours_mean": 1.0,
            "position_held_hours_median": 1.0,
            "position_unchanged_hours_max": 1.0,
            "positions_held_per_day": 1.0,
            "sharpe_ratio": -1.0,
            "sharpe_ratio_w": -1.0,
            "sortino_ratio": -1.0,
            "sortino_ratio_w": -1.0,
            "sterling_ratio": -1.0,
            "sterling_ratio_w": -1.0,
            "volume_pct_per_day_avg": -1.0,
            "volume_pct_per_day_avg_w": -1.0,
        }

        self.build_limit_checks()

    def perturb_step_digits(self, individual, change_chance=0.5):
        perturbed = []
        for i, val in enumerate(individual):
            if np.random.random() < change_chance:  # x% chance of leaving unchanged
                perturbed.append(val)
                continue
            low, high = self.bounds[i]
            if high == low:
                perturbed.append(val)
                continue

            if val != 0.0:
                exponent = math.floor(math.log10(abs(val))) - (self.sig_digits - 1)
                step = 10**exponent
            else:
                step = (high - low) * 10 ** -(self.sig_digits - 1)

            direction = np.random.choice([-1.0, 1.0])
            perturbed.append(pbr.round_dynamic(val + step * direction, self.sig_digits))

        return perturbed

    def perturb_x_pct(self, individual, magnitude=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            low, high = self.bounds[i]
            if high == low:
                perturbed.append(val)
                continue
            new_val = pbr.round_dynamic(
                val * (1 + np.random.uniform(-magnitude, magnitude)), self.sig_digits
            )
            perturbed.append(new_val)
        return perturbed

    def perturb_random_subset(self, individual, frac=0.2):
        perturbed = individual.copy()
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            low, high = self.bounds[i]
            if low != high:
                delta = (high - low) * 0.01
                step = delta * np.random.uniform(-1.0, 1.0)
                perturbed[i] = individual[i] + step
        return perturbed

    def perturb_sample_some(self, individual, frac=0.2):
        perturbed = individual.copy()
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            low, high = self.bounds[i]
            if low != high:
                perturbed[i] = np.random.uniform(low, high)
        return perturbed

    def perturb_gaussian(self, individual, scale=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            low, high = self.bounds[i]
            if high == low:
                perturbed.append(val)
                continue
            noise = np.random.normal(0, scale * (high - low))
            perturbed.append(val + noise)
        return perturbed

    def perturb_large_uniform(self, individual):
        perturbed = []
        for i in range(len(individual)):
            low, high = self.bounds[i]
            if low == high:
                perturbed.append(low)
            else:
                perturbed.append(np.random.uniform(low, high))
        return perturbed

    def evaluate(self, individual, overrides_list):
        individual[:] = enforce_bounds(individual, self.bounds, self.sig_digits)
        config = individual_to_config(individual, optimizer_overrides, overrides_list, self.config)
        individual_hash = calc_hash(individual)
        if individual_hash in self.seen_hashes:
            existing_score = self.seen_hashes[individual_hash]
            self.duplicate_counter["count"] += 1
            dup_ct = self.duplicate_counter["count"]
            perturbation_funcs = [
                self.perturb_x_pct,
                self.perturb_step_digits,
                self.perturb_gaussian,
                self.perturb_random_subset,
                self.perturb_sample_some,
                self.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = perturb_fn(individual)
                perturbed = enforce_bounds(perturbed, self.bounds, self.sig_digits)
                new_hash = calc_hash(perturbed)
                if new_hash not in self.seen_hashes:
                    logging.info(
                        f"[DUPLICATE {dup_ct}] resolved with {perturb_fn.__name__} Hash: {new_hash}"
                    )
                    individual[:] = perturbed
                    self.seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.config
                    )
                    break
            else:
                logging.info(f"[DUPLICATE {dup_ct}] All perturbations failed.")
                if existing_score is not None:
                    return existing_score
        else:
            self.seen_hashes[individual_hash] = None
        analyses = {}
        for exchange in self.exchanges:
            bot_params, _, _ = prep_backtest_args(
                config,
                [],
                exchange,
                exchange_params=self.exchange_params[exchange],
                backtest_params=self.backtest_params[exchange],
            )
            fills, equities_usd, equities_btc, analysis_usd, analysis_btc = pbr.run_backtest(
                self.shared_memory_files[exchange],
                self.hlcvs_shapes[exchange],
                self.hlcvs_dtypes[exchange].str,
                self.btc_usd_shared_memory_files[exchange],
                self.btc_usd_dtypes[exchange].str,
                bot_params,
                self.exchange_params[exchange],
                self.backtest_params[exchange],
            )
            analyses[exchange] = expand_analysis(analysis_usd, analysis_btc, fills, config)
        analyses_combined = self.combine_analyses(analyses)
        objectives = self.calc_fitness(analyses_combined)
        for i, val in enumerate(objectives):
            analyses_combined[f"w_{i}"] = val
        data = {
            **config,
            "analyses_combined": analyses_combined,
            "analyses": analyses,
        }
        self.results_queue.put(data)
        actual_hash = calc_hash(individual)
        self.seen_hashes[actual_hash] = tuple(objectives)
        return tuple(objectives)

    def combine_analyses(self, analyses):
        analyses_combined = {}
        keys = analyses[next(iter(analyses))].keys()
        for key in keys:
            values = [analysis[key] for analysis in analyses.values()]
            if not values or any([x == np.inf for x in values]) or any([x is None for x in values]):
                analyses_combined[f"{key}_mean"] = 0.0
                analyses_combined[f"{key}_min"] = 0.0
                analyses_combined[f"{key}_max"] = 0.0
                analyses_combined[f"{key}_std"] = 0.0
            else:
                try:
                    analyses_combined[f"{key}_mean"] = np.mean(values)
                    analyses_combined[f"{key}_min"] = np.min(values)
                    analyses_combined[f"{key}_max"] = np.max(values)
                    analyses_combined[f"{key}_std"] = np.std(values)
                except Exception as e:
                    print("\n\n debug\n\n")
                    print("key, values", key, values)
                    print(e)
                    traceback.print_exc()
                    raise
        return analyses_combined

    def build_limit_checks(self):
        self.limit_checks = []
        limits = self.config["optimize"].get("limits", {})
        scoring_weights = self.scoring_weights

        for i, full_key in enumerate(sorted(limits)):
            bound = limits[full_key]

            if full_key.startswith("penalize_if_greater_than_"):
                metric = full_key[len("penalize_if_greater_than_") :]
                penalize_if = "greater"
            elif full_key.startswith("penalize_if_lower_than_"):
                metric = full_key[len("penalize_if_lower_than_") :]
                penalize_if = "lower"
            else:
                # Fallback for scoring_weight-based logic
                metric = full_key
                weight = scoring_weights.get(metric)
                if weight is None:
                    continue
                penalize_if = "lower" if weight < 0 else "greater"
            suffix = "min" if penalize_if == "lower" else "max"

            self.limit_checks.append(
                {
                    "metric_key": f"{metric}_{suffix}",
                    "penalize_if": penalize_if,
                    "bound": bound,
                    "penalty_weight": 1e6,
                }
            )

    def calc_fitness(self, analyses_combined):
        modifier = 0.0
        for check in self.limit_checks:
            val = analyses_combined.get(check["metric_key"])
            if val is None:
                continue

            if check["penalize_if"] == "greater" and val > check["bound"]:
                modifier += (val - check["bound"]) * (check["penalty_weight"])
            elif check["penalize_if"] == "lower" and val < check["bound"]:
                modifier += (check["bound"] - val) * (check["penalty_weight"])

        scores = []
        for sk in sorted(self.config["optimize"]["scoring"]):
            val = analyses_combined.get(f"{sk}_mean")
            if val is None:
                val = analyses_combined.get(f"{sk}_mean")
            if val is None:
                return None
            scores.append(val * self.scoring_weights[sk] + modifier)
        return tuple(scores)

    def __del__(self):
        if hasattr(self, "mmap_contexts"):
            for mmap_context in self.mmap_contexts.values():
                mmap_context.__exit__(None, None, None)

    def __getstate__(self):
        # This method is called when pickling. We exclude mmap_contexts and shared_hlcvs_np
        state = self.__dict__.copy()
        del state["mmap_contexts"]
        del state["shared_hlcvs_np"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mmap_contexts = {}
        self.shared_hlcvs_np = {}
        for exchange in self.exchanges:
            self.mmap_contexts[exchange] = managed_mmap(
                self.shared_memory_files[exchange],
                self.hlcvs_dtypes[exchange],
                self.hlcvs_shapes[exchange],
            )
            self.shared_hlcvs_np[exchange] = self.mmap_contexts[exchange].__enter__()
            if self.shared_hlcvs_np[exchange] is None:
                print(
                    f"Warning: Unable to recreate shared memory mapping during unpickling for {exchange}."
                )


def add_extra_options(parser):
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="Start with given live configs. Single json file or dir with multiple json files",
    )


def extract_configs(path):
    cfgs = []
    if os.path.exists(path):
        if path.endswith("_all_results.bin"):
            logging.info(f"Skipping {path}")
            return []
        if path.endswith(".json"):
            try:
                cfgs.append(load_config(path, verbose=False))
                return cfgs
            except:
                return []
        if path.endswith("_pareto.txt"):
            with open(path) as f:
                for line in f.readlines():
                    try:
                        cfg = json.loads(line)
                        cfgs.append(format_config(cfg, verbose=False))
                    except Exception as e:
                        logging.error(f"Failed to load starting config {line} {e}")
    return cfgs


def get_starting_configs(starting_configs: str):
    if starting_configs is None:
        return []
    if os.path.isdir(starting_configs):
        return flatten(
            [
                get_starting_configs(os.path.join(starting_configs, f))
                for f in os.listdir(starting_configs)
            ]
        )
    return extract_configs(starting_configs)


def configs_to_individuals(cfgs, bounds, sig_digits=0):
    inds = {}
    for cfg in cfgs:
        try:
            fcfg = format_config(cfg, verbose=False)
            individual = config_to_individual(fcfg, bounds, sig_digits)
            inds[calc_hash(individual)] = individual
            # add duplicate of config, but with lowered total wallet exposure limit
            fcfg2 = deepcopy(fcfg)
            for pside in ["long", "short"]:
                fcfg2["bot"][pside]["total_wallet_exposure_limit"] *= 0.75
            individual2 = config_to_individual(fcfg2, bounds, sig_digits)
            inds[calc_hash(individual2)] = individual2
        except Exception as e:
            logging.error(f"error loading starting config: {e}")
    return list(inds.values())


async def main():
    manage_rust_compilation()
    parser = argparse.ArgumentParser(prog="optimize", description="run optimizer")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config(TEMPLATE_CONFIG_MODE)
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
        "ohlcv_rolling_window",
        "relative_volume_filter_clip_pct",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    add_extra_options(parser)
    args = parser.parse_args()
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=True)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path, verbose=True)
    old_config = deepcopy(config)
    update_config_with_args(config, args)
    config = format_config(config, verbose=True)
    await add_all_eligible_coins_to_config(config)
    try:
        # Prepare data for each exchange
        hlcvs_dict = {}
        shared_memory_files = {}
        hlcvs_shapes = {}
        hlcvs_dtypes = {}
        msss = {}

        # NEW: Store per-exchange BTC arrays in a dict,
        # and store their shared-memory file names in another dict.
        btc_usd_data_dict = {}
        btc_usd_shared_memory_files = {}
        btc_usd_dtypes = {}

        config["backtest"]["coins"] = {}
        if config["backtest"]["combine_ohlcvs"]:
            exchange = "combined"
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await prepare_hlcvs_mss(
                config, exchange
            )
            exchange_preference = defaultdict(list)
            for coin in coins:
                exchange_preference[mss[coin]["exchange"]].append(coin)
            for ex in exchange_preference:
                logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
            config["backtest"]["coins"][exchange] = coins
            hlcvs_dict[exchange] = hlcvs
            hlcvs_shapes[exchange] = hlcvs.shape
            hlcvs_dtypes[exchange] = hlcvs.dtype
            msss[exchange] = mss
            required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
            check_disk_space(tempfile.gettempdir(), required_space)
            logging.info(f"Starting to create shared memory file for {exchange}...")
            validate_array(hlcvs, "hlcvs")
            shared_memory_file = create_shared_memory_file(hlcvs)
            shared_memory_files[exchange] = shared_memory_file
            if config["backtest"].get("use_btc_collateral", False):
                # Use the fetched array
                btc_usd_data_dict[exchange] = btc_usd_prices
            else:
                # Fall back to all ones
                btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
            validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
            btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                btc_usd_data_dict[exchange]
            )
            btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
            logging.info(f"Finished creating shared memory file for {exchange}: {shared_memory_file}")
        else:
            tasks = {}
            for exchange in config["backtest"]["exchanges"]:
                tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
            for exchange in config["backtest"]["exchanges"]:
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await tasks[exchange]
                config["backtest"]["coins"][exchange] = coins
                hlcvs_dict[exchange] = hlcvs
                hlcvs_shapes[exchange] = hlcvs.shape
                hlcvs_dtypes[exchange] = hlcvs.dtype
                msss[exchange] = mss
                required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
                check_disk_space(tempfile.gettempdir(), required_space)
                logging.info(f"Starting to create shared memory file for {exchange}...")
                validate_array(hlcvs, "hlcvs")
                shared_memory_file = create_shared_memory_file(hlcvs)
                shared_memory_files[exchange] = shared_memory_file
                # Create the BTC array for this exchange
                if config["backtest"].get("use_btc_collateral", False):
                    btc_usd_data_dict[exchange] = btc_usd_prices
                else:
                    btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)

                validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
                btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                    btc_usd_data_dict[exchange]
                )
                btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
                logging.info(
                    f"Finished creating shared memory file for {exchange}: {shared_memory_file}"
                )
        exchanges = config["backtest"]["exchanges"]
        exchanges_fname = "combined" if config["backtest"]["combine_ohlcvs"] else "_".join(exchanges)
        date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
        coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
        coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
        hash_snippet = uuid4().hex[:8]
        n_days = int(
            round(
                (
                    date_to_ts(config["backtest"]["end_date"])
                    - date_to_ts(config["backtest"]["start_date"])
                )
                / (1000 * 60 * 60 * 24)
            )
        )
        results_dir = make_get_filepath(
            f"optimize_results/{date_fname}_{exchanges_fname}_{n_days}days_{coins_fname}_{hash_snippet}/"
        )
        os.makedirs(results_dir, exist_ok=True)
        config["results_dir"] = results_dir
        results_filename = os.path.join(results_dir, "all_results.bin")
        config["results_filename"] = results_filename
        overrides_list = config.get("optimize", {}).get("enable_overrides", [])

        # Create results queue and start manager process
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        seen_hashes = manager.dict()
        duplicate_counter = manager.dict()
        duplicate_counter["count"] = 0
        flush_interval = 60  # or read from your config
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        writer_process = multiprocessing.Process(
            target=results_writer_process,
            args=(results_queue, results_dir, sig_digits, flush_interval),
            kwargs={
                "compress": config["optimize"]["compress_results_file"],
                "write_all_results": config["optimize"].get("write_all_results", True),  # ← new
            },
        )
        writer_process.start()

        # Prepare BTC/USD data
        # For optimization, use the BTC/USD prices from the first exchange (or combined)
        # Since all exchanges should align in timesteps, this should be consistent
        btc_usd_data = btc_usd_prices  # Use the fetched btc_usd_prices from prepare_hlcvs_mss
        if config["backtest"].get("use_btc_collateral", False):
            logging.info("Using fetched BTC/USD prices for collateral")
        else:
            logging.info("Using default BTC/USD prices (all 1.0s) as use_btc_collateral is False")
            btc_usd_data = np.ones(hlcvs_dict[next(iter(hlcvs_dict))].shape[0], dtype=np.float64)

        validate_array(btc_usd_data, "btc_usd_data")
        btc_usd_shared_memory_file = create_shared_memory_file(btc_usd_data)

        # Initialize evaluator with results queue and BTC/USD shared memory
        evaluator = Evaluator(
            shared_memory_files=shared_memory_files,
            hlcvs_shapes=hlcvs_shapes,
            hlcvs_dtypes=hlcvs_dtypes,
            # Instead of a single file/dtype, pass dictionaries
            btc_usd_shared_memory_files=btc_usd_shared_memory_files,
            btc_usd_dtypes=btc_usd_dtypes,
            msss=msss,
            config=config,
            results_queue=results_queue,
            seen_hashes=seen_hashes,
            duplicate_counter=duplicate_counter,
        )

        logging.info(f"Finished initializing evaluator...")
        n_objectives = len(config["optimize"]["scoring"])
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * n_objectives)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        bounds = extract_bounds_tuple_list_from_config(config)
        sig_digits = config["optimize"]["round_to_n_significant_digits"]

        # Register attribute generators
        for i, (low, high) in enumerate(bounds):
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        def create_individual():
            return creator.Individual([getattr(toolbox, f"attr_{i}")() for i in range(len(bounds))])

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", evaluator.evaluate, overrides_list=overrides_list)

        # Register genetic operators
        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=20.0,
            low=[low for low, high in bounds],
            up=[high for low, high in bounds],
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=20.0,
            low=[low for low, high in bounds],
            up=[high for low, high in bounds],
            indpb=1.0 / len(bounds),
        )
        toolbox.register("select", tools.selNSGA2)

        # Parallelization setup
        logging.info(f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}")
        pool = multiprocessing.Pool(processes=config["optimize"]["n_cpus"])
        toolbox.register("map", pool.map)
        logging.info(f"Finished initializing multiprocessing pool.")

        # Create initial population
        logging.info(f"Creating initial population...")

        starting_individuals = configs_to_individuals(
            get_starting_configs(args.starting_configs),
            bounds,
            sig_digits,
        )
        if (nstart := len(starting_individuals)) > (popsize := config["optimize"]["population_size"]):
            logging.info(f"Number of starting configs greater than population size.")
            logging.info(f"Increasing population size: {popsize} -> {nstart}")
            config["optimize"]["population_size"] = nstart

        population = toolbox.population(n=config["optimize"]["population_size"])
        if starting_individuals:
            for i in range(len(starting_individuals)):
                population[i] = creator.Individual(starting_individuals[i])

            # populate up to half of the population with duplicates of random choices within starting configs
            # duplicates will be perturbed during runtime
            for i in range(len(starting_individuals), len(population) // 2):
                population[i] = deepcopy(
                    population[np.random.choice(range(len(starting_individuals)))]
                )
        for i in range(len(population)):
            population[i][:] = enforce_bounds(population[i], bounds, sig_digits)

        logging.info(f"Initial population size: {len(population)}")

        # Set up statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", np.mean, axis=0)
        # stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        # logbook.header = "gen", "evals", "std", "min", "avg", "max"
        logbook.header = "gen", "evals", "min", "max"

        hof = tools.ParetoFront()

        # Run the optimization
        logging.info(f"Starting optimize...")
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=config["optimize"]["population_size"],
            lambda_=config["optimize"]["population_size"],
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

        # Print statistics
        print(logbook)

        logging.info(f"Optimization complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Signal the writer process to shut down and wait for it
        if "results_queue" in locals():
            results_queue.put("DONE")
            writer_process.join()
        if "pool" in locals():
            logging.info("Closing and terminating the process pool...")
            pool.close()
            pool.terminate()
            pool.join()

        # Remove shared memory files (including BTC/USD)
        if "shared_memory_files" in locals():
            for shared_memory_file in shared_memory_files.values():
                if shared_memory_file and os.path.exists(shared_memory_file):
                    logging.info(f"Removing shared memory file: {shared_memory_file}")
                    try:
                        os.unlink(shared_memory_file)
                    except Exception as e:
                        logging.error(f"Error removing shared memory file: {e}")
        if "btc_usd_shared_memory_file" in locals():
            if btc_usd_shared_memory_file and os.path.exists(btc_usd_shared_memory_file):
                logging.info(f"Removing BTC/USD shared memory file: {btc_usd_shared_memory_file}")
                try:
                    os.unlink(btc_usd_shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing BTC/USD shared memory file: {e}")

        logging.info("Cleanup complete. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
