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
import passivbot_rust as pbr
import asyncio
import argparse
import multiprocessing
import time
from collections import defaultdict
from backtest import (
    prepare_hlcvs_mss,
    prep_backtest_args,
    expand_analysis,
)
from downloader import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes
from config_utils import (
    get_template_config,
    load_hjson_config,
    load_config,
    format_config,
    add_arguments_recursively,
    update_config_with_args,
    require_config_value,
    merge_negative_cli_values,
)
from pure_funcs import (
    denumpyize,
    sort_dict_keys,
    calc_hash,
    flatten,
)
from utils import date_to_ts, ts_to_date, utc_ms, make_get_filepath, format_approved_ignored_coins
from copy import deepcopy
from main import manage_rust_compilation
import numpy as np
from uuid import uuid4
import logging
import traceback
import json
import pprint
from deap import base, creator, tools, algorithms
import math
import fcntl
from optimizer_overrides import optimizer_overrides
from opt_utils import make_json_serializable, generate_incremental_diff, round_floats
from pareto_store import ParetoStore
import msgpack
from typing import Sequence, Tuple, List
from itertools import permutations
from shared_arrays import SharedArrayManager, attach_shared_array


class ResultRecorder:
    def __init__(
        self,
        *,
        results_dir: str,
        sig_digits: int,
        flush_interval: int,
        scoring_keys: Sequence[str],
        compress: bool,
        write_all_results: bool,
    ):
        self.store = ParetoStore(
            directory=results_dir,
            sig_digits=sig_digits,
            flush_interval=flush_interval,
            log_name="optimizer.pareto",
        )
        self.write_all = write_all_results
        self.compress = compress
        self.results_file = None
        self.packer = None
        if self.write_all:
            filename = os.path.join(results_dir, "all_results.bin")
            self.results_file = open(filename, "ab")
            self.packer = msgpack.Packer(use_bin_type=True)
        self.prev_data = None
        self.counter = 0
        self.scoring_keys = list(scoring_keys)

    def record(self, data: dict) -> None:
        if self.write_all and self.results_file:
            if self.compress:
                if self.prev_data is None or self.counter % 100 == 0:
                    output_data = make_json_serializable(data)
                else:
                    diff = generate_incremental_diff(self.prev_data, data)
                    output_data = make_json_serializable(diff)
                self.counter += 1
                self.prev_data = data
            else:
                output_data = data
            try:
                self.results_file.write(self.packer.pack(output_data))
                self.results_file.flush()
            except Exception as exc:
                logging.error(f"Error writing results: {exc}")
        try:
            updated = self.store.add_entry(data)
        except Exception as exc:
            logging.error(f"ParetoStore error: {exc}")
        else:
            if updated:
                metrics = data.get("analyses_combined", {}) or {}
                objective_values = [
                    metrics[key]
                    for key in sorted(metrics)
                    if key.startswith("w_") and metrics.get(key) is not None
                ]
                logging.info(
                    "Pareto update | eval=%d | front=%d | objectives=%s",
                    self.store.n_iters,
                    len(self.store._front),
                    _format_objectives(objective_values),
                )

    def flush(self) -> None:
        self.store.flush_now()

    def close(self) -> None:
        if self.results_file:
            self.results_file.close()


logging.basicConfig(
    format="%(asctime)s %(processName)-12s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


TEMPLATE_CONFIG_MODE = "v7"


def _format_objectives(values: Sequence[float]) -> str:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not values:
        return "[]"
    return "[" + ", ".join(f"{float(v):.3g}" for v in values) + "]"


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

    template_config = get_template_config(
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


def _record_individual_result(individual, evaluator_config, overrides_list, recorder):
    metrics = getattr(individual, "evaluation_metrics", {}) or {}
    config = individual_to_config(individual, optimizer_overrides, overrides_list, evaluator_config)
    data = {**config, "analyses_combined": metrics}
    recorder.record(data)
    if hasattr(individual, "evaluation_metrics"):
        del individual.evaluation_metrics


def ea_mu_plus_lambda_stream(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats,
    halloffame,
    verbose,
    recorder,
    evaluator_config,
    overrides_list,
    pool,
):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"

    start_time = time.time()
    total_evals = 0

    def evaluate_and_record(individuals):
        nonlocal total_evals
        if not individuals:
            return 0
        logging.debug("Evaluating %d candidates", len(individuals))
        pending = {}
        for idx, ind in enumerate(individuals):
            pending[pool.apply_async(toolbox.evaluate, (ind,))] = idx

        completed = 0
        try:
            while pending:
                ready = [res for res in pending if res.ready()]
                if not ready:
                    time.sleep(0.1)
                    continue
                for res in ready:
                    idx = pending.pop(res)
                    fit_values, metrics = res.get()
                    ind = individuals[idx]
                    ind.fitness.values = fit_values
                    ind.evaluation_metrics = metrics
                    _record_individual_result(ind, evaluator_config, overrides_list, recorder)
                    completed += 1
        except KeyboardInterrupt:
            logging.info("Evaluation interrupted; terminating pending tasks...")
            for res in pending:
                try:
                    res.cancel()
                except Exception:
                    pass
            raise

        total_evals += completed
        return completed

    def log_generation(gen, nevals, record):
        best = record.get("min") if record else None
        front_size = len(halloffame) if halloffame is not None else 0
        logging.info(
            "Gen %d complete | evals=%d | total=%d | front=%d | best=%s | elapsed=%.1fs",
            gen,
            nevals,
            total_evals,
            front_size,
            _format_objectives(best),
            time.time() - start_time,
        )
        if verbose and record:
            logging.debug("Logbook: %s", " ".join(f"{k}={v}" for k, v in record.items()))

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if invalid_ind:
        logging.info("Evaluating initial population (%d candidates)...", len(invalid_ind))
    nevals = evaluate_and_record(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    log_generation(0, nevals, record)

    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        nevals = evaluate_and_record(invalid_ind)

        population[:] = toolbox.select(population + offspring, mu)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        log_generation(gen, nevals, record)

    logging.info(
        "Optimization summary | generations=%d | total_evals=%d | front=%d | duration=%.1fs",
        ngen,
        total_evals,
        len(halloffame) if halloffame is not None else 0,
        time.time() - start_time,
    )
    return population, logbook


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


def validate_array(arr, name, allow_nan=True):
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf values")
    if allow_nan and np.isnan(arr).all():
        raise ValueError(f"{name} is entirely NaN")


class Evaluator:
    def __init__(
        self,
        hlcvs_specs,
        btc_usd_specs,
        msss,
        config,
        seen_hashes=None,
        duplicate_counter=None,
        timestamps=None,
        shared_array_manager: SharedArrayManager | None = None,
    ):
        logging.debug("Initializing Evaluator...")
        self.hlcvs_specs = hlcvs_specs
        self.btc_usd_specs = btc_usd_specs
        self.msss = msss
        self.timestamps = timestamps or {}
        self.exchanges = list(hlcvs_specs.keys())
        self.shared_array_manager = shared_array_manager
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}
        self.exchange_params = {}
        self.backtest_params = {}

        for exchange in self.exchanges:
            logging.debug("Preparing cached parameters for %s...", exchange)
            if self.shared_array_manager is not None:
                self.shared_hlcvs_np[exchange] = self.shared_array_manager.view(
                    self.hlcvs_specs[exchange]
                )
                btc_spec = self.btc_usd_specs.get(exchange)
                if btc_spec is not None:
                    self.shared_btc_np[exchange] = self.shared_array_manager.view(btc_spec)
            _, self.exchange_params[exchange], self.backtest_params[exchange] = prep_backtest_args(
                config, self.msss[exchange], exchange
            )

        self.config = config
        logging.debug("Evaluator initialization complete.")
        logging.info("Evaluator ready | exchanges=%d", len(self.exchanges))
        self.seen_hashes = seen_hashes if seen_hashes is not None else {}
        self.duplicate_counter = duplicate_counter if duplicate_counter is not None else {"count": 0}
        self.bounds = extract_bounds_tuple_list_from_config(self.config)
        self.sig_digits = config.get("optimize", {}).get("round_to_n_significant_digits", 6)

        for exchange in self.exchanges:
            self._populate_backtest_metadata(exchange)

        shared_metric_weights = {
            "positions_held_per_day": 1.0,
            "position_held_hours_mean": 1.0,
            "position_held_hours_max": 1.0,
            "position_held_hours_median": 1.0,
            "position_unchanged_hours_max": 1.0,
        }

        currency_metric_weights = {
            "adg": -1.0,
            "adg_per_exposure_long": -1.0,
            "adg_per_exposure_short": -1.0,
            "adg_w": -1.0,
            "adg_w_per_exposure_long": -1.0,
            "adg_w_per_exposure_short": -1.0,
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
            "equity_peak_recovery_hours": 1.0,
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
            "sharpe_ratio": -1.0,
            "sharpe_ratio_w": -1.0,
            "sortino_ratio": -1.0,
            "sortino_ratio_w": -1.0,
            "sterling_ratio": -1.0,
            "sterling_ratio_w": -1.0,
            "volume_pct_per_day_avg": -1.0,
            "volume_pct_per_day_avg_w": -1.0,
        }

        self.scoring_weights = {}
        self.scoring_weights.update(shared_metric_weights)

        for metric, weight in currency_metric_weights.items():
            self.scoring_weights[f"{metric}_usd"] = weight
            self.scoring_weights[f"{metric}_btc"] = weight
            self.scoring_weights.setdefault(metric, weight)
            self.scoring_weights.setdefault(f"usd_{metric}", weight)
            self.scoring_weights.setdefault(f"btc_{metric}", weight)

        self.build_limit_checks()

    def _ensure_attached(self, exchange: str) -> None:
        if exchange not in self.shared_hlcvs_np:
            spec = self.hlcvs_specs[exchange]
            attachment = attach_shared_array(spec)
            self._attachments["hlcvs"][exchange] = attachment
            self.shared_hlcvs_np[exchange] = attachment.array
        if exchange not in self.shared_btc_np:
            btc_spec = self.btc_usd_specs.get(exchange)
            if btc_spec is not None:
                attachment = attach_shared_array(btc_spec)
                self._attachments["btc"][exchange] = attachment
                self.shared_btc_np[exchange] = attachment.array

    def _populate_backtest_metadata(self, exchange: str) -> None:
        self._ensure_attached(exchange)
        hlcvs_arr = self.shared_hlcvs_np[exchange]
        exchange_mss = self.msss.get(exchange, {}) if isinstance(self.msss, dict) else {}
        meta = exchange_mss.get("__meta__", {}) if isinstance(exchange_mss, dict) else {}

        first_ts_list = self.timestamps.get(exchange)
        first_ts_ms = 0
        if first_ts_list is not None and len(first_ts_list) > 0:
            try:
                first_ts_ms = int(first_ts_list[0])
            except Exception:
                logging.warning(
                    "Evaluator: unable to parse first timestamp for %s from timestamps array",
                    exchange,
                )
                first_ts_ms = 0
        if first_ts_ms == 0:
            candidate_ts = (
                meta.get("requested_start_ts")
                or meta.get("effective_start_ts")
                or require_config_value(self.config, "backtest.start_date")
            )
            if isinstance(candidate_ts, (int, float)):
                first_ts_ms = int(candidate_ts)
            elif isinstance(candidate_ts, str):
                try:
                    first_ts_ms = int(date_to_ts(candidate_ts))
                except Exception:
                    first_ts_ms = 0
            if first_ts_ms:
                logging.info(
                    "Evaluator: using fallback first timestamp %s for %s", first_ts_ms, exchange
                )
            else:
                logging.warning(
                    "Evaluator: falling back to 0 first_timestamp_ms for %s; timestamps unavailable",
                    exchange,
                )
        self.backtest_params[exchange]["first_timestamp_ms"] = first_ts_ms

        candidate_start = meta.get("requested_start_ts") or require_config_value(
            self.config, "backtest.start_date"
        )
        try:
            if isinstance(candidate_start, str):
                requested_start_ts = int(date_to_ts(candidate_start))
            else:
                requested_start_ts = int(candidate_start or 0)
        except Exception:
            requested_start_ts = int(
                date_to_ts(require_config_value(self.config, "backtest.start_date"))
            )
        self.backtest_params[exchange]["requested_start_timestamp_ms"] = requested_start_ts

        coins_order = self.backtest_params[exchange].get("coins", [])
        total_steps = hlcvs_arr.shape[0]
        warmup_map = compute_per_coin_warmup_minutes(self.config)
        default_warm = int(warmup_map.get("__default__", 0))
        first_valid_indices = []
        last_valid_indices = []
        warmup_minutes = []
        trade_start_indices = []
        for idx, coin in enumerate(coins_order):
            meta_coin = exchange_mss.get(coin, {}) if isinstance(exchange_mss, dict) else {}
            first_idx = int(meta_coin.get("first_valid_index", 0))
            last_idx = int(meta_coin.get("last_valid_index", total_steps - 1))
            if first_idx >= total_steps:
                first_idx = total_steps
            if last_idx >= total_steps:
                last_idx = total_steps - 1
            first_valid_indices.append(first_idx)
            last_valid_indices.append(last_idx)
            warm = int(meta_coin.get("warmup_minutes", warmup_map.get(coin, default_warm)))
            warmup_minutes.append(warm)
            if first_idx > last_idx:
                trade_idx = first_idx
            else:
                trade_idx = min(last_idx, first_idx + warm)
            trade_start_indices.append(trade_idx)
        self.backtest_params[exchange]["first_valid_indices"] = first_valid_indices
        self.backtest_params[exchange]["last_valid_indices"] = last_valid_indices
        self.backtest_params[exchange]["warmup_minutes"] = warmup_minutes
        self.backtest_params[exchange]["trade_start_indices"] = trade_start_indices
        self.backtest_params[exchange]["global_warmup_bars"] = compute_backtest_warmup_minutes(
            self.config
        )

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
                    logging.debug(
                        "[DUPLICATE %d] resolved with %s -> %s",
                        dup_ct,
                        perturb_fn.__name__,
                        new_hash,
                    )
                    individual[:] = perturbed
                    self.seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.config
                    )
                    break
            else:
                logging.debug("[DUPLICATE %d] All perturbations failed.", dup_ct)
                if existing_score is not None:
                    return existing_score
        else:
            self.seen_hashes[individual_hash] = None
        analyses = {}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)
            bot_params_list, _, _ = prep_backtest_args(
                config,
                [],
                exchange,
                exchange_params=self.exchange_params[exchange],
                backtest_params=self.backtest_params[exchange],
            )
            run_result = pbr.run_backtest(
                self.shared_hlcvs_np[exchange],
                self.shared_btc_np[exchange],
                bot_params_list,
                self.exchange_params[exchange],
                self.backtest_params[exchange],
            )
            if len(run_result) == 5:
                fills, equities_array, _, analysis_usd, analysis_btc = run_result
            else:
                fills, equities_array, analysis_usd, analysis_btc = run_result
            analyses[exchange] = expand_analysis(
                analysis_usd, analysis_btc, fills, equities_array, config
            )

            # Explicitly drop large intermediate arrays to keep worker RSS low.
            del fills
            del equities_array
            del analysis_usd
            del analysis_btc
        analyses_combined = self.combine_analyses(analyses)
        del analyses
        objectives = self.calc_fitness(analyses_combined)
        for i, val in enumerate(objectives):
            analyses_combined[f"w_{i}"] = val
        # attach metrics to individual so the parent process can persist lean results
        individual.evaluation_metrics = analyses_combined
        actual_hash = calc_hash(individual)
        self.seen_hashes[actual_hash] = tuple(objectives)
        return tuple(objectives), analyses_combined

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
        for sk in self.config["optimize"]["scoring"]:
            if modifier:
                scores.append(modifier)
            else:
                parts = sk.split("_")
                candidates = []
                if len(parts) <= 1:
                    candidates = [sk]
                else:
                    base, rest = parts[0], parts[1:]
                    base_candidate = "_".join([base, *rest])
                    candidates.append(base_candidate)
                    for perm in permutations(rest):
                        candidate = "_".join([base, *perm])
                        candidates.append(candidate)

                extended_candidates = []
                seen = set()
                for candidate in candidates:
                    if candidate not in seen:
                        extended_candidates.append(candidate)
                        seen.add(candidate)
                    for suffix in ("usd", "btc"):
                        with_suffix = f"{candidate}_{suffix}"
                        if with_suffix not in seen:
                            extended_candidates.append(with_suffix)
                            seen.add(with_suffix)
                        parts_candidate = candidate.split("_")
                        if len(parts_candidate) >= 2:
                            inserted = "_".join(parts_candidate[:-1] + [suffix, parts_candidate[-1]])
                            if inserted not in seen:
                                extended_candidates.append(inserted)
                                seen.add(inserted)

                val = None
                weight = None
                selected_metric = None
                for candidate in extended_candidates:
                    metric_key = f"{candidate}_mean"
                    if val is None and metric_key in analyses_combined:
                        val = analyses_combined[metric_key]
                        selected_metric = candidate
                    if weight is None and candidate in self.scoring_weights:
                        weight = self.scoring_weights[candidate]
                    if val is not None and weight is not None:
                        break

                if val is None:
                    return None
                if weight is None:
                    weight = 1.0
                scores.append(val * weight)
        return tuple(scores)

    def __del__(self):
        for attachment_map in self._attachments.values():
            for attachment in attachment_map.values():
                attachment.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("shared_hlcvs_np", None)
        state.pop("shared_btc_np", None)
        state.pop("_attachments", None)
        state.pop("shared_array_manager", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.shared_array_manager = None
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)


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
    parser.add_argument(
        "-ft",
        "--fine_tune_params",
        "--fine-tune-params",
        type=str,
        default="",
        dest="fine_tune_params",
        help=(
            "Comma-separated optimize bounds keys to tune; other parameters are fixed to their current config values"
        ),
    )


def apply_fine_tune_bounds(config: dict, fine_tune_params: list[str]) -> None:
    if not fine_tune_params:
        return

    bounds = config.get("optimize", {}).get("bounds", {})
    bot_cfg = config.get("bot", {})
    fine_tune_set = set(fine_tune_params)

    for key in list(bounds.keys()):
        if key in fine_tune_set:
            continue
        try:
            pside, param = key.split("_", 1)
        except ValueError:
            logging.warning(f"fine-tune bounds: unable to parse key '{key}', skipping")
            continue
        side_cfg = bot_cfg.get(pside)
        if not isinstance(side_cfg, dict) or param not in side_cfg:
            logging.warning(
                f"fine-tune bounds: missing bot value for '{key}', leaving bounds unchanged"
            )
            continue
        current_bounds = bounds.get(key)
        target_value = None
        if isinstance(current_bounds, (list, tuple)):
            if len(current_bounds) == 1:
                target_value = current_bounds[0]
            elif len(current_bounds) >= 2:
                low, high = current_bounds[0], current_bounds[1]
                if low != high:
                    # Explicit range provided (e.g. via CLI); respect it.
                    continue
                target_value = low
        if target_value is not None:
            try:
                side_cfg[param] = float(target_value)
            except (TypeError, ValueError):
                side_cfg[param] = target_value
            continue
        value = side_cfg[param]
        try:
            value_float = float(value)
            bounds[key] = [value_float, value_float]
        except (TypeError, ValueError):
            bounds[key] = [value, value]

    missing = [key for key in fine_tune_set if key not in bounds]
    if missing:
        logging.warning(
            "fine-tune bounds: requested keys not found in optimize bounds: %s",
            ",".join(sorted(missing)),
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
    template_config = get_template_config(TEMPLATE_CONFIG_MODE)
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    add_extra_options(parser)
    raw_args = merge_negative_cli_values(sys.argv[1:])
    args = parser.parse_args(raw_args)
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=True)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path, verbose=True)
    update_config_with_args(config, args)
    config = format_config(config, verbose=False)
    logging.info(
        "Config normalized for optimization | template=%s | scoring=%s",
        TEMPLATE_CONFIG_MODE,
        ",".join(config["optimize"].get("scoring", [])),
    )
    fine_tune_params = (
        [p.strip() for p in (args.fine_tune_params or "").split(",") if p.strip()]
        if getattr(args, "fine_tune_params", "")
        else []
    )
    apply_fine_tune_bounds(config, fine_tune_params)
    if fine_tune_params:
        logging.info(
            "Fine-tuning mode active for %s",
            ", ".join(sorted(fine_tune_params)),
        )
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(config, backtest_exchanges)
    interrupted = False
    try:
        # Prepare data for each exchange
        array_manager = SharedArrayManager()
        hlcvs_specs = {}
        btc_usd_specs = {}
        msss = {}
        timestamps_dict = {}
        config["backtest"]["coins"] = {}

        if bool(require_config_value(config, "backtest.combine_ohlcvs")):
            exchange = "combined"
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = (
                await prepare_hlcvs_mss(config, exchange)
            )
            timestamps_dict[exchange] = _timestamps
            exchange_preference = defaultdict(list)
            for coin in coins:
                exchange_preference[mss[coin]["exchange"]].append(coin)
            for ex in exchange_preference:
                logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
            config["backtest"]["coins"][exchange] = coins
            msss[exchange] = mss
            validate_array(hlcvs, "hlcvs")
            hlcvs_array = np.ascontiguousarray(hlcvs, dtype=np.float64)
            hlcvs_spec, _ = array_manager.create_from(hlcvs_array)
            hlcvs_specs[exchange] = hlcvs_spec

            btc_usd_array = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
            validate_array(btc_usd_array, f"btc_usd_data for {exchange}", allow_nan=False)
            btc_usd_spec, _ = array_manager.create_from(btc_usd_array)
            btc_usd_specs[exchange] = btc_usd_spec
            del hlcvs, hlcvs_array, btc_usd_prices, btc_usd_array
        else:
            tasks = {}
            for exchange in backtest_exchanges:
                tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
            for exchange in backtest_exchanges:
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = await tasks[
                    exchange
                ]
                timestamps_dict[exchange] = _timestamps
                config["backtest"]["coins"][exchange] = coins
                msss[exchange] = mss
                validate_array(hlcvs, "hlcvs")
                hlcvs_array = np.ascontiguousarray(hlcvs, dtype=np.float64)
                hlcvs_spec, _ = array_manager.create_from(hlcvs_array)
                hlcvs_specs[exchange] = hlcvs_spec

                btc_usd_array = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
                validate_array(btc_usd_array, f"btc_usd_data for {exchange}", allow_nan=False)
                btc_usd_spec, _ = array_manager.create_from(btc_usd_array)
                btc_usd_specs[exchange] = btc_usd_spec
                del hlcvs, hlcvs_array, btc_usd_prices, btc_usd_array
        exchanges = backtest_exchanges
        exchanges_fname = (
            "combined"
            if bool(require_config_value(config, "backtest.combine_ohlcvs"))
            else "_".join(exchanges)
        )
        date_fname = ts_to_date(utc_ms())[:19].replace(":", "_")
        coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
        coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
        hash_snippet = uuid4().hex[:8]
        n_days = int(
            round(
                (
                    date_to_ts(require_config_value(config, "backtest.end_date"))
                    - date_to_ts(require_config_value(config, "backtest.start_date"))
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

        # Shared state used by workers for duplicate detection
        manager = multiprocessing.Manager()
        seen_hashes = manager.dict()
        duplicate_counter = manager.dict()
        duplicate_counter["count"] = 0

        # Initialize evaluator with shared memory references
        evaluator = Evaluator(
            hlcvs_specs=hlcvs_specs,
            btc_usd_specs=btc_usd_specs,
            msss=msss,
            config=config,
            seen_hashes=seen_hashes,
            duplicate_counter=duplicate_counter,
            timestamps=timestamps_dict,
            shared_array_manager=array_manager,
        )

        logging.info(f"Finished initializing evaluator...")
        flush_interval = 60  # or read from your config
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        recorder = ResultRecorder(
            results_dir=results_dir,
            sig_digits=sig_digits,
            flush_interval=flush_interval,
            scoring_keys=config["optimize"]["scoring"],
            compress=config["optimize"]["compress_results_file"],
            write_all_results=config["optimize"].get("write_all_results", True),
        )

        n_objectives = len(config["optimize"]["scoring"])
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * n_objectives)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        bounds = extract_bounds_tuple_list_from_config(config)
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        crossover_eta = config["optimize"].get("crossover_eta", 20.0)
        mutation_eta = config["optimize"].get("mutation_eta", 20.0)
        mutation_indpb_raw = config["optimize"].get("mutation_indpb", 0.0)
        if isinstance(mutation_indpb_raw, (int, float)) and mutation_indpb_raw > 0.0:
            mutation_indpb = max(0.0, min(1.0, float(mutation_indpb_raw)))
        else:
            mutation_indpb = 1.0 / len(bounds) if bounds else 1.0
        offspring_multiplier = config["optimize"].get("offspring_multiplier", 1.0)
        if not isinstance(offspring_multiplier, (int, float)) or offspring_multiplier <= 0.0:
            offspring_multiplier = 1.0

        # Register attribute generators
        for i, (low, high) in enumerate(bounds):
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        # Register genetic operators
        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=crossover_eta,
            low=[low for low, high in bounds],
            up=[high for low, high in bounds],
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=mutation_eta,
            low=[low for low, high in bounds],
            up=[high for low, high in bounds],
            indpb=mutation_indpb,
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluator.evaluate, overrides_list=overrides_list)

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
        population_size = config["optimize"]["population_size"]
        if (nstart := len(starting_individuals)) > population_size:
            logging.info(f"Number of starting configs greater than population size.")
            logging.info(f"Increasing population size: {population_size} -> {nstart}")
            config["optimize"]["population_size"] = nstart
            population_size = nstart

        def _make_random_individual():
            return creator.Individual([np.random.uniform(low, high) for low, high in bounds])

        population = [_make_random_individual() for _ in range(population_size)]
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
        lambda_size = max(1, int(round(config["optimize"]["population_size"] * offspring_multiplier)))
        population, logbook = ea_mu_plus_lambda_stream(
            population,
            toolbox,
            mu=config["optimize"]["population_size"],
            lambda_=lambda_size,
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=False,
            recorder=recorder,
            evaluator_config=evaluator.config,
            overrides_list=overrides_list,
            pool=pool,
        )

        logging.info("Optimization complete.")

    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Keyboard interrupt received; terminating optimization...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if "recorder" in locals():
            try:
                recorder.flush()
            except Exception:
                logging.exception("Failed to flush recorder")
            recorder.close()
        if "pool" in locals():
            if interrupted:
                logging.info("Terminating worker pool...")
                pool.terminate()
            else:
                logging.info("Closing worker pool...")
                pool.close()
            pool.join()
        if "array_manager" in locals():
            array_manager.cleanup()

        logging.info("Cleanup complete. Exiting.")
        sys.exit(130 if interrupted else 0)


if __name__ == "__main__":
    asyncio.run(main())
