import os
import sys
import argparse

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

# Rust extension check before importing compiled module
from rust_utils import check_and_maybe_compile

_rust_parser = argparse.ArgumentParser(add_help=False)
_rust_parser.add_argument("--skip-rust-compile", action="store_true", help="Skip Rust build check.")
_rust_parser.add_argument(
    "--force-rust-compile", action="store_true", help="Force rebuild of Rust extension."
)
_rust_parser.add_argument(
    "--fail-on-stale-rust",
    action="store_true",
    help="Abort if Rust extension appears stale instead of attempting rebuild.",
)
_rust_known, _rust_remaining = _rust_parser.parse_known_args()
try:
    check_and_maybe_compile(
        skip=_rust_known.skip_rust_compile
        or os.environ.get("SKIP_RUST_COMPILE", "").lower() in ("1", "true", "yes"),
        force=_rust_known.force_rust_compile,
        fail_on_stale=_rust_known.fail_on_stale_rust,
    )
except Exception as exc:
    print(f"Rust extension check failed: {exc}")
    sys.exit(1)
sys.argv = [sys.argv[0]] + _rust_remaining

from backtest import (
    prepare_hlcvs_mss,
)
import asyncio
import multiprocessing
import signal
import time
from collections import defaultdict
from config_utils import (
    get_template_config,
    load_hjson_config,
    load_config,
    format_config,
    add_config_arguments,
    update_config_with_args,
    recursive_config_update,
    require_config_value,
    merge_negative_cli_values,
    strip_config_metadata,
    get_optional_config_value,
)
from pure_funcs import (
    flatten,
    str2bool,
)
from utils import date_to_ts, ts_to_date, utc_ms, make_get_filepath, format_approved_ignored_coins
from logging_setup import configure_logging, resolve_log_level
from copy import deepcopy
from math import comb
import numpy as np
from uuid import uuid4
import logging
import traceback
import json

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

import fcntl
from optimizer_overrides import optimizer_overrides
from pareto_store import ParetoStore
from typing import List, Dict, Any
from shared_arrays import SharedArrayManager
from ohlcv_utils import align_and_aggregate_hlcvs
from optimize_suite import (
    ScenarioEvalContext,
    prepare_suite_contexts,
)
from suite_runner import (
    extract_suite_config,
    filter_scenarios_by_label,
)
from optimization.evaluator import Evaluator, SuiteEvaluator
from optimization.bounds import (
    extract_bounds_arrays,
    config_to_individual,
    apply_fine_tune_bounds,
    validate_array,
)
from optimization.problem import PassivbotProblem
from pymoo.parallelization.starmap import StarmapParallelization
from optimization.repair import SignificantDigitsRepair
from optimization.callback import ParetoWriterCallback
from optimization.output import OptimizeOutput


def compute_n_partitions(n_obj, pop_size):
    """Find n_partitions whose Das-Dennis ref point count best matches pop_size.

    The number of Das-Dennis reference points for a given n_obj and n_partitions is
    C(n_partitions + n_obj - 1, n_obj - 1). This function finds the n_partitions
    value that produces a reference point count closest to pop_size.
    """
    if n_obj < 2 or pop_size < 1:
        raise ValueError(f"n_obj must be >= 2 and pop_size >= 1, got n_obj={n_obj}, pop_size={pop_size}")
    p = 1
    while comb(p + n_obj - 1, n_obj - 1) < pop_size:
        p += 1
    # p is first where n_points >= pop_size; compare with p-1
    if p > 1:
        below = comb(p - 1 + n_obj - 1, n_obj - 1)
        above = comb(p + n_obj - 1, n_obj - 1)
        return p - 1 if (pop_size - below) <= (above - pop_size) else p
    return p


def _ignore_sigint_in_worker():
    """Ensure worker processes don't receive SIGINT so the parent controls shutdown."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass


_BOOL_LITERALS = {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}


def _looks_like_bool_token(value: str) -> bool:
    return value.lower() in _BOOL_LITERALS


def _normalize_optional_bool_flag(argv: list[str], flag: str) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if (
                next_token
                and not next_token.startswith("-")
                and not _looks_like_bool_token(next_token)
            ):
                result.append(f"{flag}=true")
                i += 1
                continue
        result.append(token)
        i += 1
    return result


def _maybe_aggregate_backtest_data(hlcvs, timestamps, btc_usd_prices, mss, config):
    candle_interval = int(config.get("backtest", {}).get("candle_interval_minutes", 1) or 1)
    if candle_interval <= 1:
        return hlcvs, timestamps, btc_usd_prices
    n_before = hlcvs.shape[0]
    hlcvs, timestamps, btc_usd_prices, offset_bars = align_and_aggregate_hlcvs(
        hlcvs, timestamps, btc_usd_prices, candle_interval
    )
    logging.debug(
        "[optimize] aggregated %dm candles: %d bars -> %d bars (trimmed %d for alignment)",
        candle_interval,
        n_before,
        hlcvs.shape[0],
        offset_bars,
    )
    meta = mss.setdefault("__meta__", {})
    meta["data_interval_minutes"] = candle_interval
    meta["candle_interval_offset_bars"] = int(offset_bars)
    if timestamps is not None and len(timestamps) > 0:
        meta["effective_start_ts"] = int(timestamps[0])
        meta["effective_start_date"] = ts_to_date(int(timestamps[0]))
    return hlcvs, timestamps, btc_usd_prices


def select_best_config(results_dir: str, *, n_objectives: int) -> str | None:
    """Select the best config from the Pareto front using pymoo pseudo-weights."""
    from pymoo.mcdm.pseudo_weights import PseudoWeights

    pareto_dir = os.path.join(results_dir, "pareto")
    if not os.path.isdir(pareto_dir):
        logging.warning("No pareto directory found in %s", results_dir)
        return None

    files = sorted(f for f in os.listdir(pareto_dir) if f.endswith(".json"))
    if not files:
        logging.warning("No pareto entries found in %s", pareto_dir)
        return None

    entries = []
    for fname in files:
        try:
            with open(os.path.join(pareto_dir, fname)) as f:
                entries.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Skipping unreadable pareto entry %s: %s", fname, exc)

    if not entries:
        logging.warning("No readable pareto entries in %s", pareto_dir)
        return None

    if len(entries) == 1:
        best = entries[0]
    else:
        F = np.array(
            [[e["objectives"][f"w_{i}"] for i in range(n_objectives)] for e in entries]
        )
        weights = np.ones(n_objectives) / n_objectives
        idx = PseudoWeights(weights).do(F)
        best = entries[idx]

    output_path = os.path.join(results_dir, "best_config.json")
    with open(output_path, "w") as f:
        json.dump(best, f, indent=4)

    objectives = best["objectives"]
    obj_str = ", ".join(f"{k}={abs(v):.4f}" for k, v in sorted(objectives.items()))
    logging.info("Best config selected via pseudo weights: %s", obj_str)
    logging.info("Written to %s", output_path)

    return output_path


logging.basicConfig(
    format="%(asctime)s %(processName)-12s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


TEMPLATE_CONFIG_MODE = "v7"






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


def configs_to_individuals(configs, xl, xu):
    individuals = []
    for c in configs:
        try:
            formatted = format_config(c)
            ind = config_to_individual(formatted, xl, xu)
            individuals.append(ind)
        except Exception as e:
            logging.warning("Could not convert starting config: %s", e)
    return individuals


async def main():
    parser = argparse.ArgumentParser(prog="optimize", description="run optimizer")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    parser.add_argument(
        "--suite",
        nargs="?",
        const="true",
        default=None,
        type=str2bool,
        metavar="y/n",
        help="Enable or disable suite mode for optimizer run (omit to use config's suite_enabled setting).",
    )
    parser.add_argument(
        "--scenarios",
        "-sc",
        type=str,
        default=None,
        metavar="LABELS",
        help="Comma-separated list of scenario labels to run (implies --suite y). "
        "Example: --scenarios base,binance_only",
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        default=None,
        help="Optional config file providing backtest.scenarios overrides.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )
    template_config = get_template_config()
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_config_arguments(parser, template_config)
    add_extra_options(parser)
    raw_args = merge_negative_cli_values(sys.argv[1:])
    raw_args = _normalize_optional_bool_flag(raw_args, "--suite")
    args = parser.parse_args(raw_args)
    initial_log_level = resolve_log_level(args.log_level, None, fallback=1)
    configure_logging(debug=initial_log_level)
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=True)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path, verbose=True)
    update_config_with_args(config, args, verbose=True)
    config = format_config(config, verbose=False)
    config_logging_value = get_optional_config_value(config, "logging.level", None)
    effective_log_level = resolve_log_level(args.log_level, config_logging_value, fallback=1)
    if effective_log_level != initial_log_level:
        configure_logging(debug=effective_log_level)
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
    cli_bounds_overrides = {
        key.split("optimize.bounds.", 1)[1]
        for key, value in vars(args).items()
        if key.startswith("optimize.bounds.") and value is not None
    }
    apply_fine_tune_bounds(config, fine_tune_params, cli_bounds_overrides)
    if fine_tune_params:
        logging.info(
            "Fine-tuning mode active for %s",
            ", ".join(sorted(fine_tune_params)),
        )
    suite_override = None
    if args.suite_config:
        logging.info("loading suite config %s", args.suite_config)
        override_cfg = load_config(args.suite_config, verbose=False)
        override_backtest = override_cfg.get("backtest", {})
        # Support both new (scenarios at top level) and legacy (suite wrapper) formats
        if "scenarios" in override_backtest:
            suite_override = {
                "scenarios": override_backtest.get("scenarios", []),
                "aggregate": override_backtest.get("aggregate", {"default": "mean"}),
            }
        elif "suite" in override_backtest:
            # Legacy format - extract from suite wrapper
            suite_override = override_backtest["suite"]
        else:
            raise ValueError(f"Suite config {args.suite_config} must define backtest.scenarios.")
    suite_cfg = extract_suite_config(config, suite_override)

    # Handle --scenarios filter (implies --suite y)
    scenario_filter = getattr(args, "scenarios", None)
    if scenario_filter:
        labels = [label.strip() for label in scenario_filter.split(",") if label.strip()]
        suite_cfg["scenarios"] = filter_scenarios_by_label(suite_cfg.get("scenarios", []), labels)
        suite_cfg["enabled"] = True  # --scenarios implies suite mode
        logging.info("Filtered to %d scenario(s): %s", len(labels), ", ".join(labels))

    # --suite CLI arg overrides config (applied after --scenarios so explicit --suite n wins)
    if args.suite is not None:
        recursive_config_update(config, "backtest.suite_enabled", bool(args.suite), verbose=True)
        suite_cfg["enabled"] = bool(args.suite)
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(config, backtest_exchanges)
    interrupted = False
    pool = None
    try:
        array_manager = SharedArrayManager()
        hlcvs_specs = {}
        btc_usd_specs = {}
        msss = {}
        timestamps_dict = {}
        config["backtest"]["coins"] = {}
        aggregate_cfg: Dict[str, Any] = {"default": "mean"}
        scenario_contexts: List[ScenarioEvalContext] = []
        suite_enabled = bool(suite_cfg.get("enabled"))

        if suite_enabled:
            scenario_contexts, aggregate_cfg = await prepare_suite_contexts(
                config,
                suite_cfg,
                shared_array_manager=array_manager,
            )
            if not scenario_contexts:
                raise ValueError("Suite configuration produced no scenarios.")
            logging.info("Optimizer suite enabled with %d scenario(s)", len(scenario_contexts))
            first_ctx = scenario_contexts[0]
            hlcvs_specs = first_ctx.hlcvs_specs
            btc_usd_specs = first_ctx.btc_usd_specs
            msss = first_ctx.msss
            timestamps_dict = first_ctx.timestamps
            config["backtest"]["coins"] = deepcopy(first_ctx.config["backtest"]["coins"])
            backtest_exchanges = sorted({ex for ctx in scenario_contexts for ex in ctx.exchanges})

            # Estimate memory usage (per-scenario SharedMemory, shared by all workers)
            total_shm_bytes = 0
            seen_specs = set()
            for ctx in scenario_contexts:
                for spec_map in (
                    ctx.hlcvs_specs,
                    ctx.btc_usd_specs,
                    ctx.master_hlcvs_specs or {},
                    ctx.master_btc_specs or {},
                ):
                    for spec in spec_map.values():
                        if spec is None:
                            continue
                        if spec.name in seen_specs:
                            continue
                        seen_specs.add(spec.name)
                        total_shm_bytes += np.prod(spec.shape) * np.dtype(spec.dtype).itemsize
            if total_shm_bytes > 0:
                total_shm_gb = total_shm_bytes / (1024**3)
                try:
                    import shutil

                    if hasattr(os, "sysconf"):
                        pages = os.sysconf("SC_PHYS_PAGES")
                        page_size = os.sysconf("SC_PAGE_SIZE")
                        available_gb = (pages * page_size) / (1024**3)
                    else:
                        available_gb = None
                    shm_gb = None
                    if os.path.exists("/dev/shm"):
                        usage = shutil.disk_usage("/dev/shm")
                        shm_gb = usage.total / (1024**3)
                except Exception:
                    available_gb = None
                    shm_gb = None
                logging.info(
                    "Memory estimate | scenarios=%d | shared_memory=%.1fGB%s",
                    len(scenario_contexts),
                    total_shm_gb,
                    f" | system={available_gb:.1f}GB" if available_gb else "",
                )
                if shm_gb is not None:
                    logging.info("Shared memory filesystem size | /dev/shm=%.1fGB", shm_gb)
                if available_gb and total_shm_gb > available_gb * 0.7:
                    logging.warning(
                        "Shared memory for scenarios (%.1fGB) is high relative to RAM (%.1fGB). "
                        "Consider using fewer/smaller scenarios.",
                        total_shm_gb,
                        available_gb,
                    )
        else:
            # New behavior: derive data strategy from exchange count
            # - Single exchange = use that exchange's data only
            # - Multiple exchanges = best-per-coin combination (combined)
            use_combined = len(backtest_exchanges) > 1

            if use_combined:
                exchange = "combined"
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = (
                    await prepare_hlcvs_mss(config, exchange)
                )
                hlcvs, _timestamps, btc_usd_prices = _maybe_aggregate_backtest_data(
                    hlcvs, _timestamps, btc_usd_prices, mss, config
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
                    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = (
                        await tasks[exchange]
                    )
                    hlcvs, _timestamps, btc_usd_prices = _maybe_aggregate_backtest_data(
                        hlcvs, _timestamps, btc_usd_prices, mss, config
                    )
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
        exchanges_fname = "combined" if len(backtest_exchanges) > 1 else "_".join(exchanges)
        date_fname = ts_to_date(utc_ms())[:19].replace(":", "_")
        coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
        suite_flag = suite_enabled or bool(args.suite)
        if suite_flag:
            coins_fname = f"suite_{len(coins)}_coins"
        else:
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

        # Initialize evaluator with shared memory references
        evaluator = Evaluator(
            hlcvs_specs=hlcvs_specs,
            btc_usd_specs=btc_usd_specs,
            msss=msss,
            config=config,
            timestamps=timestamps_dict,
            shared_array_manager=array_manager,
        )

        if suite_enabled:
            evaluator_for_pool = SuiteEvaluator(evaluator, scenario_contexts, aggregate_cfg)
        else:
            evaluator_for_pool = evaluator

        logging.info(f"Finished initializing evaluator...")
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        store = ParetoStore(directory=results_dir, sig_digits=sig_digits)
        xl, xu, _ = extract_bounds_arrays(config)

        # Parallelization setup
        logging.info(f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}")
        pool = multiprocessing.Pool(
            processes=config["optimize"]["n_cpus"],
            initializer=_ignore_sigint_in_worker,
        )
        logging.info(f"Finished initializing multiprocessing pool.")

        runner = StarmapParallelization(pool.starmap)
        problem = PassivbotProblem(
            config=config,
            evaluator=evaluator_for_pool,
            elementwise_runner=runner,
        )

        n_obj = len(config["optimize"]["scoring"])
        configured_pop_size = config["optimize"]["population_size"]
        n_partitions = compute_n_partitions(n_obj, configured_pop_size)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        n_ref = len(ref_dirs)
        # NSGA-III needs pop_size >= ref_dirs; Deb & Jain recommend smallest multiple of 4
        pop_size = n_ref + (-n_ref % 4)
        if pop_size != configured_pop_size:
            logging.info(
                f"Adjusted pop_size from {configured_pop_size} to {pop_size} "
                f"to match {n_ref} reference directions"
            )
        n_gen = config["optimize"]["iters"] // pop_size
        logging.info(
            f"Generated {n_ref} reference directions "
            f"(n_obj={n_obj}, n_partitions={n_partitions})"
        )
        logging.info(f"Using {n_gen} generations (pop_size={pop_size}, iters={config['optimize']['iters']})")

        # Seeding from --start
        starting_configs = get_starting_configs(args.starting_configs)
        X_init = None
        if starting_configs:
            logging.info("Loaded %d starting configs", len(starting_configs))
            starting_individuals = configs_to_individuals(
                starting_configs, xl, xu
            )
            if starting_individuals:
                X_init = np.array(starting_individuals)
                if len(X_init) < pop_size:
                    n_random = pop_size - len(X_init)
                    random_pop = np.random.uniform(
                        xl, xu, size=(n_random, len(xl))
                    )
                    X_init = np.vstack([X_init, random_pop])
                elif len(X_init) > pop_size:
                    X_init = X_init[:pop_size]
        else:
            logging.info("No starting configs; population will be random-initialized")

        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=X_init if X_init is not None else FloatRandomSampling(),
            crossover=SBX(
                eta=config["optimize"].get("crossover_eta", 20.0),
                prob=config["optimize"]["crossover_probability"],
            ),
            mutation=PM(
                eta=config["optimize"].get("mutation_eta", 20.0),
                prob=config["optimize"]["mutation_probability"],
            ),
            repair=SignificantDigitsRepair(sig_digits),
        )

        termination = get_termination("n_gen", n_gen)

        write_all_results = config.get("optimize", {}).get("write_all_results", True)
        callback = ParetoWriterCallback(
            store=store,
            template=config,
            scoring_keys=config["optimize"]["scoring"],
            overrides_fn=optimizer_overrides,
            overrides_list=overrides_list,
            results_dir=results_dir,
            write_all_results=write_all_results,
        )

        logging.info("Starting pymoo NSGA-III optimization...")
        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            callback=callback,
            output=OptimizeOutput(config["optimize"]["scoring"]),
            verbose=True,
            seed=1,
        )
        logging.info("Optimization complete. %d evaluations.", res.algorithm.evaluator.n_eval)

    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Keyboard interrupt received; terminating optimization...")
        if "pool" in locals():
            logging.info("Terminating worker pool...")
            pool.terminate()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if "callback" in locals():
            try:
                callback.close()
            except Exception:
                logging.exception("Failed to close callback")
        if "store" in locals():
            try:
                store.flush()
            except Exception:
                logging.exception("Failed to flush pareto store")
        if "results_dir" in locals():
            try:
                n_obj = len(config.get("optimize", {}).get("scoring", []))
                if n_obj > 0:
                    select_best_config(results_dir, n_objectives=n_obj)
            except Exception:
                logging.exception("Failed to select best config")
        if "pool" in locals() and pool is not None:
            if interrupted:
                logging.info("Joining terminated worker pool...")
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
