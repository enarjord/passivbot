#!/usr/bin/env python3
"""Optuna optimizer orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import typer

# External dependencies - owned by orchestrator, not package
from backtest import build_backtest_payload, execute_backtest, prepare_hlcvs_mss
from config_utils import load_config, require_config_value
from logging_setup import configure_logging
from optimize import build_scenario_metrics, flatten_metric_stats

# Package imports - pure Optuna logic
from optuna_optimizer import (
    apply_params_to_config,
    check_constraints,
    compute_scores,
    create_journal_storage,
    create_sampler,
    extract_bounds,
    extract_constraints,
    extract_objectives,
    extract_optuna_config,
    extract_params_from_config,
    extract_pareto,
    get_sampler_config_by_name,
    load_seed_configs,
    make_constraints_func,
    resolve_metric,
    sample_params,
    SharedArrayManager,
    SharedArraySpec,
)
from optuna_optimizer.models import Objective
from optuna_optimizer.worker import WorkerContext, WorkerInitData, get_context, init_worker


# ---------------------------------------------------------------------------
# Worker functions (run in child processes)
# ---------------------------------------------------------------------------
def _run_single_trial(_: int) -> None:
    """Run exactly one optimization trial.

    This is designed for single-trial dispatch via pool.imap_unordered(),
    allowing the parent process to interrupt between trials. The dummy
    argument is the trial index from range(n_trials).
    """
    ctx = get_context()
    ctx.study.optimize(_objective, n_trials=1)


def _objective(trial: optuna.Trial) -> tuple[float, ...]:
    """Objective function - bridges Optuna and backtest."""
    ctx = get_context()
    trial_num = trial.number

    # 1. Sample and apply params
    bot_params = sample_params(trial, ctx.bounds, fixed_params=ctx.fixed_params)
    logging.debug(f"[{trial_num}] {_format_params(bot_params)}")
    trial_config = apply_params_to_config(bot_params, ctx.config)

    # 2. Run backtests
    analyses = _run_backtests(ctx, trial_config)

    # 3. Aggregate and score
    scenario_metrics = build_scenario_metrics(analyses)
    flat_stats = flatten_metric_stats(scenario_metrics.get("stats", {}))

    resolved = {c.metric: resolve_metric(c.metric, flat_stats) for c in ctx.constraints}
    violations = check_constraints(resolved, ctx.constraints)
    trial.set_user_attr("constraint_violations", violations)

    scores = compute_scores(flat_stats, ctx.objectives, violations, ctx.penalty_weight)

    # 4. Log completion
    obj_summary = ", ".join(
        f"{obj.metric}={resolve_metric(obj.metric, flat_stats):.5f}" for obj in ctx.objectives
    )
    logging.info(f"[{trial_num}] {obj_summary}")

    return scores


def _format_params(params: dict) -> str:
    """Format params dict for logging."""
    items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
    return ", ".join(items)


def _run_backtests(ctx: WorkerContext, trial_config: dict) -> dict[str, dict]:
    """Run backtest for each exchange, return analyses keyed by exchange."""
    analyses = {}
    for exchange in ctx.exchanges:
        payload = build_backtest_payload(
            ctx.hlcvs[exchange],
            ctx.mss[exchange],
            trial_config,
            exchange,
            ctx.btc[exchange],
            ctx.timestamps[exchange],
        )
        _, _, analysis = execute_backtest(payload, trial_config)
        analyses[exchange] = analysis
    return analyses


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    hours, remainder = divmod(int(seconds), 3600)
    mins, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _log_optimization_summary(
    study: "optuna.Study",
    objectives: list[Objective],
    n_trials_planned: int,
    start_time: float,
) -> None:
    """Log comprehensive optimization summary.

    Args:
        study: Optuna study with completed trials
        objectives: List of optimization objectives
        n_trials_planned: Number of trials that were planned
        start_time: Unix timestamp when optimization started
    """
    from optuna.trial import TrialState

    trials = study.trials
    n_total = len(trials)
    elapsed = time.time() - start_time

    # Count trial states
    complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]
    n_complete = len(complete_trials)
    n_pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
    n_failed = sum(1 for t in trials if t.state == TrialState.FAIL)

    # Count constraint violations
    n_satisfied = sum(
        1 for t in complete_trials
        if not any(v > 0 for v in t.user_attrs.get("constraint_violations", []))
    )

    # Compute throughput and remaining estimate
    throughput = n_total / elapsed * 60 if elapsed > 0 else 0
    remaining_trials = n_trials_planned - n_total
    remaining_time = remaining_trials / throughput * 60 if throughput > 0 else 0

    # Get best values for each objective from Pareto front
    pareto_trials = study.best_trials
    best_values = {}
    if pareto_trials:
        for i, obj in enumerate(objectives):
            values = [t.values[i] * obj.sign for t in pareto_trials]
            best_values[obj.metric] = max(values) if obj.sign == 1 else min(values)

    # Build and log summary
    pct_complete = (n_total / n_trials_planned * 100) if n_trials_planned > 0 else 0
    pct_satisfied = (n_satisfied / n_complete * 100) if n_complete > 0 else 0

    lines = [
        "=" * 65,
        "Optimization Summary",
        "=" * 65,
        f"Trials:      {n_total}/{n_trials_planned} completed ({pct_complete:.1f}%)",
        f"             {n_complete} complete, {n_pruned} pruned, {n_failed} failed",
        f"Constraints: {n_satisfied}/{n_complete} satisfied ({pct_satisfied:.1f}%)",
        "-" * 65,
        f"Elapsed:     {_format_duration(elapsed)}",
        f"Throughput:  {throughput:.1f} trials/min",
    ]

    if remaining_trials > 0:
        lines.append(f"Remaining:   ~{_format_duration(remaining_time)} (estimated)")

    lines.append("-" * 65)

    if best_values:
        lines.append("Best values:")
        for metric, value in best_values.items():
            lines.append(f"  {metric}: {value:.6f}")
        lines.append(f"Pareto front: {len(pareto_trials)} trials")
    else:
        lines.append("No complete trials yet")

    lines.append("=" * 65)

    for line in lines:
        logging.info(line)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def _run_optimization_core(
    config: dict,
    study_dir: Path,
    study: optuna.Study,
    bounds: dict,
    constraints: list,
    objectives: list[Objective],
    sampler_config,
    n_trials: int,
    n_cpus: int,
    fixed_params: dict | None,
    penalty_weight: float,
    max_best_trials: int,
    debug_level: int,
) -> None:
    """Core optimization loop shared by run and resume.

    Args:
        config: Full configuration dict
        study_dir: Directory for study artifacts
        study: Optuna study (already created or loaded)
        bounds: Parameter bounds
        constraints: Metric constraints
        objectives: Optimization objectives
        sampler_config: Sampler configuration (recreated in workers)
        n_trials: Number of trials to run
        n_cpus: Number of worker processes
        fixed_params: Parameters to skip sampling (for fine-tune mode)
        penalty_weight: Constraint penalty weight (-1 = hard, 0 = disabled, >0 = soft)
        max_best_trials: Maximum Pareto configs to export
        debug_level: Logging verbosity
    """
    array_manager = SharedArrayManager()
    start_time = time.time()

    try:
        # Load OHLCV data into shared memory
        logging.info("Loading OHLCV data...")
        hlcvs_specs, btc_specs, ts_specs, mss = await _load_shared_data(
            config, array_manager
        )
        exchanges = list(hlcvs_specs.keys())
        logging.info(f"Loaded {len(exchanges)} exchange(s): {exchanges}")

        logging.info(f"Study: {study_dir}")
        logging.info(f"Trials: {n_trials}, Workers: {n_cpus}")

        # Run workers with single-trial dispatch for responsive interrupts
        init_data = WorkerInitData(
            hlcvs_specs=hlcvs_specs,
            btc_specs=btc_specs,
            ts_specs=ts_specs,
            mss=mss,
            config=config,
            study_dir=str(study_dir),
            bounds=bounds,
            constraints=constraints,
            objectives=objectives,
            sampler_config=sampler_config,
            fixed_params=fixed_params,
            penalty_weight=penalty_weight,
            debug_level=debug_level,
            logging_module="logging_setup",
        )
        # Get population size for generation-based dispatch
        population_size = getattr(sampler_config, "population_size", n_trials)
        n_generations = (n_trials + population_size - 1) // population_size

        with Pool(processes=n_cpus, initializer=init_worker, initargs=(init_data,)) as pool:
            try:
                trials_completed = 0
                for gen in range(n_generations):
                    batch_size = min(population_size, n_trials - trials_completed)

                    # Dispatch one generation, wait for all to complete
                    list(pool.imap_unordered(_run_single_trial, range(batch_size)))
                    trials_completed += batch_size

                    logging.info(f"Generation {gen} complete ({trials_completed}/{n_trials} trials)")
            except KeyboardInterrupt:
                logging.info("Interrupted - terminating workers...")
                pool.terminate()
                pool.join()

    finally:
        # Always attempt Pareto extraction (even on interrupt)
        journal_path = study_dir / "study.log"
        if journal_path.exists():
            try:
                storage = create_journal_storage(journal_path)
                final_study = optuna.load_study(study_name=study_dir.name, storage=storage)
                if final_study.trials:
                    _log_optimization_summary(final_study, objectives, n_trials, start_time)
                    logging.info("Extracting Pareto front, please wait...")
                    extract_pareto(final_study, study_dir, objectives, config, max_best_trials)
                else:
                    logging.warning("No trials completed, skipping Pareto extraction")
            except Exception as e:
                logging.warning(f"Could not extract Pareto front: {e}")
        else:
            logging.warning("No study.log found, skipping Pareto extraction")

        array_manager.cleanup()


async def run_optimization(
    config_path: Path,
    n_trials: int | None = None,
    n_cpus: int | None = None,
    study_name: str | None = None,
    sampler_name: str | None = None,
    fine_tune: str | None = None,
    start: Path | None = None,
    debug_level: int = 1,
) -> None:
    """Start a new optimization.

    Args:
        config_path: Path to configuration file
        n_trials: Number of trials (overrides config)
        n_cpus: Number of workers (overrides config)
        study_name: Custom study name (default: timestamp_hash)
        sampler_name: Sampler override (tpe/nsgaii/nsgaiii/gp/random)
        fine_tune: Comma-separated params to tune (others fixed)
        start: Path to seed configs file or directory
        debug_level: Logging verbosity
    """
    config = load_config(str(config_path))
    optuna_cfg = extract_optuna_config(config)
    bounds = extract_bounds(config)
    constraints = extract_constraints(config)
    objectives = extract_objectives(config)

    if not objectives:
        raise ValueError("No objectives defined in config['optimize']['objectives']")

    n_trials = n_trials or optuna_cfg.n_trials
    n_cpus = n_cpus or optuna_cfg.n_cpus
    penalty_weight = optuna_cfg.penalty_weight

    # Handle fine-tune mode
    fixed_params = None
    if fine_tune:
        tune_keys = set(fine_tune.split(","))
        fixed_params = {
            name: config.get("live", {}).get(name, bounds[name].low)
            for name in bounds
            if name not in tune_keys
        }

    # Create study directory and study
    study_dir = _create_study_dir(config, study_name)
    journal_path = study_dir / "study.log"
    storage = create_journal_storage(journal_path)

    sampler_cfg = optuna_cfg.sampler
    if sampler_name:
        sampler_cfg = get_sampler_config_by_name(sampler_name)

    # Hard mode: pass constraints_func to sampler
    constraints_func = make_constraints_func() if penalty_weight == -1 else None

    study = optuna.create_study(
        study_name=study_dir.name,
        storage=storage,
        directions=["minimize"] * len(objectives),
        sampler=create_sampler(sampler_cfg, constraints_func),
    )

    # Store fine-tune params for resume
    if fine_tune:
        study.set_user_attr("fine_tune_params", list(tune_keys))

    # Enqueue seed configs
    if start:
        seed_configs = load_seed_configs(
            start, lambda p: load_config(p, live_only=False, verbose=False)
        )
        enqueued = 0
        for cfg in seed_configs:
            params = extract_params_from_config(cfg, bounds)
            if params:
                study.enqueue_trial(params, skip_if_exists=True)
                enqueued += 1
            else:
                logging.warning("Skipped seed config (no valid params)")
        logging.info(f"Enqueued {enqueued} seed configs from {start}")

    # Save config for resume
    (study_dir / "config.json").write_text(json.dumps(config, indent=2))
    mode = "hard" if penalty_weight == -1 else "disabled" if penalty_weight == 0 else f"soft (weight={penalty_weight})"
    logging.info(f"Sampler: {sampler_cfg.name}, Constraints: {mode}")

    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params, penalty_weight,
        optuna_cfg.max_best_trials, debug_level
    )


async def _load_shared_data(config: dict, manager: SharedArrayManager):
    """Load OHLCV and create shared memory arrays."""
    hlcvs_specs: dict[str, SharedArraySpec] = {}
    btc_specs: dict[str, SharedArraySpec] = {}
    ts_specs: dict[str, SharedArraySpec] = {}
    mss_all = {}

    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    combine_ohlcvs = config.get("backtest", {}).get("combine_ohlcvs", False)
    exchanges = ["combined"] if combine_ohlcvs else backtest_exchanges
    logging.debug(f"Loading data for {len(exchanges)} exchange(s): {exchanges}")

    for exchange in exchanges:
        logging.debug(f"{exchange}: Loading OHLCV data...")
        coins, hlcvs, mss, _, _, btc_usd, timestamps = await prepare_hlcvs_mss(
            config, exchange
        )
        logging.debug(f"{exchange}: Loaded {len(coins)} coins, shape={hlcvs.shape}")

        logging.debug(f"{exchange}: Creating shared memory arrays...")
        hlcvs_spec, _ = manager.create_from(
            np.ascontiguousarray(hlcvs, dtype=np.float64)
        )
        btc_spec, _ = manager.create_from(
            np.ascontiguousarray(btc_usd, dtype=np.float64)
        )
        ts_spec, _ = manager.create_from(
            np.ascontiguousarray(timestamps, dtype=np.int64)
        )

        hlcvs_specs[exchange] = hlcvs_spec
        btc_specs[exchange] = btc_spec
        ts_specs[exchange] = ts_spec
        mss_all[exchange] = mss

        config.setdefault("backtest", {}).setdefault("coins", {})[exchange] = coins
        logging.debug(f"{exchange}: Done")

    return hlcvs_specs, btc_specs, ts_specs, mss_all


def _create_study_dir(config: dict, study_name: str | None) -> Path:
    if study_name:
        dirname = study_name
    else:
        ts = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
        h = md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        dirname = f"{ts}_{h}"

    study_dir = Path("optimize_results") / dirname
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def _derive_fixed_params(
    tune_params: list[str],
    bounds: dict,
    config: dict,
) -> dict[str, float]:
    """Derive fixed params from tune params list.

    Args:
        tune_params: List of parameter names being tuned
        bounds: All parameter bounds
        config: Full config to get default values from

    Returns:
        Dict of param name -> fixed value for non-tuned params
    """
    tune_keys = set(tune_params)
    fixed = {}

    bot = config.get("bot", {})
    for name, bound in bounds.items():
        if name in tune_keys:
            continue
        # Get value from config, falling back to bound.low
        if name.startswith("long_"):
            value = bot.get("long", {}).get(name[5:], bound.low)
        elif name.startswith("short_"):
            value = bot.get("short", {}).get(name[6:], bound.low)
        else:
            value = config.get("live", {}).get(name, bound.low)
        fixed[name] = value

    return fixed


async def resume_optimization(
    study_dir: Path,
    n_trials: int | None = None,
    n_cpus: int | None = None,
    debug_level: int = 1,
) -> None:
    """Resume an existing optimization study.

    Args:
        study_dir: Directory containing existing study
        n_trials: Additional trials to run (overrides config)
        n_cpus: Number of workers (overrides config)
        debug_level: Logging verbosity
    """
    config_path = study_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in {study_dir}")

    journal_path = study_dir / "study.log"
    if not journal_path.exists():
        raise ValueError(f"No study.log found in {study_dir}")
    storage = create_journal_storage(journal_path)

    config = load_config(str(config_path), live_only=False, verbose=False)
    optuna_cfg = extract_optuna_config(config)
    bounds = extract_bounds(config)
    constraints = extract_constraints(config)
    objectives = extract_objectives(config)

    if not objectives:
        raise ValueError("No objectives defined in config['optimize']['objectives']")

    n_trials = n_trials or optuna_cfg.n_trials
    n_cpus = n_cpus or optuna_cfg.n_cpus
    penalty_weight = optuna_cfg.penalty_weight

    # Recreate sampler from config (load_study uses default otherwise)
    sampler_cfg = optuna_cfg.sampler
    constraints_func = make_constraints_func() if penalty_weight == -1 else None
    sampler = create_sampler(sampler_cfg, constraints_func)

    # Load existing study with correct sampler
    study = optuna.load_study(study_name=study_dir.name, storage=storage, sampler=sampler)
    existing_trials = len(study.trials)

    # Restore fine-tune mode from study metadata
    fixed_params = None
    tune_params = study.user_attrs.get("fine_tune_params")
    if tune_params:
        fixed_params = _derive_fixed_params(tune_params, bounds, config)
        logging.info(f"Restored fine-tune mode: tuning {tune_params}")

    logging.info(f"Resuming study: {study_dir}")
    logging.info(f"Existing trials: {existing_trials}, Adding: {n_trials}")
    mode = "hard" if penalty_weight == -1 else "disabled" if penalty_weight == 0 else f"soft (weight={penalty_weight})"
    logging.info(f"Sampler: {sampler_cfg.name}, Constraints: {mode}")

    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params, penalty_weight,
        optuna_cfg.max_best_trials, debug_level
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: Path = typer.Argument(..., help="Config file (new) or study directory (resume)"),
    n_trials: int | None = typer.Option(None, "--n-trials", "-n", help="Number of trials"),
    n_cpus: int | None = typer.Option(None, "--n-cpus", "-c", help="Number of workers"),
    study_name: str | None = typer.Option(None, "--study-name", "-s", help="Study name (new only)"),
    sampler: str | None = typer.Option(None, "--sampler", help="Sampler: tpe/nsgaii/nsgaiii/gp/random (new only)"),
    fine_tune: str | None = typer.Option(None, "--fine-tune", "-ft", help="Params to tune (new only)"),
    start: Path | None = typer.Option(None, "--start", "-t", help="Seed configs file/dir (new only)"),
    debug_level: int = typer.Option(1, "--debug-level", "-d", help="Log level: 0=warn, 1=info, 2=debug, 3=trace"),
):
    """Run Optuna optimization.

    Pass a config file to start a new optimization.
    Pass a study directory to resume an existing optimization.
    """
    configure_logging(debug=debug_level)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        if path.is_dir():
            asyncio.run(resume_optimization(path, n_trials, n_cpus, debug_level))
        elif path.is_file():
            asyncio.run(
                run_optimization(path, n_trials, n_cpus, study_name, sampler, fine_tune, start, debug_level)
            )
        else:
            raise typer.BadParameter(f"Path does not exist: {path}")
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    app()
