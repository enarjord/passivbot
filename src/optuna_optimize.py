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
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
import typer
import passivbot_rust as pbr

# External dependencies - owned by orchestrator, not package
from backtest import build_backtest_payload, prepare_hlcvs_mss
from config_utils import load_config, require_config_value
from logging_setup import configure_logging
from optimize import build_scenario_metrics, flatten_metric_stats

# Package imports - pure Optuna logic
from optuna_optimizer import (
    BotParamsTemplate,
    build_distributions,
    check_constraints,
    compute_scores,
    create_sampler,
    dump_to_sqlite,
    extract_bounds,
    extract_constraints,
    extract_objectives,
    extract_optuna_config,
    extract_params_from_config,
    extract_pareto,
    get_sampler_config_by_name,
    load_from_sqlite,
    load_seed_configs,
    make_constraints_func,
    resolve_metric,
    sample_params,
    InMemoryJournalBackend,
)
from optuna_optimizer.models import Objective


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _format_params(params: dict) -> str:
    """Format params dict for logging."""
    items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
    return ", ".join(items)


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
    """Log optimization summary.

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

    complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]
    n_complete = len(complete_trials)

    # Count constraint violations
    n_satisfied = sum(
        1 for t in complete_trials
        if not any(v > 0 for v in t.user_attrs.get("constraint_violations", []))
    )

    # Get best values for each objective from Pareto front
    pareto_trials = study.best_trials
    best_values = {}
    if pareto_trials:
        for i, obj in enumerate(objectives):
            values = [t.values[i] * obj.sign for t in pareto_trials]
            best_values[obj.metric] = max(values) if obj.sign == 1 else min(values)

    # Build and log summary
    pct_satisfied = (n_satisfied / n_complete * 100) if n_complete > 0 else 0

    throughput = n_total / elapsed * 60 if elapsed > 0 else 0

    lines = [
        "=" * 50,
        "Optimization Complete",
        "=" * 50,
        f"Elapsed:     {_format_duration(elapsed)} ({throughput:.1f} trials/min)",
    ]

    # ETA line only if incomplete
    remaining_trials = n_trials_planned - n_total
    if remaining_trials > 0 and throughput > 0:
        remaining_time = remaining_trials / throughput * 60
        lines.append(f"Remaining:   ~{_format_duration(remaining_time)}")

    lines.append(f"Constraints: {n_satisfied}/{n_complete} satisfied ({pct_satisfied:.1f}%)")

    if best_values:
        lines.append("Best values:")
        for metric, value in best_values.items():
            lines.append(f"  {metric}: {value:.6f}")
        lines.append(f"Pareto front: {len(pareto_trials)} configs")
    else:
        lines.append("No complete trials yet")

    lines.append("=" * 50)

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
    max_best_trials: int,
    debug_level: int,
    backend: InMemoryJournalBackend,
) -> None:
    """Core optimization loop using ask/tell pattern with Rust Rayon parallelism.

    Args:
        config: Full configuration dict
        study_dir: Directory for study artifacts
        study: Optuna study (already created or loaded)
        bounds: Parameter bounds
        constraints: Metric constraints
        objectives: Optimization objectives
        sampler_config: Sampler configuration
        n_trials: Number of trials to run
        n_cpus: Number of threads for Rust Rayon thread pool
        fixed_params: Parameters to skip sampling (for fine-tune mode)
        max_best_trials: Maximum Pareto configs to export
        debug_level: Logging verbosity
        backend: In-memory journal backend
    """
    start_time = time.time()

    # Configure Rust thread pool once before any parallel work
    pbr.configure_thread_pool(n_cpus)
    logging.info(f"Configured Rayon thread pool with {n_cpus} threads")

    try:
        # Load OHLCV data once (no shared memory needed - single process)
        logging.info("Loading OHLCV data...")
        payloads, mss_dict = await _load_all_data(config)
        exchanges = list(payloads.keys())
        logging.info(f"Loaded {len(exchanges)} exchange(s): {exchanges}")

        # Create bot_params templates for each exchange (avoids deepcopy in hot loop)
        templates = {}
        for exchange in exchanges:
            coins = payloads[exchange].backtest_params["coins"]
            templates[exchange] = BotParamsTemplate.from_config(config, coins)
            logging.debug(f"Created BotParamsTemplate for {exchange} with {len(coins)} coins")

        # Build distributions once for faster suggest_float calls (~10x speedup)
        distributions = build_distributions(bounds)
        logging.debug(f"Built {len(distributions)} parameter distributions")

        logging.info(f"Study: {study_dir}")
        logging.info(f"Trials: {n_trials}, CPU cores: {n_cpus}")

        # Generation-based ask/tell loop
        population_size = sampler_config.population_size
        n_generations = (n_trials + population_size - 1) // population_size
        trials_completed = 0

        def create_trial_direct() -> optuna.Trial:
            """Create trial directly, bypassing O(n) waiting trial scan."""
            trial_id = study._storage.create_new_trial(study._study_id)
            return optuna.Trial(study, trial_id)

        try:
            for gen in range(n_generations):
                batch_size = min(population_size, n_trials - trials_completed)

                # 1. Ask: Create batch of trials
                # Gen 0: use study.ask() to consume any enqueued seed configs
                # Gen 1+: create trials directly to avoid O(n) waiting trial scan
                if gen == 0:
                    trials = [study.ask() for _ in range(batch_size)]
                else:
                    trials = [create_trial_direct() for _ in range(batch_size)]

                # 2. Sample params and build bot_params for each trial
                sampled_params_batch = []
                for trial in trials:
                    params = sample_params(trial, bounds, fixed_params=fixed_params)
                    logging.debug(f"[{trial.number}] {_format_params(params)}")
                    sampled_params_batch.append(params)

                # Build bot_params_batch for each exchange using templates (no deepcopy)
                bot_params_batch_per_exchange = {}
                for exchange, template in templates.items():
                    bot_params_batch_per_exchange[exchange] = [
                        template.build_bot_params_list(params)
                        for params in sampled_params_batch
                    ]

                # 3. Run batch backtests in Rust (parallel via Rayon)
                t_backtest_start = time.perf_counter()
                analyses_batch = _run_batch_backtests(payloads, bot_params_batch_per_exchange)
                t_backtest = time.perf_counter() - t_backtest_start

                # 4. Tell: Report results for each trial
                for trial, trial_analyses in zip(trials, analyses_batch):
                    scenario_metrics = build_scenario_metrics(trial_analyses)
                    flat_stats = flatten_metric_stats(scenario_metrics.get("stats", {}))

                    resolved = {c.metric: resolve_metric(c.metric, flat_stats) for c in constraints}
                    violations = check_constraints(resolved, constraints)
                    trial.set_user_attr("constraint_violations", violations)

                    scores = compute_scores(flat_stats, objectives)
                    study.tell(trial.number, scores)

                trials_completed += batch_size

                # Log generation progress
                logging.info(f"Gen {gen+1}/{n_generations}: {trials_completed}/{n_trials} trials | backtest {t_backtest:.1f}s")

        except KeyboardInterrupt:
            logging.info("Interrupted - stopping optimization...")

    finally:
        # Extract Pareto and log summary
        try:
            if study.trials:
                _log_optimization_summary(study, objectives, n_trials, start_time)
                logging.info("Extracting Pareto front...")
                extract_pareto(study, study_dir, objectives, config, max_best_trials)
            else:
                logging.warning("No trials completed, skipping Pareto extraction")
        except Exception as e:
            logging.warning(f"Could not extract Pareto front: {e}")

        # Dump to SQLite for resume and optuna-dashboard
        sqlite_path = study_dir / "study.db"
        logging.info(f"Saving study to {sqlite_path}...")
        dump_to_sqlite(backend, study.study_name, sqlite_path)


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

    # Create in-memory backend (single process, no IPC needed)
    backend = InMemoryJournalBackend()
    storage = JournalStorage(backend)

    sampler_cfg = optuna_cfg.sampler
    if sampler_name:
        sampler_cfg = get_sampler_config_by_name(sampler_name)

    # Always use hard constraints
    constraints_func = make_constraints_func()

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
    logging.info(f"Sampler: {sampler_cfg.name}, Constraints: hard")

    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params,
        optuna_cfg.max_best_trials, debug_level, backend
    )


async def _load_all_data(config: dict):
    """Load OHLCV data and prepare backtest payloads (no shared memory needed)."""
    payloads = {}
    mss_dict = {}

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

        # Set coins in config BEFORE build_backtest_payload (which calls prep_backtest_args)
        config.setdefault("backtest", {}).setdefault("coins", {})[exchange] = coins

        # Use build_backtest_payload to create properly structured bundle with metadata
        payload = build_backtest_payload(hlcvs, mss, config, exchange, btc_usd, timestamps)
        payloads[exchange] = payload
        mss_dict[exchange] = mss
        logging.debug(f"{exchange}: Done")

    return payloads, mss_dict


def _run_batch_backtests(
    payloads: dict,
    bot_params_batch_per_exchange: dict[str, list[list[dict]]],
) -> list[dict[str, dict]]:
    """Run batch of trials across all exchanges using Rust Rayon parallelism.

    Args:
        payloads: Dict of exchange -> BacktestPayload
        bot_params_batch_per_exchange: Dict of exchange -> list of bot_params_list per trial

    Returns:
        List of analyses dicts (one per trial), where each dict is keyed by exchange.
    """
    # Get batch size from first exchange
    first_exchange = next(iter(bot_params_batch_per_exchange))
    batch_size = len(bot_params_batch_per_exchange[first_exchange])
    analyses_batch = [{} for _ in range(batch_size)]

    for exchange, bot_params_batch in bot_params_batch_per_exchange.items():
        payload = payloads[exchange]

        # Call Rust batch API (runs trials in parallel via Rayon)
        analyses = pbr.run_backtest_batch(
            payload.bundle,           # HlcvsBundle with proper metadata
            bot_params_batch,         # List of bot_params per trial
            payload.exchange_params,  # Shared across trials
            payload.backtest_params,  # Shared across trials
        )

        # Assign results to each trial's dict
        for i, analysis in enumerate(analyses):
            analyses_batch[i][exchange] = analysis

    return analyses_batch


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

    sqlite_path = study_dir / "study.db"
    if not sqlite_path.exists():
        raise ValueError(f"No study.db found in {study_dir}")

    config = load_config(str(config_path), live_only=False, verbose=False)
    optuna_cfg = extract_optuna_config(config)
    bounds = extract_bounds(config)
    constraints = extract_constraints(config)
    objectives = extract_objectives(config)

    if not objectives:
        raise ValueError("No objectives defined in config['optimize']['objectives']")

    n_trials = n_trials or optuna_cfg.n_trials
    n_cpus = n_cpus or optuna_cfg.n_cpus

    # Recreate sampler from config
    sampler_cfg = optuna_cfg.sampler
    # Always use hard constraints
    constraints_func = make_constraints_func()
    sampler = create_sampler(sampler_cfg, constraints_func)

    # Load existing study from SQLite into memory
    shared_backend, study = load_from_sqlite(sqlite_path, study_dir.name, sampler)
    existing_trials = len(study.trials)

    # Restore fine-tune mode from study metadata
    fixed_params = None
    tune_params = study.user_attrs.get("fine_tune_params")
    if tune_params:
        fixed_params = _derive_fixed_params(tune_params, bounds, config)
        logging.info(f"Restored fine-tune mode: tuning {tune_params}")

    logging.info(f"Resuming study: {study_dir}")
    logging.info(f"Existing trials: {existing_trials}, Adding: {n_trials}")
    logging.info(f"Sampler: {sampler_cfg.name}, Constraints: hard")

    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params,
        optuna_cfg.max_best_trials, debug_level, backend
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
    sampler: str | None = typer.Option(None, "--sampler", help="Sampler: nsgaii/nsgaiii (new only)"),
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
