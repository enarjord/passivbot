"""CPU ask/tell search over authoritative asynchronous GPU backtests.

Search retains cohort ask/tell updates, with optional CPU scenario promotion.
Requests are replenished and full candidate records are persisted independently.
The problem has no CPU evaluation path, including seed bootstrap and resumption.
"""

import logging
import math
import time
from copy import deepcopy

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

from config.gpu import resolve_gpu_screening, validate_gpu_backtest_config
from optimization.backend_shared import load_starting_individuals
from optimization.backends.pymoo_backend import (
    _build_algorithm, _build_random_sampling, _prepare_resumed_algorithm,
    _reduce_starting_population, _resolve_pymoo_population_plan,
)
from optimization.callback import build_pymoo_record_entry
from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
from optimization.interrupts import no_interrupt_requested
from optimization.native_checkpoint import CHECKPOINT_VERSION, checkpoint_config, load_checkpoint, save_checkpoint
from optimization.native_datasets import NativeDatasetRegistry
from optimization.native_pipeline import ResultCadence
from optimization.native_session import NativeEvaluationSession
from optimization.progress import OptimizerProgress
from optimization.scenario_screening import screening_survivor_indices, screening_survivor_count
from optimization.fine_tune_anchors import get_anchor_plan


class NativeSearchProblem(Problem):
    def __init__(self, base):
        super().__init__(n_var=len(base.bounds), n_obj=len(base.scoring_specs),
                         n_ieq_constr=int(bool(base.limit_checks)),
                         xl=np.asarray([bound.low for bound in base.bounds]),
                         xu=np.asarray([bound.high for bound in base.bounds]))

    def _evaluate(self, *_args, **_kwargs):
        raise RuntimeError("GPU native search must use GPU completions through ask/tell")


class _Search:
    def __init__(self, *, state, session, registry, recorder, template,
                 build_config_fn, overrides_fn, overrides_list, checkpoint_path,
                 checkpoint_interval, interrupt_check, screening_policy):
        self.state, self.session, self.registry = state, session, registry
        self.recorder, self.template = recorder, template
        self.build_config_fn, self.overrides_fn = build_config_fn, overrides_fn
        self.overrides_list = overrides_list
        self.checkpoint_path, self.interval = checkpoint_path, checkpoint_interval
        self.interrupt_check = interrupt_check
        self._last_checkpoint = 0.0
        self._pending = {}
        self._cadence = ResultCadence()
        self.screening_policy = screening_policy
        self._progress = OptimizerProgress(lambda: dict(
            gen=self.state["algorithm"].n_iter or 0,
            completed=self.state["completed"], screened=self.state["screened"],
            pending=len(self._pending),
        ))

    def checkpoint(self, *, force=False):
        now = time.monotonic()
        if force or now - self._last_checkpoint >= self.interval:
            save_checkpoint(self.checkpoint_path, self.state)
            self._last_checkpoint = now

    def consume(self, completions):
        population = self.state["population"]
        for completion in completions:
            index = self._pending.pop(completion.candidate_id)
            screening = self.state["phase"] == "screening"
            if completion.stage != ("screening" if screening else "full"):
                raise RuntimeError("GPU completion stage does not match its search cohort")
            payload = completion.payload if screening else completion.require_full()
            objectives = np.asarray(payload["fitness"], dtype=np.float64)
            problem = self.state["algorithm"].problem
            if objectives.shape != (problem.n_obj,) or np.isnan(objectives).any():
                raise RuntimeError("GPU candidate objective shape or values are invalid")
            penalty = float(payload["constraint_violation"])
            if not math.isfinite(penalty) or penalty < 0:
                raise RuntimeError("GPU candidate constraint violation is invalid")
            individual = population[index]
            individual.X = np.asarray(payload["evaluation_vector"], dtype=np.float64)
            if screening:
                # Durable CPU selection evidence, never evaluated F/G/H or a
                # complete result record. No device/cache handles are retained.
                individual.screening_payload = dict(fitness=objectives.tolist(), constraint_violation=penalty)
                self.state["screened"] += 1
                self.checkpoint()
                continue
            self.recorder.record(build_pymoo_record_entry(
                vector=payload["evaluation_vector"], metrics=payload["metrics"],
                template=self.template, build_config_fn=self.build_config_fn,
                overrides_fn=self.overrides_fn, overrides_list=self.overrides_list,
            ))
            individual.F = objectives
            if problem.n_ieq_constr:
                individual.G = np.asarray([penalty if penalty > 0 else -1.0])
            individual.evaluated.update(("F", "G", "H"))
            self.state["completed"] += 1
            self.state["algorithm"].evaluator.n_eval += 1
            self.checkpoint()

    def evaluate_population(self):
        population = self.state["population"]
        screening = self.state["phase"] == "screening"
        waiting = iter(index for index, individual in enumerate(population)
                       if (not hasattr(individual, "screening_payload") if screening else
                           not {"F", "G", "H"} <= individual.evaluated))
        exhausted = False
        initial = True
        self.checkpoint(force=True)
        self._progress.transition(self.state["phase"])
        while self._pending or not exhausted:
            self.interrupt_check()
            self._progress.report()
            # Start GPU work after the first prepared candidate. Then alternate
            # bounded CPU preparation with scoring/persistence instead of filling
            # the entire admission window before pumping or consuming the service.
            started = time.perf_counter()
            timeout = (0.0 if not exhausted and len(self._pending) < self.session.max_candidates
                       else self._cadence.budget_seconds)
            while self._pending:
                polled = time.perf_counter()
                completions = self.session.poll(timeout=timeout, max_completions=self._cadence.limit)
                consumed = time.perf_counter()
                self.consume(completions)
                finished = time.perf_counter()
                # Blocking polls may include device idle time. Their record work
                # still counts; nonblocking polls include CPU fan-in/scoring too.
                cost = finished - (polled if timeout == 0 else consumed)
                self._cadence.observe(len(completions), cost)
                timeout = 0.0
                if not completions or finished - started >= self._cadence.budget_seconds:
                    break
                self.interrupt_check()
            started = time.perf_counter()
            while not exhausted and len(self._pending) < self.session.max_candidates:
                self.interrupt_check()
                try:
                    index = next(waiting)
                except StopIteration:
                    exhausted = True
                    break
                candidate_id = f"candidate:{self.state['sequence']}"
                self.state["sequence"] += 1
                plan = self.registry.planner.prepare(
                    candidate_id, population[index].X,
                    scenarios=self.screening_policy["scenarios"] if screening else None,
                )
                self.session.admit(plan)
                self._pending[candidate_id] = index
                if initial or time.perf_counter() - started >= self._cadence.budget_seconds:
                    initial = False
                    break

    def promote_screening(self):
        population = self.state["population"]
        scores = [individual.screening_payload for individual in population]
        count = screening_survivor_count(len(population), self.screening_policy)
        survivors = screening_survivor_indices(
            np.asarray([score["fitness"] for score in scores]),
            np.asarray([score["constraint_violation"] for score in scores]), count=count,
        )
        promoted = population[survivors]
        for individual in promoted:
            del individual.screening_payload
        # Rejected partial observations never enter evolutionary survival. The
        # existing complete parent population remains available to pymoo.
        self.state.update(phase="generation", population=promoted)
        self.checkpoint(force=True)
        logging.info("GPU scenario screening | candidates=%d full_suite=%d scenarios=%s",
                     len(population), len(promoted), self.screening_policy["scenarios"])

    def drain_after_stop(self):
        # Service has already stopped/drained. Persist full successes which
        # precede a cancelled/failed request; leave partial candidates unevaluated.
        while True:
            completions = self.session.poll()
            self.consume(completions)
            if not completions and not self.session.pending_request_count:
                return


def run_backend(*, config, evaluator_for_pool, recorder, overrides_list,
                starting_configs_path, get_starting_configs, configs_to_individuals,
                build_config_fn, overrides_fn, iter_starting_configs=None,
                configs_to_individuals_streaming=None, optimization_shape=None,
                checkpoint_path=None, resume=False, interrupt_check=no_interrupt_requested,
                standalone_candle_coins=None, **_cpu_only_arguments):
    validate_gpu_backtest_config(config)
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.autotune import is_auto

    base = getattr(evaluator_for_pool, "base", evaluator_for_pool)
    problem = NativeSearchProblem(base)
    population_plan = _resolve_pymoo_population_plan(config, n_obj=problem.n_obj)
    population_size = population_plan["actual_population_size"]
    ngen = max(1, int(config["optimize"]["iters"] / population_size))
    termination = get_termination("n_gen", ngen)
    seed = config["optimize"].get("seed")
    gpu = config["optimize"].get("gpu", {})
    screening_policy = resolve_gpu_screening(gpu.get("screening"))
    if screening_policy["scenarios"] and not callable(getattr(evaluator_for_pool, "score_scenario_results", None)):
        raise ValueError("GPU screening.scenarios requires a prepared scenario suite")
    batch_size = None if is_auto(gpu.get("batch_size")) else gpu["batch_size"]
    dispatch_budget = (500_000_000 if gpu.get("max_dispatch_candidate_bars") is None
                       else gpu["max_dispatch_candidate_bars"])
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("GPU native batch size must be a positive integer")
    checkpoint_interval = float(gpu.get("checkpoint_interval_seconds", 5.0))
    if not math.isfinite(checkpoint_interval) or checkpoint_interval < 0:
        raise ValueError("GPU checkpoint interval must be finite and nonnegative")
    if resume:
        state = load_checkpoint(checkpoint_path, config)
        _prepare_resumed_algorithm(state["algorithm"], problem=problem,
                                   termination=termination, seed=seed, callback=Callback())
    else:
        starts = load_starting_individuals(
            starting_configs_path=starting_configs_path, population_size=population_size,
            get_starting_configs=get_starting_configs, configs_to_individuals=configs_to_individuals,
            iter_starting_configs=iter_starting_configs,
            configs_to_individuals_streaming=configs_to_individuals_streaming,
            optimization_shape=optimization_shape, bounds=base.bounds, sig_digits=base.sig_digits,
        )
        algorithm = _build_algorithm(config=config, sampling=_build_random_sampling(base.bounds, population_size),
                                     bounds=base.bounds, sig_digits=base.sig_digits, population_plan=population_plan)
        algorithm.setup(problem, termination=termination, seed=seed, verbose=False)
        state = dict(backend="gpu", version=CHECKPOINT_VERSION, algorithm=algorithm,
                     phase="seeds" if starts else "idle", sequence=0, completed=0,
                     screened=0, population=Population.new("X", np.asarray(starts)) if starts else None)
        state[CONTRACT_KEY] = build_evaluation_contract(config)
        state["resume_config"] = checkpoint_config(config, state[CONTRACT_KEY])
        state["anchor_plan"] = deepcopy(get_anchor_plan(config))

    with NativeDatasetRegistry(evaluator_for_pool, standalone_candle_coins=standalone_candle_coins,
                               overrides_list=overrides_list) as registry:
        labels = set(screening_policy["scenarios"])
        if labels:
            unknown = labels - registry.scorer.coverage.keys()
            if unknown:
                raise ValueError(f"GPU screening.scenarios contains unknown labels: {sorted(unknown)}")
            registry.scorer.validate_coverage(
                [(binding.scenario, binding.dataset.exchange) for binding in registry.bindings
                 if binding.scenario in labels], "screening",
            )
        service = CudaBacktestService(batch_size=batch_size,
                                     max_dispatch_candidate_bars=dispatch_budget,
                                     interrupt_check=interrupt_check, tuning_mode=gpu.get("tuning_mode", "auto"))
        session = NativeEvaluationSession(service, registry.scorer,
                                          max_candidates=max(2, min(population_size, 1024)))
        search = _Search(state=state, session=session, registry=registry, recorder=recorder,
                         template=base.config, build_config_fn=build_config_fn, overrides_fn=overrides_fn,
                         overrides_list=overrides_list, checkpoint_path=checkpoint_path,
                         checkpoint_interval=checkpoint_interval, interrupt_check=interrupt_check,
                         screening_policy=screening_policy)
        try:
            registry.register(service)
            if state["phase"] == "seeds":
                search.evaluate_population()
                seeds = state["population"]
                state["algorithm"].initialization.sampling = _reduce_starting_population(
                    problem=problem, algorithm=state["algorithm"],
                    starting_individuals=seeds.get("X").tolist(),
                    payloads=[dict(F=individual.F, G=individual.G) for individual in seeds],
                    population_size=population_size, bounds=base.bounds, rng_seed=seed,
                )
                state.update(phase="idle", population=None)
                search.checkpoint(force=True)
            logging.info("Starting GPU native optimization...")
            # n_iter is the next generation after tell. A freshly configured
            # resume termination has no progress yet; also check that next index
            # so a completed checkpoint does not run an extra generation.
            while state["phase"] in {"screening", "generation"} or (
                (state["algorithm"].n_iter or 1) <= ngen and state["algorithm"].has_next()
            ):
                interrupt_check()
                if state["phase"] == "idle":
                    population = state["algorithm"].ask()
                    if population is None and state["algorithm"].termination.force_termination:
                        break
                    if population is None or not len(population):
                        raise RuntimeError("GPU native evolutionary algorithm returned no candidates")
                    parents = state["algorithm"].pop
                    screening = bool(labels) and parents is not None and len(parents) > 0
                    # All-label selection is already a full evaluation, with no
                    # partial search stage or survivor reduction.
                    screening = screening and labels != registry.scorer.coverage.keys()
                    screening = screening and screening_survivor_count(len(population), screening_policy) < len(population)
                    state.update(phase="screening" if screening else "generation", population=population)
                if state["phase"] == "screening":
                    search.evaluate_population()
                    search.promote_screening()
                search.evaluate_population()
                state["algorithm"].tell(infills=state["population"])
                state.update(phase="idle", population=None)
                search.checkpoint(force=True)
            service.close()
            search.checkpoint(force=True)
        except BaseException:
            session.stop_admission()
            try:
                service.close(cancel_pending=True)
                search.drain_after_stop()
            except BaseException:
                logging.exception("GPU native drain failed after an earlier failure")
            try:
                search.checkpoint(force=True)
            except BaseException:
                logging.exception("GPU native checkpoint failed after an earlier failure")
            raise
    logging.info("GPU native optimization complete.")
    return {"pool": None, "pool_terminated": False}
