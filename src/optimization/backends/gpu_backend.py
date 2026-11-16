"""CPU ask/tell search over authoritative asynchronous GPU backtests.

Search retains cohort ask/tell updates, with optional CPU scenario promotion.
Requests are replenished and full candidate records are persisted independently.
The problem has no CPU evaluation path, including seed bootstrap and resumption.
"""

from collections import deque
from contextlib import ExitStack
import logging
import math
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

from config.gpu import (
    GPU_TUNING_DEFAULT,
    resolve_gpu_screening,
    resolve_gpu_search_cohort_size,
    resolve_gpu_sizing,
    resolve_gpu_tuning_mode,
)
from optimization.backend_shared import load_starting_individuals
from optimization.backends.pymoo_backend import (
    _build_algorithm, _build_random_sampling, _prepare_resumed_algorithm,
    _reduce_starting_population, _resolve_pymoo_population_plan,
)
from optimization.callback import build_pymoo_record_entry
from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
from optimization.interrupts import OptimizerInterruptRequested, no_interrupt_requested
from optimization.native_checkpoint import CHECKPOINT_VERSION, checkpoint_config, load_checkpoint, save_checkpoint
from optimization.native_datasets import NativeDatasetRegistry
from optimization.native_pipeline import ResultCadence
from optimization.native_session import NativeEvaluationSession
from optimization.progress import CohortDiagnostics, OptimizerProgress, search_boundary
from optimization.scenario_screening import screening_survivor_indices, screening_survivor_count
from optimization.fine_tune_anchors import ANCHOR_GENE_KEY, get_anchor_plan
from optimization.gpu.model import CUDA_MAX_BATCH_CANDIDATES
from optimization.gpu.search_budget import CudaMaxSearchBudget
from optimization.gpu.search_sampling import max_starting_vectors


def _open_preparation_workers(planner, lifetime, *, device, available_rows, interrupt_check):
    from optimization.native_planner_capacity import cuda_plan_worker_count, mps_plan_worker_count

    interrupt_check()
    worker_count = cuda_plan_worker_count if device == "cuda" else mps_plan_worker_count
    workers = worker_count(planner, admission_slots=available_rows, available_rows=available_rows)
    if not workers:
        return None
    from optimization.native_planner_workers import OrderedCandidateWorkers

    pool = lifetime.enter_context(OrderedCandidateWorkers(
        planner, workers=workers, max_pending=min(2 * workers, available_rows),
        interrupt_check=interrupt_check,
    ))
    logging.info("%s max CPU preparation | workers=%d", "CUDA" if device == "cuda" else "Metal", workers)
    return pool


class NativeSearchProblem(Problem):
    def __init__(self, base):
        super().__init__(n_var=len(base.bounds), n_obj=len(base.scoring_specs),
                         n_ieq_constr=1,  # Metric availability always constrains feasibility.
                         xl=np.asarray([bound.low for bound in base.bounds]),
                         xu=np.asarray([bound.high for bound in base.bounds]))

    def _evaluate(self, *_args, **_kwargs):
        raise RuntimeError("GPU native search must use GPU completions through ask/tell")


class _Search:
    def __init__(self, *, state, session, registry, recorder, template,
                 build_config_fn, overrides_fn, overrides_list, checkpoint_path,
                 checkpoint_interval, interrupt_check, screening_policy,
                 preparation_lifetime=None, preparation_device=None):
        self.state, self.session, self.registry = state, session, registry
        self.recorder, self.template = recorder, template
        self.build_config_fn, self.overrides_fn = build_config_fn, overrides_fn
        self.overrides_list = overrides_list
        self.checkpoint_path, self.interval = checkpoint_path, checkpoint_interval
        self.interrupt_check = interrupt_check
        self._last_checkpoint = 0.0
        self._pending = {}
        # Ephemeral candidate-owned CPU configs never enter checkpoints, result
        # caches or the service. Admission bounds their lifetime and count.
        self._record_configs = {}
        from optimize import individual_to_config
        self._reuse_record_configs = build_config_fn is individual_to_config
        self._cadence = ResultCadence()
        self.screening_policy = screening_policy
        self._preparation_lifetime = preparation_lifetime
        self._preparation_device = preparation_device
        self._preparation_workers = None
        self._largest_preparation_window = 0
        self._progress = OptimizerProgress(lambda: dict(
            gen=self.state["algorithm"].n_iter or 0,
            completed=self.state["completed"], screened=self.state["screened"],
            prefilter_rejected=self.state.get("prefilter_rejected", 0),
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
            prepared = self._record_configs.pop(completion.candidate_id, None)
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
            self.apply_terminal(index, payload, prepared=prepared)

    def apply_terminal(self, index, payload, *, prepared=None):
        population = self.state["population"]
        problem = self.state["algorithm"].problem
        objectives = np.asarray(payload["fitness"], dtype=np.float64)
        penalty = float(payload["constraint_violation"])
        if (objectives.shape != (problem.n_obj,) or np.isnan(objectives).any()
                or not math.isfinite(penalty) or penalty < 0):
            raise RuntimeError("invalid terminal optimizer evaluation")
        individual = population[index]
        individual.X = np.asarray(payload["evaluation_vector"], dtype=np.float64)
        rejected = payload["metrics"].get("evaluation_origin") == "config_prefilter"
        # Cached duplicate completions own their evaluation vector. Rebuild if
        # it differs rather than recording a config from another canonical row.
        record_config = (prepared[1] if self._reuse_record_configs and prepared is not None
                         and tuple(payload["evaluation_vector"]) == prepared[0] else None)
        self.recorder.record(build_pymoo_record_entry(
            vector=payload["evaluation_vector"], metrics=payload["metrics"],
            template=self.template, build_config_fn=self.build_config_fn,
            overrides_fn=self.overrides_fn, overrides_list=self.overrides_list,
            prepared_config=record_config,
        ))
        individual.F = objectives
        if problem.n_ieq_constr:
            individual.G = np.asarray([penalty if penalty > 0 else -1.0])
        individual.evaluated.update(("F", "G", "H"))
        individual.prefilter_rejected = rejected
        if rejected:
            self.state["prefilter_rejected"] = self.state.get("prefilter_rejected", 0) + 1
        self.state["completed"] += 1
        self.state["algorithm"].evaluator.n_eval += 1
        self.checkpoint()

    def evaluate_population(self):
        screening = self.state["phase"] == "screening"
        datasets = frozenset(binding.dataset_id for binding in self.registry.bindings
                             if not screening or binding.scenario in self.screening_policy["scenarios"])
        diagnostics = CohortDiagnostics(
            generation=self.state["algorithm"].n_iter or 0, phase=self.state["phase"],
            counts=lambda: dict(completed=self.state["completed"], screened=self.state["screened"],
                                prefilter_rejected=self.state.get("prefilter_rejected", 0)),
            session=self.session,
        )
        status = "interrupted"
        try:
            self._evaluate_population(datasets, diagnostics)
            status = "complete"
        except Exception:
            status = "failed"
            raise
        finally:
            try:
                self.session.set_preparing(())
            finally:
                diagnostics.finish(status)

    def _workers_for_window(self, pending_count):
        if self._preparation_lifetime is None:
            return None
        # Speculative and admitted candidates share this phase's window.
        # Recheck live headroom against the new width before a pool can reuse it.
        available_rows = min(self.session.max_candidates, pending_count)
        device = self._preparation_device
        workers = self._preparation_workers
        if workers is not None:
            from optimization import native_planner_capacity as capacity

            if workers.pending_count:
                raise RuntimeError("GPU optimizer CPU preparation left pending rows across cohorts")
            if device == "cuda":
                estimate_memory, pool_fits = capacity.cuda_plan_preparation_memory, capacity.cuda_plan_pool_fits
            else:
                estimate_memory, pool_fits = capacity.mps_plan_preparation_memory, capacity.mps_plan_pool_fits
            parent_bytes, _ = estimate_memory(self.registry.planner, available_rows)
            if not pool_fits(parent_bytes=parent_bytes, workers=workers.worker_count):
                self._preparation_lifetime.close()
                self._preparation_workers = None
                self._preparation_lifetime = None
                logging.info("%s max CPU preparation | retiring workers under host pressure, continuing serially",
                             "CUDA" if device == "cuda" else "Metal")
                return None
        if available_rows < 2:
            return None
        if self._preparation_workers is None and available_rows > self._largest_preparation_window:
            self._largest_preparation_window = available_rows
            self._preparation_workers = _open_preparation_workers(
                self.registry.planner, self._preparation_lifetime,
                device=device, available_rows=available_rows, interrupt_check=self.interrupt_check,
            )
        workers = self._preparation_workers
        return workers if workers is not None and available_rows >= workers.worker_count else None

    def _evaluate_population(self, datasets, diagnostics):
        population = self.state["population"]
        screening = self.state["phase"] == "screening"
        waiting = (index for index, individual in enumerate(population)
                   if not {"F", "G", "H"} <= individual.evaluated
                   and (not screening or not hasattr(individual, "screening_payload")))
        self.checkpoint(force=True)
        self._progress.transition(self.state["phase"])
        workers = None
        if self._preparation_lifetime is not None:
            indices = list(waiting)
            workers = self._workers_for_window(len(indices))
            waiting = iter(indices)
        exhausted = False
        initial = True
        stages = self.screening_policy["scenarios"] if screening else None
        prepared_indices = deque()
        input_exhausted = False
        speculative_sequence = self.state["sequence"]
        while self._pending or not exhausted:
            self.interrupt_check()
            self._progress.report()
            self.session.set_preparing(
                datasets if not exhausted and len(self._pending) < self.session.max_candidates else (),
            )
            # Submit after the first prepared candidate. Then alternate
            # bounded CPU preparation with scoring/persistence instead of filling
            # the entire admission window before pumping or consuming the service.
            # The service may accumulate while this producer can make progress.
            started = time.perf_counter()
            timeout = (0.0 if not exhausted and len(self._pending) < self.session.max_candidates
                       else self._cadence.budget_seconds)
            while self._pending:
                polled = time.perf_counter()
                with diagnostics.measure("poll"):
                    completions = self.session.poll(timeout=timeout, max_completions=self._cadence.limit)
                consumed = time.perf_counter()
                with diagnostics.measure("consume"):
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
            self.session.set_preparing(
                datasets if not exhausted and len(self._pending) < self.session.max_candidates else (),
            )
            started = time.perf_counter()
            while not exhausted and len(self._pending) < self.session.max_candidates:
                self.interrupt_check()
                candidate_id = f"candidate:{self.state['sequence']}"
                if workers is None:
                    index = next(waiting, None)
                    if index is None:
                        exhausted = True
                        break
                    self.state["sequence"] += 1
                    with diagnostics.measure("prepare"):
                        plan = self.registry.planner.prepare(candidate_id, population[index].X, scenarios=stages)
                else:
                    while (not input_exhausted and workers.pending_count < workers.max_pending
                           and len(self._pending) + workers.pending_count < self.session.max_candidates):
                        self.interrupt_check()
                        index = next(waiting, None)
                        if index is None:
                            input_exhausted = True
                            break
                        with diagnostics.measure("prepare"):
                            workers.submit(f"candidate:{speculative_sequence}", population[index].X, scenarios=stages)
                        speculative_sequence += 1
                        prepared_indices.append(index)
                    if not prepared_indices:
                        exhausted = True
                        break
                    with diagnostics.measure("prepare"):
                        plan = workers.poll(timeout=0.002 if self._pending else 0.02,
                                            interrupt_check=self.interrupt_check)
                    if plan is None:
                        with diagnostics.measure("poll"):
                            self.session.submit_ready()
                        break
                    index = prepared_indices.popleft()
                    if plan.candidate_id != candidate_id:
                        raise RuntimeError("Parallel CPU preparation changed native candidate order")
                    self.state["sequence"] += 1
                if plan.prefilter_payload is not None:
                    with diagnostics.measure("consume"):
                        self.apply_terminal(index, plan.prefilter_payload,
                                            prepared=(plan.vector, plan.record_config))
                    if time.perf_counter() - started >= self._cadence.budget_seconds:
                        break
                    continue
                with diagnostics.measure("prepare"):
                    self.session.admit(replace(plan, record_config=None))
                self._pending[candidate_id] = index
                if plan.record_config is not None and self._reuse_record_configs:
                    self._record_configs[candidate_id] = (plan.vector, plan.record_config)
                if initial or time.perf_counter() - started >= self._cadence.budget_seconds:
                    initial = False
                    break
            if exhausted or len(self._pending) >= self.session.max_candidates:
                # Publish already-prepared rows before sealing. Otherwise the
                # worker can dispatch an earlier partial batch while this final
                # CPU preparation burst is still queued only in the session.
                with diagnostics.measure("poll"):
                    self.session.submit_ready()
                # Further preparation needs a completion, including all-alias
                # or partially cached windows. No future work is promised here.
                self.session.set_preparing(())

    def promote_screening(self):
        population = self.state["population"]
        rejected = population[[index for index, individual in enumerate(population)
                               if getattr(individual, "prefilter_rejected", False)]]
        partial = population[[index for index, individual in enumerate(population)
                              if not getattr(individual, "prefilter_rejected", False)]]
        scores = [individual.screening_payload for individual in partial]
        count = screening_survivor_count(len(partial), self.screening_policy) if len(partial) else 0
        survivors = screening_survivor_indices(
            np.asarray([score["fitness"] for score in scores]),
            np.asarray([score["constraint_violation"] for score in scores]), count=count,
        ) if count else []
        promoted = partial[survivors]
        for individual in promoted:
            del individual.screening_payload
        # Rejected partial observations never enter evolutionary survival. The
        # existing complete parent population remains available to pymoo.
        self.state.update(phase="generation", population=Population.merge(rejected, promoted))
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
    from optimization.gpu.native import NativeBacktestService, max_candidate_widths

    base = getattr(evaluator_for_pool, "base", evaluator_for_pool)
    problem = NativeSearchProblem(base)
    population_plan = _resolve_pymoo_population_plan(config, n_obj=problem.n_obj)
    population_size = population_plan["actual_population_size"]
    options = config["optimize"]
    iters = options["iters"]
    seed = options.get("seed")
    gpu = options.get("gpu", {})
    tuning_mode = resolve_gpu_tuning_mode(gpu.get("tuning_mode", GPU_TUNING_DEFAULT))
    max_throughput = tuning_mode == "max"
    cohort_limit = resolve_gpu_search_cohort_size(gpu.get("search_cohort_size"), tuning_mode=tuning_mode)
    if max_throughput:
        iters = CudaMaxSearchBudget.remaining(iters=iters, allocated=0)
    # Max budgets candidate proposals directly. This is an upper bound for
    # pymoo's bookkeeping generations, including initialization from evaluated parents.
    ngen = iters + 1 if max_throughput else max(1, int(iters / population_size))
    termination = get_termination("n_gen", ngen)
    screening_policy = resolve_gpu_screening(gpu.get("screening"))
    if screening_policy["scenarios"] and not callable(getattr(evaluator_for_pool, "score_scenario_results", None)):
        raise ValueError("GPU screening.scenarios requires a prepared scenario suite")
    batch_size = resolve_gpu_sizing(gpu.get("batch_size"), field="batch_size")
    input_memory_budget = getattr(evaluator_for_pool, "_gpu_input_memory_budget", None)
    device = None
    initial_width = None
    candidate_limit = 1024
    if max_throughput:
        from optimization.gpu.residency import CudaInputMemoryBudget, MpsInputMemoryBudget
        from optimization.gpu.runtime import gpu_device

        if isinstance(input_memory_budget, CudaInputMemoryBudget):
            device = "cuda"
        elif isinstance(input_memory_budget, MpsInputMemoryBudget):
            device = "mps"
        else:
            device = gpu_device()
        if device == "cuda":
            candidate_limit = CUDA_MAX_BATCH_CANDIDATES
        else:
            initial_width, candidate_limit = max_candidate_widths(device)
        device_label = "CUDA" if device == "cuda" else "Metal"
    dispatch_budget = resolve_gpu_sizing(
        gpu.get("max_dispatch_candidate_bars"), field="max_dispatch_candidate_bars",
    )
    if dispatch_budget is None:
        dispatch_budget = 500_000_000
    checkpoint_interval = float(gpu.get("checkpoint_interval_seconds", 5.0))
    if not math.isfinite(checkpoint_interval) or checkpoint_interval < 0:
        raise ValueError("GPU checkpoint interval must be finite and nonnegative")

    def build_initial_state(startup_limit):
        starts = load_starting_individuals(
            starting_configs_path=starting_configs_path, population_size=population_size,
            get_starting_configs=get_starting_configs, configs_to_individuals=configs_to_individuals,
            iter_starting_configs=iter_starting_configs,
            configs_to_individuals_streaming=configs_to_individuals_streaming,
            optimization_shape=optimization_shape, bounds=base.bounds, sig_digits=base.sig_digits,
        )
        authored_seed_count = len(starts)
        if max_throughput:
            width = batch_size
            if width is None:
                width = max_candidate_widths(device)[0] if device == "cuda" else initial_width
            target_width = min(width, startup_limit)
            plan = CudaMaxSearchBudget(population_size, target_width, cohort_limit)
            generated = plan.bootstrap_size(iters=iters, authored_seed_count=authored_seed_count)
            key_paths = getattr(optimization_shape, "key_paths", ())
            anchors = tuple(i for i, (key, _path) in enumerate(key_paths) if key == ANCHOR_GENE_KEY)
            starts = max_starting_vectors(
                starts, bounds=base.bounds, sig_digits=base.sig_digits,
                width=authored_seed_count + generated, random_state=np.random.default_rng(seed),
                preserve_indices=anchors,
            )
        sampling = (np.asarray(starts[:population_size]) if max_throughput
                    else _build_random_sampling(base.bounds, population_size))
        algorithm = _build_algorithm(config=config, sampling=sampling,
                                     bounds=base.bounds, sig_digits=base.sig_digits, population_plan=population_plan,
                                     bounded_duplicates=max_throughput)
        from session_artifacts import record_optimizer_search
        max_facts = (
            {"bootstrapped_individuals": len(starts), "initial_cohort_target": plan.cohort_width,
             "generated_attempt_budget": iters} if max_throughput else None
        )
        record_optimizer_search(
            config, algorithm, population_plan, starting_individuals=authored_seed_count,
            cuda_max_search=max_facts if device == "cuda" else None,
            metal_max_search=max_facts if device == "mps" else None,
        )
        algorithm.setup(problem, termination=termination, seed=seed, verbose=False)
        state = dict(backend="gpu", version=CHECKPOINT_VERSION, algorithm=algorithm,
                     phase="seeds" if starts else "idle", sequence=0, completed=0,
                     screened=0, population=Population.new("X", np.asarray(starts)) if starts else None)
        if max_throughput:
            state.update(max_search_budget=plan, max_search_allocated=len(starts) - authored_seed_count)
            logging.info(
                "%s max search | survivors=%d cohort_target=%d bootstrap_seeds=%d "
                "bootstrap_generated=%d proposal_budget=%d",
                device_label, population_size, plan.cohort_width, authored_seed_count,
                state["max_search_allocated"], iters,
            )
        state[CONTRACT_KEY] = build_evaluation_contract(config)
        state["resume_config"] = checkpoint_config(config, state[CONTRACT_KEY])
        state["anchor_plan"] = deepcopy(get_anchor_plan(config))
        return state

    if resume:
        state = load_checkpoint(checkpoint_path, config)
        _prepare_resumed_algorithm(state["algorithm"], problem=problem,
                                   termination=termination, seed=seed, callback=Callback())
    elif device != "cuda":
        # CUDA max uses registry memory estimates before it sizes its seed cohort.
        state = build_initial_state(candidate_limit)

    with NativeDatasetRegistry(evaluator_for_pool, standalone_candle_coins=standalone_candle_coins,
                               overrides_list=overrides_list) as registry:
        startup_limit = candidate_limit
        if device == "cuda" and batch_size is None:
            from optimization.native_planner_capacity import cuda_plan_parallel_window

            parallel_window = cuda_plan_parallel_window(registry.planner, candidate_limit)
            if parallel_window is not None and parallel_window < candidate_limit:
                startup_limit = parallel_window
                logging.info(
                    "CUDA max host admission | startup_limit=%d serial_candidate_s=%.3f",
                    startup_limit, registry.serial_preparation_seconds,
                )
        if not resume and device == "cuda":
            state = build_initial_state(startup_limit)
        max_search_budget = state.get("max_search_budget")
        labels = set(screening_policy["scenarios"])
        if labels:
            unknown = labels - registry.scorer.coverage.keys()
            if unknown:
                raise ValueError(f"GPU screening.scenarios contains unknown labels: {sorted(unknown)}")
            registry.scorer.validate_coverage(
                [(binding.scenario, binding.dataset.exchange) for binding in registry.bindings
                 if binding.scenario in labels], "screening",
            )
        service_options = ({"input_memory_budget": input_memory_budget}
                           if input_memory_budget is not None else {})
        input_owner = getattr(base, "shared_array_manager", None)
        if getattr(input_owner, "file_backed", False):
            service_options["on_inputs_prepared"] = input_owner.discard_file_pages
        # Seed survival waits for the complete seed cohort, so its available
        # work is independent of the later evolutionary population size. Keep
        # admission bounded while allowing the service to fill wider batches.
        cohort_width = max_search_budget.cohort_width if max_search_budget is not None else population_size
        population_window = max(2, min(cohort_width, startup_limit))
        pending_count = len(state["population"]) if (
            state["phase"] == "seeds" or (max_throughput and state["phase"] != "idle")
        ) else 0
        candidate_window = max(population_window, min(pending_count, startup_limit))
        if device == "mps":
            service_options["max_pending"] = candidate_limit
        if max_throughput and batch_size is None:
            service_options["max_cohort_width"] = min(max_search_budget.target_width, startup_limit)
            if startup_limit < candidate_limit:
                service_options["host_preparation"] = (startup_limit, registry.serial_preparation_seconds)
        service = NativeBacktestService(
            batch_size=batch_size, **service_options,
            max_dispatch_candidate_bars=dispatch_budget,
            interrupt_check=interrupt_check, tuning_mode=tuning_mode,
            auto_request_capacity=(candidate_limit if max_throughput else candidate_window)
                                  * registry.requests_per_candidate,
        )
        session = NativeEvaluationSession(
            service, registry.scorer,
            max_candidates=candidate_window,
        )
        worker_lifetime = ExitStack()
        search = _Search(state=state, session=session, registry=registry, recorder=recorder,
                         template=base.config, build_config_fn=build_config_fn, overrides_fn=overrides_fn,
                         overrides_list=overrides_list, checkpoint_path=checkpoint_path,
                         checkpoint_interval=checkpoint_interval, interrupt_check=interrupt_check,
                         screening_policy=screening_policy,
                         preparation_device=device,
                         preparation_lifetime=worker_lifetime if max_throughput else None)

        def remember_accepted_width():
            if not max_throughput:
                return
            width = service.accepted_cohort_width
            budget = state["max_search_budget"]
            if width is not None and width != budget.target_width:
                state["max_search_budget"] = replace(budget, target_width=width)
                logging.info("%s max cohort target | width=%d", device_label, state["max_search_budget"].cohort_width)

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
                    pad_sampling=not max_throughput,
                )
                session.max_candidates = population_window
                remember_accepted_width()
                state.update(phase="idle", population=None)
                search.checkpoint(force=True)
            logging.info("Starting GPU native optimization...")
            # n_iter is the next generation after tell. A freshly configured
            # resume termination has no progress yet; also check that next index
            # so a completed checkpoint does not run an extra generation.
            algorithm = state["algorithm"]
            while state["phase"] in {"screening", "generation"} or (
                (algorithm.n_iter or 1) <= ngen and algorithm.has_next()
            ):
                interrupt_check()
                if state["phase"] == "idle":
                    if max_throughput and algorithm.is_initialized:
                        target_width = service.cohort_width
                        budget = state["max_search_budget"]
                        # A live trial can use a wider cohort without becoming
                        # the baseline when the process restarts mid-trial.
                        if target_width is not None and target_width != budget.target_width:
                            budget = replace(budget, target_width=target_width)
                        slots = budget.offspring_size(iters=iters, allocated=state["max_search_allocated"])
                        if not slots:
                            break
                        algorithm.n_offsprings = slots
                        session.max_candidates = max(2, min(slots, candidate_limit))
                    else:
                        slots = 0
                    with search_boundary(algorithm.n_iter or 0, "ask"):
                        population = algorithm.ask()
                    if population is None and algorithm.termination.force_termination:
                        break
                    if population is None or not len(population):
                        raise RuntimeError("GPU native evolutionary algorithm returned no candidates")
                    if max_throughput:
                        state["max_search_allocated"] += slots
                    parents = algorithm.pop
                    screening = bool(labels) and parents is not None and len(parents) > 0
                    # All-label selection is already a full evaluation, with no
                    # partial search stage or survivor reduction.
                    screening = screening and labels != registry.scorer.coverage.keys()
                    screening = screening and screening_survivor_count(len(population), screening_policy) < len(population)
                    state.update(phase="screening" if screening else "generation", population=population)
                if state["phase"] == "screening":
                    search.evaluate_population()
                    remember_accepted_width()
                    with search_boundary(algorithm.n_iter or 0, "promotion"):
                        search.promote_screening()
                search.evaluate_population()
                with search_boundary(algorithm.n_iter or 0, "tell"):
                    algorithm.tell(infills=state["population"])
                if max_throughput:
                    # pymoo keeps the last offspring independently of survivors.
                    # It is not consumed by future mating, so avoid retaining a
                    # full wide cohort in every subsequent checkpoint.
                    algorithm.off = None
                    remember_accepted_width()
                state.update(phase="idle", population=None)
                search.checkpoint(force=True)
            worker_lifetime.close()
            service.close()
            search.checkpoint(force=True)
        except BaseException as failure:
            try:
                session.stop_admission()
            except BaseException:
                logging.exception("GPU native admission cleanup failed after an earlier failure")
            try:
                worker_lifetime.close()
            except BaseException:
                logging.exception("CPU preparation worker cleanup failed after an earlier failure")
            shutdown_failure = None
            try:
                service.close(cancel_pending=True)
            except BaseException as error:
                if isinstance(error, Exception):
                    shutdown_failure = error
                logging.exception("GPU native service close failed after an earlier failure")
            try:
                search.drain_after_stop()
            except BaseException as error:
                if shutdown_failure is None and isinstance(error, Exception):
                    shutdown_failure = error
                logging.exception("GPU native drain failed after an earlier failure")
            try:
                remember_accepted_width()
                search.checkpoint(force=True)
            except BaseException:
                logging.exception("GPU native checkpoint failed after an earlier failure")
            if isinstance(failure, OptimizerInterruptRequested) and shutdown_failure is not None:
                raise shutdown_failure
            raise
        finally:
            search._record_configs.clear()
    logging.info("GPU native optimization complete.")
    return {"pool": None, "pool_terminated": False}
