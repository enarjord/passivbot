"""Bounded, ordered CPU preparation using the canonical native planner."""

from collections import deque
from dataclasses import dataclass, replace
import logging
import math
import multiprocessing
import os
import pickle
from queue import Empty, Full, Queue
import signal
import sys
from threading import Event, Thread
import time
import traceback
from types import MappingProxyType


_STARTUP_TIMEOUT_SECONDS = 60.0
_WAIT_INTERVAL_SECONDS = 0.02
_WORKER_ENVIRONMENT = {
    "SKIP_RUST_COMPILE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class CandidatePreparationQueueFull(Full):
    """The pending window is full, including completed plans awaiting delivery."""


class CandidatePreparationError(RuntimeError):
    """The original worker initializer or candidate-preparation error."""

    def __init__(self, candidate_id, error):
        self.candidate_id = candidate_id
        self.error_type, message, self.remote_traceback = error
        operation = (
            "CPU planner startup failed" if candidate_id is None
            else f"CPU preparation failed for {candidate_id}"
        )
        super().__init__(f"{operation}: {self.error_type}: {message}")


@dataclass(frozen=True)
class _Scenario:
    label: str
    config: dict
    exchanges: tuple[str, ...]
    msss: dict
    overrides: dict


@dataclass(frozen=True)
class _Snapshot:
    config: dict
    exchanges: tuple[str, ...]
    msss: dict
    scenarios: tuple[_Scenario, ...] | None
    reducer: dict | None
    bindings: tuple
    overrides: tuple
    rust_source_stamp: str
    rust_artifact_sha256: str

    @classmethod
    def capture(cls, planner):
        source, artifact = _runtime_identity()
        evaluator = planner.scorer.evaluator
        base = planner.base
        contexts = tuple(_Scenario(ctx.label, ctx.config, tuple(ctx.exchanges), ctx.msss, ctx.overrides)
                         for ctx in evaluator.contexts) if planner.scorer.suite else None
        return cls(base.config, tuple(base.exchanges), base.msss, contexts,
                   evaluator.reducer_cfg if contexts is not None else None,
                   planner.bindings, planner.overrides, source, artifact)

    def build_planner(self):
        from optimization.native_planning import NativeCandidatePlanner
        from optimize import Evaluator, SuiteEvaluator

        if _runtime_identity(fingerprint=self.rust_source_stamp) != (self.rust_source_stamp, self.rust_artifact_sha256):
            raise RuntimeError("Planner worker's Rust extension differs from the optimizer's loaded extension")
        # The keys identify scoring exchanges. Workers own no evaluator arrays.
        base = Evaluator(dict.fromkeys(self.exchanges), {}, self.msss, self.config)
        evaluator = SuiteEvaluator(base, list(self.scenarios), self.reducer) if self.scenarios is not None else base
        return NativeCandidatePlanner(evaluator, self.bindings, overrides_list=self.overrides)


def _runtime_identity(*, fingerprint=None):
    from rust_utils import verify_loaded_runtime_extension

    verified = verify_loaded_runtime_extension(fingerprint=fingerprint)
    source = verified["runtime_compiled_source_stamp"]
    artifact = verified["runtime_compiled_sha256"]
    if not source or not artifact:
        raise RuntimeError("Parallel CPU preparation requires a verified source-stamped Rust extension")
    return source, artifact


def _plan_parameters(plan, freeze):
    convert = MappingProxyType if freeze else dict
    return replace(plan, requests=tuple(replace(row, parameters=convert(dict(row.parameters))) for row in plan.requests))


def _check_cpu_only():
    blocked = ("torch", "cupy", "optimization.gpu.runtime", "optimization.gpu.service",
               "optimization.gpu.cuda_kernel", "optimization.gpu.mps_kernel")
    if any(module in sys.modules for module in blocked):
        raise RuntimeError("Parallel CPU preparation imported a device runtime")


def _failure(error):
    return type(error).__name__, str(error), traceback.format_exc()


def _process_metrics():
    import psutil

    try:
        rss = psutil.Process().memory_info().rss
    except (OSError, psutil.Error):
        rss = None  # Optional observation cannot fail a canonical plan.
    return time.process_time(), rss


def _prepare_worker(connection):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.pop("CUPY_CACHE_DIR", None)
    with connection:
        try:
            packed_snapshot = connection.recv_bytes()
            planner = pickle.loads(packed_snapshot).build_planner()
            del packed_snapshot
            _check_cpu_only()
            connection.send((True, _process_metrics()))
        except Exception as error:
            connection.send((False, _failure(error)))
            return
        while True:
            try:
                job = connection.recv()
            except EOFError:
                return
            if job is None:
                return
            candidate_id, vector, scenarios = job
            try:
                plan = planner.prepare(candidate_id, vector, scenarios=scenarios)
                _check_cpu_only()
                wire = _plan_parameters(plan, freeze=False)
                payload = pickle.dumps(wire, protocol=pickle.HIGHEST_PROTOCOL)
                connection.send((True, payload, _process_metrics()))
            except Exception as error:
                connection.send((False, _failure(error), _process_metrics()))
                return


@dataclass(eq=False)
class _Row:
    job: tuple
    done: bool = False
    payload: bytes | None = None
    error: tuple | None = None


@dataclass(eq=False)
class _Worker:
    process: object
    connection: object
    tasks: Queue
    thread: Thread | None = None
    row: _Row | None = None
    dead: bool = False
    failure: tuple | None = None


class OrderedCandidateWorkers:
    """Spawned CPU planners with a fixed capacity and submission-order results.

    The parent owns candidate IDs, optimizer state, admission and scoring.
    Construct and use this object on that parent thread inside a context manager.
    A return of None from poll denotes no ready head row; pending_count tells
    whether to wait or advance the cohort. Later ready rows remain buffered.
    """

    def __init__(self, planner, *, workers, max_pending, interrupt_check=None):
        if type(workers) is not int or type(max_pending) is not int or not 1 <= workers <= max_pending:
            raise ValueError("planner worker count and pending capacity must satisfy 1 <= workers <= max_pending")
        self._snapshot = pickle.dumps(_Snapshot.capture(planner), protocol=pickle.HIGHEST_PROTOCOL)
        self._interrupt_check = interrupt_check
        self.worker_count = workers
        self.max_pending = max_pending
        self._workers = []
        self._ordered = deque()
        self._waiting = deque()
        self._ids = set()
        self._events = Queue()
        self._stopping = Event()
        self._started = False
        self._failed = False
        self._ordered_failure = False
        self._metrics = {}

    @property
    def pending_count(self):
        return len(self._ordered)

    @property
    def snapshot_bytes(self):
        return len(self._snapshot)

    def resource_snapshot(self):
        """Return measured per-worker CPU/RSS and the peak serialized plan size."""
        self._collect_events()
        return {pid: dict(stats) for pid, stats in self._metrics.items()}

    def __enter__(self):
        if self._started or self._stopping.is_set():
            raise RuntimeError("parallel CPU preparation cannot be started twice")
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        try:
            self._start_workers(deadline)
            self._await_ready(deadline)
        except BaseException:
            self._close_preserving_error()
            raise
        self._started = True
        return self

    def _start_workers(self, deadline):
        context = multiprocessing.get_context("spawn")
        for _ in range(self.worker_count):
            if self._interrupt_check is not None:
                self._interrupt_check()
            self._startup_remaining(deadline)
            parent, child = context.Pipe()
            process = context.Process(target=_prepare_worker, args=(child,), daemon=True)
            with child:
                try:
                    self._start_process(process)
                except BaseException:
                    parent.close()
                    raise
            worker = _Worker(process, parent, Queue(maxsize=1))
            self._workers.append(worker)
            worker.thread = Thread(target=self._relay, args=(worker,), daemon=True)
            worker.thread.start()

    def _await_ready(self, deadline):
        pending = set(self._workers)
        while True:
            if self._interrupt_check is not None:
                self._interrupt_check()
            remaining = self._startup_remaining(deadline)
            try:
                worker, kind, detail = self._events.get(timeout=min(remaining, _WAIT_INTERVAL_SECONDS))
            except Empty:
                continue
            if kind == "failed":
                raise CandidatePreparationError(None, detail)
            if kind != "ready" or worker not in pending:
                raise RuntimeError("Invalid CPU planner startup acknowledgment")
            self._handle_event(worker, kind, detail)
            pending.remove(worker)
            if not pending:
                return

    @staticmethod
    def _start_process(process):
        # Spawn can import __main__ before invoking the target. Scope the child
        # environment to start(), before that import can load Rust or BLAS.
        previous = {key: os.environ.get(key) for key in _WORKER_ENVIRONMENT}
        try:
            os.environ.update(_WORKER_ENVIRONMENT)
            process.start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _startup_remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("CPU planner workers did not initialize before the startup deadline")
        return remaining

    def _relay(self, worker):
        # Keep the socket's framing and large payload reads outside optimizer
        # polling. The parent still owns all state and chooses delivery order.
        try:
            # Keep potentially large snapshot writes off the startup thread so
            # an idle or failed child cannot block its timeout or interrupt.
            worker.connection.send_bytes(self._snapshot)
            ready, initialization = worker.connection.recv()
            if not ready:
                self._events.put((worker, "failed", initialization))
                return
            self._events.put((worker, "ready", initialization))
            while not self._stopping.is_set():
                job = worker.tasks.get()
                if job is None or self._stopping.is_set():
                    worker.connection.send(None)
                    return
                worker.connection.send(job)
                success, payload, metrics = worker.connection.recv()
                self._events.put((worker, "result", (success, payload, metrics)))
                if not success:
                    return
        except Exception as error:
            if not self._stopping.is_set():
                self._events.put((worker, "failed", _failure(error)))
        finally:
            worker.connection.close()

    def _record_metrics(self, worker, metrics, *, plan_bytes=None):
        cpu, rss = metrics
        stats = self._metrics.setdefault(worker.process.pid, dict(cpu_seconds=0.0, peak_sampled_rss_bytes=None,
                                                                 plans=0, peak_plan_bytes=0))
        stats["cpu_seconds"] = cpu
        if rss is not None:
            stats["peak_sampled_rss_bytes"] = max(stats["peak_sampled_rss_bytes"] or 0, rss)
        if plan_bytes is not None:
            stats["plans"] += 1
            stats["peak_plan_bytes"] = max(stats["peak_plan_bytes"], plan_bytes)

    def _handle_event(self, worker, kind, detail):
        if kind == "ready":
            self._record_metrics(worker, detail)
            return
        row, worker.row = worker.row, None
        if kind == "failed":
            worker.dead = True
            worker.failure = detail
            if row is not None:
                row.done, row.error = True, detail
                self._ordered_failure = True
            return
        success, payload, metrics = detail
        self._record_metrics(worker, metrics, plan_bytes=len(payload) if success else None)
        row.done = True
        if success:
            row.payload = payload
        else:
            worker.dead = True
            worker.failure = payload
            row.error = payload
            self._ordered_failure = True

    def _dispatch(self):
        if self._ordered_failure:
            # Assigned rows form a prefix. No waiting row can precede a fault.
            return
        for worker in self._workers:
            if not self._waiting:
                return
            if not worker.dead and worker.row is None:
                worker.row = self._waiting.popleft()
                worker.tasks.put_nowait(worker.row.job)
        if self._waiting and all(worker.dead for worker in self._workers):
            row = self._waiting.popleft()
            row.done = True
            row.error = self._workers[0].failure

    def _collect_events(self):
        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            self._handle_event(*event)

    def _require_running(self):
        if not self._started or self._stopping.is_set() or self._failed:
            raise RuntimeError("parallel CPU preparation is unavailable")

    def submit(self, candidate_id, vector, *, scenarios=None):
        self._require_running()
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in self._ids:
            raise ValueError("candidate ID must be nonempty and unique among pending plans")
        if self.pending_count >= self.max_pending:
            raise CandidatePreparationQueueFull("CPU preparation window is full")
        row = _Row((candidate_id, tuple(vector), None if scenarios is None else tuple(scenarios)))
        self._ids.add(candidate_id)
        self._ordered.append(row)
        self._waiting.append(row)
        self._collect_events()
        self._dispatch()

    def poll(self, *, timeout=0.0, interrupt_check=None):
        self._require_running()
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("CPU preparation poll timeout must be finite and nonnegative")
        if not self._ordered:
            return None
        row = self._ordered[0]
        deadline = time.monotonic() + timeout
        while True:
            if interrupt_check is not None:
                interrupt_check()
            self._collect_events()
            if not row.done:
                self._dispatch()
            if row.done:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                event = self._events.get(timeout=min(remaining, _WAIT_INTERVAL_SECONDS)
                                         if interrupt_check is not None else remaining)
            except Empty:
                continue
            self._handle_event(*event)
        candidate_id = row.job[0]
        if row.error is not None:
            self._failed = True
            raise CandidatePreparationError(candidate_id, row.error)
        try:
            from optimization.native_planning import CandidatePlan

            plan = pickle.loads(row.payload)
            if type(plan) is not CandidatePlan or plan.candidate_id != candidate_id:
                raise RuntimeError("worker returned the wrong candidate plan")
            plan = _plan_parameters(plan, freeze=True)
        except Exception:
            self._failed = True
            raise
        self._ordered.popleft()
        self._ids.remove(candidate_id)
        self._dispatch()
        return plan

    def close(self):
        if self._stopping.is_set():
            return
        self._stopping.set()
        cancelling = not self._started or self.pending_count != 0
        for worker in self._workers:
            if worker.process.is_alive() and cancelling:
                worker.process.terminate()
            if worker.thread is not None and worker.thread.is_alive():
                try:
                    worker.tasks.put_nowait(None)
                except Full:
                    pass  # A cancelled worker already has an unconsumed job.
        for stop in (None, "terminate", "kill"):
            deadline = time.monotonic() + 1
            for worker in self._workers:
                if stop is not None and worker.process.is_alive():
                    getattr(worker.process, stop)()
            for worker in self._workers:
                worker.process.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(worker.process.is_alive() for worker in self._workers):
            raise RuntimeError("CPU planner worker did not stop")
        relay_deadline = time.monotonic() + 1
        for worker in self._workers:
            if worker.thread is not None and worker.thread.ident is not None:
                worker.thread.join(timeout=max(0.0, relay_deadline - time.monotonic()))
            worker.connection.close()
        self._collect_events()
        for worker in self._workers:
            worker.process.close()
        transport_alive = any(worker.thread is not None and worker.thread.is_alive() for worker in self._workers)
        self._workers.clear()
        self._ordered.clear()
        self._waiting.clear()
        self._ids.clear()
        self._snapshot = b""
        if transport_alive:
            raise RuntimeError("CPU planner worker transport did not stop")

    def _close_preserving_error(self):
        try:
            self.close()
        except Exception:
            logging.exception("CPU planner worker cleanup failed after an earlier failure")

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self.close()
        else:
            self._close_preserving_error()
