"""Bounded CPU orchestration over a black-box asynchronous backtest service.

The caller drives polling, selection and persistence on the CPU. Future callbacks
only enqueue notifications; scoring never occupies a GPU worker's completion path.
The service is exclusively borrowed and its lifetime belongs to the caller.
"""

from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future
from copy import deepcopy
import json
import math
from queue import Empty, SimpleQueue

from optimization.gpu.executor import BacktestQueueFull, BacktestRequest, BacktestResult
from optimization.native_results import CandidateCompletion


class CandidateQueueFull(RuntimeError):
    """CPU candidate admission would exceed its bounded collection window."""


class NativeEvaluationSession:
    def __init__(self, service, scorer, *, max_candidates=128, cache_size=1024):
        for name, value in (("max_candidates", max_candidates), ("cache_size", cache_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.service, self.scorer = service, scorer
        self.max_candidates, self.cache_size = max_candidates, cache_size
        self._active, self._primary, self._aliases = {}, {}, {}
        self._queued = deque()
        self._futures = {}
        self._notifications = SimpleQueue()
        self._ready = deque()
        self._cache = OrderedDict()
        self._row_cache = OrderedDict()
        self._stopped = False
        self._failure = None
        self._submission_failure = None

    @property
    def active_candidate_ids(self):
        return tuple(self._active)

    @property
    def pending_request_count(self):
        return len(self._queued) + len(self._futures)

    @staticmethod
    def _key(plan):
        # The full-candidate identity is shared, but cached screening observations
        # must never satisfy another stage or a different scenario subset.
        return plan.stage, tuple(slot.dataset_id for slot in plan.slots), plan.effective_key

    def admit(self, plan):
        if self._failure is not None:
            raise RuntimeError("CPU evaluation session failed") from self._failure
        if self._stopped:
            raise RuntimeError("CPU evaluation session admission is stopped")
        if plan.candidate_id in self._active:
            raise ValueError("candidate is already active")
        if len(self._active) >= self.max_candidates:
            raise CandidateQueueFull("CPU candidate collection window is full")
        collector = plan.collector(self.scorer)
        slots = {slot.request_id: slot.dataset_id for slot in plan.slots}
        requests = {request.request_id: request.dataset_id for request in plan.requests}
        if requests != slots or len(requests) != len(plan.requests):
            raise ValueError("candidate requests must exactly match its result slots")
        other_ids = {request.request_id for _owner, request in self._queued}
        other_ids.update(request.request_id for _owner, request in self._futures.values())
        if other_ids & requests.keys():
            raise ValueError("request identities collide with admitted work")
        snapshots = tuple(BacktestRequest(request.request_id, request.dataset_id,
                                          deepcopy(dict(request.parameters)))
                          for request in plan.requests)
        key = self._key(plan)
        self._active[plan.candidate_id] = (plan, collector)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._ready.append(self._reuse(plan, self._cache[key]))
        elif key in self._primary:
            self._aliases.setdefault(key, []).append(plan.candidate_id)
        else:
            self._primary[key] = plan.candidate_id
            self._queued.extend((plan.candidate_id, request) for request in snapshots)

    @staticmethod
    def _reuse(plan, payload):
        payload = deepcopy(payload)
        payload["evaluation_vector"] = list(plan.vector)
        return CandidateCompletion(plan.candidate_id, plan.stage, payload)

    @staticmethod
    def _row_key(plan, request):
        # Keep request parameters explicit: future scenario plans may share one
        # prepared dataset while overriding different dynamic execution values.
        return (plan.effective_key, request.dataset_id,
                json.dumps(dict(request.parameters), sort_keys=True, allow_nan=False))

    def _pump(self):
        if self._submission_failure is not None:
            if not self._futures and not self._ready:
                raise self._submission_failure
            return
        while self._queued and not self._stopped:
            candidate_id, request = self._queued[0]
            try:
                plan, _collector = self._active[candidate_id]
                key = self._row_key(plan, request)
                if key in self._row_cache:
                    # Reuse identified simulator evidence, never a partial score.
                    # Full collection must still validate every required slot.
                    self._row_cache.move_to_end(key)
                    row = self._row_cache[key]
                    future = Future()
                    future.set_result(BacktestResult(
                        request.request_id, request.dataset_id, deepcopy(row.metrics), row.liquidated,
                    ))
                else:
                    future = self.service.submit(request)
            except BacktestQueueFull:
                if not self._futures:
                    raise RuntimeError("borrowed backtest service is full without session-owned work")
                return
            except Exception as failure:
                if not self._futures:
                    raise
                # A fast worker can fail while we are replenishing the queue.
                # Consume earlier successes and the original future error before
                # a subsequent submit's service-failed wrapper can poison fan-in.
                self._submission_failure = failure
                return
            self._queued.popleft()
            self._futures[future] = (candidate_id, request)
            future.add_done_callback(self._notifications.put)

    def _complete(self, complete):
        plan, _collector = self._active[complete.candidate_id]
        key = self._key(plan)
        self._cache[key] = deepcopy(complete.payload)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        self._primary.pop(key, None)
        self._ready.append(complete)
        for alias_id in self._aliases.pop(key, []):
            alias, _collector = self._active[alias_id]
            self._ready.append(self._reuse(alias, complete.payload))

    def poll(self, *, timeout=0.0, max_results=256, max_completions=None):
        """Bound device notifications separately from returned complete candidates.

        A large suite can consume many notifications before yielding one candidate;
        cached duplicates can yield many candidates from one notification. Keep both
        bounds independent so CPU persistence cadence need not throttle suite fan-in.
        """
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        if max_completions is None:
            max_completions = max_results
        if isinstance(max_completions, bool) or not isinstance(max_completions, int) or max_completions < 1:
            raise ValueError("max_completions must be a positive integer")
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("poll timeout must be finite and nonnegative")
        if self._failure is not None:
            raise RuntimeError("CPU evaluation session failed") from self._failure
        try:
            self._pump()
            processed = 0
            while processed < max_results and not self._ready:
                try:
                    future = self._notifications.get(timeout=timeout if processed == 0 else 0)
                except Empty:
                    break
                processed += 1
                candidate_id, _request = self._futures.pop(future)
                try:
                    row = future.result()
                except CancelledError:
                    if not self._stopped:
                        raise RuntimeError("an admitted backtest was cancelled unexpectedly")
                    continue
                if not isinstance(row, BacktestResult) or (
                    row.request_id, row.dataset_id
                ) != (_request.request_id, _request.dataset_id):
                    raise RuntimeError("backtest result identity does not match its submitted request")
                plan, collector = self._active[candidate_id]
                complete = collector.add_result(row)
                # Only collector-validated simulator rows enter this bounded,
                # run-local cache. Keys include the complete effective candidate
                # identity and prepared dataset, independent of screening stage.
                key = self._row_key(plan, _request)
                self._row_cache[key] = deepcopy(row)
                self._row_cache.move_to_end(key)
                while len(self._row_cache) > self.cache_size:
                    self._row_cache.popitem(last=False)
                if complete is not None:
                    self._complete(complete)
                self._pump()
            ready = []
            while self._ready and len(ready) < max_completions:
                complete = self._ready.popleft()
                del self._active[complete.candidate_id]
                ready.append(complete)
            return ready
        except BaseException as failure:
            self._failure = failure
            self.stop_admission()
            raise

    def stop_admission(self):
        """Drop unsubmitted work and cancel pending futures; preserve ready results.

        Running requests remain service-owned. Close/drain the service, then poll
        available notifications to retain candidates whose full work completed.
        Partial candidates may be re-created and rerun on GPU after resumption.
        """
        self._stopped = True
        self._queued.clear()
        for future in tuple(self._futures):
            future.cancel()
