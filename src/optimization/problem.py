from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from optimization.backend_shared import drain_async_results
from optimization.bounds import Bound
from optimization.callback import build_pymoo_record_entry


def _evaluate_pymoo_worker(
    evaluator,
    vector: Sequence[float],
    overrides_list: Sequence[str] | None,
    n_obj: int,
    has_constraints: bool,
) -> dict[str, Any]:
    evaluated_vector = list(float(v) for v in vector)
    objectives, constraint_violation, metrics = evaluator.evaluate(
        evaluated_vector,
        list(overrides_list or []),
    )
    objectives_arr = np.asarray(objectives, dtype=np.float64)
    if len(objectives_arr) != int(n_obj):
        raise ValueError(
            f"pymoo objective length mismatch: expected {int(n_obj)}, got {len(objectives_arr)}"
        )
    payload = {
        "F": objectives_arr,
        "metrics": metrics or {},
        "evaluation_vector": np.asarray(evaluated_vector, dtype=np.float64),
    }
    if has_constraints:
        violation = float(constraint_violation)
        payload["G"] = np.asarray([violation if violation > 0.0 else -1.0], dtype=np.float64)
    return payload


class PymooEvaluatorAdapter:
    def __init__(self, evaluator, *, overrides_list: Sequence[str] | None = None):
        self.evaluator = evaluator
        self.overrides_list = list(overrides_list or [])

    @property
    def has_constraints(self) -> bool:
        if hasattr(self.evaluator, "limit_checks"):
            return bool(getattr(self.evaluator, "limit_checks"))
        base = getattr(self.evaluator, "base", None)
        return bool(getattr(base, "limit_checks", []))

    def evaluate(self, vector: Sequence[float]) -> dict[str, Any]:
        evaluated_vector = list(float(v) for v in vector)
        objectives, constraint_violation, metrics = self.evaluator.evaluate(
            evaluated_vector,
            self.overrides_list,
        )
        return {
            "objectives": list(objectives),
            "constraint_violation": float(constraint_violation),
            "metrics": metrics or {},
            "evaluation_vector": np.asarray(evaluated_vector, dtype=np.float64),
        }


class PymooAsyncRecordingRunner:
    def __init__(
        self,
        *,
        evaluator,
        has_constraints: bool,
        n_obj: int,
        pool,
        recorder,
        template: dict,
        build_config_fn,
        overrides_fn,
        overrides_list: Sequence[str] | None = None,
        poll_interval_seconds: float = 0.05,
    ):
        self.evaluator = evaluator
        self.has_constraints = bool(has_constraints)
        self.n_obj = int(n_obj)
        self.pool = pool
        self.recorder = recorder
        self.template = template
        self.build_config_fn = build_config_fn
        self.overrides_fn = overrides_fn
        self.overrides_list = list(overrides_list or [])
        self.poll_interval_seconds = max(0.0, float(poll_interval_seconds))

    def _record_result(self, vector, metrics) -> None:
        entry = build_pymoo_record_entry(
            vector=vector,
            metrics=metrics,
            template=self.template,
            build_config_fn=self.build_config_fn,
            overrides_fn=self.overrides_fn,
            overrides_list=self.overrides_list,
        )
        self.recorder.record(entry)

    def __call__(self, _f, X):
        xs = list(X)
        ordered_results: list[dict[str, Any] | None] = [None] * len(xs)
        pending = {
            self.pool.apply_async(
                _evaluate_pymoo_worker,
                (
                    self.evaluator,
                    x,
                    self.overrides_list,
                    self.n_obj,
                    self.has_constraints,
                ),
            ): (idx, x)
            for idx, x in enumerate(xs)
        }

        def _on_result(context, payload):
            idx, x = context
            self._record_result(
                payload.get("evaluation_vector", x),
                payload.get("metrics") or {},
            )
            slim_payload = dict(payload)
            slim_payload.pop("metrics", None)
            slim_payload.pop("evaluation_vector", None)
            ordered_results[idx] = slim_payload

        drain_async_results(
            pending,
            poll_interval_seconds=self.poll_interval_seconds,
            on_result=_on_result,
        )

        return [payload for payload in ordered_results if payload is not None]


class PassivbotProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        bounds: Sequence[Bound],
        scoring_keys: Sequence[str],
        evaluator_adapter: PymooEvaluatorAdapter,
        **kwargs,
    ):
        self.bounds = list(bounds)
        self.scoring_keys = list(scoring_keys)
        self.evaluator_adapter = evaluator_adapter
        xl = np.asarray([bound.low for bound in self.bounds], dtype=np.float64)
        xu = np.asarray([bound.high for bound in self.bounds], dtype=np.float64)
        super().__init__(
            n_var=len(self.bounds),
            n_obj=len(self.scoring_keys),
            n_ieq_constr=1 if evaluator_adapter.has_constraints else 0,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        payload = self.evaluator_adapter.evaluate(x)
        objectives = np.asarray(payload["objectives"], dtype=np.float64)
        if len(objectives) != self.n_obj:
            raise ValueError(
                f"pymoo objective length mismatch: expected {self.n_obj}, got {len(objectives)}"
            )
        out["F"] = objectives
        if self.n_ieq_constr:
            violation = float(payload["constraint_violation"])
            out["G"] = np.asarray([violation if violation > 0.0 else -1.0], dtype=np.float64)
        out["metrics"] = payload["metrics"]
        out["evaluation_vector"] = payload["evaluation_vector"]
