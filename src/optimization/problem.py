from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from optimization.bounds import Bound


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
