from __future__ import annotations

from typing import Any, Sequence


def build_evaluation_payload(
    fit_values: Sequence[float],
    penalty: float,
    metrics_payload: dict | None,
    individual,
) -> dict[str, Any]:
    return {
        "fitness": tuple(fit_values),
        "constraint_violation": penalty,
        "metrics": metrics_payload,
        "evaluation_vector": list(individual),
    }


def unpack_evaluation_payload(payload):
    if isinstance(payload, dict):
        try:
            fit_values = tuple(payload["fitness"])
            penalty = payload["constraint_violation"]
        except KeyError as exc:
            raise ValueError(f"missing optimizer evaluation payload field: {exc.args[0]}") from exc
        return fit_values, penalty, payload.get("metrics"), payload.get("evaluation_vector")
    fit_values, penalty, metrics = payload
    return tuple(fit_values), penalty, metrics, None


def apply_evaluation_payload(individual, payload):
    fit_values, penalty, metrics, evaluation_vector = unpack_evaluation_payload(payload)
    if evaluation_vector is not None:
        individual[:] = list(evaluation_vector)
    individual.fitness.values = fit_values
    individual.fitness.constraint_violation = penalty
    individual.constraint_violation = penalty
    return metrics
