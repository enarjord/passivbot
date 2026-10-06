"""CPU-owned fan-in and canonical scoring of asynchronous backtest results.

This module neither submits simulations nor chooses candidates. A completed candidate
can be scored/persisted independently of other in-flight candidates; a screening result
is explicitly separate from a recordable full evaluation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from config.metrics import resolve_metric_value
from config.scoring import to_engine_value
from metrics_schema import MetricAggregationError, build_scenario_metrics, flatten_metric_stats
from optimization.evaluation_payload import build_evaluation_payload
from optimization.gpu.executor import BacktestResult


@dataclass(frozen=True)
class ResultSlot:
    request_id: str
    dataset_id: str
    scenario: str
    exchange: str
    metrics: tuple[str, ...]

    def __post_init__(self):
        for value in (self.request_id, self.dataset_id, self.scenario, self.exchange):
            if not isinstance(value, str) or not value:
                raise ValueError("result slot identities must be nonempty strings")
        if isinstance(self.metrics, str):
            raise ValueError("slot metrics must be a collection of names")
        metrics = tuple(dict.fromkeys(self.metrics))
        if not metrics or any(not isinstance(name, str) or not name for name in metrics):
            raise ValueError("slot metrics must be nonempty names")
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class CandidateCompletion:
    candidate_id: str
    stage: str
    payload: dict

    def require_full(self):
        """The persistence/selection path must not admit partial suite observations."""
        if self.stage != "full":
            raise ValueError("screening results cannot be recorded as complete candidates")
        return self.payload


class CanonicalResultScorer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.suite = callable(getattr(evaluator, "score_scenario_results", None))
        self.base = evaluator.base if self.suite else evaluator
        if self.suite:
            self.coverage = {ctx.label: frozenset(ctx.exchanges) for ctx in evaluator.contexts}
            if len(self.coverage) != len(evaluator.contexts):
                raise ValueError("suite scenario labels must be unique")
        else:
            self.coverage = {"base": frozenset(evaluator.exchanges)}
        if not self.coverage or any(not exchanges for exchanges in self.coverage.values()):
            raise ValueError("scoring requires prepared scenario/exchange coverage")

    def required_metrics(self, scenario):
        metrics = set()
        for index, spec in enumerate(self.base.scoring_specs):
            selected = self.evaluator.objective_bases[index].scenario if self.suite else None
            if selected is None or selected == scenario:
                metrics.add(spec.metric)
        for check in self.base.limit_checks:
            if check.get("scenario") is None or check["scenario"] == scenario:
                metrics.add(check["metric"])
        return frozenset(metrics)

    def validate_coverage(self, pairs, stage):
        if stage not in {"full", "screening"}:
            raise ValueError("evaluation stage must be full or screening")
        selected = {scenario for scenario, _exchange in pairs}
        if not selected or selected - self.coverage.keys():
            raise ValueError("evaluation contains missing or unknown scenarios")
        expected = {(label, exchange) for label in selected for exchange in self.coverage[label]}
        if set(pairs) != expected or (stage == "full" and selected != self.coverage.keys()):
            raise ValueError("evaluation must contain complete required scenario/exchange coverage")
        required = {check["scenario"] for check in self.base.limit_checks if check.get("scenario") is not None}
        if self.suite:
            required.update(basis.scenario for basis in self.evaluator.objective_bases if basis.scenario is not None)
        if required - selected:
            raise ValueError("screening must retain explicitly selected objective/limit scenarios")

    def score(self, vector, analyses):
        liquidated = any(row["liquidated"] for venues in analyses.values() for row in venues.values())
        try:
            if self.suite:
                from suite_runner import ScenarioResult, SuiteScenario
                rows = [ScenarioResult(
                    scenario=SuiteScenario(label, None, None, None, None),
                    per_exchange=analyses[label], metrics=build_scenario_metrics(analyses[label]),
                    elapsed_seconds=0.0, output_path=None,
                ) for label in self.coverage if label in analyses]
                scored = self.evaluator.score_scenario_results(rows)
                objectives, penalty = scored["objectives"], scored["constraint_violation"]
                metrics = dict(
                    objectives={f"w_{index}": value for index, value in enumerate(objectives)},
                    unpenalized_objectives=scored["unpenalized_objectives"],
                    suite_metrics=scored["suite_metrics"], constraint_violation=penalty,
                )
                if self.evaluator.objective_scenario is not None:
                    metrics["objective_scenario"] = self.evaluator.objective_scenario
            else:
                scenario = build_scenario_metrics(analyses["base"])
                objectives, penalty, raw = self.base.calc_fitness(
                    flatten_metric_stats(scenario["stats"]), return_raw_objectives=True,
                )
                metrics = dict(scenario, objectives=raw, constraint_violation=penalty,
                               unpenalized_objectives=tuple(to_engine_value(spec, raw[spec.metric])
                                                           for spec in self.base.scoring_specs))
        except MetricAggregationError as error:
            from optimize import _build_invalid_candidate_metrics
            objectives, penalty, metrics = _build_invalid_candidate_metrics(
                self.base.config["optimize"]["scoring"], f"{type(error).__name__}: {error}",
                include_stats=not self.suite, include_suite_metrics=self.suite,
            )
        metrics["liquidated"] = liquidated
        return build_evaluation_payload(objectives, penalty, metrics, vector)


class CandidateEvaluation:
    """One bounded candidate's result collection, independent of completion order."""

    def __init__(self, candidate_id, vector, slots, scorer: CanonicalResultScorer, *, stage="full"):
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate identity must be a nonempty string")
        self.candidate_id, self.stage, self.scorer = candidate_id, stage, scorer
        self.vector = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in self.vector):
            raise ValueError("candidate values must be finite before simulation")
        slots = tuple(slots)
        if not slots or any(not isinstance(slot, ResultSlot) for slot in slots):
            raise ValueError("candidate evaluation requires result slots")
        self._slots = {slot.request_id: slot for slot in slots}
        pairs = {(slot.scenario, slot.exchange) for slot in slots}
        if len(self._slots) != len(slots) or len(pairs) != len(slots):
            raise ValueError("request and scenario/exchange slots must be unique")
        scorer.validate_coverage(pairs, stage)
        for slot in slots:
            missing = scorer.required_metrics(slot.scenario) - set(slot.metrics)
            if missing:
                raise ValueError(f"result slot is missing required scoring/limit metrics: {sorted(missing)}")
        self._rows = {}
        self._complete = False

    @property
    def pending_request_ids(self):
        return tuple(name for name in self._slots if name not in self._rows) if not self._complete else ()

    def add_result(self, result: BacktestResult):
        if not isinstance(result, BacktestResult):
            raise TypeError("expected an identified backtest result")
        if self._complete or result.request_id in self._rows:
            raise RuntimeError("candidate result has already been consumed")
        slot = self._slots.get(result.request_id)
        if slot is None or slot.dataset_id != result.dataset_id:
            raise RuntimeError("result identity does not match its candidate evaluation slot")
        if not isinstance(result.liquidated, bool):
            raise RuntimeError("authoritative results require actual simulator liquidation status")
        if not isinstance(result.metrics, Mapping) or not result.metrics:
            raise RuntimeError("backtest result requires metric values")
        row = dict(result.metrics)
        if any(not isinstance(value, (int, float)) or math.isnan(value) for value in row.values()):
            raise RuntimeError("backtest result contains malformed metric values")
        missing = [name for name in slot.metrics if resolve_metric_value(row, name) is None]
        if missing:
            raise RuntimeError(f"backtest result is missing requested metrics: {missing}")
        row["liquidated"] = result.liquidated
        self._rows[result.request_id] = row
        if len(self._rows) != len(self._slots):
            return None
        analyses = {}
        for request_id, slot in self._slots.items():
            analyses.setdefault(slot.scenario, {})[slot.exchange] = self._rows[request_id]
        payload = self.scorer.score(self.vector, analyses)
        self._rows.clear()
        self._complete = True
        return CandidateCompletion(self.candidate_id, self.stage, payload)
