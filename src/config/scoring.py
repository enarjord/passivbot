from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .access import require_config_dict
from .limits import resolve_reducer_mode
from .log_output import log_config_message
from .metrics import canonicalize_metric_name
from .reducers import REDUCER_ALIASES, normalize_reducer, reducer_from_aliases

OBJECTIVE_GOALS = ("min", "max")
SCORING_ENTRY_FIELDS = {"metric", "goal", "scenario", *REDUCER_ALIASES}


class ScenarioSelection(Enum):
    INHERIT = "inherit"


DEFAULT_OBJECTIVE_GOALS = {
    "positions_held_per_day": "min",
    "positions_held_per_day_w": "min",
    "position_held_hours_mean": "min",
    "position_held_hours_max": "min",
    "position_held_hours_median": "min",
    "position_unchanged_hours_max": "min",
    "entry_interval_hours_mean": "min",
    "entry_interval_hours_median": "min",
    "entry_interval_hours_p95": "min",
    "entry_interval_hours_p99": "min",
    "entry_interval_hours_max": "min",
    "fills_active_days_count": "max",
    "fills_active_days_ratio": "max",
    "fills_active_symbols_count": "max",
    "fills_count": "max",
    "fills_count_close": "max",
    "fills_count_entry": "max",
    "fills_count_long": "max",
    "fills_count_short": "max",
    "fills_gap_longest_days": "min",
    "fills_gap_mean_hours": "min",
    "fills_gap_median_hours": "min",
    "fills_gap_p95_hours": "min",
    "fills_gap_p99_hours": "min",
    "fills_per_day": "max",
    "fills_per_day_close": "max",
    "fills_per_day_entry": "max",
    "fills_per_day_long": "max",
    "fills_per_day_per_position_slot": "max",
    "fills_per_day_per_position_slot_long": "max",
    "fills_per_day_per_position_slot_short": "max",
    "fills_per_day_short": "max",
    "fills_top_symbol_share": "min",
    "backtest_completion_ratio": "max",
    "high_exposure_hours_mean_long": "min",
    "high_exposure_hours_max_long": "min",
    "high_exposure_hours_mean_short": "min",
    "high_exposure_hours_max_short": "min",
    "adg_pnl": "max",
    "adg_pnl_w": "max",
    "gain_strategy_eq": "max",
    "adg_strategy_eq": "max",
    "mdg_strategy_eq": "max",
    "sharpe_ratio_strategy_eq": "max",
    "sortino_ratio_strategy_eq": "max",
    "omega_ratio_strategy_eq": "max",
    "expected_shortfall_1pct_strategy_eq": "min",
    "calmar_ratio_strategy_eq": "max",
    "sterling_ratio_strategy_eq": "max",
    "adg_strategy_eq_w": "max",
    "mdg_strategy_eq_w": "max",
    "sharpe_ratio_strategy_eq_w": "max",
    "sortino_ratio_strategy_eq_w": "max",
    "omega_ratio_strategy_eq_w": "max",
    "calmar_ratio_strategy_eq_w": "max",
    "sterling_ratio_strategy_eq_w": "max",
    "drawdown_worst_strategy_eq": "min",
    "drawdown_worst_mean_1pct_strategy_eq": "min",
    "strategy_eq_underwater_pct_mean": "min",
    "strategy_eq_underwater_pct_median": "min",
    "strategy_eq_recovery_days_mean": "min",
    "strategy_eq_recovery_days_median": "min",
    "strategy_eq_recovery_days_p95": "min",
    "strategy_eq_recovery_days_p99": "min",
    "strategy_eq_recovery_days_mean_worst_5pct": "min",
    "strategy_eq_recovery_days_mean_worst_1pct": "min",
    "strategy_eq_recovery_days_max": "min",
    "peak_recovery_hours_strategy_eq": "min",
    "mdg_pnl": "max",
    "mdg_pnl_w": "max",
    "sharpe_ratio_pnl": "max",
    "sharpe_ratio_pnl_w": "max",
    "sortino_ratio_pnl": "max",
    "sortino_ratio_pnl_w": "max",
    "adg": "max",
    "adg_per_exposure_long": "max",
    "adg_per_exposure_short": "max",
    "adg_w": "max",
    "adg_w_per_exposure_long": "max",
    "adg_w_per_exposure_short": "max",
    "calmar_ratio": "max",
    "calmar_ratio_w": "max",
    "drawdown_worst": "min",
    "drawdown_worst_mean_1pct": "min",
    "equity_balance_diff_neg_max": "min",
    "equity_balance_diff_neg_mean": "min",
    "equity_balance_diff_pos_max": "min",
    "equity_balance_diff_pos_mean": "min",
    "paper_loss_ratio": "max",
    "paper_loss_mean_ratio": "max",
    "exposure_ratio": "max",
    "exposure_mean_ratio": "max",
    "equity_choppiness": "min",
    "equity_choppiness_w": "min",
    "equity_jerkiness": "min",
    "equity_jerkiness_w": "min",
    "peak_recovery_hours_equity": "min",
    "expected_shortfall_1pct": "min",
    "exponential_fit_error": "min",
    "exponential_fit_error_w": "min",
    "gain": "max",
    "gain_per_exposure_long": "max",
    "gain_per_exposure_short": "max",
    "loss_profit_ratio": "min",
    "loss_profit_ratio_w": "min",
    "mdg": "max",
    "mdg_per_exposure_long": "max",
    "mdg_per_exposure_short": "max",
    "mdg_w": "max",
    "mdg_w_per_exposure_long": "max",
    "mdg_w_per_exposure_short": "max",
    "omega_ratio": "max",
    "omega_ratio_w": "max",
    "sharpe_ratio": "max",
    "sharpe_ratio_w": "max",
    "sortino_ratio": "max",
    "sortino_ratio_w": "max",
    "sterling_ratio": "max",
    "sterling_ratio_w": "max",
    "paper_loss_ratio_w": "max",
    "paper_loss_mean_ratio_w": "max",
    "exposure_ratio_w": "max",
    "exposure_mean_ratio_w": "max",
    "total_wallet_exposure_max": "min",
    "total_wallet_exposure_mean": "min",
    "total_wallet_exposure_median": "min",
    "volume_pct_per_day_avg": "max",
    "volume_pct_per_day_avg_w": "max",
    "entry_initial_balance_pct_long": "max",
    "entry_initial_balance_pct_short": "max",
}


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str
    goal: str
    scenario: str | None | ScenarioSelection = ScenarioSelection.INHERIT
    reducer: str | None = None

    @property
    def engine_sign(self) -> float:
        return -1.0 if self.goal == "max" else 1.0

    def to_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"metric": self.metric, "goal": self.goal}
        if self.scenario is not ScenarioSelection.INHERIT:
            payload["scenario"] = self.scenario
        if self.reducer is not None:
            payload["reducer"] = self.reducer
        return payload


@dataclass(frozen=True)
class ObjectiveBasis:
    scenario: str | None
    reducer: str | None


def _normalize_goal(value: Any, *, path: str) -> str:
    goal = str(value or "").strip().lower()
    if goal not in OBJECTIVE_GOALS:
        allowed = ", ".join(OBJECTIVE_GOALS)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {value!r}")
    return goal


def _normalize_scenario(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a scenario label string or null, got {value!r}")
    scenario = value.strip()
    if not scenario:
        raise ValueError(f"{path} must be a non-empty scenario label or null")
    return scenario


def default_objective_goal(metric: str) -> str | None:
    canonical = canonicalize_metric_name(metric)
    goal = DEFAULT_OBJECTIVE_GOALS.get(canonical) or DEFAULT_OBJECTIVE_GOALS.get(str(metric).strip())
    if goal is not None:
        return goal
    if canonical.endswith(("_usd", "_btc")):
        base_metric = canonical.rsplit("_", 1)[0]
        goal = DEFAULT_OBJECTIVE_GOALS.get(base_metric)
        if goal is not None:
            return goal
    return None


def _normalize_spec(
    item: Any,
    *,
    index: int,
    unknown_goal: str,
) -> tuple[ObjectiveSpec, bool]:
    if isinstance(item, ObjectiveSpec):
        scenario = item.scenario
        if scenario is not ScenarioSelection.INHERIT:
            scenario = _normalize_scenario(
                scenario,
                path=f"config.optimize.scoring[{index}].scenario",
            )
        spec = ObjectiveSpec(
            metric=canonicalize_metric_name(item.metric),
            goal=_normalize_goal(item.goal, path=f"config.optimize.scoring[{index}].goal"),
            scenario=scenario,
            reducer=normalize_reducer(
                item.reducer,
                path=f"config.optimize.scoring[{index}].reducer",
            ),
        )
        if isinstance(spec.scenario, str) and spec.reducer is not None:
            raise ValueError(
                f"config.optimize.scoring[{index}] cannot set both a named scenario "
                "and reducer"
            )
        return spec, False

    if isinstance(item, str):
        metric = canonicalize_metric_name(item.strip())
        if not metric:
            raise ValueError(f"config.optimize.scoring[{index}] must be a non-empty metric name")
        goal = default_objective_goal(metric)
        if goal is None:
            if unknown_goal == "error":
                raise ValueError(
                    f"config.optimize.scoring[{index}]={item!r} has no default optimization goal; "
                    "use the object form with explicit goal"
                )
            goal = unknown_goal
        return ObjectiveSpec(metric=metric, goal=goal), True

    if isinstance(item, dict):
        unknown_fields = sorted(set(item) - SCORING_ENTRY_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"config.optimize.scoring[{index}] has unknown field(s): "
                f"{', '.join(unknown_fields)}"
            )
        metric = canonicalize_metric_name(str(item.get("metric", "")).strip())
        if not metric:
            raise ValueError(f"config.optimize.scoring[{index}].metric must be a non-empty string")
        raw_goal = item.get("goal")
        if raw_goal is None:
            goal = default_objective_goal(metric)
            if goal is None:
                if unknown_goal == "error":
                    raise ValueError(
                        f"config.optimize.scoring[{index}] for metric {metric!r} must set goal"
                    )
                goal = unknown_goal
        else:
            goal = _normalize_goal(raw_goal, path=f"config.optimize.scoring[{index}].goal")
        scenario = (
            _normalize_scenario(
                item.get("scenario"),
                path=f"config.optimize.scoring[{index}].scenario",
            )
            if "scenario" in item
            else ScenarioSelection.INHERIT
        )
        reducer, _reducer_present = reducer_from_aliases(
            item,
            path=f"config.optimize.scoring[{index}]",
        )
        if isinstance(scenario, str) and reducer is not None:
            raise ValueError(
                f"config.optimize.scoring[{index}] cannot set both a named scenario "
                "and reducer"
            )
        return ObjectiveSpec(
            metric=metric,
            goal=goal,
            scenario=scenario,
            reducer=reducer,
        ), False

    raise ValueError(
        "config.optimize.scoring entries must be strings or objects like "
        "{metric: ..., goal: min|max, scenario: label|null, reducer: mean|min|max|std|median}"
    )


def normalize_scoring_entries(
    scoring: Sequence[Any] | None,
    *,
    dedupe: bool = True,
    unknown_goal: str = "error",
) -> tuple[list[ObjectiveSpec], bool]:
    if scoring is None:
        return [], False
    if not isinstance(scoring, (list, tuple)):
        raise ValueError(
            "config.optimize.scoring must be a list of metric names or scoring objects"
        )
    normalized: list[ObjectiveSpec] = []
    changed = False
    seen_metrics: set[str] = set()
    for idx, item in enumerate(scoring):
        spec, converted_legacy = _normalize_spec(item, index=idx, unknown_goal=unknown_goal)
        changed = changed or converted_legacy or not isinstance(item, ObjectiveSpec)
        if dedupe and spec.metric in seen_metrics:
            continue
        normalized.append(spec)
        seen_metrics.add(spec.metric)
    return normalized, changed


def normalize_scoring_config(config: dict, *, verbose: bool = True, tracker=None) -> None:
    optimize_cfg = require_config_dict(config, "optimize")
    before = optimize_cfg.get("scoring", [])
    normalized, changed = normalize_scoring_entries(before, unknown_goal="error")
    normalized_payload = [spec.to_config() for spec in normalized]
    optimize_cfg["scoring"] = normalized_payload
    if changed and before != normalized_payload:
        log_config_message(
            verbose,
            20,
            "normalized optimize.scoring to canonical schema (%d entries)",
            len(normalized_payload),
        )
        if tracker is not None:
            tracker.update(["optimize", "scoring"], before, normalized_payload)


def extract_objective_specs(config_or_scoring: Any) -> list[ObjectiveSpec]:
    if isinstance(config_or_scoring, dict):
        scoring = config_or_scoring.get("optimize", {}).get("scoring", [])
    else:
        scoring = config_or_scoring
    normalized, _ = normalize_scoring_entries(scoring, dedupe=False, unknown_goal="min")
    return normalized


def objective_metric_names(config_or_scoring: Any) -> list[str]:
    return [spec.metric for spec in extract_objective_specs(config_or_scoring)]


def objective_goal_map(config_or_scoring: Any) -> dict[str, str]:
    return {spec.metric: spec.goal for spec in extract_objective_specs(config_or_scoring)}


def objective_index_map(config_or_scoring: Any) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for idx, spec in enumerate(extract_objective_specs(config_or_scoring)):
        mapping.setdefault(spec.metric, []).append(idx)
    return mapping


def objective_spec_by_metric(config_or_scoring: Any) -> dict[str, ObjectiveSpec]:
    return {spec.metric: spec for spec in extract_objective_specs(config_or_scoring)}


def resolve_objective_basis(
    spec: ObjectiveSpec,
    *,
    default_scenario: str | None,
    reducer_cfg: dict[str, Any] | None,
) -> ObjectiveBasis:
    scenario = default_scenario if spec.scenario is ScenarioSelection.INHERIT else spec.scenario
    if scenario is not None:
        if spec.reducer is not None:
            raise ValueError(
                f"scoring objective {spec.metric!r} resolves to scenario {scenario!r} "
                f"and cannot also use reducer {spec.reducer!r}; set scenario to null "
                "to use suite reduction"
            )
        return ObjectiveBasis(scenario=scenario, reducer=None)
    reducer = spec.reducer or resolve_reducer_mode(spec.metric, reducer_cfg)
    return ObjectiveBasis(scenario=None, reducer=reducer)


def to_engine_value(spec: ObjectiveSpec, raw_value: float) -> float:
    return float(raw_value) * spec.engine_sign


def from_engine_value(spec: ObjectiveSpec, engine_value: float) -> float:
    return float(engine_value) * spec.engine_sign


def engine_space_fitness_weights(config_or_scoring: Any) -> tuple[float, ...]:
    """
    Shared optimizer fitness weights for engine-space objectives.

    Engine-space values are already normalized so that lower is better for every
    objective, regardless of the original raw metric goal.
    """
    return tuple(-1.0 for _ in extract_objective_specs(config_or_scoring))


def dominates_objectives(
    lhs: Sequence[float],
    rhs: Sequence[float],
    specs: Sequence[ObjectiveSpec],
) -> bool:
    better_in_one = False
    for a, b, spec in zip(lhs, rhs, specs):
        if spec.goal == "max":
            if a > b:
                better_in_one = True
            elif a < b:
                return False
        else:
            if a < b:
                better_in_one = True
            elif a > b:
                return False
    return better_in_one


def objective_display_name(spec: ObjectiveSpec) -> str:
    basis = ""
    if isinstance(spec.scenario, str):
        basis = f", scenario={spec.scenario}"
    elif spec.scenario is None:
        basis = ", reducer"
    if spec.reducer is not None:
        basis += f"={spec.reducer}"
    return f"{spec.metric} ({spec.goal}{basis})"


def default_scoring_weights() -> dict[str, float]:
    weights: dict[str, float] = {}
    for metric, goal in DEFAULT_OBJECTIVE_GOALS.items():
        weight = -1.0 if goal == "max" else 1.0
        weights[metric] = weight
        canonical = canonicalize_metric_name(metric)
        weights.setdefault(canonical, weight)
    return weights
