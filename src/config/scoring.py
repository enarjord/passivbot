from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .access import require_config_dict
from .log_output import log_config_message
from .metrics import canonicalize_metric_name

OBJECTIVE_GOALS = ("min", "max")

DEFAULT_OBJECTIVE_GOALS = {
    "positions_held_per_day": "min",
    "positions_held_per_day_w": "min",
    "position_held_hours_mean": "min",
    "position_held_hours_max": "min",
    "position_held_hours_median": "min",
    "position_unchanged_hours_max": "min",
    "high_exposure_hours_mean_long": "min",
    "high_exposure_hours_max_long": "min",
    "high_exposure_hours_mean_short": "min",
    "high_exposure_hours_max_short": "min",
    "adg_pnl": "max",
    "adg_pnl_w": "max",
    "gain_strategy_pnl_rebased": "max",
    "adg_strategy_pnl_rebased": "max",
    "mdg_strategy_pnl_rebased": "max",
    "sharpe_ratio_strategy_pnl_rebased": "max",
    "sortino_ratio_strategy_pnl_rebased": "max",
    "omega_ratio_strategy_pnl_rebased": "max",
    "expected_shortfall_1pct_strategy_pnl_rebased": "min",
    "calmar_ratio_strategy_pnl_rebased": "max",
    "sterling_ratio_strategy_pnl_rebased": "max",
    "adg_strategy_pnl_rebased_w": "max",
    "mdg_strategy_pnl_rebased_w": "max",
    "sharpe_ratio_strategy_pnl_rebased_w": "max",
    "sortino_ratio_strategy_pnl_rebased_w": "max",
    "omega_ratio_strategy_pnl_rebased_w": "max",
    "calmar_ratio_strategy_pnl_rebased_w": "max",
    "sterling_ratio_strategy_pnl_rebased_w": "max",
    "drawdown_worst_hsl": "min",
    "drawdown_worst_mean_1pct_hsl": "min",
    "peak_recovery_hours_hsl": "min",
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

    @property
    def engine_sign(self) -> float:
        return -1.0 if self.goal == "max" else 1.0

    def to_config(self) -> dict[str, str]:
        return {"metric": self.metric, "goal": self.goal}


def _normalize_goal(value: Any, *, path: str) -> str:
    goal = str(value or "").strip().lower()
    if goal not in OBJECTIVE_GOALS:
        allowed = ", ".join(OBJECTIVE_GOALS)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {value!r}")
    return goal


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
        spec = ObjectiveSpec(
            metric=canonicalize_metric_name(item.metric),
            goal=_normalize_goal(item.goal, path=f"config.optimize.scoring[{index}].goal"),
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
        return ObjectiveSpec(metric=metric, goal=goal), False

    raise ValueError(
        "config.optimize.scoring entries must be strings or objects like "
        "{metric: ..., goal: min|max}"
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
    return f"{spec.metric} ({spec.goal})"


def default_scoring_weights() -> dict[str, float]:
    weights: dict[str, float] = {}
    for metric, goal in DEFAULT_OBJECTIVE_GOALS.items():
        weight = -1.0 if goal == "max" else 1.0
        weights[metric] = weight
        canonical = canonicalize_metric_name(metric)
        weights.setdefault(canonical, weight)
    return weights
