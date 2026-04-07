from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from config.metrics import ANALYSIS_SHARED_KEYS, CURRENCY_METRICS, SHARED_METRICS
from config.scoring import extract_objective_specs
from utils import trim_analysis_aliases


@dataclass(frozen=True)
class VisibleAnalysis:
    analysis: dict
    shown_count: int
    total_count: int
    resolved_metric_names: tuple[str, ...]


def collect_visible_metric_requests(config: dict) -> tuple[list[str], list[str], object]:
    optimize_cfg = (config or {}).get("optimize", {}) or {}
    derived = []
    for spec in extract_objective_specs(optimize_cfg.get("scoring", []) or []):
        metric = spec.metric
        if metric.endswith(("_usd", "_btc")) and metric.rsplit("_", 1)[0] in CURRENCY_METRICS:
            metric = metric.rsplit("_", 1)[0]
        derived.append(metric)
    derived.extend(
        limit["metric"]
        for limit in (optimize_cfg.get("limits", []) or [])
        if isinstance(limit, dict) and limit.get("metric")
    )
    visible_cfg = (config or {}).get("backtest", {}).get("visible_metrics")
    explicit = [] if visible_cfg is None else _normalize_visible_metrics_config(visible_cfg)
    return derived, explicit, visible_cfg


def resolve_visible_metric_names(
    config: dict,
    analysis_keys: Iterable[str],
) -> list[str]:
    ordered_keys = list(analysis_keys)
    key_set = set(ordered_keys)
    derived, explicit, visible_cfg = collect_visible_metric_requests(config)
    if visible_cfg == []:
        return ordered_keys

    resolved = []
    resolved_seen = set()
    unresolved_explicit = []
    for metric in [*derived, *explicit]:
        matches = _expand_metric_name(metric, ordered_keys, key_set)
        if matches:
            for match in matches:
                if match not in resolved_seen:
                    resolved.append(match)
                    resolved_seen.add(match)
        elif metric in explicit:
            unresolved_explicit.append(metric)
    if unresolved_explicit:
        available = ", ".join(sorted(ordered_keys))
        raise ValueError(
            "unknown backtest.visible_metrics entries: "
            + ", ".join(unresolved_explicit)
            + f" | available metrics: {available}"
        )
    return resolved


def filter_analysis_for_visibility(analysis: dict, config: dict) -> VisibleAnalysis:
    trimmed = trim_analysis_aliases(analysis)
    visible_names = resolve_visible_metric_names(config, trimmed.keys())
    filtered = {key: trimmed[key] for key in visible_names}
    return VisibleAnalysis(
        analysis=filtered,
        shown_count=len(filtered),
        total_count=len(trimmed),
        resolved_metric_names=tuple(visible_names),
    )


def validate_visible_metrics_config(config: dict) -> None:
    _derived, explicit, visible_cfg = collect_visible_metric_requests(config)
    if visible_cfg in (None, []):
        return
    known_metrics = _known_visible_metric_names()
    unresolved_explicit = [
        metric for metric in explicit if not _metric_name_is_known(metric, known_metrics)
    ]
    if unresolved_explicit:
        available = ", ".join(sorted(known_metrics))
        raise ValueError(
            "unknown backtest.visible_metrics entries: "
            + ", ".join(unresolved_explicit)
            + f" | available metrics: {available}"
        )


def _normalize_visible_metrics_config(value) -> list[str]:
    if value == []:
        return []
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            "backtest.visible_metrics must be null, [], or a list/tuple/set of metric names"
        )
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("backtest.visible_metrics entries must be non-empty strings")
        normalized.append(item.strip())
    return normalized


def _expand_metric_name(metric: str, ordered_keys: Sequence[str], key_set: set[str]) -> list[str]:
    if metric in key_set:
        return [metric]
    if metric in SHARED_METRICS:
        return [metric] if metric in key_set else []
    if metric in CURRENCY_METRICS:
        matches = [key for key in ordered_keys if key in {f"{metric}_usd", f"{metric}_btc"}]
        if matches:
            return matches
    prefixed = [key for key in ordered_keys if key.startswith(f"{metric}_")]
    if prefixed:
        return prefixed
    return []


def _known_visible_metric_names() -> set[str]:
    known = set(CURRENCY_METRICS) | set(ANALYSIS_SHARED_KEYS)
    known |= {
        f"{metric}_{suffix}"
        for metric in CURRENCY_METRICS
        for suffix in ("usd", "btc")
    }
    return known


def _metric_name_is_known(metric: str, known_metrics: set[str]) -> bool:
    if metric in known_metrics:
        return True
    if metric.endswith(("_usd", "_btc")) and metric.rsplit("_", 1)[0] in CURRENCY_METRICS:
        return True
    return any(known_metric.startswith(f"{metric}_") for known_metric in known_metrics)
