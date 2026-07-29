from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.scoring import ObjectiveSpec, default_scoring_weights
from pareto_compress import compress_candidates
from pareto_explorer import (
    ParetoCandidate,
    filter_candidates_with_limits,
    load_candidates,
)

_IGNORED_METADATA_FILENAMES = {"selection.json"}


@dataclass(frozen=True)
class StartingConfigSelection:
    pareto_dir: Path
    candidates: tuple[ParetoCandidate, ...]
    scoring_specs: tuple[ObjectiveSpec, ...]
    loaded_count: int
    filtered_count: int
    selected_count: int


def _require_metric_bearing_artifacts(
    pareto_dir: Path,
    candidates: Sequence[ParetoCandidate],
    source_path: str,
) -> None:
    candidate_paths = {candidate.path.resolve() for candidate in candidates}
    raw_source = Path(source_path).expanduser()
    json_paths = (
        [raw_source.resolve()]
        if raw_source.is_file()
        else sorted(pareto_dir.glob("*.json"))
    )
    unsupported = [
        path.name
        for path in json_paths
        if path.name not in _IGNORED_METADATA_FILENAMES
        and path.resolve() not in candidate_paths
    ]
    if unsupported:
        preview = ", ".join(unsupported[:5])
        suffix = (
            "" if len(unsupported) <= 5 else f", ... (+{len(unsupported) - 5} more)"
        )
        raise ValueError(
            "Starting-config metric preselection requires every JSON input to be a "
            "metric-bearing Pareto artifact; unsupported files: "
            f"{preview}{suffix}. Omit --filter-starting-configs and "
            "--compress-starting-configs to re-evaluate ordinary seed configs."
        )


def select_starting_config_artifacts(
    path: str,
    *,
    limits: Sequence[Mapping[str, Any]],
    aggregate_cfg: Mapping[str, Any] | None,
    filter_by_limits: bool,
    max_count: int | None,
    scenario_labels: Sequence[str] | None = None,
) -> StartingConfigSelection:
    logging.warning(
        "Starting-config metric preselection trusts stored Pareto metrics and objectives "
        "without verifying that their coins, exchanges, date range, scenarios, or backtest "
        "settings match this optimization run. Use it only for comparable artifacts; omit "
        "--filter-starting-configs and --compress-starting-configs to re-evaluate all seed configs."
    )
    try:
        pareto_dir, loaded, scoring_specs = load_candidates(path)
    except ValueError as exc:
        raise ValueError(
            "Starting-config metric preselection could not load complete metric-bearing "
            f"Pareto artifacts: {exc}. "
            "Omit --filter-starting-configs and --compress-starting-configs to re-evaluate "
            "ordinary seed configs."
        ) from exc
    _require_metric_bearing_artifacts(pareto_dir, loaded, path)

    candidates = list(loaded)
    active_limits: list[dict[str, Any]] = []
    if filter_by_limits:
        candidates, active_limits = filter_candidates_with_limits(
            candidates,
            limits,
            aggregate_cfg=aggregate_cfg,
            scenario_labels=scenario_labels,
            scoring_weights=default_scoring_weights(),
        )
        if not candidates:
            raise ValueError(
                "No starting configs remained after applying the optimizer's effective limits."
            )

    filtered_count = len(candidates)
    if max_count is not None and len(candidates) > int(max_count):
        members, _objective_ranges, _truncated_anchors = compress_candidates(
            candidates,
            scoring_specs,
            count=int(max_count),
            method="anchors-farthest",
        )
        candidates = [member.candidate for member in members]

    logging.info(
        "Starting-config metric preselection | loaded=%d | retained_after_limits=%d "
        "| selected=%d | active_limits=%d | max=%s",
        len(loaded),
        filtered_count,
        len(candidates),
        len(active_limits),
        str(max_count) if max_count is not None else "none",
    )
    return StartingConfigSelection(
        pareto_dir=pareto_dir,
        candidates=tuple(candidates),
        scoring_specs=tuple(scoring_specs),
        loaded_count=len(loaded),
        filtered_count=filtered_count,
        selected_count=len(candidates),
    )
