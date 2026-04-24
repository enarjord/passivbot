from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.merge_paretos import build_merged_configs, main, _load_fronts


SCORING = [
    {"metric": "metric_a", "goal": "max"},
    {"metric": "metric_b", "goal": "min"},
]


def _side_config(side: str, value: float, *, enabled: bool) -> dict:
    return {
        "n_positions": 1.0 if enabled else 0.0,
        "total_wallet_exposure_limit": 1.0 if enabled else 0.0,
        f"{side}_marker": value,
    }


def _write_candidate(
    pareto_dir: Path,
    name: str,
    *,
    enabled_side: str,
    marker: float,
    metric_a: float,
    metric_b: float,
    bounds: dict,
) -> None:
    payload = {
        "bot": {
            "common": {"shared_flag": True},
            "long": _side_config("long", marker, enabled=enabled_side == "long"),
            "short": _side_config("short", marker, enabled=enabled_side == "short"),
        },
        "optimize": {
            "scoring": SCORING,
            "bounds": bounds,
        },
        "metrics": {
            "objectives": {"metric_a": metric_a, "metric_b": metric_b},
            "stats": {
                "metric_a": {"mean": metric_a},
                "metric_b": {"mean": metric_b},
            },
        },
    }
    with open(pareto_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _make_front(tmp_path: Path, name: str, *, enabled_side: str) -> Path:
    pareto_dir = tmp_path / name / "pareto"
    pareto_dir.mkdir(parents=True)
    side_bounds = {
        "long": {
            "long_entry_initial_qty_pct": [0.10, 0.50, 0.02],
            "shared_param": [1.0, 2.0, 0.1],
        },
        "short": {
            "short_entry_initial_qty_pct": [0.20, 0.80, 0.04],
            "shared_param": [0.5, 3.0, 0.2],
        },
    }[enabled_side]
    values = [
        ("balanced", 0.55, 0.45),
        ("best_a", 0.90, 0.80),
        ("best_b", 0.20, 0.10),
        ("extra", 0.45, 0.35),
        ("extra_2", 0.35, 0.65),
        ("extra_3", 0.62, 0.70),
        ("extra_4", 0.25, 0.55),
        ("extra_5", 0.72, 0.60),
    ]
    for idx, (label, metric_a, metric_b) in enumerate(values):
        bounds = dict(side_bounds)
        if enabled_side == "long" and idx == 2:
            bounds["long_entry_initial_qty_pct"] = [0.05, 0.65, 0.01]
        if enabled_side == "short" and idx == 1:
            bounds["short_entry_initial_qty_pct"] = [0.15, 0.90, 0.02]
        _write_candidate(
            pareto_dir,
            label,
            enabled_side=enabled_side,
            marker=float(idx + 1),
            metric_a=metric_a,
            metric_b=metric_b,
            bounds=bounds,
        )
    return pareto_dir.parent


def test_merge_paretos_accepts_run_or_pareto_dirs_and_caps_outputs(tmp_path: Path):
    long_run = _make_front(tmp_path, "long_run", enabled_side="long")
    short_run = _make_front(tmp_path, "short_run", enabled_side="short")
    output_dir = tmp_path / "merged"

    exit_code = main([str(long_run), str(short_run / "pareto"), str(output_dir), "--max", "5"])

    assert exit_code == 0
    index = json.loads((output_dir / "index.json").read_text())
    assert index["count"] == 5
    assert index["stats"]["capped"] is True
    assert len(index["files"]) == 5

    merged = json.loads((output_dir / index["files"][0]).read_text())
    assert merged["bot"]["long"]["n_positions"] == 1.0
    assert merged["bot"]["short"]["n_positions"] == 1.0
    assert merged["bot"]["common"] == {"shared_flag": True}
    assert merged["merge_paretos"]["phase"] == "core"
    assert merged["optimize"]["bounds"]["long_entry_initial_qty_pct"] == [0.05, 0.65, 0.01]
    assert merged["optimize"]["bounds"]["short_entry_initial_qty_pct"] == [0.15, 0.9, 0.02]
    assert merged["optimize"]["bounds"]["shared_param"] == [0.5, 3, 0.1]


def test_merge_paretos_fills_after_core_until_max(tmp_path: Path):
    long_run = _make_front(tmp_path, "long_run", enabled_side="long")
    short_run = _make_front(tmp_path, "short_run", enabled_side="short")
    fronts = _load_fronts([long_run, short_run])

    configs, stats = build_merged_configs(fronts, max_outputs=20)

    assert len(configs) == 20
    assert stats.core_pair_added > 0
    assert stats.fill_pair_added > 0
    assert {config["merge_paretos"]["phase"] for config in configs} == {"core", "fill"}


def test_merge_paretos_fails_without_enabled_side(tmp_path: Path):
    long_run = _make_front(tmp_path, "long_run", enabled_side="long")
    fronts = _load_fronts([long_run, long_run])

    with pytest.raises(ValueError, match="No enabled short"):
        build_merged_configs(fronts, max_outputs=10)
