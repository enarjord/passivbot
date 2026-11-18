from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.merge_paretos import build_merged_configs, main, _load_fronts
from tools.merge_paretos import _side_is_enabled


SCORING = [
    {"metric": "omega_ratio_strategy_equity", "goal": "max"},
    {"metric": "ulcer_index_fraction_strategy_equity", "goal": "min"},
]


def _metric_stats(value: float) -> dict:
    return {"mean": value, "min": value, "max": value, "std": 0.0, "median": value}


def _side_config(side: str, value: float, *, enabled: bool) -> dict:
    return {
        "n_positions": 1.0 if enabled else 0.0,
        "total_wallet_exposure_limit": 1.0 if enabled else 0.0,
        f"{side}_marker": value,
    }


def _nested_side_config(side: str, value: float, *, enabled: bool) -> dict:
    return {
        "risk": {
            "n_positions": 1.0,
            "total_wallet_exposure_limit_ratio": 1.0 if enabled else 0.0,
        },
        "strategy": {"marker": value},
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
            "definition_version": 2,
            "objectives": {"omega_ratio_strategy_equity": metric_a, "ulcer_index_fraction_strategy_equity": metric_b},
            "stats": {
                "omega_ratio_strategy_equity": _metric_stats(metric_a),
                "ulcer_index_fraction_strategy_equity": _metric_stats(metric_b),
            },
        },
    }
    with open(pareto_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_nested_candidate(
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
            "long": _nested_side_config("long", marker, enabled=enabled_side == "long"),
            "short": _nested_side_config("short", marker, enabled=enabled_side == "short"),
        },
        "optimize": {
            "scoring": SCORING,
            "bounds": bounds,
        },
        "metrics": {
            "definition_version": 2,
            "objectives": {"omega_ratio_strategy_equity": metric_a, "ulcer_index_fraction_strategy_equity": metric_b},
            "stats": {
                "omega_ratio_strategy_equity": _metric_stats(metric_a),
                "ulcer_index_fraction_strategy_equity": _metric_stats(metric_b),
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


def _make_nested_front(tmp_path: Path, name: str, *, enabled_side: str) -> Path:
    pareto_dir = tmp_path / name / "pareto"
    pareto_dir.mkdir(parents=True)
    side_bounds = {
        "long": {
            "long": {
                "risk": {
                    "n_positions": [1.0, 1.0, 1.0],
                    "total_wallet_exposure_limit_ratio": [0.75, 1.50, 0.01],
                },
                "strategy": {"initial_qty_pct": [0.10, 0.50, 0.02]},
            },
            "short": {
                "risk": {
                    "n_positions": [1.0, 1.0, 1.0],
                    "total_wallet_exposure_limit_ratio": [0.0, 0.0, 0.01],
                },
                "strategy": {"initial_qty_pct": [0.20, 0.80, 0.04]},
            },
        },
        "short": {
            "long": {
                "risk": {
                    "n_positions": [1.0, 1.0, 1.0],
                    "total_wallet_exposure_limit_ratio": [0.0, 0.0, 0.01],
                },
                "strategy": {"initial_qty_pct": [0.10, 0.50, 0.02]},
            },
            "short": {
                "risk": {
                    "n_positions": [1.0, 1.0, 1.0],
                    "total_wallet_exposure_limit_ratio": [0.65, 1.40, 0.01],
                },
                "strategy": {"initial_qty_pct": [0.20, 0.80, 0.04]},
            },
        },
    }[enabled_side]
    for idx, (label, metric_a, metric_b) in enumerate(
        [("balanced", 0.55, 0.45), ("best_a", 0.90, 0.80), ("best_b", 0.20, 0.10)]
    ):
        _write_nested_candidate(
            pareto_dir,
            label,
            enabled_side=enabled_side,
            marker=float(idx + 1),
            metric_a=metric_a,
            metric_b=metric_b,
            bounds=side_bounds,
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
    rendered = (output_dir / index["files"][0]).read_text()
    assert '"long_entry_initial_qty_pct": [0.05, 0.65, 0.01]' in rendered
    assert rendered.endswith("\n")
    assert (output_dir / "index.json").read_text() == json.dumps(index, indent=2, sort_keys=True) + "\n"


def test_merge_paretos_supports_nested_v8_side_risk_and_bounds(tmp_path: Path):
    long_run = _make_nested_front(tmp_path, "long_run", enabled_side="long")
    short_run = _make_nested_front(tmp_path, "short_run", enabled_side="short")
    output_dir = tmp_path / "merged"

    exit_code = main([str(long_run), str(short_run / "pareto"), str(output_dir), "--max", "5"])

    assert exit_code == 0
    index = json.loads((output_dir / "index.json").read_text())
    assert index["count"] == 5

    merged = json.loads((output_dir / index["files"][0]).read_text())
    assert merged["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.0
    assert merged["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.0
    assert merged["optimize"]["bounds"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == [
        0.75,
        1.5,
        0.01,
    ]
    assert merged["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] == [
        0.65,
        1.4,
        0.01,
    ]


def test_merge_paretos_supports_canonical_v9_risk_and_bound_union(tmp_path: Path):
    fronts = [_make_nested_front(tmp_path, side, enabled_side=side) for side in ("long", "short")]
    originals = {}
    for front in fronts:
        for path in (front / "pareto").glob("*.json"):
            entry = json.loads(path.read_text())
            for side in ("long", "short"):
                risk = entry["bot"][side]["risk"]
                risk["max_positions"] = risk.pop("n_positions")
                bounds = entry["optimize"]["bounds"][side]["risk"]
                bounds["max_positions"] = bounds.pop("n_positions")
            path.write_text(json.dumps(entry))
            originals[path] = path.read_bytes()
    output = tmp_path / "canonical-merged"
    assert main([str(p) for p in fronts] + [str(output), "--max", "5"]) == 0
    index = json.loads((output / "index.json").read_text())
    assert index["count"] == 5
    for filename in index["files"]:
        merged = json.loads((output / filename).read_text())
        for side, exposure_bounds in [("long", [.75, 1.5, .01]), ("short", [.65, 1.4, .01])]:
            assert merged["bot"][side]["risk"]["max_positions"] == 1.0
            assert "n_positions" not in merged["bot"][side]["risk"]
            bounds = merged["optimize"]["bounds"][side]["risk"]
            assert bounds["max_positions"] == [1, 1, 1]
            assert "n_positions" not in bounds
            assert bounds["total_wallet_exposure_limit_ratio"] == exposure_bounds
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def _write_current_v9_candidate(pareto_dir, side, initial_qty):
    from config.load import prepare_config
    from config.schema import get_template_config
    from optimization.callback import build_pymoo_record_entry

    source = get_template_config()
    bot = source["bot"]
    for pside in ("long", "short"):
        bot[pside] = {"enabled": pside == side}
    strategy = bot["global"]["strategy"]["trailing_martingale"]
    strategy["entry"]["initial_qty_fraction"] = initial_qty
    bot["global"]["risk"].update(max_positions=2, total_wallet_exposure_limit_ratio=1.0)
    source["optimize"].update(
        scoring=SCORING,
        bounds={"global": {"strategy": {"trailing_martingale": {
            "entry": {"initial_qty_fraction": [0.01, 0.1]},
        }}}},
    )
    prepared = prepare_config(source, verbose=False)
    record = build_pymoo_record_entry(
        vector=[], template=prepared, prepared_config=prepared, overrides_fn=None, build_config_fn=None,
        metrics={
            "definition_version": 2,
            "objectives": {"omega_ratio_strategy_equity": 2.0, "ulcer_index_fraction_strategy_equity": 0.2},
        },
    )
    assert set(record["config"]["bot"][side]) == {"enabled"}
    pareto_dir.mkdir(parents=True)
    path = pareto_dir / f"{side}.json"
    path.write_text(json.dumps(record))
    return path


def test_merge_paretos_current_results_are_usable_optimizer_seeds(tmp_path):
    from config.load import load_prepared_config
    from optimization.bounds import Bound
    from optimize import configs_to_individuals_streaming, get_starting_configs

    long_path = _write_current_v9_candidate(tmp_path / "long" / "pareto", "long", 0.031)
    short_path = _write_current_v9_candidate(tmp_path / "short" / "pareto", "short", 0.078)
    output = tmp_path / "merged"
    assert main([str(long_path.parent), str(short_path.parent), str(output)]) == 0
    index = json.loads((output / "index.json").read_text())
    assert index["count"] == 1
    path = output / index["files"][0]
    record = json.loads(path.read_text())
    assert record["result_version"] == 1
    provenance = record["provenance"]["merge_paretos"]
    assert provenance["long"]["source"] == str(long_path)
    assert provenance["short"]["source"] == str(short_path)
    assert record["config"]["config_version"] == "v9.0.0"
    assert "merge_paretos" not in record["config"]
    bounds = record["config"]["optimize"]["bounds"]
    assert bounds["global"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_fraction"] == [0.01, 0.1]

    loaded = load_prepared_config(str(path), allow_result_records=True, verbose=False)
    [seed] = get_starting_configs(str(output))
    for side, expected in (("long", 0.031), ("short", 0.078)):
        assert loaded["bot"][side]["enabled"] is True
        for source in (loaded, seed):
            assert source["bot"][side]["strategy"]["trailing_martingale"]["entry"]["initial_qty_fraction"] == expected
    key_paths = [(side, ("bot", side, "strategy", "trailing_martingale", "entry", "initial_qty_fraction"))
                 for side in ("long", "short")]
    bounds = [Bound.from_config(side, [0.01, 0.1]) for side, _ in key_paths]
    assert configs_to_individuals_streaming([seed], bounds, sig_digits=6, key_paths=key_paths) == (
        [(0.031, 0.078)], 1,
    )


def test_merge_paretos_requires_consistent_schema_for_current_results(tmp_path):
    long_path = _write_current_v9_candidate(tmp_path / "long" / "pareto", "long", 0.031)
    old_short = _make_front(tmp_path, "old-short", enabled_side="short")
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="historical Pareto inputs"):
        build_merged_configs(_load_fronts([long_path.parent, old_short]), max_outputs=5)
    assert not output.exists()


@pytest.mark.parametrize("count,exposure,enabled", [(2, .5, True), (0, .5, False), (2, 0, False)])
def test_canonical_pareto_side_enablement(count, exposure, enabled):
    assert _side_is_enabled({"risk": {"max_positions": count, "total_wallet_exposure_limit_ratio": exposure}}) is enabled


@pytest.mark.parametrize("alias_patch", [
    {"n_positions": 0},
    {"total_wallet_exposure_limit": 0},
    {"risk": {"n_positions": 0}},
    {"risk": {"total_wallet_exposure_limit": 0}},
])
def test_canonical_pareto_policy_rejects_conflicting_historical_aliases(alias_patch):
    side = {"risk": {"max_positions": 2, "total_wallet_exposure_limit_ratio": .5}}
    for key, value in alias_patch.items():
        if key == "risk":
            side[key].update(value)
        else:
            side[key] = value
    with pytest.raises(ValueError, match="Conflicting"):
        _side_is_enabled(side)


def test_historical_pareto_reader_accepts_consistent_duplicate_policy_values():
    assert _side_is_enabled({
        "n_positions": 2, "total_wallet_exposure_limit": .5,
        "risk": {"max_positions": 2, "n_positions": 2, "total_wallet_exposure_limit_ratio": .5},
    })


@pytest.mark.parametrize("count,exposure,enabled", [(2, .5, True), (0, .5, False), (2, 0, False)])
def test_released_v8_grouped_pareto_side_enablement(count, exposure, enabled):
    assert _side_is_enabled({
        "risk": {"n_positions": count, "total_wallet_exposure_limit": exposure}
    }) is enabled


def test_historical_pareto_reader_keeps_current_exposure_authoritative():
    assert not _side_is_enabled({
        "risk": {
            "n_positions": 2,
            "total_wallet_exposure_limit_ratio": 0,
            "total_wallet_exposure_limit": .5,
        }
    })


def test_merge_paretos_preserves_released_v8_grouped_capacity(tmp_path: Path):
    fronts = [_make_nested_front(tmp_path, side, enabled_side=side) for side in ("long", "short")]
    originals = {}
    for front in fronts:
        for path in (front / "pareto").glob("*.json"):
            entry = json.loads(path.read_text())
            for side in ("long", "short"):
                risk = entry["bot"][side]["risk"]
                risk["total_wallet_exposure_limit"] = risk.pop("total_wallet_exposure_limit_ratio")
                bounds = entry["optimize"]["bounds"][side]["risk"]
                bounds["total_wallet_exposure_limit"] = bounds.pop("total_wallet_exposure_limit_ratio")
            path.write_text(json.dumps(entry))
            originals[path] = path.read_bytes()
    output = tmp_path / "released-v8-merged"
    assert main([str(p) for p in fronts] + [str(output), "--max", "5"]) == 0
    index = json.loads((output / "index.json").read_text())
    assert index["count"] == 5
    for filename in index["files"]:
        merged = json.loads((output / filename).read_text())
        for side in ("long", "short"):
            assert merged["bot"][side]["risk"]["total_wallet_exposure_limit"] == 1.0
            assert "total_wallet_exposure_limit_ratio" not in merged["bot"][side]["risk"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())


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
