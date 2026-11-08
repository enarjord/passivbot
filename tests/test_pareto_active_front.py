import json

import pytest

from pareto_explorer import build_parser, run_from_args


@pytest.fixture
def front(tmp_path):
    for name, values in {
        "a": [1.0, 0.7, 0.0],
        "b": [0.8, 1.0, 0.0],
        "c": [-100.0, 0.9, 1.0],
        "d": [0.5, 0.5, 2.0],
    }.items():
        payload = {
            "optimize": {"scoring": [
                {"metric": key, "goal": "max"}
                for key in ("cagr_fraction_strategy_equity", "omega_ratio_strategy_equity", "sortino_ratio_strategy_equity")
            ]},
            "metrics": {"definition_version": 2, "objectives": dict(zip(
                ("cagr_fraction_strategy_equity", "omega_ratio_strategy_equity", "sortino_ratio_strategy_equity"), values,
            )), "stats": {"pain_index_fraction_strategy_equity": {
                key: 0.0 if key == "std" else values[0] + 101.0
                for key in ("mean", "min", "max", "std", "median")
            }}},
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    return tmp_path


def run(front, capsys, *flags):
    result = run_from_args(build_parser().parse_args([str(front), *flags, "--json"]))
    return result, json.loads(capsys.readouterr().out)


def test_subset_pruning_precedes_normalization_and_ranking(front, capsys):
    before = {p.name: p.read_bytes() for p in front.glob("*.json")}
    result, output = run(front, capsys, "-o", "cagr_fraction_strategy_equity,omega_ratio_strategy_equity", "--show-top", "10")
    assert result.candidate.path.stem == "a"
    assert output["loaded_count"] == output["retained_count"] == 4
    assert output["active_front_count"] == 2
    assert result.details["selected_utilities"] == {"cagr_fraction_strategy_equity": 1.0, "omega_ratio_strategy_equity": 0.0}
    assert {row["file"] for row in output["top_candidates"]} == {"a.json", "b.json"}
    assert before == {p.name: p.read_bytes() for p in front.glob("*.json")}


def test_full_objective_front_stays_intact(front, capsys):
    _, output = run(front, capsys)
    assert output["loaded_count"] == output["active_front_count"] == 4


def test_limits_and_export_keep_separate_post_limit_pool(front, capsys, tmp_path):
    export = tmp_path.parent / (tmp_path.name + "-export")
    _, output = run(front, capsys, "-o", "cagr_fraction_strategy_equity,omega_ratio_strategy_equity", "-l", "cagr_fraction_strategy_equity>=0",
                    "-f", str(export))
    assert output["loaded_count"] == 4
    assert output["retained_count"] == 3
    assert output["active_front_count"] == 2
    assert output["saved_filtered"]["count"] == 3
    assert (export / "d.json").exists()


@pytest.mark.parametrize("method", ["ideal", "utility", "knee", "outranking"])
def test_all_selectors_receive_active_front(front, capsys, method):
    _, output = run(front, capsys, "-m", method, "-o", "cagr_fraction_strategy_equity,omega_ratio_strategy_equity")
    assert output["active_front_count"] == 2


def test_reference_targets_define_active_objectives(front, capsys):
    _, output = run(front, capsys, "-m", "reference", "--target", "cagr_fraction_strategy_equity=1",
                    "--target", "omega_ratio_strategy_equity=1")
    assert output["active_front_count"] == 2
    assert output["selection_count"] == 4
    assert output["normalization_scope"] == "post_limit_pool"


def test_reference_keeps_dominated_exact_target(front, capsys):
    result, output = run(front, capsys, "-m", "reference", "--target", "cagr_fraction_strategy_equity=0.5",
                         "--target", "omega_ratio_strategy_equity=0.5")
    assert result.candidate.path.stem == "d"
    assert output["active_front_count"] == 2
    assert output["selection_count"] == 4
    assert output["selected"]["score"] == pytest.approx(0.0)


@pytest.mark.parametrize("method", ["ideal", "utility", "outranking", "reference"])
def test_zero_weight_metrics_do_not_affect_front_or_normalization(front, capsys, method):
    flags = ["-m", method, "--weight", "sortino_ratio_strategy_equity=0"]
    if method == "reference":
        flags += ["--target", "cagr_fraction_strategy_equity=1", "--target", "omega_ratio_strategy_equity=1", "--target", "sortino_ratio_strategy_equity=1"]
    first, output = run(front, capsys, *flags)
    path = front / "c.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["objectives"]["sortino_ratio_strategy_equity"] = -1000.0
    path.write_text(json.dumps(payload))
    second, changed = run(front, capsys, *flags)
    assert output["active_front_count"] == changed["active_front_count"] == 2
    assert first.candidate.path.name == second.candidate.path.name
    assert first.score == second.score
    assert "sortino_ratio_strategy_equity" not in first.objective_values


def test_all_zero_weights_fail(front, capsys):
    with pytest.raises(ValueError, match="At least one weight"):
        run(front, capsys, "-o", "cagr_fraction_strategy_equity", "--weight", "cagr_fraction_strategy_equity=0")


def test_reference_preserves_exact_target_for_min_objective(front, capsys):
    result, output = run(front, capsys, "-m", "reference", "--target", "pain_index_fraction_strategy_equity=102")
    assert result.candidate.path.stem == "a"
    assert output["active_front_count"] == 1
    assert output["selection_count"] == 4


def test_reference_text_distinguishes_front_and_ranked_pool(front, capsys):
    run_from_args(build_parser().parse_args([
        str(front), "-m", "reference", "--target", "cagr_fraction_strategy_equity=0.5", "--target", "omega_ratio_strategy_equity=0.5",
    ]))
    output = capsys.readouterr().out
    assert "| Active front          | 2" in output
    assert "| Ranked candidates     | 4" in output
    assert "post-limit pool (reference targets)" in output


def test_lexicographic_priority_defines_active_objectives(front, capsys):
    result, output = run(front, capsys, "-m", "lexicographic", "--priority", "omega_ratio_strategy_equity")
    assert output["active_front_count"] == 1
    assert result.candidate.path.stem == "b"


def test_non_scoring_min_cagr_fraction_strategy_equitynd_duplicate_vectors(front, capsys):
    original = (front / "c.json").read_bytes()
    (front / "z_duplicate.json").write_bytes(original)
    result, output = run(front, capsys, "-o", "pain_index_fraction_strategy_equity")
    assert output["loaded_count"] == 5
    assert output["active_front_count"] == 1
    assert result.candidate.path.stem == "c"


@pytest.mark.parametrize(
    "saved_goal, expected_goal, winner",
    [(None, "min", "short"), ("max", "max", "long")],
)
def test_requested_peak_recovery_direction(tmp_path, capsys, saved_goal, expected_goal, winner):
    growth = "cagr_fraction_strategy_equity_ew"
    ulcer = "ulcer_index_fraction_strategy_equity"
    recovery = "peak_recovery_max_days_strategy_equity"
    scoring = [{"metric": growth, "goal": "max"}, {"metric": ulcer, "goal": "min"}]
    if saved_goal is not None:
        scoring.append({"metric": recovery, "goal": saved_goal})
    for name, days in (("short", 10.0), ("long", 50.0)):
        objectives = {growth: 0.2, ulcer: 0.1}
        if saved_goal is not None:
            objectives[recovery] = days
        payload = {
            "optimize": {"scoring": scoring},
            "metrics": {
                "definition_version": 2,
                "objectives": objectives,
                "stats": {recovery: {"mean": days}},
            },
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))

    result, record = run(tmp_path, capsys, "-o", f"{growth},{ulcer},{recovery}")

    assert record["loaded_count"] == 2
    assert record["active_front_count"] == 1
    assert record["objectives"][-1]["goal"] == expected_goal
    assert result.candidate.path.stem == winner


def test_text_reports_three_counts(front, capsys):
    run_from_args(build_parser().parse_args([str(front), "-o", "cagr_fraction_strategy_equity,omega_ratio_strategy_equity"]))
    output = capsys.readouterr().out
    assert "| Loaded candidates     | 4" in output
    assert "| Retained after limits | 4" in output
    assert "| Active front          | 2" in output


def test_versioned_record_identifies_loaded_bytes_and_reports_raw_changes(front, capsys):
    import hashlib

    _, record = run(front, capsys, "--show-top", "4")
    assert record["schema_version"] == 1
    assert record["metric_definition_version"] == 2
    assert record["settings"]["scaling"] == "atan_front_minmax"
    winner = record["selected"]
    assert winner["file_sha256"] == hashlib.sha256((front / winner["file"]).read_bytes()).hexdigest()
    assert record["top_candidates"][0]["file"] == winner["file"]
    runner_up = record["top_candidates"][1]
    for metric, value in runner_up["objectives"].items():
        assert runner_up["objective_deltas"][metric] == pytest.approx(
            value - winner["objectives"][metric]
        )
    assert runner_up["score_margin"] == pytest.approx(winner["score"] - runner_up["score"])
    assert record["sensitivity"]["check_count"] == 6

    # A byte change leaves the parsed objectives alone but changes the source identity.
    path = front / winner["file"]
    path.write_bytes(path.read_bytes() + b"\n")
    _, changed = run(front, capsys)
    assert changed["selected"]["file"] == winner["file"]
    assert changed["selected"]["file_sha256"] != winner["file_sha256"]
    assert changed["candidate_set_sha256"] != record["candidate_set_sha256"]


def test_aspiration_uses_active_front_and_exports_target_probes(front, capsys):
    cagr = "cagr_fraction_strategy_equity"
    omega = "omega_ratio_strategy_equity"
    _, record = run(
        front, capsys,
        "--method", "aspiration",
        "--target", f"{cagr}=0.5", "--target", f"{omega}=0.5",
        "--scale", f"{cagr}=1", "--scale", f"{omega}=1",
        "--sensitivity-weight-fraction", "0",
        "--sensitivity-target-scales", "0.5",
    )
    assert record["settings"]["scaling"] == "fixed_raw"
    assert record["active_front_count"] == record["selection_count"] == 2
    assert record["selected"]["file"] != "d.json"
    assert {check["kind"] for check in record["sensitivity"]["checks"]} == {"target_step"}
    assert record["sensitivity"]["check_count"] == 4
    assert "secondary_score" in record["top_candidates"][0]
