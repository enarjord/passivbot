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
                for key in ("metric_a", "metric_b", "metric_c")
            ]},
            "metrics": {"objectives": dict(zip(
                ("metric_a", "metric_b", "metric_c"), values,
            )), "stats": {"fills_gap_p95_hours": {
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
    result, output = run(front, capsys, "-o", "metric_a,metric_b", "--show-top", "10")
    assert result.candidate.path.stem == "a"
    assert output["loaded_count"] == output["retained_count"] == 4
    assert output["active_front_count"] == 2
    assert result.details["selected_utilities"] == {"metric_a": 1.0, "metric_b": 0.0}
    assert {row["file"] for row in output["top_candidates"]} == {"a.json", "b.json"}
    assert before == {p.name: p.read_bytes() for p in front.glob("*.json")}


def test_full_objective_front_stays_intact(front, capsys):
    _, output = run(front, capsys)
    assert output["loaded_count"] == output["active_front_count"] == 4


def test_limits_and_export_keep_separate_post_limit_pool(front, capsys, tmp_path):
    export = tmp_path.parent / (tmp_path.name + "-export")
    _, output = run(front, capsys, "-o", "metric_a,metric_b", "-l", "metric_a>=0",
                    "-f", str(export))
    assert output["loaded_count"] == 4
    assert output["retained_count"] == 3
    assert output["active_front_count"] == 2
    assert output["saved_filtered"]["count"] == 3
    assert (export / "d.json").exists()


@pytest.mark.parametrize("method", ["ideal", "utility", "knee", "outranking"])
def test_all_selectors_receive_active_front(front, capsys, method):
    _, output = run(front, capsys, "-m", method, "-o", "metric_a,metric_b")
    assert output["active_front_count"] == 2


def test_reference_targets_define_active_objectives(front, capsys):
    _, output = run(front, capsys, "-m", "reference", "--target", "metric_a=1",
                    "--target", "metric_b=1")
    assert output["active_front_count"] == 2
    assert output["selection_count"] == 4
    assert output["normalization_scope"] == "post_limit_pool"


def test_reference_keeps_dominated_exact_target(front, capsys):
    result, output = run(front, capsys, "-m", "reference", "--target", "metric_a=0.5",
                         "--target", "metric_b=0.5")
    assert result.candidate.path.stem == "d"
    assert output["active_front_count"] == 2
    assert output["selection_count"] == 4
    assert output["selected"]["score"] == pytest.approx(0.0)


@pytest.mark.parametrize("method", ["ideal", "utility", "outranking", "reference"])
def test_zero_weight_metrics_do_not_affect_front_or_normalization(front, capsys, method):
    flags = ["-m", method, "--weight", "metric_c=0"]
    if method == "reference":
        flags += ["--target", "metric_a=1", "--target", "metric_b=1", "--target", "metric_c=1"]
    first, output = run(front, capsys, *flags)
    path = front / "c.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["objectives"]["metric_c"] = -1000.0
    path.write_text(json.dumps(payload))
    second, changed = run(front, capsys, *flags)
    assert output["active_front_count"] == changed["active_front_count"] == 2
    assert first.candidate.path.name == second.candidate.path.name
    assert first.score == second.score
    assert "metric_c" not in first.objective_values


def test_all_zero_weights_fail(front, capsys):
    with pytest.raises(ValueError, match="At least one weight"):
        run(front, capsys, "-o", "metric_a", "--weight", "metric_a=0")


def test_reference_preserves_exact_target_for_min_objective(front, capsys):
    result, output = run(front, capsys, "-m", "reference", "--target", "fills_gap_p95_hours=102")
    assert result.candidate.path.stem == "a"
    assert output["active_front_count"] == 1
    assert output["selection_count"] == 4


def test_reference_text_distinguishes_front_and_ranked_pool(front, capsys):
    run_from_args(build_parser().parse_args([
        str(front), "-m", "reference", "--target", "metric_a=0.5", "--target", "metric_b=0.5",
    ]))
    output = capsys.readouterr().out
    assert "| Active front          | 2" in output
    assert "| Ranked candidates     | 4" in output
    assert "post-limit pool (reference targets)" in output


def test_lexicographic_priority_defines_active_objectives(front, capsys):
    result, output = run(front, capsys, "-m", "lexicographic", "--priority", "metric_b")
    assert output["active_front_count"] == 1
    assert result.candidate.path.stem == "b"


def test_non_scoring_min_metric_and_duplicate_vectors(front, capsys):
    original = (front / "c.json").read_bytes()
    (front / "z_duplicate.json").write_bytes(original)
    result, output = run(front, capsys, "-o", "fills_gap_p95_hours")
    assert output["loaded_count"] == 5
    assert output["active_front_count"] == 1
    assert result.candidate.path.stem == "c"


def test_text_reports_three_counts(front, capsys):
    run_from_args(build_parser().parse_args([str(front), "-o", "metric_a,metric_b"]))
    output = capsys.readouterr().out
    assert "| Loaded candidates     | 4" in output
    assert "| Retained after limits | 4" in output
    assert "| Active front          | 2" in output
