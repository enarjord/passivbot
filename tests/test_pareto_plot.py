import json
from pathlib import Path

import pytest

from config.scoring import ObjectiveSpec
from pareto_explorer import load_candidates
from tools import pareto_plot
from passivbot_cli.main import main as cli_main


METRICS = ["adg_strategy_eq", "strategy_eq_underwater_pct_mean", "sortino_ratio_strategy_eq"]
SPECS = [ObjectiveSpec(metric=name, goal=goal) for name, goal in zip(METRICS, ["max", "min", "max"])]


def write_candidate(path, values=(0.02, 0.1, 2.0), *, layout="named"):
    objectives = dict(zip(METRICS, values))
    entry = {"optimize": {"scoring": [spec.to_config() for spec in SPECS]}}
    if layout == "named":
        entry["metrics"] = {"objectives": objectives}
    elif layout == "legacy":
        entry["metrics"] = {"objectives": {f"w_{i}": -value if spec.goal == "max" else value
                                          for i, (spec, value) in enumerate(zip(SPECS, values))}}
    elif layout == "suite":
        entry["suite_metrics"] = {"metrics": {name: {"aggregated": value,
                                                   "stats": {stat: value + 10 for stat in ("mean", "min", "max", "std", "median")},
                                                   "scenarios": {"example": value + 20}}
                                               for name, value in objectives.items()}}
    path.write_text(json.dumps(entry))


@pytest.fixture
def front(tmp_path):
    path = tmp_path / "run" / "pareto"
    path.mkdir(parents=True)
    # Includes dominated, coincident and constant-axis projections: keep every member.
    for name, values in [("a", (0.02, 0.1, 2.0)), ("b", (0.01, 0.2, 2.0)),
                         ("c", (0.02, 0.1, 2.0))]:
        write_candidate(path / f"{name}.json", values)
    (path / "selection.json").write_text('{"selected": ["a.json"]}')
    return path


def test_dataset_includes_objectives_and_preserves_members(front):
    _, candidates, specs = load_candidates(front)
    data = pareto_plot.build_dataset(candidates, specs)
    assert data["names"] == ["a.json", "b.json", "c.json"]
    metrics = {metric["key"]: metric for metric in data["metrics"]}
    assert metrics[METRICS[0]]["values"] == [0.02, 0.01, 0.02]
    assert metrics[METRICS[1]]["values"] == [0.1, 0.2, 0.1]
    assert metrics[METRICS[0]]["goal"] == "max"
    assert metrics[METRICS[1]]["goal"] == "min"
    assert pareto_plot.select_metrics([], data) == METRICS[:2]


@pytest.mark.parametrize("layout", ["named", "legacy", "suite"])
def test_formats_preserve_native_values_not_engine_sign_or_scenario_mean(tmp_path, layout):
    path = tmp_path / "candidate.json"
    write_candidate(path, layout=layout)
    _, candidates, specs = load_candidates(path)
    data = pareto_plot.build_dataset(candidates, specs)
    metrics = {metric["key"]: metric for metric in data["metrics"]}
    for name, value in zip(METRICS, [0.02, 0.1, 2.0]):
        assert metrics[name]["values"] == [value]
    if layout == "suite":
        assert metrics["stats.adg_strategy_eq.mean"]["values"] == [10.02]
        assert metrics["stats.adg_strategy_eq.std"]["goal"] is None


def test_requested_axis_order_and_aliases():
    data = {"metrics": [{"key": name, "count": 1} for name in ["adg_usd", METRICS[1]]]}
    assert pareto_plot.select_metrics([METRICS[1], "adg"], data) == [METRICS[1], "adg_usd"]
    with pytest.raises(ValueError, match="distinct"):
        pareto_plot.select_metrics(["adg", "adg_usd"], data)


@pytest.mark.parametrize("metrics", [[METRICS[0]], METRICS + ["other"], [METRICS[0]] * 2, ["unknown", METRICS[0]]])
def test_invalid_selection(metrics):
    with pytest.raises(ValueError):
        pareto_plot.select_metrics(metrics, {"metrics": [{"key": spec.metric, "count": 1} for spec in SPECS]})


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "bad"])
def test_bad_objectives_fail_without_output(tmp_path, bad):
    source = tmp_path / "candidate.json"
    write_candidate(source, (bad, 0.1, 2.0))
    output = tmp_path / "plot.html"
    with pytest.raises(SystemExit) as exc:
        pareto_plot.main([str(source), *METRICS[:2], "-o", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_cli_dispatch_offline_html_and_source_preservation(front, tmp_path):
    before = {p.name: p.read_bytes() for p in front.iterdir()}
    for dimensions in (2, 3):
        output = tmp_path / "plots" / f"{dimensions}d.html"
        assert cli_main(["tool", "pareto-plot", str(front.parent), *METRICS[:dimensions],
                         "--output", str(output)]) == 0
        page = output.read_text()
        assert "Plotly.react" in page
        assert "plotly.js v" in page
        assert '<script src=' not in page
        assert "a.json" in page
        assert "optimize" not in json.dumps(read_payload(page))  # no full config in export
    assert {p.name: p.read_bytes() for p in front.iterdir()} == before


def test_list_metrics_without_plot(front, monkeypatch, capsys):
    def no_plot(*args):
        pytest.fail("Listing metrics must not render a figure")
    monkeypatch.setattr(pareto_plot, "render_html", no_plot)
    assert pareto_plot.main([str(front), "--list-metrics"]) == 0
    assert f"{METRICS[1]} (min;" in capsys.readouterr().out


def test_output_overwrite_and_extension_guards(front, tmp_path):
    output = tmp_path / "plot.html"
    output.write_text("keep")
    args = [str(front), *METRICS[:2], "-o", str(output)]
    with pytest.raises(SystemExit):
        pareto_plot.main(args)
    assert output.read_text() == "keep"
    assert pareto_plot.main(args + ["--force"]) == 0
    source = front / "a.json"
    with pytest.raises(SystemExit):
        pareto_plot.main([str(front), *METRICS[:2], "-o", str(source), "--force"])
    alias = tmp_path / "source.html"
    alias.symlink_to(source)
    with pytest.raises(SystemExit):
        pareto_plot.main([str(front), *METRICS[:2], "-o", str(alias), "--force"])
    assert json.loads(source.read_text())["optimize"]


def test_missing_empty_and_malformed_inputs(tmp_path):
    for path in [tmp_path / "missing", tmp_path]:
        with pytest.raises(SystemExit):
            pareto_plot.main([str(path), *METRICS[:2]])
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(SystemExit):
        pareto_plot.main([str(tmp_path), *METRICS[:2]])


def test_browser_open_is_opt_in(front, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pareto_plot.webbrowser, "open", lambda uri: calls.append(uri) or True)
    monkeypatch.chdir(tmp_path)
    args = [str(front), *METRICS[:2]]
    assert pareto_plot.main(args) == 0
    assert not calls
    assert pareto_plot.main(args + ["--force", "--open"]) == 0
    assert calls == [(tmp_path / "pareto-plot-2d.html").as_uri()]


def read_payload(page):
    start = page.index('<script id="pareto-data" type="application/json">')
    return json.loads(page[start:].split(">", 1)[1].split("</script>", 1)[0])


def test_export_without_cli_axes_contains_all_metrics(front, tmp_path):
    output = tmp_path / "all.html"
    assert pareto_plot.main([str(front), "-o", str(output)]) == 0
    data = read_payload(output.read_text())
    assert data["selected"] == METRICS[:2]
    assert {metric["key"] for metric in data["metrics"]} == set(METRICS)


def test_non_objective_metrics_missing_values_and_explicit_statistics(front):
    from dataclasses import replace
    _, candidates, specs = load_candidates(front)
    candidates[0] = replace(candidates[0], aggregated_values={"extra": 9.0},
                            stats_flat={"extra_mean": 2.0, "extra_std": 3.0,
                                        "position_held_hours_mean_mean": 7.0,
                                        "unavailable_mean": float("nan")})
    data = pareto_plot.build_dataset(candidates, specs)
    metrics = {metric["key"]: metric for metric in data["metrics"]}
    assert metrics["extra"]["values"] == [9.0, None, None]
    assert metrics["extra"]["goal"] is None
    assert metrics["extra"]["count"] == 1
    assert metrics["stats.extra.mean"]["values"] == [2.0, None, None]
    assert metrics["stats.extra.std"]["goal"] is None
    assert metrics["position_held_hours_mean"]["values"] == [7.0, None, None]
    assert metrics["unavailable"]["count"] == 0
    assert metrics["unavailable"]["min"] is None
    assert pareto_plot.select_metrics([METRICS[0], "extra"], data) == [METRICS[0], "extra"]
    with pytest.raises(ValueError, match="unavailable"):
        pareto_plot.select_metrics([METRICS[0], "unavailable"], data)
    json.dumps(data, allow_nan=False)


def test_html_escapes_data_without_changing_values():
    hostile = '</script><script>alert("x")</script>'
    data = {"names": [hostile], "metrics": [{"key": hostile, "values": [1.0]}]}
    page = pareto_plot.render_html(data, [hostile, "safe"])
    assert hostile not in page
    assert read_payload(page)["names"] == [hostile]


def test_browser_logic(tmp_path):
    import shutil
    import subprocess
    from tools.pareto_plot_page import SCRIPT
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed only to test the embedded browser logic")
    script = tmp_path / "pareto-plot.cjs"
    script.write_text(SCRIPT)
    subprocess.run([node, str(Path(__file__).with_name("pareto_plot_browser_logic.cjs")), str(script)],
                   check=True, capture_output=True, text=True)


def test_statistics_inherit_explicit_scoring_direction(tmp_path):
    path = tmp_path / "candidate.json"
    write_candidate(path, layout="suite")
    _, candidates, specs = load_candidates(path)
    specs[0] = ObjectiveSpec(metric=METRICS[0], goal="min")
    data = pareto_plot.build_dataset(candidates, specs)
    metrics = {metric["key"]: metric for metric in data["metrics"]}
    assert metrics["stats.adg_strategy_eq.mean"]["goal"] == "min"
    assert metrics["stats.adg_strategy_eq.std"]["goal"] is None
