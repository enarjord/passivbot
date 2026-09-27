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


@pytest.mark.parametrize("dimensions", [2, 3])
def test_figure_preserves_coordinates_members_and_goals(front, dimensions):
    _, candidates, specs = load_candidates(front)
    selected = pareto_plot.select_metrics(METRICS[:dimensions], specs)
    fig = pareto_plot.build_figure(candidates, selected)
    trace = fig.data[0]
    assert trace.type == ("scatter" if dimensions == 2 else "scatter3d")
    assert list(trace.x) == [0.02, 0.01, 0.02]
    assert list(trace.y) == [0.1, 0.2, 0.1]
    assert list(trace.text) == ["a.json", "b.json", "c.json"]
    assert METRICS[0] in trace.hovertemplate
    if dimensions == 3:
        assert list(trace.z) == [2.0] * 3
        assert fig.layout.scene.dragmode == "orbit"
        assert "lower is better" in fig.layout.scene.yaxis.title.text
    else:
        assert "higher is better" in fig.layout.xaxis.title.text
        assert "lower is better" in fig.layout.yaxis.title.text


@pytest.mark.parametrize("layout", ["named", "legacy", "suite"])
def test_formats_plot_native_values_not_engine_sign_or_scenario_mean(tmp_path, layout):
    path = tmp_path / "candidate.json"
    write_candidate(path, layout=layout)
    _, candidates, specs = load_candidates(path)
    fig = pareto_plot.build_figure(candidates, specs)
    assert list(fig.data[0].x) == [0.02]
    assert list(fig.data[0].y) == [0.1]
    assert list(fig.data[0].z) == [2.0]


def test_requested_axis_order_and_aliases():
    specs = [ObjectiveSpec(metric="adg_usd", goal="max"), SPECS[1]]
    assert pareto_plot.select_metrics([METRICS[1], "adg"], specs) == specs[::-1]
    with pytest.raises(ValueError, match="distinct"):
        pareto_plot.select_metrics(["adg", "adg_usd"], specs)


@pytest.mark.parametrize("metrics", [[], [METRICS[0]], METRICS + ["other"], [METRICS[0]] * 2, ["unknown", METRICS[0]]])
def test_invalid_selection(metrics):
    with pytest.raises(ValueError):
        pareto_plot.select_metrics(metrics, SPECS)


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
        assert "Plotly.newPlot" in page
        assert "plotly.js v" in page
        assert '<script src=' not in page
        assert "a.json" in page
        assert "optimize" not in page.split('Plotly.newPlot')[-1]  # no full config in figure
    assert {p.name: p.read_bytes() for p in front.iterdir()} == before


def test_list_metrics_without_plot(front, monkeypatch, capsys):
    def no_plot(*args):
        pytest.fail("Listing metrics must not render a figure")
    monkeypatch.setattr(pareto_plot, "build_figure", no_plot)
    assert pareto_plot.main([str(front), "--list-metrics"]) == 0
    assert f"{METRICS[1]} (min)" in capsys.readouterr().out


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
