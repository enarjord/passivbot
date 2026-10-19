"""Saved result provenance must not become authored configuration authority."""

from copy import deepcopy
import json
import shlex

import pytest

from config.load import load_input_config, load_prepared_config, prepare_config
from config.result_records import extract_result_config
from config.schema import get_template_config
from pareto_explorer import build_parser, run_from_args
from tools.clean_config import cleanup_config


def result_record():
    config = get_template_config()
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    config["backtest"].update(start_date="2024-01-01", end_date="2024-01-02")
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    config["backtest"]["coins"] = {"combined": ["BTC"]}
    config["metrics"] = {"objectives": {"adg_strategy_eq": 0.001}}
    config["optimizer_evaluation_contract"] = {"version": 5}
    return config


@pytest.mark.parametrize("wrapped", [False, True])
def test_saved_results_load_like_clean_exports_without_changing_sources(tmp_path, wrapped):
    record = result_record()
    source = {"config": record, "analysis": {}, "timestamp": 123} if wrapped else record
    path = tmp_path / "result.json"
    original = json.dumps(source)
    path.write_text(original)

    loaded = load_prepared_config(str(path), verbose=False, log_info=False)
    expected = prepare_config(
        cleanup_config(source, base_config_path=str(path)),
        base_config_path=str(path),
        verbose=False,
    )

    for section in ("bot", "live", "coin_overrides", "backtest", "optimize"):
        assert loaded[section] == expected[section]
    assert "metrics" not in loaded
    assert "optimizer_evaluation_contract" not in loaded
    assert "coins" not in loaded["backtest"]
    assert path.read_text() == original


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("section", [None, "backtest", "live", "bot"])
def test_result_extraction_does_not_hide_unknown_authored_fields(tmp_path, wrapped, section):
    record = result_record()
    target = record if section is None else record[section]
    target["unknown_field"] = 1
    source = {"config": record, "analysis": {}} if wrapped else record
    path = tmp_path / "result.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="unknown configuration field"):
        load_prepared_config(str(path), verbose=False, log_info=False)


def test_authored_dataset_membership_stays_invalid_without_result_metadata(tmp_path):
    config = get_template_config()
    config["backtest"]["coins"] = {"combined": ["BTC"]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"config.backtest.coins is an unknown"):
        load_prepared_config(str(path), verbose=False, log_info=False)


@pytest.mark.parametrize("marker", [
    "analysis", "metrics", "suite_metrics", "optimizer_anchor",
    "optimizer_evaluation_contract",
])
def test_known_result_formats_extract_without_mutating_input(marker):
    source = get_template_config()
    source[marker] = {}
    source["backtest"]["coins"] = {"combined": ["BTC"]}
    original = deepcopy(source)

    extracted = extract_result_config(source)

    assert extracted == get_template_config()
    assert source == original


@pytest.mark.parametrize("version", [None, "v8.6.0", "v10.0.0"])
def test_result_records_still_require_current_schema(tmp_path, version):
    record = result_record()
    record["config_version"] = version
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"config": record, "analysis": {}}))

    with pytest.raises(ValueError, match="config.config_version"):
        load_prepared_config(str(path), verbose=False, log_info=False)


def test_result_input_snapshots_preserve_config_before_cli_overrides(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result_record()))
    source, base_path, snapshot = load_input_config(str(path), log_info=False)
    original = deepcopy(source)
    source["backtest"]["starting_balance"] = 1234.0
    prepared = prepare_config(
        source, base_config_path=base_path, raw_snapshot=snapshot, verbose=False,
    )

    assert snapshot == original
    assert prepared["_raw"] == original
    assert prepared["_raw_effective"]["backtest"]["starting_balance"] == 1234.0
    assert "metrics" not in snapshot


def test_backtest_command_printed_by_pareto_loads_selected_result(tmp_path, capsys):
    pareto = tmp_path / "pareto"
    pareto.mkdir()
    path = pareto / "candidate.json"
    path.write_text(json.dumps(result_record()))
    original = path.read_bytes()
    selected = run_from_args(
        build_parser().parse_args([str(pareto), "--method", "ideal"])
    )
    output = capsys.readouterr().out
    command = next(
        line.removeprefix("Backtest command: ")
        for line in output.splitlines() if line.startswith("Backtest command: ")
    )
    executable, subcommand, selected_path = shlex.split(command)

    assert (executable, subcommand) == ("passivbot", "backtest")
    assert selected.candidate.path == path
    loaded = load_prepared_config(selected_path, verbose=False, log_info=False)
    assert loaded["live"]["approved_coins"] == {"long": ["BTC"], "short": []}
    assert path.read_bytes() == original
