"""Offline command integration for explicit directional policy refactoring."""

from copy import deepcopy
import json

import pytest

from config.load import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from passivbot_cli.main import main as cli_main
from released_v8_fixtures import without_metric_selection
from tools.clean_config import cleanup_config, render_source
from tools.migrate_config_v9 import migrate


def policy():
    return {
        "config_version": "v9.0.0",
        "live": {},
        "bot": {
            "global": {"universe": {"approved_coins": ["BTC"]}},
            "long": {"enabled": True, "risk": {
                "max_positions": 3, "total_wallet_exposure_limit_ratio": 1.5,
            }},
            "short": {"enabled": True, "risk": {
                "max_positions": 1, "total_wallet_exposure_limit_ratio": 0.0,
            }, "unstuck": {"close_fraction": 0.15}},
        },
        "optimize": {"bounds": {
            "long": {"unstuck": {"close_fraction": [0.02, 0.06]}},
            "short": {"risk": {"total_wallet_exposure_limit_ratio": [0.0, 0.0]}},
        }},
        "backtest": {"scenarios": [{"label": "two", "overrides": {
            "bot.long.risk.max_positions": 2,
        }}]},
    }


@pytest.mark.parametrize("mode", ["full", "live", "backtest", "optimize", "compact", "resolved"])
def test_cleanup_optional_global_layout_keeps_omitted_side_out_of_final_export(mode):
    source = policy()
    original = deepcopy(source)
    report = {}
    result = cleanup_config(source, mode=mode, disabled_sides="omit",
                            bot_layout="global", reshape_report=report)
    assert source == original
    assert result["bot"]["long"] == {"enabled": True}
    assert "short" not in result["bot"]
    assert "enabled" not in result["bot"]["global"]
    assert report["changes"] and report["warnings"]
    before, after = (prepare_config(c, live_only=True, verbose=False) for c in (source, result))
    assert before["bot"]["long"] == after["bot"]["long"]
    assert after["bot"]["short"]["enabled"] is False
    assert cleanup_config(result, mode=mode, disabled_sides="omit", bot_layout="global") == result
    if "optimize" in result:
        assert "short" not in result["optimize"]["bounds"]
        assert result["optimize"]["bounds"]["long"] == source["optimize"]["bounds"]["long"]


@pytest.mark.parametrize("mode", ["full", "compact", "resolved"])
def test_cleanup_side_layout_does_not_rehydrate_global(mode):
    result = cleanup_config(policy(), mode=mode, disabled_sides="omit", bot_layout="side")
    assert not result["bot"].get("global")
    assert "short" not in result["bot"]
    assert result["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]


def test_cleanup_disable_retains_dormant_profile():
    source = policy()
    result = cleanup_config(source, disabled_sides="disable")
    assert result["bot"]["short"]["enabled"] is False
    assert result["bot"]["short"]["unstuck"] == source["bot"]["short"]["unstuck"]


def test_live_projection_audits_optimizer_reachability_before_dropping_optimize():
    source = policy()
    source["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="short"):
        cleanup_config(source, mode="live", disabled_sides="omit")


def test_cli_reports_refactoring_and_preserves_input(tmp_path, capsys):
    source, destination = tmp_path / "source.json", tmp_path / "output.json"
    source.write_text(json.dumps(policy()))
    original = source.read_bytes()
    assert cli_main(["tool", "clean-config", str(source), str(destination),
                     "--disabled-sides", "omit", "--bot-layout", "global"]) == 0
    output = json.loads(destination.read_text())
    assert output["bot"]["long"] == {"enabled": True}
    assert "short" not in output["bot"]
    assert source.read_bytes() == original
    captured = capsys.readouterr()
    assert "Refactor:" in captured.err and "Warning:" in captured.err


def test_dry_run_reports_without_writing(tmp_path, capsys):
    source, destination = tmp_path / "source.json", tmp_path / "output.json"
    source.write_text(json.dumps(policy()))
    assert cli_main(["tool", "clean-config", str(source), str(destination),
                     "--disabled-sides", "omit", "--dry-run"]) == 0
    assert not destination.exists()
    assert "Refactor:" in capsys.readouterr().err


def test_batch_refactoring_conflict_does_not_publish_earlier_files(tmp_path):
    source, destination = tmp_path / "inputs", tmp_path / "outputs"
    source.mkdir()
    (source / "a.json").write_text(json.dumps(policy()))
    conflicting = policy()
    conflicting["backtest"]["scenarios"][0]["overrides"]["bot.short.risk.total_wallet_exposure_limit_ratio"] = 1.0
    (source / "b.json").write_text(json.dumps(conflicting))
    assert cli_main(["tool", "clean-config", str(source), str(destination),
                     "--disabled-sides", "omit"]) == 2
    assert not destination.exists()


def test_format_rejects_semantic_options_before_loading_input(tmp_path, capsys):
    assert cli_main(["tool", "clean-config", str(tmp_path / "missing.json"), "--dry-run",
                     "--mode", "format", "--bot-layout", "global"]) == 2
    assert "--mode format" in capsys.readouterr().err
    with pytest.raises(ValueError, match="format"):
        render_source('{}', source=tmp_path / "input.json", mode="format", disabled_sides="omit")


def test_preserve_flags_keep_existing_rendering_exact(tmp_path):
    source = tmp_path / "source.json"
    text = json.dumps(policy())
    for mode in ("full", "compact", "resolved", "format"):
        assert render_source(text, source=source, mode=mode) == render_source(
            text, source=source, mode=mode, disabled_sides="preserve", bot_layout="preserve")


@pytest.mark.parametrize("interface", ["programmatic", "cli"])
def test_migration_optional_flags_produce_lean_profile_and_report_removals(tmp_path, interface):
    source = without_metric_selection(get_released_v8_defaults())
    original = deepcopy(source)
    if interface == "programmatic":
        result = migrate(source, best_effort=True, disabled_sides="omit", bot_layout="global")
        output, changes, warnings = result.config, result.changes, result.warnings
    else:
        input_path, output_path, report_path = (tmp_path / name for name in ("input.json", "output.json", "report.json"))
        input_path.write_text(json.dumps(source))
        assert cli_main(["tool", "migrate-config-v9", str(input_path), str(output_path),
                         "--best-effort", "--disabled-sides", "omit", "--bot-layout", "global",
                         "--report", str(report_path)]) == 0
        output = json.loads(output_path.read_text())
        report = json.loads(report_path.read_text())
        changes, warnings = report["changes"], report["warnings"]
        assert json.loads(input_path.read_text()) == original
    assert source == original
    assert output["bot"]["long"] == {"enabled": True}
    assert "short" not in output["bot"]
    assert "short" not in output["optimize"]["bounds"]
    assert any(change["path"] == "bot.short" for change in changes)
    assert warnings
