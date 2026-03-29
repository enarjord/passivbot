import os
import sys
from pathlib import Path

import pytest

from passivbot_cli import main as cli_main
from cli_utils import expand_help_all_argv


def test_root_help_lists_primary_commands(capsys):
    assert cli_main.main(["-h"]) == 0

    out = capsys.readouterr().out
    assert "live" in out
    assert "backtest" in out
    assert "optimize" in out
    assert "download" in out
    assert "tool" in out
    assert 'pip install -e ".[full]"' in out


def test_dispatch_core_command_forwards_module_and_prog(monkeypatch):
    captured = {}
    original_argv = sys.argv[:]

    def fake_run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["optimize", "--suite", "y"]) == 0

    assert captured["module_name"] == "optimize"
    assert captured["run_name"] == "__main__"
    assert captured["argv"] == ["passivbot optimize", "--suite", "y"]
    assert captured["prog_env"] == "passivbot optimize"
    assert sys.argv == original_argv
    assert os.environ.get("PASSIVBOT_CLI_PROG") is None


def test_help_subcommand_forwards_to_command_help(monkeypatch):
    captured = {}

    def fake_run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: ["deap"])

    assert cli_main.main(["help", "backtest"]) == 0

    assert captured["module_name"] == "backtest"
    assert captured["argv"] == ["passivbot backtest", "-h"]


def test_tool_help_lists_supported_tools(capsys):
    assert cli_main.main(["tool", "-h"]) == 0

    out = capsys.readouterr().out
    assert "monitor-relay" in out
    assert "pareto-dash" in out
    assert "streamline-json" in out
    assert "verify-hlcvs-data" in out
    assert "requires full install" in out


def test_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "pareto-dash", "--data-root", "optimize_results"]) == 0

    assert captured["module_name"] == "tools.pareto_dash"
    assert captured["run_name"] == "__main__"
    assert captured["argv"] == [
        "passivbot tool pareto-dash",
        "--data-root",
        "optimize_results",
    ]
    assert captured["prog_env"] == "passivbot tool pareto-dash"


def test_monitor_relay_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "monitor-relay", "--port", "9000"]) == 0

    assert captured["module_name"] == "tools.monitor_relay"
    assert captured["run_name"] == "__main__"
    assert captured["argv"] == ["passivbot tool monitor-relay", "--port", "9000"]
    assert captured["prog_env"] == "passivbot tool monitor-relay"


def test_unknown_command_exits_with_error():
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["unknown"])

    assert exc.value.code == 2


def test_full_install_hint_is_shown_for_missing_optional_dependency(monkeypatch, capsys):
    error = ModuleNotFoundError("No module named 'deap'")
    error.name = "deap"

    def fake_run_module(module_name, run_name):
        raise error

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["optimize", "--iters", "10"]) == 2

    captured = capsys.readouterr()
    assert 'pip install -e ".[full]"' in captured.err
    assert "passivbot optimize requires the full Passivbot install." in captured.err


def test_requires_full_command_fails_immediately_without_full_install(monkeypatch, capsys):
    def fail_if_called(module_name, run_name):
        raise AssertionError("run_module should not be called without the full install")

    monkeypatch.setattr(cli_main.runpy, "run_module", fail_if_called)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: ["aiohttp"])

    assert cli_main.main(["backtest", "-s", "XMR"]) == 2

    captured = capsys.readouterr()
    assert 'pip install -e ".[full]"' in captured.err
    assert "passivbot backtest requires the full Passivbot install." in captured.err


def test_expand_help_all_argv_appends_help_when_needed():
    assert expand_help_all_argv(["--help-all"]) == ["--help-all", "--help"]
    assert expand_help_all_argv(["--help-all", "-s", "XMR"]) == ["--help-all", "-s", "XMR", "--help"]


def test_expand_help_all_argv_preserves_explicit_help():
    assert expand_help_all_argv(["--help-all", "-h"]) == ["--help-all", "-h"]


def _configure_real_cli_module_test(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SKIP_RUST_COMPILE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))


@pytest.mark.parametrize(
    ("argv", "expected_text"),
    [
        (
            [
                "live",
                "--live.approved_coins",
                "BTC,ETH",
                "--live_ignored_coins",
                "DOGE",
                "--live.minimum_coin_age_days",
                "5",
                "--live.filter_by_min_effective_cost",
                "y",
                "--live.hedge_mode",
                "n",
                "--live_leverage",
                "3",
                "-h",
            ],
            "usage: passivbot live [config_path] [options]",
        ),
        (
            [
                "backtest",
                "--backtest.exchanges",
                "bybit",
                "--backtest_start_date",
                "2025",
                "--backtest.end_date",
                "now",
                "--backtest_starting_balance",
                "1000",
                "--live.approved_coins",
                "BTC",
                "-h",
            ],
            "usage: passivbot backtest [config_path] [options]",
        ),
        (
            [
                "optimize",
                "--optimize.iters",
                "10",
                "--optimize_n_cpus",
                "2",
                "--optimize.scoring",
                "adg_pnl",
                "--backtest_start_date",
                "2025",
                "--optimizer-backend",
                "deap",
                "-h",
            ],
            "usage: passivbot optimize [config_path] [options]",
        ),
    ],
)
def test_legacy_aliases_still_work_through_real_command_modules(
    monkeypatch, tmp_path, capsys, argv, expected_text
):
    _configure_real_cli_module_test(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(argv) == 0

    captured = capsys.readouterr()
    assert expected_text in captured.out
