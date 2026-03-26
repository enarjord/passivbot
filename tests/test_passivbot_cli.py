import os
import sys

import pytest

from passivbot_cli import main as cli_main


def test_root_help_lists_primary_commands(capsys):
    assert cli_main.main(["-h"]) == 0

    out = capsys.readouterr().out
    assert "live" in out
    assert "backtest" in out
    assert "optimize" in out
    assert "download" in out
    assert "tool" in out


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

    assert cli_main.main(["help", "backtest"]) == 0

    assert captured["module_name"] == "backtest"
    assert captured["argv"] == ["passivbot backtest", "-h"]


def test_tool_help_lists_supported_tools(capsys):
    assert cli_main.main(["tool", "-h"]) == 0

    out = capsys.readouterr().out
    assert "pareto-dash" in out
    assert "streamline-json" in out
    assert "verify-hlcvs-data" in out


def test_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_run_module(module_name, run_name):
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)

    assert cli_main.main(["tool", "pareto-dash", "--data-root", "optimize_results"]) == 0

    assert captured["module_name"] == "tools.pareto_dash"
    assert captured["run_name"] == "__main__"
    assert captured["argv"] == [
        "passivbot tool pareto-dash",
        "--data-root",
        "optimize_results",
    ]
    assert captured["prog_env"] == "passivbot tool pareto-dash"


def test_unknown_command_exits_with_error():
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["unknown"])

    assert exc.value.code == 2
