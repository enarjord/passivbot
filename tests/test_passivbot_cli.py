import os
import sys
from pathlib import Path
import asyncio

import pytest

from passivbot_cli import main as cli_main
from cli_utils import expand_help_all_argv, help_requested
import ohlcv_download


def test_root_help_lists_primary_commands(capsys):
    assert cli_main.main(["-h"]) == 0

    out = capsys.readouterr().out
    assert "live" in out
    assert "backtest" in out
    assert "optimize" in out
    assert "download" in out
    assert "tool" in out
    assert 'python3 -m pip install -e ".[full]"' in out


def test_dispatch_core_command_forwards_module_and_prog(monkeypatch):
    captured = {}
    original_argv = sys.argv[:]

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["optimize", "--suite", "y"]) == 0

    assert captured["module_name"] == "optimize"
    assert captured["argv"] == ["passivbot optimize", "--suite", "y"]
    assert captured["prog_env"] == "passivbot optimize"
    assert sys.argv == original_argv
    assert os.environ.get("PASSIVBOT_CLI_PROG") is None


def test_download_dispatch_forwards_new_module_and_prog(monkeypatch):
    captured = {}
    original_argv = sys.argv[:]

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["download", "configs/examples/default_trailing_grid_long_npos10.json"]) == 0

    assert captured["module_name"] == "ohlcv_download"
    assert captured["argv"] == [
        "passivbot download",
        "configs/examples/default_trailing_grid_long_npos10.json",
    ]
    assert captured["prog_env"] == "passivbot download"
    assert sys.argv == original_argv
    assert os.environ.get("PASSIVBOT_CLI_PROG") is None


def test_download_help_is_scoped_to_download_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["passivbot download", "-h"],
    )

    with pytest.raises(SystemExit) as exc:
        asyncio.run(ohlcv_download.main())

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--symbols" in out
    assert "--exchanges" in out
    assert "--start-date" in out
    assert "--end-date" in out
    assert "--user" not in out
    assert "--iters" not in out
    assert "--monitor.enabled" not in out


def test_help_subcommand_forwards_to_command_help(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: ["deap"])

    assert cli_main.main(["help", "backtest"]) == 0

    assert captured["module_name"] == "backtest"
    assert captured["argv"] == ["passivbot backtest", "-h"]


def test_help_all_request_skips_full_install_gate(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: ["aiohttp"])

    assert cli_main.main(["backtest", "--help-all"]) == 0

    assert captured["module_name"] == "backtest"
    assert captured["argv"] == ["passivbot backtest", "--help-all"]


def test_tool_help_lists_supported_tools(capsys):
    assert cli_main.main(["tool", "-h"]) == 0

    out = capsys.readouterr().out
    assert "monitor-dev" in out
    assert "monitor-relay" in out
    assert "monitor-web" in out
    assert "monitor-tui" in out
    assert "pareto" in out
    assert "pareto-dash" in out
    assert "pareto-explorer" in out
    assert "streamline-json" in out
    assert "verify-hlcvs-data" in out
    assert "requires full install" in out


def test_console_main_reexecs_into_active_virtualenv(monkeypatch, tmp_path):
    venv_prefix = tmp_path / "venv"
    python_path = venv_prefix / "bin" / "python"
    script_path = venv_prefix / "bin" / "passivbot"
    script_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    script_path.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_prefix))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv(cli_main.ENV_REEXEC_GUARD_ENV, raising=False)
    monkeypatch.delenv(cli_main.ENV_MISMATCH_IGNORE_ENV, raising=False)
    monkeypatch.setattr(cli_main.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(cli_main.sys, "argv", ["passivbot", "backtest", "cfg.json"])

    captured = {}

    def fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.os, "execv", fake_execv)

    with pytest.raises(SystemExit) as exc:
        cli_main.console_main()

    assert exc.value.code == 0
    assert captured["path"] == str(python_path.resolve())
    assert captured["argv"] == [str(python_path.resolve()), str(script_path.resolve()), "backtest", "cfg.json"]
    assert os.environ.get(cli_main.ENV_REEXEC_GUARD_ENV) == "1"


def test_console_main_fails_loudly_when_active_env_has_no_passivbot(monkeypatch, tmp_path):
    venv_prefix = tmp_path / "venv"
    (venv_prefix / "bin").mkdir(parents=True)

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_prefix))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv(cli_main.ENV_REEXEC_GUARD_ENV, raising=False)
    monkeypatch.delenv(cli_main.ENV_MISMATCH_IGNORE_ENV, raising=False)
    monkeypatch.setattr(cli_main.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(cli_main.sys, "argv", ["passivbot", "backtest", "cfg.json"])

    with pytest.raises(SystemExit) as exc:
        cli_main.console_main()

    message = str(exc.value)
    assert "active environment mismatch" in message
    assert str(venv_prefix.resolve()) in message
    assert "/usr/bin/python3" in message
    assert f"{venv_prefix / 'bin' / 'passivbot'}" in message
    assert f"{venv_prefix / 'bin' / 'python'} -m pip install -e \".[full]\"" in message


def test_console_main_accepts_symlinked_virtualenv_python(monkeypatch, tmp_path):
    venv_prefix = tmp_path / "venv"
    (venv_prefix / "bin").mkdir(parents=True)

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_prefix))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv(cli_main.ENV_REEXEC_GUARD_ENV, raising=False)
    monkeypatch.delenv(cli_main.ENV_MISMATCH_IGNORE_ENV, raising=False)
    monkeypatch.setattr(cli_main.sys, "executable", str(venv_prefix / "bin" / "python"))
    monkeypatch.setattr(cli_main.sys, "argv", ["passivbot", "-h"])
    monkeypatch.setattr(cli_main, "main", lambda argv=None: 0)

    with pytest.raises(SystemExit) as exc:
        cli_main.console_main()

    assert exc.value.code == 0


def test_console_main_accepts_venv_with_base_interpreter_realpath(monkeypatch, tmp_path):
    venv_prefix = tmp_path / "venv"
    python_path = venv_prefix / "bin" / "python"
    script_path = venv_prefix / "bin" / "passivbot"
    script_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    script_path.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_prefix))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv(cli_main.ENV_REEXEC_GUARD_ENV, raising=False)
    monkeypatch.delenv(cli_main.ENV_MISMATCH_IGNORE_ENV, raising=False)
    monkeypatch.setattr(cli_main.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(cli_main.sys, "prefix", str(venv_prefix))
    monkeypatch.setattr(cli_main.sys, "exec_prefix", str(venv_prefix))
    monkeypatch.setattr(cli_main.sys, "argv", [str(script_path), "-h"])
    monkeypatch.setattr(cli_main, "main", lambda argv=None: 0)

    with pytest.raises(SystemExit) as exc:
        cli_main.console_main()

    assert exc.value.code == 0


def test_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "pareto-dash", "--data-root", "optimize_results"]) == 0

    assert captured["module_name"] == "tools.pareto_dash"
    assert captured["argv"] == [
        "passivbot tool pareto-dash",
        "--data-root",
        "optimize_results",
    ]
    assert captured["prog_env"] == "passivbot tool pareto-dash"


def test_pareto_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "pareto", "optimize_results/example/pareto", "-m", "knee"]) == 0

    assert captured["module_name"] == "tools.pareto_explorer"
    assert captured["argv"] == [
        "passivbot tool pareto",
        "optimize_results/example/pareto",
        "-m",
        "knee",
    ]
    assert captured["prog_env"] == "passivbot tool pareto"


def test_monitor_relay_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "monitor-relay", "--port", "9000"]) == 0

    assert captured["module_name"] == "tools.monitor_relay"
    assert captured["argv"] == ["passivbot tool monitor-relay", "--port", "9000"]
    assert captured["prog_env"] == "passivbot tool monitor-relay"


def test_monitor_dev_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "monitor-dev", "--exchange", "bitget"]) == 0

    assert captured["module_name"] == "tools.monitor_dev"
    assert captured["argv"] == ["passivbot tool monitor-dev", "--exchange", "bitget"]
    assert captured["prog_env"] == "passivbot tool monitor-dev"


def test_monitor_web_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "monitor-web", "--open-browser"]) == 0

    assert captured["module_name"] == "tools.monitor_web"
    assert captured["argv"] == ["passivbot tool monitor-web", "--open-browser"]
    assert captured["prog_env"] == "passivbot tool monitor-web"


def test_monitor_tui_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "monitor-tui", "--focus-symbol", "BTC"]) == 0

    assert captured["module_name"] == "tools.monitor_tui"
    assert captured["argv"] == ["passivbot tool monitor-tui", "--focus-symbol", "BTC"]
    assert captured["prog_env"] == "passivbot tool monitor-tui"


def test_unknown_command_exits_with_error():
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["unknown"])

    assert exc.value.code == 2


def test_full_install_hint_is_shown_for_missing_optional_dependency(monkeypatch, capsys):
    error = ModuleNotFoundError("No module named 'deap'")
    error.name = "deap"

    def fake_invoke_module_main(module_name):
        raise error

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["optimize", "--iters", "10"]) == 2

    captured = capsys.readouterr()
    assert 'python3 -m pip install -e ".[full]"' in captured.err
    assert "passivbot optimize requires the full Passivbot install." in captured.err


def test_requires_full_command_fails_immediately_without_full_install(monkeypatch, capsys):
    def fail_if_called(module_name):
        raise AssertionError("_invoke_module_main should not be called without the full install")

    monkeypatch.setattr(cli_main, "_invoke_module_main", fail_if_called)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: ["aiohttp"])

    assert cli_main.main(["backtest", "-s", "XMR"]) == 2

    captured = capsys.readouterr()
    assert 'python3 -m pip install -e ".[full]"' in captured.err
    assert "passivbot backtest requires the full Passivbot install." in captured.err


def test_run_module_falls_back_to_runpy_when_module_has_no_main(monkeypatch):
    captured = {}

    class ModuleWithoutMain:
        pass

    def fake_import_module(module_name):
        captured["module_name"] = module_name
        return ModuleWithoutMain()

    def fake_run_module(module_name, run_name):
        captured["runpy_module_name"] = module_name
        captured["run_name"] = run_name
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(cli_main.runpy, "run_module", fake_run_module)

    assert cli_main._run_module("tools.generate_mcap_list", "passivbot tool generate-mcap-list", []) == 0

    assert captured["module_name"] == "tools.generate_mcap_list"
    assert captured["runpy_module_name"] == "tools.generate_mcap_list"
    assert captured["run_name"] == "__main__"


def test_invoke_module_main_runs_async_entrypoint(monkeypatch):
    class AsyncModule:
        async def main(self):
            return 7

    monkeypatch.setattr(cli_main.importlib, "import_module", lambda module_name: AsyncModule())

    assert cli_main._invoke_module_main("optimize") == (True, 7)


def test_expand_help_all_argv_appends_help_when_needed():
    assert expand_help_all_argv(["--help-all"]) == ["--help-all", "--help"]
    assert expand_help_all_argv(["--help-all", "-s", "XMR"]) == ["--help-all", "-s", "XMR", "--help"]


def test_expand_help_all_argv_preserves_explicit_help():
    assert expand_help_all_argv(["--help-all", "-h"]) == ["--help-all", "-h"]


def test_help_requested_treats_help_all_as_help():
    assert help_requested(["--help-all"]) is True
    assert help_requested(["-h"]) is True
    assert help_requested(["--help"]) is True
    assert help_requested(["--iters", "10"]) is False


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
