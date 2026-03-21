import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_tool_help(relative_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_monitor_relay_tool_help_runs_without_circular_import():
    result = _run_tool_help("src/tools/monitor_relay.py")
    assert result.returncode == 0, result.stderr
    assert "Serve read-only Passivbot monitor snapshots and live streams." in result.stdout


def test_monitor_tui_tool_help_runs_without_circular_import():
    result = _run_tool_help("src/tools/monitor_tui.py")
    assert result.returncode == 0, result.stderr
    assert "Minimal terminal dashboard for the Passivbot monitor relay." in result.stdout
    assert "--focus-symbol" in result.stdout


def test_monitor_dev_tool_help_runs_without_import_errors():
    result = _run_tool_help("src/tools/monitor_dev.py")
    assert result.returncode == 0, result.stderr
    assert "Launch the monitor relay if needed" in result.stdout
    assert "bot-log" in result.stdout
    assert "--focus-symbol" in result.stdout
