import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from monitor_dev import _relay_launch_env, resolve_latest_log_file, wait_for_relay
from monitor_tui import MonitorTuiClient


def test_resolve_latest_log_file_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "logs" / "explicit.log"
    explicit.parent.mkdir(parents=True, exist_ok=True)
    explicit.write_text("x", encoding="utf-8")

    resolved = resolve_latest_log_file(
        logs_dir=str(tmp_path / "logs"),
        explicit_log_file=str(explicit),
    )

    assert resolved == str(explicit)


def test_resolve_latest_log_file_picks_newest_log(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    older = logs_dir / "older.log"
    newer = logs_dir / "newer.log"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    older.touch()
    newer.touch()
    newer_mtime = newer.stat().st_mtime + 10
    older_mtime = older.stat().st_mtime
    Path(older).touch()
    Path(newer).touch()
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    resolved = resolve_latest_log_file(logs_dir=str(logs_dir))

    assert resolved == str(newer)


def test_resolve_latest_log_file_returns_none_when_missing(tmp_path):
    assert resolve_latest_log_file(logs_dir=str(tmp_path / "missing")) is None


def test_relay_launch_env_prepends_repo_src(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "existing:path")

    env = _relay_launch_env(repo_root=str(repo_root))

    assert env["PYTHONPATH"] == os.pathsep.join([str(src_root.resolve()), "existing:path"])


@pytest.mark.asyncio
async def test_wait_for_relay_reports_early_exit_with_log_excerpt(tmp_path):
    relay_log = tmp_path / "relay.log"
    relay_log.write_text("boom line 1\nboom line 2\n", encoding="utf-8")

    class DummyProcess:
        returncode = 7

        def poll(self):
            return self.returncode

    with pytest.raises(RuntimeError, match="relay exited early with code 7") as excinfo:
        await wait_for_relay(
            "http://127.0.0.1:8765",
            timeout_seconds=0.5,
            process=DummyProcess(),
            relay_log_file=str(relay_log),
        )

    assert "boom line 2" in str(excinfo.value)


def test_monitor_tui_client_bootstraps_and_polls_log_tail(tmp_path):
    log_file = tmp_path / "logs" / "bot.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("line one\nline two\n", encoding="utf-8")

    client = MonitorTuiClient(
        relay_url="http://127.0.0.1:8765",
        log_file=str(log_file),
        log_bootstrap_lines=2,
    )

    assert client.state.followed_log_file == str(log_file)
    assert list(client.state.recent_log_lines) == ["line one", "line two"]

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("line three\n")

    client._poll_log_tail_once()

    assert list(client.state.recent_log_lines)[-1] == "line three"


def test_monitor_dev_tool_help_runs_without_import_errors():
    result = subprocess.run(
        [sys.executable, "src/tools/monitor_dev.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--relay-url" in result.stdout
    assert "--log-file" in result.stdout
