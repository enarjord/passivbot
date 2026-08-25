import os
import subprocess
import sys
from pathlib import Path

import pytest

from logging_setup import STABLE_LOG_POINTER_HEADER
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


def test_monitor_tui_client_follows_stable_pointer_across_run_changes(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    stable_log = logs_dir / "bot.log"
    first_run_log = logs_dir / "first_run.log"
    second_run_log = logs_dir / "second_run.log"
    first_run_log.write_text("first run line\n", encoding="utf-8")
    stable_log.write_text(
        f"{STABLE_LOG_POINTER_HEADER}\n{first_run_log.resolve()}\n",
        encoding="utf-8",
    )

    client = MonitorTuiClient(
        relay_url="http://127.0.0.1:8765",
        log_file=str(stable_log),
        log_bootstrap_lines=2,
    )

    assert client.state.followed_log_file == str(stable_log)
    assert list(client.state.recent_log_lines) == ["first run line"]

    second_run_log.write_text("second run line\n", encoding="utf-8")
    stable_log.write_text(
        f"{STABLE_LOG_POINTER_HEADER}\n{second_run_log.resolve()}\n",
        encoding="utf-8",
    )
    client._poll_log_tail_once()

    assert list(client.state.recent_log_lines)[-1] == "second run line"


def test_monitor_tui_follows_auto_selected_pointer_file(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    archived_log = logs_dir / "20260406_140000_passivbot_live.log"
    stable_log = logs_dir / "bot.log"
    stable_log.write_text(
        f"{STABLE_LOG_POINTER_HEADER}\n{archived_log.resolve()}\n",
        encoding="utf-8",
    )
    archived_log.write_text("archived line\n", encoding="utf-8")
    archive_mtime = stable_log.stat().st_mtime + 10
    os.utime(archived_log, (archive_mtime, archive_mtime))

    selected_log = resolve_latest_log_file(logs_dir=str(logs_dir))
    assert selected_log == str(stable_log)

    client = MonitorTuiClient(
        relay_url="http://127.0.0.1:8765",
        log_file=selected_log,
        log_bootstrap_lines=2,
    )
    assert list(client.state.recent_log_lines) == ["archived line"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX stable aliases use symlinks")
def test_monitor_dev_prefers_stable_symlink_over_its_archive_target(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    archived_log = logs_dir / "20260406_140000_passivbot_live.log"
    stable_log = logs_dir / "1account.log"
    archived_log.write_text("archived line\n", encoding="utf-8")
    stable_log.symlink_to(archived_log.name)

    assert stable_log.stat().st_mtime == archived_log.stat().st_mtime
    assert stable_log.name < archived_log.name
    assert resolve_latest_log_file(logs_dir=str(logs_dir)) == str(stable_log)


def test_monitor_dev_prefers_newer_unrelated_archive_over_old_stable_pointer(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    older_archive = logs_dir / "older_archive.log"
    stable_log = logs_dir / "bot.log"
    newer_archive = logs_dir / "newer_archive.log"
    older_archive.write_text("older\n", encoding="utf-8")
    stable_log.write_text(
        f"{STABLE_LOG_POINTER_HEADER}\n{older_archive.resolve()}\n",
        encoding="utf-8",
    )
    newer_archive.write_text("newer\n", encoding="utf-8")
    newer_mtime = older_archive.stat().st_mtime + 10
    os.utime(newer_archive, (newer_mtime, newer_mtime))

    assert resolve_latest_log_file(logs_dir=str(logs_dir)) == str(newer_archive)


def test_monitor_dev_imports_without_unix_terminal_modules():
    code = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "sys.modules['termios'] = None; "
        "sys.modules['tty'] = None; "
        "import monitor_dev, monitor_tui; "
        "assert not monitor_tui.TERMINAL_CONTROL_SUPPORTED"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
