from pathlib import Path

from monitor_dev import resolve_latest_log_file


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
    import os

    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    resolved = resolve_latest_log_file(logs_dir=str(logs_dir))

    assert resolved == str(newer)


def test_resolve_latest_log_file_returns_none_when_missing(tmp_path):
    assert resolve_latest_log_file(logs_dir=str(tmp_path / "missing")) is None
