from __future__ import annotations

import argparse
from pathlib import Path

import sync_tar


def test_resolve_pull_destination_uses_parent_for_globbed_destination(tmp_path):
    destination = tmp_path / "logs" / "20260318*"
    resolved = sync_tar._resolve_pull_destination(str(destination), "20260318*")
    assert resolved == (tmp_path / "logs").resolve()


def test_build_remote_tar_cmd_uses_find_for_patterns():
    cmd = sync_tar._build_remote_tar_cmd("/tmp/archive.tar.gz", "/root/passivbot/logs", "20260318*")
    assert "find . -mindepth 1 -maxdepth 1 -name '20260318*' -print0" in cmd
    assert "tar --null -czf /tmp/archive.tar.gz --files-from -" in cmd
    assert "No remote matches found for pattern: 20260318*" in cmd


def test_handle_pull_glob_archives_remote_matches_and_extracts_to_parent(tmp_path, monkeypatch):
    ssh_calls: list[tuple[str, str, bool]] = []
    scp_calls: list[tuple[str, str, Path]] = []
    extract_calls: list[tuple[Path, Path]] = []

    def fake_ssh_exec(remote: str, command: str, *, check: bool = True) -> None:
        ssh_calls.append((remote, command, check))

    def fake_scp_from_remote(remote: str, remote_path: str, local_path: Path) -> None:
        scp_calls.append((remote, remote_path, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake archive")

    def fake_extract_archive(archive_path: Path, destination: Path) -> None:
        extract_calls.append((archive_path, destination))

    monkeypatch.setattr(sync_tar, "ssh_exec", fake_ssh_exec)
    monkeypatch.setattr(sync_tar, "scp_from_remote", fake_scp_from_remote)
    monkeypatch.setattr(sync_tar, "extract_archive", fake_extract_archive)

    args = argparse.Namespace(
        remote_source="/root/passivbot/logs/20260318*",
        destination=str(tmp_path / "logs" / "20260318*"),
        remote="vps3",
        extract=True,
    )

    sync_tar.handle_pull(args)

    assert len(ssh_calls) == 2
    assert ssh_calls[0][0] == "vps3"
    assert "find . -mindepth 1 -maxdepth 1 -name '20260318*' -print0" in ssh_calls[0][1]
    assert ssh_calls[0][1].startswith("cd /root/passivbot/logs && ")
    assert ssh_calls[1][1].startswith("rm -f /tmp/")

    assert len(scp_calls) == 1
    local_archive = scp_calls[0][2]
    assert "*" not in local_archive.name
    assert local_archive.parent == (tmp_path / "logs").resolve()

    assert extract_calls == [(local_archive, (tmp_path / "logs").resolve())]
    assert not local_archive.exists()
