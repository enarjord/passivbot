#!/usr/bin/env python3
"""
sync_tar.py
~~~~~~~~~~~

Utility for archiving directories, transferring them via scp, and restoring
their contents. Designed to work symmetrically on local and remote hosts so
that you can push or pull directories by invoking the same helper on either
side of the connection.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
    return subprocess.run(cmd, check=check)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_archive_name(source: Path) -> str:
    base = source.name.rstrip("/")
    return f"{base}_{_timestamp()}.tar.gz"


def _contains_glob(text: str) -> bool:
    return any(ch in text for ch in "*?[")


def _safe_archive_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "archive"


def _resolve_pull_destination(destination: str, remote_base: str) -> Path:
    dest_path = Path(destination).resolve()
    if _contains_glob(dest_path.name):
        return dest_path.parent
    if dest_path.suffix and dest_path.name == remote_base:
        return dest_path.parent
    return dest_path


def _build_remote_tar_cmd(remote_tmp: str, remote_dir: str, remote_base: str) -> str:
    quoted_tmp = shlex.quote(remote_tmp)
    quoted_dir = shlex.quote(remote_dir)
    quoted_base = shlex.quote(remote_base)
    no_match_msg = shlex.quote(f"No remote matches found for pattern: {remote_base}")
    return (
        f"cd {quoted_dir} && "
        f"matches=$(find . -mindepth 1 -maxdepth 1 -name {quoted_base} -print) && "
        f"if [ -z \"$matches\" ]; then echo {no_match_msg} >&2; exit 2; fi && "
        f"find . -mindepth 1 -maxdepth 1 -name {quoted_base} -print0 | "
        f"tar --null -czf {quoted_tmp} --files-from -"
    )


def create_archive(source_path: Path, archive_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating archive {archive_path} from {source_path} ...")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(source_path, arcname=source_path.name)
    print("Archive created.")


def extract_archive(archive_path: Path, destination: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} into {destination} ...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(destination)
    print("Extraction complete.")


def scp_transfer(local_path: Path, remote: str, remote_path: str) -> None:
    _run(["scp", str(local_path), f"{remote}:{remote_path}"])


def scp_from_remote(remote: str, remote_path: str, local_path: Path) -> None:
    _run(["scp", f"{remote}:{remote_path}", str(local_path)])


def ssh_exec(remote: str, command: str, *, check: bool = True) -> None:
    _run(["ssh", remote, command], check=check)


def handle_push(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    if args.archive_name:
        archive_name = (
            args.archive_name
            if args.archive_name.endswith(".tar.gz")
            else f"{args.archive_name}.tar.gz"
        )
    else:
        archive_name = _default_archive_name(source)

    archive_path = Path(tempfile.gettempdir()) / archive_name
    create_archive(source, archive_path)

    remote_archive = args.remote_path.rstrip("/") + f"/{archive_name}"
    try:
        if args.remote_path:
            remote_dir_cmd = f"mkdir -p {shlex.quote(args.remote_path)}"
            ssh_exec(args.remote, remote_dir_cmd)
        scp_transfer(archive_path, args.remote, remote_archive)

        if args.remote_extract:
            remote_extract_cmd = (
                f"mkdir -p {shlex.quote(args.remote_path)} && "
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(args.remote_path)} && "
                f"rm -f {shlex.quote(remote_archive)}"
            )
            ssh_exec(args.remote, remote_extract_cmd)
            print("Remote extraction complete.")
        print("Transfer complete.")
    finally:
        if archive_path.exists():
            archive_path.unlink()
            print(f"Removed local archive {archive_path}")


def handle_pull(args: argparse.Namespace) -> None:
    remote_source = args.remote_source
    remote = args.remote

    if ":" in remote_source and not remote_source.startswith("/"):
        inferred_remote, path = remote_source.split(":", 1)
        remote_path = path
        if remote and remote != inferred_remote:
            raise ValueError(
                f"Remote specified twice with different values: '{remote}' vs '{inferred_remote}'"
            )
        remote = inferred_remote
    else:
        remote_path = remote_source

    if not remote:
        raise ValueError("Remote host must be specified via --remote or within remote_source.")

    remote_path = remote_path.rstrip("/")
    remote_dir = str(Path(remote_path).parent)
    remote_base = Path(remote_path).name

    archive_basename = f"{_safe_archive_component(remote_base)}_{_timestamp()}.tar.gz"
    remote_tmp = f"/tmp/{archive_basename}"
    local_dest = _resolve_pull_destination(args.destination, remote_base)
    local_dest.mkdir(parents=True, exist_ok=True)
    local_archive = local_dest / archive_basename

    tar_cmd = _build_remote_tar_cmd(remote_tmp, remote_dir, remote_base)
    ssh_exec(remote, tar_cmd)
    local_dest.mkdir(parents=True, exist_ok=True)

    try:
        scp_from_remote(remote, remote_tmp, local_archive)
    finally:
        ssh_exec(remote, f"rm -f {shlex.quote(remote_tmp)}")

    if args.extract:
        extract_archive(local_archive, local_dest)
        extracted_root = local_dest / remote_base
        if not _contains_glob(remote_base) and extracted_root.is_dir():
            for child in extracted_root.iterdir():
                target = local_dest / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child), target)
            extracted_root.rmdir()
        local_archive.unlink()
        print(f"Removed downloaded archive {local_archive}")
    else:
        print(f"Archive stored at {local_archive}")


def handle_extract(args: argparse.Namespace) -> None:
    archive = Path(args.archive).resolve()
    destination = Path(args.destination).resolve() if args.destination else archive.parent
    extract_archive(archive, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive and transfer files or directories via scp.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push = subparsers.add_parser(
        "push", help="Archive a local file or directory and push it to a remote host."
    )
    push.add_argument("source", help="Local source file or directory to archive and transfer.")
    push.add_argument("remote", help="Remote host (can be SSH alias, host, or user@host).")
    push.add_argument(
        "remote_path",
        nargs="?",
        default=".",
        help="Remote path to store the archive/extracted data (default: home directory).",
    )
    push.add_argument(
        "--remote-extract",
        action="store_true",
        help="Extract the archive on the remote host and remove the archive afterwards.",
    )
    push.add_argument(
        "--archive-name",
        help="Custom archive name (defaults to <dir>_<timestamp>.tar.gz).",
    )
    push.set_defaults(func=handle_push)

    pull = subparsers.add_parser(
        "pull", help="Archive a remote directory and pull it to the local machine."
    )
    pull.add_argument(
        "remote_source",
        help="Remote directory to archive (e.g. vps3:/opt/project or /opt/project if remote provided separately).",
    )
    pull.add_argument(
        "destination",
        nargs="?",
        default=".",
        help="Local directory to store/extract the archive (default: current directory).",
    )
    pull.add_argument(
        "--remote",
        help="Remote host (alias, host, or user@host). Optional if provided in remote_source.",
    )
    pull.add_argument(
        "--extract",
        action="store_true",
        help="Extract the downloaded archive locally and remove the archive afterwards.",
    )
    pull.set_defaults(func=handle_pull)

    extract = subparsers.add_parser("extract", help="Extract a local tar.gz archive.")
    extract.add_argument("archive", help="Local archive to extract.")
    extract.add_argument(
        "destination",
        nargs="?",
        help="Destination directory (default: archive directory).",
    )
    extract.set_defaults(func=handle_extract)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
