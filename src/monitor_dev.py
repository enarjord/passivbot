from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from monitor_tui import MonitorTuiClient


def resolve_latest_log_file(*, logs_dir: str = "logs", explicit_log_file: Optional[str] = None) -> Optional[str]:
    if explicit_log_file:
        return str(Path(explicit_log_file).expanduser())
    root = Path(logs_dir).expanduser()
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*.log") if path.is_file()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return str(latest)


def relay_healthcheck(relay_url: str, timeout_seconds: float = 1.0) -> bool:
    url = relay_url.rstrip("/") + "/health"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return int(response.status) == 200
    except (OSError, URLError):
        return False


def _relay_launch_env(*, repo_root: str) -> dict[str, str]:
    env = os.environ.copy()
    src_root = str((Path(repo_root) / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_root if not existing else os.pathsep.join([src_root, existing])
    )
    return env


def _read_relay_log_excerpt(relay_log_file: str, *, max_lines: int = 20) -> str:
    path = Path(relay_log_file)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    excerpt = "\n".join(lines[-max_lines:])
    return excerpt.strip()


def launch_relay_subprocess(
    *,
    repo_root: str,
    monitor_root: str,
    host: str,
    port: int,
    poll_interval_ms: int,
    queue_size: int,
    relay_log_file: str,
) -> subprocess.Popen:
    relay_log_path = Path(relay_log_file)
    relay_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(relay_log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.monitor_relay",
            "--monitor-root",
            monitor_root,
            "--host",
            host,
            "--port",
            str(port),
            "--poll-interval-ms",
            str(poll_interval_ms),
            "--queue-size",
            str(queue_size),
        ],
        cwd=repo_root,
        env=_relay_launch_env(repo_root=repo_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._passivbot_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def stop_relay_subprocess(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
    log_handle = getattr(process, "_passivbot_log_handle", None)
    if log_handle is not None:
        log_handle.close()


async def wait_for_relay(
    relay_url: str,
    *,
    timeout_seconds: float = 10.0,
    process: Optional[subprocess.Popen] = None,
    relay_log_file: Optional[str] = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if relay_healthcheck(relay_url, timeout_seconds=0.5):
            return
        if process is not None and process.poll() is not None:
            excerpt = _read_relay_log_excerpt(relay_log_file or "")
            details = f"relay exited early with code {process.returncode}"
            if excerpt:
                details += f"\nRecent relay log:\n{excerpt}"
            raise RuntimeError(details)
        await asyncio.sleep(0.2)
    raise RuntimeError(f"relay did not become healthy at {relay_url}/health within {timeout_seconds}s")


async def run_monitor_dev(
    *,
    relay_url: str,
    exchange: Optional[str] = None,
    user: Optional[str] = None,
    focus_symbol: Optional[str] = None,
    monitor_root: str = "monitor",
    logs_dir: str = "logs",
    explicit_log_file: Optional[str] = None,
    snapshot_refresh_seconds: float = 2.0,
    render_interval_ms: int = 250,
    log_poll_interval_ms: int = 500,
    log_bootstrap_lines: int = 12,
    relay_poll_interval_ms: int = 250,
    relay_queue_size: int = 1000,
    relay_log_file: str = "tmp/monitor_dev/relay.log",
    repo_root: str = ".",
) -> None:
    log_file = resolve_latest_log_file(logs_dir=logs_dir, explicit_log_file=explicit_log_file)
    parsed = urlsplit(relay_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if log_file:
        logging.info("[monitor-dev] following bot log %s", log_file)
    relay_process: Optional[subprocess.Popen] = None
    if relay_healthcheck(relay_url):
        logging.info("[monitor-dev] using existing relay at %s", relay_url)
    else:
        if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            raise ValueError(
                f"cannot auto-launch relay for non-local relay host {host}; start it manually or use a local relay URL"
            )
        logging.info("[monitor-dev] launching relay at %s", relay_url)
        relay_process = launch_relay_subprocess(
            repo_root=repo_root,
            monitor_root=monitor_root,
            host=host,
            port=port,
            poll_interval_ms=relay_poll_interval_ms,
            queue_size=relay_queue_size,
            relay_log_file=relay_log_file,
        )
        try:
            await wait_for_relay(
                relay_url,
                process=relay_process,
                relay_log_file=relay_log_file,
            )
        except (OSError, RuntimeError):
            stop_relay_subprocess(relay_process)
            raise
    try:
        client = MonitorTuiClient(
            relay_url=relay_url,
            exchange=exchange,
            user=user,
            focus_symbol=focus_symbol,
            snapshot_refresh_seconds=snapshot_refresh_seconds,
            render_interval_ms=render_interval_ms,
            log_file=log_file,
            log_poll_interval_ms=log_poll_interval_ms,
            log_bootstrap_lines=log_bootstrap_lines,
        )
        await client.run()
    finally:
        stop_relay_subprocess(relay_process)
