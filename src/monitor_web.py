from __future__ import annotations

import asyncio
import logging
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from monitor_dev import (
    launch_relay_subprocess,
    relay_healthcheck,
    stop_relay_subprocess,
    wait_for_relay,
)


def build_dashboard_url(
    relay_url: str,
    *,
    exchange: Optional[str] = None,
    user: Optional[str] = None,
    focus_symbol: Optional[str] = None,
) -> str:
    parts = urlsplit(relay_url)
    query = {}
    if exchange:
        query["exchange"] = str(exchange)
    if user:
        query["user"] = str(user)
    if focus_symbol:
        query["symbol"] = str(focus_symbol)
    path = "/dashboard"
    return urlunsplit(
        (
            parts.scheme or "http",
            parts.netloc,
            path,
            urlencode(query),
            "",
        )
    )


async def run_monitor_web(
    *,
    relay_url: str,
    exchange: Optional[str] = None,
    user: Optional[str] = None,
    focus_symbol: Optional[str] = None,
    monitor_root: str = "monitor",
    relay_poll_interval_ms: int = 250,
    relay_queue_size: int = 1000,
    relay_log_file: str = "tmp/monitor_web/relay.log",
    repo_root: str = ".",
    open_browser: bool = False,
) -> None:
    parsed = urlsplit(relay_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    relay_process: Optional[subprocess.Popen] = None

    if relay_healthcheck(relay_url):
        logging.info("[monitor-web] using existing relay at %s", relay_url)
    else:
        if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            raise ValueError(
                f"cannot auto-launch relay for non-local relay host {host}; start it manually or use a local relay URL"
            )
        logging.info("[monitor-web] launching relay at %s", relay_url)
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

    dashboard_url = build_dashboard_url(
        relay_url,
        exchange=exchange,
        user=user,
        focus_symbol=focus_symbol,
    )
    logging.info("[monitor-web] dashboard available at %s", dashboard_url)
    print(dashboard_url, flush=True)

    if open_browser:
        logging.info("[monitor-web] opening default browser")
        webbrowser.open(dashboard_url, new=2)

    try:
        while True:
            await asyncio.sleep(3600.0)
    finally:
        stop_relay_subprocess(relay_process)
