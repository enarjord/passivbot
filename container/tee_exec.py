from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("Usage: tee_exec.py <log_file> -- <command> [args...]")

    log_file = Path(sys.argv[1]).expanduser()
    command = sys.argv[3:]
    log_file.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _forward(sig, _frame):
        if process.poll() is None:
            process.send_signal(sig)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"=== passivbot container log start pid={os.getpid()} ===\n")
        handle.write(f"Command: {' '.join(command)}\n")
        handle.flush()
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
