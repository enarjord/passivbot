"""
Capture memory diagnostics for optimizer runs into a single JSON file.

Examples
--------
Watch the newest running optimizer process whose command line contains ``src/optimize.py``::

    python3 src/tools/capture_optimize_memory.py --wait --output /tmp/opt_mem.json

Watch a specific optimizer process by PID::

    python3 src/tools/capture_optimize_memory.py --pid 12345 --output /tmp/opt_mem.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    rss_kb: int
    vsz_kb: int
    cpu_pct: float
    mem_pct: float
    elapsed_s: int
    args: str


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def collect_ps() -> list[ProcessInfo]:
    code, stdout, stderr = run_cmd(
        ["ps", "-eo", "pid=,ppid=,rss=,vsz=,%cpu=,%mem=,etimes=,args="]
    )
    if code != 0:
        raise RuntimeError(f"ps failed: {stderr.strip()}")
    rows: list[ProcessInfo] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, rss, vsz, pcpu, pmem, etimes, args = parts
        try:
            rows.append(
                ProcessInfo(
                    pid=int(pid),
                    ppid=int(ppid),
                    rss_kb=int(rss),
                    vsz_kb=int(vsz),
                    cpu_pct=float(pcpu),
                    mem_pct=float(pmem),
                    elapsed_s=int(etimes),
                    args=args,
                )
            )
        except ValueError:
            continue
    return rows


def find_target_pid(match: str) -> int | None:
    candidates = [
        row
        for row in collect_ps()
        if match in row.args and "capture_optimize_memory.py" not in row.args
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row.elapsed_s, row.pid), reverse=False)
    return candidates[0].pid


def build_descendant_tree(rows: list[ProcessInfo], root_pid: int) -> list[ProcessInfo]:
    by_ppid: dict[int, list[ProcessInfo]] = {}
    by_pid: dict[int, ProcessInfo] = {}
    for row in rows:
        by_pid[row.pid] = row
        by_ppid.setdefault(row.ppid, []).append(row)

    result: list[ProcessInfo] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        row = by_pid.get(pid)
        if row is not None:
            result.append(row)
        for child in by_ppid.get(pid, []):
            stack.append(child.pid)
    result.sort(key=lambda row: (row.ppid, row.pid))
    return result


def collect_free_bytes() -> dict[str, int] | None:
    code, stdout, _stderr = run_cmd(["free", "-b"])
    if code != 0:
        return None
    lines = [line.split() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    for row in lines:
        if row[0].lower().startswith("mem:") and len(row) >= 7:
            return {
                "total": int(row[1]),
                "used": int(row[2]),
                "free": int(row[3]),
                "shared": int(row[4]),
                "buff_cache": int(row[5]),
                "available": int(row[6]),
            }
    return None


def collect_proc_meminfo() -> dict[str, int] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    wanted = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "Shmem",
        "SwapTotal",
        "SwapFree",
    }
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key not in wanted:
            continue
        parts = value.strip().split()
        if not parts:
            continue
        try:
            kb = int(parts[0])
        except ValueError:
            continue
        out[key] = kb * 1024
    return out


def collect_df_shm() -> dict[str, int] | None:
    shm = Path("/dev/shm")
    if not shm.exists():
        return None
    usage = shutil.disk_usage(shm)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
    }


def collect_shm_listing(limit: int = 50) -> list[dict[str, Any]] | None:
    shm = Path("/dev/shm")
    if not shm.exists():
        return None
    entries = []
    for path in shm.iterdir():
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": path.name,
                "size_bytes": int(stat.st_size),
                "mtime": int(stat.st_mtime),
            }
        )
    entries.sort(key=lambda item: item["size_bytes"], reverse=True)
    return entries[:limit]


def make_sample(rows: list[ProcessInfo], root_pid: int, top_n_global: int) -> dict[str, Any]:
    tree = build_descendant_tree(rows, root_pid)
    total_rss_kb = sum(row.rss_kb for row in tree)
    total_vsz_kb = sum(row.vsz_kb for row in tree)
    top_global = sorted(rows, key=lambda row: row.rss_kb, reverse=True)[:top_n_global]
    return {
        "timestamp": utc_now_iso(),
        "target_pid": root_pid,
        "system_free_b": collect_free_bytes(),
        "proc_meminfo_b": collect_proc_meminfo(),
        "dev_shm_b": collect_df_shm(),
        "tree_totals": {
            "process_count": len(tree),
            "rss_kb": total_rss_kb,
            "vsz_kb": total_vsz_kb,
        },
        "process_tree": [asdict(row) for row in tree],
        "top_global_rss": [asdict(row) for row in top_global],
        "dev_shm_top_entries": collect_shm_listing(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture optimizer memory diagnostics to JSON.")
    parser.add_argument("--pid", type=int, default=None, help="PID of the running optimize process.")
    parser.add_argument(
        "--match",
        type=str,
        default="src/optimize.py",
        help="Substring used to locate the optimize process when --pid is omitted.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for a matching optimize process to appear instead of failing immediately.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between samples.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional max duration in seconds. 0 means until the target exits.",
    )
    parser.add_argument(
        "--top-global",
        type=int,
        default=20,
        help="How many top-RSS global processes to include per sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimize_memory_capture.json",
        help="Output JSON filepath.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    root_pid = args.pid
    if root_pid is None:
        while root_pid is None:
            root_pid = find_target_pid(args.match)
            if root_pid is not None or not args.wait:
                break
            time.sleep(max(0.2, args.interval))
        if root_pid is None:
            raise SystemExit(f"No running process matched {args.match!r}")

    header = {
        "captured_at": utc_now_iso(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": vars(args),
        "root_pid": root_pid,
    }
    samples = []
    start = time.monotonic()

    while True:
        rows = collect_ps()
        live_pids = {row.pid for row in rows}
        if root_pid not in live_pids:
            samples.append(
                {
                    "timestamp": utc_now_iso(),
                    "target_pid": root_pid,
                    "status": "exited",
                    "system_free_b": collect_free_bytes(),
                    "proc_meminfo_b": collect_proc_meminfo(),
                    "dev_shm_b": collect_df_shm(),
                    "dev_shm_top_entries": collect_shm_listing(),
                }
            )
            break

        samples.append(make_sample(rows, root_pid, args.top_global))

        if args.duration > 0 and (time.monotonic() - start) >= args.duration:
            break
        time.sleep(max(0.1, args.interval))

    payload = {
        "header": header,
        "samples": samples,
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
