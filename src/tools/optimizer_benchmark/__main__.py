"""Offline optimizer benchmark: catalogue, freeze, run, compare."""

import argparse
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import subprocess
import sys
import time

from .catalogue import catalogue
from .report import compare, markdown, validate_policies
from .workloads import freeze, harness_digest, load_manifest


def modes(values):
    result = []
    for value in values:
        engine, size = value.split(":")
        if engine not in {"cpu", "gpu", "search-cpu", "search-gpu"}:
            raise ValueError(f"unknown measurement engine {engine}")
        if size != "auto" and (not size.isdigit() or not 1 <= int(size) <= 4096):
            raise ValueError("workers/width must be 1..4096 or GPU auto")
        if engine.endswith("cpu") and (size == "auto" or int(size) > 64):
            raise ValueError("CPU worker count must be 1..64")
        result.append((value, engine, size))
    return result


def bounded_process(command, *, cwd, environment, log, timeout, max_rss, min_available):
    import psutil
    reason = None
    peak_rss, minimum_available = 0, None
    started = time.monotonic()
    with open(log, "w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    reason = "timeout"
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                    available = psutil.virtual_memory().available
                    peak_rss = max(peak_rss, rss)
                    minimum_available = available if minimum_available is None else min(minimum_available, available)
                    if rss > max_rss:
                        reason = "process_tree_memory_limit"
                    if available < min_available:
                        reason = "system_memory_headroom_limit"
                except psutil.NoSuchProcess:
                    pass  # A worker can exit between poll and its resource observation.
                if reason:
                    break
                time.sleep(.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        return dict(exit_code=process.returncode, termination_reason=reason,
                    sampled_peak_process_tree_rss_bytes=peak_rss,
                    sampled_minimum_system_available_bytes=minimum_available,
                    wall_seconds=time.monotonic() - started)


def read_worker_report(output, termination):
    """Keep failed or interrupted writes from aborting the rest of a campaign."""
    try:
        result = json.loads(output.read_text())
        if not isinstance(result, dict) or result.get("status") not in {
            "completed", "unsupported", "bounded_stop", "error",
        }:
            raise ValueError("worker report is not a status-bearing object")
    except (OSError, UnicodeError, ValueError) as error:
        result = dict(status="error", error="worker produced no complete report",
                      report_error=f"{type(error).__name__}: {error}")
    if termination["termination_reason"]:
        result.update(status="bounded_stop", reason=termination["termination_reason"])
    elif termination["exit_code"] != 0 and result.get("status") == "completed":
        result.update(status="error", error="worker exited unsuccessfully after writing its report")
    result["process"] = termination
    return result


def run(args):
    manifest = load_manifest(args.workload)
    parsed_modes = modes(args.modes)
    if args.phase_profile and any(engine.startswith("search-") for _, engine, _ in parsed_modes):
        raise ValueError("--phase-profile supports frozen replay modes, not production search")
    if len(args.backend) != 2 or len({item[0] for item in args.backend}) != 2:
        raise ValueError("supply two uniquely labelled --backend LABEL CHECKOUT PYTHON arguments")
    preflight = {}
    for label, root, python in args.backend:
        check = subprocess.run([str(Path(python).absolute()), "-c",
            "import sys,json; sys.path.insert(0,'src'); import passivbot_rust; "
            "from rust_utils import verify_loaded_runtime_extension; r=verify_loaded_runtime_extension(); "
            "assert r.get('runtime_compiled_source_stamp') == r.get('expected_source_fingerprint') "
            "and r.get('runtime_compiled_source_stamp'); print(json.dumps(r))"],
            cwd=root, capture_output=True, text=True, timeout=30)
        if check.returncode:
            raise ValueError(f"{label} runtime preflight failed; rebuild and verify that checkout: {check.stderr.strip()}")
        preflight[label] = json.loads(check.stdout)
    destination = Path(args.output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("report destination must be empty; use a new experiment directory")
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    fingerprint = harness_digest()
    from tools.gpu_parity import DEFAULT_TOLERANCES
    policies = {name: vars(policy) for name, policy in DEFAULT_TOLERANCES.items()}
    if args.policies:
        policies.update(json.loads(Path(args.policies).read_text()))
    validate_policies(policies)
    report = dict(version=1, workload_sha256=manifest["sha256"],
        harness_sha256=fingerprint,
        backends={label: dict(checkout=str(Path(root).resolve()), python=str(Path(python).absolute()),
                             commit=subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip())
                  for label, root, python in args.backend},
        runtime_preflight=preflight,
        hardware=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                      logical_cpus=os.cpu_count()), policies=policies, runs=[],
        bounds=dict(timeout_seconds=args.timeout, max_process_tree_rss_gib=args.max_rss_gib,
                    minimum_available_gib=args.min_available_gib),
        limitations=["Synthetic bounded workloads do not establish full-history performance or merge readiness.",
            "Legacy direct replay excludes master GPU optimizer CPU-validation overhead; search modes measure the production backend.",
            "Warm timing excludes worker startup and first use. First use can reuse on-disk compiler caches.",
            "Configured feature coverage is not proof that every controller transition occurred.",
            "Physical runner width is observed candidates in runner calls, not hardware occupancy."])
    experiment = 0
    import fnmatch
    entries = [entry for entry in manifest["entries"] if not args.case or
               any(fnmatch.fnmatchcase(entry["case"]["name"], pattern) for pattern in args.case)]
    if not entries:
        raise ValueError("case selection is empty")
    for entry_index, entry in enumerate(entries):
        for mode_index, (mode, engine, size) in enumerate(parsed_modes):
            order = list(args.backend)
            if (entry_index + mode_index) % 2:
                order.reverse()
            for label, root, python in order:
                experiment += 1
                print(f"[{experiment}] {label} {mode} {entry['case']['name']} seed={entry['seed']}", flush=True)
                output = destination / f"{experiment:04d}.json"
                log = destination / f"{experiment:04d}.log"
                command = [str(Path(python).absolute()), "-m", "optimizer_benchmark.worker",
                    "--backend", str(Path(root).resolve()), "--workload", str(Path(args.workload).resolve()),
                    "--entry", entry["id"], "--engine", engine, "--width", size if not engine.endswith("cpu") else "8",
                    "--workers", size if engine.endswith("cpu") else str(args.search_workers),
                    "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                    "--tune-seconds", str(args.tune_seconds), "--search-iters", str(args.search_iters),
                    "--dispatch-budget", str(args.dispatch_budget), "--output", str(output)]
                command += ["--harness-sha256", fingerprint]
                if args.phase_profile:
                    command.append("--phase-profile")
                environment = dict(os.environ, PYTHONPATH=str(source.parent),
                    PASSIVBOT_RUST_VERIFY="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    PASSIVBOT_GPU_PROFILE="1" if args.phase_profile else "0",
                    MKL_NUM_THREADS="1")
                termination = bounded_process(command, cwd=root, environment=environment, log=log,
                    timeout=args.timeout, max_rss=args.max_rss_gib * 1024**3,
                    min_available=args.min_available_gib * 1024**3)
                result = read_worker_report(output, termination)
                report["runs"].append(dict(entry=entry["id"], label=label, mode=mode, result=result,
                                           log_file=log.name, raw_file=output.name))
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                comparison = compare(report, manifest)
                (destination / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
                (destination / "comparison.md").write_text(markdown(comparison))
                print(f"  {result['status']} ({termination['wall_seconds']:.2f}s total)", flush=True)
    return 2 if any(r["result"]["status"] not in {"completed", "unsupported"} for r in report["runs"]) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalogue")
    prepare = sub.add_parser("freeze")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--profile", choices=("smoke", "standard", "extended"), default="smoke")
    prepare.add_argument("--case", action="append", default=[])
    prepare.add_argument("--candidates", type=int)
    prepare.add_argument("--bars", type=int)
    prepare.add_argument("--coins", type=int, default=4)
    prepare.add_argument("--seeds", nargs="+", type=int)
    prepare.add_argument("--random-seeds", type=int, default=0)
    prepare.add_argument("--boundary-candidates", action="store_true")
    prepare.add_argument("--bounds-config", help="Read optimize.bounds from a private canonical config; derived bounds and candidates are saved in the private workload")
    execute = sub.add_parser("run")
    execute.add_argument("--workload", required=True)
    execute.add_argument("--case", action="append", default=[])
    execute.add_argument("--backend", nargs=3, action="append", required=True, metavar=("LABEL", "CHECKOUT", "PYTHON"))
    execute.add_argument("--output", required=True)
    execute.add_argument("--modes", nargs="+", default=["cpu:1", "cpu:4", "gpu:1", "gpu:8", "gpu:auto"])
    execute.add_argument("--warmup", type=int, default=1)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--timeout", type=float, default=180)
    execute.add_argument("--max-rss-gib", type=float, default=6)
    execute.add_argument("--min-available-gib", type=float, default=1)
    execute.add_argument("--tune-seconds", type=float, default=0)
    execute.add_argument("--search-iters", type=int, default=64)
    execute.add_argument("--search-workers", type=int, default=4)
    execute.add_argument("--dispatch-budget", type=int, default=50_000_000)
    execute.add_argument("--phase-profile", action="store_true", help="Diagnostic GPU phase timings; adds synchronization overhead")
    execute.add_argument("--policies", help="Explicit per-metric numerical tolerance JSON")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--report", required=True)
    comparison.add_argument("--workload", required=True)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalogue":
            print(json.dumps([case.record() for case in catalogue()], indent=2))
            return 0
        if args.command == "compare":
            report = json.loads(Path(args.report).read_text())
            manifest = load_manifest(args.workload)
            if report["workload_sha256"] != manifest["sha256"]:
                raise ValueError("report and workload identities differ")
            Path(args.output).write_text(markdown(compare(report, manifest)))
            return 0
        if args.command == "freeze":
            from rust_utils import verify_loaded_runtime_extension
            import passivbot_rust
            verify_loaded_runtime_extension()
            if args.random_seeds < 0 or args.random_seeds > 1000:
                raise ValueError("random seed count must be 0..1000")
            seeds = [secrets.randbelow(2**32) for _ in range(args.random_seeds)] if args.random_seeds else (args.seeds or ([7] if args.profile == "smoke" else [7, 43, 101]))
            if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
                raise ValueError("seeds must be unique unsigned 32-bit integers")
            bounds = json.loads(Path(args.bounds_config).read_text())["optimize"]["bounds"] if args.bounds_config else None
            manifest = freeze(args.output, profile=args.profile, patterns=args.case,
                              candidates=args.candidates if args.candidates is not None else 8 if args.profile == "smoke" else 128,
                              bars=args.bars if args.bars is not None else 512 if args.profile == "smoke" else 10080,
                              coins=args.coins, seeds=seeds, bounds_config=bounds,
                              boundary_candidates=args.boundary_candidates)
            print(json.dumps(dict(workload_sha256=manifest["sha256"], cases=len(manifest["entries"]), seeds=seeds)))
            return 0
        if not 1 <= args.repeats <= 100 or not 1 <= args.warmup <= 10:
            raise ValueError("require 1..100 repetitions and 1..10 warmups")
        if any(not 0 < value < float("inf") for value in (args.timeout, args.max_rss_gib, args.min_available_gib)):
            raise ValueError("time and memory limits must be positive and finite")
        if not 0 <= args.tune_seconds < float("inf") or args.search_iters < 1 or args.dispatch_budget < 1:
            raise ValueError("require finite nonnegative tuning duration, positive iterations and dispatch budget")
        return run(args)
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
