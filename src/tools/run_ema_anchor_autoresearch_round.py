import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from tools.score_ema_anchor_autoresearch import score_suite


DEFAULT_BASELINE_ITERS = 100000
DEFAULT_CANDIDATE_ITERS = 3000
DEFAULT_CPUS = 3
DEFAULT_SYMBOL = "XMR"
DEFAULT_START_DATE = "2024-10-01"
DEFAULT_END_DATE = "2026-04-01"
DEFAULT_CONFIG = "configs/examples/ema_anchor.json"
DEFAULT_FINE_TUNE_PARAMS = [
    "long_base_qty_pct",
    "short_base_qty_pct",
    "long_ema_span_0",
    "short_ema_span_0",
    "long_ema_span_1",
    "short_ema_span_1",
    "long_offset",
    "short_offset",
    "long_offset_psize_weight",
    "short_offset_psize_weight",
    "long_offset_volatility_ema_span_minutes",
    "short_offset_volatility_ema_span_minutes",
    "long_offset_volatility_1m_weight",
    "short_offset_volatility_1m_weight",
    "long_entry_volatility_ema_span_hours",
    "short_entry_volatility_ema_span_hours",
    "long_offset_volatility_1h_weight",
    "short_offset_volatility_1h_weight",
    "long_entry_double_down_factor",
    "short_entry_double_down_factor",
    "long_n_positions",
    "short_n_positions",
    "long_total_wallet_exposure_limit",
    "short_total_wallet_exposure_limit",
    "long_unstuck_ema_dist",
    "short_unstuck_ema_dist",
]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H_%M_%S")


def _default_results_dir(mode: str, symbol: str) -> str:
    return f"optimize_results/autoresearch_{mode}_{_timestamp()}_{symbol}"


def build_optimize_command(
    *,
    config_path: str,
    symbol: str,
    start_date: str,
    end_date: str,
    n_cpus: int,
    iterations: int,
    results_dir: str,
    baseline_pareto: str | None = None,
    fine_tune_params: list[str] | None = None,
) -> list[str]:
    cmd = [
        "passivbot",
        "optimize",
        config_path,
        "-s",
        symbol,
        "-sd",
        start_date,
        "-ed",
        end_date,
        "-c",
        str(n_cpus),
        "-i",
        str(iterations),
        "--backtest.candle_interval_minutes",
        "1",
        "-t",
        results_dir,
    ]
    if baseline_pareto:
        cmd.extend(["--start", baseline_pareto])
    if fine_tune_params:
        cmd.extend(["--fine_tune_params", ",".join(fine_tune_params)])
    return cmd


def _resolved_passivbot_command() -> str:
    executable = shutil.which("passivbot")
    if executable:
        return executable
    venv_candidate = Path(sys.executable).resolve().parent / "passivbot"
    if venv_candidate.exists():
        return str(venv_candidate)
    repo_candidate = Path(__file__).resolve().parents[2] / "venv" / "bin" / "passivbot"
    if repo_candidate.exists():
        return str(repo_candidate)
    raise FileNotFoundError("passivbot executable not found on PATH")


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


def _score_and_print(results_dir: str) -> dict:
    payload = score_suite([Path(results_dir) / "pareto"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a constrained ema_anchor autoresearch round.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def _common(subparser, default_iters: int):
        subparser.add_argument("--config", default=DEFAULT_CONFIG)
        subparser.add_argument("--symbol", default=DEFAULT_SYMBOL)
        subparser.add_argument("--start-date", default=DEFAULT_START_DATE)
        subparser.add_argument("--end-date", default=DEFAULT_END_DATE)
        subparser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
        subparser.add_argument("--iterations", type=int, default=default_iters)
        subparser.add_argument("--results-dir", default=None)
        subparser.add_argument("--run", action="store_true")

    baseline = subparsers.add_parser("baseline", help="Build the reference ema_anchor baseline.")
    _common(baseline, DEFAULT_BASELINE_ITERS)

    candidate = subparsers.add_parser(
        "candidate",
        help="Evaluate a strategy-code candidate with warm-started fine-tune optimize.",
    )
    _common(candidate, DEFAULT_CANDIDATE_ITERS)
    candidate.add_argument(
        "--baseline-pareto",
        required=True,
        help="Path to the baseline pareto dir or directory containing pareto/",
    )
    candidate.add_argument(
        "--fine-tune-params",
        default=",".join(DEFAULT_FINE_TUNE_PARAMS),
        help="Comma-separated optimize keys to leave tunable in candidate mode.",
    )

    args = parser.parse_args()
    results_dir = args.results_dir or _default_results_dir(args.mode, args.symbol)
    fine_tune_params = None
    baseline_pareto = None
    if args.mode == "candidate":
        fine_tune_params = [p.strip() for p in args.fine_tune_params.split(",") if p.strip()]
        baseline_pareto = args.baseline_pareto
        baseline_path = Path(baseline_pareto)
        if baseline_path.is_dir() and baseline_path.name != "pareto":
            baseline_pareto = str(baseline_path / "pareto")

    cmd = build_optimize_command(
        config_path=args.config,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        n_cpus=args.cpus,
        iterations=args.iterations,
        results_dir=results_dir,
        baseline_pareto=baseline_pareto,
        fine_tune_params=fine_tune_params,
    )
    cmd[0] = _resolved_passivbot_command()
    print("COMMAND:")
    print(" ".join(cmd))

    if not args.run:
        return 0

    _run(cmd)
    _score_and_print(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
