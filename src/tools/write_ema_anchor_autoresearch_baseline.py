import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tools.score_ema_anchor_autoresearch import DEFAULT_CONSTRAINTS
from tools.score_ema_anchor_autoresearch import score_suite


def resolve_baseline_dirs(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"baseline path not found: {candidate}")
    if candidate.is_dir() and candidate.name == "pareto":
        return candidate.parent, candidate
    pareto_dir = candidate / "pareto" if candidate.is_dir() else None
    if pareto_dir is not None and pareto_dir.is_dir():
        return candidate, pareto_dir
    raise ValueError("baseline path must be an optimize run dir or a pareto dir")


def build_baseline_manifest(path: str | Path) -> dict:
    run_dir, pareto_dir = resolve_baseline_dirs(path)
    scored = score_suite([pareto_dir], constraints=DEFAULT_CONSTRAINTS)
    best = scored.get("best_result")
    if best is None:
        raise ValueError(f"no Pareto members found in {pareto_dir}")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "baseline_run_dir": str(run_dir.resolve()),
        "baseline_pareto_dir": str(pareto_dir.resolve()),
        "constraints": scored["constraints"],
        "suite_passed": scored["suite_passed"],
        "suite_score": scored["suite_score"],
        "n_candidates": len(scored["results"]),
        "best_member_path": str(Path(best["path"]).resolve()),
        "best_score": best["score"],
        "best_passed": best["passed"],
        "best_metrics": best["metrics"],
        "best_violations": best["violations"],
        "candidate_command_template": (
            "PYTHONPATH=src python3 src/tools/run_ema_anchor_autoresearch_round.py "
            f"candidate --baseline-pareto {pareto_dir.resolve()} --run"
        ),
    }


def write_baseline_manifest(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    require_pass: bool = True,
) -> Path:
    manifest = build_baseline_manifest(path)
    if require_pass and not manifest["best_passed"]:
        raise ValueError("best baseline candidate does not satisfy constrained scorer")
    run_dir = Path(manifest["baseline_run_dir"])
    destination = Path(output_path) if output_path is not None else run_dir / "baseline.json"
    with destination.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write baseline.json for an ema_anchor autoresearch optimize run."
    )
    parser.add_argument("path", help="Optimize run dir or pareto dir")
    parser.add_argument("-o", "--output", default=None, help="Optional output path")
    parser.add_argument(
        "--allow-failing-best",
        action="store_true",
        help="Write baseline.json even if the best Pareto member fails constraints",
    )
    args = parser.parse_args()

    destination = write_baseline_manifest(
        args.path,
        output_path=args.output,
        require_pass=not args.allow_failing_best,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
