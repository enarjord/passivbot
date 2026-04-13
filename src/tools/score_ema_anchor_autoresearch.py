import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmaAnchorResearchConstraints:
    min_adg_strategy_pnl_rebased: float = 0.0
    max_drawdown_worst_hsl: float = 0.35
    max_peak_recovery_hours_hsl: float = 336.0
    min_fills_per_day: float = 0.25
    max_hours_no_fills_max: float = 168.0
    max_hours_no_fills_mean: float = 72.0
    max_hours_no_fills_median: float = 48.0


DEFAULT_CONSTRAINTS = EmaAnchorResearchConstraints()


def _require_metric(analysis: dict, key: str) -> float:
    if key not in analysis:
        raise KeyError(f"analysis missing required metric {key!r}")
    value = analysis[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"analysis metric {key!r} must be numeric") from exc


def resolve_analysis_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "analysis.json"
    if not candidate.exists():
        raise FileNotFoundError(f"analysis path not found: {candidate}")
    return candidate


def load_analysis(path: str | Path) -> dict:
    resolved = resolve_analysis_path(path)
    with resolved.open(encoding="utf-8") as f:
        return json.load(f)


def score_analysis(
    analysis: dict,
    *,
    constraints: EmaAnchorResearchConstraints = DEFAULT_CONSTRAINTS,
) -> dict:
    adg = _require_metric(analysis, "adg_strategy_pnl_rebased")
    drawdown = _require_metric(analysis, "drawdown_worst_hsl")
    recovery = _require_metric(analysis, "peak_recovery_hours_hsl")
    fills_per_day = _require_metric(analysis, "fills_per_day")
    no_fills_max = _require_metric(analysis, "hours_no_fills_max")
    no_fills_mean = _require_metric(analysis, "hours_no_fills_mean")
    no_fills_median = _require_metric(analysis, "hours_no_fills_median")
    liquidated = bool(analysis.get("liquidated", False))

    violations = []
    if liquidated:
        violations.append({"metric": "liquidated", "kind": "flag", "actual": True, "limit": False})
    if adg < constraints.min_adg_strategy_pnl_rebased:
        violations.append(
            {
                "metric": "adg_strategy_pnl_rebased",
                "kind": "min",
                "actual": adg,
                "limit": constraints.min_adg_strategy_pnl_rebased,
            }
        )
    for metric, actual, limit in (
        ("drawdown_worst_hsl", drawdown, constraints.max_drawdown_worst_hsl),
        ("peak_recovery_hours_hsl", recovery, constraints.max_peak_recovery_hours_hsl),
        ("hours_no_fills_max", no_fills_max, constraints.max_hours_no_fills_max),
        ("hours_no_fills_mean", no_fills_mean, constraints.max_hours_no_fills_mean),
        ("hours_no_fills_median", no_fills_median, constraints.max_hours_no_fills_median),
    ):
        if actual > limit:
            violations.append({"metric": metric, "kind": "max", "actual": actual, "limit": limit})
    if fills_per_day < constraints.min_fills_per_day:
        violations.append(
            {
                "metric": "fills_per_day",
                "kind": "min",
                "actual": fills_per_day,
                "limit": constraints.min_fills_per_day,
            }
        )

    if violations:
        penalty = 10.0 if liquidated else 0.0
        for violation in violations:
            if violation["kind"] == "flag":
                continue
            limit = abs(float(violation["limit"])) or 1.0
            actual = float(violation["actual"])
            if violation["kind"] == "max":
                penalty += max(0.0, (actual - limit) / limit)
            else:
                penalty += max(0.0, (limit - actual) / limit)
        return {
            "passed": False,
            "score": -penalty,
            "violations": violations,
            "metrics": {
                "adg_strategy_pnl_rebased": adg,
                "drawdown_worst_hsl": drawdown,
                "peak_recovery_hours_hsl": recovery,
                "fills_per_day": fills_per_day,
                "hours_no_fills_max": no_fills_max,
                "hours_no_fills_mean": no_fills_mean,
                "hours_no_fills_median": no_fills_median,
                "liquidated": liquidated,
            },
        }

    denominator = (
        1.0
        + drawdown / constraints.max_drawdown_worst_hsl
        + recovery / constraints.max_peak_recovery_hours_hsl
        + no_fills_max / constraints.max_hours_no_fills_max
        + no_fills_mean / constraints.max_hours_no_fills_mean
        + no_fills_median / constraints.max_hours_no_fills_median
    )
    fill_bonus = min(fills_per_day / constraints.min_fills_per_day, 4.0) * 0.0001
    score = adg / denominator + fill_bonus
    return {
        "passed": True,
        "score": score,
        "violations": [],
        "metrics": {
            "adg_strategy_pnl_rebased": adg,
            "drawdown_worst_hsl": drawdown,
            "peak_recovery_hours_hsl": recovery,
            "fills_per_day": fills_per_day,
            "hours_no_fills_max": no_fills_max,
            "hours_no_fills_mean": no_fills_mean,
            "hours_no_fills_median": no_fills_median,
            "liquidated": liquidated,
        },
    }


def score_suite(
    paths: list[str | Path],
    *,
    constraints: EmaAnchorResearchConstraints = DEFAULT_CONSTRAINTS,
) -> dict:
    results = []
    for path in paths:
        resolved = resolve_analysis_path(path)
        analysis = load_analysis(resolved)
        result = score_analysis(analysis, constraints=constraints)
        result["path"] = str(resolved)
        results.append(result)
    suite_score = sum(result["score"] for result in results) / len(results) if results else 0.0
    return {
        "constraints": asdict(constraints),
        "suite_passed": all(result["passed"] for result in results),
        "suite_score": suite_score,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score ema_anchor backtest artifacts for constrained autoresearch."
    )
    parser.add_argument("paths", nargs="+", help="analysis.json path(s) or backtest artifact dir(s)")
    parser.add_argument("--require-pass", action="store_true", help="exit nonzero if any run fails constraints")
    args = parser.parse_args()

    payload = score_suite(args.paths)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pass and not payload["suite_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
