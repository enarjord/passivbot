"""Comparable completed work, numerical gates, and explicit missing evidence."""

import math
import statistics


def validate_policies(policies):
    if not isinstance(policies, dict):
        raise ValueError("numerical policies must be an object")
    for name, policy in policies.items():
        if not isinstance(name, str) or not name or not isinstance(policy, dict):
            raise ValueError("each policy needs a metric name and an object")
        if set(policy) - {"absolute", "relative", "matching_infinity"} or not {"absolute", "relative"} <= set(policy):
            raise ValueError(f"{name} policy requires absolute/relative and optional matching_infinity")
        for field in ("absolute", "relative"):
            value = policy[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}.{field} must be finite and nonnegative")
        if not isinstance(policy.get("matching_infinity", False), bool):
            raise ValueError(f"{name}.matching_infinity must be boolean")


def compare_values(reference, observed, metrics, policies):
    """Standalone comparison so older target branches need no parity module."""
    validate_policies(policies)
    rows = {}
    for name in metrics:
        left, right, policy = reference.get(name), observed.get(name), policies.get(name)
        row = dict(cpu=left, gpu=right)
        if left is None or right is None:
            status = "missing_both" if left is None and right is None else "missing_cpu" if left is None else "missing_gpu"
        else:
            left, right = float(left), float(right)
            if not (math.isfinite(left) and math.isfinite(right)):
                status = "sentinel_match" if policy and policy.get("matching_infinity", False) and math.isinf(left) and left == right else "non_finite"
            elif policy is None:
                status = "unassessed"
            else:
                error = abs(left - right)
                allowed = policy["absolute"] + policy["relative"] * abs(left)
                status = "match" if math.isfinite(error) and math.isfinite(allowed) and error <= allowed else "mismatch"
                row.update(absolute_error=error, allowed_error=allowed)
        row["status"] = status
        rows[name] = row
    return dict(passed=bool(rows) and all(row["status"] in {"match", "sentinel_match"} for row in rows.values()), metrics=rows)


def summary(result):
    samples = result.get("samples", [])
    if result.get("status") != "completed" or not samples:
        return dict(status=result.get("status", "missing"))
    seconds = [s["seconds"] for s in samples]
    total_seconds = sum(seconds)
    completed = sum(s["completed"] for s in samples)
    dispatches = [d["candidates"] for s in samples for d in s.get("dispatches", [])]
    latencies = [r["latency_seconds"] for s in samples for r in s["results"] if "latency_seconds" in r]
    completion_latencies = [v for s in samples for v in s.get("completion_latencies_seconds", [])]
    if result.get("isolated_requests"):
        latencies = completion_latencies
    def p95(values):
        return sorted(values)[max(0, math.ceil(len(values) * .95) - 1)] if values else None
    return dict(status="completed", median_batch_seconds=statistics.median(seconds),
                min_batch_seconds=min(seconds), max_batch_seconds=max(seconds),
                candidates_per_second=completed / total_seconds,
                completed=completed, measured_seconds=total_seconds, repetitions=len(samples),
                max_observed_runner_width=max(dispatches, default=None),
                dispatch_size_counts={str(n): dispatches.count(n) for n in sorted(set(dispatches))},
                single_candidate_latency_median_seconds=statistics.median(latencies) if latencies else None,
                single_candidate_latency_p95_seconds=p95(latencies),
                request_latency_p95_seconds=p95(completion_latencies))


def numerical_comparison(reference, observed, metrics, policies, checks=()):
    if not reference or not observed or reference.get("status") != "completed" or observed.get("status") != "completed":
        return dict(status="unassessed", reason="both completed measurements required")
    a, b = reference["samples"][0]["results"], observed["samples"][0]["results"]
    if not a or not b:
        return dict(status="unassessed", reason="no candidate results")
    if len(a) != len(b):
        return dict(status="mismatch", reason="different candidate count")
    comparisons = []
    for left, right in zip(a, b):
        result = compare_values(left["metrics"], right["metrics"], metrics, policies)
        liquidation_assessed = left.get("liquidated") is not None and right.get("liquidated") is not None
        result["liquidation"] = ("match" if left["liquidated"] == right["liquidated"] else "mismatch") if liquidation_assessed else "unassessed"
        result["passed"] = result["passed"] and result["liquidation"] == "match"
        limits = []
        for check in checks:
            from limit_utils import compute_limit_violation
            values = [row["metrics"].get(check["metric"]) for row in (left, right)]
            if check.get("scenario") is not None or any(value is None or not math.isfinite(float(value)) for value in values):
                limits.append(dict(metric=check["metric"], status="unassessed"))
            else:
                violations = [compute_limit_violation(check, float(value)) for value in values]
                limits.append(dict(metric=check["metric"], reference_violation=violations[0],
                    observed_violation=violations[1], status="match" if (violations[0] > 0) == (violations[1] > 0) else "flip"))
        result["limits"] = limits
        result["passed"] = result["passed"] and all(row["status"] == "match" for row in limits)
        comparisons.append(result)
    variation = any(len({sample["results_sha256"] for sample in result["samples"]
                         if "results_sha256" in sample}) > 1 for result in (reference, observed))
    if variation:
        return dict(status="repeat_variation_unassessed", candidates=len(a), rows=comparisons,
                    reason="warm cohort outputs changed; compact repeated evidence requires investigation")
    return dict(status="passed" if all(r["passed"] for r in comparisons) else "failed_or_incomplete",
                passed_candidates=sum(r["passed"] for r in comparisons), candidates=len(a), rows=comparisons)


def tuning_comparison(automatic, fixed):
    measured = [summary(r) for r in fixed if r and r.get("status") == "completed"]
    windows = automatic.get("tuning_windows", []) if automatic else []
    if not measured:
        return dict(status="unassessed", reason="no completed fixed-width comparison")
    target = max(row["candidates_per_second"] for row in measured)
    samples = automatic.get("samples", []) if automatic else []
    reached = None
    for i in range(2, len(samples)):
        if all(sample["completed"] / sample["seconds"] >= target * .95 for sample in samples[i - 2:i + 1]):
            reached = samples[i]["elapsed_seconds"]
            break
    return dict(status="observed" if windows else "insufficient_tuner_evidence",
        completed_windows=len(windows), best_measured_fixed_candidates_per_second=target,
        first_tuning_window_seconds=automatic.get("first_tuning_window_seconds") if automatic else None,
        time_to_three_cohorts_within_five_percent_seconds=reached,
        scope="bounded fixed-width sweep; cohort throughput target is separate from controller window rates")


def ranking_assessment(reference, observed):
    if not reference or not observed or reference.get("status") != "completed" or observed.get("status") != "completed":
        return dict(status="unassessed")
    left = reference["samples"][0]["results"]
    right = observed["samples"][0]["results"]
    if not left or len(left) != len(right) or len(left) > 512:
        return dict(status="unassessed", reason="requires 1..512 matching candidate rows")
    objectives = {"adg_strategy_eq": "max", "drawdown_worst_strategy_eq": "min"}
    try:
        vectors = [[tuple(float(row["metrics"][name]) * (-1 if direction == "max" else 1)
                          for name, direction in objectives.items()) for row in rows] for rows in (left, right)]
    except (KeyError, TypeError, ValueError):
        return dict(status="unassessed", reason="missing or invalid objective")
    if any(not math.isfinite(value) for rows in vectors for row in rows for value in row):
        return dict(status="unassessed", reason="nonfinite objective")
    def front(rows):
        return [i for i, row in enumerate(rows) if not any(
            all(a <= b for a, b in zip(other, row)) and any(a < b for a, b in zip(other, row))
            for j, other in enumerate(rows) if i != j)]
    fronts = [front(rows) for rows in vectors]
    flips, regret = {}, {}
    for axis, name in enumerate(objectives):
        relation = lambda a, b: (a > b) - (a < b)
        flips[name] = sum(relation(vectors[0][i][axis], vectors[0][j][axis]) !=
                          relation(vectors[1][i][axis], vectors[1][j][axis])
                          for i in range(len(left)) for j in range(i + 1, len(left)))
        chosen = min(range(len(right)), key=lambda i: vectors[1][i][axis])
        regret[name] = vectors[0][chosen][axis] - min(row[axis] for row in vectors[0])
    return dict(status="assessed", objectives=objectives, reference_front=fronts[0], observed_front=fronts[1],
                front_members_match=fronts[0] == fronts[1], pair_order_disagreements=flips,
                reference_regret_at_observed_best=regret)


def compare(report, manifest):
    labels = list(report["backends"])
    if len(labels) != 2:
        raise ValueError("comparison requires exactly two backend labels")
    rows = []
    for entry in manifest["entries"]:
        from limit_utils import expand_limit_checks
        checks = expand_limit_checks(entry.get("diagnostic_limits", []),
            {"adg_strategy_eq": -1, "drawdown_worst_strategy_eq": 1}, penalty_weight=1.0)
        rows_for_case = [r for r in report["runs"] if r["entry"] == entry["id"]]
        modes = list(dict.fromkeys(r["mode"] for r in rows_for_case))
        def find(label, mode):
            return next((r["result"] for r in rows_for_case if r["label"] == label and r["mode"] == mode), None)
        for mode in modes:
            first, second = find(labels[0], mode), find(labels[1], mode)
            summaries = [summary(r or {}) for r in (first, second)]
            speedup = summaries[0]["median_batch_seconds"] / summaries[1]["median_batch_seconds"] if all(s["status"] == "completed" for s in summaries) else None
            metrics = numerical_comparison(first, second, entry["metrics"], report["policies"], checks) if not mode.startswith("search") else dict(status="separate_search_reference_validation_required")
            gpu_parity = {}
            if mode.startswith("gpu"):
                for label in labels:
                    gpu_parity[label] = numerical_comparison(find(label, "cpu:1"), find(label, mode),
                                                             entry["metrics"], report["policies"], checks)
                    gpu_parity[label]["ranking"] = ranking_assessment(find(label, "cpu:1"), find(label, mode))
            tuning = {label: tuning_comparison(find(label, mode),
                [find(label, other) for other in modes if other.startswith("gpu:") and other != "gpu:auto"])
                for label in labels} if mode == "gpu:auto" else {}
            rows.append(dict(case=entry["case"]["name"], seed=entry["seed"], mode=mode,
                summaries=dict(zip(labels, summaries)), speedup_second_over_first=speedup,
                branch_numerical_comparison=metrics, gpu_cpu_comparison=gpu_parity,
                branch_ranking=ranking_assessment(first, second) if not mode.startswith("search") else None,
                tuning=tuning,
                equivalent_work_speedup=speedup if metrics["status"] == "passed" and
                    all(p["status"] == "passed" for p in gpu_parity.values()) else None))
    return dict(labels=labels, rows=rows,
                interpretation="speedup = first branch median time / second branch median time; >1 favors second; failed/incomplete correctness does not establish equivalent work speedup")


def markdown(comparison):
    labels = comparison["labels"]
    lines = ["# Optimizer benchmark comparison", "", comparison["interpretation"], "",
             f"| Case / seed | Mode | {labels[0]} c/s | {labels[1]} c/s | Time ratio | Correctness |",
             "|---|---|---:|---:|---:|---|"]
    for row in comparison["rows"]:
        values = [row["summaries"][label].get("candidates_per_second") for label in labels]
        formatted = [f"{v:.3f}" if v is not None else "unavailable" for v in values]
        ratio = row["speedup_second_over_first"]
        status = row["branch_numerical_comparison"]["status"]
        if row["gpu_cpu_comparison"]:
            status += "; CPU/GPU " + "/".join(row["gpu_cpu_comparison"][label]["status"] for label in labels)
        lines.append(f"| {row['case']} / {row['seed']} | {row['mode']} | {formatted[0]} | {formatted[1]} | {ratio:.3f} | {status} |" if ratio is not None else
                     f"| {row['case']} / {row['seed']} | {row['mode']} | {formatted[0]} | {formatted[1]} | unavailable | {status} |")
    return "\n".join(lines) + "\n"
