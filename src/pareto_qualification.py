"""Attach supplied native replay evidence to exact Pareto result file identities."""

from datetime import date, datetime, time, timezone
import json
from pathlib import Path
import re

from config.metrics import CORE_METRIC_DEFINITION_VERSION, KNOWN_METRICS
from metrics_schema import decode_metric_value


SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
_METRIC_STATUSES = {"undefined", "invalid", "positive_infinity", "negative_infinity"}


def _object(value, path, required=(), optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = sorted(set(required) - value.keys())
    unexpected = sorted(value.keys() - set(required) - set(optional))
    if missing or unexpected:
        raise ValueError(f"{path} has missing fields {missing} and unexpected fields {unexpected}")


def _text(value, path):
    if not isinstance(value, str) or not value.strip() or not value.isprintable():
        raise ValueError(f"{path} must be a nonempty, printable string")


def _digest(value, path):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{path} must be a lowercase SHA-256 hex digest")


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} must be one of {', '.join(choices)}")


def _timestamp(value, path, *, date_boundary=None):
    if isinstance(value, str):
        try:
            if _UTC_TIMESTAMP.fullmatch(value):
                return datetime.fromisoformat(value[:-1] + "+00:00")
            if date_boundary is not None and _DATE.fullmatch(value):
                return datetime.combine(date.fromisoformat(value), date_boundary, timezone.utc)
        except ValueError:
            pass
    form = "a UTC date or timestamp" if date_boundary is not None else "a UTC timestamp"
    raise ValueError(f"{path} must be {form} in ISO 8601 format")


def _validate_metrics(metrics, path):
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} must be a metric object")
    names = metrics.keys() - {"metric_status"}
    if not names or not names <= KNOWN_METRICS:
        raise ValueError(f"{path} must contain known raw metrics")
    statuses = metrics.get("metric_status", {})
    if not isinstance(statuses, dict) or statuses.keys() - names:
        raise ValueError(f"{path}.metric_status must refer only to present metrics")
    for name in names:
        value = metrics[name]
        if name in statuses:
            status = statuses[name]
            status_path = f"{path}.metric_status.{name}"
            _object(status, status_path, ("status",), ("reason",))
            _choice(status["status"], sorted(_METRIC_STATUSES), f"{status_path}.status")
            if "reason" in status:
                _text(status["reason"], f"{status_path}.reason")
            if value is not None:
                raise ValueError(f"{path}.{name} must be null with an explicit status")
        else:
            try:
                decode_metric_value(metrics, name)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}.{name}: {exc}") from exc


def _validate_result(result, path, window):
    _object(
        result,
        path,
        ("status",),
        ("reason", "report", "effective_start_date", "effective_end_date", "liquidated", "metrics"),
    )
    status = result["status"]
    _choice(status, ("completed", "terminated", "unavailable"), f"{path}.status")
    if status != "completed" or "reason" in result:
        _text(result.get("reason"), f"{path}.reason")
    if status == "unavailable":
        if result.keys() - {"status", "reason", "report"}:
            raise ValueError(f"{path} cannot contain replay facts with unavailable status")
        if "report" in result:
            _text(result["report"], f"{path}.report")
        return
    _text(result.get("report"), f"{path}.report")
    start = _timestamp(result.get("effective_start_date"), f"{path}.effective_start_date")
    end = _timestamp(result.get("effective_end_date"), f"{path}.effective_end_date")
    if not window[0] <= start <= end <= window[1]:
        raise ValueError(f"{path} effective replay window must be ordered and within its scenario")
    if not isinstance(result.get("liquidated"), bool):
        raise ValueError(f"{path}.liquidated must be a boolean")
    if status == "completed" and result["liquidated"]:
        raise ValueError(f"{path} must declare terminated status for a liquidated replay")
    if status == "completed" or "metrics" in result:
        _validate_metrics(result.get("metrics"), f"{path}.metrics")


def _validate_subject(subject, path, windows):
    _object(subject, path, ("effective_config_sha256", "results"))
    _digest(subject["effective_config_sha256"], f"{path}.effective_config_sha256")
    results = subject["results"]
    if not isinstance(results, dict) or not results or results.keys() - windows.keys():
        raise ValueError(f"{path}.results must refer to at least one defined scenario")
    for label, result in results.items():
        _validate_result(result, f"{path}.results.{label}", windows[label])


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("qualification JSON contains a duplicate object key")
    return value


def load_qualification(path):
    """Validate a JSON sidecar. Evidence and report references are operator supplied."""
    if path is None:
        return None
    path = Path(path).expanduser()
    try:
        with path.open(encoding="utf-8") as stream:
            evidence = json.load(stream, object_pairs_hook=_unique_object)
        _validate_evidence(evidence)
    except ValueError as exc:
        raise ValueError(f"Invalid qualification evidence {path}: {exc}") from exc
    return evidence


def _validate_evidence(evidence):
    _object(
        evidence,
        "qualification",
        ("schema_version", "definition_version", "engine", "engine_sha256", "scenarios", "candidates"),
        ("incumbent",),
    )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("definition_version", CORE_METRIC_DEFINITION_VERSION),
    ):
        if type(evidence[key]) is not int or evidence[key] != expected:
            raise ValueError(f"qualification.{key} must be {expected}")
    _choice(evidence["engine"], ("rust_cpu",), "qualification.engine")
    _digest(evidence["engine_sha256"], "qualification.engine_sha256")
    scenarios = evidence["scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("qualification.scenarios must be a nonempty object")
    windows = {}
    for label, scenario in scenarios.items():
        path = f"qualification.scenarios.{label}"
        _text(label, "qualification scenario label")
        _object(scenario, path, ("split", "execution", "start_date", "end_date", "inputs_sha256"))
        _choice(scenario["split"], ("development", "holdout", "forward"), f"{path}.split")
        _choice(scenario["execution"], ("baseline", "stress"), f"{path}.execution")
        _digest(scenario["inputs_sha256"], f"{path}.inputs_sha256")
        start = _timestamp(
            scenario["start_date"], f"{path}.start_date", date_boundary=time.min
        )
        end = _timestamp(scenario["end_date"], f"{path}.end_date", date_boundary=time.max)
        if start > end:
            raise ValueError(f"{path} start_date is after end_date")
        windows[label] = (start, end)
    candidates = evidence["candidates"]
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("qualification.candidates must be a nonempty object")
    for digest, subject in candidates.items():
        _digest(digest, "qualification candidate key")
        _validate_subject(subject, f"qualification.candidates.{digest}", windows)
    if "incumbent" in evidence:
        _validate_subject(evidence["incumbent"], "qualification.incumbent", windows)


def qualification_for_candidates(evidence, candidate_sha256s):
    """Include only evidence matching the exact bytes of the supplied candidate files."""
    if evidence is None:
        return None
    candidates = evidence["candidates"]
    return {
        **evidence,
        "candidates": {
            digest: candidates[digest] for digest in candidate_sha256s if digest in candidates
        },
    }
