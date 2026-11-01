"""Optional preparation references, separate from evaluation/fitness identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from simulation_preparation import PREPARATION_REPORT_SCHEMA_VERSION


@dataclass(frozen=True)
class PreparationReference:
    report: dict
    sha256: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate preparation manifest key: {key}")
        result[key] = value
    return result


def _projection(report: dict) -> dict:
    """Keep ordered input facts; omit only storage and acquisition locators."""
    if not isinstance(report, dict):
        raise ValueError("Preparation manifest must be an object")
    if type(report.get("schema_version")) is not int or report["schema_version"] != PREPARATION_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported preparation manifest schema_version")
    if report.get("kind") != "prepared-inputs" or report.get("consumer") != "optimize":
        raise ValueError("Expected a completed optimizer preparation manifest, not a plan")
    for field in ("suite", "datasets", "cases"):
        if field not in report:
            raise ValueError(f"Preparation manifest missing {field}")
    if not isinstance(report["suite"], dict):
        raise ValueError("Preparation manifest suite must be an object")
    for field in ("datasets", "cases"):
        if not isinstance(report[field], list) or not report[field]:
            raise ValueError(f"Preparation manifest {field} must be a nonempty list")
    datasets = []
    for dataset in report["datasets"]:
        if not isinstance(dataset, dict):
            raise ValueError("Preparation manifest dataset must be an object")
        if dataset.get("manifest_present") is not True or not isinstance(dataset.get("manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", dataset["manifest_sha256"]):
            raise ValueError("Preparation manifest requires verified dataset manifest hashes")
        projected = deepcopy(dataset)
        # A transferred cache can live at another root. This is not an identity
        # exception for any evaluation contract or saved fitness.
        projected.pop("cache_dir", None)
        datasets.append(projected)
    return {"schema_version": report["schema_version"], "kind": report["kind"],
            "consumer": report["consumer"], "suite": deepcopy(report["suite"]),
            "datasets": datasets, "cases": deepcopy(report["cases"])}


def load_preparation_reference(path) -> PreparationReference | None:
    if path is None:
        return None
    raw = Path(path).read_bytes()
    report = json.loads(raw, object_pairs_hook=_unique_object,
                        parse_constant=_reject_nonfinite)
    _projection(report)
    return PreparationReference(report, hashlib.sha256(raw).hexdigest())


def _reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _difference(expected, actual, path="inputs") -> str | None:
    # Bool/int and int/float equality is not typed input equality. Preserve
    # ordered lists and literal dates rather than comparing normalized clocks.
    if type(expected) is not type(actual):
        return path
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return path
        for key in expected:
            difference = _difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return path
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif expected != actual:
        return path
    return None


def preparation_metadata(actual: dict, reference: PreparationReference | None) -> dict:
    """Check after owner loading and before any candidate fitness is evaluated."""
    if reference is not None:
        difference = _difference(_projection(reference.report), _projection(actual))
        if difference:
            raise ValueError(f"Prepared inputs differ from preparation manifest at {difference}")
    return {"report": deepcopy(actual),
            "requested_manifest_sha256": None if reference is None else reference.sha256,
            "reference_verified": reference is not None}
