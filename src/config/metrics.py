"""Exact versioned selectors from Rust; no formula or currency aliases."""

from collections.abc import Mapping
from types import MappingProxyType
import math
import passivbot_rust as _native

if not callable(getattr(_native, "get_core_metric_catalog", None)):
    raise RuntimeError(
        "Rust extension lacks the versioned core metric catalog; rebuild the extension"
    )
_catalog = _native.get_core_metric_catalog()
CORE_METRIC_DEFINITION_VERSION = 1
if _catalog.get("definition_version") != CORE_METRIC_DEFINITION_VERSION:
    raise RuntimeError("Core metric definition mismatch; rebuild the Rust extension")
_entries = _catalog.get("metrics")
if not isinstance(_entries, list) or len(_entries) != 25:
    raise RuntimeError("Malformed native core metric catalog")
CORE_METRIC_CATALOG = MappingProxyType(
    {item["name"]: MappingProxyType(dict(item)) for item in _entries}
)
if len(CORE_METRIC_CATALOG) != 25 or any(
    item["role"] not in {"maximize", "minimize", "limit"}
    or not isinstance(item.get("requires_minute_observations"), bool)
    for item in _entries
):
    raise RuntimeError("Malformed native core metric catalog")
DEFAULT_EW_HALF_LIFE_DAYS = float(_catalog["default_ew_half_life_days"])
if not math.isfinite(DEFAULT_EW_HALF_LIFE_DAYS) or DEFAULT_EW_HALF_LIFE_DAYS <= 0:
    raise RuntimeError("Malformed native EW half-life default")
KNOWN_METRICS = frozenset(CORE_METRIC_CATALOG)
EXPOSURE_RMS_METRICS = frozenset(name for name in KNOWN_METRICS if "exposure" in name and "_rms_" in name)
STAT_SUFFIXES = ("min", "max", "mean", "std", "median")


def canonical_metric_name(metric: str) -> str:
    return str(metric).strip()


def split_metric_stat_suffix(name: str) -> tuple[str, str | None]:
    metric = canonical_metric_name(name)
    if metric in KNOWN_METRICS:
        return metric, None
    for suffix in STAT_SUFFIXES:
        marker = "_" + suffix
        if metric.endswith(marker) and metric[: -len(marker)] in KNOWN_METRICS:
            return metric[: -len(marker)], suffix
    return metric, None


def metric_base_name(name: str) -> str:
    return split_metric_stat_suffix(name)[0]


def metric_stat_key(name: str, reducer: str = "mean") -> str:
    base, suffix = split_metric_stat_suffix(name)
    return base + "_" + (suffix or reducer)


def canonicalize_metric_name(metric: str) -> str:
    return canonical_metric_name(metric)


def reject_retired_metric(metric: str, *, path: str) -> None:
    if canonical_metric_name(metric) not in KNOWN_METRICS:
        raise ValueError(
            f"{path}: unknown or retired core metric {metric!r}; choose an explicit supported metric and re-evaluate results and limits"
        )


def resolve_metric_value(metrics: Mapping[str, object], requested_name: str):
    name = canonical_metric_name(requested_name)
    if name not in metrics:
        return None
    from metrics_schema import decode_metric_value

    return decode_metric_value(metrics, name)


def canonicalize_limit_name(limit_key: str) -> str:
    for old, current in (
        ("lower_bound_", "penalize_if_greater_than_"),
        ("upper_bound_", "penalize_if_lower_than_"),
    ):
        if limit_key.startswith(old):
            return current + canonical_metric_name(limit_key[len(old) :])
    return canonical_metric_name(limit_key)


def validate_core_metric_requests(config, metrics):
    interval = config.get("backtest", {}).get("candle_interval_minutes", 1)
    for name in metrics:
        reject_retired_metric(name, path="metric_requests")
        base = metric_base_name(name)
        if interval != 1 and CORE_METRIC_CATALOG[base].get(
            "requires_minute_observations", False
        ):
            raise ValueError(f"metric {base!r} requires one-minute replay observations")
        if base in {
            "volatility_annual_fraction_strategy_equity",
            "sharpe_ratio_strategy_equity",
            "sortino_ratio_strategy_equity",
            "omega_ratio_strategy_equity",
        } and (interval != int(interval) or interval <= 0 or 1440 % int(interval)):
            raise ValueError(
                f"metric {base!r} requires a cadence supplying complete UTC daily closes"
            )
