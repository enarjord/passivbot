"""Native-owned GPU core observation ABI and result materialization.

Python selects and transports requested quantities. Rust owns the packet layout,
metric formulas, sample validity and nonfinite status handling.
"""

from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=1)
def observation_catalog():
    import passivbot_rust
    from config.metrics import CORE_METRIC_CATALOG, CORE_METRIC_DEFINITION_VERSION

    value = passivbot_rust.get_core_gpu_observation_catalog()
    if not isinstance(value, dict) or value.get("definition_version") != CORE_METRIC_DEFINITION_VERSION:
        raise RuntimeError("GPU observation definition disagrees with the native core catalog")
    fields, bits = value.get("fields"), value.get("metric_bits")
    if (not isinstance(fields, list) or not fields or len(fields) != len(set(fields))
            or any(not isinstance(field, str) or not field for field in fields)
            or not isinstance(bits, dict) or set(bits) != set(CORE_METRIC_CATALOG)
            or any(isinstance(bit, bool) or not isinstance(bit, int) or bit <= 0
                   or bit & (bit - 1) for bit in bits.values())
            or len(set(bits.values())) != len(bits)):
        raise RuntimeError("Malformed native GPU core observation catalog")
    return value


def metric_mask(metric_names):
    from config.metrics import metric_base_name

    bits = observation_catalog()["metric_bits"]
    mask = 0
    for requested in metric_names:
        name = metric_base_name(requested)
        try:
            mask |= bits[name]
        except KeyError as error:
            raise ValueError(f"GPU core metric is not implemented: {requested!r}") from error
    return mask


def with_core_metrics(source, mask, half_life_days):
    if (isinstance(mask, bool) or not isinstance(mask, int) or mask < 0
            or mask & ~sum(observation_catalog()["metric_bits"].values())):
        raise ValueError("Invalid native GPU core metric mask")
    if not isinstance(half_life_days, (int, float)) or isinstance(half_life_days, bool) or not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("GPU core EW half-life must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        encoded_half_life = np.float32(half_life_days)
    if not np.isfinite(encoded_half_life) or encoded_half_life <= 0:
        raise ValueError("GPU core EW half-life must encode as a positive finite float32")
    if "CORE_OBSERVATION_FIELDS" in source or "CORE_WRITE_FIELDS" in source:
        raise RuntimeError("GPU core observation source was not composed by Rust")
    literal = format(float(encoded_half_life), ".17g")
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return (f"#define PASSIVBOT_CORE_REQUEST_MASK {mask}u\n"
            f"#define PASSIVBOT_CORE_HALF_LIFE_DAYS {literal}f\n" + source)


def finalize_observations(packet, metric_names, *, origin_timestamp_ms,
                          observation_interval_ms, intended_end_timestamp_ms,
                          ew_half_life_days, hsl_effective_policy_enabled=None):
    import passivbot_rust

    if hasattr(packet, "detach"):
        packet = packet.detach().cpu().numpy()
    values = np.asarray(packet)
    if (values.ndim != 2 or values.dtype != np.dtype(np.float32)
            or values.shape[1] != len(observation_catalog()["fields"])):
        raise RuntimeError("Malformed GPU core observation packet")
    rows = passivbot_rust.core_metrics_from_gpu_observations(
        np.ascontiguousarray(values), list(metric_names), int(origin_timestamp_ms),
        int(observation_interval_ms), int(intended_end_timestamp_ms), ew_half_life_days,
        hsl_effective_policy_enabled,
    )
    if not isinstance(rows, list) or len(rows) != values.shape[0]:
        raise RuntimeError("Native GPU metric finalization changed candidate count")
    return rows
