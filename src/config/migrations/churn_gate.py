import logging
import math
from typing import Optional

from config.transform_log import ConfigTransformTracker


LEGACY_DISTANCE_KEY = "initial_entry_exec_max_market_dist_pct"
CHURN_DISTANCE_KEY = "order_replacement_churn_gate_market_dist_pct"
CHURN_ACTIVATION_KEY = "order_replacement_churn_gate_activation_count"


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    if verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)


def _as_finite_float(value: object, *, path: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be finite")
    return normalized


def _get_explicit_activation_count(live: dict) -> Optional[int]:
    if CHURN_ACTIVATION_KEY not in live:
        return None
    activation = live[CHURN_ACTIVATION_KEY]
    if isinstance(activation, bool) or not isinstance(activation, int):
        raise TypeError(f"config.live.{CHURN_ACTIVATION_KEY} must be an integer")
    return activation


def migrate_initial_entry_distance_gate(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    """Migrate the retired initial-entry-only distance gate.

    Positive legacy values map directly to the new near-market exemption distance.
    Null and non-positive values disabled the old gate, so they map to the new
    gate's explicit activation-count disable sentinel. Missing new tuning fields
    are subsequently hydrated from the canonical template.
    """

    live = result.get("live")
    if not isinstance(live, dict) or LEGACY_DISTANCE_KEY not in live:
        return

    legacy_raw = live[LEGACY_DISTANCE_KEY]
    legacy_value = (
        0.0
        if legacy_raw is None
        else _as_finite_float(
            legacy_raw,
            path=f"config.live.{LEGACY_DISTANCE_KEY}",
        )
    )
    explicit_activation = _get_explicit_activation_count(live)

    if legacy_value <= 0.0:
        if explicit_activation is not None:
            if explicit_activation != 0:
                raise ValueError(
                    f"config.live.{LEGACY_DISTANCE_KEY}={legacy_raw!r} disabled the "
                    "retired gate, but config.live."
                    f"{CHURN_ACTIVATION_KEY}={explicit_activation!r} enables the "
                    "replacement; "
                    "remove the legacy field or resolve the conflict explicitly"
                )
            live.pop(LEGACY_DISTANCE_KEY)
            if tracker is not None:
                tracker.remove(["live", LEGACY_DISTANCE_KEY], legacy_raw)
        else:
            live.pop(LEGACY_DISTANCE_KEY)
            live[CHURN_ACTIVATION_KEY] = 0
            if tracker is not None:
                tracker.rename(
                    ["live", LEGACY_DISTANCE_KEY],
                    ["live", CHURN_ACTIVATION_KEY],
                    0,
                )
        _log_config(
            verbose,
            logging.INFO,
            "migrated live.%s=%s (disabled) -> live.%s=0",
            LEGACY_DISTANCE_KEY,
            legacy_raw,
            CHURN_ACTIVATION_KEY,
        )
        return

    if legacy_value >= 1.0:
        raise ValueError(
            f"config.live.{LEGACY_DISTANCE_KEY} must be less than 1.0 to migrate; "
            f"got {legacy_raw!r}"
        )

    if explicit_activation == 0:
        raise ValueError(
            f"config.live.{LEGACY_DISTANCE_KEY}={legacy_raw!r} enabled the retired "
            f"gate, but config.live.{CHURN_ACTIVATION_KEY}=0 disables the replacement; "
            "remove the legacy field or resolve the conflict explicitly"
        )

    if CHURN_DISTANCE_KEY in live:
        replacement_raw = live[CHURN_DISTANCE_KEY]
        replacement_value = _as_finite_float(
            replacement_raw,
            path=f"config.live.{CHURN_DISTANCE_KEY}",
        )
        if replacement_value != legacy_value:
            raise ValueError(
                f"config.live.{LEGACY_DISTANCE_KEY}={legacy_raw!r} conflicts with "
                f"config.live.{CHURN_DISTANCE_KEY}={replacement_raw!r}; remove the "
                "legacy field or resolve the conflict explicitly"
            )
        live.pop(LEGACY_DISTANCE_KEY)
        if tracker is not None:
            tracker.remove(["live", LEGACY_DISTANCE_KEY], legacy_raw)
    else:
        live.pop(LEGACY_DISTANCE_KEY)
        live[CHURN_DISTANCE_KEY] = legacy_value
        if tracker is not None:
            tracker.rename(
                ["live", LEGACY_DISTANCE_KEY],
                ["live", CHURN_DISTANCE_KEY],
                legacy_value,
            )

    _log_config(
        verbose,
        logging.INFO,
        "migrated live.%s=%s -> live.%s=%s; other churn-gate settings use "
        "canonical defaults unless explicitly configured",
        LEGACY_DISTANCE_KEY,
        legacy_raw,
        CHURN_DISTANCE_KEY,
        legacy_value,
    )
