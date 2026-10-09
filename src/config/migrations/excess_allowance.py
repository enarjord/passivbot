"""Retire the released raw/bounded selector without silently changing risk policy."""

import logging


def retire_excess_allowance_mode(node, *, path="", tracker=None):
    if isinstance(node, dict):
        for key in list(node):
            if str(key).startswith("_"):
                continue
            location = f"{path}.{key}" if path else str(key)
            if str(key).split(".")[-1] in {
                "we_excess_allowance_mode",
                "risk_we_excess_allowance_mode",
            }:
                value = node[key]
                normalized = value.strip().lower() if isinstance(value, str) else value
                if normalized not in (None, "bounded"):
                    message = (
                        f"{location}={value!r} is no longer supported: excess allowance is always bounded by side TWEL. "
                        "Configuration loading stopped; this configuration will not be used for trading. "
                        "Remove this field only after reviewing the reduced exposure headroom, "
                        "then re-backtest this configuration. Legacy raw sizing cannot be preserved automatically."
                    )
                    logging.error(message)
                    raise ValueError(message)
                del node[key]
                if tracker is not None:
                    tracker.remove(location.split("."), value)
                logging.warning(
                    "Removed obsolete %s=%r; bounded excess allowance is unchanged",
                    location,
                    value,
                )
            else:
                retire_excess_allowance_mode(node[key], path=location, tracker=tracker)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            retire_excess_allowance_mode(
                value, path=f"{path}[{index}]", tracker=tracker
            )
