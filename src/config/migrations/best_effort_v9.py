"""Observable guesses for the opt-in offline converter; runtime loading stays strict."""

from copy import deepcopy
import logging

from config.risk_memory import DEFAULT_LOOKBACK_EMA_SPAN_MINUTES, normalize_lookback_ema_span_minutes


def record_guess(changes, path, old, new, reason, *, log_prefix=""):
    """A None prefix keeps validation-only decisions out of the warning stream."""
    item = dict(path=path, old=deepcopy(old), new=deepcopy(new),
                reason=reason, best_effort=True)
    changes.append(item)
    if log_prefix is None:
        return

    def display(value):
        rendered = repr(value)
        return rendered if len(rendered) <= 240 else rendered[:240] + "... (full value in report)"

    logging.getLogger(__name__).warning(
        "[best-effort] %s: %s -> %s; %s", log_prefix + path, display(old), display(new), reason,
    )


def infer_memory_span(value, path, changes, *, missing=False):
    from .v9 import _validate_legacy_gate_window

    if missing:
        span = DEFAULT_LOOKBACK_EMA_SPAN_MINUTES
        reason = "missing rolling window; use canonical 30-day memory default"
    else:
        _validate_legacy_gate_window(value, path, days=True)
        if isinstance(value, str) and value.strip().lower() == "all":
            span = DEFAULT_LOOKBACK_EMA_SPAN_MINUTES
            reason = "all-history has no finite equivalent; use canonical 30-day memory default"
        else:
            span = max(1.0, float(value) * 1440.0)
            reason = "rolling days multiplied by 1440, floored at 1 minute; EMA behavior differs"
    span = normalize_lookback_ema_span_minutes(span)
    record_guess(changes, path, value, span, reason)
    return span


_RESULT_METADATA_KEYS = ("analysis", "metrics", "suite_metrics", "optimizer_anchor",
                         "optimizer_evaluation_contract")


def record_envelope_metadata(source, changes):
    for key in _RESULT_METADATA_KEYS:
        if key in source:
            record_guess(changes, key, source[key], None,
                         "discard outer saved-result metadata; old fitness and evaluator snapshots are invalid after migration")


def remove_result_metadata(config, changes, *, wrapped=False, path_prefix=""):
    """Match the export tool's exact saved-result allowlist, never unknown fields."""
    result_record = wrapped or any(key in config for key in _RESULT_METADATA_KEYS)
    for key in _RESULT_METADATA_KEYS:
        if key in config:
            record_guess(changes, path_prefix + key, config.pop(key), None,
                         "discard saved-result metadata; old fitness and evaluator snapshots are invalid after migration")
    if result_record:
        backtest = config.get("backtest", {})
        for key in ("coins", "cache_dir"):
            if key in backtest:
                record_guess(changes, f"{path_prefix}backtest.{key}", backtest.pop(key), None,
                             "discard derived dataset metadata; retain authored live/scenario coin selection")
