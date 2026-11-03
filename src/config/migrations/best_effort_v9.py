"""Observable guesses for the opt-in offline converter; runtime loading stays strict."""

from copy import deepcopy
import logging

from config.risk_memory import DEFAULT_LOOKBACK_EMA_SPAN_MINUTES, normalize_lookback_ema_span_minutes
from config.result_records import (
    RESULT_METADATA_KEYS,
    RESULT_RUNTIME_BACKTEST_KEYS,
    RESULT_RUNTIME_ROOT_KEYS,
)


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


def _metadata_keys():
    from config.authored import _METADATA_KEYS

    return (*sorted(RESULT_METADATA_KEYS), *sorted(_METADATA_KEYS))


def record_envelope_metadata(source, changes):
    for key in _metadata_keys():
        if key in source:
            record_guess(changes, key, source[key], None,
                         "discard outer saved-result metadata; old fitness and evaluator snapshots are invalid after migration")


def _is_explicit_coin_list(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(coin, str) and coin and coin == coin.strip() and coin.lower() != "all"
        and not any(char in coin for char in ",[]{}")
        and not coin.startswith(("./", "../", "/", "~"))
        and not coin.lower().endswith((".json", ".hjson", ".txt"))
        for coin in value
    )


def _discard_redundant_coin_snapshot(config, backtest, changes, *, path_prefix, base_config_path):
    from .shared_policy import is_shared_approved_source

    snapshot = backtest["coins"]
    live = config.get("live", {})
    approved = live.get("approved_coins") if isinstance(live, dict) else None
    ignored = live.get("ignored_coins", []) if isinstance(live, dict) else None
    if isinstance(ignored, dict) and ignored.keys() <= {"long", "short"}:
        no_ignores = all(value == [] for value in ignored.values())
    else:
        no_ignores = ignored == []
    if isinstance(approved, dict) and approved.keys() == {"long", "short"}:
        approvals = list(approved.values())
    else:
        approvals = [approved]
    path = f"{path_prefix}backtest.coins"
    if not (
        isinstance(snapshot, dict) and snapshot and no_ignores
        and all(_is_explicit_coin_list(coins)
                and is_shared_approved_source(coins, base_config_path=base_config_path)
                for coins in approvals)
        and all(
            isinstance(exchange, str) and exchange and exchange == exchange.strip()
            and _is_explicit_coin_list(coins)
            and all(set(coins) == set(side) for side in approvals)
            for exchange, coins in snapshot.items()
        )
    ):
        raise ValueError(
            f"{path} is an unknown configuration field and could not be verified as redundant "
            "with explicit live.approved_coins lists and no live.ignored_coins. "
            "Check the intended coin selection and "
            "remove the dataset snapshot before retrying."
        )
    record_guess(changes, path, backtest.pop("coins"), None,
                 "discard resolved dataset snapshot matching authored live.approved_coins")
    cache_dirs = backtest.get("cache_dir")
    if isinstance(cache_dirs, dict) and cache_dirs.keys() == snapshot.keys() and all(
        isinstance(directory, str) and directory.strip() for directory in cache_dirs.values()
    ):
        record_guess(changes, f"{path_prefix}backtest.cache_dir", backtest.pop("cache_dir"), None,
                     "discard cache paths for the verified redundant dataset snapshot")


def remove_result_metadata(config, changes, *, wrapped=False, path_prefix="",
                           base_config_path="", recognized_only=False, log_prefix=""):
    """Remove known result metadata or a proven redundant coin snapshot."""
    result_record = wrapped or any(key in config for key in RESULT_METADATA_KEYS)
    if recognized_only and not result_record:
        return
    for key in _metadata_keys():
        if key in config:
            record_guess(changes, path_prefix + key, config.pop(key), None,
                         "discard metadata; migrate authored config without cached results or hydration snapshots",
                         log_prefix=log_prefix)
    backtest = config.get("backtest", {})
    if result_record:
        for key in RESULT_RUNTIME_ROOT_KEYS:
            if key in config:
                record_guess(changes, path_prefix + key, config.pop(key), None,
                             "discard saved-result runtime metadata", log_prefix=log_prefix)
        if not isinstance(backtest, dict):
            return
        for key in RESULT_RUNTIME_BACKTEST_KEYS:
            if key in backtest:
                record_guess(changes, f"{path_prefix}backtest.{key}", backtest.pop(key), None,
                             "discard saved-result backtest metadata; retain authored live/scenario policy",
                             log_prefix=log_prefix)
    elif isinstance(backtest, dict):
        for key in ("coins", "cache_dir"):
            if backtest.get(key) == {}:
                record_guess(changes, f"{path_prefix}backtest.{key}", backtest.pop(key), None,
                             "discard empty backtest preparation metadata", log_prefix=log_prefix)
        if "coins" in backtest:
            _discard_redundant_coin_snapshot(config, backtest, changes, path_prefix=path_prefix,
                                             base_config_path=base_config_path)
