"""Saved results carry an authored config, observations, and replay provenance."""

from copy import deepcopy


RESULT_VERSION_KEY = "result_version"
RESULT_VERSION = 1
_RESULT_ENVELOPE_KEYS = frozenset(
    {RESULT_VERSION_KEY, "config", "provenance", "metrics", "suite_metrics"}
)
RESULT_METADATA_KEYS = frozenset({
    "analysis", "metrics", "suite_metrics", "optimizer_anchor",
    "optimizer_evaluation_contract",
})
RESULT_RUNTIME_ROOT_KEYS = ("disable_plotting", "_hlcvs_dataset_override_meta")
RESULT_RUNTIME_BACKTEST_KEYS = (
    "coins", "cache_dir", "cm_debug_level",
    "cm_progress_log_interval_seconds", "cm_remote_fetch_bar",
)


def build_result_record(config, *, provenance=None, metrics=None, suite_metrics=None):
    """Package an already exported authored config."""
    record = {
        RESULT_VERSION_KEY: RESULT_VERSION,
        "config": config,
        "provenance": {} if provenance is None else provenance,
    }
    if metrics is not None:
        record["metrics"] = metrics
    if suite_metrics is not None:
        record["suite_metrics"] = suite_metrics
    return record


def _validate_envelope(source):
    version = source[RESULT_VERSION_KEY]
    if type(version) is not int or version != RESULT_VERSION:
        raise ValueError(f"unsupported saved result version: {version!r}")
    unexpected = source.keys() - _RESULT_ENVELOPE_KEYS
    if unexpected:
        raise ValueError(f"unknown saved result fields: {', '.join(sorted(map(str, unexpected)))}")
    for field in ("config", "provenance"):
        if not isinstance(source.get(field), dict):
            raise TypeError(f"saved result {field} must be an object")
    config = source["config"]
    if not RESULT_METADATA_KEYS.isdisjoint(config):
        raise ValueError("saved result config must keep observations and provenance outside config")
    backtest = config.get("backtest")
    if isinstance(backtest, dict) and "coins" in backtest:
        raise ValueError("saved result config.backtest.coins belongs in provenance")


def _legacy_config_view(source, *, allow_unmarked_wrapper=False):
    result_record = not RESULT_METADATA_KEYS.isdisjoint(source)
    wrapped = source.get("config")
    if "bot" not in source and isinstance(wrapped, dict):
        if (
            result_record or allow_unmarked_wrapper
            or not RESULT_METADATA_KEYS.isdisjoint(wrapped)
        ):
            return wrapped, True
    return source, result_record


def result_config_view(source):
    """Borrow the configuration for readers of saved or plain records."""
    if RESULT_VERSION_KEY in source:
        _validate_envelope(source)
        return source["config"]
    return _legacy_config_view(source)[0]


def result_provenance_view(source):
    """Read replay evidence from a saved result, including pre-envelope records."""
    if RESULT_VERSION_KEY in source:
        _validate_envelope(source)
        return source["provenance"]
    config = result_config_view(source)
    provenance = {}
    for key in ("optimizer_anchor", "optimizer_evaluation_contract"):
        if key in source:
            provenance[key] = source[key]
        elif key in config:
            provenance[key] = config[key]
    backtest = config.get("backtest")
    if isinstance(backtest, dict) and "coins" in backtest:
        provenance["backtest_coins"] = backtest["coins"]
    return provenance


def extract_result_config(source: dict, *, allow_unmarked_wrapper: bool = False) -> dict:
    """Copy config input, removing only known result-record fields.

    Result records may contain the config inline or under ``config``. Cleanup
    exports may also explicitly unwrap documents without result metadata. Concrete
    dataset membership is provenance in these records, while authored membership
    remains live/scenario policy. Ordinary configs and unknown nested fields
    still pass intact to strict schema validation. No migration occurs here.
    """
    if not isinstance(source, dict):
        raise TypeError("config root must be an object")
    if RESULT_VERSION_KEY in source:
        return deepcopy(result_config_view(source))
    payload, result_record = _legacy_config_view(
        source, allow_unmarked_wrapper=allow_unmarked_wrapper,
    )
    payload = deepcopy(payload)
    for key in RESULT_METADATA_KEYS:
        payload.pop(key, None)
    if result_record:
        for key in RESULT_RUNTIME_ROOT_KEYS:
            payload.pop(key, None)
        if isinstance(payload.get("backtest"), dict):
            for key in RESULT_RUNTIME_BACKTEST_KEYS:
                payload["backtest"].pop(key, None)
    return payload
