"""Extract authored configuration from saved results without interpreting provenance."""

from copy import deepcopy


RESULT_METADATA_KEYS = frozenset({
    "analysis", "metrics", "suite_metrics", "optimizer_anchor",
    "optimizer_evaluation_contract",
})


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
    payload = source
    result_record = any(key in payload for key in RESULT_METADATA_KEYS)
    if "bot" not in payload and isinstance(payload.get("config"), dict):
        wrapped = payload["config"]
        if (
            result_record or allow_unmarked_wrapper
            or any(key in wrapped for key in RESULT_METADATA_KEYS)
        ):
            payload = wrapped
            result_record = True
    payload = deepcopy(payload)
    for key in RESULT_METADATA_KEYS:
        payload.pop(key, None)
    if result_record:
        for key in ("disable_plotting", "_hlcvs_dataset_override_meta"):
            payload.pop(key, None)
        if isinstance(payload.get("backtest"), dict):
            for key in (
                "coins", "cache_dir", "cm_debug_level",
                "cm_progress_log_interval_seconds", "cm_remote_fetch_bar",
            ):
                payload["backtest"].pop(key, None)
    return payload
