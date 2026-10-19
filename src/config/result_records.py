"""Extract authored configuration from saved results without interpreting provenance."""

from copy import deepcopy


RESULT_METADATA_KEYS = frozenset({
    "analysis", "metrics", "suite_metrics", "optimizer_anchor",
    "optimizer_evaluation_contract",
})


def extract_result_config(source: dict) -> dict:
    """Copy config input, removing only known result-record fields.

    Result records may contain the config inline or under ``config``. Concrete
    dataset membership is provenance in these records, while authored membership
    remains live/scenario policy. Ordinary configs and unknown nested fields
    still pass intact to strict schema validation. No migration occurs here.
    """
    if not isinstance(source, dict):
        raise TypeError("config root must be an object")
    payload = source
    result_record = any(key in payload for key in RESULT_METADATA_KEYS)
    if "bot" not in payload and isinstance(payload.get("config"), dict):
        payload = payload["config"]
        result_record = True
    payload = deepcopy(payload)
    for key in RESULT_METADATA_KEYS:
        payload.pop(key, None)
    if result_record and isinstance(payload.get("backtest"), dict):
        payload["backtest"].pop("coins", None)
    return payload
