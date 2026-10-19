import logging

import hjson

from pure_funcs import remove_OD


def _unique_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"configuration field {key!r} is defined more than once")
        result[key] = value
    return result


def load_raw_config(config_path: str, *, log_errors: bool = True) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return remove_OD(hjson.load(f, object_pairs_hook=_unique_members))
    except Exception:
        if log_errors:
            logging.exception("failed to load config file %s", config_path)
        raise
