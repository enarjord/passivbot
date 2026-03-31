import logging

import hjson

from pure_funcs import remove_OD


def load_raw_config(config_path: str, *, log_errors: bool = True) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return remove_OD(hjson.load(f))
    except Exception:
        if log_errors:
            logging.exception("failed to load config file %s", config_path)
        raise
