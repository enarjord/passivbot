import msgpack
from typing import Any
import passivbot_rust as pbr

_DIFF_DELETE_KEY = "__passivbot_diff_delete__"
_DIFF_DELETE_MARKER = {_DIFF_DELETE_KEY: True}


def _is_diff_delete_marker(value: Any) -> bool:
    return isinstance(value, dict) and value == _DIFF_DELETE_MARKER


def _dict_diff(d1: dict, d2: dict) -> dict:
    diff = {}
    for k in d1.keys() - d2.keys():
        diff[k] = _DIFF_DELETE_MARKER.copy()
    for k in d2:
        if k not in d1:
            diff[k] = d2[k]
        elif isinstance(d2[k], dict) and isinstance(d1.get(k), dict):
            nested = _dict_diff(d1[k], d2[k])
            if nested:
                diff[k] = nested
        elif d1[k] != d2[k]:
            diff[k] = d2[k]
    return diff


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [make_json_serializable(e) for e in obj]
    elif isinstance(obj, list):
        return [make_json_serializable(e) for e in obj]
    else:
        return obj


def generate_diffs(dictlist):
    """Yield diffs between consecutive dicts in dictlist, supporting nested dicts."""
    prev = {}
    for d in dictlist:
        if not prev:
            yield d
        else:
            yield _dict_diff(prev, d)
        prev = d


def deep_updated(base, diff):
    out = {}  # build a fresh dict
    keys = base.keys() | diff.keys()
    for k in keys:
        if k in diff:
            v2 = diff[k]
            if _is_diff_delete_marker(v2):
                continue
            if isinstance(v2, dict) and isinstance(base.get(k), dict):
                out[k] = deep_updated(base[k], v2)
            else:
                out[k] = v2
        else:
            out[k] = base[k]
    return out


def generate_incremental_diff(prev, current):
    """Return the diff between two dicts."""
    return _dict_diff(prev or {}, current)


def apply_diffs(difflist, base=None):
    """Yield full dicts by applying diffs, supporting nested dicts."""
    current = base or {}
    for d in difflist:
        current = deep_updated(current, d)
        yield current


def load_results(filepath):
    """
    Generator that yields each full config by applying diffs.
    No need to distinguish between full configs and diffs.
    """
    with open(filepath, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        current = {}
        for entry in unpacker:
            for full_config in apply_diffs([entry], base=current):
                current = full_config
            yield current


def round_floats(obj: Any, sig_digits: int = 6) -> Any:
    if isinstance(obj, float):
        return pbr.round_dynamic(obj, sig_digits)
    elif isinstance(obj, dict):
        return {k: round_floats(v, sig_digits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, sig_digits) for v in obj]
    else:
        return obj


def round_floats_sig_digits(obj: Any, sig_digits: int) -> Any:
    if isinstance(obj, float):
        return pbr.round_dynamic(obj, sig_digits)
    elif isinstance(obj, dict):
        return {k: round_floats_sig_digits(v, sig_digits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats_sig_digits(v, sig_digits) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([round_floats_sig_digits(v, sig_digits) for v in obj])
    else:
        return obj


def round_floats_step(obj: Any, step: float) -> Any:
    if isinstance(obj, float):
        return pbr.round_(obj, step)
    elif isinstance(obj, dict):
        return {k: round_floats_step(v, step) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats_step(v, step) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([round_floats_step(v, step) for v in obj])
    else:
        return obj


def quantize_floats(obj: Any, sig_digits: int = None, step: float = None) -> Any:
    """
    if step is given, round by step
    else, round by sig_digits
    """
    if step is None:
        if sig_digits is None:
            raise Exception("must provide sig_digits or step")
        return round_floats_sig_digits(obj, sig_digits)
    else:
        return round_floats_step(obj, step)
