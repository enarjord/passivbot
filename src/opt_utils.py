import json
import logging
import math
import msgpack
from typing import Any
import passivbot_rust as pbr


def dominates(p0, p1):
    better_in_one = False
    for a, b in zip(p0, p1):
        if a < b:
            better_in_one = True
        elif a > b:
            return False
    return better_in_one


def dominates_d(x, y, higher_is_better):
    better_in_one = False
    for xi, yi, hib in zip(x, y, higher_is_better):
        if hib:
            if xi > yi:
                better_in_one = True
            elif xi < yi:
                return False
        else:
            if xi < yi:
                better_in_one = True
            elif xi > yi:
                return False
    return better_in_one


def update_pareto_front(new_index, new_obj, current_front, objectives_dict, higher_is_better):
    for idx in current_front:
        if dominates_d(objectives_dict[idx], new_obj, higher_is_better):
            return current_front
    new_front = [
        idx
        for idx in current_front
        if not dominates_d(new_obj, objectives_dict[idx], higher_is_better)
    ]
    new_front.append(new_index)
    return new_front


def calc_dist(p0, p1):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))


def calc_normalized_dist(point, ideal, w0_min, w0_max, w1_min, w1_max):
    norm_point = [
        (p - min_v) / (max_v - min_v) if max_v > min_v else p
        for p, min_v, max_v in zip(point, mins, maxs)
    ]
    norm_ideal = [
        (i - min_v) / (max_v - min_v) if max_v > min_v else i
        for i, min_v, max_v in zip(ideal, mins, maxs)
    ]
    return math.sqrt(sum((p - i) ** 2 for p, i in zip(norm_point, norm_ideal)))


def format_distance(dist: float) -> str:
    """Format distance to fixed-width string for lexicographical sorting."""
    return f"{dist:08.4f}"


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [make_json_serializable(e) for e in obj]
    elif isinstance(obj, list):
        return [make_json_serializable(e) for e in obj]
    else:
        return obj


def gprint(verbose):
    return print if verbose else (lambda *args, **kwargs: None)


def generate_diffs(dictlist):
    """Yield diffs between consecutive dicts in dictlist, supporting nested dicts."""

    def dict_diff(d1, d2):
        diff = {}
        for k in d2:
            if k not in d1:
                diff[k] = d2[k]
            elif isinstance(d2[k], dict) and isinstance(d1.get(k), dict):
                nested = dict_diff(d1[k], d2[k])
                if nested:
                    diff[k] = nested
            elif d1[k] != d2[k]:
                diff[k] = d2[k]
        return diff

    prev = {}
    for d in dictlist:
        if not prev:
            yield d
        else:
            yield dict_diff(prev, d)
        prev = d


def deep_updated(base, diff):
    out = {}  # build a fresh dict
    keys = base.keys() | diff.keys()
    for k in keys:
        if k in diff:
            v2 = diff[k]
            if isinstance(v2, dict) and isinstance(base.get(k), dict):
                out[k] = deep_updated(base[k], v2)
            else:
                out[k] = v2
        else:
            out[k] = base[k]
    return out


def generate_incremental_diff(prev, current):
    """Return the diff between two dicts."""

    def dict_diff(d1, d2):
        diff = {}
        for k in d2:
            if k not in d1:
                diff[k] = d2[k]
            elif isinstance(d2[k], dict) and isinstance(d1.get(k), dict):
                nested = dict_diff(d1[k], d2[k])
                if nested:
                    diff[k] = nested
            elif d1[k] != d2[k]:
                diff[k] = d2[k]
        return diff

    return dict_diff(prev or {}, current)


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
