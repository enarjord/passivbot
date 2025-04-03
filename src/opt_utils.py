import json
import logging
import math
import msgpack


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
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def calc_normalized_dist(point, ideal, w0_min, w0_max, w1_min, w1_max):
    """Calculate normalized Euclidean distance from ideal point."""
    if w0_max > w0_min:
        norm_w0 = (point[0] - w0_min) / (w0_max - w0_min)
        ideal_w0 = (ideal[0] - w0_min) / (w0_max - w0_min)
    else:
        norm_w0 = point[0]
        ideal_w0 = ideal[0]
    if w1_max > w1_min:
        norm_w1 = (point[1] - w1_min) / (w1_max - w1_min)
        ideal_w1 = (ideal[1] - w1_min) / (w1_max - w1_min)
    else:
        norm_w1 = point[1]
        ideal_w1 = ideal[1]
    return math.sqrt((norm_w0 - ideal_w0) ** 2 + (norm_w1 - ideal_w1) ** 2)


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

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    current = base or {}
    for diff in difflist:
        current = deep_update(current, diff.copy())
        yield current.copy()


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
