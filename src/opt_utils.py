import json
import logging
import math


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
