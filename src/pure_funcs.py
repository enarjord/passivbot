import datetime
import pprint
import json
import re
from collections import OrderedDict
from hashlib import sha256

import numpy as np


__all__ = [
    "safe_filename",
    "numpyize",
    "denumpyize",
    "ts_to_date",
    "config_pretty_str",
    "sort_dict_keys",
    "filter_orders",
    "flatten",
    "floatify",
    "shorten_custom_id",
    "determine_pos_side_ccxt",
    "calc_hash",
    "ensure_millis",
    "multi_replace",
    "str2bool",
    "determine_side_from_order_tuple",
    "remove_OD",
    "log_dict_changes",
]


def safe_filename(symbol: str) -> str:
    """Convert a symbol to a filesystem-safe string."""
    return re.sub(r'[<>:"/\|?*]', "_", symbol)


def numpyize(x):
    if isinstance(x, (list, tuple)):
        return np.array([numpyize(e) for e in x])
    if isinstance(x, dict):
        return {k: numpyize(v) for k, v in x.items()}
    return x


def denumpyize(x):
    if isinstance(x, (np.float64, np.float32, np.float16)):
        return float(x)
    if isinstance(x, (np.int64, np.int32, np.int16, np.int8)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [denumpyize(e) for e in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (dict, OrderedDict)):
        return {k: denumpyize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [denumpyize(z) for z in x]
    if isinstance(x, tuple):
        return tuple(denumpyize(z) for z in x)
    return x


def ts_to_date(timestamp: float) -> str:
    if timestamp > 253402297199:
        dt = datetime.datetime.utcfromtimestamp(timestamp / 1000)
    else:
        dt = datetime.datetime.utcfromtimestamp(timestamp)
    return dt.isoformat().replace(" ", "T")


def config_pretty_str(config: dict) -> str:
    pretty = pprint.pformat(config)
    for before, after in [("'", '"'), ("True", "true"), ("False", "false"), ("None", "null")]:
        pretty = pretty.replace(before, after)
    return pretty


def sort_dict_keys(d):
    if isinstance(d, list):
        return [sort_dict_keys(e) for e in d]
    if not isinstance(d, dict):
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def filter_orders(
    actual_orders,
    ideal_orders,
    keys=("symbol", "side", "qty", "price"),
):
    """Return orders to cancel and to create by comparing actual vs ideal."""

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []

    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_cropped = [{k: o[k] for k in keys} for o in actual_orders]

    for cropped, original in zip(ideal_cropped, ideal_orders):
        matches = [(a_c, a_o) for a_c, a_o in zip(actual_cropped, actual_orders) if a_c == cropped]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(original)
    return actual_orders, orders_to_create


def flatten(nested):
    return [item for sublist in nested for item in sublist]


def floatify(xs):
    if isinstance(xs, (int, float)):
        return float(xs)
    if isinstance(xs, str):
        try:
            return float(xs)
        except (ValueError, TypeError):
            return xs
    if isinstance(xs, bool):
        return xs
    if isinstance(xs, list):
        return [floatify(x) for x in xs]
    if isinstance(xs, tuple):
        return tuple(floatify(x) for x in xs)
    if isinstance(xs, dict):
        return {k: floatify(v) for k, v in xs.items()}
    return xs


def shorten_custom_id(id_: str) -> str:
    replacements = [
        ("clock", "clk"),
        ("close", "cls"),
        ("entry", "etr"),
        ("_", ""),
        ("normal", "nrml"),
        ("long", "lng"),
        ("short", "shrt"),
        ("primary", "prm"),
        ("unstuck", "ustk"),
        ("partial", "prtl"),
        ("panic", "pnc"),
    ]
    for before, after in replacements:
        id_ = id_.replace(before, after)
    return id_


def determine_pos_side_ccxt(open_order: dict) -> str:
    info = open_order.get("info", open_order)
    if "positionIdx" in info:
        idx = float(info["positionIdx"])
        if idx == 1.0:
            return "long"
        if idx == 2.0:
            return "short"

    keys_map = {key.lower().replace("_", ""): key for key in info}
    for pos_key in ("posside", "positionside"):
        if pos_key in keys_map:
            return info[keys_map[pos_key]].lower()

    if info.get("side", "").lower() == "buy":
        if "reduceonly" in keys_map:
            return "long" if not info[keys_map["reduceonly"]] else "short"
        if "closedsize" in keys_map:
            return "long" if float(info[keys_map["closedsize"]]) != 0.0 else "short"

    for key in ["order_link_id", "clOrdId", "clientOid", "orderLinkId"]:
        if key in info:
            value = info[key].lower()
            if "long" in value or "lng" in value:
                return "long"
            if "short" in value or "shrt" in value:
                return "short"
    return "both"


def calc_hash(data) -> str:
    data_string = json.dumps(data, sort_keys=True)
    return sha256(data_string.encode("utf-8")).hexdigest()


def ensure_millis(timestamp):
    """Normalize various timestamp formats to milliseconds."""
    if not isinstance(timestamp, (int, float)):
        raise TypeError("Timestamp must be an int or float")

    ts = float(timestamp)
    if ts > 1e16:  # nanoseconds
        return ts / 1e6
    if ts > 1e14:  # microseconds
        return ts / 1e3
    if ts > 1e11:  # milliseconds
        return ts
    if ts > 1e9:  # seconds with decimals
        return ts * 1e3
    if ts > 1e6:  # seconds
        return ts * 1e3
    raise ValueError("Timestamp value too small or unrecognized format")


def multi_replace(input_data, replacements):
    if isinstance(input_data, str):
        for old, new in replacements:
            input_data = input_data.replace(old, new)
        return input_data
    if isinstance(input_data, list):
        return [multi_replace(item, replacements) for item in input_data]
    if isinstance(input_data, dict):
        return {key: multi_replace(value, replacements) for key, value in input_data.items()}
    return input_data


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


def determine_side_from_order_tuple(order_tuple):
    side_info = order_tuple[2]
    if "long" in side_info:
        if "entry" in side_info:
            return "buy"
        if "close" in side_info:
            return "sell"
    elif "short" in side_info:
        if "entry" in side_info:
            return "sell"
        if "close" in side_info:
            return "buy"
    raise ValueError(f"malformed order tuple {order_tuple}")


def remove_OD(d):
    if isinstance(d, dict):
        return {k: remove_OD(v) for k, v in d.items()}
    if isinstance(d, list):
        return [remove_OD(x) for x in d]
    return d


def log_dict_changes(d1, d2, parent_key=""):
    """Return a summary of differences between two nested dictionaries."""

    changes = {"added": [], "removed": [], "changed": []}
    if not d1:
        changes["added"].extend([f"{parent_key}{k}: {v}" for k, v in (d2 or {}).items()])
        return changes
    if not d2:
        changes["removed"].extend([f"{parent_key}{k}: {v}" for k, v in (d1 or {}).items()])
        return changes

    for key in sorted(set(d1.keys()) | set(d2.keys())):
        new_parent = f"{parent_key}{key}." if parent_key else f"{key}."
        if key in d1 and key in d2:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested = log_dict_changes(d1[key], d2[key], new_parent)
                for change_type, items in nested.items():
                    changes[change_type].extend(items)
            elif d1[key] != d2[key]:
                changes["changed"].append(f"{parent_key}{key}: {d1[key]} -> {d2[key]}")
        elif key in d2:
            if isinstance(d2[key], dict):
                nested = log_dict_changes({}, d2[key], new_parent)
                changes["added"].extend(nested["added"])
            else:
                changes["added"].append(f"{parent_key}{key}: {d2[key]}")
        else:
            if isinstance(d1[key], dict):
                nested = log_dict_changes(d1[key], {}, new_parent)
                changes["removed"].extend(nested["removed"])
            else:
                changes["removed"].append(f"{parent_key}{key}: {d1[key]}")
    return changes
