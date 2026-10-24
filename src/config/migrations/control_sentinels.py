"""Released-v8 control sentinels become explicit final-v9 policy values."""

import math

from config.control_sentinels import (
    FORCED_MODE_PATHS, WARMUP_CAP_PATH, WARMUP_CONCURRENCY_PATH,
    optional_forced_mode, warmup_cap, warmup_concurrency,
)


def migrate_control_sentinels(document, *, record):
    def number(value, path):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{path} must be numeric, not a boolean or object")
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"{path} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{path} must be finite")
        return result

    def convert(parts, value, path):
        if any(parts[-len(mode):] == mode for mode in FORCED_MODE_PATHS):
            return optional_forced_mode(None if value == "" else value, path=path)
        if parts[-len(WARMUP_CAP_PATH):] == WARMUP_CAP_PATH:
            numeric = None if value is None else number(value, path)
            return warmup_cap(None if numeric == 0 else numeric, path=path)
        if parts[-len(WARMUP_CONCURRENCY_PATH):] == WARMUP_CONCURRENCY_PATH:
            if value == "auto":
                return value
            numeric = number(value, path)
            if numeric == 0:
                return "auto"
            if not numeric.is_integer():
                raise ValueError(f"{path} must be an integer")
            return warmup_concurrency(int(numeric), path=path)
        return value

    def visit(node, parts=(), path=""):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, parts, f"{path}[{index}]")
        elif isinstance(node, dict):
            for key, old in list(node.items()):
                if key.startswith("_"):
                    continue
                child_parts = (*parts, *key.split("."))
                current = f"{path}.{key}".lstrip(".")
                new = convert(child_parts, old, current)
                if new != old or type(new) is not type(old):
                    node[key] = new
                    record(current, old, new, "explicit released-v8 control sentinel conversion")
                visit(node[key], child_parts, current)

    visit(document)
