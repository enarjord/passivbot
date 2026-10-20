"""Explicit released-v8 structural conversion of operational live settings."""

from copy import deepcopy

from config.field_registry import LIVE_OPERATIONAL_PATHS


def operational_live_selector(path):
    """Retarget released full selectors, including nested coin override selectors."""
    for old, current in LIVE_OPERATIONAL_PATHS.items():
        source = f"live.{old}"
        for prefix in ("", "."):
            marker = prefix + source
            start = 0 if not prefix else path.find(marker)
            if start < 0 or (not prefix and not path.startswith(source)):
                continue
            end = start + len(marker)
            if end == len(path) or path[end] == ".":
                return path[:start] + prefix + ".".join(current) + path[end:]
    return path


def move_operational_live(document, *, record, path=""):
    """Move only authored leaves; sparse coin documents never receive defaults."""
    if isinstance(document, list):
        for index, child in enumerate(document):
            move_operational_live(child, record=record, path=f"{path}[{index}]")
        return
    if not isinstance(document, dict):
        return
    live = document.get("live")
    if isinstance(live, dict):
        for old, (_, group, leaf) in LIVE_OPERATIONAL_PATHS.items():
            if old not in live:
                continue
            value = live.pop(old)
            block = live.setdefault(group, {})
            source = f"{path}.live.{old}".lstrip(".")
            target = f"{path}.live.{group}.{leaf}".lstrip(".")
            if not isinstance(block, dict) or (leaf in block and block[leaf] != value):
                raise ValueError(f"{source} conflicts with {target}")
            block[leaf] = deepcopy(value)
            record(
                source,
                value,
                {target: value},
                "grouped released operational live field",
            )
    for key, child in document.items():
        if not str(key).startswith("_"):
            move_operational_live(
                child, record=record, path=f"{path}.{key}".lstrip(".")
            )
