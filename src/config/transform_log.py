import time
from copy import deepcopy
from typing import Any, Iterable, List, Optional, Union


def _normalize_path(path: Union[str, Iterable[Any]]) -> str:
    if isinstance(path, str):
        return path
    if isinstance(path, Iterable):
        parts = []
        for item in path:
            if item is None:
                continue
            text = str(item).strip(".")
            if text:
                parts.append(text)
        return ".".join(parts)
    return str(path)


def _summarize_value(value: Any, *, max_str: int = 80, max_seq: int = 6) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= max_str else value[: max_str - 3] + "..."
    if isinstance(value, (list, tuple)):
        if len(value) <= max_seq:
            return [_summarize_value(v) for v in value]
        preview = [_summarize_value(v) for v in value[:max_seq]]
        preview.append(f"... (+{len(value) - max_seq})")
        return preview
    if isinstance(value, dict):
        keys = list(value.keys())
        preview_keys = keys[:max_seq]
        preview = {str(k): _summarize_value(value[k]) for k in preview_keys}
        if len(keys) > max_seq:
            preview["..."] = f"{len(keys) - max_seq} more keys"
        return {"__dict__": preview}
    return f"<{type(value).__name__}>"


class ConfigTransformTracker:
    def __init__(self) -> None:
        self._events: List[dict] = []

    def _push(self, event: dict) -> None:
        self._events.append(event)

    def add(self, path: Union[str, Iterable[Any]], value: Any = None) -> None:
        self._push(
            {
                "action": "add",
                "path": _normalize_path(path),
                "value": _summarize_value(value),
            }
        )

    def remove(self, path: Union[str, Iterable[Any]], value: Any = None) -> None:
        self._push(
            {
                "action": "remove",
                "path": _normalize_path(path),
                "value": _summarize_value(value),
            }
        )

    def rename(
        self,
        old_path: Union[str, Iterable[Any]],
        new_path: Union[str, Iterable[Any]],
        value: Any = None,
    ) -> None:
        self._push(
            {
                "action": "rename",
                "from": _normalize_path(old_path),
                "to": _normalize_path(new_path),
                "value": _summarize_value(value),
            }
        )

    def update(
        self,
        path: Union[str, Iterable[Any]],
        old_value: Any,
        new_value: Any,
    ) -> None:
        self._push(
            {
                "action": "update",
                "path": _normalize_path(path),
                "old": _summarize_value(old_value),
                "new": _summarize_value(new_value),
            }
        )

    def extend(self, events: List[dict]) -> None:
        for event in events:
            self._push(deepcopy(event))

    def summary(self) -> List[dict]:
        return deepcopy(self._events)

    def merge_details(self, base: Optional[dict] = None) -> dict:
        details = {} if base is None else deepcopy(base)
        if self._events:
            details = details or {}
            details["changes"] = self.summary()
        return details


def record_transform(config: dict, step: str, details: Optional[dict] = None) -> None:
    if not isinstance(config, dict):
        return
    entry = {"step": step, "ts_ms": int(time.time() * 1000)}
    if details:
        entry["details"] = deepcopy(details)
    config.setdefault("_transform_log", []).append(entry)
