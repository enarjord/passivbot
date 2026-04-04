import logging
from typing import Iterable, Optional

from .log_output import log_config_message


def add_missing_keys_recursively(src, dst, parent=None, verbose=True, tracker=None):
    if parent is None:
        parent = []
    for key in src:
        if key not in dst:
            log_config_message(verbose, logging.INFO, "Added missing %s to config.", ".".join(parent + [key]))
            dst[key] = src[key]
            if tracker is not None:
                tracker.add(parent + [key], src[key])
        elif isinstance(src[key], dict) and isinstance(dst.get(key), dict):
            add_missing_keys_recursively(src[key], dst[key], parent + [key], verbose, tracker=tracker)
        elif isinstance(src[key], dict):
            log_config_message(
                verbose,
                logging.INFO,
                "Skipping template subtree %s (template is dict, config is %s)",
                ".".join(parent + [key]),
                type(dst.get(key)).__name__,
            )
            continue
        else:
            if key not in dst:
                log_config_message(
                    verbose,
                    logging.INFO,
                    "Adding missing key -> val %s -> %s to config",
                    ".".join(parent + [key]),
                    src[key],
                )
                dst[key] = src[key]
                if tracker is not None:
                    tracker.add(parent + [key], src[key])


def remove_unused_keys_recursively(
    src,
    dst,
    parent=None,
    verbose=True,
    preserve: Optional[Iterable[Iterable[str]]] = None,
    tracker=None,
):
    if parent is None:
        parent = []
        if preserve is None:
            preserve_set = set()
        else:
            preserve_set = {tuple(p) for p in preserve}
    else:
        preserve_set = getattr(remove_unused_keys_recursively, "_preserve_set", set())

    def _path_is_preserved(path: Iterable[str]) -> bool:
        if not preserve_set:
            return False
        path_tuple = tuple(path)
        for preserved in preserve_set:
            if path_tuple[: len(preserved)] == preserved:
                return True
        return False

    if parent == []:
        remove_unused_keys_recursively._preserve_set = preserve_set

    if _path_is_preserved(parent):
        return
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return

    for key in list(dst.keys()):
        if isinstance(key, str):
            continue
        removed = dst.pop(key)
        current_path = parent + [str(key)]
        log_config_message(
            verbose, logging.INFO, "Removed unused key from config: %s", ".".join(current_path)
        )
        if tracker is not None:
            tracker.remove(current_path, removed)

    def _sort_key(value) -> tuple[str, str]:
        return (type(value).__name__, str(value))

    for key in sorted(list(dst.keys()), key=_sort_key):
        current_path = parent + [key]
        if _path_is_preserved(current_path):
            continue
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key not in src:
            removed = dst.pop(key)
            log_config_message(
                verbose,
                logging.INFO,
                "Removed unused key from config: %s",
                ".".join(map(str, current_path)),
            )
            if tracker is not None:
                tracker.remove(current_path, removed)
            continue
        src_val = src[key]
        dst_val = dst[key]
        if isinstance(dst_val, dict) and isinstance(src_val, dict):
            remove_unused_keys_recursively(
                src_val, dst_val, current_path, verbose=verbose, tracker=tracker
            )

    if parent == [] and hasattr(remove_unused_keys_recursively, "_preserve_set"):
        delattr(remove_unused_keys_recursively, "_preserve_set")
