"""Move the public cooldown config while retaining its runtime/optimizer key."""

import logging
from copy import deepcopy

OLD = "risk.entry_cooldown_minutes"
NEW = "entry_cooldown.base_duration_minutes"


def _put(target, key, value, *, path):
    if key in target and target[key] != value:
        logging.warning(
            "[config] Conflicting entry cooldown values at %s: explicit canonical "
            "value %r wins over legacy value %r. Review this config before use.",
            path,
            target[key],
            value,
        )
    else:
        target.setdefault(key, value)


def migrate_entry_cooldown_tree(document, *, path="config", tracker=None):
    """Migrate authored leaves before defaults or independently authored patches merge."""
    if isinstance(document, list):
        for index, item in enumerate(document):
            migrate_entry_cooldown_tree(item, path=f"{path}.{index}", tracker=tracker)
        return
    if not isinstance(document, dict):
        return
    risk = document.get("risk")
    if isinstance(risk, dict) and "entry_cooldown_minutes" in risk:
        cooldown = document.setdefault("entry_cooldown", {})
        if not isinstance(cooldown, dict):
            raise TypeError(f"{path}.entry_cooldown must be a mapping")
        value = risk.pop("entry_cooldown_minutes")
        _put(cooldown, "base_duration_minutes", value, path=f"{path}.{NEW}")
        if tracker is not None:
            prefix = path.split(".")[1:]
            tracker.rename(
                [*prefix, *OLD.split(".")],
                [*prefix, *NEW.split(".")],
                cooldown["base_duration_minutes"],
            )
    for key, value in list(document.items()):
        if not isinstance(key, str) or key.startswith("_"):
            continue
        migrate_entry_cooldown_tree(value, path=f"{path}.{key}", tracker=tracker)
        if key == OLD or key.endswith("." + OLD):
            new = key[: -len(OLD)] + NEW
            document.pop(key)
            _put(document, new, value, path=f"{path}.{new}")


def migrate_entry_cooldown(config, *, tracker=None):
    from ..param_paths import resolve_dotted_config_path
    from suite_runner import _normalize_scenario_overrides

    for scenario in config.get("backtest", {}).get("scenarios", []):
        if not isinstance(scenario, dict) or not isinstance(
            scenario.get("overrides"), dict
        ):
            continue
        overrides = _normalize_scenario_overrides(scenario["overrides"])
        expanded = {}
        for key, value in overrides.items():
            # Dotted group replacements are atomic in the suite runner. Extract
            # the moved leaf while retaining the rest of the replaced risk group.
            if (
                key.endswith(".risk")
                and isinstance(value, dict)
                and "entry_cooldown_minutes" in value
            ):
                value = deepcopy(value)
                expanded[key + ".entry_cooldown_minutes"] = value.pop(
                    "entry_cooldown_minutes"
                )
            elif (
                key.endswith(".entry_cooldown")
                and isinstance(value, dict)
                and "base_duration_minutes" in value
            ):
                value = deepcopy(value)
                expanded[key + ".base_duration_minutes"] = value.pop(
                    "base_duration_minutes"
                )
                if not value:
                    continue
            expanded[key] = value
        items = []
        for key, value in expanded.items():
            resolved = resolve_dotted_config_path(config, key)
            canonical = (
                ".".join(resolved)
                if resolved and resolved[-2:] == tuple(NEW.split("."))
                else key
            )
            items.append((key, canonical, value))
        if expanded != overrides or any(
            key != canonical for key, canonical, _ in items
        ):
            migrated = {}
            # Canonical spellings win regardless of source insertion order.
            for key, canonical, value in sorted(
                items, key=lambda item: (not item[0].endswith(NEW), item[0] != item[1])
            ):
                _put(migrated, canonical, deepcopy(value), path=f"scenario.{canonical}")
            scenario["overrides"] = migrated
    migrate_entry_cooldown_tree(config, tracker=tracker)
