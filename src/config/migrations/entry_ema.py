"""Move trailing-martingale price horizons to their entry-owned config paths."""

import logging
from copy import deepcopy

SPANS = ("ema_span_0", "ema_span_1")
KIND = "trailing_martingale"


def _move(mapping, old, new, *, path, tracker=None):
    value = mapping.pop(old)
    if new in mapping:
        if mapping[new] != value:
            logging.warning(
                "[config] Cannot preserve conflicting EMA values at %s: discarded legacy %s=%r; "
                "explicit %s=%r wins. Review this config before use.",
                path,
                old,
                value,
                new,
                mapping[new],
            )
    else:
        mapping[new] = value
    if tracker is not None:
        tracker.rename([*path.split("."), old], [*path.split("."), new], mapping[new])


def migrate_entry_ema_tree(document, *, path="config", tracker=None):
    """Migrate explicit leaves only, including partial and external coin patches.

    Called before merging independently authored patches, so ordinary file/inline
    precedence is retained. Metadata and unrelated strategies are left alone.
    """
    if isinstance(document, list):
        for index, item in enumerate(document):
            migrate_entry_ema_tree(item, path=f"{path}.{index}", tracker=tracker)
        return
    if not isinstance(document, dict):
        return
    for key, value in list(document.items()):
        if not isinstance(key, str) or key.startswith("_"):
            continue
        child_path = f"{path}.{key}"
        if key.split(".")[-1] == KIND and isinstance(value, dict):
            for span in SPANS:
                if span not in value:
                    continue
                entry = value.setdefault("entry", {})
                if not isinstance(entry, dict):
                    raise TypeError(f"{child_path}.entry must be a mapping")
                old = value.pop(span)
                if span in entry and entry[span] != old:
                    logging.warning(
                        "[config] Cannot preserve conflicting EMA values: %s.%s=%r is discarded; "
                        "explicit %s.entry.%s=%r wins. Review this config before use.",
                        child_path,
                        span,
                        old,
                        child_path,
                        span,
                        entry[span],
                    )
                else:
                    entry.setdefault(span, old)
                if tracker is not None:
                    tracker.rename(
                        child_path.split(".")[1:] + [span],
                        child_path.split(".")[1:] + ["entry", span],
                        entry[span],
                    )
        migrate_entry_ema_tree(value, path=child_path, tracker=tracker)
        for span in SPANS:
            suffix = f"{KIND}.{span}"
            if key == suffix or key.endswith("." + suffix):
                new = key[: -len(span)] + "entry." + span
                _move(document, key, new, path=path, tracker=tracker)
                break


def migrate_entry_ema_spans(config, *, tracker=None):
    # Scenario mappings can mix nested paths, dotted paths, and active-strategy
    # aliases. Resolve leaf collisions before any defaults/overrides are applied.
    from ..param_paths import resolve_dotted_config_path
    from suite_runner import _normalize_scenario_overrides

    for scenario in config.get("backtest", {}).get("scenarios", []):
        if not isinstance(scenario, dict) or not isinstance(
            scenario.get("overrides"), dict
        ):
            continue
        overrides = _normalize_scenario_overrides(scenario["overrides"])
        migrated = {}
        # Canonical keys win independently of input insertion order.
        items = []
        for key, value in overrides.items():
            resolved = resolve_dotted_config_path(config, key)
            canonical = (
                ".".join(resolved)
                if resolved[-3:] in [(KIND, "entry", span) for span in SPANS]
                else key
            )
            items.append((key, canonical, value))
        if not any(key != canonical for key, canonical, _ in items):
            continue
        for key, canonical, value in sorted(
            items, key=lambda item: (".entry.ema_span_" in item[0], item[0] == item[1])
        ):
            if canonical in migrated and migrated[canonical] != value:
                logging.warning(
                    "[config] Cannot preserve conflicting scenario EMA/override values for %s "
                    "in %r; explicit canonical path wins. Review this config before use.",
                    canonical,
                    scenario.get("label", "<unnamed>"),
                )
            migrated[canonical] = deepcopy(value)
        scenario["overrides"] = migrated
    migrate_entry_ema_tree(config, tracker=tracker)
