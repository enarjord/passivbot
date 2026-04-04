from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from config.parse import load_raw_config


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def load_base_config() -> dict[str, Any]:
    inline = os.environ.get("PB_CONFIG_INLINE", "").strip()
    if inline:
        return parse_jsonish(inline, source="PB_CONFIG_INLINE")

    config_path = os.environ.get("PB_CONFIG_PATH", "").strip()
    if config_path:
        return load_raw_config(config_path)
    return {}


def parse_jsonish(raw: str, *, source: str) -> Any:
    try:
        import hjson

        return hjson.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Failed to parse {source}: {exc}") from exc


def parse_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise SystemExit(f"{name} must be one of: {sorted(TRUE_VALUES | FALSE_VALUES)}")


def parse_approved_coins(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped[0] in "[{":
        return parse_jsonish(stripped, source="PB_APPROVED_COINS")
    return [item.strip() for item in stripped.split(",") if item.strip()]


def set_nested(config: dict[str, Any], path: list[str], value: Any) -> None:
    current = config
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


def maybe_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    output_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/run/passivbot/config.runtime.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_base_config()
    if not isinstance(config, dict):
        raise SystemExit("Rendered container config must be a dictionary at the top level")

    set_nested(config, ["live", "user"], os.environ["PB_USER"])

    approved_coins = os.environ.get("PB_APPROVED_COINS", "").strip()
    if approved_coins:
        set_nested(config, ["live", "approved_coins"], parse_approved_coins(approved_coins))

    log_level = os.environ.get("PB_LOG_LEVEL", "").strip()
    if log_level:
        set_nested(config, ["logging", "level"], maybe_int(log_level))

    monitor_enabled = parse_bool_env("PB_MONITOR_ENABLED")
    if monitor_enabled is not None:
        set_nested(config, ["monitor", "enabled"], monitor_enabled)

    monitor_root = os.environ.get("PB_MONITOR_ROOT", "").strip()
    if monitor_root:
        set_nested(config, ["monitor", "root_dir"], monitor_root)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
