def require_config_value(config: dict, dotted_path: str):
    parts = dotted_path.split(".")
    if not parts:
        raise KeyError("empty dotted_path")
    current = config
    traversed = []
    for part in parts:
        traversed.append(part)
        if not isinstance(current, dict):
            raise KeyError(
                f"config path {'/'.join(traversed[:-1])} is not a dict (required for '{dotted_path}')"
            )
        if part not in current:
            raise KeyError(f"config missing required key '{'.'.join(traversed)}'")
        current = current[part]
    return current


def require_config_dict(config: dict, dotted_path: str) -> dict:
    value = require_config_value(config, dotted_path)
    if not isinstance(value, dict):
        raise TypeError(f"config.{dotted_path} must be a dict; got {type(value).__name__}")
    return value


def get_optional_config_value(config: dict, dotted_path: str, default=None):
    parts = dotted_path.split(".")
    if not parts:
        return default
    current = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def require_live_value(config: dict, key: str):
    return require_config_value(config, f"live.{key}")


def get_optional_live_value(config: dict, key: str, default=None):
    return get_optional_config_value(config, f"live.{key}", default)
