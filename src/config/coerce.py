from copy import deepcopy
from typing import Optional

from .access import require_config_dict
from .pnl_lookback import normalize_pnls_max_lookback_days_config_value
from .schema import get_template_config


HSL_COOLDOWN_POSITION_POLICIES = (
    "normal",
    "panic",
    "tp_only",
    "graceful_stop",
    "manual",
)
HSL_SIGNAL_MODES = ("pside", "unified")
MONITOR_BOOL_KEYS = (
    "enabled",
    "retain_price_ticks",
    "retain_candles",
    "retain_fills",
    "compress_rotated_segments",
    "emit_completed_candles",
    "include_raw_fill_payloads",
)
LOGGING_BOOL_KEYS = (
    "persist_to_file",
    "rotation",
)
PYMOO_ALGORITHMS = ("auto", "nsga2", "nsga3")
PYMOO_REF_DIR_METHODS = ("das_dennis",)


def normalize_hsl_cooldown_position_policy(
    value, path: str = "live.hsl_position_during_cooldown_policy"
) -> str:
    policy = str(value)
    if policy not in HSL_COOLDOWN_POSITION_POLICIES:
        allowed = ", ".join(HSL_COOLDOWN_POSITION_POLICIES)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {policy!r}")
    return policy


def normalize_hsl_signal_mode(value, path: str = "live.hsl_signal_mode") -> str:
    mode = str(value)
    if mode not in HSL_SIGNAL_MODES:
        allowed = ", ".join(HSL_SIGNAL_MODES)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {mode!r}")
    return mode


def normalize_monitor_config(config: dict) -> None:
    monitor_cfg = require_config_dict(config, "monitor")
    root_dir = str(monitor_cfg["root_dir"]).strip()
    if not root_dir:
        raise ValueError("config.monitor.root_dir must be a non-empty string")
    monitor_cfg["root_dir"] = root_dir

    for key in MONITOR_BOOL_KEYS:
        monitor_cfg[key] = bool(monitor_cfg[key])

    numeric_rules = (
        ("snapshot_interval_seconds", float, lambda x: x > 0.0, "must be > 0"),
        ("checkpoint_interval_minutes", float, lambda x: x >= 0.0, "must be >= 0"),
        ("event_rotation_mb", float, lambda x: x > 0.0, "must be > 0"),
        ("event_rotation_minutes", float, lambda x: x > 0.0, "must be > 0"),
        ("retain_days", float, lambda x: x >= 0.0, "must be >= 0"),
        ("max_total_bytes", int, lambda x: x > 0, "must be > 0"),
        ("price_tick_min_interval_ms", int, lambda x: x >= 0, "must be >= 0"),
    )
    for key, caster, predicate, message in numeric_rules:
        try:
            value = caster(monitor_cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config.monitor.{key} {message}") from exc
        if not predicate(value):
            raise ValueError(f"config.monitor.{key} {message}")
        monitor_cfg[key] = value


def normalize_logging_config(config: dict) -> None:
    logging_cfg = require_config_dict(config, "logging")
    log_dir = str(logging_cfg["dir"]).strip()
    if not log_dir:
        raise ValueError("config.logging.dir must be a non-empty string")
    logging_cfg["dir"] = log_dir

    for key in LOGGING_BOOL_KEYS:
        logging_cfg[key] = bool(logging_cfg[key])

    numeric_rules = (
        ("max_bytes_mb", float, lambda x: x > 0.0, "must be > 0"),
        ("backup_count", int, lambda x: x >= 0, "must be >= 0"),
    )
    for key, caster, predicate, message in numeric_rules:
        try:
            value = caster(logging_cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config.logging.{key} {message}") from exc
        if not predicate(value):
            raise ValueError(f"config.logging.{key} {message}")
        logging_cfg[key] = value


def normalize_pymoo_algorithm(value, path: str = "config.optimize.pymoo.algorithm") -> str:
    algorithm = str(value).strip().lower()
    if algorithm not in PYMOO_ALGORITHMS:
        allowed = ", ".join(PYMOO_ALGORITHMS)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {value!r}")
    return algorithm


def normalize_pymoo_ref_dir_method(
    value, path: str = "config.optimize.pymoo.algorithms.nsga3.ref_dirs.method"
) -> str:
    method = str(value).strip().lower().replace("-", "_")
    if method not in PYMOO_REF_DIR_METHODS:
        allowed = ", ".join(PYMOO_REF_DIR_METHODS)
        raise ValueError(f"{path} must be one of {{{allowed}}}, got {value!r}")
    return method


def normalize_pymoo_probability(value, path: str, *, allow_auto: bool = False) -> str | float:
    if allow_auto and isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        allowed = "a number in [0, 1]" + (" or 'auto'" if allow_auto else "")
        raise ValueError(f"{path} must be {allowed}, got {value!r}") from exc
    if not 0.0 <= normalized <= 1.0:
        allowed = "a number in [0, 1]" + (" or 'auto'" if allow_auto else "")
        raise ValueError(f"{path} must be {allowed}, got {value!r}")
    return normalized


def normalize_pymoo_positive_float(value, path: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be > 0, got {value!r}") from exc
    if normalized <= 0.0:
        raise ValueError(f"{path} must be > 0, got {value!r}")
    return normalized


def normalize_pymoo_n_partitions(
    value,
    path: str = "config.optimize.pymoo.algorithms.nsga3.ref_dirs.n_partitions",
) -> str | int:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "auto":
            return "auto"
        try:
            value = int(stripped)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be 'auto' or an integer >= 1, got {value!r}") from exc
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be 'auto' or an integer >= 1, got {value!r}") from exc
    if normalized < 1:
        raise ValueError(f"{path} must be 'auto' or an integer >= 1, got {value!r}")
    return normalized


def normalize_pymoo_config(config: dict, raw_optimize: Optional[dict] = None) -> None:
    optimize_cfg = require_config_dict(config, "optimize")
    template_pymoo = get_template_config()["optimize"]["pymoo"]

    source_optimize = raw_optimize if isinstance(raw_optimize, dict) else optimize_cfg
    raw_pymoo = source_optimize.get("pymoo", {})
    pymoo_cfg = deepcopy(raw_pymoo) if isinstance(raw_pymoo, dict) else {}
    shared = deepcopy(pymoo_cfg.get("shared", {})) if isinstance(pymoo_cfg.get("shared"), dict) else {}
    algorithms = (
        deepcopy(pymoo_cfg.get("algorithms", {}))
        if isinstance(pymoo_cfg.get("algorithms"), dict)
        else {}
    )
    nsga3_cfg = (
        deepcopy(algorithms.get("nsga3", {})) if isinstance(algorithms.get("nsga3"), dict) else {}
    )
    ref_dirs_cfg = (
        deepcopy(nsga3_cfg.get("ref_dirs", {}))
        if isinstance(nsga3_cfg.get("ref_dirs"), dict)
        else {}
    )

    legacy_crossover_eta = source_optimize.get("crossover_eta")
    legacy_crossover_probability = source_optimize.get("crossover_probability")
    legacy_mutation_eta = source_optimize.get("mutation_eta")
    legacy_mutation_indpb = source_optimize.get("mutation_indpb")

    shared_defaults = template_pymoo["shared"]
    ref_dir_defaults = template_pymoo["algorithms"]["nsga3"]["ref_dirs"]

    mutation_prob_var = shared.get("mutation_prob_var")
    if mutation_prob_var is None:
        if legacy_mutation_indpb is not None and float(legacy_mutation_indpb) > 0.0:
            mutation_prob_var = legacy_mutation_indpb
        else:
            mutation_prob_var = shared_defaults["mutation_prob_var"]

    normalized_shared = {
        "crossover_eta": normalize_pymoo_positive_float(
            shared.get(
                "crossover_eta",
                legacy_crossover_eta
                if legacy_crossover_eta is not None
                else shared_defaults["crossover_eta"],
            ),
            "config.optimize.pymoo.shared.crossover_eta",
        ),
        "crossover_prob_var": normalize_pymoo_probability(
            shared.get(
                "crossover_prob_var",
                legacy_crossover_probability
                if legacy_crossover_probability is not None
                else shared_defaults["crossover_prob_var"],
            ),
            "config.optimize.pymoo.shared.crossover_prob_var",
        ),
        "mutation_eta": normalize_pymoo_positive_float(
            shared.get(
                "mutation_eta",
                legacy_mutation_eta if legacy_mutation_eta is not None else shared_defaults["mutation_eta"],
            ),
            "config.optimize.pymoo.shared.mutation_eta",
        ),
        "mutation_prob_var": normalize_pymoo_probability(
            mutation_prob_var,
            "config.optimize.pymoo.shared.mutation_prob_var",
            allow_auto=True,
        ),
        "eliminate_duplicates": bool(
            shared.get("eliminate_duplicates", shared_defaults["eliminate_duplicates"])
        ),
    }

    normalized_ref_dirs = {
        "method": normalize_pymoo_ref_dir_method(
            ref_dirs_cfg.get("method", ref_dir_defaults["method"])
        ),
        "n_partitions": normalize_pymoo_n_partitions(
            ref_dirs_cfg.get("n_partitions", ref_dir_defaults["n_partitions"])
        ),
    }

    optimize_cfg["pymoo"] = {
        "algorithm": normalize_pymoo_algorithm(
            pymoo_cfg.get("algorithm", template_pymoo["algorithm"])
        ),
        "shared": normalized_shared,
        "algorithms": {
            "nsga2": {},
            "nsga3": {"ref_dirs": normalized_ref_dirs},
        },
    }


def normalize_validation_fields(config: dict, *, raw_optimize=None) -> None:
    require_config_dict(config, "monitor")
    config["live"]["hsl_signal_mode"] = normalize_hsl_signal_mode(config["live"]["hsl_signal_mode"])
    config["live"]["hsl_position_during_cooldown_policy"] = (
        normalize_hsl_cooldown_position_policy(config["live"]["hsl_position_during_cooldown_policy"])
    )
    config["live"]["pnls_max_lookback_days"] = normalize_pnls_max_lookback_days_config_value(
        config["live"]["pnls_max_lookback_days"]
    )
    normalize_logging_config(config)
    normalize_monitor_config(config)
    normalize_pymoo_config(config, raw_optimize=raw_optimize)
