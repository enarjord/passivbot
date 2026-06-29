from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from config.shared_bot import BOT_GROUP_FIELD_MAP, get_grouped_bot_value
from live.smoke_report import _user_safe_display_path


DEFAULT_SAMPLE_SIZE = 8
SIDES = ("long", "short")
_MISSING = object()
CACHE_LIVE_KEYS = (
    "defer_broad_candle_warmup",
    "enable_archive_candle_fetch",
    "fills_confirmation_overlap_minutes",
    "fills_recent_overlap_minutes",
    "force_cold_startup",
    "max_active_candle_tail_gap_minutes",
    "max_disk_candles_per_symbol_per_tf",
    "max_memory_candles_per_symbol",
    "max_ohlcv_fetches_per_minute",
    "max_warmup_minutes",
    "pnls_max_lookback_days",
    "warmup_concurrency",
    "warmup_jitter_seconds",
    "warmup_ratio",
)
CACHE_SETTING_CHECK_KEYS = (
    "defer_broad_candle_warmup",
    "enable_archive_candle_fetch",
    "fills_confirmation_overlap_minutes",
    "fills_recent_overlap_minutes",
    "force_cold_startup",
    "max_active_candle_tail_gap_minutes",
    "max_disk_candles_per_symbol_per_tf",
    "max_memory_candles_per_symbol",
    "max_warmup_minutes",
    "pnls_max_lookback_days",
    "warmup_ratio",
)
FORAGER_LIVE_KEYS = (
    "forager_score_hysteresis_pct",
    "max_forager_candle_refresh_seconds",
    "max_forager_candle_staleness_minutes",
)
DIFF_SETTING_PATHS = (
    ("identity", "live.user", ("live", "user")),
    ("identity", "live.exchange", ("live", "exchange")),
    ("identity", "backtest.exchanges", ("backtest", "exchanges")),
    ("hsl", "live.hsl_signal_mode", ("live", "hsl_signal_mode")),
    (
        "hsl",
        "live.hsl_position_during_cooldown_policy",
        ("live", "hsl_position_during_cooldown_policy"),
    ),
    ("hsl", "bot.long.hsl.enabled", ("bot", "long", "hsl", "enabled")),
    ("hsl", "bot.short.hsl.enabled", ("bot", "short", "hsl", "enabled")),
    ("forager", "bot.long.risk.n_positions", ("bot", "long", "risk", "n_positions")),
    ("forager", "bot.short.risk.n_positions", ("bot", "short", "risk", "n_positions")),
    *(
        ("forager", f"live.{key}", ("live", key))
        for key in FORAGER_LIVE_KEYS
    ),
    *(
        ("cache", f"live.{key}", ("live", key))
        for key in CACHE_LIVE_KEYS
    ),
)


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    issue = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if path is not None:
        issue["path"] = path
    return issue


def _section(value: Any, name: str, issues: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    issues.append(
        _issue(
            "error",
            "required_section_invalid",
            f"required config section {name!r} is missing or is not an object",
            path=name,
        )
    )
    return {}


def _selected_values(section: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: section[key] for key in keys if key in section}


def _string_sample(values: list[Any], *, limit: int) -> list[str]:
    sample: list[str] = []
    for value in values[:limit]:
        text = str(value)
        sample.append(text[:120] + "...<truncated>" if len(text) > 120 else text)
    return sample


def _display_value(value: Any) -> dict[str, Any]:
    if value is _MISSING:
        return {"present": False}
    return {"present": True, "value": value}


def _setting_value(section: dict[str, Any], key: str) -> Any:
    return section[key] if key in section else _MISSING


def _setting_display(section: dict[str, Any], key: str) -> dict[str, Any]:
    return _display_value(_setting_value(section, key))


def _path_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    if len(path) == 4 and path[0] == "bot" and path[1] in SIDES:
        group_name = path[2]
        local_key = path[3]
        flat_key = BOT_GROUP_FIELD_MAP.get(group_name, {}).get(local_key)
        bot = config.get("bot")
        side_config = bot.get(path[1]) if isinstance(bot, dict) else None
        if flat_key is not None:
            return get_grouped_bot_value(side_config, flat_key, default=_MISSING)
    value: Any = config
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    display_path = _user_safe_display_path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = getattr(exc, "strerror", None) or type(exc).__name__
        return (
            None,
            display_path,
            [
                _issue(
                    "error",
                    f"{label}_read_failed",
                    f"could not read {label} config {display_path}: {detail}",
                    path=display_path,
                )
            ],
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            None,
            display_path,
            [
                _issue(
                    "error",
                    f"{label}_json_decode_failed",
                    f"invalid {label} JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
                    path=display_path,
                )
            ],
        )
    if not isinstance(parsed, dict):
        return (
            None,
            display_path,
            [
                _issue(
                    "error",
                    f"{label}_config_root_invalid",
                    f"{label} config root must be a JSON object",
                    path=display_path,
                )
            ],
        )
    return parsed, display_path, []


def _coin_list_summary(value: Any, *, sample_size: int) -> dict[str, Any]:
    if value is None:
        return {
            "present": False,
            "mode": "missing",
            "count": None,
            "sample": [],
            "truncated": 0,
        }
    if isinstance(value, str):
        if value.lower() == "all":
            return {
                "present": True,
                "mode": "all",
                "count": None,
                "sample": [],
                "truncated": 0,
            }
        return {
            "present": True,
            "mode": "scalar",
            "count": 1,
            "sample": _string_sample([value], limit=sample_size),
            "truncated": 0,
        }
    if isinstance(value, list):
        return {
            "present": True,
            "mode": "list",
            "count": len(value),
            "sample": _string_sample(value, limit=sample_size),
            "truncated": max(0, len(value) - sample_size),
        }
    return {
        "present": True,
        "mode": "invalid",
        "count": None,
        "sample": [],
        "truncated": 0,
    }


def _coin_list_diff_value(value: Any) -> tuple[dict[str, Any], set[str] | None]:
    summary = _coin_list_summary(value, sample_size=0)
    if isinstance(value, list):
        return summary, {str(item) for item in value}
    if isinstance(value, str) and value.lower() != "all":
        return summary, {value}
    return summary, None


def _side_coin_summary(value: Any, *, sample_size: int) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            side: _coin_list_summary(
                value[side] if side in value else None,
                sample_size=sample_size,
            )
            for side in SIDES
        }
    summary = _coin_list_summary(value, sample_size=sample_size)
    return {side: dict(summary) for side in SIDES}


def _hsl_side_report(side_config: dict[str, Any]) -> dict[str, Any]:
    hsl_values = {
        key: get_grouped_bot_value(side_config, flat_key, default=None)
        for key, flat_key in (
            ("enabled", "hsl_enabled"),
            ("red_threshold", "hsl_red_threshold"),
            ("cooldown_minutes_after_red", "hsl_cooldown_minutes_after_red"),
            ("no_restart_drawdown_threshold", "hsl_no_restart_drawdown_threshold"),
            ("ema_span_minutes", "hsl_ema_span_minutes"),
            ("tier_ratios", "hsl_tier_ratios"),
            ("orange_tier_mode", "hsl_orange_tier_mode"),
            ("panic_close_order_type", "hsl_panic_close_order_type"),
        )
    }
    tier_ratios = (
        hsl_values["tier_ratios"] if isinstance(hsl_values.get("tier_ratios"), dict) else {}
    )
    present = any(value is not None for value in hsl_values.values())
    return {
        "present": present,
        "enabled": hsl_values["enabled"],
        "red_threshold": hsl_values["red_threshold"],
        "cooldown_minutes_after_red": hsl_values["cooldown_minutes_after_red"],
        "no_restart_drawdown_threshold": hsl_values["no_restart_drawdown_threshold"],
        "ema_span_minutes": hsl_values["ema_span_minutes"],
        "tier_ratios": {
            key: tier_ratios[key]
            for key in ("yellow", "orange")
            if key in tier_ratios
        },
        "orange_tier_mode": hsl_values["orange_tier_mode"],
        "panic_close_order_type": hsl_values["panic_close_order_type"],
    }


def _forager_side_report(side_config: dict[str, Any]) -> dict[str, Any]:
    forager_values = {
        key: get_grouped_bot_value(side_config, flat_key, default=None)
        for key, flat_key in (
            ("volatility_ema_span_1m", "forager_volatility_ema_span_1m"),
            ("volume_drop_pct", "forager_volume_drop_pct"),
            ("volume_ema_span_1m", "forager_volume_ema_span_1m"),
        )
    }
    report = {
        "n_positions": get_grouped_bot_value(
            side_config, "n_positions", default=None
        ),
        "forager_present": any(value is not None for value in forager_values.values()),
    }
    if report["forager_present"]:
        report["settings"] = {
            key: value
            for key, value in forager_values.items()
            if value is not None
        }
    return report


def _identity_report(config: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    user = live.get("user")
    exchange = live.get("exchange")
    user_exchange_hint = None
    if isinstance(user, str) and "_" in user:
        user_exchange_hint = user.split("_", 1)[0]
    backtest = config["backtest"] if isinstance(config.get("backtest"), dict) else {}
    exchanges = (
        backtest["exchanges"]
        if isinstance(backtest.get("exchanges"), list)
        else None
    )
    return {
        "account": user,
        "user": user,
        "exchange": exchange,
        "user_exchange_hint": user_exchange_hint,
        "backtest_exchanges": exchanges,
        "exchange_source": "live.exchange" if exchange is not None else "not_in_config",
    }


def _numeric_status(value: Any, *, allow_zero: bool = False, allow_all: bool = False) -> str:
    if value is _MISSING:
        return "missing"
    if allow_all and isinstance(value, str) and value.lower() == "all":
        return "all"
    if isinstance(value, bool):
        return "invalid_bool"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if number < 0:
        return "negative"
    if number == 0 and not allow_zero:
        return "zero"
    return "present"


def _bool_status(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if isinstance(value, bool):
        return "present"
    return "invalid"


def _surface_report(name: str) -> dict[str, Any]:
    return {
        "surface": name,
        "status": "settings_compatible",
        "evidence": [],
        "attention": [],
        "not_proven": [],
    }


def _append_attention(surface: dict[str, Any], code: str, message: str) -> None:
    surface["attention"].append({"code": code, "message": message})


def _append_not_proven(surface: dict[str, Any], code: str, message: str) -> None:
    surface["not_proven"].append({"code": code, "message": message})


def _finalize_surface(surface: dict[str, Any]) -> dict[str, Any]:
    if surface["attention"]:
        surface["status"] = "attention"
    elif surface["not_proven"]:
        surface["status"] = "settings_compatible_artifacts_not_checked"
    return surface


def _any_hsl_enabled(hsl_sides: dict[str, dict[str, Any]]) -> bool:
    return any(side_report.get("enabled") is True for side_report in hsl_sides.values())


def _balance_override_report(
    live: dict[str, Any],
    *,
    override_value: Any = _MISSING,
) -> dict[str, Any]:
    source = "argument" if override_value is not _MISSING else "live.balance_override"
    raw_value = override_value if override_value is not _MISSING else live.get("balance_override", _MISSING)
    if raw_value is _MISSING:
        return {"active": False, "source": "none", "present": False}
    if raw_value in (None, ""):
        return {
            "active": False,
            "source": source,
            "present": True,
            "value": raw_value,
        }
    if isinstance(raw_value, bool):
        return {
            "active": True,
            "source": source,
            "present": True,
            "value_type": type(raw_value).__name__,
            "status": "invalid_bool",
        }
    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        return {
            "active": True,
            "source": source,
            "present": True,
            "value_type": type(raw_value).__name__,
            "status": "invalid",
        }
    if not math.isfinite(number) or number <= 0.0:
        return {
            "active": True,
            "source": source,
            "present": True,
            "value": number if math.isfinite(number) else None,
            "value_type": type(raw_value).__name__,
            "status": "invalid",
        }
    return {
        "active": True,
        "source": source,
        "present": True,
        "value": number,
        "status": "valid",
    }


def _effective_hsl_signal_mode(live: dict[str, Any]) -> str:
    raw = live.get("hsl_signal_mode", "unified")
    return str(raw)


def _bot_side_config(config: dict[str, Any], side: str) -> dict[str, Any]:
    bot = config.get("bot")
    if not isinstance(bot, dict):
        return {}
    side_config = bot.get(side)
    return side_config if isinstance(side_config, dict) else {}


def _cache_root_hints(identity: dict[str, Any]) -> dict[str, Any]:
    exchange = identity.get("exchange") or identity.get("user_exchange_hint")
    user = identity.get("user")
    fill_root: dict[str, Any] = {"available": False}
    if isinstance(exchange, str) and exchange and isinstance(user, str) and user:
        fill_root = {
            "available": True,
            "path": f"caches/fill_events/{exchange}/{user}",
        }
    else:
        fill_root["reason"] = "live.user or exchange hint unavailable"
    return {
        "default_cache_root": "caches",
        "candle_v2_root": "caches/ohlcvs",
        "fill_events_root": fill_root,
        "cache_integrity_doctor_root_argument": "caches",
    }


def _cache_readiness_report(
    live: dict[str, Any],
    *,
    identity: dict[str, Any],
    hsl_sides: dict[str, dict[str, Any]],
    balance_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {
        key: _setting_display(live, key)
        for key in CACHE_SETTING_CHECK_KEYS
    }
    candles = _surface_report("candles")
    fills = _surface_report("fills")
    hsl = _surface_report("hsl")

    if live.get("force_cold_startup") is True:
        _append_attention(
            candles,
            "force_cold_startup_enabled",
            "live.force_cold_startup=true disables warm-cache startup reuse",
        )
    force_status = _bool_status(_setting_value(live, "force_cold_startup"))
    if force_status == "missing":
        _append_attention(
            candles,
            "force_cold_startup_missing",
            "live.force_cold_startup is missing; startup cache intent is implicit",
        )
    elif force_status == "invalid":
        _append_attention(
            candles,
            "force_cold_startup_invalid",
            "live.force_cold_startup should be boolean when present",
        )

    defer_status = _bool_status(_setting_value(live, "defer_broad_candle_warmup"))
    if defer_status == "missing":
        _append_attention(
            candles,
            "defer_broad_candle_warmup_missing",
            "live.defer_broad_candle_warmup is missing; broad warmup behavior is implicit",
        )
    elif defer_status == "invalid":
        _append_attention(
            candles,
            "defer_broad_candle_warmup_invalid",
            "live.defer_broad_candle_warmup should be boolean when present",
        )
    elif live.get("defer_broad_candle_warmup") is False:
        _append_attention(
            candles,
            "broad_candle_warmup_blocking",
            "live.defer_broad_candle_warmup=false keeps broad candle warmup on the startup path",
        )

    for key in (
        "max_warmup_minutes",
        "max_active_candle_tail_gap_minutes",
        "max_disk_candles_per_symbol_per_tf",
        "max_memory_candles_per_symbol",
    ):
        status = _numeric_status(_setting_value(live, key), allow_zero=key == "max_warmup_minutes")
        if status == "missing":
            _append_attention(
                candles,
                f"{key}_missing",
                f"live.{key} is missing; candle cache sizing/readiness intent is implicit",
            )
        elif status != "present":
            _append_attention(
                candles,
                f"{key}_{status}",
                f"live.{key} is {status}; candle cache readiness may be degraded",
            )

    warmup_ratio_status = _numeric_status(_setting_value(live, "warmup_ratio"), allow_zero=True)
    if warmup_ratio_status == "missing":
        _append_attention(
            candles,
            "warmup_ratio_missing",
            "live.warmup_ratio is missing; EMA warmup buffer intent is implicit",
        )
    elif warmup_ratio_status != "present":
        _append_attention(
            candles,
            f"warmup_ratio_{warmup_ratio_status}",
            f"live.warmup_ratio is {warmup_ratio_status}; EMA warmup buffer is suspicious",
        )

    archive_status = _bool_status(_setting_value(live, "enable_archive_candle_fetch"))
    if archive_status == "missing":
        candles["evidence"].append(
            {
                "code": "archive_candle_fetch_default",
                "message": (
                    "live.enable_archive_candle_fetch is missing; archive fetch "
                    "remains defaulted by runtime"
                ),
            }
        )
    elif archive_status == "invalid":
        _append_attention(
            candles,
            "enable_archive_candle_fetch_invalid",
            "live.enable_archive_candle_fetch should be boolean when present",
        )
    else:
        candles["evidence"].append(
            {
                "code": "archive_candle_fetch_configured",
                "value": bool(live.get("enable_archive_candle_fetch")),
            }
        )
    _append_not_proven(
        candles,
        "local_candle_artifacts_not_scanned",
        "preflight inspects config only; run cache-integrity-doctor to prove local candle coverage",
    )

    lookback_status = _numeric_status(
        _setting_value(live, "pnls_max_lookback_days"),
        allow_all=True,
    )
    if lookback_status in {"missing", "invalid", "invalid_bool", "negative", "zero"}:
        _append_attention(
            fills,
            f"pnls_max_lookback_days_{lookback_status}",
            "live.pnls_max_lookback_days must be positive or 'all' for fill/PnL lookback consumers",
        )
    else:
        fills["evidence"].append(
            {
                "code": "pnls_max_lookback_days_configured",
                "value": live.get("pnls_max_lookback_days"),
            }
        )
    for key in ("fills_recent_overlap_minutes", "fills_confirmation_overlap_minutes"):
        status = _numeric_status(_setting_value(live, key), allow_zero=True)
        if status == "missing":
            _append_attention(
                fills,
                f"{key}_missing",
                f"live.{key} is missing; fill-cache overlap intent is implicit",
            )
        elif status != "present":
            _append_attention(
                fills,
                f"{key}_{status}",
                f"live.{key} is {status}; fill-cache overlap setting is suspicious",
            )
    _append_not_proven(
        fills,
        "local_fill_artifacts_not_scanned",
        "preflight inspects config only; it does not prove fill-cache coverage or contract metadata",
    )

    if _any_hsl_enabled(hsl_sides):
        hsl["evidence"].append(
            {"code": "hsl_enabled", "message": "one or more HSL sides are enabled"}
        )
        signal_mode = _effective_hsl_signal_mode(live)
        if (
            isinstance(balance_override, dict)
            and balance_override.get("active") is True
            and signal_mode in {"unified", "pside"}
        ):
            _append_attention(
                hsl,
                "hsl_balance_override_account_level_replay_unsafe",
                (
                    "HSL account-level history replay is unsafe with an active "
                    "balance override; runtime will fail before replay for "
                    f"live.hsl_signal_mode={signal_mode!r}"
                ),
            )
        if lookback_status in {"missing", "invalid", "invalid_bool", "negative", "zero"}:
            _append_attention(
                hsl,
                "hsl_fill_lookback_unready",
                "HSL is enabled but live.pnls_max_lookback_days is not a usable positive/'all' value",
            )
        _append_not_proven(
            hsl,
            "local_hsl_artifacts_not_scanned",
            (
                "preflight cannot prove local HSL status/cooldown artifacts; "
                "use hsl-startup-preview for monitor-derived HSL observations"
            ),
        )
    else:
        hsl["status"] = "disabled"
        hsl["evidence"].append(
            {"code": "hsl_disabled", "message": "no HSL side is enabled in config"}
        )

    surfaces = {
        "candles": _finalize_surface(candles),
        "fills": _finalize_surface(fills),
        "hsl": _finalize_surface(hsl),
    }
    attention_count = sum(len(surface["attention"]) for surface in surfaces.values())
    not_proven_count = sum(len(surface["not_proven"]) for surface in surfaces.values())
    status = (
        "attention"
        if attention_count
        else "settings_compatible_artifacts_not_checked"
        if not_proven_count
        else "settings_compatible"
    )
    return {
        "status": status,
        "checks": checks,
        "root_hints": _cache_root_hints(identity),
        "surfaces": surfaces,
        "summary": {
            "attention_count": attention_count,
            "not_proven_count": not_proven_count,
            "disabled_surface_count": sum(
                1 for surface in surfaces.values() if surface["status"] == "disabled"
            ),
        },
        "notes": [
            "config_only_cache_readiness",
            "does_not_scan_cache_artifacts",
            "does_not_enforce_startup_policy",
        ],
    }


def _coin_value_for_side(config: dict[str, Any], group_name: str, side: str) -> Any:
    live = config["live"] if isinstance(config.get("live"), dict) else {}
    group = (
        live[group_name]
        if isinstance(live.get(group_name), dict)
        else live.get(group_name)
    )
    if isinstance(group, dict):
        return group[side] if side in group else None
    return group


def _universe_diff_changes(
    baseline: dict[str, Any],
    target: dict[str, Any],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for group_name in ("approved_coins", "ignored_coins"):
        for side in SIDES:
            before_summary, before_set = _coin_list_diff_value(
                _coin_value_for_side(baseline, group_name, side)
            )
            after_summary, after_set = _coin_list_diff_value(
                _coin_value_for_side(target, group_name, side)
            )
            if before_summary == after_summary and before_set == after_set:
                continue
            change = {
                "category": "universe",
                "field": f"live.{group_name}.{side}",
                "before": {
                    "present": before_summary["present"],
                    "mode": before_summary["mode"],
                    "count": before_summary["count"],
                },
                "after": {
                    "present": after_summary["present"],
                    "mode": after_summary["mode"],
                    "count": after_summary["count"],
                },
            }
            if before_set is not None and after_set is not None:
                added = sorted(after_set - before_set)
                removed = sorted(before_set - after_set)
                change["added_count"] = len(added)
                change["removed_count"] = len(removed)
                change["added_sample"] = _string_sample(added, limit=sample_size)
                change["removed_sample"] = _string_sample(removed, limit=sample_size)
                change["added_truncated"] = max(0, len(added) - sample_size)
                change["removed_truncated"] = max(0, len(removed) - sample_size)
            changes.append(change)
    return changes


def build_live_config_diff_report(
    baseline_config_path: str | Path,
    target_config_path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    sample_size = max(0, int(sample_size))
    baseline, baseline_display_path, baseline_issues = _load_json_object(
        Path(baseline_config_path).expanduser(),
        label="baseline",
    )
    target, target_display_path, target_issues = _load_json_object(
        Path(target_config_path).expanduser(),
        label="target",
    )
    issues = baseline_issues + target_issues
    if baseline is None or target is None:
        return {
            "ok": False,
            "baseline_config_path": baseline_display_path,
            "target_config_path": target_display_path,
            "issues": issues,
            "summary": {"change_count": 0, "category_counts": {}},
        }

    changes: list[dict[str, Any]] = []
    for category, field, path in DIFF_SETTING_PATHS:
        before = _path_value(baseline, path)
        after = _path_value(target, path)
        if before != after:
            changes.append(
                {
                    "category": category,
                    "field": field,
                    "before": _display_value(before),
                    "after": _display_value(after),
                }
            )
    changes.extend(
        _universe_diff_changes(baseline, target, sample_size=sample_size)
    )
    baseline_live = baseline["live"] if isinstance(baseline.get("live"), dict) else {}
    target_live = target["live"] if isinstance(target.get("live"), dict) else {}
    baseline_hsl_sides = {
        side: _hsl_side_report(_bot_side_config(baseline, side))
        for side in SIDES
    }
    target_hsl_sides = {
        side: _hsl_side_report(_bot_side_config(target, side))
        for side in SIDES
    }
    baseline_identity = _identity_report(baseline, baseline_live)
    target_identity = _identity_report(target, target_live)
    baseline_readiness = _cache_readiness_report(
        baseline_live,
        identity=baseline_identity,
        hsl_sides=baseline_hsl_sides,
        balance_override=_balance_override_report(baseline_live),
    )
    target_readiness = _cache_readiness_report(
        target_live,
        identity=target_identity,
        hsl_sides=target_hsl_sides,
        balance_override=_balance_override_report(target_live),
    )
    if baseline_readiness["status"] != target_readiness["status"]:
        changes.append(
            {
                "category": "cache",
                "field": "cache.readiness.status",
                "before": {"present": True, "value": baseline_readiness["status"]},
                "after": {"present": True, "value": target_readiness["status"]},
            }
        )
    if baseline_readiness["summary"] != target_readiness["summary"]:
        changes.append(
            {
                "category": "cache",
                "field": "cache.readiness.summary",
                "before": {"present": True, "value": baseline_readiness["summary"]},
                "after": {"present": True, "value": target_readiness["summary"]},
            }
        )
    category_counts = Counter(change["category"] for change in changes)
    return {
        "ok": True,
        "baseline_config_path": baseline_display_path,
        "target_config_path": target_display_path,
        "issues": [],
        "changes": changes,
        "summary": {
            "change_count": len(changes),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "notes": [
            "offline_read_only_config_diff",
            "does_not_load_credentials_or_contact_exchanges",
            "does_not_enforce_live_startup_policy",
        ],
    }


def _collect_shape_warnings(
    *,
    approved: dict[str, Any],
    ignored: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    for group_name, group in (("approved_coins", approved), ("ignored_coins", ignored)):
        for side in SIDES:
            if group[side]["mode"] == "invalid":
                issues.append(
                    _issue(
                        "warning",
                        "coin_list_shape_invalid",
                        (
                            f"live.{group_name}.{side} is not a list, 'all', "
                            "scalar, or missing"
                        ),
                        path=f"live.{group_name}.{side}",
                    )
                )


def build_live_config_preflight_report(
    config_path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    compare_config_path: str | Path | None = None,
    balance_override: Any = _MISSING,
) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    display_path = _user_safe_display_path(path)
    issues: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = getattr(exc, "strerror", None) or type(exc).__name__
        return {
            "ok": False,
            "config_path": display_path,
            "issues": [
                _issue(
                    "error",
                    "read_failed",
                    f"could not read config {display_path}: {detail}",
                    path=display_path,
                )
            ],
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "config_path": display_path,
            "issues": [
                _issue(
                    "error",
                    "json_decode_failed",
                    f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
                    path=display_path,
                )
            ],
        }
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "config_path": display_path,
            "issues": [
                _issue("error", "config_root_invalid", "config root must be a JSON object")
            ],
        }

    live = _section(parsed["live"] if "live" in parsed else None, "live", issues)
    bot = _section(parsed["bot"] if "bot" in parsed else None, "bot", issues)
    side_configs = {
        side: bot[side] if isinstance(bot.get(side), dict) else {}
        for side in SIDES
    }
    for side, side_config in side_configs.items():
        if not side_config and bot:
            issues.append(
                _issue(
                    "warning",
                    "side_config_missing",
                    f"bot.{side} is missing or is not an object",
                    path=f"bot.{side}",
                )
            )

    sample_size = max(0, int(sample_size))
    approved = _side_coin_summary(live.get("approved_coins"), sample_size=sample_size)
    ignored = _side_coin_summary(live.get("ignored_coins"), sample_size=sample_size)
    _collect_shape_warnings(approved=approved, ignored=ignored, issues=issues)

    forager_by_side = {
        side: _forager_side_report(side_config)
        for side, side_config in side_configs.items()
    }
    n_positions_values = [
        value
        for value in (forager_by_side[side]["n_positions"] for side in SIDES)
        if isinstance(value, (int, float))
    ]
    severity_counts = Counter(issue["severity"] for issue in issues)
    identity_report = _identity_report(parsed, live)
    hsl_sides = {
        side: _hsl_side_report(side_config)
        for side, side_config in side_configs.items()
    }
    balance_override_report = _balance_override_report(live, override_value=balance_override)
    hsl_signal_mode = _effective_hsl_signal_mode(live)
    if balance_override_report.get("status") in {"invalid", "invalid_bool"}:
        issues.append(
            _issue(
                "error",
                f"balance_override_{balance_override_report['status']}",
                "balance override is present but is not a positive finite number",
                path=(
                    "argument.balance_override"
                    if balance_override_report.get("source") == "argument"
                    else "live.balance_override"
                ),
            )
        )
    if (
        _any_hsl_enabled(hsl_sides)
        and balance_override_report.get("active") is True
        and hsl_signal_mode in {"unified", "pside"}
    ):
        issues.append(
            _issue(
                "error",
                "hsl_balance_override_account_level_replay_unsafe",
                (
                    "HSL signal modes 'unified' and 'pside' reconstruct "
                    "account-level equity history and are unsafe with an active "
                    "balance override; use hsl_signal_mode='coin', remove the "
                    "balance override, disable HSL, or initialize an explicit "
                    "HSL baseline/checkpoint before live trading"
                ),
                path=(
                    "argument.balance_override"
                    if balance_override_report.get("source") == "argument"
                    else "live.balance_override"
                ),
            )
        )
    severity_counts = Counter(issue["severity"] for issue in issues)
    report = {
        "ok": "error" not in severity_counts,
        "config_path": display_path,
        "config_version": parsed.get("config_version"),
        "identity": identity_report,
        "hsl": {
            "signal_mode": live.get("hsl_signal_mode"),
            "effective_signal_mode": hsl_signal_mode,
            "cooldown_position_policy": live.get("hsl_position_during_cooldown_policy"),
            "balance_override": balance_override_report,
            "sides": hsl_sides,
        },
        "universe": {
            "approved_coins": approved,
            "ignored_coins": ignored,
        },
        "forager": {
            "live_settings": _selected_values(live, FORAGER_LIVE_KEYS),
            "sides": forager_by_side,
            "total_configured_n_positions": sum(float(value) for value in n_positions_values),
        },
        "cache": {
            "live_settings": _selected_values(live, CACHE_LIVE_KEYS),
            "readiness": _cache_readiness_report(
                live,
                identity=identity_report,
                hsl_sides=hsl_sides,
                balance_override=balance_override_report,
            ),
        },
        "issues": issues,
        "summary": {
            "error_count": int(severity_counts["error"]),
            "warning_count": int(severity_counts["warning"]),
        },
        "notes": [
            "offline_read_only_report",
            "does_not_load_credentials_or_contact_exchanges",
            "does_not_enforce_live_startup_policy",
        ],
    }
    if compare_config_path is not None:
        diff_report = build_live_config_diff_report(
            compare_config_path,
            path,
            sample_size=sample_size,
        )
        report["diff"] = diff_report
        issues.extend(diff_report["issues"])
        severity_counts = Counter(issue["severity"] for issue in issues)
        report["ok"] = "error" not in severity_counts
        report["summary"] = {
            "error_count": int(severity_counts["error"]),
            "warning_count": int(severity_counts["warning"]),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-config-preflight",
        description="Read-only offline preflight report for risk-relevant live config facts.",
    )
    parser.add_argument("config_path", help="Live config JSON file to inspect.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum approved/ignored coin sample size per side.",
    )
    parser.add_argument(
        "--compare",
        dest="compare_config_path",
        help="Optional baseline live config JSON file for read-only diff reporting.",
    )
    parser.add_argument(
        "--balance-override",
        dest="balance_override",
        default=_MISSING,
        help=(
            "Optional balance override from the intended live launch command. "
            "Use this when preflighting a run that will pass -bo/--balance-override."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_live_config_preflight_report(
        args.config_path,
        sample_size=int(args.sample_size),
        compare_config_path=args.compare_config_path,
        balance_override=args.balance_override,
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
