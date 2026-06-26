from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE_SIZE = 8
SIDES = ("long", "short")
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
FORAGER_LIVE_KEYS = (
    "forager_score_hysteresis_pct",
    "max_forager_candle_refresh_seconds",
    "max_forager_candle_staleness_minutes",
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
    hsl = side_config["hsl"] if isinstance(side_config.get("hsl"), dict) else {}
    tier_ratios = hsl["tier_ratios"] if isinstance(hsl.get("tier_ratios"), dict) else {}
    return {
        "present": bool(hsl),
        "enabled": hsl.get("enabled"),
        "red_threshold": hsl.get("red_threshold"),
        "cooldown_minutes_after_red": hsl.get("cooldown_minutes_after_red"),
        "no_restart_drawdown_threshold": hsl.get("no_restart_drawdown_threshold"),
        "ema_span_minutes": hsl.get("ema_span_minutes"),
        "tier_ratios": {
            key: tier_ratios[key]
            for key in ("yellow", "orange")
            if key in tier_ratios
        },
        "orange_tier_mode": hsl.get("orange_tier_mode"),
        "panic_close_order_type": hsl.get("panic_close_order_type"),
    }


def _forager_side_report(side_config: dict[str, Any]) -> dict[str, Any]:
    risk = side_config["risk"] if isinstance(side_config.get("risk"), dict) else {}
    forager = (
        side_config["forager"]
        if isinstance(side_config.get("forager"), dict)
        else {}
    )
    report = {
        "n_positions": risk.get("n_positions"),
        "forager_present": bool(forager),
    }
    if forager:
        report["settings"] = {
            key: forager[key]
            for key in (
                "volatility_ema_span_1m",
                "volume_drop_pct",
                "volume_ema_span_1m",
            )
            if key in forager
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
) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    issues: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False,
            "config_path": str(path),
            "issues": [
                _issue(
                    "error",
                    "read_failed",
                    f"could not read config: {exc}",
                    path=str(path),
                )
            ],
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "config_path": str(path),
            "issues": [
                _issue(
                    "error",
                    "json_decode_failed",
                    f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
                    path=str(path),
                )
            ],
        }
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "config_path": str(path),
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
    return {
        "ok": "error" not in severity_counts,
        "config_path": str(path),
        "config_version": parsed.get("config_version"),
        "identity": _identity_report(parsed, live),
        "hsl": {
            "signal_mode": live.get("hsl_signal_mode"),
            "cooldown_position_policy": live.get("hsl_position_during_cooldown_policy"),
            "sides": {
                side: _hsl_side_report(side_config)
                for side, side_config in side_configs.items()
            },
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
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
