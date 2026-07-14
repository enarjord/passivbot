from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config.shared_bot import get_grouped_bot_value
from live.event_bus import EventTypes, LIVE_EVENT_MONITOR_PAYLOAD_KEY
from live.event_query import discover_event_files
from live.smoke_report import _user_safe_display_path


SIDES = ("long", "short")
HSL_EVENT_TYPES = {
    EventTypes.HSL_STATUS,
    EventTypes.HSL_RED_TRIGGERED,
    EventTypes.HSL_COOLDOWN_STARTED,
    EventTypes.HSL_COOLDOWN_ENDED,
}
HSL_DATA_KEYS = (
    "signal_mode",
    "tier",
    "previous_tier",
    "drawdown_score",
    "drawdown_raw",
    "drawdown_ema",
    "dist_to_red",
    "red_threshold",
    "cooldown_until_ms",
    "cooldown_remaining",
    "cooldown_remaining_seconds",
    "last_red_ts",
    "pending_red_since_ms",
    "slot_budget",
    "realized_pnl",
    "peak_realized_pnl",
    "unrealized_pnl",
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


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
    }


def _load_config(config_path: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = Path(config_path).expanduser()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = getattr(exc, "strerror", None) or type(exc).__name__
        return None, [
            _issue(
                "error",
                "config_read_failed",
                f"could not read config: {detail}",
                path=_user_safe_display_path(path),
            )
        ]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [
            _issue(
                "error",
                "config_json_decode_failed",
                f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
                path=_user_safe_display_path(path),
            )
        ]
    if not isinstance(parsed, dict):
        return None, [
            _issue("error", "config_root_invalid", "config root must be a JSON object")
        ]
    return parsed, []


def _section(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _hsl_side_config(config: dict[str, Any], pside: str) -> dict[str, Any]:
    bot = _section(config.get("bot"))
    side_config = _section(bot.get(pside))
    hsl_values = {
        key: get_grouped_bot_value(side_config, flat_key, default=None)
        for key, flat_key in (
            ("enabled", "hsl_enabled"),
            ("red_threshold", "hsl_red_threshold"),
            ("cooldown_minutes_after_red", "hsl_cooldown_minutes_after_red"),
            ("no_restart_drawdown_threshold", "hsl_no_restart_drawdown_threshold"),
            ("restart_after_red_policy", "hsl_restart_after_red_policy"),
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


def _config_report(config: dict[str, Any]) -> dict[str, Any]:
    live = _section(config.get("live"))
    return {
        "config_version": config.get("config_version"),
        "identity": {
            "user": live.get("user"),
            "account": live.get("user"),
            "exchange": live.get("exchange"),
        },
        "hsl": {
            "signal_mode": live.get("hsl_signal_mode"),
            "cooldown_position_policy": live.get("hsl_position_during_cooldown_policy"),
            "sides": {pside: _hsl_side_config(config, pside) for pside in SIDES},
        },
    }


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _event_sort_key(record: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(record.get("latest_ts") or -1),
        int(record.get("latest_seq") or -1),
        str(record.get("latest_path") or ""),
        int(record.get("latest_line") or 0),
    )


def _bot_key(live_event: dict[str, Any], row: dict[str, Any]) -> str:
    exchange = live_event.get("exchange") or row.get("exchange") or "unknown_exchange"
    user = live_event.get("user") or row.get("user") or "unknown_user"
    return f"{exchange}/{user}"


def _bounded_hsl_data(live_event: dict[str, Any]) -> dict[str, Any]:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    out: dict[str, Any] = {}
    for key in HSL_DATA_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int):
            out[key] = value
        elif isinstance(value, float):
            if value == value and value not in (float("inf"), float("-inf")):
                out[key] = value
        elif isinstance(value, str):
            out[key] = value[:160] + "...<truncated>" if len(value) > 160 else value
    return out


def _target_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("bot") or "unknown_exchange/unknown_user"),
        str(record.get("pside") or "unknown_pside"),
        str(record.get("symbol") or ""),
    )


def _status_from_event(record: dict[str, Any]) -> str:
    data = record.get("latest_data") if isinstance(record.get("latest_data"), dict) else {}
    tier = data.get("tier")
    if tier not in (None, ""):
        return str(tier)
    reason_code = record.get("reason_code")
    if reason_code in {"green", "yellow", "orange", "red", "cooldown_active"}:
        return "red" if reason_code == "cooldown_active" else str(reason_code)
    event_type = record.get("event_type")
    if event_type in {EventTypes.HSL_RED_TRIGGERED, EventTypes.HSL_COOLDOWN_STARTED}:
        return "red"
    return "unknown"


def _cooldown_preview(data: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
    cooldown_until_ms = data.get("cooldown_until_ms")
    remaining_seconds = data.get("cooldown_remaining_seconds")
    if cooldown_until_ms is None and remaining_seconds is None and data.get("cooldown_remaining") is None:
        return _unavailable("no cooldown fields were present in the latest local HSL event")
    out: dict[str, Any] = {
        "available": True,
        "source": "latest_local_hsl_event",
    }
    if cooldown_until_ms is not None:
        try:
            until_ms = int(cooldown_until_ms)
            out["cooldown_until_ms"] = until_ms
            out["remaining_seconds_at_preview"] = max(0.0, (until_ms - int(now_ms)) / 1000.0)
            out["status_at_preview"] = "active" if int(now_ms) < until_ms else "elapsed_by_wall_clock"
        except (TypeError, ValueError):
            out["cooldown_until_ms"] = cooldown_until_ms
            out["status_at_preview"] = "unknown_invalid_timestamp"
    if remaining_seconds is not None:
        out["last_observed_remaining_seconds"] = remaining_seconds
    if data.get("cooldown_remaining") is not None:
        out["last_observed_remaining"] = data.get("cooldown_remaining")
    return out


def _status_record_preview(record: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
    data = record.get("latest_data") if isinstance(record.get("latest_data"), dict) else {}
    status = _status_from_event(record)
    drawdown_to_red = (
        {
            "available": True,
            "source": "latest_local_hsl_event",
            "value": data["dist_to_red"],
            "note": "last observed distance to red; not recomputed from current exchange state",
        }
        if data.get("dist_to_red") is not None
        else _unavailable("latest local HSL event did not include dist_to_red")
    )
    current_drawdown = _unavailable(
        "offline preview does not contact exchanges or replay fresh fill/account state"
    )
    last_observed_drawdown = {
        key: data[key]
        for key in ("drawdown_raw", "drawdown_ema", "drawdown_score", "red_threshold")
        if data.get(key) is not None
    }
    return {
        key: value
        for key, value in {
            "bot": record.get("bot"),
            "pside": record.get("pside"),
            "symbol": record.get("symbol"),
            "status": status,
            "event_type": record.get("event_type"),
            "reason_code": record.get("reason_code"),
            "latest_ts": record.get("latest_ts"),
            "last_observed_drawdown": last_observed_drawdown,
            "drawdown_to_red": drawdown_to_red,
            "current_drawdown": current_drawdown,
            "cooldown": _cooldown_preview(data, now_ms=now_ms),
            "latest_data": data,
        }.items()
        if value not in (None, {}, [])
    }


def _scan_hsl_events(
    monitor_root: str | Path,
    *,
    include_rotated: bool,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, Any]:
    try:
        files = discover_event_files(monitor_root, include_rotated=include_rotated)
    except FileNotFoundError:
        return {
            "available": False,
            "root": str(Path(monitor_root).expanduser()),
            "files_scanned": 0,
            "rows_scanned": 0,
            "hsl_events_seen": 0,
            "latest_by_target": [],
            "issues": [
                _issue(
                    "warning",
                    "monitor_root_missing",
                    "monitor root or event segment does not exist",
                    path=str(monitor_root),
                )
            ],
        }

    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    event_type_counts: Counter[str] = Counter()
    rows_scanned = 0
    hsl_events_seen = 0
    for path in files:
        try:
            handle = _open_text(path)
        except OSError as exc:
            issues.append(
                _issue(
                    "warning",
                    "event_file_read_failed",
                    f"could not read event file: {exc}",
                    path=str(path),
                )
            )
            continue
        with handle:
            for line_no, line in enumerate(handle, start=1):
                rows_scanned += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    issues.append(
                        _issue(
                            "warning",
                            "event_json_decode_failed",
                            "skipped malformed monitor event row",
                            path=f"{path}:{line_no}",
                        )
                    )
                    continue
                if not isinstance(row, dict):
                    continue
                ts = row.get("ts")
                try:
                    ts_int = int(ts)
                except (TypeError, ValueError):
                    ts_int = None
                if since_ms is not None and (ts_int is None or ts_int < since_ms):
                    continue
                if until_ms is not None and (ts_int is None or ts_int > until_ms):
                    continue
                live_event = _live_event_payload(row)
                if live_event is None:
                    continue
                event_type = live_event.get("event_type") or row.get("kind")
                if event_type not in HSL_EVENT_TYPES:
                    continue
                hsl_events_seen += 1
                event_type_counts[str(event_type)] += 1
                record = {
                    "bot": _bot_key(live_event, row),
                    "event_type": event_type,
                    "reason_code": live_event.get("reason_code"),
                    "status": live_event.get("status"),
                    "level": live_event.get("level"),
                    "exchange": live_event.get("exchange") or row.get("exchange"),
                    "user": live_event.get("user") or row.get("user"),
                    "symbol": live_event.get("symbol") or row.get("symbol"),
                    "pside": live_event.get("pside") or row.get("pside"),
                    "latest_ts": ts_int,
                    "latest_seq": row.get("seq"),
                    "latest_path": str(path),
                    "latest_line": int(line_no),
                    "latest_data": _bounded_hsl_data(live_event),
                }
                key = _target_key(record)
                existing = latest.get(key)
                if existing is None or _event_sort_key(record) > _event_sort_key(existing):
                    latest[key] = record

    latest_by_target = sorted(
        latest.values(),
        key=lambda item: (
            str(item.get("bot") or ""),
            str(item.get("pside") or ""),
            str(item.get("symbol") or ""),
        ),
    )
    return {
        "available": bool(files),
        "root": str(Path(monitor_root).expanduser()),
        "include_rotated": bool(include_rotated),
        "files_scanned": len(files),
        "rows_scanned": rows_scanned,
        "hsl_events_seen": hsl_events_seen,
        "event_types": dict(event_type_counts.most_common()),
        "latest_by_target": latest_by_target,
        "issues": issues,
    }


def build_hsl_startup_preview_report(
    config_path: str | Path,
    *,
    monitor_root: str | Path | None = "monitor",
    include_rotated: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    config, config_issues = _load_config(config_path)
    if config is None:
        return {
            "ok": False,
            "tool": "hsl-startup-preview",
            "config_path": _user_safe_display_path(config_path),
            "issues": config_issues,
        }

    issues = list(config_issues)
    event_scan: dict[str, Any]
    if monitor_root is None or str(monitor_root).strip() == "":
        event_scan = {
            "available": False,
            "root": None,
            "include_rotated": bool(include_rotated),
            "files_scanned": 0,
            "rows_scanned": 0,
            "hsl_events_seen": 0,
            "event_types": {},
            "latest_by_target": [],
            "issues": [],
        }
    else:
        event_scan = _scan_hsl_events(
            monitor_root,
            include_rotated=include_rotated,
            since_ms=since_ms,
            until_ms=until_ms,
        )
    issues.extend(event_scan.get("issues") or [])

    latest_by_target = event_scan["latest_by_target"]
    statuses = [
        _status_record_preview(record, now_ms=int(now_ms))
        for record in latest_by_target
    ]
    status_counts = Counter(str(item.get("status") or "unknown") for item in statuses)
    return {
        "ok": not any(issue.get("severity") == "error" for issue in issues),
        "tool": "hsl-startup-preview",
        "config_path": _user_safe_display_path(config_path),
        "preview_time_ms": int(now_ms),
        "inputs": {
            "config": {
                "available": True,
                "path": _user_safe_display_path(config_path),
            },
            "monitor_events": {
                key: event_scan.get(key)
                for key in (
                    "available",
                    "root",
                    "include_rotated",
                    "files_scanned",
                    "rows_scanned",
                    "hsl_events_seen",
                    "event_types",
                )
            },
            "account_state": _unavailable(
                "not loaded; this tool is local/offline and does not contact exchanges"
            ),
            "fill_history": _unavailable(
                "not replayed in this first slice; latest local HSL events are reported when present"
            ),
            "current_drawdown": _unavailable(
                "requires fresh balance, positions, fills, and unrealized PnL"
            ),
            "startup_panic_order_prediction": _unavailable(
                "requires live startup replay, current positions/open orders, market snapshots, and Rust planning"
            ),
        },
        "config": _config_report(config),
        "hsl_status": {
            "available": bool(statuses),
            "source": "latest_local_monitor_hsl_events" if statuses else None,
            "latest_by_target": statuses,
            "counts_by_status": dict(status_counts.most_common()),
        },
        "startup_panic_orders": {
            "available": False,
            "would_emit": None,
            "reason": (
                "offline first-slice preview does not predict panic orders without "
                "current exchange/account state and order planning"
            ),
        },
        "issues": issues,
        "summary": {
            "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "hsl_targets_with_local_status": len(statuses),
            "status_counts": dict(status_counts.most_common()),
        },
        "notes": [
            "offline_read_only_report",
            "does_not_load_credentials_or_contact_exchanges",
            "does_not_predict_current_panic_orders",
            "latest local HSL events are observations, not fresh startup replay",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool hsl-startup-preview",
        description=(
            "Read-only offline HSL startup preview from config and local monitor "
            "events."
        ),
    )
    parser.add_argument("config_path", help="Live config JSON file to inspect.")
    parser.add_argument(
        "--monitor-root",
        default="monitor",
        help=(
            "Monitor root, bot root, events directory, or NDJSON segment to scan. "
            "Use an empty string to skip local event scanning."
        ),
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help="Also scan rotated/compressed monitor event segments.",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        help="Only include local HSL events at or after this epoch-ms timestamp.",
    )
    parser.add_argument(
        "--until-ms",
        type=int,
        help="Only include local HSL events at or before this epoch-ms timestamp.",
    )
    parser.add_argument(
        "--now-ms",
        type=int,
        help="Preview wall-clock epoch-ms used for cooldown remaining calculations.",
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
    if args.since_ms is not None and args.until_ms is not None and args.since_ms > args.until_ms:
        parser.error("--since-ms must be <= --until-ms")
    report = build_hsl_startup_preview_report(
        args.config_path,
        monitor_root=args.monitor_root if str(args.monitor_root).strip() else None,
        include_rotated=bool(args.include_rotated),
        since_ms=args.since_ms,
        until_ms=args.until_ms,
        now_ms=args.now_ms,
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
