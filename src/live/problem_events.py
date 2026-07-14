from __future__ import annotations

from typing import Any

from live.event_bus import EventTypes


def is_problem_event(live_event: dict[str, Any]) -> bool:
    level = str(live_event.get("level") or "").lower()
    status = str(live_event.get("status") or "").lower()
    event_type = str(live_event.get("event_type") or "")
    reason_code = str(live_event.get("reason_code") or "")
    if (
        event_type == EventTypes.HSL_REPLAY_FAILED
        and reason_code == "shutdown_cancelled"
    ):
        return False
    return (
        level in {"error", "critical"}
        or status in {"failed", "degraded"}
        or event_type == EventTypes.SINK_DEGRADED
    )


def is_hard_problem_event(live_event: dict[str, Any]) -> bool:
    level = str(live_event.get("level") or "").lower()
    event_type = str(live_event.get("event_type") or "")
    return level in {"error", "critical"} or event_type == EventTypes.SINK_DEGRADED
