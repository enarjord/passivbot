"""Stateless controller reference over reconstructed, scope-level episode traces.

Input episode boundaries are exchange-derived evidence supplied by reconstruction,
not old controller state. Numerical estimates remain usable without such a boundary.
The production scope builder must independently establish this trace from snapshots.
"""

from dataclasses import dataclass

from hsl_reference import signal


@dataclass(frozen=True)
class Point:
    observation: object
    exposed: bool
    flatten: bool = False  # supported scope flatten after the risk sample


@dataclass(frozen=True)
class Episode:
    points: tuple
    entry_reference: object = None


@dataclass(frozen=True)
class Decision:
    timestamp: int
    action: str
    red_at: int | None
    flat_at: int | None
    reason: str
    raw: object
    ema: object
    numeric_range_approximation: bool = False


def replay(episodes, *, now, start, budget, span, threshold, cooldown,
           restart="always", intervention="panic"):
    """Replay the entire bounded trace afresh; no prior decision argument.

    Completed episodes include their final risk sample before the supported flat.
    The next episode's baseline must preserve opening fees and subsequent losses.
    The scope builder owns that sample composition, not the controller.
    Historical RED is retained only while it is reproducible in these inputs.
    """
    if start > now or cooldown < 0 or restart not in ("always", "never") or intervention not in ("panic", "normal"):
        raise ValueError("invalid controller configuration")
    previous_time = None
    previous_flat = False
    for episode in episodes:
        if not episode.points:
            raise ValueError("empty episode")
        if previous_time is not None and not previous_flat:
            raise ValueError("episode reset without supported flatten")
        for i, point in enumerate(episode.points):
            t = point.observation.timestamp
            if t > now:
                raise ValueError("future trace observation")
            if previous_time is not None and t < previous_time:
                raise ValueError("unordered trace")
            if point.flatten and (point.exposed or i != len(episode.points) - 1):
                raise ValueError("invalid flatten point")
            previous_time = t
        previous_flat = episode.points[-1].flatten

    decisions = []
    red_at = flat_at = None
    for episode in episodes:
        points = [p for p in episode.points if start <= p.observation.timestamp <= now]
        if not points:
            continue
        if episode.entry_reference is not None and len(points) > 1:
            raise ValueError("entry reference requires a singleton episode")
        # Never seed an active interval with an expired synthetic reference.
        reference = episode.entry_reference if points[0] == episode.points[0] else None
        risk = signal([p.observation for p in points], budget, span, threshold,
                      entry_reference=reference, anchor=episodes[-1].points[-1].observation)
        for point, raw, ema, red_now in zip(points, risk.raw, risk.ema, risk.panic):
            t = point.observation.timestamp
            reason = "green"
            if flat_at is not None and restart == "always" and t >= flat_at + cooldown:
                red_at = flat_at = None
                reason = "cooldown_complete"
            if flat_at is not None and point.exposed:
                if intervention == "normal":
                    red_at = flat_at = None
                    reason = "normal_intervention"
                else:
                    red_at, flat_at = t, None
                    reason = "panic_intervention"
            # A final flatten sample can discover RED before resetting the
            # episode. A flat sample without boundary evidence cannot invent a stop.
            if red_now and (point.exposed or point.flatten) and red_at is None:
                red_at, flat_at = t, None
                reason = "drawdown"
            if point.flatten and red_at is not None:
                flat_at = t
                reason = "stop_flattened"
            if flat_at is not None and restart == "always" and t >= flat_at + cooldown:
                red_at = flat_at = None
                reason = "cooldown_complete"
            action = "normal" if red_at is None else ("panic" if point.exposed else "halted")
            decisions.append(Decision(t, action, red_at, flat_at, reason, raw, ema))
    if not decisions:
        raise ValueError("no in-window current trace")
    if decisions[-1].timestamp != now:
        raise ValueError("trace must end at the current observation")
    return tuple(decisions)
