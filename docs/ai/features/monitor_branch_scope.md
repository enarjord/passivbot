# Monitor Branch Scope

Read this when working on `feature/monitor-relay-dashboard` or when transplanting monitor work between branches.

## Purpose

This branch is the clean monitor foundation extracted from a larger dev branch.

It exists so monitor work can move independently of unrelated HSL/risk/config churn.

## Source Of Truth

1. The monitor root on disk is the source of truth for observability.
2. The bot writes:
   - one atomic `state.latest.json` snapshot
   - append-only `events/current.ndjson`
   - append-only `history/*.current.ndjson`
   - rotated/compressed segments
   - periodic snapshot checkpoints
3. The relay is a read-only network adapter over that disk state. It must not become a second state engine.

## Scope In

Keep these in scope on the monitor branch:

1. monitor publisher config and file layout
2. snapshot/event/history schema
3. relay HTTP/WebSocket behavior
4. browser dashboard served by the relay
5. TUI and monitor dev launcher
6. generic monitor payloads for:
   - positions
   - orders
   - market
   - forager
   - unstuck
   - trailing
   - recent activity
7. minimal `Passivbot` integration needed to emit monitor data

## Scope Out

Keep these out of scope unless explicitly requested:

1. standalone trailing diagnostics UI/tooling
2. HSL-specific monitor additions that only make sense on the HSL branch
3. unrelated HSL runtime/policy work
4. unrelated optimizer/backtest/config refactors

## Trailing Rule

Trailing data is part of core observability and belongs in the monitor payloads.

That means this branch should include:

1. trailing fields in snapshots
2. trailing-related event/history visibility when applicable
3. small pure helpers needed to derive trailing monitor payloads

It should not include the separate diagnostics product around that data.

Current example:

1. `src/trailing_diagnostics.py` is present as a pure helper module because `src/passivbot_monitor.py` uses it to build trailing monitor payloads.
2. The standalone diagnostics CLI/tool remains intentionally out of scope.

## Passivbot Integration Boundary

1. Keep monitor-specific logic in `src/passivbot_monitor.py` where possible.
2. Keep `src/passivbot.py` changes narrow:
   - publisher init
   - snapshot flush hooks
   - event emission hooks
   - helper binding
3. Do not spread ad hoc monitor writes through unrelated live-bot codepaths.

## HSL Boundary

The extracted monitor branch is intentionally generic.

Current rule:

1. generic monitor infrastructure belongs here
2. HSL-specific monitor fields/events can be added later when merging monitor work back into `feature/equity-hard-stop-loss_codex`

Do not preemptively drag HSL runtime complexity into this branch.

## When Merging Back Into HSL Branch

Expect the later merge to add only a small amount of HSL-specific monitor surface, for example:

1. HSL snapshot fields
2. HSL event kinds
3. any HSL-specific dashboard panels

The relay/disk/dashboard core should not need to be redesigned for that.

## Test Strategy

1. Exhaustive monitor contract tests belong in focused monitor unit/integration tests.
2. The monitor branch should stay green on the monitor-focused test slice.
3. If adding monitor payloads for existing bot behavior, prefer tests that assert:
   - snapshot shape
   - event emission
   - relay forwarding
   - consumer rendering

## Key Files

- `src/monitor_publisher.py`
- `src/passivbot_monitor.py`
- `src/monitor_relay.py`
- `src/monitor_tui.py`
- `src/monitor_dev.py`
- `src/monitor_dashboard_static/`
- `tests/test_monitor_publisher.py`
- `tests/test_passivbot_monitor.py`
- `tests/test_monitor_relay.py`
- `tests/test_monitor_tui.py`
