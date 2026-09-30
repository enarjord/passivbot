# Live Shutdown And Warm-Cache Restart Handoff

## Purpose

Repeated VPS5 live-bot restarts during the v8 logging-overhaul work exposed two
separate operational problems worth handing to a dedicated agent:

1. Ctrl-C shutdown can still wait on work that is no longer useful once the bot
   is stopping.
2. Short-downtime restarts still do too much cold-start warmup/replay work even
   when local cache data should already prove most required coverage.

This handoff is for another agent to fork from `v8` and implement the work in
two separate PRs. Keep both PRs reviewed, tested, and live-smoked separately.

## Evidence From Recent VPS5 Restarts

Observed during repeated restarts of the five VPS5 bots:

These are not trading-logic bugs by themselves. They are operational latency and
developer-feedback-loop problems that also increase VPS load.

## Shared Rules

- Branch each PR from current `v8`.
- Keep Rust order/risk behavior authoritative. Do not change trading decisions
  in Python under the name of shutdown or warmup optimization.
- Fresh account-critical state remains mandatory on every startup:
  positions, balance, and open orders must be fetched or the bot must fail/defer
  according to the existing error contract.
- HSL/stateless safety must not be weakened. Local caches may speed startup only
  when they prove the same state that a cold reconstruction would produce, or
  when the bot can compute a bounded delta from a proven checkpoint.
- Local cache is a performance cache, not an unverified behavior source.
- Add regression tests for every new cancellation/cache-reuse path.
- Add structured events or existing event-bus helpers where useful, but keep the
  behavioral PRs separate from the logging-overhaul PRs.

## PR 1: Faster Ctrl-C Shutdown

### Goal

When Ctrl-C or process stop is requested, the shutdown intent should propagate
into long-running live paths quickly. The bot should stop starting new non-cleanup
work, cancel or abandon work that is no longer useful, close sessions to interrupt
slow I/O, flush monitor/event output on a short bounded deadline, and exit.

### Target Contract

### Likely Code Areas

### Implementation Notes

- Prefer an `asyncio.Event` or equivalent single shutdown primitive in addition
  to existing `stop_signal_received`/`_shutdown_in_progress`, then bridge old
  checks to the new primitive.
- Replace long sleeps with `_sleep_unless_shutdown`.
- Add shutdown checks around loops that process many symbols, many rows, or many
  candle windows.
- For lock waits, sleep in short chunks and bail when shutdown is requested.
- When shutting down, cancel known background tasks before waiting on them.
- Close CCXT sessions early enough to break slow network calls, but not before
  the code has stopped enqueueing normal work.
- Track interrupted component names for structured stop diagnostics.

### Tests

Add focused async tests with fake tasks/exchanges:

### VPS Smoke Acceptance

On VPS5, restart all five configured bots and then Ctrl-C all of them from tmux.
Record:

- time from signal to process exit per bot
- whether any bot exceeds 15s
- whether any bot exceeds 30s
- last shutdown log/event per bot
- monitor/event pipeline close result

Expected target:

- Idle or normal-loop bots exit in a few seconds.
- Bots inside candle/HSL/fill/account work should usually exit under 15s.
- If an exchange/network call cannot be interrupted, it must be visible with the
  component and elapsed time, and shutdown must still have an upper bound.

## PR 2: Faster Warm-Cache Restart

### Goal

### Target Contract

- Always fetch fresh account-critical state on startup.
- Use cache only when metadata proves:
  - source exchange/user/config match
  - symbol/timeframe/pside requirements match the current config
  - coverage reaches the required start and latest completed candle target
  - cache generation/index is valid
  - synthetic/no-trade gaps are explicitly proven by the candle policy
- If proof is missing, stale, non-finite, or incompatible, fall back to current
  cold-refresh behavior with an observable reason.
- For short downtime, fetch only the missing tail/delta ranges when coverage is
  otherwise proven.
- For forager candidates, preserve the existing forager staleness contract from
  `docs/ai/error_contract.md`: stale-but-within-policy candidates are not
  arbitrarily excluded, and volume/log-range ranking features carry forward only
  when their age/provenance is valid.

### Likely Code Areas

### Implementation Notes

### Tests

Add tests around cache proof and startup routing:

- Short downtime with complete candle cache uses delta-only candle fetch.
- Missing latest tail triggers only tail fetch, not full window rebuild.
- Missing/invalid cache metadata falls back to cold path.
- Non-finite EMA/ranking feature data blocks fast-path reuse for the affected
  surface.
- Forager candidates inside allowed staleness retain valid carried
  volume/log-range features; candidates beyond the cap become unavailable with a
  reason.
- Fast path and cold path produce equivalent HSL current drawdown/red status for
  controlled fixtures.

### VPS Smoke Acceptance

On VPS5:

Expected target:

## Suggested Review Prompt

Review the PR against current `v8`. Read `AGENTS.md`,
`docs/ai/principles.md`, and `docs/ai/error_contract.md` first. This work is
operational shutdown/startup behavior, not trading logic. Verify it preserves
Rust order/risk authority, stateless restart safety, and hard-fail behavior for
trading-critical inputs. Look especially for hidden fallback defaults, local
cache changing behavior without proof, shutdown cancellation leaving corrupt
cache files, and any path that can still block Ctrl-C indefinitely.

Findings first, ordered by severity. Include exact file/line references,
repro/test suggestions, and whether the issue affects PR 1 shutdown, PR 2
warm-cache restart, or both.
