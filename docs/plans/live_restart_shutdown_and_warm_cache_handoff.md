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

- Binance and Hyperliquid often stopped quickly.
- Kucoin sometimes took over two minutes after Ctrl-C while it was in candle
  warmup/fetch-lock work.
- Gateio sometimes waited for in-flight fills/account refresh work before exit.
- OKX usually exited promptly, but can still be inside exchange/websocket work.
- Gateio startup with HSL coin history replay reached READY only after HSL
  reconstruction applied `880111` rows in about `114.9s`, despite the bot having
  been restarted after a short downtime.

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

- A single shutdown intent is observable by all long-running loops.
- Long candle warmup/fetch loops, forager refresh, HSL replay, fill refresh,
  staged account refresh, background maintainers, executor waits, and lock waits
  check the shutdown intent at bounded intervals.
- After shutdown starts, do not start new normal planning/execution work.
- In-flight exchange writes cannot be unsent, but shutdown should not enqueue new
  writes after the stop intent is visible.
- In-flight non-critical fetches may be cancelled or interrupted by session close.
- Cache writes must remain atomic or be safely abandoned; do not leave partially
  written cache shards that later look valid.
- Monitor/event pipeline flush has a bounded deadline. Flush failure degrades
  observability only; it must not hang shutdown indefinitely.
- A repeated signal may still force immediate exit, preserving the current
  operator escape hatch.

### Likely Code Areas

- `src/passivbot.py`
  - signal handling around `_handle_shutdown_signal`
  - `_shutdown_requested`, `_raise_if_shutdown_requested`,
    `_sleep_unless_shutdown`
  - `shutdown_gracefully`
  - startup and warmup calls around `start_bot`
  - HSL replay paths
- `src/live/state_refresh.py`
  - staged account refresh and routine fill prefetch
- candle/warmup paths in `Passivbot`
  - `warmup_trading_ready_candles`
  - `start_background_candle_warmup`
  - active/forager candle refresh loops
  - candlestick manager stop callback/lock waits
- websocket/order maintainer tasks

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

- Ctrl-C during candle warmup cancels/returns promptly and closes sessions.
- Ctrl-C during forager refresh does not continue refreshing the remaining
  candidate universe.
- Ctrl-C during HSL replay stops row/symbol processing before replay completion.
- Ctrl-C while a lock wait is active exits the wait promptly.
- Ctrl-C while staged account/fill refresh is sleeping or awaiting a fake fetch
  does not wait for the full fake fetch duration when session/task cancellation
  is possible.
- Shutdown still flushes monitor/events before publisher close when possible.
- Repeated signal still forces immediate exit.
- No new order creates are started after shutdown is requested.

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

A short-downtime restart should not repeat cold-start candle warmup, broad
forager warmup, or full HSL replay when local caches can prove coverage and a
delta refresh is sufficient. Startup should explain which surfaces were reused,
which were refreshed, and why cold-start work was required.

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
- For HSL:
  - `hsl_signal_mode=coin` can often avoid candle-equity replay and should use
    fill/PnL cache proof plus bounded delta where valid.
  - `pside` and `unified` modes that require equity replay may use a persisted
    replay checkpoint only if it is keyed by config, fills coverage, candle
    coverage, exchange/user, and replay code version. Otherwise cold replay.
  - A checkpoint must never change the red/current-drawdown contract. Current
    state still determines whether to panic now.

### Likely Code Areas

- startup flow in `src/passivbot.py`
- candle index/cache coverage functions
- `CandlestickManager` cache metadata and gap policy
- fill/PnL cache coverage and HSL replay code
- startup timing diagnostics
- event bus emitters for cache reuse/degraded startup evidence

### Implementation Notes

- First add measurement if needed: startup events should separate account fetch,
  active candle warmup, forager/background warmup, HSL replay load, HSL replay
  apply, and market-ready timing.
- Add a cache-coverage decision object rather than scattering booleans:
  `accepted`, `reason_code`, `covered_start_ms`, `covered_end_ms`,
  `latest_required_ms`, `delta_ranges`, `source_fingerprint`, and
  `cold_path_required`.
- Cache proof should be strict and boring. The fast path is allowed only after
  the proof object says it is equivalent to cold reconstruction plus delta.
- Persist HSL replay checkpoints only after successful replay completion. Use a
  code/config/input fingerprint and invalidate aggressively.
- Keep the existing cold path intact and easy to force for debugging.

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
- HSL coin mode uses fill/PnL proof and delta without candle-equity replay when
  valid.
- HSL pside/unified checkpoint is accepted only when all fingerprints match.
- HSL checkpoint mismatch, missing fills coverage, or candle coverage gap falls
  back to cold replay.
- Fast path and cold path produce equivalent HSL current drawdown/red status for
  controlled fixtures.

### VPS Smoke Acceptance

On VPS5:

1. Start all configured bots from a cold-ish state and record startup timings.
2. Stop all bots.
3. Restart within a few minutes and record startup timings again.
4. Confirm account-critical fetches still occur on every startup.
5. Confirm logs/events explicitly say which cache surfaces were reused and which
   deltas were fetched.
6. Confirm Gateio HSL startup no longer re-applies the full historical replay
   when a valid checkpoint/delta path exists.
7. Confirm no hard errors, no missing EMA regressions, and no order/risk
   behavior changes.

Expected target:

- Short-downtime restarts should be materially faster than cold starts.
- Gateio HSL replay should avoid multi-minute full replay when cache/checkpoint
  proof is valid.
- If fast-path proof is invalid, startup may remain slow, but the reason should
  be explicit and searchable.

## Suggested Review Prompt

Review the PR against current `v8`. Read `AGENTS.md`,
`docs/ai/principles.yaml`, and `docs/ai/error_contract.md` first. This work is
operational shutdown/startup behavior, not trading logic. Verify it preserves
Rust order/risk authority, stateless restart safety, and hard-fail behavior for
trading-critical inputs. Look especially for hidden fallback defaults, local
cache changing behavior without proof, shutdown cancellation leaving corrupt
cache files, and any path that can still block Ctrl-C indefinitely.

Findings first, ordered by severity. Include exact file/line references,
repro/test suggestions, and whether the issue affects PR 1 shutdown, PR 2
warm-cache restart, or both.
