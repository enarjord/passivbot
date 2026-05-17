# Staged Live Open-Tail EMA Projection Plan

## Objective

Replace the current bounded open-tail EMA carry-forward policy with a bounded, provisional,
in-memory EMA projection for staged live planning.

When the latest completed 1m candle is missing but the gap is a pure open-ended tail within
`live.max_active_candle_tail_gap_minutes`, live should compute temporary EMA inputs as if the
missing completed minutes were no-trade candles. The projection must not persist synthetic candles
to disk or runtime candle caches. Once real candles arrive, normal candle truth and existing EMA
cache invalidation/recompute behavior take over.

## Current State

Relevant current behavior:

1. `src/passivbot.py::_completed_candle_freshness_signature()` permits bounded active open-tail
   gaps by stamping a `tail_gap_fallback` signature.
2. `src/passivbot.py::_log_completed_candle_tail_gap_fallbacks()` logs the policy as
   `action=carry_forward_latest_real`.
3. `src/passivbot.py::_load_orchestrator_ema_bundle()` reads close, quote-volume, and log-range
   EMAs through `CandlestickManager` helpers.
4. `src/passivbot.py::fetch_close_map()` has a previous-close-EMA fallback when the close EMA read
   fails, bounded by `_close_ema_fallback_max_age_ms()`.
5. `src/candlestick_manager.py::get_completed_candle_health()` reports enough open-tail metadata:
   `open_tail_gap`, `missing_spans`, `latest_expected_ts`, `last_cached_ts`, `tail_gap_age_ms`, and
   `tail_gap_candles`.
6. `src/candlestick_manager.py` already keeps synthetic tracking and invalidates EMA caches when
   real data replaces runtime synthetic rows, but open-tail projection must not use that persisted
   synthetic path.

Known gap:

Live currently carries forward latest real EMA state during a bounded open tail. Backtests can see
the future and replay bounded no-trade gaps as zero-candles, so live can keep close/log-range/
quote-volume EMAs hotter or staler than the equivalent no-trade backtest path.

## Desired Contract

For each active 1m open-tail symbol:

1. Projection is allowed only when `_completed_candle_tail_gap_fallback_signature()` would allow
   the same gap today.
2. Projection uses the real cached candle immediately before the open tail as the seed row.
3. Each projected no-trade minute uses:
   - `open = previous close`
   - `high = previous close`
   - `low = previous close`
   - `close = previous close`
   - `volume/base volume = 0.0`
   - quote volume derived from zero base volume, therefore `0.0`
   - log range `log(high / low) = 0.0`
4. Projection applies only to live orchestrator EMA inputs for the bounded open tail.
5. Projection does not write synthetic rows to disk, runtime candle caches, shard indexes, or known
   gap metadata.
6. Projected EMA values must not be cached as normal EMA values under a real `end_ts`, because real
   candles may arrive for the same timestamps.
7. Projection is stateless per EMA read. The bot must not maintain a projected candle timeline such
   as `[..., 11:58_real, 11:59_projected, 12:00_projected]` as authoritative state for the next
   cycle. Every cycle must start from the currently available real candle cache plus any existing
   bounded-gap logic.
8. Real candles always win over projected rows. If a later fetch returns real candles for timestamps
   previously projected, the next EMA calculation must recompute from the real data; no rollback
   step should be needed because projected candles and projected EMA values were never persisted or
   cached as authoritative state.
9. When real candles arrive, the freshness signature becomes normal candle truth and the projection
   state/logging recovers visibly.
10. If projection cannot be computed from complete local real-candle data, the bot must keep the
   existing loud block/fallback behavior rather than inventing values.

## Non-Goals

1. Do not change Rust order behavior.
2. Do not change backtest candle materialization rules.
3. Do not persist open-tail synthetic candles.
4. Do not widen the allowed tail-gap threshold.
5. Do not relax missing EMA hard failures outside the explicitly allowed open-tail projection path.
6. Do not refactor `src/passivbot.py` into `src/live/` as part of this work.

## Implementation Plan

### Phase 1: Add a Read-Only Projection Primitive

Add a narrow helper in `src/candlestick_manager.py` that returns projected EMA metrics without
mutating candle storage. A reasonable shape:

```python
async def get_projected_open_tail_ema_metrics(
    self,
    symbol: str,
    spans_by_metric: dict[str, float],
    *,
    latest_expected_ts: int,
    last_cached_ts: int,
    max_tail_gap_ms: int,
    timeframe: str = "1m",
) -> dict[str, float]:
    ...
```

Implementation details:

1. Support only 1m initially. Raise for other timeframes.
2. Validate `latest_expected_ts >= last_cached_ts`, finite positive spans, and tail age within
   `max_tail_gap_ms`.
3. Fetch the largest required candle window through `latest_expected_ts`, allowing existing
   bounded-gap standardization to include any real candles that have arrived since the previous
   cycle.
4. Build a throwaway in-memory array for this call only:
   - Preserve all real candles returned for the requested window.
   - Preserve existing bounded internal synthetic behavior where the gap is between two real
     candles.
   - Append projected no-trade rows only for the still-open tail after the newest real/bounded
     candle in the window.
5. Compute each metric using the same `_ema_series()` math as normal EMA helpers.
6. Return plain floats plus optional metadata if useful, but do not populate `_ema_cache`.
7. Raise with actionable context if local candles are insufficient.

Preferred metric keys for this first pass:

1. `close`
2. `qv`
3. `log_range`

Keep this helper local to `CandlestickManager`; do not introduce a new service abstraction yet.

### Phase 2: Capture Projection Eligibility In `Passivbot`

Add a small helper in `src/passivbot.py` that converts completed-candle health into a projection
descriptor:

```python
def _active_tail_gap_projection_context(self, symbol: str) -> dict | None:
    ...
```

It should:

1. Call `self.cm.get_completed_candle_health(symbol, {"1m": 1}, now_ms=...)`.
2. Reuse the same eligibility logic as `_completed_candle_tail_gap_fallback_signature()`.
3. Return `latest_expected_ts`, `last_cached_ts`, `tail_gap_age_ms`, `tail_gap_candles`, and
   `max_tail_gap_ms`.
4. Return `None` when there is no allowed open-tail gap.
5. Never fetch remote data.

Avoid duplicating the tail-gap rules. If needed, split `_completed_candle_tail_gap_fallback_signature()`
into a lower-level parser plus the current signature wrapper.

### Phase 3: Route Live Orchestrator EMA Reads Through Projection

Update `_load_orchestrator_ema_bundle()` so projected values are used only for symbols with an
allowed open-tail projection context.

Recommended flow:

1. Compute projection context per symbol before the local `fetch_*` closures.
2. For symbols with context, call the new `CandlestickManager` projection helper for the spans
   required by:
   - `m1_close_emas`
   - `m1_volume_emas` / quote-volume input
   - `m1_log_range_emas`
3. Keep normal `get_latest_ema_log_range(..., timeframe="1h")` for H1. Do not project H1.
4. For close EMAs, projection should be preferred over previous-close-EMA fallback when the only
   issue is an allowed open-tail gap.
5. For non-tail failures, keep the current fail-loud behavior and stale previous-close fallback
   rules.
6. Record projection metadata on the bot for logging/tests, for example:
   `_orchestrator_ema_projection_symbols` and `_orchestrator_ema_projection_details`.

Do not make secondary forager cache-only symbols perform remote fetches because of projection.
If a cache-only secondary lacks enough local real candles to project, mark it unavailable as today.

### Phase 4: Logging And Operator Visibility

Change the visible policy from carry-forward to projection.

Required logs:

1. First projected active tail in a window:
   - tag: `[candle]`
   - action: `project_open_tail_ema`
   - symbols/count
   - newest expected timestamp
   - oldest real timestamp
   - max projected tail age
   - configured max age
2. Near threshold:
   - keep WARNING behavior analogous to current carry-forward warning.
3. Recovery:
   - existing recovery log can remain, but change action wording to
     `resume_completed_candle_truth`.
4. Projection failure:
   - WARNING for active symbols when projection was eligible but local data was insufficient.
   - Include `symbol`, `tail_gap_age_ms`, `latest_expected_ts`, `last_cached_ts`, and error type.

Update wording in docs from `carry_forward_latest_real` to projection once behavior changes.

### Phase 5: Tests First Around The Contract

Add focused unit tests before or alongside implementation.

`tests/test_live_candle_budget.py`:

1. Existing bounded tail-gap freshness tests should continue passing.
2. Add a projection test where a three-minute open tail projects close, quote-volume, and log-range
   EMAs from local candles and does not call remote fetch.
3. Add a parity test comparing:
   - projected open-tail EMA result
   - normal EMA result from an explicit bounded no-trade candle array ending at the same timestamp
4. Add an anomaly replacement test:
   - Cycle A projects `11:59` and `12:00` from last real `11:58`.
   - Cycle B sees real `11:59` and `12:01`, with `12:00` still missing.
   - The EMA result must recompute from `11:59_real` and `12:01_real`, with only `12:00` handled
     by bounded-gap synthesis.
   - The Cycle A projected EMA values must not be reused.
5. Add a no-persistence test proving runtime cache length, disk shards, `_synthetic_timestamps`,
   and `_ema_cache` are not mutated as normal real/synthetic candles.
6. Add a threshold test proving projection is refused once `tail_gap_age_ms` exceeds
   `live.max_active_candle_tail_gap_minutes`.
7. Add a recovery test proving normal completed-candle truth wins once real candles exist.

`tests/test_missing_ema_fix.py`:

1. Add a close-EMA test proving allowed open-tail projection is used before previous-close fallback.
2. Keep the stale previous-close fallback test intact for non-projection failures.

`tests/test_unstucking_safeguards.py` or a narrower orchestrator test:

1. Add an active-symbol path proving projected EMAs make the symbol tradable when all other inputs
   are valid.
2. Add a failure path proving missing local seed candles still marks the symbol unavailable or raises
   visibly according to the current active/secondary contract.

### Phase 6: Documentation Updates

Update after tests and behavior are in place:

1. `docs/plans/staged_live_data_reconciliation_plan.md`
   - Mark the open-tail EMA projection task done.
   - Mark relevant test bullets done.
   - Replace carry-forward language where it no longer applies.
2. `docs/ai/features/candlestick_manager.md`
   - Replace "known parity risk" with the new projection contract.
   - Keep a warning that projection is provisional and non-persistent.
3. `docs/configuration.md`
   - Update `max_active_candle_tail_gap_minutes` wording so users know bounded tails project EMA
     inputs rather than simply carrying forward latest real EMA state.
4. `CHANGELOG.md`
   - Add an Unreleased entry because live behavior changes for bounded missing tail candles.

## Validation Commands

Minimum targeted validation:

```bash
./venv/bin/python -m pytest \
  tests/test_live_candle_budget.py \
  tests/test_missing_ema_fix.py \
  tests/test_unstucking_safeguards.py::test_orchestrator_marks_ema_unavailable_symbols_non_tradable
```

If implementation touches Rust/PyO3 boundaries unexpectedly, stop and reassess. This plan should
not require Rust changes.

Recommended broader validation:

```bash
./venv/bin/python -m pytest \
  tests/test_live_candle_budget.py \
  tests/test_missing_ema_fix.py \
  tests/test_passivbot_balance_split.py \
  tests/test_unstucking_safeguards.py
```

Optional fake-live validation if the unit tests expose enough hooks:

```bash
./venv/bin/python -m pytest tests/test_run_fake_live.py -m fake_live
```

## Acceptance Criteria

The work is complete when:

1. Bounded active open-tail live planning uses projected EMA inputs instead of latest-real
   carry-forward for 1m close, quote-volume, and log-range EMAs.
2. Projection is bounded by `live.max_active_candle_tail_gap_minutes`.
3. Projection is read-only and leaves disk/runtime candle storage untouched.
4. Real candles arriving for previously projected timestamps replace projection automatically
   through normal reads; projected EMA values from earlier cycles are not reused.
5. Symbols are not allowed to trade on fabricated inputs if local real candles are insufficient to
   seed the projection.
6. Existing non-tail missing EMA failures still fail loudly.
7. Secondary forager cache-only behavior remains cache-only and does not create new remote fetch
   pressure.
8. Logs make projection, near-limit warning, failure, and recovery visible.
9. Targeted tests cover projection, parity, no-persistence, threshold refusal, recovery, and
   previous-close fallback precedence.
10. User-facing docs and changelog describe the behavior change.

## Risks And Mitigations

1. Risk: projection accidentally caches values as real EMA results.
   Mitigation: do not write `_ema_cache` in the projection helper; add no-cache tests.
2. Risk: projection changes inactive forager behavior and increases remote calls.
   Mitigation: keep cache-only symbols cache-only; add a test around `allow_remote_fetch=False`.
3. Risk: quote-volume/log-range semantics drift from bounded no-trade synthetic candles.
   Mitigation: build parity tests against explicit zero-volume/flat OHLC rows.
4. Risk: duplicate tail-gap eligibility logic diverges.
   Mitigation: share parsing/validation between freshness signature and projection context.
5. Risk: live hides a genuine exchange outage.
   Mitigation: preserve threshold blocking, WARNING near threshold, and failure logs when projection
   cannot be computed.
6. Risk: late real candles arrive for timestamps that were projected in an earlier cycle.
   Mitigation: make projection stateless per read, prefer real candles in the fetched window, and
   add the anomaly replacement regression test described above.

## Suggested Commit Slices

1. Add projection helper and pure `CandlestickManager` unit tests.
2. Wire projection into `_load_orchestrator_ema_bundle()` with passivbot-level tests.
3. Update logging and recovery wording.
4. Update docs and changelog.
