# Live Event Registry

This file documents the stable live event tag and reason-code registries in
`src/live/event_bus.py`. Use these values for structured live events when the
value is query-facing or repeated across producers.

Do not add aliases for temporary branch-local spellings. Prefer adding a new
registry value only when operators or tools should be able to filter for it
across time.

## Event Types

- `action.planned`
- `balance.changed`
- `bot.ready`
- `bot.shutdown.stage`
- `bot.started`
- `bot.startup_timing`
- `bot.stopped`
- `bot.stopping`
- `cache.flush.completed`
- `cache.load.completed`
- `cache.warmup_decision`
- `candle.coverage_checked`
- `candle.tail_projected`
- `config.market_compatibility`
- `cycle.completed`
- `cycle.degraded`
- `cycle.started`
- `data_packet.updated`
- `ema.bundle.completed`
- `ema.bundle.started`
- `ema.fallback_used`
- `ema.unavailable`
- `exchange.config_refresh`
- `exchange.time_sync`
- `execution.ambiguous`
- `execution.cancel_ambiguous_terminal`
- `execution.cancel_connector_call_started`
- `execution.cancel_failed`
- `execution.cancel_sent`
- `execution.cancel_succeeded`
- `execution.confirmation_requested`
- `execution.confirmation_satisfied`
- `execution.confirmation_timeout`
- `execution.create_connector_call_started`
- `execution.create_deferred`
- `execution.create_failed`
- `execution.create_rejected`
- `execution.create_sent`
- `execution.create_skipped`
- `execution.create_succeeded`
- `entry.initial_distance_gate_blocked`
- `entry.initial_distance_gate_cleared`
- `entry.initial_eligibility`
- `entry.min_effective_cost_blocked`
- `fill.ingested`
- `fills.refresh_summary`
- `forager.feature_unavailable`
- `forager.eligibility_changed`
- `forager.selection`
- `health.summary`
- `market.snapshot_diagnostic_skipped`
- `hsl.cooldown_ended`
- `hsl.cooldown_started`
- `hsl.raw_red_pending`
- `hsl.red_finalized_without_order`
- `hsl.red_triggered`
- `hsl.replay.cache`
- `hsl.replay.completed`
- `hsl.replay.failed`
- `hsl.replay.progress`
- `hsl.replay.started`
- `hsl.status`
- `hsl.transition`
- `order_wave.completed`
- `order_wave.started`
- `planning.defer_summary`
- `planning.symbol_state`
- `planning.unavailable`
- `position.changed`
- `remote_call.failed`
- `remote_call.started`
- `remote_call.succeeded`
- `remote_call.throttled`
- `risk.mode_changed`
- `risk.entry_cooldown_delta_anchored`
- `risk.realized_loss_gate_blocked`
- `rust_orchestrator.called`
- `rust_orchestrator.returned`
- `sink.degraded`
- `snapshot.built`
- `state.refresh_progress`
- `state.refresh_timing`
- `trailing.status`
- `unstuck.selection`
- `unstuck.status`
- `websocket.reconnect`

## Fresh-Entry Eligibility Contract

Completed normal live order plans emit one `entry.initial_eligibility` event
to structured and monitor sinks only. The producer observes the existing Rust
plan, Python reconciliation, and local pre-connector filters; it does not add
or re-evaluate a trading gate.

Each evaluated symbol/position-side record has one outcome, in this precedence:

1. `eligible`: an initial-entry order survived the final batch cap and was
   selected for the connector-bound `execute_orders` list immediately before
   invocation.
2. `blocked_candidate`: an initial candidate existed but no candidate reached
   that boundary.
3. `already_satisfied`: reconciliation removed the candidate because an open
   order matched exactly or within the configured order-match tolerance.
4. `protective_only`: the pair had protective/reduce-only/panic actions but no
   initial candidate.
5. `no_candidate`: the evaluated pair had no initial candidate; the stable
   default reason is `rust_no_initial_candidate`.

Stable per-record `reason_counts` values are:

- `batch_capacity`
- `conversion_zero_or_duplicate`
- `debug_mode`
- `exact_reconciliation_match`
- `freshness_creation_guardrail`
- `hsl_replay_pending`
- `initial_entry_distance_gate`
- `limit_order_create_market_distance`
- `low_balance`
- `malformed_actual_orders`
- `mode_filter`
- `order_match_tolerance`
- `pending_exchange_config`
- `pre_create_market_snapshot_unavailable`
- `pre_create_planning_snapshot_invalid`
- `protective_actions_only`
- `recent_execution`
- `rust_no_initial_candidate`
- `state_change_detected`
- `trailing_unavailable`
- `unclassified_candidate` for an initial candidate not accounted for by a
  known observation point
- `reason_overflow` when a record exceeds its bounded reason-key limit

`eligible` does not claim connector invocation completed, exchange acceptance,
or order acknowledgement. Connector and exchange outcomes remain in the
`execution.*` event family. The event retains full aggregate counts, samples at
most 32 deterministic pair records, and contains no order price, quantity, raw
payload, path, secret, or exception text. Planning failures, deferrals,
shutdown interruption, and diagnostic tracing failures omit the event instead
of emitting a misleading candidate-free result. Diagnostic and sink failures
must not change order lists or execution results.

## Connector Call Boundary Contract

`execution.create_connector_call_started` and
`execution.cancel_connector_call_started` are emitted to structured and monitor
sinks only, immediately before Passivbot calls the concrete
`cca.create_order` or `cca.cancel_order` coroutine. They mean only that local
execution reached that connector call site. They do not claim that bytes were
sent, that the exchange received or accepted the request, or that an order was
acknowledged.

The fixed payload fields are `action=create|cancel`,
`connector_method=cca.create_order|cca.cancel_order`, and
`connector_route=base|hyperliquid|okx`, plus bounded order shape fields and
finite numeric values. Normal order-plan calls retain their cycle,
`order_wave_id`, and `action_id` correlation. The events must not contain raw
connector params, vault addresses, URLs, responses, exception text, paths, or
arbitrary payloads. Diagnostic and sink failures must not prevent or alter the
connector call.

These events are distinct from the existing `execution.create_sent` and
`execution.cancel_sent` pre-call submission-intent events and from terminal
exchange outcome events. They do not define another startup performance
milestone.

## HSL Replay Timing Fields

For coin-mode `hsl.replay.completed`, `full_elapsed_s` measures total replay
time, while `protective_elapsed_s` and `startup_blocking_elapsed_s` measure the
time until all currently held pairs are protectively ready. Use
`replay_loop_elapsed_s` for the replay-loop-only duration. Cache status events
use `hsl.replay.cache` with `cache_status=hit|miss|rejected`; misses and
rejections are non-authoritative performance-cache outcomes and should fall
back to exchange-derived replay.

Coin mode emits one `hsl.replay.progress` event with `stage=pair_replay` when
the first frozen replay candidate starts, keeps in-pair progress time-throttled,
and emits one terminal progress sample for each pair. `applied_rows` and
`total_applied_rows` retain state-update counts; `scanned_rows`,
`total_scanned_rows`, `scanned_rows_per_second`, and `pair_elapsed_s` expose
candidate-row scan cost, including pairs which apply no rows. `is_held_pair`,
`is_cooldown_pair`, and `pair_idx` expose the deterministic
held/cooldown/remaining ordering without controlling it.
After the held batch completes, `stage=held_protective_ready` exposes bounded
`ready_pairs`, `pending_pairs`, and `protective_elapsed_s` fields. Remaining
pairs continue in the same replay task; `hsl.replay.completed` remains the
full-replay terminal event.

## Event Tags

- `account`
- `action`
- `authoritative`
- `availability`
- `balance`
- `bundle`
- `cache`
- `candle`
- `confirmation`
- `coverage`
- `cycle`
- `defer`
- `degraded`
- `ema`
- `entry`
- `execution`
- `exchange`
- `fallback`
- `fill`
- `fills`
- `flush`
- `forager`
- `gate`
- `health`
- `load`
- `logging`
- `market`
- `mode`
- `order`
- `planning`
- `position`
- `refresh`
- `remote_call`
- `resource`
- `risk`
- `rust`
- `selection`
- `sink`
- `snapshot`
- `state`
- `summary`
- `tail`
- `timeout`
- `time_sync`
- `trailing`
- `unavailable`
- `unstuck`
- `warmup`
- `wave`
- `websocket`

## Reason Codes

- `authoritative_confirmation`
- `authoritative_confirmation_timeout`
- `balance_changed`
- `candle_disk_flush_completed`
- `candle_disk_load_completed`
- `config_market_unsupported`
- `config_hip3_account_mode_unsupported`
- `config_isolated_only_market_blocked`
- `config_stock_perp_unavailable_market`
- `config_stock_perp_wrong_exchange`
- `connector_call_started`
- `ema_fallback_used`
- `exchange_acknowledged`
- `exchange_config_refresh`
- `exchange_config_refresh_failed`
- `exchange_exception`
- `exchange_time_sync`
- `exchange_time_sync_unavailable`
- `execution_loop_error_burst`
- `fresh_entry_eligibility`
- `entry_cooldown_position_delta`
- `fill_cache_doctor_report`
- `fill_cache_quarantined`
- `fill_cache_ready`
- `fill_cache_rebuild_started`
- `forager_eligibility_membership_changed`
- `hsl_balance_override_account_level_replay_unsafe`
- `hsl_history_empty`
- `hsl_history_inputs_loaded`
- `hsl_held_protective_ready`
- `hsl_price_history_fetch_completed`
- `hsl_price_history_fetch_started`
- `hsl_price_history_symbol_fetch_completed`
- `hsl_price_history_symbol_fetch_started`
- `hsl_raw_red_pending_ema_confirmation`
- `hsl_red_finalized_without_exchange_order`
- `hsl_replay_cache_hit`
- `hsl_replay_cache_miss`
- `hsl_replay_cache_rejected`
- `hsl_replay_cache_write_failed`
- `hsl_replay_cache_written`
- `hsl_replay_pending`
- `hsl_timeline_replay_completed`
- `hsl_timeline_replay_started`
- `initial_entry_distance_gate`
- `length_mismatch`
- `limit_order_create_market_distance`
- `low_balance`
- `market_snapshot_diagnostic_skipped`
- `min_effective_cost_blocked`
- `new_fill`
- `open_tail_projection`
- `optional_ema_dropped`
- `pending_exchange_config`
- `periodic_health_summary`
- `pre_create_market_snapshot_unavailable`
- `pre_create_planning_snapshot_invalid`
- `queue_full`
- `ranking_features_unavailable`
- `recent_execution`
- `remote_fetch`
- `realized_loss_gate_blocked`
- `required_candle_disk_coverage`
- `required_ema_unavailable`
- `rust_output_actions`
- `pipeline_closing`
- `snapshot_symbol_state`
- `startup_phase_ready`
- `staged_refresh_progress`
- `staged_refresh_timing`
- `state_change_detected`
- `submitted_to_exchange`
- `trailing_status`
- `unstuck_selection`
- `unstuck_status`
- `warmup_cache_decision`
- `websocket_reconnect`

## Forager Eligibility Change Events

`forager.eligibility_changed` uses reason code
`forager_eligibility_membership_changed` for existing approved/ignored membership
updates. It routes to structured and monitor sinks only, not console or text
sinks. Its data payload is limited to `source` (`config_sources` or
`live_value`), `list_kind`, `operation`, and ordered per-pside `changes` rows.
Each change row contains its total `count` and at most 12 sorted `symbols`; no
config path, raw source, or full list is retained. Emission is best-effort and
must not affect eligibility refresh behavior.

## Configured Market Compatibility Events

`config.market_compatibility` records configured symbols removed by the
existing eligible-market filter. It routes to structured and monitor sinks
only and preserves the existing text-log warning. Unsupported approved symbols
use status `degraded`, making them non-hard problem events; unsupported ignored
symbols use status `skipped` because they do not weaken configured trading
intent.

The event envelope identifies one affected `pside`. The payload contains
`list_kind`, total `skipped_count`, at most 12 sorted `skipped_symbols`, a
truncation flag, bounded `reason_counts`, and bounded
per-reason samples. Symbol samples are value-redacted and length-bounded before
durable publication. Stable reasons distinguish generic unsupported markets,
stock perps configured on a non-Hyperliquid exchange, and unavailable
Hyperliquid stock-perp markets. Emission is best-effort and must not change
coin filtering, exchange calls, or trading behavior.

Before the existing Hyperliquid non-unified HIP-3 startup guard raises,
`config_hip3_account_mode_unsupported` records one hard failed compatibility
event. Its account-level payload contains only a bounded, pre-redacted account
abstraction, a stable action, and count/sample summaries for approved,
position, open-order, isolated-only, and live-isolated symbols. The producer
requires enqueue and waits at most 0.1 seconds for a best-effort terminal flush;
emission or flush failure never suppresses or replaces the fatal startup error.

When the existing generic CCXT margin-policy filter blocks isolated-only
markets from new entries under cross-margin preference,
`config_isolated_only_market_blocked` records one bounded per-side degraded
compatibility event. Its payload contains only a stable action and margin
capability/preference plus count/sample/truncation context for blocked symbols.
Event enqueue failure is retryable independently of the existing text-warning
dedupe; observability must not alter filtering or existing-state handling.

Dynamic helpers are part of the same contract:

- `authoritative_reason_code(surface)` emits `authoritative_<surface>`.
- `sink_failed_reason_code(name)` emits `<name>_sink_failed`.

## Debug Profiles

Set `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` to one or more comma-separated values
to add bounded, opt-in diagnostic details for selected live event families.
`all` enables every profile. Empty, false-like, or `none` values disable debug
profile enrichment.

- `cache`
- `candles`
- `ema`
- `execution`
- `fills`
- `forager`
- `hsl`
- `remote_calls`
- `rust`
- `startup`
- `state`
