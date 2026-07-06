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
- `execution.cancel_failed`
- `execution.cancel_sent`
- `execution.cancel_succeeded`
- `execution.confirmation_requested`
- `execution.confirmation_satisfied`
- `execution.confirmation_timeout`
- `execution.create_deferred`
- `execution.create_failed`
- `execution.create_rejected`
- `execution.create_sent`
- `execution.create_skipped`
- `execution.create_succeeded`
- `entry.initial_distance_gate_blocked`
- `entry.initial_distance_gate_cleared`
- `entry.min_effective_cost_blocked`
- `fill.ingested`
- `fills.refresh_summary`
- `forager.feature_unavailable`
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

## HSL Replay Timing Fields

For `hsl.replay.completed`, `full_elapsed_s`, `startup_blocking_elapsed_s`, and
`elapsed_s` measure total blocking startup time for the replay. Use
`replay_loop_elapsed_s` for the replay-loop-only duration. Cache status events
use `hsl.replay.cache` with `cache_status=hit|miss|rejected`; misses and
rejections are non-authoritative performance-cache outcomes and should fall
back to exchange-derived replay.

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

## Reason Codes

- `authoritative_confirmation`
- `authoritative_confirmation_timeout`
- `balance_changed`
- `candle_disk_flush_completed`
- `candle_disk_load_completed`
- `ema_fallback_used`
- `exchange_acknowledged`
- `exchange_config_refresh`
- `exchange_config_refresh_failed`
- `exchange_exception`
- `exchange_time_sync`
- `exchange_time_sync_unavailable`
- `execution_loop_error_burst`
- `entry_cooldown_position_delta`
- `fill_cache_doctor_report`
- `fill_cache_quarantined`
- `fill_cache_ready`
- `fill_cache_rebuild_started`
- `hsl_balance_override_account_level_replay_unsafe`
- `hsl_history_empty`
- `hsl_history_inputs_loaded`
- `hsl_price_history_fetch_completed`
- `hsl_price_history_fetch_started`
- `hsl_price_history_symbol_fetch_completed`
- `hsl_price_history_symbol_fetch_started`
- `hsl_raw_red_pending_ema_confirmation`
- `hsl_red_finalized_without_exchange_order`
- `hsl_replay_cache_hit`
- `hsl_replay_cache_miss`
- `hsl_replay_cache_rejected`
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
