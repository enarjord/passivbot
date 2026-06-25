# Live Event Registry

This file documents the stable live event tag and reason-code registries in
`src/live/event_bus.py`. Use these values for structured live events when the
value is query-facing or repeated across producers.

Do not add aliases for temporary branch-local spellings. Prefer adding a new
registry value only when operators or tools should be able to filter for it
across time.

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
- `fallback`
- `fill`
- `fills`
- `flush`
- `forager`
- `gate`
- `health`
- `load`
- `logging`
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
- `unavailable`
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
- `exchange_exception`
- `length_mismatch`
- `new_fill`
- `open_tail_projection`
- `optional_ema_dropped`
- `periodic_health_summary`
- `queue_full`
- `ranking_features_unavailable`
- `remote_fetch`
- `required_candle_disk_coverage`
- `required_ema_unavailable`
- `rust_output_actions`
- `pipeline_closing`
- `snapshot_symbol_state`
- `startup_phase_ready`
- `submitted_to_exchange`
- `warmup_cache_decision`

Dynamic helpers are part of the same contract:

- `authoritative_reason_code(surface)` emits `authoritative_<surface>`.
- `sink_failed_reason_code(name)` emits `<name>_sink_failed`.
