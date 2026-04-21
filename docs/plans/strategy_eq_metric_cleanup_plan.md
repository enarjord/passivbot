# Strategy Equity Metric Cleanup Plan

## Purpose

Define the implementation plan for replacing the ambiguous `*_hsl` analysis risk metrics and the verbose
`*_strategy_pnl_rebased` metrics with a clearer canonical `*_strategy_eq` metric family.

This is a planning document only. It describes the intended behavior, migration path, and test plan before code changes begin.

## Problem

Current backtest analysis has overlapping metric families:

- `*_pnl`
- `*_usd`
- `*_btc`
- `*_strategy_pnl_rebased`
- `*_hsl`
- `hard_stop_*`

The current `*_hsl` risk metrics are muddy because they are not purely hard-stop operational metrics. They are partly derived from the same synthetic strategy-equity path as `*_strategy_pnl_rebased`, but their drawdown and recovery logic uses separate peak-reference handling.

This creates confusion such as:

- `peak_recovery_hours_hsl` can be much larger than `peak_recovery_hours_pnl`
- `peak_recovery_hours_hsl` can be nonzero even when per-side HSL is disabled
- `drawdown_worst_hsl` sounds like a hard-stop event metric, but it is effectively a strategy-equity drawdown metric
- `strategy_pnl_rebased` is accurate but too verbose for routine metric names

## Target Metric Contract

### `*_pnl`

Meaning:

- realized PnL only
- no unrealized PnL
- no collateral repricing
- shared across USD/BTC analysis output

Canonical PnL input:

```text
net_realized_pnl = fill.pnl + fill.fee_paid
net_realized_pnl_cumsum = cumsum(net_realized_pnl)
```

Target behavior:

- `adg_pnl`: mean daily net realized PnL ratio
- `mdg_pnl`: median daily net realized PnL ratio
- `sharpe_ratio_pnl`: Sharpe over daily net realized PnL ratios
- `sortino_ratio_pnl`: Sortino over daily net realized PnL ratios
- `peak_recovery_hours_pnl`: longest duration from a net-realized-PnL cumsum peak to the next exceeded peak, including the open interval from the last peak to the final backtest timestamp

Important change:

- current `adg_pnl` / `mdg_pnl` already use net daily PnL ratios
- current `peak_recovery_hours_pnl` uses gross `fill.pnl` and ignores open tail to backtest end
- implementation should standardize `peak_recovery_hours_pnl` on net PnL and include the open tail

### `*_usd`

Meaning:

- actual account equity expressed in USD
- includes realized PnL, fees, unrealized PnL, cash wallet state, and BTC collateral repricing effects

Conceptual series:

```text
equity_usd = usd_total_balance + upnl_usd
```

Examples:

- `adg_usd`
- `drawdown_worst_usd`
- `peak_recovery_hours_equity_usd`

No naming change is planned for this family.

### `*_btc`

Meaning:

- actual account equity expressed in BTC
- includes the same account effects as USD equity, denominated by BTC price

Conceptual series:

```text
equity_btc = btc_total_balance + upnl_usd / btc_usd_price
```

Examples:

- `adg_btc`
- `drawdown_worst_btc`
- `peak_recovery_hours_equity_btc`

No naming change is planned for this family.

### `*_strategy_eq`

Meaning:

- collateral-agnostic synthetic strategy equity
- includes realized net PnL plus unrealized PnL
- rebased to `starting_balance`
- independent of USD/BTC collateral denomination effects

Canonical series:

```text
net_realized_pnl_cumsum = cumsum(fill.pnl + fill.fee_paid)
upnl_series = long_upnl + short_upnl
strategy_pnl = net_realized_pnl_cumsum.reindex(upnl_series.index).ffill() + upnl_series
strategy_eq = starting_balance + strategy_pnl
```

Canonical metric names:

- `gain_strategy_eq`
- `adg_strategy_eq`
- `mdg_strategy_eq`
- `sharpe_ratio_strategy_eq`
- `sortino_ratio_strategy_eq`
- `omega_ratio_strategy_eq`
- `expected_shortfall_1pct_strategy_eq`
- `calmar_ratio_strategy_eq`
- `sterling_ratio_strategy_eq`
- `drawdown_worst_strategy_eq`
- `drawdown_worst_mean_1pct_strategy_eq`
- `peak_recovery_hours_strategy_eq`

Weighted/subset variants:

- `adg_strategy_eq_w`
- `mdg_strategy_eq_w`
- `sharpe_ratio_strategy_eq_w`
- `sortino_ratio_strategy_eq_w`
- `omega_ratio_strategy_eq_w`
- `calmar_ratio_strategy_eq_w`
- `sterling_ratio_strategy_eq_w`

Optional EMA drawdown variants:

- `drawdown_worst_ema_strategy_eq`
- `drawdown_worst_mean_1pct_ema_strategy_eq`

These should only remain if the EMA drawdown signal is still useful outside HSL tiering. Otherwise, keep EMA drawdown internal to hard-stop logic.

### `hard_stop_*`

Meaning:

- operational hard-stop telemetry only
- no strategy-equity performance/risk metrics

Examples:

- `hard_stop_triggers`
- `hard_stop_triggers_per_year`
- `hard_stop_restarts`
- `hard_stop_restarts_per_year`
- `hard_stop_time_in_yellow_pct`
- `hard_stop_time_in_orange_pct`
- `hard_stop_time_in_red_pct`
- `hard_stop_duration_minutes_mean`
- `hard_stop_duration_minutes_max`
- `hard_stop_trigger_drawdown_mean`
- `hard_stop_panic_close_loss_sum`
- `hard_stop_panic_close_loss_max`
- `hard_stop_flatten_time_minutes_mean`
- `hard_stop_post_restart_retrigger_pct`

## Deprecated / Alias Metric Names

Canonical names should be used by all code, docs, config defaults, optimizer objectives, suite examples, and tests.

The old names should remain accepted through centralized aliasing for at least one release cycle:

| Old name | Canonical alias |
| --- | --- |
| `gain_strategy_pnl_rebased` | `gain_strategy_eq` |
| `adg_strategy_pnl_rebased` | `adg_strategy_eq` |
| `mdg_strategy_pnl_rebased` | `mdg_strategy_eq` |
| `sharpe_ratio_strategy_pnl_rebased` | `sharpe_ratio_strategy_eq` |
| `sortino_ratio_strategy_pnl_rebased` | `sortino_ratio_strategy_eq` |
| `omega_ratio_strategy_pnl_rebased` | `omega_ratio_strategy_eq` |
| `expected_shortfall_1pct_strategy_pnl_rebased` | `expected_shortfall_1pct_strategy_eq` |
| `calmar_ratio_strategy_pnl_rebased` | `calmar_ratio_strategy_eq` |
| `sterling_ratio_strategy_pnl_rebased` | `sterling_ratio_strategy_eq` |
| `adg_strategy_pnl_rebased_w` | `adg_strategy_eq_w` |
| `mdg_strategy_pnl_rebased_w` | `mdg_strategy_eq_w` |
| `sharpe_ratio_strategy_pnl_rebased_w` | `sharpe_ratio_strategy_eq_w` |
| `sortino_ratio_strategy_pnl_rebased_w` | `sortino_ratio_strategy_eq_w` |
| `omega_ratio_strategy_pnl_rebased_w` | `omega_ratio_strategy_eq_w` |
| `calmar_ratio_strategy_pnl_rebased_w` | `calmar_ratio_strategy_eq_w` |
| `sterling_ratio_strategy_pnl_rebased_w` | `sterling_ratio_strategy_eq_w` |
| `drawdown_worst_hsl` | `drawdown_worst_strategy_eq` |
| `drawdown_worst_mean_1pct_hsl` | `drawdown_worst_mean_1pct_strategy_eq` |
| `peak_recovery_hours_hsl` | `peak_recovery_hours_strategy_eq` |
| `drawdown_worst_ema_hsl` | `drawdown_worst_ema_strategy_eq` |
| `drawdown_worst_mean_1pct_ema_hsl` | `drawdown_worst_mean_1pct_ema_strategy_eq` |

Per-side aliases should follow the same pattern:

- `drawdown_worst_hsl_long` -> `drawdown_worst_strategy_eq_long`
- `drawdown_worst_hsl_short` -> `drawdown_worst_strategy_eq_short`
- `peak_recovery_hours_hsl_long` -> `peak_recovery_hours_strategy_eq_long`
- `peak_recovery_hours_hsl_short` -> `peak_recovery_hours_strategy_eq_short`

## Alias Architecture

Aliases must be centralized. Do not scatter ad hoc replacements through optimizer, pareto, suite runner, or docs tooling.

Add a canonical metric alias module, likely under `src/config/metrics.py` or a sibling module.

Required API shape:

```python
canonical_metric_name(name: str) -> str
canonicalize_metric_mapping(mapping: dict) -> dict
canonicalize_metric_list(values: list[str]) -> list[str]
```

The alias layer should be used by:

- config parsing
- config formatting
- config migration / hydration
- optimizer objective parsing
- optimizer limit parsing
- suite aggregate config parsing
- pareto tool objective/limit/target parsing
- analysis visibility config
- docs/examples validation if applicable

Behavior:

- canonical names are stored in prepared configs
- old names are accepted as input aliases
- warning/deprecation logging can be added, but should be coalesced and not noisy
- output JSON should include canonical names
- old alias fields may optionally be emitted for one release, but canonical fields must be authoritative

Review question:

- Decide whether analysis output should include old alias fields for one release, or only accept old names in config/query tooling.

Recommendation:

- emit canonical fields only in new `analysis.json`
- accept old names in optimizer, pareto, limits, objectives, aggregate config, and visibility filters
- this avoids bloating analysis output while preserving user workflows

## Rust Implementation Plan

### 1. Net realized PnL recovery

Update `peak_recovery_hours_pnl` to:

- accumulate `fill.pnl + fill.fee_paid`
- measure peak-to-next-exceeded-peak duration
- include open tail from the last peak timestamp to final backtest timestamp

This requires passing the final backtest timestamp into the PnL recovery helper.

Candidate signature:

```rust
fn calc_peak_recovery_hours_pnl(fills: &[Fill], final_timestamp_ms: Option<u64>) -> f64
```

Use `timestamps_ms.last()` from the full backtest equity series when available.

### 2. Canonical strategy equity series

Keep or refine the existing strategy-equity sampling:

```rust
strategy_pnl = pnl_cumsum_running_net + unrealized_pnl
strategy_eq = starting_balance + strategy_pnl
```

The key change is semantic cleanup:

- metrics derived from this series should be named `*_strategy_eq`
- drawdown and peak recovery should use this series directly
- avoid comparing actual USD equity to a separate strategy peak reference for canonical strategy-equity risk metrics

### 3. Strategy-equity drawdown metrics

Compute strategy-equity drawdown directly:

```text
drawdown = 1 - strategy_eq / strategy_eq.cummax()
```

Expose:

- `drawdown_worst_strategy_eq`
- `drawdown_worst_mean_1pct_strategy_eq`
- per-side variants if per-side strategy-equity series are enabled

### 4. Strategy-equity peak recovery

Compute:

```text
peak_recovery_hours_strategy_eq = longest peak-to-next-exceeded-peak duration on strategy_eq, including open tail to final timestamp
```

This should use the same conceptual recovery helper as USD/BTC equity and PnL, but with explicit open-tail handling.

### 5. HSL runtime drawdown

For hard-stop tiering, use the same collateral-agnostic strategy-equity concept.

For rolling `pnls_max_lookback_days` behavior, the runtime should be equivalent to:

```python
cutoff = now - pnls_max_lookback_days
rolling_net_pnls = fills_after_cutoff.pnl_plus_fee
rolling_upnls = upnl_samples_after_cutoff
rolling_pnl_cumsum = rolling_net_pnls.cumsum().reindex(rolling_upnls.index).ffill()
rolling_strategy_pnl = rolling_pnl_cumsum + rolling_upnls
rolling_strategy_eq = baseline + rolling_strategy_pnl
rolling_drawdown = 1 - rolling_strategy_eq / rolling_strategy_eq.cummax()
```

Implementation can remain incremental in Rust:

- maintain rolling net realized PnL samples
- sample UPNL each timestep
- derive current rolling strategy equity
- maintain rolling peak for HSL tier logic
- compute HSL drawdown from strategy equity against that rolling peak

Important:

- HSL operational decisions may keep rolling-window semantics
- canonical `*_strategy_eq` analysis metrics should be full-series unless explicitly named rolling

### 6. Rust structs and PyO3 export

Update Rust structs:

- `Analysis`
- `HardStopMetrics`
- `StrategyEquityMetrics`
- Python export in `python.rs`

Add canonical fields:

- `*_strategy_eq`

Preserve old internal fields only as needed during transition.

## Python Implementation Plan

### 1. Metric alias centralization

Add/extend a central alias map in config metrics code.

Use it in:

- `src/config/metrics.py`
- `src/config/scoring.py`
- `src/config/schema.py`
- `src/config/limits.py`
- `src/limit_utils.py`
- `src/optimize.py`
- `src/pareto_explorer.py`
- `src/pareto_store.py`
- `src/suite_runner.py`
- `src/analysis_visibility.py`
- tools that parse metric names, including pareto and iterative tools

### 2. Defaults and examples

Update defaults to canonical metric names:

- optimize objectives
- optimize limits
- backtest aggregate defaults
- suite examples
- docs examples

### 3. Output handling

Preferred behavior:

- write canonical metric names to `analysis.json`
- do not duplicate deprecated alias fields
- allow old names when reading older Pareto directories or applying limits/objectives

If compatibility requires old fields in output for one release, add them only in the Python expansion layer with a clear TODO/removal note.

## Documentation Plan

Update:

- `CHANGELOG.md`
- `docs/tools.md`
- `docs/suite_examples.md`
- metric-related docs under `docs/`
- optimizer / pareto examples

Document the metric families:

- `*_pnl`: net realized PnL only
- `*_usd`: actual USD equity
- `*_btc`: actual BTC equity
- `*_strategy_eq`: collateral-agnostic strategy equity
- `hard_stop_*`: operational hard-stop telemetry

Avoid using `*_hsl` in new docs except in a compatibility/deprecation note.

## Testing Plan

### Rust tests

Add or update tests for:

1. `peak_recovery_hours_pnl` uses net realized PnL, including fees.
2. `peak_recovery_hours_pnl` includes open tail to final timestamp.
3. `strategy_eq = starting_balance + net_realized_pnl_cumsum + upnl`.
4. `drawdown_worst_strategy_eq` is computed from strategy-equity cummax.
5. `peak_recovery_hours_strategy_eq` includes open tail.
6. Old `*_hsl` aliases, if still exported, match canonical `*_strategy_eq` values.

### Python tests

Add or update tests for:

1. Metric aliases in optimizer objectives.
2. Metric aliases in optimizer limits.
3. Metric aliases in suite aggregate config.
4. Metric aliases in `passivbot tool pareto`.
5. Metric aliases in analysis visibility.
6. Prepared configs store canonical names.
7. Existing old configs using `*_hsl` and `*_strategy_pnl_rebased` still load.

### End-to-end smoke

Run one short backtest and verify:

- `analysis.json` contains canonical `*_strategy_eq` metrics
- `peak_recovery_hours_pnl` changes only where fees/open-tail semantics require it
- old metric names still work in CLI limits/objectives

Example compatibility checks:

```bash
passivbot optimize ... -l "peak_recovery_hours_hsl<5000"
passivbot optimize ... -l "peak_recovery_hours_strategy_eq<5000"
passivbot tool pareto ... -l "drawdown_worst_hsl<0.8"
passivbot tool pareto ... -l "drawdown_worst_strategy_eq<0.8"
```

Both old and new forms should resolve to the same canonical metric.

## Rollout Plan

Phase 1:

- add canonical fields
- add aliases
- update config defaults and docs
- keep old names accepted

Phase 2:

- update optimizer/pareto examples and scoring to use canonical names only
- monitor whether old aliases are still used

Phase 3:

- decide whether to keep aliases indefinitely or warn more aggressively
- remove old output fields if they were emitted temporarily

## Open Review Points

1. Should `analysis.json` emit old alias fields for one release, or only canonical fields?
   - recommendation: canonical fields only, aliases accepted at read/query/config boundaries

2. Should EMA drawdown metrics remain public under `*_strategy_eq`, or become internal HSL-only state?
   - recommendation: keep only if actively used in optimizer/scoring; otherwise keep internal

3. Should per-side strategy-equity metrics be kept?
   - recommendation: keep per-side drawdown/recovery only if HSL per-side reporting remains valuable; otherwise avoid expanding public metric surface

4. Should `peak_recovery_hours_equity_usd/btc` also include open tail?
   - recommendation: yes, for consistency, but handle as a separate explicit change if current behavior differs and users rely on it

