# Migrate released v8 configurations to v9

Normal loading accepts only the final canonical `config_version: "v9.0.0"` layout.
It applies documented defaults and rejects unknown, conflicting and retired fields.
Formatting does not migrate policy. Unreleased intermediate v9 layouts, `pb_multi`
and pre-v8 layouts are not supported compatibility formats.

The explicit offline converter writes config and a change report. It makes no exchange
requests and starts no bot. Keep original source/config pairs and inspect the report
before backtesting a proposed replacement.

## Structural changes

| Released field | Final v9 field or action |
|---|---|
| `live.hsl_signal_mode` | `bot.risk.hsl.scope`: `coin`, `pside` or `portfolio` |
| `bot.hsl` portfolio policy | `bot.risk.hsl` alongside its scope |
| HSL `ema_span_minutes` | `drawdown_ratio_ema_span_minutes` in the same policy |
| `live.max_realized_loss_pct` | `bot.risk.loss_gate.max_drawdown_ratio` |
| Unstuck `threshold` | `bot.<side>.unstuck.exposure_trigger_ratio` |
| Risk `n_positions` | `bot.<side>.risk.max_positions` |
| `backtest.liquidation_threshold` | `backtest.equity_floor_ratio` |
| `live.auto_gs` | `live.disapproved_position_mode`: `graceful_stop` or `manual` |
| `forager.volume_drop_pct` | Removed; volume-based ranking remains |
| Optional total-exposure entry gate | Removed; the entry cap always applies |
| `trailing_grid_v7`, Defx and Paradex | Removed; select a retained strategy/venue explicitly |
| Legacy `_w` metrics | Removed; select replacement formulas explicitly |
| DEAP search | Removed; choose an explicit pymoo NSGA-II/NSGA-III operator policy |

V9 uses separate slow EMA spans for each HSL policy, global unstuck realized memory and
global loss-gate realized memory. Local unstuck settings remain under `bot.<side>.unstuck`.
The global span is `bot.unstuck.memory_ema_span_minutes`; the gate span is
`bot.risk.loss_gate.memory_ema_span_minutes`; HSL uses its own policy's
`memory_ema_span_minutes`. HSL ratio smoothing is a separate faster span.

Scenario overrides and fixed optimizer overrides convert released flat shared fields
to their canonical side-owned groups. For example, `bot.long.n_positions` becomes
`bot.long.risk.max_positions`, whether authored as a dotted selector or nested maps.
Supported side shorthand such as `long.n_positions` follows the same conversion;
`fixed_params` selectors use it as well. Dictionary members, authored values and
canonical strategy paths remain intact. The converter adds no other override fields.
Unknown fields and conflicting aliases still fail rather than selecting a value.

## Choose semantics explicitly

A rolling duration is not an equivalent EMA span. Zero/minimum-resolution and all-history
windows have no equivalent span. Choose a slow span with `--lookback-ema-span-minutes`;
the converter initializes the independent slow spans from that explicit choice. To choose
a different canonical policy span, use `--memory-span PATH=MINUTES`. Each authored
scenario/fixed override of an old rolling window needs its own choice at the exact path
reported by the converter. Old rolling-window search bounds require explicit replacement
bounds rather than a unit conversion. Omitted new span bounds remain fixed.

```sh
passivbot tool migrate-config-v9 input.v8.json \
  --lookback-ema-span-minutes 43200.0 --dry-run --diff

passivbot tool migrate-config-v9 input.v8.json output.v9.json \
  --lookback-ema-span-minutes 43200.0 \
  --memory-span bot.unstuck.memory_ema_span_minutes=21600.0 \
  --report migration-report.json
```

For each effective active old realized-loss gate, including authored stored scenarios and
fixed optimizer overrides, add `--acknowledge-realized-loss-gate-memory-change` after
reviewing the new memory policy. The gate allowance is
`max(0, r * (raw_balance + D_gate) - D_gate)`; TWEL does not multiply `r`.

HSL resumes after a qualified finite cooldown and the ordinary admission gates. Any
authored `restart_after_red_policy="never"` requires `--acknowledge-never-removal`,
including dormant policies. An ambiguous enabled null policy must be corrected explicitly.
The converter preserves sparse file inheritance and reports materialized referenced coin
patches; file values precede inline values. Conflicting aliases fail rather than choosing
one by insertion order.

Retired metrics require a specific replacement and a separate formula acknowledgment:

```sh
passivbot tool migrate-config-v9 input.v8.json output.v9.json \
  --lookback-ema-span-minutes 43200.0 \
  --metric-replacement adg_strategy_eq_w=adg_strategy_eq \
  --acknowledge-metric-formula-change --report migration-report.json
```

This example deliberately selects ordinary full-period growth. `_ew` is another distinct,
opt-in formula and requires `backtest.ew_half_life_days`. Neither is an alias for `_w`.
Objectives, limits, visibility lists and reducer mappings all require explicit choices.
DEAP inputs additionally require an authored `optimize.pymoo.shared` policy and
`--acknowledge-search-backend-change`; outer genetic gates and offspring policies are
not equivalent. Existing pymoo fallback settings are materialized in the report.

Released v8 GPU inputs are converted explicitly as well. The report records removal
of the retired automatic lean parallelism, drift monitoring, exact-worker queue,
seed-bootstrap and per-generation validation controls. Supported GPU tuning, batch,
dispatch, checkpoint and scenario-screening settings remain authored. Unknown GPU or
seed-bootstrap fields still fail validation. `optimize.gpu.population_size` keeps the
existing backend-specific migration to `optimize.population_size`, including its
precedence and legacy minimum of eight for GPU search.

Retired GPU controls in `fixed_params` or `fixed_runtime_overrides` require an explicit
new selector or policy; the converter does not delete those selectors. Scenarios cannot
set optimizer policy. GPU and pymoo backend selections remain unchanged, while DEAP
still requires its separate acknowledgment. Start a fresh optimization after conversion;
retired search controls do not imply equivalent search evolution or reusable fitness.

## Output, rollback and validation

Without an output argument the tool writes `input.v9.json`. Existing output, backup and
report files are never overwritten. Dry run writes no config or backup; an explicitly
requested report may still be written. In-place replacement requires `--in-place` and
creates a backup, by default `input.pre-v9-backup.json`; `--backup` selects another path.
Parsing, policy, validation and serialization failures leave the input unchanged.
Already canonical final-v9 input is idempotent; retuning it is an explicit config edit.

Reload the output through the normal loader and backtest the exact intended dates, fees,
market data and scenarios. Freshly evaluate every optimizer seed; old fitness and incompatible
checkpoints cannot be resumed. Public examples now select ordinary metrics explicitly and
are starting points, not translations of a private trading policy.

Live risk acquisition has its own bounded horizon, including supported cooldown, capped by
`live.risk_bootstrap_max_lookback_minutes`. This is not a finite exponential-memory cutoff.
HSL retains compact forward state while pruning factual support; cold restart or a correction
requiring discarded history can need observable approximate reconstruction. Unknown prefix
error can move HSL timing in either direction. The independent unstuck and loss-gate origin
contract is unchanged. See [V9 risk model](v9_risk_model.md) and
[HSL validation](hsl_testing.md). Offline checks do not establish actual-account readiness.
