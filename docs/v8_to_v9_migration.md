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
| `live.hsl_signal_mode` | `bot.global.hsl.scope`: `coin`, `position_side` or `portfolio`; `pside` becomes `position_side`, `unified` becomes `portfolio` |
| `bot.<side>.hsl.red_threshold` | `bot.<side>.hsl.drawdown_ratio_threshold`, with the same fractional value |
| `bot.<side>.unstuck.ema_dist` | `bot.<side>.unstuck.ema_distance_fraction`, with the same signed fractional value |
| `bot.hsl` portfolio policy | `bot.global.hsl` alongside its scope |
| HSL `ema_span_minutes` | `drawdown_ratio_ema_span_minutes` in the same policy |
| HSL `cooldown_minutes_after_red` | `cooldown_after_red_minutes` in the same policy |
| Strategy and unstuck `ema_span_0`, `ema_span_1` | `ema_span_0_minutes`, `ema_span_1_minutes`; trailing-martingale spans live under `entry` |
| Forager/strategy `*_ema_span_1m` | `*_ema_span_minutes`, with the same floating minute horizon |
| Strategy `*_ema_span_1h` | `*_ema_span_1h_candles`, with the same floating hourly-candle count |
| Authored `recv_window_ms`, `fee_conversion_max_age_ms`, `price_tick_min_interval_ms`, startup-budget `elapsed_ms`/`since_previous_ms` | `_milliseconds` suffixes in their final operational groups |
| `live.max_realized_loss_pct` | `bot.global.loss_gate.max_drawdown_ratio` |
| Unstuck `threshold` | `bot.<side>.unstuck.exposure_trigger_ratio` |
| Risk `n_positions` | `bot.<side>.risk.max_positions` |
| `backtest.liquidation_threshold` | `backtest.equity_floor_ratio` |
| `live.auto_gs` | `live.modes.disapproved_position`: `graceful_stop` or `manual` |
| `live.strategy_kind`, `live.hedge_mode` | Same leaves under `bot.global` |
| `live.approved_coins`, `live.ignored_coins` | Separate `bot.<side>.universe` sources |
| Shared minimum age | Copy `live.minimum_coin_age_days` to `bot.long.universe.minimum_coin_age_days` and `bot.short.universe.minimum_coin_age_days`; report consistent all-mode initial-entry admission |
| Shared minimum-cost filter | `bot.global.universe`; simulation minimum-cost override remains nullable |
| Shared market-order policy | `bot.global.execution`; `market_order_near_touch_threshold` becomes `market_order_near_touch_fraction` with the same value |
| `live.warmup_ratio` | `bot.global.warmup.span_multiplier`, with the same value |
| Exposure-enforcer numeric thresholds | `bot.<side>.risk.<enforcer>.threshold_ratio`, with the same value |
| `live.forager_score_hysteresis_pct`, warmup controls | `bot.global.forager.score_hysteresis_margin`, `bot.global.warmup` |
| Scale-one sizing, allowance, fee, fill and execution `_pct` fields | Public `_fraction` names; values are unchanged |
| `fee_pct_fallback`, `fee_pct_sanity_abs_max` | `live.fills.fallback_fee_fraction`, `live.fills.max_abs_fee_fraction` |
| TM `entry.initial_ema_dist`, EMA-anchor `offset` | `entry.initial_ema_distance_fraction`, `offset_fraction` |
| Coin-specific `normal` mode | Retained operational mode plus `bot.<side>.forager.always_include` |
| Flat operational `live` fields | `live.account`, `api`, `candles`, `startup`, `fills`, `execution`, `recovery`, `modes`; long/short mode leaves become `live.modes.long`/`short` |
| Empty forced modes, zero warmup cap, zero warmup concurrency | `null` modes, `bot.global.warmup.max_minutes: null`, and `live.startup.warmup_concurrency: "auto"`; each conversion is reported in every authored layer |
| Stale `backtest.max_warmup_minutes` | Removed and reported; released v8 read the shared cap from `live.max_warmup_minutes`, even when the stale value differed |
| Numeric strings in `backtest.maker_fee_override`/`taker_fee_override` | Reported as the same finite numeric fees; `null` retains exchange fees, and booleans require an explicit numeric choice |
| `forager.volume_drop_pct` | Removed; volume-based ranking remains |
| Optional total-exposure entry gate | Removed; the entry cap always applies |
| `trailing_grid_v7`, Defx and Paradex | Removed; select a retained strategy/venue explicitly |
| Legacy `_w` metrics | Removed; select replacement formulas explicitly |
| DEAP search | Removed; choose an explicit pymoo NSGA-II/NSGA-III operator policy |

V9 uses separate slow EMA spans for each HSL policy, global unstuck realized memory and
global loss-gate realized memory. Local unstuck settings remain under `bot.<side>.unstuck`.
The global span is `bot.global.unstuck.memory_ema_span_minutes`; the gate span is
`bot.global.loss_gate.memory_ema_span_minutes`; HSL uses its own policy's
`memory_ema_span_minutes`. HSL ratio smoothing is a separate faster span.

These unit spelling changes do not rescale values. Price and minute-volatility horizons remain
floating-point minutes, including backtests with aggregated candle intervals; hourly horizons
remain floating-point counts of calendar-hour candles. Native parameter and optimizer identifiers
keep their established spellings.

Scenario overrides and fixed optimizer overrides convert released flat shared fields
to their canonical side-owned groups. For example, `bot.long.n_positions` becomes
`bot.long.risk.max_positions`, whether authored as a dotted selector or nested maps.
Supported side shorthand such as `long.n_positions` follows the same conversion;
`fixed_params` selectors use it as well. Dictionary members, authored values and
canonical strategy paths remain intact. Unknown fields and conflicting aliases still fail
rather than selecting a value.

Released coin-specific `normal` modes retain their operational exemptions and their forced
Forager selection. The converter records generated `always_include` lists, including explicit
empty lists when a fixed or scenario mode replacement must clear inherited selection. A scenario
can run over the ordinary root or optimizer-fixed root: its generated list contains only coins
normal in both effective contexts. Retained operational `normal` flags preserve the remaining
context-specific selections. These are deterministic conversions, not best-effort choices.

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
  --memory-span bot.global.unstuck.memory_ema_span_minutes=21600.0 \
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
Best-effort migration can infer a supported `_ew` formula, but still requires
`--metric-replacement` when no supported exponential counterpart exists. Selecting an ordinary
full-period metric remains a different formula, including when another objective already uses it.
Existing scoring normalization consolidates duplicate objectives created by an explicit choice;
inspect the resulting objective list before starting a new search.
`backtest.visible_metrics: null` is retained: it derives displayed metrics from objectives and
limits. An empty list instead displays all metrics.
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

## Opt-in best-effort migration

Use `--best-effort` to propose a replacement without supplying every semantic choice:

```sh
passivbot tool migrate-config-v9 input.v8.json output.v9.json \
  --best-effort --dry-run --diff --report migration-report.json
```

Remove `--dry-run` after inspecting the proposal. Every inferred choice emits a
`[best-effort]` warning with its path, old value, new value and rationale
(large values are abbreviated in the console and retained in full in the report). The
same decisions have `"best_effort": true` in the report's `changes` list. Warnings
are emitted as choices are made, including before a later validation failure.

| Missing choice or saved-result field | Best-effort policy |
|---|---|
| Retired `_w` metric | Replace `_w` with `_ew` when that metric exists, preserving currency and supported statistic suffixes |
| EW half-life absent or `null`, with an EW metric requested | Set global `backtest.ew_half_life_days` to `180.0` |
| Root rolling-day memory | Multiply `live.pnls_max_lookback_days` by `1440.0` to initialize independent slow EMA spans |
| Scenario or fixed-override rolling-day memory | Convert each authored duration separately, preserving its scope |
| Zero or sub-minute rolling window | Use the minimum `1.0` minute span |
| Missing or all-history root window | Use the canonical `43200.0` minute (30-day) span |
| All-history scenario/fixed window | Use the same canonical 30-day span |
| `restart_after_red_policy="never"` | Remove it and adopt finite cooldown plus ordinary admission gates |
| Retired `we_excess_allowance_mode="bounded"` | Remove selector; retain the bounded allowance contract (`legacy_raw` still fails) |
| Active realized-loss gate | Adopt independent decaying realized memory and report each effective context |
| Saved-result `analysis`, `metrics`, `suite_metrics`, `optimizer_anchor`, `optimizer_evaluation_contract`, and recognized hydration metadata | Remove obsolete result/cache snapshots before inference; discard derived `backtest.coins`/`cache_dir` only for recognized result records |
| DEAP backend | Switch to pymoo, retain authored shared policy, approximate missing operator fields from old controls or canonical pymoo defaults |

Explicit `--lookback-ema-span-minutes`, `--memory-span` and
`--metric-replacement` choices take precedence. Authored EW half-lives and pymoo
shared settings are preserved. For DEAP, candidate-level `mutation_probability`
initializes `pymoo.shared.mutation_prob`, and per-variable `mutation_indpb`
initializes `mutation_prob_per_variable`; the old zero `mutation_indpb` sentinel
uses `auto`. Validation-only passes do not emit duplicate guessed decisions.
Explicit acknowledgment flags retain their explicit-consent provenance when combined
with best-effort mode. Best-effort mode supplies missing consent for formula,
restart, gate-memory and search-algorithm changes; those changes are guesses,
not equivalence claims. Review limit thresholds, re-tune risk settings, run fresh
backtests and start fresh optimizations.

Unknown fields, conflicting aliases, malformed values, ambiguous enabled null
restart policies, retired strategies/venues, unsupported metric counterparts and
rolling-window search bounds still require explicit correction. Normal config
loading and migration without `--best-effort` remain strict. Referenced coin
files are materialized in the output and remain unchanged on disk.

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
`live.startup.risk_bootstrap_max_lookback_minutes`. This is not a finite exponential-memory cutoff.
HSL retains compact forward state while pruning factual support; cold restart or a correction
requiring discarded history can need observable approximate reconstruction. Unknown prefix
error can move HSL timing in either direction. The independent unstuck and loss-gate origin
contract is unchanged. See [V9 risk model](v9_risk_model.md) and
[HSL validation](hsl_testing.md). Offline checks do not establish actual-account readiness.

Released-v8 age behavior changes deliberately: the side gate now applies to every live and
backtest selection mode, including static and forced-normal selection. The converter records
this change for the root and scenario/fixed patches. Both copied values preserve fractional
days; zero disables age admission. It does not stop held-position management or delay warmup
until after the age threshold. Coin-universe overrides remain unsupported.
