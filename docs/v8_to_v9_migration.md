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
| `live.hsl_signal_mode` | `bot.portfolio.hsl.scope`: `coin`, `position_side` or `portfolio`; `pside` becomes `position_side`, `unified` becomes `portfolio` |
| `bot.<side>.hsl.red_threshold` | `bot.<side>.hsl.drawdown_threshold_fraction`, with the same fractional value |
| `bot.<side>.unstuck.ema_dist` | `bot.<side>.unstuck.ema_distance_fraction`, with the same signed fractional value |
| `bot.hsl` portfolio policy | `bot.portfolio.hsl` alongside its scope |
| HSL `ema_span_minutes` | `drawdown_fraction_ema_span_minutes` in the same policy |
| HSL `cooldown_minutes_after_red` | `cooldown_after_red_minutes` in the same policy |
| Strategy and unstuck `ema_span_0`, `ema_span_1` | `ema_span_0_minutes`, `ema_span_1_minutes`; trailing-martingale spans live under `entry` |
| Pymoo shared `crossover_eta`, `mutation_eta` | `crossover_distribution_index`, `mutation_distribution_index` |
| Pymoo shared `crossover_prob_var`, `mutation_prob`, `mutation_prob_per_variable` | `crossover_probability_per_variable`, `mutation_probability`, `mutation_probability_per_variable`; same probability gates and auto formulas |
| TM `entry.double_down_factor` | `entry.reentry_qty_multiplier`, scaling the absolute current position quantity for an additional entry |
| TM `entry.threshold_we_weight`, `entry.retracement_we_weight`, `close.threshold_we_weight` | Same role names ending in `_exposure_ratio_coefficient`; values and formulas are unchanged |
| EMA-anchor `entry_double_down_factor`, `offset_psize_weight` | `entry_qty_exposure_coefficient`, `inventory_offset_exposure_coefficient`; same-side sizing and signed inventory shifts retain their original formulas |
| Strategy `*_volatility_1m_weight` / `*_volatility_1h_weight` | Corresponding `*_log_range_1m_coefficient` / `*_log_range_1h_coefficient`; the indicator remains EMA of `ln(high / low)` |
| Forager `score_weights.volatility` | `score_weights.log_range`, preserving relative weights and their normalization |
| Forager/strategy log-range span names containing `volatility` | Corresponding names containing `log_range`, retaining the same minute or hourly-candle unit |
| Forager/strategy `*_ema_span_1m` | `*_ema_span_minutes`, with the same floating minute horizon |
| Strategy `*_ema_span_1h` | `*_ema_span_1h_candles`, with the same floating hourly-candle count |
| Authored `recv_window_ms`, `fee_conversion_max_age_ms`, `price_tick_min_interval_ms`, startup-budget `elapsed_ms`/`since_previous_ms` | `_milliseconds` suffixes in their final operational groups |
| `live.max_realized_loss_pct` | `bot.portfolio.loss_gate.max_drawdown_fraction` |
| `bot.<side>.risk.total_wallet_exposure_limit` | `bot.<side>.risk.total_wallet_exposure_limit_ratio`, with the same notional-to-balance value |
| `coin_overrides.<coin>.bot.<side>.wallet_exposure_limit` | `coin_overrides.<coin>.bot.<side>.wallet_exposure_limit_ratio`, with the same per-coin value |
| Unstuck `threshold` | `bot.<side>.unstuck.exposure_trigger_ratio` |
| Risk `n_positions` | `bot.<side>.risk.max_positions` |
| `backtest.btc_collateral_cap`, `backtest.btc_collateral_ltv_cap` | `backtest.btc_collateral_target_ratio`, `backtest.max_usd_debt_to_wallet_value_ratio`; the same target and USD-debt/total-wallet-value limit, excluding unrealized trading PnL |
| `backtest.maker_fee_override`, `backtest.taker_fee_override` | `backtest.maker_fee_override_fraction`, `backtest.taker_fee_override_fraction`; signed values and `null` are unchanged |
| `backtest.liquidation_threshold` | `backtest.equity_floor_fraction` |
| `live.auto_gs` | `live.modes.disapproved_position`: `graceful_stop` or `manual` |
| `live.strategy_kind`, `live.hedge_mode` | Same leaves under `bot.portfolio` |
| `live.approved_coins` | Common inline approvals become `bot.global.universe.approved_coins`; directional approvals and file bindings remain explicit `bot.<side>.universe.approved_coins` sources |
| `live.ignored_coins` | Separate `bot.<side>.universe.ignored_coins` sources |
| Shared minimum age | Copy `live.minimum_coin_age_days` to `bot.long.universe.minimum_coin_age_days` and `bot.short.universe.minimum_coin_age_days`; report consistent all-mode initial-entry admission |
| Shared minimum-cost filter | Copy to both `bot.<side>.universe.filter_by_min_effective_cost` leaves; simulation override remains nullable |
| Shared market-order policy | `bot.portfolio.execution`; `market_order_near_touch_threshold` becomes `market_order_near_touch_fraction` with the same value |
| `live.warmup_ratio` | `bot.portfolio.warmup.activation_span_fraction`, with the same value |
| Exposure-enforcer numeric thresholds | `bot.<side>.risk.<enforcer>.threshold_ratio`, with the same value |
| `live.forager_score_hysteresis_pct`, warmup controls | `bot.global.forager.score_hysteresis_margin`, `bot.portfolio.warmup` |
| Scale-one sizing, allowance, fee, fill and execution `_pct` fields | Public `_fraction` names; values are unchanged |
| `fee_pct_fallback`, `fee_pct_sanity_abs_max` | `live.fills.fallback_fee_fraction`, `live.fills.max_abs_fee_fraction` |
| TM `entry.initial_ema_dist`, EMA-anchor `offset` | `entry.initial_ema_distance_fraction`, `offset_fraction` |
| Coin-specific `normal` mode | Retained operational mode plus `bot.<side>.forager.always_include` |
| Flat operational `live` fields | `live.account`, `api`, `candles`, `startup`, `fills`, `execution`, `recovery`, `modes`; long/short mode leaves become `live.modes.long`/`short` |
| Empty forced modes, zero warmup cap, zero warmup concurrency | `null` modes, `bot.portfolio.warmup.max_minutes: null`, and `live.startup.warmup_concurrency: "auto"`; each conversion is reported in every authored layer |
| Stale `backtest.max_warmup_minutes` | Removed and reported; released v8 read the shared cap from `live.max_warmup_minutes`, even when the stale value differed |
| Numeric strings in `backtest.maker_fee_override`/`taker_fee_override` | Reported as the same finite numeric fees; `null` retains exchange fees, and booleans require an explicit numeric choice |
| `forager.volume_drop_pct` | Removed; volume-based ranking remains |
| Optional total-exposure entry gate | Removed; the entry cap always applies |
| `trailing_grid_v7`, Defx and Paradex | Removed; select a retained strategy/venue explicitly |
| Legacy `_w` metrics | Removed; select replacement formulas explicitly |
| DEAP search | Removed; choose an explicit pymoo NSGA-II/NSGA-III operator policy |

V9 uses `bot.global` for inherited directional defaults and `bot.portfolio` for
account-wide policy. The converter retains frozen released-v8 side parameters and search
ranges as explicit side policy, including equal-valued pins; it does not replace old asymmetric
profiles with the new common starter. It explicitly writes `enabled: true` for both sides and
reports the choice, preserving search reachability when a side starts with zero capacity.
Zero capacity still prevents entries; it no longer implicitly sets held-position management
to Manual. New v9 configs instead default to disabled sides. No intermediate dev-v9 migration
is added.

V9 uses separate slow EMA spans for each HSL policy, account-wide unstuck realized memory and
account-wide loss-gate realized memory. Local unstuck settings remain under `bot.<side>.unstuck`.

Released flat unstuck fields and optimizer IDs are translated by a frozen versioned map,
including root optimizer bounds, fixed selectors, referenced coin files and inline overrides.
Current native identifiers use `_fraction`, `_ratio` and `_minutes`. Conflicting released
spellings are rejected before defaults are applied, and conversions report final public paths.

The shared span is `bot.portfolio.unstuck.memory_ema_span_minutes`; the gate span is
`bot.portfolio.loss_gate.memory_ema_span_minutes`; HSL uses its own policy's
`memory_ema_span_minutes`. HSL ratio smoothing is a separate faster span.

These unit spelling changes do not rescale values. Price and minute-log-range horizons remain
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
  --memory-span bot.portfolio.unstuck.memory_ema_span_minutes=21600.0 \
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

Core metric definition version 1 changes observation populations, clocks and formulas.
Strict migration requires `--acknowledge-metric-formula-change` for every selected
released metric, including retained names such as `loss_profit_ratio`. Each newly
retired selector also requires an explicit `--metric-replacement OLD=NEW` choice:

```sh
passivbot tool migrate-config-v9 input.v8.json output.v9.json \
  --lookback-ema-span-minutes 43200.0 \
  --metric-replacement adg_strategy_eq=cagr_fraction_strategy_equity \
  --metric-replacement adg_strategy_eq_w=cagr_fraction_strategy_equity_ew \
  --metric-replacement strategy_eq_underwater_pct_mean=pain_index_fraction_strategy_equity \
  --acknowledge-metric-formula-change --report migration-report.json
```

Repeat the replacement flag for every retired selector reported in the input's
objectives, limits or reducer mappings. These examples are deliberate formula choices;
CAGR, CAGR EW and minute-sampled Pain Index do not preserve the old formulas or their
thresholds. Turnover, HSL episode loss and other revised quantities also require fresh
thresholds and backtests. See the [current metric definitions](metrics.md).

Strict conversion requires these explicit choices. Best-effort conversion instead uses
the bounded inference/removal policy below; it does not restore retired runtime aliases.
An explicitly chosen replacement must
be a current BASE selector; `_usd`, `_btc` and outer-statistic aliases are unsupported.
A verified released outer suffix such as `_max` moves into the existing `reducer` field.
Conflicting reducer choices fail, and identical objectives created by formula choices
are consolidated with their old/new lists and counts recorded in the report.

Only `cagr_fraction_strategy_equity_ew` retains exponential recency weighting. Its
native default half-life is 180 days; omitted or `null` values resolve to that default.
Explicit half-lives must be finite and positive, including fractional days below one.
One resolved value applies across the campaign; scenarios cannot override it.

Annualized metrics use 365.25 days per year. Volatility, Sharpe, Sortino and Omega use
complete UTC daily returns, with no risk-free-return subtraction. Drawdown, Ulcer,
Pain, recovery, CAGR EW and exposure need minute observations. Coarser replay preserves
its actual cadence: ordinary reports mark unavailable quantities explicitly, and an
optimizer selecting one of those minute quantities fails preflight. Daily quantities
require a cadence supplying the complete UTC-close grid; no interpolation creates
missing observations.

The terminal-only `backtest.visible_metrics` setting is removed during explicit v8
migration, with its removal recorded in the report. Standalone backtests display their
complete core analysis. Current v9 configurations must omit this retired field.

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

Use `--best-effort` to infer close metric replacements and remove verified retired
objectives and limits that have no meaningful compact-core counterpart. Review the warnings
and report: removing a limit removes that optimization constraint.

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
| Strategy-equity ADG, including its rebased alias | Infer unsmoothed annual CAGR; infer CAGR EW for the legacy weighted ADG form |
| `volume_pct_per_day_avg`, including its weighted form | Infer all-day, pre-fill-balance-normalized gross turnover; report changed sampling and any removed recency weighting |
| `strategy_eq_underwater_pct_mean` | Infer minute-sampled Pain Index instead of mean daily worst drawdown |
| `strategy_eq_recovery_days_max` or `peak_recovery_days_strategy_eq` | Infer `peak_recovery_max_days_strategy_equity`, preserving any outer reducer and limit direction/value; report the formula change |
| `position_held_hours_max` | Infer `position_held_max_days`; divide scalar and range bounds by 24 |
| `long_short_profit_ratio` or `pnl_ratio_long_short` | Infer short net profit contribution with the complement transformations below |
| Other verified retired metric selectors, including median growth, unsupported weighted/currency/per-side metrics and HSL restart counts | Remove the entire objective/limit and its stale named reducer rules; record the full original entry and path with `new: null` |
| Retained metric selector crossing core definition version 1 | Report the changed definition and require fresh thresholds/results |
| EW half-life absent or `null`, with CAGR EW explicitly selected | Resolve and report the native 180-day default |
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
`--metric-replacement` choices take precedence over inference and removal. Explicit metric
choices retain their authored bounds and goals; supply values appropriate to the chosen formula.
Authored EW half-lives and pymoo
shared settings are preserved. For DEAP, candidate-level `mutation_probability`
initializes `pymoo.shared.mutation_probability`, and per-variable `mutation_indpb`
initializes `mutation_probability_per_variable`; the old zero `mutation_indpb` sentinel
uses `auto`. Validation-only passes do not emit duplicate guessed decisions.
Explicit acknowledgment flags retain their explicit-consent provenance when combined
with best-effort mode. Best-effort mode supplies missing acknowledgment for retained
metric definitions and explicit formula replacements, restart, gate-memory and search-algorithm
changes. Inferred metric choices remain identified as guesses in warnings and the report,
even with `--acknowledge-metric-formula-change`. The old recovery maximum measures waiting
for a strictly higher equity observation; the current metric measures running-peak recovery
episodes, ending when equity regains the peak. Peak ties recover and flat equity has zero
recovery under the current formula. These decisions make no equivalence claim.
Review limit thresholds, re-tune risk settings, run fresh backtests and start fresh optimizations.

Inferred daily growth limits use the nominal annual bound
`expm1(365.25 * log1p(daily_bound))`. This changes units; it does not restore the old
smoothed or weighted growth formula, and nonlinear annualization does not commute with
scenario averaging. Standard-deviation growth limits and bounds with no finite nominal
conversion are removed with warnings.
The canonical `penalize_if` mode selects the active scalar or range. Inactive bound fields
are discarded with a warning during inferred numerical conversion, so an unused range
cannot redirect or invalidate conversion of the actual scalar constraint.

For long profit share `L`, the inferred short contribution is nominally `1 - L`.
Scoring goals and scalar inequality directions reverse, bounds become `1 - bound`,
and ranges become `[1 - high, 1 - low]`. Minimum/maximum reducers swap; mean/median
retain their names. Standard deviation retains its original goal, bound and direction because
`std(1 - L) = std(L)`. These choices resolve the original reducer aliases, active legacy suite
policy and named-scenario selection before rewriting. Automatic limit directions that cannot
be safely reversed are removed. The new quantity also includes fees and open-position PnL,
requires positive combined net profit, and does not preserve the old zero-total fallback.

If all objectives are removed, the output retains explicit `scoring: []` and reports that
decision; it does not insert new objectives. Choose an objective before optimizing. An inferred
limit-only quantity without an authored scoring goal is likewise removed from scoring with a
warning. Legacy outer-statistic shorthand without a declared scoring goal is removed; use
an objective object with an explicit goal to retain it. Unknown selector spellings and
malformed policy still fail instead of being discarded.

Unknown fields, conflicting aliases, malformed values, ambiguous enabled null
restart policies, retired strategies/venues, unsupported selections outside objective/limit/reducer
collections and
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
