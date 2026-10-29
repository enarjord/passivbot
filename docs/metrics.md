# Metrics reference

Passivbot uses **26 core quantities**, exposed through 27 exact selectors because position
utilization has separate long and short scopes. The generated [quantity reference](ai/generated/metric_fields.md)
lists the selectors and units; the [numerical contract](ai/features/backtest_metrics.md) owns
their clocks, formulas and validity rules. There are no automatic USD/BTC, unit, rate or
presentation copies of these quantities.

An optimizer computes its selected objectives, enabled limits and necessary validity facts.
A standalone backtest produces the core report. Every numeric metric supports penalties below
and/or above a bound, independently or together. Score directions below are recommendations;
an explicitly authored goal can override them. Risk-minimizing objectives need suitable growth
or activity conditions.

## Equity, units and sampling

`E` is portfolio strategy equity: initial trading capital, net realized trading cashflows and
marked trading UPNL. Trading fees and rebates enter once. Direct collateral appreciation and
collateral-conversion operations/costs are excluded from strategy performance, while their
effects on actual account balance, position sizing and subsequent trading remain. With USD-only
collateral, published USD equity and strategy equity are identical by construction.

Use exactly **365.25 calendar days per year**. With actual elapsed evaluation days `D`,
`Y = D / 365.25`. Return and drawdown fractions use scale one: `0.05` means 5%. Ratios are
dimensionless; duration units appear in their names. Human percentage formatting includes `%`.

Volatility, Sharpe, Sortino and Omega use the same **complete UTC daily-close simple returns**:
`r_i = E_close_i / E_close_(i-1) - 1`. Partial boundary days do not create returns, and the first
equity level is a reference rather than an artificial zero return. Missing required interior
observations make the selected statistic unavailable.

**Passivbot does not subtract a risk-free rate or benchmark return.** Sortino and Omega have a
fixed zero cutoff. There is no excess-return or configurable return-target machinery.

Drawdown, Ulcer, Pain, peak recovery, CAGR EW and exposure use minute observations. The initial
pre-trading baseline counts once; a factual terminal mark counts once, including an early
partial minute. Flat periods and zero drawdowns remain included. Coarser replay remains usable
for supported quantities, with its cadence recorded; minute-only metrics are unavailable and
cannot be selected for optimization at that cadence. No interpolation creates supposed minute facts.

## Complete core list

Let `mu = mean(r)`, `s = sample_std(r, ddof=1)`, and
`d_t = 1 - E_t / running_max(E)`. S denotes a useful score and L a primarily useful limit.
Every S metric is also a limit.

| Metric selector | Definition/formula | Usual role |
|---|---|---|
| `cagr_fraction_strategy_equity` | `(E_end / E_start)^(1/Y) - 1` | S maximize |
| `volatility_annual_fraction_strategy_equity` | `s * sqrt(365.25)` | L |
| `sharpe_ratio_strategy_equity` | `mu / s * sqrt(365.25)` | S maximize |
| `sortino_ratio_strategy_equity` | `mu / sqrt(mean(min(r, 0)^2)) * sqrt(365.25)` | S maximize |
| `omega_ratio_strategy_equity` | `sum(max(r, 0)) / sum(max(-r, 0))`; not annualized | S maximize |
| `drawdown_max_fraction_strategy_equity` | `max(d)` | L |
| `ulcer_index_fraction_strategy_equity` | `sqrt(mean(d^2))` | S minimize |
| `pain_index_fraction_strategy_equity` | `mean(d)` | S minimize |
| `cagr_fraction_strategy_equity_ew` | Annualized calendar-weighted log growth, below | S maximize |
| `backtest_completion_fraction` | Covered duration / intended duration, clipped to `[0,1]` | L |
| `fills_gap_time_weighted_mean_hours` | `sum(g_i^2) / sum(g_i)`, for portfolio no-fill gap hours `g_i` including evaluation boundaries | S minimize |
| `fills_gap_max_days` | Longest gap between evaluation boundaries and consecutive portfolio fills, in days | L |
| `peak_recovery_max_days_strategy_equity` | Longest underwater episode until regaining its previous high, in days | L |
| `position_held_max_days` | Longest coin/position-side open-to-flat episode, including open tails | L |
| `gross_turnover_ratio_per_day` | `sum(abs(fill_notional_i) / pre_fill_balance_i) / D` | S maximize when desired |
| `position_held_time_weighted_mean_hours` | `sum(h_i^2) / sum(h_i)`, for holding episode hours `h_i` | S minimize |
| `position_exposure_utilization_rms_ratio_long` / `_short` | Budget-relative position RMS, below | S minimize |
| `hsl_panic_episode_loss_max_fraction` | Maximum HSL execution-episode loss / balance before its first HSL fill | L |
| `hsl_panic_episode_loss_mean_fraction` | Arithmetic mean of those episode loss fractions | L |
| `hsl_execution_episode_count` | Actual execution episodes with at least one HSL fill, including zero-loss episodes | L |
| `hsl_triggers_per_year` | Observed trigger count / `Y` | L |
| `gross_exposure_rms_ratio_strategy_equity` | `sqrt(mean(g_t^2))`, with gross marked exposure `g_t` below | S minimize |
| `gross_exposure_max_ratio_strategy_equity` | `max(g_t)` | L |
| `hsl_red_time_fraction` | Union reported RED seconds / observed HSL seconds | L |
| `loss_profit_ratio` | Negative net realized cashflow magnitude / positive net realized cashflow | S minimize |
| `short_net_profit_contribution_fraction` | Net short strategy profit / positive combined net strategy profit | L |

Sortino includes **all daily returns** in its downside denominator: nonnegative days contribute
zero, and division is by the total number of daily returns. It does not use clipped-return
standard deviation or a negative-day-only average. Sharpe uses arithmetic mean returns, not CAGR.
Volatility and Sharpe need at least two daily returns for sample dispersion.

CAGR uses factual endpoints with no three-day or seven-day smoothing. Growth and daily return
statistics require valid positive equity support; a later recovery does not repair an intermediate
nonpositive equity observation. Drawdown, Pain and Ulcer are not annualized.

Requested peak recovery includes `metric_metadata.peak_recovery_tail_censored`: true for an
unfinished underwater tail, false when recovered or flat, and null when support is invalid.

## Exponential recency weighting

The single EW quantity is CAGR EW. `backtest.ew_half_life_days` resolves to **180 days** by
default, including an omitted or null input; an explicit finite positive value sets one common
campaign half-life. Scenario overrides are rejected.

For minute intervals, define `z_k = log(E_k / E_(k-1))`, interval days `delta_k`, and
`w_k = 2^(-age_of_interval_endpoint_days / half_life_days)`:

```text
CAGR_EW = expm1(365.25 * sum(w_k * z_k) / sum(w_k * delta_k))
```

Flat intervals remain in the denominator. As half-life tends to infinity, the result approaches
ordinary CAGR on the same window. This is a declared Passivbot definition of recency-weighted
growth. There are no EW Sharpe/Sortino/Omega or currency variants.

## Effective backtest period

The declared post-warmup boundary starts evaluation even when there is no trading capacity or
no fills. Actual observation clocks determine elapsed time and coverage. Completion keeps the
intended denominator when replay terminates early. A nonpositive post-warmup window is an explicit
error for a standalone backtest and an invalid candidate in optimization.

`metric_metadata` records actual/intended clocks, observation cadence, definition version and
relevant sample counts. Uncomputed group counts are omitted. `effective_start_date` and
`effective_end_date` are actual UTC ISO boundaries, outside numeric aggregation.
Liquidation is separate support metadata: final-minute liquidation can have completion `1`.

Fill gaps include the leading and trailing evaluation intervals; no fills gives one gap equal
to the full duration. `fills_gap_time_weighted_mean_hours` computes
`sum(gap_hours^2) / sum(gap_hours)` over the same boundary-inclusive gaps. With no
fills it equals the actual evaluated duration in hours, including early termination.
Duplicate fill timestamps add no elapsed gap and do not change the weighted mean.
Recovery starts at the last high before a decline and ends at its first recovery; a plateau at the high is recovered.
Include unfinished recovery and holding episodes through evaluation end. Position additions
and partial closes do not reset holding starts. With no positive-duration position episode,
both holding metrics are zero.

## Turnover and exposure

Turnover counts every entry and exit at execution-price notional, including instrument
multiplier. Its denominator is each fill's raw, unrounded USD-valued account balance immediately
before execution, excluding UPNL. Include inactive and partial days, with no round-trip halving.
At unchanged balance 1000, volume 500 in one day gives `0.5`; volume 1200 over two days gives
`0.6`, including a day without trades.

Position utilization retains nominal runtime WEL/TWEL budgets, excluding excess allowance:

```text
e_p,t = abs(entry-price position notional_p,t) / raw_account_balance_t
u_side_squared_t = sum_over_positions(e_p,t^2 / WEL_p,t) / TWEL_side,t
utilization_side_rms = sqrt(mean(u_side_squared_t))

g_t = sum_over_positions(abs(marked position notional_p,t)) / E_t
```

Flat capacity remains in the minute population. Positive load with invalid balance/budget is
unavailable. Scaling exposure and both budgets proportionally leaves utilization unchanged;
gross exposure separately measures absolute strategy load. Long and short gross notionals add.

## HSL execution episodes

An execution episode belongs to one actual controller scope and contains at least one HSL fill.
Flattening or retirement of its panic intent ends it. A later execution starts a new episode;
RED cooldown after flattening does not create another loss episode. Include observed unfinished
episode loss once at evaluation end.

```text
net_fill_pnl_i = realized_pnl_i + signed_fee_i
episode_loss_e = sum(max(-net_fill_pnl_i, 0))
episode_fraction_e = episode_loss_e / raw_balance_before_first_HSL_fill_e
```

Costs are negative signed fees and rebates positive. Profitable fills do not offset losses.
Zero-loss executed episodes count as zero in the mean; triggers without executed HSL fills do
not enter that mean. With no execution episodes, both loss aggregates are zero with count
metadata. A nonpositive/invalid balance anchor makes the selected aggregate unavailable.

Use an optimizer limit to require an executed stop during a selected evaluation window:

```json
{"metric": "hsl_execution_episode_count", "penalize_if": "less_than", "value": 1}
```

This counts execution episodes, including zero-loss and unfinished episodes with an HSL fill.
Multiple fills in one episode count once. A trigger without an HSL fill does not satisfy it.
Panic-tagged fills after the controller retires HSL intent do not count.
The count is not annualized. For suites, select the intended base scenario explicitly if the
requirement should not apply to every scenario through an aggregate reducer.

Reported RED includes enforced restart cooldown and can outlast immediate panic permission.
RED time is the union across enabled controllers, excluding warmup. In coin/position-side mode
it does not mean the whole bot was inactive. Missing coverage is not known GREEN time.
Disabled HSL remains distinguishable from enabled HSL that was never triggered.

## Profit accounting

For realized loss/profit, split each actual fill's `x_i = realized_pnl_i + signed_fee_i`
into positive and negative net cashflow. Include entry fees, HSL fills and other modeled trading
cashflows once. Positive cashflow 100 and negative magnitude 70 give `0.7`, leaving net realized
profit 30. Above one means net realized loss. UPNL is excluded. Actual event grouping is part of
this definition; no inverse profit-factor or closed-trade variant is maintained.

For short contribution:

```text
P_side = net_realized_trading_cashflows_side + UPNL_side_end - UPNL_side_start
short_contribution = P_short / (P_long + P_short)
```

Use position side, not buy/sell direction. Include trading costs and rebates; exclude collateral
appreciation. Require positive combined net profit. Long/short profits 100/0, 0/100 and 100/100
give `0`, `1` and `0.5`. A losing short gives a negative share; a losing long gives a share above
one. Do not clamp or take absolute profits. Bounds `0.1` and `0.9` require at least 10% of positive
combined net profit from each side. Nonpositive combined profit is unavailable and cannot satisfy
an enabled contribution limit.

## Results, scores and limits

Raw analysis uses finite values or `null`, with a sparse `metric_status` map distinguishing
positive/negative infinity, undefined and invalid inputs. It uses standard JSON, without
`NaN`/`Infinity` tokens, epsilon floors, arbitrary ratio caps or friendly `0/0` fallbacks.
True nonzero/zero ratios can be infinite; zero/zero is undefined. A standard report can contain
unavailable metrics. Only a selected objective or enabled limit requiring one rejects a candidate.

Limits compare raw extended-real values. Positive infinity passes a finite lower bound and fails
a finite upper bound, with the converse for negative infinity. Unavailable values cannot satisfy
limits. Both penalty directions remain available together, with existing equality semantics.

Authored selectors use exact base names; outer reductions use reducer fields/maps. Reduce raw
values across every required source before scoring; missing scenarios/exchanges are never dropped.
Compute only requested reductions. Mixed positive/negative infinities make a mean undefined;
any infinite input makes standard deviation unavailable.

For minimization engines, a reduced score uses `(2/pi) * atan(goal_sign * raw_value)`, with sign
`+1` for minimization and `-1` for maximization. Genuine infinities map to finite endpoints.
This preserves mathematical ordering/Pareto dominance but changes crowding geometry; it is
versioned in the evaluation contract. Feasibility takes precedence. Raw values remain in results
and limits; finite invalid-candidate transport placeholders are not real metric values.

Standalone `analysis.json` and structured `config.json` results retain definition metadata and
statuses. Saved fitness/checkpoints require the current definition and implementation identity.
Historical results keep their original definitions; changed metrics and thresholds need explicit
review/re-evaluation. See [released-v8 migration](v8_to_v9_migration.md).

The normal equity CSV keeps factual native observations and float64 precision. The
`balance_sample_divider` setting affects plots only. Fills remain exported; no dedicated
post-backtest analysis tool or retired-metric compatibility subsystem is added.
