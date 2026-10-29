# Core backtest metrics, definition version 1

This contract owns the arithmetic of `passivbot-rust/src/core_metrics.rs` and any
producer declaring `CORE_METRIC_DEFINITION_VERSION = 1`. An unversioned historical
analysis does not acquire these definitions by retaining the same field names.
Rust owns formulas and accumulators; Python owns requests, orchestration and results.

## Population, clocks and units

- Use exactly 365.25 calendar days per year. Elapsed days are actual timestamp
  differences divided by 86400000; years are elapsed days divided by 365.25.
- Establish an initial strategy-equity boundary before evaluated trading, after
  declared warmup. Inactivity and zero trading capacity do not postpone this boundary.
  A close-valued observation uses its completed-close clock, not its candle-open label.
- Strategy equity is initial trading capital plus net realized trading cashflows and
  marked trading UPNL. Direct collateral revaluation and collateral conversion operations,
  including their conversion costs, are outside this strategy-P&L basis. They remain
  factual account changes affecting balances, sizing and subsequent trading. Ordinary
  executed strategy-fill fees and every other modeled trading cashflow enter once.
- Preserve factual economic equity at termination. Display floors do not alter metric
  inputs or conceal loss. Liquidation detection, risk and order behavior retain their owners.
- Daily statistics use complete, contiguous UTC-midnight close pairs admitted within
  evaluation. A pair supplies one simple return `r = E_close / E_previous_close - 1`.
  Exclude partial boundary days, reject missing interior support, and never insert a
  fabricated initial zero return. Record admitted clocks and return count.
- Minute path statistics use the initial baseline and actual evaluated minute marks,
  including flat periods and zero drawdowns. Count a terminal economic mark once;
  an early terminal event does not invent another elapsed minute. Arithmetic observation
  means give the retained partial-terminal mark one observation, without fractional-time
  reweighting. Coarser observations must not be interpolated into supposed minute facts.
  Declared coarser replay remains supported for factual growth/daily/activity/HSL quantities;
  minute-only path, EW and exposure metrics are unavailable and rejected as optimizer requests.
- Returns, volatility, drawdown and fractions use scale one; ratios are dimensionless.
  Durations state days or hours. Human percentage formatting includes `%`.
- No risk-free or benchmark subtraction, minimum-return input, terminal CAGR smoothing,
  or configurable Sortino/Omega threshold belongs to this contract.

Let `E` be strategy equity, `D` elapsed evaluated days, `Y = D / 365.25`,
`mu = mean(r)`, `s = sample_std(r, ddof=1)`, and
`d_t = 1 - E_t / max(E_0, ..., E_t)`. Zero drawdowns remain in both drawdown means.

## Canonical quantities

S denotes a suitable score as well as a limit; L denotes a primarily useful limit.
Directions are recommendations. Every numeric selector accepts lower and upper
penalties independently or together. Risk-minimizing scores need appropriate growth
or activity conditions. There are 26 quantities and 27 selectors because position
utilization retains explicit long and short scopes.

| Quantity | Canonical selector | Formula | Role |
|---|---|---|---|
| CAGR | `cagr_fraction_strategy_equity` | `(E_end / E_start)^(1/Y) - 1` | S maximize |
| Annualized volatility | `volatility_annual_fraction_strategy_equity` | `s * sqrt(365.25)` | L |
| Annualized Sharpe | `sharpe_ratio_strategy_equity` | `mu / s * sqrt(365.25)` | S maximize |
| Annualized Sortino | `sortino_ratio_strategy_equity` | `mu / sqrt(mean(min(r, 0)^2)) * sqrt(365.25)` | S maximize |
| Omega | `omega_ratio_strategy_equity` | `sum(max(r, 0)) / sum(max(-r, 0))` | S maximize |
| Maximum drawdown | `drawdown_max_fraction_strategy_equity` | `max(d)` | L |
| Ulcer Index | `ulcer_index_fraction_strategy_equity` | `sqrt(mean(d^2))` | S minimize |
| Pain Index | `pain_index_fraction_strategy_equity` | `mean(d)` | S minimize |
| CAGR EW | `cagr_fraction_strategy_equity_ew` | Calendar-weighted log growth defined below | S maximize |
| Completion | `backtest_completion_fraction` | `clip(covered_seconds / intended_seconds, 0, 1)` | L |
| Time-weighted portfolio fill gap | `fills_gap_time_weighted_mean_hours` | `sum(g_i^2) / sum(g_i)`, for gap hours `g_i` in `[start, sorted unique fills, end]` | S minimize |
| Maximum portfolio fill gap | `fills_gap_max_days` | Longest gap in `[start, sorted fills, end]`, divided by 86400 seconds | L |
| Maximum peak recovery | `peak_recovery_max_days_strategy_equity` | Longest last-high-to-recovery episode, in days | L |
| Maximum position holding | `position_held_max_days` | `max(h_i) / 24`, for episode hours `h_i` | L |
| Gross turnover/day | `gross_turnover_ratio_per_day` | `sum(abs(fill_notional_i) / B_i_pre) / D` | S maximize when explicitly desired |
| Time-weighted position holding | `position_held_time_weighted_mean_hours` | `sum(h_i^2) / sum(h_i)` | S minimize |
| Position utilization RMS | `position_exposure_utilization_rms_ratio_long`, `position_exposure_utilization_rms_ratio_short` | Budget-relative per-side RMS defined below | S minimize |
| Maximum HSL episode loss | `hsl_panic_episode_loss_max_fraction` | `max(L_e)` | L |
| Mean HSL episode loss | `hsl_panic_episode_loss_mean_fraction` | Arithmetic `mean(L_e)` | L |
| HSL execution count | `hsl_execution_episode_count` | Actual execution episodes containing at least one HSL-attributed fill, including zero-loss episodes | L |
| HSL trigger rate | `hsl_triggers_per_year` | Actual observed trigger count divided by `Y` | L |
| Gross strategy exposure RMS | `gross_exposure_rms_ratio_strategy_equity` | `sqrt(mean(g_t^2))` | S minimize |
| Maximum gross strategy exposure | `gross_exposure_max_ratio_strategy_equity` | `max(g_t)` | L |
| HSL RED time | `hsl_red_time_fraction` | Union RED seconds divided by observed HSL seconds | L |
| Realized loss/profit | `loss_profit_ratio` | `sum(max(-x_i, 0)) / sum(max(x_i, 0))` | S minimize |
| Short net profit contribution | `short_net_profit_contribution_fraction` | `P_short / (P_long + P_short)`, for positive combined P&L | L |

### Growth and daily returns

CAGR uses factual endpoints and the full actual window. Use log/expm1 evaluation for
numerical stability. Every admitted strategy-equity observation must be positive for
log growth; recovery after an intermediate nonpositive observation does not repair it.

For CAGR EW, use minute log returns `z_k = log(E_k / E_(k-1))`, interval days `delta_k`,
and `w_k = 2^(-age_of_interval_endpoint_days / half_life_days)`:

```text
CAGR_EW = expm1(365.25 * sum(w_k * z_k) / sum(w_k * delta_k))
```

One fixed positive half-life applies across a campaign. The source-owned default is 180 days; omitted/null configuration resolves to that value. Decaying prior numerator and
denominator before adding the newest interval is equivalent and needs constant storage.
As half-life tends to infinity, this tends to ordinary CAGR on the same window.

Sharpe uses arithmetic mean daily returns. Sortino's downside denominator is an RMS
over all admitted days: nonnegative returns contribute zero, and the divisor is the
total daily-return count. It is not clipped-return standard deviation or a conditional
negative-day average. Omega uses the same daily population and no annualization factor.
Square-root annualization is a reporting convention, not a correction for serial dependence.

### Durations and turnover

Recovery begins at the last high before a decline and ends at the first return to
that high or above. A plateau at the high is recovered; a plateau below it remains
underwater. Include an unfinished episode through actual evaluation end, marked censored.
With no underwater episode the maximum is zero. Requested recovery includes
`metric_metadata.peak_recovery_tail_censored`: true for an unfinished tail, false
when recovered/flat, and null when support is invalid. Omit it when recovery is unrequested.

Position episodes are keyed by coin and position side. A first nonzero position opens
an episode; additions and partial closes retain its start; flattening ends it. Include
open tails through actual end. Holding duration is not notional-weighted. With no
positive-duration holding episode, both holding aggregates are zero with count metadata.

Fill gaps include start and end boundaries. Duplicate timestamps add zero elapsed time.
A no-fill run has a maximum gap equal to its full evaluated duration. The time-weighted
mean is `sum(g_i^2) / sum(g_i)` in hours, with all portfolio gaps including both
boundaries. Its denominator is the full actual evaluated duration; with no fills
it equals that duration in hours. Same-clock fills contribute zero and do not
change the result. Early termination uses actual end, excluding the unobserved tail.

Turnover counts every entry and exit using execution-price absolute notional with the
instrument's canonical conversion, including contract multiplier. `B_i_pre` is raw,
unrounded USD-valued account balance immediately before that fill, excluding UPNL and
using causal collateral valuation. Include inactive and partial days. Do not halve
round-trip volume. Invalid/nonpositive counted balances make the metric unavailable.
At unchanged balance 1000, volume 500 in one day yields 0.5; volume 1200 over two days
yields 0.6 even if all fills occur on the first day.

### Exposure

```text
e_p,t = abs(entry-price position notional_p,t) / raw_account_balance_t
u_side,t_squared = sum_over_positions(e_p,t^2 / WEL_p,t) / TWEL_side,t
position_utilization_side = sqrt(mean(u_side,t_squared))

g_t = sum_over_positions(abs(marked position notional_p,t)) / E_t
```

WEL/TWEL are contemporaneous nominal runtime budgets, excluding excess allowance.
Include flat periods and unused capacity. A flat side contributes zero; positive load
with invalid balance or budget is unavailable. Proportional scaling of exposure and
both budgets leaves utilization unchanged. Gross exposure instead measures absolute
strategy load, uses marked prices, adds long and short notionals, and requires positive
strategy equity. No implicit portfolio sum/max of side utilization is accepted.

### HSL

An execution episode belongs to one actual controller scope and includes at least one
HSL-attributed fill. Flattening or retirement of panic intent ends it. Do not split an
episode by candle or pool unrelated controllers. RED cooldown after flattening does not
create a new loss episode. An unfinished episode contributes observed loss once at end.

```text
net_fill_pnl_i = realized_pnl_i + signed_fee_i
L_e = sum_over_episode(max(-net_fill_pnl_i, 0)) / B_e_pre_first_fill
```

Fees are signed balance impacts: costs negative, rebates positive, included once.
Profitable fills do not offset losses. Normalize once by raw causal account balance
before the first HSL fill. Zero-loss executed episodes count as zero in the arithmetic
episode mean; triggers without executed fills do not enter it. With no execution
episodes, maximum and mean are zero with episode-count metadata. Invalid anchors make
the selected loss aggregate unavailable, not a smaller sample of surviving episodes.

`hsl_execution_episode_count` counts actual execution episodes over the evaluated window,
including zero-loss and unfinished episodes with an attributed fill. Triggers without an HSL
fill do not count. Panic-tagged fills after the controller retires HSL intent do not count.
A lower limit of one requires an executed stop without requiring a loss;
no execution episodes gives zero. Invalid execution-count support makes the selector unavailable.

Count actual observed trigger/lifecycle events. Do not reconstruct hypothetical old
triggers. RED time retains reported controller RED, including enforced restart cooldown,
which can outlast a current raw signal breach. Count union time with any enabled scope
RED, excluding warmup. Overlapping controllers do not multiply elapsed duration. In
coin/position-side mode this does not imply that the entire portfolio was inactive.
Disabled HSL and enabled HSL with zero observed triggers remain distinguishable.

### Profit accounting

For realized loss/profit, each actual fill contributes `x_i = pnl_i + signed_fee_i`,
including entry fees and HSL fills. Any separately modeled trading cashflow enters once
as its own signed accounting event. Positive and negative net event cashflows define
the two sums; UPNL is excluded. Profit 100 and loss 70 give 0.7 and net realized profit
30. The actual event grouping is part of the definition: regrouping offsetting events
can change the ratio even when total net P&L is unchanged. Do not add inverse profit
factor, automatic side copies or closed-trade alternatives.

For short contribution, use additive side strategy P&L:

```text
P_side = net_realized_trading_cashflows_side + UPNL_side_end - UPNL_side_start
short_contribution = P_short / (P_long + P_short)
```

Use `position_side`, never buy/sell direction. Include fees, rebates, HSL losses and all
modeled trading cashflows once. Combined side profit reconciles with strategy-equity
change within declared numerical tolerance. Marked UPNL prevents realizing winners
while hiding open side losses. Missing required attribution is unavailable.

Require positive combined profit. Long/short profits 100/0, 0/100 and 100/100 give
0, 1 and 0.5. With positive total, a losing short produces a negative share; a losing
long produces a share above one. Do not clamp or take absolute values. Bounds 0.1 and
0.9 require at least 10% of positive net profit from each side. Nonpositive combined
profit is unavailable and cannot satisfy an enabled contribution limit; two losing
sides must not appear balanced.

## Status, selection and result integrity

Keep status separate from a finite optional numeric value: finite, positive infinity,
negative infinity, undefined and invalid are distinct outcomes. A genuine nonzero/zero
ratio can be infinite; 0/0 is undefined. Nonfinite required facts, invalid clocks,
missing support and numerical overflow are invalid/unavailable, not genuine infinity.
No epsilon floor, arbitrary ratio cap or friendly empty-input fallback alters a formula.

No-trade flat equity can have zero growth, volatility, drawdown, exposure and turnover.
Its 0/0 return or loss/profit ratios remain undefined. Completion one does not prove no
liquidation: preserve termination status even on the final evaluated minute.

Compute selected objectives, enabled limits and required validity facts only. A regular
report may request the core catalog. Request masks must not alter order decisions,
fills, balances, risk state or termination. Keep compact support metadata outside the
numeric metric catalog: definition version, actual/intended clocks, daily-return count,
half-life, observation count, HSL counts, scope policy and termination/censoring status.
  Omit support counts for groups that were not computed rather than implying zero observations.
Unknown core selectors fail explicitly. Every required scenario/exchange must supply
the requested value/status; missing or invalid sources must not be dropped from a mean.

Selected unavailable metrics cannot satisfy limits or receive favorable fitness.
Unselected unavailable diagnostics have no scoring effect. Raw extended-real values
govern limits: positive infinity passes a finite lower bound and fails a finite upper
bound, with the converse for negative infinity. An infinite violation enters the finite
invalid-candidate path; it does not produce an infinite optimizer penalty vector.

Reduce raw values before objective adaptation. `min`/`max` use extended-real ordering;
`mean` propagates a single infinity sign but opposing infinities are undefined; an even
`median` averages its middle pair with the same rule. Any infinite input makes `std`
unavailable. Compute only requested reductions, with complete required source coverage.

For minimization engines, adapt a reduced objective as follows:

```text
sign = +1 for a minimization goal, -1 for a maximization goal
engine_objective = (2 / pi) * atan(sign * raw_objective)
```

Genuine infinite endpoints map to finite -1/+1. Raw metric serialization and limit
comparisons remain unchanged. This common monotonic adapter preserves mathematical
ordering/Pareto dominance but changes objective-space distances and crowding; the
evaluation contract must version it explicitly. Compare feasibility before objective
quality on every backend. A finite invalid-candidate transport placeholder is not a raw
metric value and must not be persisted as one.

Definition versions and implementation fingerprints jointly guard stored fitness reuse.
Historical results retain their original definitions and must not be silently relabelled.

Use bounded accumulators: daily sufficient statistics, calendar-decayed growth sums,
running peak/path moments, per-open-position episode state, fill-gap/notional sums,
exposure moments, side cashflows/boundary UPNL and per-controller HSL episode state.
Retain trading-owned histories even when report-only histories are removed.

## Required evidence

Independent fixtures cover clocks and partial days, constant/flat returns, zero and
infinite denominators, negative equity, brief/sustained drawdowns, peak ties, open tails,
turnover examples and instrument multipliers, exposure scaling, HSL episode attribution
and RED unions, fees/rebates, signed side contributions, selected/all-core equivalence,
and unchanged fills/account state. Native Python tests verify the built source fingerprint.
Actual supported Metal/CUDA execution is required for a runtime parity claim; absent
hardware is missing evidence. CPU collateral support does not imply GPU collateral support.

The implementation does not add a retired-metric analysis tool, a compatibility metric
subsystem or generic custom-metric framework. Additional quantities are separate requests.
