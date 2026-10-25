# Numeric Names And Units

Names for new or renamed owned interfaces describe the quantity, its role, and its unit.
These conventions apply to configuration, native parameters, metrics, tools, and documentation.
They do not change formulas, authorize aliases, or infer a field's accepted domain.

## Quantities

| Quantity | Naming | Representation |
| --- | --- | --- |
| Proportion, allocation, relative change, or relative price offset | `_fraction` | Scale one: `0.025` means 2.5%. State the denominator and sign convention. |
| Quotient or relative level | `_ratio` | Numerator divided by denominator, generally without subtracting one. State both operands. |
| Percentage | `_percent` | Scale 100: `2.5` means 2.5%. Prefer fractions for calculations and config. |
| Basis points | `_bps` | Scale 10,000: `250` means 2.5%. |
| Difference between percentages | `_percentage_points` | Absolute percentage-point difference, not relative percentage change. |
| Probability | `_probability` | Scale one in `[0, 1]`; identify the event and candidate/variable scope. |
| Direct scaling | `_multiplier` | `result = input * multiplier`; one is neutral. |
| Incremental allowance or padding | `_fraction` | `result = input * (1 + fraction)`; zero is neutral. |
| Formula sensitivity | `_coefficient` | Document the input and output units. A coefficient need not be bounded. |
| Relative weighting | `_weight` or `weights.*` | Authored weights need not sum to one; distinguish raw weights from normalized values. |
| Normalized score | `_score` | State its definition and domain; a bounded score is not necessarily a population share. |
| Frequency or time rate | `_per_second`, `_per_day`, etc. | State the time denominator. A win proportion is not a temporal rate. |
| Logarithmic return | `_log_return` | `ln(new / reference)`; simple return is `expm1(log_return)`. |
| Logarithmic level quotient | `_log_ratio` | Identify the operands; do not imply a chronological return. |

`_pct` is ambiguous in existing interfaces. Do not introduce it for new owned numeric values.
Renaming an existing scale-one `_pct` value to `_fraction` does not divide it by 100.
Renaming a real scale-100 `_pct` value to `_percent` does not rescale it either.

Fractions and ratios overlap mathematically. The naming distinction communicates the semantic
quantity, not a theorem about its domain. A fraction can be signed or exceed one when its
definition permits that: returns, price offsets, rebates, and excess allowances are examples.
Keep bounds, nullability, sentinels, rounding, and clipping explicit in their owning contracts.

Use `offset` for signed displacements and `distance` for nonnegative magnitudes where practical.
Price helpers state their reference and sign: `calc_abs_relative_difference_fraction` uses the
absolute reference denominator; `calc_pside_adverse_price_change_fraction` is positive for a
change adverse to long/short; `calc_order_price_gap_fraction` is positive on the passive side of
a buy/sell order. A signed gap can be negative when an order crosses the market. Their existing
zero, nonfinite-input and invalid-side behavior remains owned by the native implementation.
Conventional financial names such as `sharpe_ratio`, `sortino_ratio`, `omega_ratio`,
`calmar_ratio`, and `sterling_ratio` remain useful. Document their actual estimators, sampling,
annualization, floors and clipping; the familiar name alone does not establish those formulas.

## Roles And Qualifiers

Threshold, limit, cap, floor, tolerance, margin, and buffer describe roles, not units. Prefer
`quantity_role_unit`, such as `drawdown_threshold_fraction`, `exposure_limit_ratio`, or
`latency_threshold_seconds`. A numeric threshold is distinct from the event it triggers.
A target should not be called a cap when the implementation actively seeks that target.

Generic heterogeneous containers remain generic: optimizer limit `value`/`range` and prefilter
`min`/`max`/`scale` inherit units from their selected metric. Do not suffix all such values with
`_fraction`. Relative score margins, financial margin, and absolute price tolerances also need
their distinct meanings retained.

Time spans retain explicit units and floating-point precision. A multiplier is not a duration.
Do not silently reinterpret a dual-purpose control: for example, existing warmup acquisition
and activation formulas have different baselines that are explicitly documented by their owner.

Percentile and tail qualifiers describe observation selection, not output scale. `p95` is a
percentile; `worst_1pct` identifies a tail population. A worst-tail mean is not the percentile
at its cutoff. An `expected_shortfall_1pct_fraction` therefore has a legitimate percentage
selection qualifier and a fraction-valued result. Percentile helper arguments may genuinely
use 0 through 100; do not convert them through a spelling-only change.

## Metrics And Metadata

For new or renamed metric families, use
`<base_with_unit>_<curve>[_<position_side>][_ew]` when those qualifiers apply.
Use `usd`, `btc`, and `strategy_equity` consistently for the curve basis and put exponential
weighting last. For example, `adg_fraction_usd_ew` and `sharpe_ratio_btc_ew` disclose different
quantities without turning a conventional quotient into a fraction.

Within a metric base, prefer statistic before unit, for example `drawdown_max_fraction` or
`entry_interval_p95_hours`. Preserve a metric's intrinsic statistic separately from outer
suite/exchange aggregation in structured result records.

Config field and metric metadata must make these facts discoverable where applicable:

- unit and numeric scale
- numerator, denominator, reference or quantity basis
- sign convention, domain, missing values and sentinels
- sampling interval, observation population, weighting and aggregation
- curve/currency/position-side scope and formula definition

Reuse the configuration field registry's existing `unit`, `denominator`, domain and description
metadata. Metric definitions should be owned once and reused by result serializers, scoring,
CPU/GPU capability declarations and generated documentation. Do not duplicate trading formulas
in a naming validator or move an existing owner's defaults into descriptive metadata.

## Interface Boundaries And Changes

Normalize external units at owned adapter boundaries. Preserve exact exchange, CCXT, library,
and other external protocol keys even when their names are misleading. A wire field name is
not proof of its scale; use the actual interface contract.

Existing native IDs and adapters may retain prior spellings until a coordinated rename updates
both ends. A code-wide rename includes native structs, Python bindings, optimizer keys, packets,
CPU/GPU code, identities, validation, tests, examples and active documentation as applicable.
Do not advertise an internal/native name as another accepted authored-config spelling.

Preserve frozen released-version source names, explicit migration source keys, external formats,
and accurately labelled historical quotations. Explicit released-v8 conversion reports final
names without silently rescaling their values. Do not add aliases or migrations for earlier
unpublished v9 names, including through result readers or generic metric canonicalization.

A rename must preserve values, formulas, domains, precedence and optimizer direction. Correcting
a measurement population, fee treatment, denominator or formula is a separate behavior change
with its own evidence and validation; a better name must not conceal that change.
