# Optimizer configuration prefilters

`optimize.prefilter` can reject a candidate before its CPU or native GPU backtest.
It is purely opt-in: the default `checks` list is empty, and each authored check
also defaults to `enabled: false`. Passing candidates follow the existing replay
and scoring path. These settings never change live trading behavior.

For example, explicitly enable a normalized grid-span range of 10% to 35%:

```json
{
  "optimize": {
    "prefilter": {
      "checks": [
        {
          "metric": "grid_span",
          "enabled": true,
          "min": 0.10,
          "max": 0.35,
          "scale": 0.10,
          "weight": 1.0,
          "sides": ["long"]
        }
      ]
    }
  }
}
```

This is a policy example, not a recommended universal range. Optional `min` and
`max` bounds are inclusive. An enabled check needs at least one bound. All checks
currently require `live.strategy_kind = "trailing_martingale"`.

## Available checks

| Metric | Measurement |
|---|---|
| `entry_close_span` | Entry-to-close landmark distance divided by average position price at each exposure probe. Includes the normalized EMA entry gate; handles partial initial entries and immediate trailing closes separately. |
| `entry_retracement` | Effective rebound/pullback required to confirm a trailing reentry. Zero for passive or partial initial entries. |
| `close_retracement` | Effective reversal required to confirm a trailing close. |
| `entry_cooldown_minutes` | Effective cooldown, including exposure/adverse weights and floor/ceiling. |
| `grid_span` | Absolute last-entry price / initial-entry price minus one, along the normalized capacity-filling ladder below. |

Percentage measurements use fractions: `0.10` means 10%. Entry/close landmark
span uses average position price as denominator, not the entry landmark. These
measurements identify configured geometry; they do not forecast time to a fill.

## Explicit probe assumptions

The policy defaults are:

```json
{
  "checks": [],
  "exposure_ratios": [0.25, 0.5, 0.8, 0.95],
  "volatility_ema_1m": 0.0,
  "volatility_ema_1h": 0.0,
  "adverse_directionality": 0.0,
  "grid_max_entries": 1024,
  "penalty_weight": 1000000.0
}
```

Exposure probes are fractions of the effective per-position allowance, including
the bounded excess allowance. They must remain below the Rust full-capacity
boundary of `0.999`. Cooldown exposure weights use Rust's base-WEL denominator.
Inherited WEL uses the backtest allocation master's total and configured slots.
With dynamic WEL enabled, a single entry-eligible coin has one effective slot;
one configured slot likewise has an unambiguous denominator. Multiple eligible
coins and multiple slots have a candle-dependent grow-only tradability count.
This foundation raises an explicit unsupported-input error for inherited WEL in
that case rather than guessing a denominator. Fixed WEL mode and explicit coin
WEL allocations support multiple coins. Disabled filters impose no such limit.
Volatility inputs are declared log-range EMA values, not estimates derived from
EMA spans. Adverse directionality is an explicit value in `[0, 1]`.

The grid probe keeps balance and the EMA reference at 1. It asks Rust for each
successive entry, fills it, and updates signed position size and average price.
For trailing entries it supplies an adverse excursion just beyond activation and
the minimum confirmation reversal. Passive entries use their recursive prices.
Rust owns EMA gating, sizing, double-down factors and exposure cropping. The probe
ends at 99.9% of capacity. Price and quantity steps are normalized to `1e-12`,
minimum quantity is `1e-12`, minimum cost and fees are zero.

This is a hypothetical entry-only ladder, not a replay or an exchange-executable
price prediction. It excludes closes, risk reducers, elapsed cooldown time,
balance changes and changing EMAs/volatility. A full initial entry has zero grid
span. An unreachable or bounded-out ladder is rejected by an enabled grid-span
check, with at least one normalized violation unit; its partial span cannot
establish that the requested complete-grid range was satisfied.

## Constraint and evolutionary feedback

For every enabled check, the normalized violation is:

```text
max(0, min_bound - measurement, measurement - max_bound) / scale
```

Omitted bounds contribute nothing. `scale` and `weight` are positive and default
to 1. Each check uses its worst violation across applicable coin/side/exposure
probes. The candidate constraint is `penalty_weight * sum(weight * violation)`.
Suites use the worst complete scenario constraint, including scenarios outside
an optional GPU screening subset. Repeating probes does not multiply penalties.
Tiny positive violations are floored to `1e-9` to remain distinct from feasibility
in the existing evolutionary comparator.

Zero proceeds to replay. A positive constraint completes the candidate without
replay: each minimizing fitness component is the constraint, while raw numeric
metric defaults use its negative for maximizing objectives and its positive for
minimizing objectives. Metric directions use the canonical registry, with explicit
objective goals taking precedence. Directionless diagnostic metrics receive the
positive constraint. Existing backtest limits are not reapplied to these defaults.

Records retain `evaluation_origin: "config_prefilter"`, `backtest_performed: false`
and the violated measurements. They are optimizer penalty evaluations, not measured
performance. The same F/G values reach ordinary CPU and native GPU evolutionary
selection, even during scenario screening. Rejections consume candidate evaluation
budget but submit no simulator work. Filter policy and definition version are part
of saved-fitness identity; changing enabled policy requires a fresh run.

Malformed settings, unsupported strategies and non-finite results raise errors;
they are not silently converted into passing candidates. Coin membership and
override files must already be resolved by canonical preparation. The pure Python
API is `optimization.prefilter.config_prefilter(config, prefilter_params) -> float`;
the corresponding inspection helper returns explanatory contributions as well.
