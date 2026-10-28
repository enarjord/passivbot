# Recipe: Long-Only Unstuck Template

This recipe constructs a multi-coin trailing-martingale template with fixed nominal exposure and
position count, HSL disabled, both exposure enforcers disabled, and unstuck enabled. It illustrates
the [campaign runbook](../runbooks/optimization_campaign.md), including focused adaptation from an
older starting config followed by wider discovery and qualification. It is a procedure, not a
claim that this style or any resulting config has passed final robustness tests.

## Declare The Style Before Searching

Copy an appropriate authored config, or use the public
[trailing-martingale example](../../../configs/examples/default_trailing_martingale_long.json) as a
starting profile. Preserve deliberately chosen bounds; default example ranges are not a substitute
for the user's search design. Migrate older schemas explicitly before anchoring a v9 search.

For this style, set these effective policies in the bot and retain them through every stage:

| Policy | Required value |
|---|---|
| `bot.portfolio.strategy_kind` | `trailing_martingale` |
| `bot.long.enabled` / `bot.short.enabled` | `true` / `false` |
| `bot.global.hsl.enabled` | `false`; no enabled side/coin override |
| `bot.global.risk.total_wallet_exposure_limit_ratio` | `1.5` |
| `bot.global.risk.max_positions` | `10` |
| `bot.global.risk.position_exposure_enforcer.enabled` | `false` |
| `bot.global.risk.total_exposure_enforcer.enabled` | `false` |
| `bot.global.unstuck.enabled` | `true` |
| `backtest.candle_interval_minutes` | `1` |
| `backtest.ew_half_life_days` | `180` |

Fix the corresponding numeric bounds to `[1.5, 1.5]` and `[10, 10]`. Align their starting bot values
too. Inspect explicit long-side and coin overrides so they do not override the global policy.
If the source config uses portfolio-scope HSL, disable its explicit portfolio HSL policy as well.
Do not use optimizer-only runtime overrides to make an otherwise different exported config appear
to satisfy the style. Keep the approved coin universe explicit and inspect effective suite subsets.

TWEL is a nominal per-side entry budget. With exposure enforcers disabled, it is not a guaranteed
ceiling on realized marked exposure. Keep `we_excess_allowance_fraction` as a separate, deliberately
bounded search choice. Unstuck manages loss within its own policy; enabling it is not a substitute
for a loss or liquidation guarantee. Evaluate actual drawdown, exposure, holding tails, and costs.

## Use Four Complementary Objectives

```json
{
  "scoring": [
    {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
    {"metric": "cagr_fraction_strategy_equity_ew", "goal": "max"},
    {"metric": "position_held_time_weighted_mean_hours", "goal": "min"},
    {"metric": "ulcer_index_fraction_strategy_equity", "goal": "min"}
  ],
  "limits": [
    {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.70},
    {"metric": "backtest_completion_fraction", "penalize_if": "less_than", "value": 0.99}
  ]
}
```

These are entries under `optimize`, not a complete config. They favor whole-window growth, recent
growth, shorter position episodes, and less persistent/deep drawdown. Time-weighted holding is
`sum(hours^2) / sum(hours)`, not a simple average over fills or a notional-weighted measure. Ulcer
uses the drawdown path including flat/zero-drawdown periods. See the
[metric definitions](../features/backtest_metrics.md); preserve their statuses and support counts.

The initial limits are broad discovery guards, not a claim that a 70% drawdown is acceptable for
deployment. Inspect the observed feasible pool before adding a positive-growth or activity floor.
Do not select a no-trade or money-losing member merely because it minimizes holding time or Ulcer.
Also inspect maximum holding time and recovery tails: a favorable weighted mean can hide a bad
individual episode. Keep the four scoring objectives stable when adding diagnostic metrics.

## Recent-Window Adaptation, Then Full Search

Create private stage configs and separate seed/anchor directories before using these commands.
The paths below are placeholders for those artifacts; no command supplies the missing configs or
authorizes remote execution. Set a fixed campaign cutoff in every stage config. A useful initial
window is 2025-01-01 through that cutoff, with suite mode disabled.

If the starting strategy is credible but newer risk/selection controls need adaptation:

```bash
passivbot optimize campaign/stage-1.json --optimizer-backend gpu --suite n \
  -t campaign/anchors \
  -ft global.entry_cooldown,global.forager,global.unstuck,global.risk.we_excess_allowance_fraction,portfolio
```

The selectors address shared v9 policy and portfolio memory/settings. Include excess allowance
explicitly; it is outside the `global.unstuck` subtree. `portfolio` is a broad selector: review its
actual matching bounds and fix any portfolio policy that must not move. It does not make omitted
or already-fixed bounds tunable. Fine-tune with anchors fixes non-selected optimizer coordinates
from each anchor; the stage config still owns boolean/runtime policy. Inspect selector expansion,
anchor clamping, participating Forager weight groups, and actual tunable counts before continuing.

Review finite cooldown caps, span units, lower bounds, and normalized initial values. Search-space
sanity can expose a missing required parameter or a starting value outside its fixed bound. Resolve
those explicitly, rather than inventing a trading input or silently changing the whole search.

For four objectives, automatic pymoo selection uses NSGA-III; use the logged population/reference
direction counts in budgeting. An illustrative first assessment is 4,096 complete evaluations,
followed by another 1,024 when productive. Around 1% or less fixed-reference hypervolume gain over
the additional interval can support moving on if useful tradeoffs and CPU checks also stabilize.
These numbers are planning heuristics, not universal sample sizes or convergence guarantees.

Promote a diverse pool including growth, EW, holding, low-Ulcer, and useful intermediate choices,
plus any protected CPU-tested leaders. Materialize those configs against the next stage's original
full bounds and fixed policy. Then start fresh without `-ft`:

```bash
passivbot optimize campaign/stage-2-full.json --optimizer-backend gpu --suite n \
  -t campaign/diverse-seeds
```

This searches all independently tunable authored bounds. Do not pass a phase-one result as the sole
base config if its saved bounds have already been collapsed by fine-tune. Keep some exploration
capacity instead of filling a huge initial pool with nearly identical seeds. Fresh GPU seed
evaluation does not include independent CPU backtesting.

## Expand History And Scenarios

When broad discovery is mature, freshly evaluate diverse seeds on 2023-01-01 onward, then
2021-04-01 onward, through the same frozen cutoff. Prepare and verify genuinely older candles and
their maximum warmup prefix; relabelling or slicing the recent dataset does not expand history.
Preflight each larger workload before allocating it. Use the runbook's explicit alternative plan
if continuous GPU history will not fit.

Then introduce the authored base and coin-subset scenarios, followed by selected temporal regimes
and plausible execution/cost stress. Check the actual approved-coin intersections and coverage;
scenario lists must not silently broaden the intended universe. Fix aggregation and inspect each
scenario's risk rather than relying only on its mean. Continue search where the broader tests
expose weaknesses. Every changed evaluation policy gets fresh scores.

Before final promotion, inspect whether broad positive-growth/activity limits retain useful
diversity. Choose any stricter risk tolerance from the intended use and observed evidence. Do not
ratchet limits until every point is infeasible, or quietly redefine success to keep one winner.

Optionally polish several distinct tradeoffs after broad discovery. For each one, create a config
containing that candidate's bot values and a copy of the original authored bounds:

```bash
passivbot optimize campaign/polish-balanced.json --optimizer-backend gpu \
  --polish-bounds-fraction 0.25 --polish-bounds-mode clamp \
  -t campaign/balanced-anchor
```

Polish centers on the **base config's current bot values**, not automatically on every `--start`
member. Clamp mode preserves the original domain and fixed bounds. A zero-centered relative bound
stays zero. Preserve the broader search artifacts and confirm useful alternatives; one narrow
search around the highest-CAGR point is not a diverse confirmation campaign.

## Lessons To Carry Into Qualification

- Restoring broad indicator bounds can increase shared optimizer activation even on unchanged
  candles. Separate matched-activation CPU diagnostics from ordinary candidate CPU replay. The
  latter qualifies the exported artifact; neither requires editing factual dataset metadata.
- Normalization or clamping can change a seed's effective bot. Freshly score the actual prepared
  seed and retain the source-to-effective relationship; old anchor metrics do not transfer.
- A shorter holding-time tradeoff may accept more drawdown; a low-Ulcer point may sacrifice growth
  or retain long individual positions. Report these tradeoffs rather than declaring a universal
  winner from one improved objective.
- Sampled CPU/GPU ordering agreement is evidence for the sampled set. It does not establish blanket
  numerical parity or ensure the entire GPU front has the same CPU membership.
- Stage counts and a growing front are progress indicators. The finish is a useful CPU-rebuilt
  feasible front on the frozen final evaluation, supported by confirmation and reported limitations.

Preserve earlier CPU leaders and near-front candidates for that final replay. Deliver a small
representative selection alongside the full qualified front so the user can choose a tradeoff.
The recipe specifies how to search for this style; it does not promise a profitable or robust
configuration exists within the declared bounds.
