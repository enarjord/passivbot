# Equity Hard Stop Loss

V9 HSL uses forward exponential currency drawdown, an independent EMA of the drawdown
ratio, and finite cooldown after a qualifying flatten. Rust owns the model across CPU,
supported GPU execution and live fact reconstruction. See the exact
[numerical and bootstrap reference](v9_risk_model.md) and the
[v8 migration guide](v8_to_v9_migration.md) before using older configurations.

## Scopes and budgets

| `bot.global.hsl.scope` | Scope and RED closes | Current base budget | Policy |
|---|---|---|---|
| `coin` | One symbol/position-side pair | Raw balance / that side's applicable slot count | `bot.long.hsl`, `bot.short.hsl`; coin patches allowed |
| `pside` | Every position on one side | Raw balance | `bot.long.hsl`, `bot.short.hsl` |
| `portfolio` | All positions | Raw balance | Explicit `bot.global.hsl` |

Portfolio protection includes existing exposure on disabled entry sides. A hedged scope is
flat only when gross inventory is zero. Replacing a constituent while an aggregate remains
exposed does not start a new episode. Coin mode does not invent slots for an inactive side.
TWEL does not directly multiply the HSL budget.

`bot.<side>.hsl.scale_budget_with_excess_allowance` defaults to `false`. In coin mode only,
`true` multiplies its slot budget by bounded authored exposure headroom:

```text
base_wel = authored WEL, or side TWEL / applicable slots if automatic
allowance = min(max(0, we_excess_allowance_pct), max(0, side TWEL / base_wel - 1))
budget = raw balance / applicable slots * (1 + allowance)
```

A nonpositive base WEL grants no headroom. The switch may vary per coin; coin-specific
WEL/excess values apply. It uses configured headroom rather than current utilization.
The option is rejected in pside/portfolio modes.

## Drawdown and ratio smoothing

Each HSL policy's `memory_ema_span_minutes` controls slow decay. Its
`drawdown_ratio_ema_span_minutes` independently controls fast smoothing of the sampled ratio. Both are
float EMA spans in minutes. For one observation, omitting numerical rounding details:

```text
alpha_slow = 2 / (memory_ema_span_minutes + 1)
alpha_fast = 2 / (drawdown_ratio_ema_span_minutes + 1)
loss = previous_UPNL - current_UPNL - net_realized_cashflow
D = max(0, (1 - alpha_slow) * D_previous + loss) # first event of a new minute
E = current_budget + current_UPNL
R = D / (E + D)
M = (1 - alpha_fast) * M_previous_minute + alpha_fast * R
RED = min(R, M) > red_threshold
```

Currency D and dimensionless R/M are different quantities. Historical ratios retain their
contemporaneous budgets; current balance/slot changes do not renormalize old ratios. Entry
and exit fees, signed rebates and all supported trading cashflows are consumed exactly once.
Deposits/withdrawals and unrelated-scope budget changes are not trading losses.

At a proven flat-to-exposed transition, initialize D=M=0 before entry economics. Partial
closes do not reset the episode. Each elapsed minute applies slow decay exactly once;
ordered events within that minute update D without a second slow decay. A changed same-minute
observation replaces fast smoothing from the same previous-minute baseline. Pure queries
advance no clock and consume no fill twice. Nonpositive finite equity/denominator is RED;
nonfinite or missing required inputs are errors, not a neutral signal.

For example, with both spans 3.0, a first-minute loss of 100 against a 1000 budget gives
D=100, E=900, R=0.1 and M=0.05 from the zero seed. A threshold of 0.04 is RED; a threshold
of 0.05 is GREEN because the comparison is strict. The exact f32 graph, causal candle/fill
ordering and unsupported cashflow branches are specified in the reference.

Current GREEN retires unfinished panic intent, including after a partial close. HSL does
not preserve a prior panic order as continuing strategy authority.

## Qualified finite cooldown

At the exposed-to-flat transition, use the latest causal assessment immediately before the
final fill and the final accounting assessment, including realized PnL and fees:

```text
qualified = pre_final_fill_RED OR post_final_accounting_RED
cooldown_until = flatten_timestamp + duration if qualified else no_deadline
in_cooldown = flat AND now < cooldown_until
```

Any close can qualify, including a manual close without a Passivbot order tag. An earlier
RED followed by GREEN while still exposed is insufficient by itself. A final accounting
GREEN does not erase a qualifying pre-fill RED. For example, threshold 0.099 with a latest
pre-fill score 0.09912 and final score 0.09884 still starts the configured finite cooldown.

After qualification, later GREEN, budget changes or repeated flat observations do not
cancel, renew or extend the deadline. Corrected facts require shared reconstruction.
Actual new exposure clears the old terminal record and starts a new episode. Zero cooldown
permits immediate restart subject to normal strategy/risk gates. There is no `never`
restart policy; an operator or external supervisor owns permanent shutdown.

## Live reconstruction and availability

Forward CPU/GPU simulation keeps constant retained risk state per scope and processes new
facts once. Fills, budget changes and diagnostics do not rescan historical candles. Live
startup/repair can reconstruct bounded history through the same Rust core.

Live targets five slow spans and at least every active policy's configured cooldown, within
`live.risk_bootstrap_max_lookback_minutes` (maximum 90 days). An enabled cooldown exceeding
that cap fails live capability validation. Coverage, unknown prefix, shortened targets, ambiguous fills, missing historical
marks and estimated budgets are observable. The same retained boundary and ordered facts reproduce the same fold. Routine pruning
keeps compact forward state; restart, incompatible policy or corrections requiring discarded
history may need explicit approximate reconstruction. Unknown exposed prefixes use the
approximation in the reference;
a finite suffix cannot prove equality to an unknown infinite past. Cooldown comes from
factual flatten reconstruction, never a local permission journal or old order tag.

Fresh current balance, positions, open orders and required marks remain mandatory for
protective exchange actions. Historical approximation does not make stale current facts
fresh. Ordinary strategy and unrelated risk consumers retain their own readiness needs.
See the [failure contract](ai/error_contract.md).

## Migration, reporting and qualification

V9 configs use `config_version="v9.0.0"`, coin/pside/portfolio modes, and no restart field.
The explicit migration tool accepts released v8 and final canonical v9 and requires acknowledgment
of an active old realized-loss gate's memory change. V8 also needs a slow-memory choice and
acknowledgement of any removed Never policy. It retains original configs for version-matched
rollback and never reuses old optimizer fitness. See [configuration](configuration.md#hsl-configuration).

HSL risk diagnostics are model-dependent. Use ordinary account-equity metrics from full
fills/equity for v8/v9 economic comparison. Optional `backtest.hsl_detailed_report=true`
adds risk samples/plots; optimizer replay omits extra diagnostic arrays while required
account/objective metrics remain complete.

Follow the [validation matrix](hsl_testing.md). Supported-device numerical execution,
end-to-end performance and exact intended-config backtests are separate evidence gates.
Offline qualification authorizes no actual live trial or deployment; those use the
[separately approved live validation/rollback procedure](hsl_live_validation.md).
