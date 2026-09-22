# Revised Equity Hard Stop Loss

Select `live.hsl_engine=revised` at startup to use this engine. `legacy` remains the
implicit default. The selector applies to live execution, the offline fake runner,
backtests and CPU optimization; GPU optimization does not support revised HSL.
Changing engines requires a restart and an engine-compatible configuration.

## Signals and scopes

All three modes use the same reconstructed equity-peak drawdown and EMA calculation.
They differ in which positions contribute and which positions a RED signal closes.

| `live.hsl_signal_mode` | Controller and RED scope | Current budget | Policy configuration |
|---|---|---|---|
| `coin` | One symbol and position side | Raw balance / that side's applicable position-slot count | `bot.long.hsl`, `bot.short.hsl`; coin overrides allowed |
| `pside` | All positions on one side | Raw balance | `bot.long.hsl`, `bot.short.hsl` |
| `unified` | One controller for the whole portfolio | Raw balance | Explicit `bot.hsl` |

Unified protection includes exposure on either side even if ordinary entries on that
side are disabled. It does not inherit policy from the long or short blocks. Coin mode
never substitutes a divisor for an inactive zero-slot side. Live coin budgets use configured
slots; a backtest's explicitly enabled tradability-aware slot policy remains applicable.
TWEL does not multiply the HSL budget.

For each scope, add realized cashflows and unrealized PnL in currency before computing
drawdown. Within the configured lookback and current reset episode:

```text
X[t] = cumulative net realized PnL[t] + unrealized PnL[t]
current_equity = budget + unrealized PnL[last]
equity[t] = current_equity + X[t] - X[last]
peak[t] = cumulative_max(equity)[t]
raw_drawdown[t] = (peak[t] - equity[t]) / peak[t]
smoothed[t] = EMA(raw_drawdown, span=ema_span_minutes, adjust=False)[t]
score = min(raw_drawdown[last], smoothed[last])
RED = score > red_threshold
```

Only the current score authorizes panic. Historical threshold crossings do not latch RED.
Recovery to GREEN retires unfilled panic orders, including after a partial panic close.

The first raw drawdown seeds the EMA. Fractional spans are preserved. Repeated updates
within a minute replace that minute's sample rather than repeatedly advancing the EMA.
Known fill boundaries are evaluated before an episode reset or reopening. Nonpositive
historical peaks and extreme numeric histories use explicit, reported approximations.
Thresholds are not numerically interchangeable with legacy coin HSL, which uses a
realized-PnL peak and a different denominator. Reevaluate configurations and optimizer fitness.

## Best-effort history

Missing, delayed, conflicting or incomplete historical fills and candles are estimation
inputs, not a requirement for perfect reconstruction before HSL can act. The shared Rust
evaluator preserves usable cashflows and reconciles history to current positions, reporting
approximation reasons. Restored evidence replaces estimates on subsequent evaluation.

Historical valuation uses minute closes for both long and short positions. When true 1m
candles are absent, the finest available completed 5m/15m/1h candle is expanded along a
deterministic OHLC zigzag. Remaining gaps use forward-fill, then backfill for a missing
prefix inside the lookback. Live valuation ends at a fresh current mark. A historical
wick that recovers before the close is not an additional HSL sample.

With no usable history, current size, basis and mark still define current UPNL. The
estimator uses an entry-equity reference `budget` and one drawdown sample; it does
not invent an earlier zero sample to dilute the EMA. For loss `L`, the minimal signal is
`L / budget`: a 100 loss against a 1,000 budget gives 10%.
Current equity may be zero or negative; this is a loss signal, not missing input. This is
part of the same evaluator, not a separately timed fallback. Available history is not
arbitrarily discarded to use this shorter path.

Estimation can produce earlier or later stops than complete evidence. Completely missing
realized losses cannot be recovered from current positions alone. Diagnostics expose this
uncertainty; they are not trading authority. Fresh usable current balance, positions and
required marks remain mandatory, and exchange writes also require current order state.
Other strategy consumers retain their own input requirements.

## Stops, cooldown and restart

Trading has GREEN and RED behavior; there are no YELLOW/ORANGE trading tiers or separate
terminal drawdown threshold. Only the latest episode matters. Evaluate its terminal accounting sample, including the
final realized PnL and fees, before resetting the peak and EMA. If that terminal score
exceeds RED, the flat scope is eligible for cooldown regardless of the closing order type.
An earlier RED followed by terminal GREEN does not start cooldown. For an eligible scope:

- `restart_after_red_policy=always` permits restart after cooldown. Zero cooldown means
  no wait, subject to current HSL and ordinary strategy requirements.
- `restart_after_red_policy=never` prevents automatic restart while the stop remains
  inside the configured lookback. It is not a permanent, locally persisted halt.

Fresh exchange positions establish flatness even if the closing fill is missing. Prefer
its reconstructed timestamp; otherwise estimate cooldown from the last retained causal
fill in that scope. Repeated observations do not renew it. A delayed final fill may correct
the anchor and restore remaining cooldown. No retained fills means no historical cooldown.
Aggregate flatness requires every selected position to be zero, not zero net exposure.

Any renewed exposure clears the preceding cooldown and begins normal evaluation of the
new episode. There is no revised `hsl_position_during_cooldown_policy` or forced re-panic.
Fresh losses in that new episode can independently produce current RED.

Cooldown is reconstructed each evaluation. A balance/budget change or corrected history
can remove or restore terminal RED; any remaining duration is measured from the original
estimated flatten timestamp, never from the time of reclassification. This also applies
to the lookback-bounded `never` restriction.

All history and lifecycle anchors outside the lookback are forgotten, including `never`
halts. Equal exchange evidence and configuration reconstruct
equal decisions after restart. A previous RED decision that produced no recoverable
exchange evidence has no separate local authority. There is no revised emergency journal.
Repeated stops under `always` have no second, terminal accumulated-loss threshold.

## Configuration and live validation

Follow the [migration rules](configuration.md#experimental-revised-hsl-configuration):
enabled scopes need an explicit restart choice and a 1–90 day lookback; unified requires
an explicit portfolio policy. Removed tier fields, intervention values, optimizer paths
and cached fitness must not silently become different policies.

Offline reference, unit, native parity and fake-exchange tests do not establish actual
exchange execution correctness. Use the [live validation and rollback checklist](hsl_revised_live_validation.md)
for an operator-approved trial. Existing live bots and the legacy default are unchanged.
