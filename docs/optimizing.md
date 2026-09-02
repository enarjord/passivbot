# Optimizing

Passivbot configurations can be optimized using a multi-objective evolutionary algorithm to balance performance metrics while meeting constraints.

The canonical defaults live in `src/config/schema.py`. The example config
`configs/examples/default_trailing_martingale_long.json` mirrors those defaults exactly. For the
recommended config workflow, see [Config Workflow](config_workflow.md).

Optimization requires the full install profile:

```bash
pip install -e ".[full]"
```

## Running Optimization

```bash
passivbot optimize [path/to/config.json]
```

- Defaults to the in-code schema in `src/config/schema.py` if no config is specified
- Use existing configs as starting points: `--start path/to/config(s)`
- Enable suite scenarios defined in `backtest.scenarios` with `--suite [y/n]` (omit value to enable)
- Layer additional scenario definitions via `--suite-config path/to/file.json`

The canonical default profile keeps `backtest.suite_enabled = false`, so optimize runs are
single-scenario by default unless you explicitly enable suite mode.

Example:
```bash
passivbot optimize configs/examples/default_trailing_martingale_long.json --start configs/starting_pool/
```

Most config parameters can be modified via CLI. `passivbot optimize -h` for more info.

### Scoring Objectives

`optimize.scoring` defines the metrics the optimizer tries to improve. The canonical form is a
list of objective objects:

```json
"scoring": [
  {"metric": "adg_strategy_eq", "goal": "max"},
  {"metric": "drawdown_worst_strategy_eq", "goal": "min"}
]
```

- `metric` is a canonical backtest metric name.
- `goal: "max"` means higher raw metric values are better.
- `goal: "min"` means lower raw metric values are better.

The older shorthand form is still accepted for built-in metrics with known default goals:

```json
"scoring": ["adg_strategy_eq", "drawdown_worst_strategy_eq"]
```

Passivbot normalizes that shorthand to the canonical object form during config formatting. For
custom or newly added metrics, use the explicit object form so the optimizer knows whether to
minimize or maximize the metric.

### Backend Selection

Passivbot supports three optimizer backends:

- `optimize.backend: deap`
  - Uses the existing DEAP evolutionary backend.
- `optimize.backend: pymoo`
  - Uses pymoo. This is now the default optimizer backend.
  - The default pymoo algorithm mode is `auto`: Passivbot uses `nsga2` when optimizing `3` or fewer objectives, and `nsga3` when optimizing more than `3`.
- `optimize.backend: gpu`
  - Uses an Apple Metal screening proxy and exact Rust validation. This backend is experimental and
    deliberately limited to the scope documented below.

Example:

```json
{
  "optimize": {
    "backend": "pymoo",
    "pymoo": {
      "algorithm": "auto"
    }
  }
}
```

### Apple MPS GPU Backend (Experimental)

The GPU backend is an additive research backend for Apple Silicon. Install the normal optimizer
dependencies plus its optional PyTorch runtime:

```bash
python3 -m pip install -e ".[full,gpu-mps]"
```

Select it with `--backend gpu` or `optimize.backend: "gpu"`. Normal live operation, backtesting,
and the DEAP/pymoo CPU optimizers do not import or require PyTorch.

The supported slice is intentionally narrow:

- Apple Silicon with `torch.backends.mps.is_available()`
- one prepared dataset per independent run or suite scenario; single- and multi-coin EMA Anchor
  and Trailing Martingale runs accept any positive integer
  `backtest.candle_interval_minutes`; the dataset may be an
  individual exchange or the canonical combined multi-exchange dataset
- `strategy_kind: ema_anchor` or `trailing_martingale`, with long-only, short-only, or
  long+short enabled for one coin
- long-only, short-only, or dual-side hedge-mode and one-way multi-coin EMA-anchor and
  trailing-martingale runs
  for up to 64 coins, with fixed or dynamic wallet-exposure allocation and independent per-side
  Forager selection; non-suite dual-side runs may use different effective long and short coin
  universes. With `backtest.dynamic_wel_by_tradability: true`, each side's WEL denominator grows
  with the maximum observed eligible coin count up to `n_positions`; with `false`, it remains the
  configured `n_positions`. Current Forager selection still uses the current tradable set in both
  modes;
  `live.forager_score_hysteresis_pct` preserves flat incumbent candidates when a challenger's
  normalized Forager-score lead is within the configured gap
- each enabled side's `n_positions` pinned to `1` and wallet-exposure limit kept positive
  for single-coin runs; supported multi-coin bounds may vary `n_positions` between `1` and the
  prepared coin count
- hedge mode and one-way mode; one-way flat-side arbitration uses the active strategy's Rust rule
- suite mode with exactly one prepared dataset per scenario, including individual-exchange
  comparisons and combined multi-exchange scenarios;
  single-coin EMA-anchor and trailing-martingale scenarios keep their existing directional
  support, while both strategies' suites may also use different multi-coin subsets of up to 64
  coins when every effective scenario shares the same supported side topology;
  scenario date, coin, ignored-coin, exchange selection, and fail-closed `bot.long`/`bot.short`
  config overrides are supported; scenario-local `coin_overrides` for supported multi-coin
  EMA-anchor and trailing-martingale runs, starting balance, maker fee, liquidation threshold,
  taker fee, market-order slippage, minimum-effective-cost filtering, finite or all-history PnL
  lookback, Forager hysteresis, hedge mode, HSL signal mode, dynamic-WEL denominator mode,
  ordinary-market enablement, and its near-touch threshold are also supported when the resulting
  scenario remains within the scope below, while other non-bot override paths remain unsupported;
  combined scenarios may use
  canonical per-coin source assignments, while an individual-exchange scenario fails closed if
  an effective assignment for one of its prepared coins selects another exchange
- static `coin_overrides` for each enabled side of single- and multi-coin EMA-anchor and
  trailing-martingale runs: `live.forced_mode_<side>: normal`, active-strategy parameters,
  `risk.entry_cooldown_minutes`, and explicit
  `wallet_exposure_limit`, `risk.we_excess_allowance_pct`, and all six `unstuck` leaves are
  supported. Static single-coin values are applied after each optimizer candidate, preserving
  exact Rust's override precedence. Checkpoint identity records the resolved exact override values
  before float32 Metal packing. Eligible forced-normal symbols in multi-coin runs reserve
  active slots before Forager ranking and may expand the active-set cap beyond `n_positions`; the
  configured WEL denominator mode remains unchanged, matching exact Rust. Trailing Martingale also supports
  per-coin
  `risk.position_exposure_enforcer_enabled` and
  `risk.position_exposure_enforcer_threshold`; per-coin
  `risk.we_excess_allowance_mode`, modeled leaves for disabled sides, and other override leaves
  fail closed. Non-`normal` forced modes remain accepted for either side because they are
  backtest-inert. Trailing
  Martingale also resolves all four `entry.ema_gate_mode` values per coin and side. In one-sided
  `live.hsl_signal_mode: coin` runs, all ten HSL
  leaves documented in `coin_overrides.md` are also supported. Fused dual-side EMA Anchor and
  Trailing Martingale coin mode resolve the same HSL leaves independently for long and short.
  CPU-compatible per-coin `live.leverage` and non-`normal` forced modes on either side are accepted
  for composed live-config portability, but neither affects an exact CPU backtest; GPU optimization
  therefore warns that they are backtest-inert and records them in checkpoint identity.
- one-sided single-coin and multi-coin EMA Anchor and Trailing Martingale runs support HSL with
  `coin`, `pside`, or `unified` signals and both resting-limit and market panic closes. Unified and
  pside multi-coin runs use one portfolio controller; coin mode uses an independent controller per
  coin and scales its loss budget by the configured denominator mode's effective position-slot
  count. Market panic orders fill on
  the next valid bar at its close shifted adversely by `backtest.market_order_slippage_pct`, rounded
  directionally to the exchange price step, and charged the resolved taker fee. The Metal proxy
  models tunable RED
  threshold, drawdown-EMA span, and cooldown, plus fixed yellow/orange ratios, orange entry
  suppression, RED latching, panic flattening, two-sample flat confirmation,
  positive-cooldown restart, zero-cooldown indefinite halt, cumulative no-restart peak tracking,
  effective coin-slot scaling, and terminal no-restart policy. For a finite
  `live.pnls_max_lookback_days`, the single-coin `coin`-mode directional kernels expire
  candidate-local realized-PnL fill events with the same rolling-window rule as exact Rust.
  Other HSL topologies deliberately retain all-history realized-PnL and strategy-equity peaks as
  a conservative envelope over Rust's rolling peak: they may trigger HSL early after an old peak
  ages out, but cannot suppress a drawdown for that reason. The selected history must have no
  internal invalid candles between its first and last valid samples. Thresholds in the
  float32-unrepresentable interval immediately below `1.0` fail closed. Exact validation and
  drift gates remain authoritative. Optimization may use normalized HSL lifecycle and risk
  signals: yearly trigger/restart rates, time in RED, halt-duration summaries, trigger drawdown,
  post-restart retriggers, panic-loss drawdown mean/max, and halt-to-restart equity loss. Raw event
  counts, yellow/orange occupancy, absolute panic-loss totals/maxima, minimum panic-loss drawdown,
  and mean flatten time remain exact-analysis diagnostics.
  One-sided coin-mode multi-coin runs may resolve all canonical HSL settings independently per
  coin, including HSL enablement and limit/market panic execution. Dual-side multi-coin EMA Anchor
  and Trailing Martingale use fused shared-account kernels for unified, pside, and coin signals,
  so shared event-loss, warning-tier overlap, and the other HSL lifecycle/panic-loss metrics are
  available. The worst and mean-worst-1% EMA-smoothed strategy-equity drawdown metrics are also
  available for the account and each side. The mean-worst-1% proxy uses an opt-in bounded
  logarithmic histogram whose only approximation is the partially selected cutoff bin; exact Rust
  validation and the rolling drift gates remain authoritative. Per-side
  `peak_recovery_hours_strategy_eq_{long,short}` and
  `peak_recovery_days_strategy_eq_{long,short}` retain the longest interval until strategy equity
  strictly exceeds its prior controller peak, including an unrecovered tail through the backtest
  end. Raw per-side `drawdown_worst_strategy_eq_{long,short}` and
  `drawdown_worst_mean_1pct_strategy_eq_{long,short}` are also available through opt-in bounded
  accumulators. The latter first retains each observed day's worst full-resolution controller
  drawdown and then applies the same logarithmic tail approximation used by the EMA-smoothed
  metric; only its partially selected cutoff bin is approximate. Exact Rust validation retains
  authority over both the proxy fill path and tail reduction. Single-coin, one-sided multi-coin,
  and fused shared-account dual-side
  multi-coin EMA Anchor and Trailing Martingale also support the global
  `strategy_eq_recovery_days_{mean,median,p95,p99,mean_worst_5pct,mean_worst_1pct}` metrics. The
  strategy kernels emit opt-in candidate-relative hourly strategy-equity samples plus mandatory
  initial and terminal/liquidation endpoints, and a bounded Metal postprocessor applies the same
  strict time-to-exceed and percentile/tail definitions as exact Rust. The hourly proxy is
  approximate; exact Rust validation and rolling drift gates remain authoritative. At an internal
  gap whose raw H/L/C values are all NaN, Metal mirrors exact Rust by advancing the tracked
  timeline with balance-only strategy equity while blocking fills, order generation, and unrealized
  PnL. Exact Rust classifies the same row as non-tradable without treating the internal gap as a
  delist during mandatory validation. Only the GPU optimizer operation explicitly requests this
  input exception; a persisted GPU backend setting does not relax other commands. At coarser candle
  intervals, complete NaN minutes are ignored inside a mixed aggregation bucket and an all-gap
  bucket remains a non-tradable gap; malformed non-gap prices cannot be hidden by a later valid
  minute in the same bucket. A gap must be strictly internal: first-valid candles and forced-delist
  endpoints remain finite positive prices, and the raw endpoint guard scales with the configured
  candle interval. Pending directional orders are cleared at a gap before the next valid candle.
  Normal CPU input validation stays strict.
  Finite non-positive, partially invalid, or float32-unrepresentable prices remain fail-closed
  for GPU screening. The single-coin coin-HSL ring coalesces realized-PnL components from the same
  candle and retains up to 8,192 event candles in the configured finite lookback; an overflow
  beyond that bounded capacity still fails closed with a conservative full-horizon recovery
  penalty. Independent dual-side multi-coin summaries remain
  fail closed for these metrics because they cannot reconstruct one shared portfolio-equity curve.
  Compatible suites may use the supported topologies.
- single-coin EMA Anchor and Trailing Martingale support auto-unstuck for long-only,
  short-only, hedge-mode dual-side, one-way, and compatible suite runs. One- and dual-side
  multi-coin runs and compatible suites also support auto-unstuck, including static per-coin
  overrides. Metal
  models the enable and EMA-gating toggles, tunable close percentage, EMA distance, loss allowance,
  and exposure threshold. It derives the allowance from a conservative all-history realized
  net-PnL peak, admits at most one least-stuck eligible position per portfolio, scales a
  losing close to its own allowance subject to exchange minimums, and lets that close compete with
  the position's WEL/TWEL reducer before ordinary closes consume the remaining realized-loss
  budget. The fused dual-side multi-coin kernel chooses globally across both directional surfaces
  using exact Rust's price-difference, symbol-index, and long-before-short tie ordering. Exact Rust
  remains authoritative for the configured rolling PnL lookback
- single- and multi-coin EMA Anchor and Trailing Martingale runs support bounded and legacy-raw
  `risk.we_excess_allowance_pct`, `risk.total_exposure_entry_gate_enabled`, and
  `risk.total_exposure_enforcer_threshold` across long-only, short-only, dual-side, and compatible
  suites. For multi-coin runs, allowance is applied to each symbol's dynamic or overridden WEL;
  the configured side-level bounded or legacy-raw mode applies to all symbols. Per-coin mode
  overrides remain outside the canonical config contract.
  Bounded mode limits only the added allowance: when the base WEL is at or below the side TWEL,
  the allowance cannot raise it past TWEL; an explicit base WEL already above TWEL is left
  unchanged. Legacy-raw mode applies the raw multiplier. The optional side-wide entry gate caps
  aggregate entries at TWEL times its positive threshold (never above raw TWEL). Disabling the gate
  permits aggregate entries beyond TWEL while each symbol remains subject to its allowed WEL
- `position_held_hours_mean`, `position_held_days_mean`, `positions_held_per_day`,
  `position_unchanged_hours_max`, and `position_unchanged_days_max` may be used for scoring and
  limits in single-coin and multi-coin runs. Metal counts each completed position and open tail,
  sums its holding duration, and tracks the latest fill independently for each coin and position
  side. Fused dual-side multi-coin kernels truncate those aggregates at shared portfolio
  liquidation; exact Rust validation remains authoritative
- `total_wallet_exposure_max` and `total_wallet_exposure_mean` may be used for scoring and limits
  in single-coin and multi-coin runs. Metal samples absolute net long-minus-short wallet
  exposure after every non-liquidating equity update, including flat zero-exposure samples, and
  reduces it with bounded maximum and online-mean accumulators. Fused dual-side multi-coin kernels
  maintain the shared minute-level net exposure series
- canonical USD account-equity metrics may use the same MPS surface as their strategy-equity
  counterparts while BTC collateral is disabled: `adg_usd`, `mdg_usd`,
  `sharpe_ratio_usd`, `sortino_ratio_usd`, `omega_ratio_usd`,
  `expected_shortfall_1pct_usd`, `calmar_ratio_usd`, `sterling_ratio_usd`,
  `drawdown_worst_usd`, and `drawdown_worst_mean_1pct_usd`, plus the available `_w_usd` weighted
  variants. `exposure_ratio_usd` and `exposure_mean_ratio_usd` combine proxy ADG with the maximum
  and mean total-wallet-exposure accumulators. The exposure ratios support single-coin and
  multi-coin runs, including fused dual-side kernels
- canonical USD ADG, MDG, weighted ADG, and weighted MDG per-configured-exposure metrics are
  supported for both long and short sides. The proxy divides the matching validated equity metric
  by each candidate's effective side `total_wallet_exposure_limit`, after applying any exact-last
  suite override, and returns zero for a zero-exposure side as the CPU analysis does
- canonical USD `equity_choppiness`, `equity_jerkiness`, and `exponential_fit_error` scoring and
  limits use the proxy's active daily closing-equity samples and the exact Rust formulas. Candidates
  without fills retain Rust's default value of `1.0` for all three metrics. Their weighted variants
  and `volume_pct_per_day_avg_w` apply the same ten trailing slices as exact Rust to the compact
  daily proxy series. Weighted volume excludes an ambiguous partial UTC cutoff day instead of
  admitting pre-cutoff fills; exact validation and rolling drift gates remain authoritative
- Trailing Martingale supports `risk.position_exposure_enforcer_enabled` and a tunable
  `risk.position_exposure_enforcer_threshold` for single-coin long, short, and dual-side runs,
  one- and dual-side multi-coin runs, and compatible suites. When current position exposure
  exceeds the allowance-adjusted WEL times the positive threshold, Metal gives the passive reducer
  precedence over the normal strategy close and sizes it strictly below that target. Static per-coin
  overrides may change both fields in one- and dual-side multi-coin runs. The fused kernel computes
  WEL and TWEL repair independently for each directional surface against the same pre-fill shared
  account snapshot, matching exact Rust order generation. EMA Anchor position-exposure repair
  remains fail closed because its exact Rust strategy path does not use this reducer
- single- and multi-coin EMA Anchor model the cumulative realized-loss gate, including entry and
  close fees, shared long/short loss-budget accounting, and lossy total-exposure repairs. For
  multi-coin runs, Metal executable-touch-sizes each TWEL and unstuck candidate, finalizes it with
  the ordinary strategy close, and considers the current largest finalized candidate globally
  across symbols and sides. A rejected candidate advances only that position to its smaller
  fallback. Accepted protective losses spend one shared allowance before ordinary closes are
  checked in canonical long-then-short symbol order; HSL panic closes remain exempt. These are
  generation-time decisions retained with the pending orders rather than reclassified at fill
  time. The proxy uses a conservative all-history loss envelope, so it may block a close that exact
  Rust admits after old PnL ages out of `live.pnls_max_lookback_days`. Single-coin and one-sided
  multi-coin Trailing Martingale permit the one selected auto-unstuck reducer to consume that same
  conservative budget, while ordinary and exposure-repair closes retain the stricter zero-loss
  envelope. Those Trailing Martingale restrictions avoid unsafe cross-side loss-budget reservation
  and per-candle enumeration of its recursive 500-rung close ladder. Exact validation applies the
  configured rolling allowance and remains authoritative
- BTC collateral remains disabled
- `backtest.filter_by_min_effective_cost` may be enabled or disabled. When enabled, Metal uses the
  projected initial-entry cost test with the effective wallet-exposure limit, including dynamic
  position counts and static per-coin wallet-exposure, allowance, and initial-quantity overrides.
  The screening proxy compares each coin against its highest executable minimum over the prepared
  window, rounds that threshold upward, and discounts the projected float32 product so boundary
  rounding cannot turn a just-below-threshold proxy projection into an admission. To remain
  conservative across float32 proxy versus float64 Rust path divergence, Metal uses the configured
  liquidation floor—not proxy balance—as the guaranteed cash-balance lower bound while the entire
  portfolio is flat and Metal has not rejected a candidate that exact Rust may still admit. For
  multicoin and dual-side single-coin runs, the first independently selected or arbitrated candidate
  set also exhausts this bound because later proxy/exact selection may diverge without a proxy fill.
  Once any of those events occurs, the liquidation floor bounds equity but no longer proves a lower
  bound for exact cash. Metal therefore keeps that uncertainty for the rest of the candidate
  backtest and immediately fails every later flat coin/side closed, even if its own portfolio remains
  or becomes flat again. These candidates are
  removed before Forager selection and one-way long/short arbitration; every open position remains
  managed. This supports single- and
  multi-coin, one- and dual-side EMA Anchor and Trailing Martingale runs and compatible suites.
  The all-history minimum and whole-portfolio-flat bound may produce proxy false negatives, which
  exact validation may admit. Runs that depend on filling several slots sequentially while earlier
  positions remain open may therefore accumulate more proxy/exact rank drift and can halt at the
  configured safety threshold. A finite positive `backtest.liquidation_threshold` is required
- `live.market_orders_allowed: false` for the complete strategy/risk surface. Single-coin EMA
  Anchor supports `true` for long-only, short-only, hedge-mode dual-side, one-way, and compatible
  suite scenarios, including HSL, one-sided minimum-effective-cost filtering, auto-unstuck,
  total-exposure repair, and realized-loss gating. Single-coin Trailing Martingale supports the
  same directional and position-mode topologies with HSL and one-sided minimum-effective-cost
  filtering. Entry and close retracement bounds may cross the recursive/trailing mode boundary:
  each candidate independently selects recursive mode at a nonpositive base or trailing mode at a
  positive base. An extremely small positive float64 value that would underflow during Metal
  packing is raised to the smallest normal float32 solely in the screening proxy so its mode sign
  agrees with exact Rust.
  Auto-unstuck, position- and total-exposure repair, and realized-loss gating may remain enabled or
  tunable. The Metal proxy classifies each generated order against the
  current candle close using
  `live.market_order_near_touch_threshold`, retains that execution intent for the pending order,
  and fills promoted orders on the next valid candle at its adversely slipped, directionally
  rounded close with the taker fee. Market closes and short entries are resized at the executable
  touch before they are retained. Protective reducers are independently classified and resized
  before reducer selection and aggregate allocation; adverse slippage and taker fees participate
  in realized-loss gating. For recursive Trailing Martingale entries, every ladder rung is
  classified against the immutable generation market snapshot only when the original passive rung
  is strictly next-candle reachable, then streamed through the strict total-exposure entry gate at
  each limit price or executable market touch. Immutable strategy sizing uses its wallet-exposure
  allowance separately from that portfolio gate; a partially retained TWEL boundary rung ends the
  nearest-order prefix so no farther rung can reappear. For recursive Trailing Martingale closes,
  exact passive next-candle reachability still decides whether Rust emits only the next close or
  expands the immutable ladder. Market policy cannot expose an unexpanded suffix. Once expanded,
  every merged price group is classified and executable-touch-sized against the immutable
  generation market, then aggregate-trimmed to the position and ordered with any protective
  reducer before next-candle adverse slippage, taker fees, and realized-loss gating are applied.
  Strategy WEL reachability remains part of the pre-gate expansion decision even when that reducer
  is subsequently rejected. Its passive quantity seeds the remaining immutable rungs before any
  market minimum resize, and a WEL sharing the following ordinary rung's quantized price is merged
  into that ordinary group.
  Multi-coin EMA Anchor supports ordinary market entries and ordinary strategy closes for
  long-only, short-only, and fused long+short runs, including compatible suites, static coin
  overrides, forager selection, the strict total-exposure entry gate, TWEL repair, auto-unstuck,
  realized-loss gating, and HSL. The proxy stores generation-time market intent, fills it on the
  next valid candle with adverse directional slippage and the coin's taker fee, uses
  executable-touch minimum sizing for short entries and all closes, and accounts for market-touch
  cost while allocating the portfolio entry cap. Protective reducers participate in the same
  finalized-quantity ordering, shared loss budget, and per-position fallback used by exact Rust;
  their market slippage and taker fees are included in the projected loss. Static coin overrides
  may enable or tune unstuck. Multi-coin Trailing Martingale supports ordinary market entries and
  ordinary strategy closes for long-only, short-only, fused long+short, hedge-mode, one-way, and
  compatible suite runs. Entry and close retracement bounds may cross the recursive/trailing mode
  boundary, and static coin overrides may select either mode per coin. It uses
  the same retained generation-time intent, next-valid-candle adverse fill, taker-fee,
  executable-touch minimum sizing for short entries and all closes, and portfolio entry-cap
  accounting contract. Passive next-candle reachability alone exposes a recursive close suffix;
  market promotion cannot expose an otherwise unexpanded ladder. Every emitted duplicate-merged
  group is classified and minimum-sized against the immutable generation market before aggregate
  trimming to the position. Recursive entries likewise expose their suffix only after strict
  passive first-rung reachability, retain immutable strategy sizing, classify each emitted rung at
  the generation market, and stage one order when entry cooldown is positive. With
  `risk.total_exposure_entry_gate_enabled: true`, the proxy performs one deterministic global merge
  of first and recursive suffix rungs across coins, retains nearest orders, and can crop one
  minimum-valid boundary strictly below the TWEL cap. HSL, static coin overrides, and forager
  selection remain supported. Position- and total-exposure repair plus the one globally selected
  auto-unstuck reducer may promote to market, executable-touch-size before finalized reducer
  selection, and retain adverse slippage plus taker fees. Static coin overrides may enable or tune
  unstuck; recursive-grid reconstruction keeps the independently generated ordinary strategy
  ladder when market sizing enlarges the external reducer. Realized-loss gating composes with these
  market paths: the proxy projects adverse slippage and taker fees and keeps a conservative
  zero-loss envelope for ordinary and exposure-repair closes, while exact Rust applies the shared
  peak-balance allowance to every validation.
- single-coin runs may include invalid candles after the selected coin's final valid candle. A
  shorter tail remains non-tradable, excludes the open position's unrealized PnL, and continues
  balance-only equity and elapsed-time accounting through the prepared endpoint. When at least
  1,400 prepared candles follow the final valid candle, Metal mirrors exact Rust's forced delist:
  it closes any open long and short at that candle with adverse market slippage, directional price
  rounding, and the taker fee, records the panic fill and position/fill aggregates, clears pending
  orders, and then continues the same balance-only tail. HSL-enabled runs also continue hard-stop
  sampling, rolling-PnL expiry, tier accounting, and restart checks without treating stale orders
  as blocking
- multi-coin runs may include staggered invalid tails while at least one prepared coin remains
  valid through the endpoint and at least one candle whose packed float32 H/L/C values remain
  finite and positive inside the coins' declared first-to-last-valid ranges covers every timestep
  after coverage begins. Tailed coins are non-tradable, contribute no unrealized PnL, and cannot
  leave stale orders blocking HSL. When a tail crosses exact Rust's 1,400-candle forced-delist
  boundary, Metal closes that coin's long and then short with the same adverse market execution,
  taker-fee, panic-loss, fill, duration, and realized-PnL accounting, then clears both sides' pending
  orders for that coin. Each such coin's final H/L/C must remain finite and positive after float32
  packing; an unrepresentable final candle fails before dispatch. Dynamic tradability,
  portfolio/coin HSL, equity, and elapsed-time accounting continue on the surviving timeline
- multi-coin histories may also contain declared all-invalid gaps between disjoint coin histories, a
  tail after every coin has ended, or internal timestamps where every declared coin's raw H/L/C is
  NaN. Once portfolio equity tracking starts, Metal continues exact Rust's balance-only
  equity, exposure, daily, recovery, and elapsed-time accounting through those periods. Tradability
  observed only before the warmup/requested-start guard does not activate tracking inside a later
  gap; the next eligible coin does. Per-coin candle validity still blocks fills, order generation,
  and unrealized PnL. Finite but non-positive, partially invalid, or float32-unrepresentable H/L/C
  values remain fail-closed for GPU screening. HSL-enabled coins retain the
  stricter contiguous-candle requirement documented above, and a forced-delist endpoint must remain
  finite and positive after float32 packing because it supplies an executable close

#### Deliberate current limitations

The following boundaries are intentional rather than silent fallbacks:

- `trailing_grid_v7` is outside the Apple MPS implementation. Use `optimize.backend: "pymoo"` or
  `"deap"` for it; GPU optimization never substitutes EMA Anchor or Trailing Martingale behavior.
- `backtest.btc_collateral_cap` must be zero. Positive BTC collateral changes the simulated
  account state and is not approximated by the screening proxy; use a CPU optimizer when it is
  required. BTC-denominated scoring with a zero collateral cap remains supported as described
  below.
- Each prepared single run or suite scenario is capped at 64 coins. Split the scenario universe or
  use a CPU optimizer for a larger portfolio.
- Only metrics and limits represented by the MPS screening surface are accepted. An unsupported
  metric is named in the startup error and must be removed or run on a CPU backend. Exact Rust
  validation does not turn an unmodeled proxy objective into a safe evolutionary search.
- Suite scenario override paths require explicit proxy shadow semantics. Unmodeled paths fail
  before historical-data preparation rather than being applied only to exact validations.

GPU startup checks the strategy, zero-collateral contract, suite override paths, optional PyTorch
installation, and MPS availability before historical-data preparation. The prepared-data stage
then enforces the 64-coin ceiling and topology-specific requirements. CPU optimizers, backtesting,
and live operation do not import or probe the optional GPU runtime.

Unsupported combinations fail before optimization begins. Dual-side multi-coin EMA Anchor and
Trailing Martingale use fused shared-account Metal kernels in hedge and one-way modes. Every
accepted metric still comes from the unchanged exact Rust portfolio backtest, and classification,
rank, and drift gates halt material disagreement. Unmodeled non-bot suite scenario overrides are
not silently approximated by this release.

For a supported suite, each Metal candidate is dispatched across every prepared scenario. The GPU
path then calls the same canonical suite reducer and scenario-selection logic as the CPU optimizer
for aggregate reducers, named-scenario objectives, and named-scenario or suite-reduced limits.
Every exact validation still runs the unchanged Rust backtest for every scenario; only those exact
suite metrics enter `all_results.bin` and the Pareto front. A scenario may select a different coin
subset, date window, individual exchange, canonical combined multi-exchange dataset, or a strict
multi-exchange subset represented by one prepared dataset per exchange. In the last case, Metal
evaluates each prepared exchange independently and the GPU path combines those proxy analyses with
the same canonical per-scenario metric statistics as the CPU suite before applying suite reducers.
Scenario `overrides` require
explicit candidate-shadowing behavior in the proxy. This slice models `bot.long` and `bot.short`
overrides: the canonical exact
suite evaluator still applies them last, after candidate materialization, while each scenario's
Metal proxy shadows the corresponding candidate parameters with the same effective values. Every
overridden scenario is rechecked against the directional GPU scope, so an override cannot silently
enable an unsupported exposure-repair path, an invalid
position count, or another unsupported behavior. In a multicoin suite, a scenario with fewer coins must
keep the effective `n_positions` range within that subset, either through common bounds or an
explicit scenario override. Metal uses the strategy-specific single-coin or multicoin kernel
independently for each scenario, then feeds all results to the same suite reducer. Scenario-local
`coin_overrides` for supported
multi-coin EMA Anchor and Trailing Martingale runs, `backtest.starting_balance`,
`backtest.maker_fee_override`, `backtest.taker_fee_override`,
`backtest.market_order_slippage_pct`, `backtest.filter_by_min_effective_cost`,
`backtest.liquidation_threshold`, `live.pnls_max_lookback_days`,
`live.forager_score_hysteresis_pct`, `live.hedge_mode`, `live.hsl_signal_mode`,
`live.market_orders_allowed`, and `live.market_order_near_touch_threshold` are accepted because
every scenario proxy consumes them through the canonical backtest payload and then passes the same
fail-closed scope checks. Combined scenario `coin_sources` use the same prepared per-coin candles
and market settings as exact Rust; their resolved OHLCV and market-settings exchanges are part of
checkpoint
identity. Other non-bot paths remain rejected until their proxy semantics are modeled. The
effective external suite definition and any `--scenarios` filter are
stored in the run contract and checkpoint identity, with dynamic scenario dates resolved to the
prepared concrete dates. The checkpoint signature also records each scenario's ordered effective
coins, side topology, and prepared candle window, so resume fails closed if preparation changes.

Ordinary `-t/--start` seeding and fine-tuning with `-ft/--fine-tune-params` use the same optimizer
shape as the CPU backends. When `-t` and `-ft` are combined, the GPU population includes the
discrete anchor id alongside the selected tunable bounds. Anchor-fixed values are supplied to the
screening proxy before each candidate's tunable values, while the unchanged exact Rust path
materializes the same anchor and tunable vector. The complete range across all anchors is checked
against the GPU scope: an anchor cannot change enabled sides or introduce unsupported risk
behavior. Base-config runtime policy fields still win over anchor configs as described in
[Fine-Tuning Specific Parameters](#fine-tuning-specific-parameters).

Starting configs are normalized, clamped, quantized, and deduplicated through the shared optimizer
loader before either backend evaluates them. CPU optimization retains its exact-all contract: every
deduplicated seed is evaluated by the exact Rust backtester before population trimming. GPU
optimization uses `optimize.gpu.seed_bootstrap`: the default `auto` mode also exact-evaluates every
seed when the pool contains at most `max_exact` entries (128 by default). Larger pools are screened
once by the full-history Metal proxy; a deterministic, constraint-aware mixture of diverse proxy-
Pareto members, per-objective extremes, and off-front probes is then exact-evaluated, capped at
`max_exact`. The authoritative Pareto archive contains only those exact results, so screened mode
does not claim that the unevaluated remainder of the seed pool has an exact Pareto classification.
Exact-selected seeds are placed first in the initial GPU population, followed by proxy-diverse
seeds and then random candidates. Their exact objective values are never inserted into the proxy
NSGA-II fitness matrix.

The V8 `optimize.enable_overrides` values `mirror_short_from_long` and
`lossless_close_trailing` are applied to Metal candidates in the same order as exact candidate
materialization. Mirroring may be used with the supported single-coin directional scopes; short
genes that exact materialization overwrites are omitted from the proxy search dimensions.
`lossless_close_trailing` is available only with `strategy_kind: trailing_martingale`. The legacy
`forward_tp_grid` and `backward_tp_grid` values remain unsupported because the GPU backend does not
support `trailing_grid_v7`.

`optimize.fixed_runtime_overrides` is also applied in exact candidate-materialization order: after
anchor and tunable values, and before `optimize.enable_overrides`. A fixed override that shadows a
supported GPU optimizer bound removes that gene from the Metal search and supplies the fixed value
to both proxy screening and exact Rust validation. The effective overridden config is checked
against the same fail-closed GPU scope, so an override cannot silently enable unsupported behavior.
Exact finalized boundary configs are validated before either optimizer backend starts. Config
normalization preserves user-defined dotted leaf paths, rejects malformed or unknown paths,
rejects aliases that resolve to the same setting, and rejects mapping-level replacements. When a
fixed value disables dependent trailing-martingale parameters, the Metal search removes and
hash-canonicalizes those dead genes using the same rule as exact candidate materialization.

Proxy scoring and limits are likewise fail-closed. The supported strategy-equity surface includes
ADG, MDG, Sharpe, Sortino, Omega, Calmar, Sterling, expected shortfall, worst and worst-1%
drawdown, mean underwater percentage, recovery distributions and position-held duration,
position-unchanged duration, initial-entry balance percentage for each side, volume per active day,
total-wallet-exposure maximum and mean, USD exposure ratios, backtest completion, and weighted
variants of ADG, MDG, Sharpe, Sortino, Omega, Calmar, and Sterling. With BTC collateral disabled,
the corresponding canonical USD account-equity names share this strategy-equity surface. The GPU
proxy also supports BTC-denominated ADG, MDG, Omega, equity shape, unweighted exposure
ratios, per-side exposure-normalized ADG/MDG, and the existing weighted
ADG/MDG/Omega and equity-shape variants. It converts the compact USD daily closing-equity surface
with the canonical prepared BTC/USD price at each UTC day end, including a candidate-specific
final endpoint after early liquidation. For BTC Sharpe, Sortino, expected shortfall, worst and
worst-1% drawdown, Calmar, and Sterling, Metal conditionally retains a
synchronized BTC-equity surface containing each UTC day's close, minimum, and worst full-curve
drawdown. Weighted BTC Sharpe, Sortino, Calmar, and Sterling require suffix-local intraday minima
and drawdown reconstruction and remain fail-closed. Runs that
do not request one of these intraday-risk metrics keep the smaller existing kernel ABI and output
buffers. Weighted BTC exposure ratios still require
suffix-local exposure series and remain fail-closed.
Unweighted `equity_balance_diff_neg_{max,mean}_{usd,btc}` and
`paper_loss{,_mean}_ratio_{usd,btc}` are supported through an opt-in full-resolution accumulator.
The BTC balance baseline is rebased at each proxy fill; when the first fill establishes that
baseline, the kernel replays earlier tracked BTC prices without repeating the strategy simulation.
The retained negative maxima and means supply the paper-loss denominators and retain tight parity
coverage.
The prepared BTC series must contain at least one sample in every covered UTC day; unusually coarse
intervals which skip a whole UTC day fail closed for BTC scoring.
Positive `backtest.btc_collateral_cap` remains unsupported and fails before GPU optimization.
Daily USD equity choppiness, jerkiness, and exponential fit error are reduced from that same active
daily closing-equity surface with Rust's no-fill defaults and short-series behavior.
Gross close-fill loss/profit ratios are supported both in aggregate and separately for long and
short. `pnl_ratio_long_short` uses each side's signed realized PnL and Rust's neutral `0.5` result
when combined signed PnL is zero. Directional kernels
retain the four gross side sums, while one-sided and dual-side multi-coin dispatches preserve the
same side partition before reduction.
Full-run fill activity is supported for single-coin and multi-coin topologies through combined
fills per day, entry-to-close ratio, combined per-configured-position-slot rate, active-day ratio,
active-symbol count, and top-symbol fill share. Metal classifies every actual proxy fill by role,
position side, and—when needed—coin. All rates use the same first-to-last analyzed-equity timestamp
span as Rust. Fused dual-side kernels retain the shared intraday-liquidation cutoff.
Initial-entry allocation uses the candidate's effective position count and the same
first-coin strategy/allowance override precedence as exact Rust. Fill-gap longest, p95, and
time-weighted-mean metrics are also supported. Metal coalesces multiple fills in the same candle and
records positive inter-candle gaps
in a 128-bin logarithmic histogram; the proxy decodes each occupied bin with a float32-safe upper
edge and adds the exact leading and trailing gaps. This deliberately overestimates the minimizing
fill-gap summaries when exact Rust has same-candle zero gaps or a value inside a histogram bin.
Trailing Martingale also supports `entry_interval_hours_{mean,median,p95,p99,max}` for single-coin
and multi-coin, long-only, short-only, and fused long+short runs. Metal records gaps between
normal initial entries independently for each coin and position side. The proxy mean and maximum
come from direct Metal accumulators with integer-safe event counts; median, p95, and p99 use
conservative upper edges from a bounded logarithmic histogram. EMA Anchor emits no
`EntryInitialNormal` order types,
so these metrics retain exact Rust's canonical zero values without allocating the optional output
surface. Runs that do not request an entry-interval metric keep their existing kernel ABI and
dispatch cost.

Exact Rust metrics remain authoritative. Metrics intentionally kept for exact analysis rather than
proxy optimization include raw gain, realized-PnL growth/risk, positive equity-balance divergence,
completed-only account-equity recovery, raw or split fill counts/rates, raw HSL event counts and
absolute loss totals, the self-relative high-exposure duration family, and the legacy global
recovery/profit aliases. Requests for these metrics as GPU objectives or proxy-side limits fail
before MPS setup. They remain
available in normal Rust backtests, exact optimizer validation output, and CPU optimization.

The backend is hybrid rather than a replacement backtester:

1. pymoo NSGA-II proposes large normalized candidate batches.
2. A Rust-owned Metal screening program evaluates every candidate against candle data resident on
   MPS; Python only prepares buffers and dispatches the program. EMA-anchor and
   trailing-martingale use separate single-coin and multi-coin kernels. Directional runs keep
   separate long/short indicator,
   trailing, and position state with one shared balance and the exact Rust fill ordering. Python
   also precomputes strict high/low crossing boundaries as integer price ticks so float32 Metal
   comparisons preserve Rust's decimal-tick fill decisions. Candle-derived touches are classified
   from the original float64 data. EMA uses Rust-compatible directional ticks. Trailing-martingale
   uses those ticks to choose the controlling raw/target value before float32 can collapse nearby
   prices, then mirrors Rust's directional entry finalization and nearest-tick close finalization.
   The multi-coin trailing-martingale screening kernel retains per-coin EMA, volatility, trailing,
   position, cooldown, and pending-order state plus shared portfolio allocation. It stages one
   entry and close per coin per candle; exact Rust validation remains responsible for authoritative
   recursive same-candle ladders, and the normal constraint/rank/drift gates halt if that screening
   approximation stops ordering candidates reliably.
   Raw-touch close minimum quantities are computed from the original float64 price before close-
   price finalization, and their ordering relative to aligned quantity steps is retained across
   float32 transport. Tick-aligned computed targets remain on their exchange tick; residual
   float32 arithmetic drift is measured by exact validation.
   For one-sided single-coin Trailing Martingale runs, the backend automatically selects a
   Rust-owned long-only or short-only Metal variant, including supported coin-HSL configurations.
   Compile-time side constants remove the opposite-side fill, indicator, trailing, and HSL update
   paths. HSL lifecycle and drawdown telemetry is compiled only when a requested proxy objective or
   limit consumes it; ordinary strategy-equity and fill metrics keep the controller behavior but
   omit those unused accumulators. Dual-side runs retain the generic kernel. The selected variant
   is logged as `GPU MPS specialized kernel selected`; exact Rust validation and the normal drift
   gates remain authoritative.
3. In suite mode, the same candidate batch is screened once per scenario and reduced with the
   canonical suite scoring and limit contract.
4. Diverse proxy-front candidates and broad drift probes are sent to the unchanged Rust backtester.
5. Only exact Rust results enter `all_results.bin` and the persisted Pareto front.
6. Rolling rank and constraint-agreement gates independently stop the run if proxy/exact agreement
   falls below `drift_halt` after sufficient evidence. Constraint classification is monitored over
   all validations and independently for proxy-front candidates and broad probes. An isolated
   disagreement is retained as drift evidence rather than aborting immediately; the exact Rust
   result remains authoritative and an exact-infeasible candidate cannot enter the Pareto front.
   Feasibility disagreements are not rank-comparable and already count against the constraint
   gates, so rank correlation uses only classification-agreeing samples and requires at least eight
   comparable broad probes. Window and exact-budget validation reserve enough total probes to retain
   those eight whenever the configured probe constraint-agreement gate has not already failed.

`optimize.iters` remains the number of evolutionary exact Rust validations. Any exact seed-
bootstrap evaluations are additional and are reported separately. GPU screening counts and
throughput are also logged separately. `n_cpus` controls the exact-validation worker pool; MPS
device scheduling is managed by Metal.

GPU-specific settings live under `optimize.gpu`:

```json
{
  "optimize": {
    "backend": "gpu",
    "gpu": {
      "auto_lean_parallelism": true,
      "batch_size": null,
      "max_dispatch_candidate_bars": null,
      "checkpoint_interval_seconds": 5.0,
      "drift_halt": 0.6,
      "drift_min_samples": 32,
      "drift_probes": 4,
      "drift_window": 128,
      "exact_workers": 0,
      "max_pending_exact": 0,
      "population_size": null,
      "seed_bootstrap": {
        "max_exact": 128,
        "mode": "auto"
      },
      "successive_halving": {
        "enabled": false,
        "history_fractions": [0.25, 0.5, 1.0],
        "min_survivors": 64,
        "survival_fraction": 0.5
      },
      "validate_per_generation": 8
    }
  }
}
```

The CPU-side NSGA-II proposal stage uses the same `optimize.pymoo.shared` crossover, mutation, and
duplicate-elimination controls as the ordinary pymoo optimizer.

- `population_size` is the NSGA-II proxy population. The general default is 1024 so long-history
  runs reach their first exact Rust validation batch four times sooner than the former 4096
  default. `null` or an omitted key requests this automatic default. On a detected Apple M3 family
  device, with all three sizing keys left automatic, `auto_lean_parallelism` raises the effective
  population to 2304 only for a proven one-sided Trailing Martingale kernel with no coin overrides
  and with HSL, unstuck, exposure reducers, market orders, the realized-loss gate, recursive
  entry/close modes, volatility weights, and opt-in metric feature paths all compiled out. Metric
  surfaces needing BTC analysis, entry intervals, recovery distributions, equity-balance
  divergence, HSL diagnostics, or other optional proxy output paths retain the general envelope.
  This supplies enough resident work to hide divergent replay latency on the benchmarked target.
  Set `auto_lean_parallelism` to `false`, or set any of `population_size`, `batch_size`, or
  `max_dispatch_candidate_bars` to a number (including the ordinary default number), to retain the
  configured sizing unchanged.
- `batch_size` is the requested upper bound on candidates per MPS dispatch. Because Apple Silicon
  shares the GPU with WindowServer, the backend transparently splits a batch when its
  candidates-by-candles-by-coins-by-enabled-sides workload would make one Metal command buffer too
  long. The effective dispatch size is logged as `GPU MPS dispatch safety cap active`; population
  size, candidate order, NSGA-II ask/tell semantics, and the number of proxy evaluations are
  unchanged. Ctrl+C is polled between those bounded dispatches. If it arrives during a generation,
  that incomplete ask/tell transaction is discarded and the last complete checkpoint is retained.
  A topology whose single candidate already exceeds the safety envelope fails closed with guidance
  to shorten the date range or reduce its coin count.
- `max_dispatch_candidate_bars` sets that MPS work envelope. The default is 1 billion, allowing
  roughly 512 candidates per dispatch across 1.95 million one-sided candle bars on a dedicated
  optimization Mac. Set it to `500000000` for the former conservative behavior when desktop
  responsiveness is more important than GPU throughput. Larger values increase the time before
  Ctrl+C can be observed and may make the shared display GPU temporarily unresponsive;
  `batch_size` remains an independent upper bound.
  When `auto_lean_parallelism` detects an Apple M3 family device and proves the lean one-sided
  Trailing Martingale shape described above, it raises the effective envelope to 4.5 billion
  together with the 2304 population. That exact shape completed approximately 4.45 billion
  candidate-bars in about 20 seconds on the supported M3 benchmark. Other Apple Silicon families
  and all heavier shapes retain the 1-billion default; the optimizer does not apply the wider
  envelope to coin-overridden, HSL, multicoin, suite, market-order, reducer, recursive-mode, or
  active-volatility kernels, nor to kernels with optional metric feature paths enabled.
- `seed_bootstrap.mode` controls `-t/--start` handling. `auto` exact-evaluates all deduplicated seeds
  up to `seed_bootstrap.max_exact`, then switches to full-history proxy screening plus capped exact
  validation for larger pools. `exact` forces exact evaluation of every seed even above the cap;
  `screened` always performs proxy screening and validates at most the cap; and `legacy` restores
  the former behavior of copying seeds directly into the first proxy population without an
  authoritative bootstrap archive. Bootstrap exact evaluations are recorded in `all_results.bin`
  and the Pareto store but do not consume `optimize.iters`, which remains the subsequent
  evolutionary exact-validation budget. Checkpoints preserve incomplete bootstrap plans and can
  recover a seed result durably flushed immediately before a restart. Anchored fine-tune context
  is checkpoint-owned as well, so resume does not require the original starting-config files.
- `successive_halving.enabled` opts a non-suite, single-coin Trailing Martingale run into
  progressively longer recent-history suffixes. The default `history_fractions` are 25%, 50%, and
  100%, measured backwards from the configured end date; each partial suffix receives the normal
  indicator warmup immediately before its scoring boundary. After each partial rung,
  constraint-aware Pareto selection retains `survival_fraction` (50% by default), subject to
  `min_survivors` (64 by default). With the default 1024 population this means 1024 candidates on
  the most recent 25%, 512 on the most recent 50%, and 256 on full history. Shorter rungs use the
  same 500 million candidate-bars safety envelope, so they can safely dispatch more candidates
  together; the cap itself is not raised. Partial-rung rows are marked ineligible in NSGA-II, and
  only the full-history survivors may enter exact Rust validation, proxy-front selection, broad
  probes, or drift evidence. The ladder remains disabled by default because it trades some search
  breadth for throughput and deliberately biases early filtering toward recent market behavior.
  The suffix-window semantics are included in checkpoint identity, so checkpoints made with the
  former prefix behavior do not resume. EMA Anchor, multicoin, and suite runs fail closed if this
  opt-in is requested.
- `validate_per_generation` caps exact candidates selected from each proxy generation.
- `drift_probes` reserves at least part of that validation budget for candidates away from the
  proxy front.
  If the complete feasible proxy front occupies nearly the whole population, the optimizer uses
  every genuinely off-front candidate available and fills the remaining exact quota with diverse
  true-front candidates. Front members are never relabeled as broad probes. The separate broad-
  probe gates activate only after enough truthful off-front evidence has accumulated; allocation
  shortfalls and recovery are logged.
- `drift_window`, `drift_min_samples`, and `drift_halt` configure the rolling rank and optimizer-
  limit classification safety gates. Broad-probe Spearman correlation plus aggregate,
  proxy-front, and broad-probe constraint agreement must each remain at or above `drift_halt`.
  `drift_halt` must be greater than zero and at most one.
  At least eight samples of a validation class are required before its independent low agreement
  can halt a run, so `drift_window` and `optimize.iters` must be large enough to retain and reach
  eight true proxy-front validations even when the complete feasible proxy front contributes only
  one candidate per generation. Off-front validations stay truthfully classified as broad
  probes rather than being relabeled as front evidence, and scarce or duplicate probes give their
  unused exact slots back to true-front candidates. Front membership is carried independently
  through exact-result persistence and resume recovery. If duplicate filtering leaves no novel
  proxy-front candidate, the optimizer revalidates a previously exact current-front member; if its
  exact job is still in flight, selection waits for that job first. This preserves truthful front
  evidence without relabeling broad probes, and repeated front disagreements still activate the
  rolling front gate. `drift_probes` must remain below
  `validate_per_generation` so each generation requests proxy-front safety evidence. A partial
  final validation batch scales its reserved probe count down proportionally.
- `exact_workers: 0` inherits `optimize.n_cpus`; a positive value overrides it for this backend.
- `max_pending_exact: 0` defaults to twice the exact-worker count.
  It must be at least `validate_per_generation` so throttling cannot change the configured
  proxy-front/broad-probe evidence allocation; the backend waits for that capacity before
  screening another generation.
- `checkpoint_interval_seconds` bounds generation-level optimizer-state checkpoint writes. Exact
   result batches are checkpointed immediately, and each durable result carries the proxy/exact
   safety evidence needed to recover if its flush outruns the companion checkpoint. A final
   evidence-budget check applies to fresh and resumed runs and includes recovered class membership,
   the rolling-window suffix, discarded pending work, and all full or partial validation batches.
   Exact worker results are consumed in submission order even if workers finish out of order,
   preserving the modeled batch sequence. Resume fails closed if the mandatory proxy-front gate can
   no longer reach its minimum sample count. Broad probes remain opportunistic across restart, just
   as in an uninterrupted run: recovered truthful probes are retained, and their independent gates
   activate only after enough off-front evidence exists. A final checkpoint is always written on
   successful completion. Checkpoint identity includes the complete optimizer shape (including
   fixed and dormant dimensions, quantization, runtime optimizer overrides, and effective NSGA-II
   population/variation policy) plus every prepared proxy's effective execution contract:
  strategy and side topology, ordered coins, hashes of prepared candle values and timestamps,
  starting balance, valid/trade windows, fully resolved fixed proxy parameters, liquidation and
  exposure policy, finite coin-HSL rolling capacity where applicable, resolved maker/taker fees and
  market settings, and other modeled execution settings.
   Suite checkpoints record that contract independently for every scenario. Changing any of these
   inputs makes resume fail closed instead of mixing evolutionary state from incompatible runs.
   Checkpoints written before this expanded identity contract are intentionally incompatible.

The proxy is a float32 ranking model, not an authoritative simulator. Exact Rust metrics and configs
remain the only stored optimization results. The screening source is owned and exported by the
Rust extension; it does not replace or modify the exact Rust backtester. Credit: the Torch
metric-reduction work was adapted from RustyCZ's Passivbot GPU branch at commit `7c529bc73`; the
MPS Metal integration and hybrid validation gates are specific to this implementation.

#### Profiling Apple MPS optimization

Set `PASSIVBOT_GPU_PROFILE=1` to emit structured `[gpu-profile]` JSON records. Profiling is disabled
by default because its synchronization points deliberately trade throughput for trustworthy phase
boundaries.

```bash
PASSIVBOT_GPU_PROFILE=1 passivbot optimize path/to/public-or-local-config.json
```

Each generation record separates NSGA ask/tell and orchestration from proxy work, exact-validation
queue/wait and worker time, result persistence, and checkpoint writes. Each prepared proxy also
reports candidate materialization and Metal parameter packing, upload/buffer clearing, runner-local
cold compilation versus warm library lookup, kernel execution, device-to-host transfer, metric
reduction, and remaining host overhead. Shape metadata includes requested and actual dispatch batch
sizes, dispatch chunk and strategy-kernel counts, candidate-bars, candle/coin/side counts, requested
optional metric features, dispatch-proven Trailing Martingale specializations, and cold/warm
dispatch counts. Single-coin profiles also report how many candidates terminated before the end of
the evaluated history, estimated candidate-bars after termination, the corresponding fraction of
total candidate-bars, and terminal-step p50/p90. These estimates make the ceiling for a future
early-exit optimization visible without adding work to unprofiled runs. Exact Rust evaluation remains
authoritative; profiling fields are diagnostic log output and are not added to retained optimization
results. Chunked proxy generations that run for at least 30 seconds also emit rate-limited ordinary
progress with completed chunks and candidates, elapsed time, and ETA. This progress remains
available without enabling profiling or adding synchronization points.

Single-coin Trailing Martingale runners inspect the packed rows in each dispatch before choosing a
cached Metal library. When every enabled-side candidate has positive entry or close retracement,
the corresponding recursive grid path is compiled out. When every enabled-side candidate disables
entry retracement, the entry path instead compiles out trailing trigger and retracement work while
retaining recursive ladder expansion. Mixed entry modes keep the generic path. When every
enabled-side candidate disables the position/total-exposure enforcers and auto-unstuck, those
reducer paths are compiled out as a group. Fixed run settings similarly specialize ordinary market
execution and the realized-loss gate. When all eight entry/close threshold and retracement
volatility weights are exactly zero for
every active row, the selected kernel also omits the 1-minute and 1-hour volatility EMA state,
candle-range loads, and volatility-weight arithmetic. Any nonzero, non-finite, or mixed row keeps
the full volatility path. A mixed dispatch keeps each relevant full path, so this changes compiled
work rather than proxy semantics. A run may record an additional cold compile if its dispatch
feature shape changes.
HSL topology likewise considers enabled sides only; an HSL setting left on an exposure-disabled
side does not make the active side pay for an HSL controller.

For comparable local MPS measurements, first confirm another optimizer is not using the device,
then run each case in a fresh process. The harness uses only fixed-seed, in-memory synthetic candles
and candidate matrices; it never reads exchange credentials, local cache data, configs, or prior
optimization results.

```bash
passivbot tool gpu-proxy-benchmark --case ema-single-long
passivbot tool gpu-proxy-benchmark --case tm-single-long
passivbot tool gpu-proxy-benchmark --case tm-single-long-entry-ladder
passivbot tool gpu-proxy-benchmark --case tm-single-long-static-close
passivbot tool gpu-proxy-benchmark --case tm-single-long-close-ladder
passivbot tool gpu-proxy-benchmark --case tm-single-long-hsl
passivbot tool gpu-proxy-benchmark --case ema-multicoin-overhead
passivbot tool gpu-proxy-benchmark --case ema-multicoin-overrides
# Hold one candidate matrix constant while measuring dispatch chunking:
passivbot tool gpu-proxy-benchmark --case ema-single-long \
  --candidates 4096 --dispatch-batch-size 1024
```

The single-coin cases default to 60,000 one-minute candles; the overhead-sensitive multicoin cases
default to 4,320 candles and eight coins. Every report records the seed and workload shape, cold
compile/run timing, a fixture hash, and warm p50 across five repeated runs, including
candidates/second, kernel time, maximum chunk wall time, dispatch count, device transfer, and host
overhead. The HSL case uses the same fixed candles and candidate generation as ordinary Trailing
Martingale while enabling a deterministic long-side HSL lifecycle with the production-equivalent
30-day finite coin-PnL lookback in `coin` signal mode. It derives optional HSL diagnostic
compilation from the same requested metric surface as the production proxy; the default case
requests no HSL-only diagnostics. Reports record the effective lookback in bars and numeric
signal-mode ID. Use identical arguments and immutable commits for before/after comparisons. The
harness calls the production proxy evaluation
path, including its full output transfer, metric reduction, and result materialization. Run one
`--case` per process when comparing cold compilation; `--case all` is convenient for smoke checks,
and shared-cache hit/miss state still keeps later cold/warm labels accurate.
`--dispatch-batch-size` defaults to `--candidates`; setting it lower preserves the fixed candidate
matrix and reports the actual chunk and strategy-dispatch counts, which isolates dispatch
saturation from population-size changes.
The static-close and close-ladder cases hold the Trailing Martingale fixture constant while
switching only the recursive close inputs. Reports include the number of candidates whose fixed
parameters can emit multiple close rungs. When every active candidate side disables WEL/TWEL
enforcers and auto-unstuck, the directional kernel skips reducer candidate construction and
performs the one ordinary close-grid allocation directly; exact Rust validation remains
authoritative.
The ordinary and entry-ladder Trailing Martingale cases likewise hold candles, candidate count,
seed, and all other parameters constant while switching entry retracement from positive to zero.
Reports include the number of recursive-entry candidates. Homogeneous recursive-entry dispatches
select the recursive-only Metal variant; mixed entry modes retain the generic kernel.

### Pymoo Configuration

Pymoo-specific settings live under `optimize.pymoo`:

```json
{
  "optimize": {
    "backend": "pymoo",
    "population_size": null,
    "pymoo": {
      "algorithm": "auto",
      "shared": {
        "crossover_eta": 20.0,
        "crossover_prob_var": 0.5,
        "mutation_eta": 20.0,
        "mutation_prob_var": "auto",
        "eliminate_duplicates": true
      },
      "algorithms": {
        "nsga2": {},
        "nsga3": {
          "ref_dirs": {
            "method": "das_dennis",
            "n_partitions": "auto"
          }
        }
      }
    }
  }
}
```

#### NSGA-III, Reference Directions, and `das_dennis`

`nsga3` is a many-objective evolutionary algorithm. Unlike NSGA-II, it does not rely only on
crowding distance to spread candidates across the Pareto front. Instead, it uses a set of
reference directions in objective space and tries to keep the population distributed across them.

Passivbot uses the `das_dennis` method to generate those reference directions. This is the
standard simplex-partition method for NSGA-III and is a sensible default for Passivbot optimize
runs.

The main NSGA-III-specific knob is:

- `optimize.pymoo.algorithms.nsga3.ref_dirs.n_partitions`
  - Controls how fine the reference-direction grid is.
  - Higher values generate more reference directions, which increases diversity resolution but also
    makes each generation heavier.
  - With the default 11-objective Passivbot scoring set, common reference-direction counts are:
    - `n_partitions = 2` -> `66`
    - `n_partitions = 3` -> `286`
    - `n_partitions = 4` -> `1001`
  - Default is `"auto"`. For the default 11-objective setup, Passivbot currently resolves that to
    `n_partitions = 3`, which gives `286` reference directions.
  - In auto mode, Passivbot chooses the largest Das-Dennis grid whose reference-direction count
    fits within the resolved NSGA-III population budget. This preserves the NSGA-III invariant that
    `population_size >= reference directions`.

- `optimize.population_size`
  - For `pymoo` + `nsga3`, `null` means “auto”.
  - In that case Passivbot uses the default NSGA-III population budget of `500`, then resolves
    auto reference directions to the finest Das-Dennis grid that fits inside that budget.
  - For the default 11-objective setup, that means `population_size = 500` with `286` reference
    directions.
  - For a 10-objective setup, auto keeps `population_size = 500` and uses `220` reference
    directions (`n_partitions = 3`) because the next grid would require `715` reference directions.
  - For `pymoo` + `nsga2`, `null` means “auto” and currently resolves to `250`.
  - Set an explicit integer when you want to change the per-generation evaluation budget and the
    maximum auto reference-direction grid size. For example, `population_size = 1000` allows the
    10-objective auto resolver to use `715` reference directions (`n_partitions = 4`).
  - For `deap`, Passivbot currently falls back to its legacy fixed default when `null` is left in
    place.

#### Shared Pymoo Hyperparameters

The `shared` block controls the SBX crossover and polynomial mutation operators used by both
`nsga2` and `nsga3`.

Current meaning of the main pymoo knobs:

- `optimize.pymoo.algorithm`
  - `auto`, `nsga2`, or `nsga3`.
  - Default is `auto`.
  - `auto` chooses `nsga2` when `len(optimize.scoring) <= 3`, otherwise `nsga3`.
  - Use explicit `nsga2` or `nsga3` only when you want to override that default selection.
- `optimize.pymoo.shared.crossover_prob_var`
  - Per-variable SBX crossover probability.
  - Higher values mix more parameters between parents on each crossover.
  - Default `0.5` is a conservative middle ground for Passivbot's parameter space.
- `optimize.pymoo.shared.crossover_eta`
  - SBX distribution index.
  - Higher values keep offspring closer to the parents; lower values explore more aggressively.
  - Default `20` is a standard conservative setting and is usually a good starting point.
- `optimize.pymoo.shared.mutation_prob_var`
  - Per-variable polynomial-mutation probability.
  - `"auto"` means `1 / n_params`.
  - This is the default and is usually the right choice for Passivbot's parameter counts because
    it scales automatically with the number of tunable parameters.
- `optimize.pymoo.shared.mutation_eta`
  - Polynomial-mutation distribution index.
  - Higher values make smaller, more local mutations.
  - Default `20` keeps mutation fairly local, which is usually appropriate for expensive
    backtests.
- `optimize.pymoo.shared.eliminate_duplicates`
  - Skip duplicate candidates before wasting a full backtest on them.
  - Default `true`.
  - Recommended for Passivbot because each evaluation is relatively expensive.
- `optimize.pymoo.algorithms.nsga3.ref_dirs.method`
  - Reference-direction generator for NSGA-III.
  - Currently `das_dennis`.

Recommended defaults for typical Passivbot runs:

- Use `optimize.backend: pymoo` with `optimize.pymoo.algorithm: auto`.
- Keep `mutation_prob_var: "auto"`.
- Keep `crossover_eta: 20` and `mutation_eta: 20` unless you have a specific reason to make
  variation much more local or much more aggressive.
- Keep `crossover_prob_var: 0.5` unless you have evidence that crossover is either too timid or
  too disruptive for your runs.
- Leave `population_size: null` and `ref_dirs.n_partitions: "auto"` for default pymoo behavior:
  NSGA-II resolves null population size to `250`, while NSGA-III uses a default population budget
  of `500` and chooses the finest compatible reference-direction grid.
- Keep `pareto_max_size: 1000` unless archived front updates become a measured bottleneck for your
  machine or workflow.
- If you need more or less exploration pressure, change `population_size` first. It is the main
  budget/coarseness knob for NSGA-III auto reference directions. Use explicit `n_partitions` only
  when you specifically want to force a grid resolution.

Practical interpretation for the default shared block:

```json
"shared": {
  "crossover_eta": 20,
  "crossover_prob_var": 0.5,
  "eliminate_duplicates": true,
  "mutation_eta": 20,
  "mutation_prob_var": "auto"
}
```

- `crossover_eta: 20`
  - conservative crossover; offspring stay fairly close to parents
- `crossover_prob_var: 0.5`
  - each parameter has a 50% chance of participating in crossover
- `mutation_eta: 20`
  - conservative mutation; most mutations are relatively local
- `mutation_prob_var: "auto"`
  - mutate each parameter with probability `1 / n_params`
- `eliminate_duplicates: true`
  - do not spend backtests on duplicate candidates

These defaults are intentionally conservative. For most Passivbot optimize runs, scoring choice,
suite design, and evaluation budget matter more than fine-tuning these operator settings.

Algorithm selection under the default `auto` mode:

- `1` to `3` objectives -> `nsga2`
- `4+` objectives -> `nsga3`

That means the default 11-objective Passivbot template uses `nsga3`, while small custom scoring
lists automatically fall back to `nsga2`.

### Candle Interval

For faster optimization runs, you can aggregate 1-minute data into coarser candles before the
backtest loop runs. This reduces the number of bars processed per iteration.

Set `backtest.candle_interval_minutes` to a value greater than 1:

```json
{
  "backtest": {
    "candle_interval_minutes": 5
  }
}
```

Trade-offs:

- Intra-interval fill ordering is lost (fills occur only at the aggregated bar boundaries).
- Metrics are still time-correct because analysis uses timestamps rather than bar indices.
- The Apple MPS backend supports aggregated intervals for single- and multi-coin EMA Anchor and
  Trailing Martingale runs in long-only, short-only, and fused long+short form. It converts
  minute-denominated strategy and Forager EMA spans, static per-coin overrides,
  and elapsed-time cooldowns to candle periods, compounds HSL's one-minute EMA decay over each
  candle, and preserves Rust's boundary-crossing behavior when an interval does not evenly divide
  an hour; exact Rust validation remains authoritative.

### Fine-Tuning Specific Parameters

When you only want to adjust a handful of parameters and keep everything else fixed, use
`--fine_tune_params` (short: `-ft`). Provide a comma-separated list of dotted config-path
selectors to keep tunable; all other bounds are locked to their current config values
before the run starts. Selectors match full path segments by prefix or suffix, not partial
substrings.
The leading `bot.` may be omitted for side-local paths, so `long.risk` is equivalent to
`bot.long.risk`. A `*` segment may be used as a one-segment wildcard, for example
`*.strategy.close` matches both long and short active-strategy close params. A leaf selector
such as `we_excess_allowance_pct` matches every bound whose config path ends with that
parameter name.

```bash
passivbot optimize configs/examples/default_trailing_martingale_long.json \
  --fine_tune_params long.risk,long.forager,long.unstuck
```

Behind the scenes the optimizer sets every unlisted bound to `[value, value]`, so the GA
can mutate only the parameters you specified. Bounds for the listed parameters remain as
configured. The optimizer logs each selector expansion on separate sorted lines before
the run starts.

`optimize.fixed_params` provides the config-file equivalent for selectors that should always
be fixed to their current config values. It uses the same dotted path matching as
`--fine_tune_params`.

Useful examples:

```json
"optimize": {
  "fixed_params": ["long.strategy"]
}
```

This fixes every optimizer bound under `bot.long.strategy.<active_strategy>`, such as all
`trailing_martingale` entry/close thresholds, retracements, EMA spans, and volatility spans.
It does not fix `bot.long.risk`, `bot.long.forager`, or `bot.long.unstuck`.

```json
"optimize": {
  "fixed_params": ["long.strategy.close", "long.hsl"]
}
```

This fixes only the active long strategy's close subtree plus long HSL bounds.

Internally, `--fine_tune_params` and `optimize.fixed_params` are merged into one effective
fixed-parameter set before bounds are collapsed.

### Polishing Around A Selected Config

Use `--polish-pct` to narrow every configured optimizer bound around the current config
value before the run starts:

```bash
passivbot optimize path/to/config.json --polish-pct 0.25
```

By default this keeps the polished bounds inside the original `optimize.bounds` domain and
leaves fixed bounds fixed. `--polish-bounds-mode` changes that policy:

- `clamp`: default behavior; intersect polished bounds with the original bounds.
- `override-tunable`: allow tunable bounds to escape the original bounds; fixed bounds stay fixed.
- `override-all`: allow tunable bounds to escape the original bounds and expand fixed bounds too.

Polish still uses relative bounds: `[value * (1 - pct), value * (1 + pct)]`. A current
value of `0.0` therefore remains fixed at `[0.0, 0.0]`.

`optimize.fixed_params` is applied after polish. That makes it the right way to polish all
bounds while pinning selected parameters to the config value:

```bash
passivbot optimize path/to/config.json \
  --polish-pct 0.25 \
  --polish-bounds-mode override-all \
  --optimize.fixed_params long.risk.n_positions,long.risk.total_wallet_exposure_limit
```

When `--fine_tune_params` is combined with `--start`, the base optimizer config remains the run
policy. Anchor configs provide values only for optimizer-bound bot parameters that are fixed by the
anchor plan; boolean toggles and other non-bound runtime policy fields such as
`bot.long.hsl.enabled` continue to come from the base config or explicit runtime overrides. Seed and
anchor values outside `optimize.bounds` are clamped into bounds during seed loading and logged in
aggregate with counts, source examples, key/path, original value samples, bound, and clamped value.

`optimize.fixed_runtime_overrides` is different: it overrides runtime config values only during
optimize evaluations, without changing the stored/live config value. This is useful for
operator-risk settings such as:

```json
"optimize": {
  "fixed_runtime_overrides": {
    "bot.long.hsl.no_restart_drawdown_threshold": 1.0,
    "bot.short.hsl.no_restart_drawdown_threshold": 1.0
  }
}
```

That default override disables terminal no-restart during optimizer evaluations so candidates can
be constrained through `drawdown_worst_strategy_eq`, `drawdown_worst_ema_strategy_eq`,
`drawdown_worst_mean_1pct_strategy_eq`, `drawdown_worst_mean_1pct_ema_strategy_eq`, and
`strategy_eq_recovery_days_max` instead of being prematurely truncated.

When you provide many starting configs to a CPU optimizer, it bounds how many seed evaluations may
be in flight at once. For the DEAP backend, the same cap also applies to generation offspring
evaluations:

```json
"optimize": {
  "max_pending_starting_evals_per_cpu": 1
}
```

Effective cap:

- `max_pending = n_cpus * max_pending_starting_evals_per_cpu`
- All provided starting configs are still exact-evaluated before a CPU optimizer trims them down to
  its initial population. GPU `auto` and `screened` seed bootstrapping instead use the capped policy
  described above.

This is mainly a memory-control knob for large seed pools and DEAP generation batches, especially
in suite mode where each candidate returns a larger metrics payload. Lower it first if the VPS
spikes RAM during seed or offspring evaluation.

### Optimizer Suites

The optimizer reuses the backtest suite configuration and allows every candidate to
be evaluated across multiple scenarios before scoring. Each scenario can override coins,
date ranges, exchanges, `coin_sources`, and bot parameters via `overrides`. The optimizer
prepares a single shared dataset that covers the union of the requested data so additional
scenarios add minimal overhead.

Key fields (directly under `backtest`):

- `backtest.suite_enabled`: master toggle for suite mode, can also be set with `--suite [y/n]`
- `backtest.scenarios`: list of scenario dictionaries (same schema as backtest scenarios)
- `backtest.reducer`: default per-metric reducers for combining scenario results (default:
  `{"default": "mean"}`)

`optimize.objective_scenario` sets the default scoring basis. Set it to a unique scenario label
(commonly `base`) to read objectives from that scenario by default, or leave it `null` to use suite
aggregation by default. The equivalent global CLI override is `--objective-scenario LABEL`.
Each `optimize.limits` entry independently uses suite aggregation by default or may select one
named scenario.

Each object-form `optimize.scoring` entry may override the default:

```json
{
  "backtest": {
    "reducer": {"default": "mean"}
  },
  "optimize": {
    "objective_scenario": "base",
    "scoring": [
      {"metric": "adg_strategy_eq", "goal": "max"},
      {
        "metric": "strategy_eq_underwater_pct_mean",
        "goal": "min",
        "scenario": null
      },
      {
        "metric": "strategy_eq_recovery_days_max",
        "goal": "min",
        "scenario": null,
        "reducer": "max"
      }
    ]
  }
}
```

An omitted `scenario` inherits `optimize.objective_scenario`, a named value selects that scenario,
and explicit `null` selects suite reduction. A reduced objective without its own `reducer`
uses the metric-specific `backtest.reducer` rule and then its `default`; an explicit `reducer`
may be `mean`, `min`, `max`, `std`, or `median`. A named scenario and `reducer` are mutually
exclusive. This permits representative base-period performance objectives alongside mean or
worst-case stress objectives while limits retain their independent suite-wide contract.
Each metric may still appear only once in `optimize.scoring`; scoring the same metric from multiple
bases is not supported in this first version.

Limits may combine a scenario-specific threshold with a separate suite-wide threshold for the same
metric:

```json
"limits": [
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  }
]
```

The first limit reads only `base`; the second reads the maximum across scenarios. Their violations
are evaluated independently and their penalties accumulate. A named `scenario` and `reducer` are
mutually exclusive. An omitted or null `scenario` uses suite aggregation; with no explicit `reducer`,
the limit falls back through the metric-specific and default `backtest.reducer` rules.

Suite mode is opt-in. The default schema/example config does not enable it automatically.

During evaluation the optimizer records:

- Per-scenario combined metrics (the same mean/min/max/std set produced by standalone
  backtests). These are exposed on each individual as `<label>__{metric}`.
- Aggregated metrics computed with the `backtest.reducer` rules (default `mean`).
  These values feed `optimize.limits` and suite-reduced `optimize.scoring` entries.

`reducer` is canonical across suite configuration, scoring entries, limits, CLI entries, and
serialized configs. For backward compatibility, the input aliases `aggregate`, `stat`, and
`scenario_stat` are accepted at the same schema positions (`field` is additionally accepted for
legacy limits). Same-valued aliases collapse to `reducer`; conflicting aliases fail validation.
Historical Pareto artifact payloads keep their existing `suite_metrics.aggregate`, `aggregated`,
and `stats` keys and remain readable.

See [Suite Examples](suite_examples.md) for practical scenario configurations including exchange
comparisons, date range testing, and parameter sensitivity analysis.

Result directories stay under `optimize_results/`, but the coin portion of the folder
name switches to `suite_{n}_coins` to make suite runs easy to locate.

Each evaluation written to disk now includes a compact `suite_metrics` payload:

```json
"suite_metrics": {
  "aggregate": {
    "aggregated": {"adg_btc_w": 0.0012, "...": "..."},
    "stats": {"adg_btc_w": {"mean": 0.0011, "min": 0.0008, "max": 0.0014, "std": 1.5e-4}}
  },
  "scenarios": {
    "scenario_a": {"stats": {"adg_btc_w": {"mean": 0.0012, "min": 0.0011, ...}}},
    "scenario_b": {"stats": {"adg_btc_w": {"mean": 0.0009, ...}}}
  }
}
```

Pareto members store a compact metrics payload under `metrics.stats` (and `suite_metrics` when suite
mode is enabled) instead of the older `analyses_combined` / per-exchange analysis blocks.

## Optimization Process

- Uses a multi-objective evolutionary backend (`deap`, `pymoo`, or experimental `gpu`)
- `pymoo` defaults to NSGA-III for many-objective runs, with NSGA-II still available explicitly
- Backtests across historical OHLCV data
- Uses multiprocessing with shared memory for reduced RAM load
- Maintains Pareto front of best-performing configurations
- Enforces constraints via `optimize.limits`
- Optimizes for multiple metrics via `optimize.scoring`
- Avoids duplicates through hash tracking and perturbation
- Logs starting-config dedup statistics at startup, including how many raw configs collapsed after quantization and how many extra TWEL-scaled variants survived

Per-coin warmup inside an optimizer run is sized from `optimize.bounds`, not from the `bot.*` template values. The optimizer treats each optimized field as if it were at its upper bound when computing how much history each coin needs, so every individual evaluated in the same run trades on an identical window. Standalone `passivbot backtest <config.json>` is unaffected and still sizes warmup from whatever bot values the config you hand it contains.

## Output Structure

Each optimization run creates a directory:
```
optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{n_days}days_{coin_label}_{hash}/
```

Contents:
- `all_results.bin`: Binary log of all evaluated configs (msgpack format)
- `pareto/`: JSON files for Pareto-optimal configurations
  - Named `{hash}.json`
  - Files are added/removed over time as the Pareto front updates and is pruned to `optimize.pareto_max_size`

## Analyzing Results

Full analysis is included in each member of the Pareto front. Two helper tools are available:

```bash
# Single-candidate selector
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto

# Interactive dashboard (recommended)
passivbot tool pareto-dash --data-root optimize_results

# Static matplotlib plotter
python3 src/pareto_store.py optimize_results/.../pareto/
```

`passivbot tool pareto` is the quickest way to promote one config from a large Pareto set. It
loads the JSON artifacts, optionally filters them with `--limit` / `--limits`, then chooses one
candidate using a named decision rule. It accepts either a `pareto/` directory, an optimize run
directory, or no path at all, in which case it falls back to the newest local
`optimize_results/.../pareto` by lexicographic run-directory name, considering only runs whose
`pareto/` subdirectory contains at least one `*.json` candidate. It also shows the retained
front's ideal point for the active objectives. Recommended workflow:

1. apply hard filters with `--limit`
2. use `-m reference` if you already know your target ADG / drawdown / recovery regime
3. otherwise start with the default `ideal` selector
4. switch to `-m knee` if you specifically want a balanced-compromise heuristic
5. use `--show-top N` to inspect the shortlist before promoting one config
6. use `--json` if you want to script the selection

Available methods:

- `knee`: approximate balanced compromise point
- `reference`: closest to user-specified targets via `--target metric=value`
- `ideal`: closest to the observed ideal point; default
- `utility`: weighted scalarization via `--weight metric=value`
- `lexicographic`: strict objective priority via `--priority metric_a,metric_b,...`
- `outranking`: simplified PROMETHEE-style pairwise ranking

These are practical selection heuristics for large Passivbot Pareto fronts rather than fully
formal MCDM implementations. For most real runs, `knee`, `reference`, and `utility` are the most
useful methods.

`-o` / `--objectives` can also reference stored metrics outside the original `optimize.scoring`
list, for example `sharpe_ratio_strategy_eq`, as long as that metric is present in the
saved Pareto JSON and Passivbot has a known default min/max direction for it.

Example:

```bash
passivbot tool pareto \
  -o sharpe_ratio_strategy_eq,adg_strategy_eq,strategy_eq_recovery_days_max \
  -m ideal
```

`pareto_dash.py` scans one or more optimization runs and launches a Plotly Dash app with:

- Scatter/histogram views for any metrics or objectives
- Defaults to the metrics listed in `config.optimize.scoring`, so the scatter/histogram
  immediately highlight your optimization objectives when the app loads
- Scenario-aware box plots (per-metric distributions broken down by suite scenario)
- Correlation heat maps and parameter-vs-metric scatter plots for quick diagnostics
- Streaming history chart sourced from `all_results.bin`
- CSV export of the current run's dataset for offline analysis

Use the full install profile (`pip install -e ".[full]"`) if the dashboard dependencies are not already present.
The legacy `pareto_store.py` script still supports quick 2D/3D matplotlib plots if a GUI
isn't needed.

## Optimization Limits

To enforce constraints during optimization, populate `optimize.limits` with a list of limit
objects. Each object describes when to penalize a result:

- `metric`: canonical metric name (e.g. `drawdown_worst_btc`, `loss_profit_ratio`, `adg`).
- `penalize_if`: comparison operator. Use `<`, `<=`, `>`, `>=`, `==` (or aliases like `less_than`
  / `greater_than`), `outside_range`
  to keep a metric within `[low, high]`, or `inside_range` to forbid a band.
- `value`: numeric threshold for `<`/`>` limits.
- `range`: `[low, high]` for the range-based operators.
- Optional `scenario`: select one named suite scenario. Omitted or explicit `null` uses suite
  aggregation. Named scenarios cannot be combined with `reducer`.
- Optional `reducer`: for suite aggregation, override the statistic to compare against (`min`, `max`,
  `mean`, `std`, or `median`).
  Without `reducer`, Passivbot resolves the metric through `backtest.reducer`: first a
  metric-specific reducer rule, then `backtest.reducer.default`, then `mean`.

Example:

```json
"limits": [
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "scenario": "base",
    "value": 0.5
  },
  {
    "metric": "drawdown_worst_strategy_eq",
    "penalize_if": "greater_than",
    "reducer": "max",
    "value": 0.7
  },
  {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.05, 0.7]},
  {"metric": "adg", "penalize_if": "<", "value": 0.0008, "reducer": "mean"}
]
```

CLI overrides can replace the full limit set with the same JSON/HJSON payload:

```bash
passivbot optimize --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]'
```

For quicker one-off edits, use repeatable `--limit` entries. The symbolic scalar operators
in `--limit` are written as keep conditions, matching `pareto_store.py` filtering:
- `--limit 'adg > 0.0008'` means keep only results with `adg > 0.0008`
- `--limit 'drawdown_worst <= 0.35'` means keep only results with `drawdown_worst <= 0.35`

```bash
passivbot optimize \
  --clear-limits \
  --limit 'drawdown_worst <= 0.35' \
  --limit 'drawdown_worst_strategy_eq <= 0.5 scenario=base' \
  --limit 'backtest_completion_ratio>=1.0' \
  --limit 'loss_profit_ratio outside_range [0.05,0.7]' \
  --limit 'adg > 0.0008 reducer=mean'
```

You can also combine both forms. `--limits` loads a whole list first, and each `--limit`
appends one more canonical entry:

```bash
passivbot optimize \
  --limits '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]' \
  --limit 'strategy_eq_recovery_days_max <= 21'
```

Semantics:

- `--limits` replaces `config.optimize.limits` for that run.
- `--limit` is repeatable and appends one parsed entry to that replacement set.
- `--limit` string expressions use keep-condition semantics for scalar operators (`>`, `>=`, `<`,
  `<=`, `==`). Explicit JSON/HJSON limit objects still use direct `penalize_if` semantics.
- `--clear-limits` starts from an empty limit list before any `--limits` or `--limit` entries are applied.

Limit violations do not disqualify a config. They produce positive penalty scores in optimizer
engine space: if a limit metric matches a configured scoring metric, the penalty replaces that
objective's engine score for the candidate; unmatched/global penalties affect every objective.
Metric names may include `_usd` / `_btc` suffixes to lock a denomination; when omitted, USD is
assumed.

## Performance Metrics

Backtest statistics originate in the Rust engine (`passivbot-rust/src/analysis.rs`) and are
augmented in Python (`src/backtest.py`). The optimizer aggregates them per exchange and then
over all exchanges before scoring.

### How Metrics Feed Scoring

- `optimize.scoring` lists the objective metrics. Each entry becomes a fitness component in
  sorted order.
- Each scoring entry is normalized to `{metric, goal}`. `goal: "max"` means higher raw metric
  values are better; `goal: "min"` means lower raw metric values are better.
- For every metric, `Evaluator.combine_analyses` computes mean/min/max/std across all
  exchanges in the run. The scoring logic uses the mean (`{metric}_mean`).
- Internally, Passivbot converts all objectives into optimizer engine space where lower is better,
  so both the `deap` and `pymoo` backends receive consistent minimization-style values.
- Penalties from `optimize.limits` replace affected engine objective scores when a bound is
  violated, turning constraint breaches into very poor scores while preserving the raw metrics
  stored for inspection.
- Metrics are emitted with both USD and BTC suffixes (for example, `adg_usd` and `adg_btc`).
- `_btc` metrics use BTC-denominated balance/equity as the numeraire even when
  `backtest.btc_collateral_cap = 0`, so they can be used to compare strategy performance against
  passive BTC exposure.
- The tables below reference the base metric names for brevity; append `_usd` or `_btc` to select the denomination you want to use.
- Exposure-normalized variants (e.g., `adg_per_exposure_long`) divide the base metric by
  that side’s configured `total_wallet_exposure_limit`, letting you compare bots that use
  different leverage budgets.

### Returns & Growth
| Metric | Description |
|--------|-------------|
| `adg`, `adg_w` | Average Daily Gain (smoothed geometric) and its recency-biased counterpart |
| `mdg`, `mdg_w` | Median Daily Gain and its recency-biased counterpart |
| `gain` | Final balance gain (end/start ratio) |
| `adg_strategy_eq`, `adg_strategy_eq_w` | Collateral-agnostic geometric growth on the synthetic strategy-equity curve |
| `mdg_strategy_eq`, `mdg_strategy_eq_w` | Median-day version of the same strategy-equity growth family |
| `*_per_exposure_{long,short}` | Above metrics divided by the configured exposure limit per side |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `drawdown_worst` | Maximum peak-to-trough drawdown |
| `drawdown_worst_mean_1pct` | Mean of worst 1% daily worst drawdowns, computed from full-resolution drawdowns before daily reduction |
| `drawdown_worst_strategy_eq` | Worst drawdown on collateral-agnostic strategy equity |
| `drawdown_worst_ema_strategy_eq` | Worst EMA-smoothed strategy-equity drawdown, shared as `max(long, short)` |
| `drawdown_worst_mean_1pct_strategy_eq` | Mean of worst 1% daily worst strategy-equity drawdowns, computed from full-resolution strategy-equity drawdowns before daily reduction |
| `drawdown_worst_mean_1pct_strategy_eq_{long,short}` | Mean of worst 1% daily worst strategy-equity drawdowns for the long or short HSL controller |
| `drawdown_worst_mean_1pct_ema_strategy_eq` | Mean of worst 1% EMA-smoothed strategy-equity drawdown samples, shared as `max(long, short)` |
| `expected_shortfall_1pct` | Mean of worst 1% daily losses (CVaR) |
| `equity_balance_diff_neg_max` / `pos_max` | Largest divergence between equity and account balance (negative side tracks only drawdowns below balance; positive side tracks only run-ups above balance) |
| `equity_balance_diff_neg_mean` / `pos_mean` | Average divergence between equity and balance (split by sign as above) |

### Ratios & Efficiency
| Metric | Description |
|--------|-------------|
| `sharpe_ratio`, `sharpe_ratio_w` | Return-to-volatility ratio and its recency-biased variant |
| `sortino_ratio`, `sortino_ratio_w` | Return-to-downside-volatility ratio |
| `calmar_ratio`, `calmar_ratio_w` | Return divided by maximum drawdown |
| `sterling_ratio`, `sterling_ratio_w` | Return divided by the average of the worst 1% drawdowns |
| `omega_ratio`, `omega_ratio_w` | Sum of positive returns / sum of absolute negative returns |
| `*_strategy_eq`, `*_strategy_eq_w` ratios | Collateral-agnostic ratio family using the strategy-equity curve |

### Position & Execution Metrics
| Metric | Description |
|--------|-------------|
| `positions_held_per_day` | Average number of unique positions opened per day |
| `position_held_hours_{mean,median,max}`, `position_held_days_{mean,median,max}` | Holding-time statistics in hours and equivalent days |
| `position_unchanged_hours_max`, `position_unchanged_days_max` | Longest span without modifying an existing position, in hours and equivalent days |
| `fills_gap_time_weighted_mean_hours` | Time-weighted mean portfolio no-fill gap: `sum(gap_hours^2) / sum(gap_hours)` over unique fill timestamps and the analysis boundaries. Lower values reward distributing fills through long droughts instead of clustering activity. |
| `volume_pct_per_day_avg`, `volume_pct_per_day_avg_w` | Average traded volume as % of account per day, with recency bias |
| `peak_recovery_hours_equity_usd`, `_btc`; `peak_recovery_days_equity_usd`, `_btc` | Longest time the equity curve stayed below its prior peak before recovering, per denomination, in hours and equivalent days. Available for scoring and limit checks (e.g. `{"metric": "peak_recovery_days_equity_usd", "penalize_if": ">", "value": 7}`). |
| `peak_recovery_hours_pnl`, `peak_recovery_days_pnl` | Longest recovery time of cumulative realised PnL (USD), in hours and equivalent days. Useful for monitoring realised drawdown recovery latency. |
| `strategy_eq_recovery_days_mean`, `strategy_eq_recovery_days_median`, `strategy_eq_recovery_days_p95`, `strategy_eq_recovery_days_p99`, `strategy_eq_recovery_days_mean_worst_5pct`, `strategy_eq_recovery_days_mean_worst_1pct`, `strategy_eq_recovery_days_max` | Per-sample strategy-equity time-to-exceed distribution in days. Each sample measures how long until a later strategy-equity sample strictly exceeds it; unrecovered samples use the open tail to the backtest end. |
| `peak_recovery_hours_strategy_eq`, `peak_recovery_days_strategy_eq` | Legacy max-recovery metrics. `peak_recovery_days_strategy_eq` is an alias for `strategy_eq_recovery_days_max`; the hours variant remains available for older configs. |
| `peak_recovery_hours_strategy_eq_{long,short}`, `peak_recovery_days_strategy_eq_{long,short}` | Longest strict time-to-exceed interval for the long or short HSL strategy-equity controller, including an unrecovered tail through the backtest end. |
| `high_exposure_hours_{mean,max}_long`, `high_exposure_days_{mean,max}_long` | Mean / maximum duration of continuous periods where total long wallet exposure exceeded the daily-resampled average long TWE, in hours and equivalent days |
| `high_exposure_hours_{mean,max}_short`, `high_exposure_days_{mean,max}_short` | Mean / maximum duration of continuous periods where total short wallet exposure exceeded the daily-resampled average short TWE, in hours and equivalent days |

### Equity Curve Quality
| Metric | Description |
|--------|-------------|
| `equity_choppiness`, `equity_choppiness_w` | Normalized total variation (lower is smoother) |
| `equity_jerkiness`, `equity_jerkiness_w` | Normalized mean absolute second derivative |
| `exponential_fit_error`, `exponential_fit_error_w` | MSE from a log-linear equity fit |

> Metrics with the `*_w` suffix use recency-weighted means: the metric is evaluated on ten
> overlapping slices of the equity curve (full history, last 1/2, last 1/3, …, last 1/10)
> and averaged. This biases the score toward recent behavior without ignoring the past.

The equity-balance difference metrics are derived by computing `(equity - balance) / balance`
minute-by-minute. Positive deviations contribute exclusively to the `*_pos_*` metrics, while
negative deviations contribute exclusively to the `*_neg_*` metrics; no cross-contamination
occurs. This mirrors the separation implemented in `passivbot-rust/src/analysis.rs` and helps
highlight asymmetric behavior in bots whose equity routinely sits above or below the
account’s wallet exposure limit baseline.

## Utilities

Loading results programmatically:
```python
from opt_utils import load_results

for config in load_results("optimize_results/.../all_results.bin"):
    # Work with config
```

### Monitoring Optimizer Memory Usage

The script `src/tools/capture_optimize_memory.py` samples an active optimizer process tree and
host memory state into a JSON report. This is useful when validating that shared-memory datasets
are behaving as expected on a given machine.

```bash
PYTHONPATH=src python3 src/tools/capture_optimize_memory.py --wait --output tmp/optimize_memory.json
```

Use `--pid <pid>` to monitor a specific optimizer process instead of waiting for the newest
matching process.

### Profiling Suite Optimizer Evaluation

Set `PASSIVBOT_OPTIMIZE_PROFILE=1` to log per-candidate suite evaluator timings:

```bash
PASSIVBOT_OPTIMIZE_PROFILE=1 passivbot optimize path/to/config.json
```

The profile is opt-in and emits `[opt-profile]` INFO lines for suite evaluation phases such as
scenario config assembly, runtime compilation, payload construction, Rust backtest execution,
metric combining, aggregate metric building, and Pareto result recording. When enabled, candidate
metrics also include a `profile` block so retained Pareto configs can be inspected after the run.
