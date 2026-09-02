# Changelog

All notable user-facing changes will be documented in this file.

## Unreleased

- Added `fills_gap_time_weighted_mean_hours` as an exact backtest and CPU/GPU optimizer metric.
  It minimizes `sum(gap_hours^2) / sum(gap_hours)` over unique portfolio fill timestamps and the
  analysis boundaries, providing smoother selection pressure against long no-fill periods than a
  single maximum-gap objective. The EMA Anchor example now uses it in optimizer scoring.

- Expanded `passivbot tool trailing-inspect` with a config-first overview of both long and short
  behavior across example volatility and exposure regimes. The report now includes threshold,
  retracement, nominal confirmation, and emitted-order reference prices from a configurable anchor;
  active/dormant side status; and plain-language explanations of sensitivity, entry cooldown and
  ladder staging, EMA gating, and recursive closes. The existing detailed single-scenario and JSON
  interfaces remain available.

- GPU optimization now bootstraps `--start` configs into an authoritative exact-Rust seed archive.
  The default `auto` policy exact-evaluates up to 128 deduplicated seeds; larger pools receive one
  full-history Metal proxy screen followed by capped exact validation of diverse proxy-Pareto
  members, objective extremes, and broad probes. Bootstrap evaluations are additional to
  `optimize.iters`, exact and proxy fitness remain isolated, and crash-safe checkpoints preserve
  incomplete seed plans and anchored fine-tune context. CPU optimization retains exact evaluation
  of every starting config.

- Apple MPS single-coin Trailing Martingale now selects a recursive-entry-only Metal variant when
  every active candidate uses nonpositive entry retracement. The variant compiles out trailing
  entry retracement/trigger work while preserving the recursive ladder and bitwise generic-kernel
  outputs. The deterministic GPU benchmark adds a fixed recursive-entry case for repeatable
  before/after measurements. Exact Rust validation, CPU optimization, backtests, and live behavior
  are unchanged.

- Apple MPS single-coin Trailing Martingale now uses a reducer-free recursive-close fast path when
  every active candidate side disables WEL/TWEL enforcers and auto-unstuck, avoiding redundant
  reducer allocations while preserving the ordinary close ladder. The deterministic GPU benchmark
  adds paired static-close and recursive-close cases so this cost remains reproducible and visible.

- Apple M3-family MPS optimization now auto-selects a 2304-candidate population and 4.5-billion
  candidate-bar dispatch envelope for the proven lean one-sided Trailing Martingale kernel,
  restoring roughly 110–117 proxy candidates/second in the long-history benchmark while keeping
  command buffers near 20 seconds. Other Apple Silicon families plus coin-overridden, HSL,
  multicoin, suite, market-order, reducer, recursive-mode, and active-volatility kernels retain the
  1-billion safety envelope. Runs requesting opt-in proxy metric features, including BTC analysis,
  entry intervals, recovery distributions, equity-balance divergence, and HSL diagnostics, also
  retain the 1-billion envelope. Set `optimize.gpu.auto_lean_parallelism` to `false`, or set any
  GPU sizing value explicitly, to disable the automatic selection.

- Apple MPS optimization now defaults to a 1-billion candidate-bar dispatch envelope instead of
  500 million, approximately doubling per-dispatch parallelism on long one-sided histories.
  `optimize.gpu.max_dispatch_candidate_bars` may restore the former `500000000` conservative
  setting; larger envelopes increase interrupt latency and may temporarily reduce desktop
  responsiveness while a Metal dispatch is running.

- Bitunix now defers live exchange actions when a locked-fund balance does not reconcile with the
  exchange-calculated maximum-transfer amount, including after restart, preventing a transient
  account response from reporting unchanged locked funds in both `available` and `used` balance
  components.
- Live `balance_override` values must now be positive finite numbers; booleans and other invalid
  values fail during bot initialization instead of being coerced into a sizing balance.

- Apple MPS successive-halving optimization now evaluates its opt-in 25% and 50% rungs on the
  most recent portion of the configured date range, with the normal indicator warmup immediately
  preceding each scoring window. The final rung remains the full configured history. Checkpoint
  identity distinguishes these recent-suffix semantics from the former historical-prefix behavior.
- Apple MPS single-coin `coin`-mode HSL screening now retains up to 8,192 realized-PnL event
  candles in the finite lookback window and coalesces all same-candle entry fees and close PnL.
  High-cadence candidates with more than 2,048 valid close events no longer receive an artificial
  early-stop penalty from fill multiplicity crossing the old capacity, which could poison
  proxy/exact drift evidence;
  overflow remains bounded and fail closed. The effective ring capacity is part of checkpoint
  identity, so finite-lookback coin-HSL checkpoints from the old capacity cannot mix stale proxy
  evidence with new evaluations. Exact Rust validation, CPU optimization, backtests, and live
  behavior are unchanged.
- Apple MPS single-coin Trailing Martingale screening now compiles out the 1-minute and 1-hour
  volatility EMA state, candle-range loads, and volatility-weight arithmetic when every active
  candidate in a dispatch uses zero volatility weights for entry and close thresholds and
  retracements. Mixed or nonzero-weight dispatches retain the full kernel. Opt-in GPU profiles now
  also report terminated-candidate counts and estimated post-termination candidate-bars, making
  future early-exit work measurable. Exact Rust validation, CPU optimization, backtests, and live
  behavior are unchanged.

- Apple MPS single-coin Trailing Martingale optimization now offers disabled-by-default
  successive-halving proxy screening. The default opt-in ladder evaluates 1024 candidates on 25%
  of history, 512 on 50%, and 256 on the complete history, while preserving the existing
  candidate-bars dispatch safety envelope. Only complete-history survivors are eligible for exact
  Rust validation and proxy/exact drift evidence; CPU optimization, backtests, live behavior, and
  ordinary GPU runs are unchanged.

- Apple MPS single-coin Trailing Martingale screening now selects dispatch-proven Metal variants
  that compile out recursive entry/close grids when every active candidate uses positive
  retracement, and compile out WEL/TWEL enforcers plus auto-unstuck when every active candidate has
  those reducers disabled. Ordinary market execution and the realized-loss gate are likewise
  removed only when their fixed run settings disable them. Mixed or enabled-feature dispatches
  keep the full kernel. HSL configured only on an exposure-disabled side no longer promotes the
  active side to an HSL kernel. Exact Rust validation remains unchanged and authoritative.

- The Apple MPS optimizer now defaults to a 1024-candidate NSGA-II population instead of 4096,
  allowing long-history runs to reach exact Rust validation four times sooner. Explicit population
  sizes and the independent 4096-candidate dispatch-batch upper bound remain unchanged.

- Apple MPS single-coin Trailing Martingale optimization now compiles dedicated long-only and
  short-only kernels while HSL is enabled, allowing Metal to eliminate the inactive strategy side
  and use a one-controller HSL update. HSL lifecycle, drawdown, tail, and recovery accumulators are
  compiled only when requested by a proxy objective or limit; controller behavior remains active
  when those diagnostics are omitted. The existing dispatch safety envelope remains unchanged.
  Dual-side, EMA Anchor, multi-coin, HSL-disabled, and CPU paths retain their existing kernel and
  dispatch contracts, and exact Rust validation remains authoritative.

- Long-running Apple MPS proxy generations now emit rate-limited dispatch progress with elapsed
  time and ETA after 30 seconds, while opt-in profiles record each dispatch chunk's wall time. The
  deterministic offline benchmark adds a single-side Trailing Martingale case with HSL enabled so
  HSL-heavy dispatch sizing and kernel changes can be measured directly.

- The deterministic Apple MPS HSL benchmark now matches the production proxy feature contract:
  it compiles HSL-only diagnostics only when requested and exercises the default 30-day finite
  coin-PnL lookback in `coin` signal mode, recording that lookback and mode in each report. This
  prevents benchmark-only HSL work from overstating or bypassing normal optimizer cost; optimizer
  behavior is unchanged.

- Apple MPS EMA Anchor optimization now compiles a one-side kernel when HSL is disabled, allowing
  Metal to eliminate the inactive side and HSL state from the hot candle loop. Dual-side and HSL
  runs keep the generic kernel, and exact Rust validation remains authoritative. The offline GPU
  benchmark also accepts an independent dispatch batch size so saturation studies can hold the
  candidate matrix constant while varying only dispatch chunking.

- Apple MPS optimization now provides disabled-by-default structured profiling for candidate
  materialization and packing, buffer upload/clearing, cold and warm shader-library work, kernel
  execution, device transfer, metric reduction, NSGA orchestration, exact-validation queue/work,
  result persistence, and checkpointing. Profile records include actual dispatch shape,
  candidate-bars, topology, optional metric features, and cold/warm state. A deterministic offline
  `passivbot tool gpu-proxy-benchmark` harness covers long-history EMA Anchor and Trailing
  Martingale, short-history multicoin, and coin-override workloads with fixed seeds, cold timing,
  and warm p50 throughput. Profiling remains opt-in and exact Rust results remain authoritative.

- Apple MPS optimization now keeps exact-analysis diagnostics separate from proxy objectives and
  proxy-side limits. Raw gain/PnL, positive equity-balance divergence, completed-only account
  recovery, raw or split fill activity, raw HSL event/absolute-loss metrics, legacy recovery/profit
  aliases, and the self-relative high-exposure duration family remain available from exact Rust
  backtests but fail closed when requested as GPU optimization signals. Removing high-exposure proxy scoring
  also removes its second complete Metal strategy replay, so every valid EMA Anchor and Trailing
  Martingale proxy evaluation dispatches its strategy kernel once. CPU optimizers, backtesting, and
  live operation are unchanged.

- GPU optimization now runs an Apple MPS capability preflight before loading historical data.
  Missing MPS support, unsupported strategies, positive `backtest.btc_collateral_cap`, and
  unmodeled suite override paths fail immediately with the unsupported setting, the CPU-backend
  fallback, and a documentation pointer. Successful starts log the strategy, zero-collateral
  contract, and 64-coin-per-scenario ceiling. CPU optimization and backtesting do not import or
  probe the optional GPU runtime.

- Apple MPS optimization now accepts the same per-coin live-only `leverage` and non-`normal`
  `forced_mode_<side>` values on either enabled or disabled sides as CPU optimization. These values
  do not affect CPU backtests, so the MPS proxy also leaves them inert, emits an explicit warning
  instead of silently ignoring them, and retains their exact values in checkpoint identity. This
  lets composed live configs move between CPU and GPU optimization without deleting valid live-only
  coin settings.

- Apple MPS suite optimization now supports scenarios spanning a strict subset of multiple
  exchanges. Metal evaluates each prepared exchange dataset independently, combines their proxy
  metrics with the canonical CPU per-scenario mean/minimum/maximum/standard-deviation/median
  contract, and only then applies suite reducers, named-scenario objectives, and limits. Exact Rust
  suite validation remains authoritative, and every prepared exchange dataset remains part of
  checkpoint identity.

- Apple MPS Trailing Martingale optimization now supports normal-initial-entry interval mean,
  median, p95, p99, and maximum metrics across single-coin, multi-coin, long-only, short-only,
  and fused long+short runs. Metal tracks intervals independently per coin and position side;
  aggregate and per-bin event counts use integer-safe buffers, while percentile screening uses
  conservative logarithmic histogram bounds under mandatory exact Rust validation and rolling
  drift gates. EMA Anchor retains exact Rust's canonical zero values for this metric family.
  Runs that do not request an entry-interval metric keep their existing kernel ABI and dispatch
  cost.

- Backtest analysis now aligns realized balance changes with tracked equity samples by timestamp
  when timestamp data is available. This fixes equity-vs-balance and paper-loss metrics after a
  warmup period, where absolute fill candle indices were previously compared with equity-series
  offsets and could leave a stale balance in the analysis.

- Apple MPS single- and multi-coin EMA Anchor and Trailing Martingale optimization now supports
  unweighted positive and negative equity-vs-balance maximum and mean metrics, plus paper-loss
  maximum and mean ratios, in USD and BTC. Metal enables a separate compact accumulator only when
  one of these metrics is requested, so ordinary GPU runs retain their existing kernel ABI and
  dispatch cost. BTC balance is rebased at each proxy fill, with tracked pre-fill samples replayed
  against the first-fill baseline; positive sign-filtered means remain an online approximation
  under mandatory exact Rust validation and rolling drift gates.

- Apple MPS single- and multi-coin EMA Anchor and Trailing Martingale optimization now accepts
  BTC-denominated account-equity scoring and limits while `backtest.btc_collateral_cap` is zero.
  The proxy converts its compact USD daily equity surface with the canonical prepared BTC/USD
  series and supports gain, ADG, MDG, Omega, equity shape, unweighted exposure ratios, peak
  recovery, per-exposure forms, Sharpe, Sortino, expected shortfall, worst and worst-1% drawdown,
  Calmar, and Sterling. UTC
  day-end conversion is exact for the compact surface, including candidate-specific liquidation
  endpoints; recovery remains a compact daily approximation under mandatory exact Rust validation
  and rolling drift gates. Metal conditionally retains synchronized BTC day-end equity, daily
  minima, and full-curve daily worst drawdowns only when one of the intraday-risk metrics is
  requested; USD-only and close-equity-only BTC runs retain their previous kernel ABI, output
  width, and dispatch cost. Weighted BTC Sharpe/Sortino/Calmar/Sterling and exposure ratios remain
  fail-closed until the proxy owns suffix-local intraday minima, drawdown, and exposure series. BTC inputs enter checkpoint
  identity only when a BTC metric is requested. Positive BTC collateral remains fail-closed
  pending its separate simulation slice.

- Apple MPS multi-coin EMA Anchor and Trailing Martingale optimization now supports both wallet-
  exposure denominator modes. With `backtest.dynamic_wel_by_tradability: false`, Metal divides
  total wallet exposure by each side's configured `n_positions`, matching exact Rust across
  long-only, short-only, hedge-mode, one-way, HSL, exposure repair, coin overrides, and compatible
  suite scenarios. The existing dynamic mode continues to use its grow-only observed-tradability
  denominator, while current Forager selection remains based on currently tradable coins in both
  modes.

- Apple MPS single-coin EMA Anchor and Trailing Martingale optimization now supports the same
  modeled static `coin_overrides` leaves as multi-coin runs. Static values retain exact Rust's
  last-write precedence over optimizer candidates for long-only, short-only, hedge-mode, and
  one-way runs. Multi-coin base rows and per-coin override rows now also read the canonical risk
  entry-cooldown payload key, preserving configured and overridden cooldowns in GPU screening.
  GPU checkpoint identity retains the resolved float64 override values before Metal packing.

- Apple MPS multi-coin EMA Anchor and Trailing Martingale optimization now supports
  `coin_overrides.<coin>.live.forced_mode_<side>: normal` for every enabled side. Long-only,
  short-only, and fused long+short kernels reserve eligible forced-normal symbols before Forager
  ranking and expand the active-set cap when forced symbols outnumber configured positions, while
  retaining exact Rust's separately configured wallet-exposure denominator, one-way initial-entry
  exclusion, minimum-effective-cost gate, and per-coin HSL eligibility.

- Apple MPS multi-coin EMA Anchor and Trailing Martingale optimization now continues through
  declared periods where no prepared coin is valid, including gaps between disjoint coin histories
  and tails after every coin has ended. Long-only, short-only, and fused long+short kernels keep
  exact Rust's balance-only equity, HSL, exposure, restart, daily, recovery, and elapsed-time
  accounting after tracking begins, but do not activate tracking from tradability seen only before
  the requested-start guard. Validity still blocks fills, orders, and unrealized PnL. A timestep
  covered by a declared valid range whose raw H/L/C values are all NaN is likewise treated
  as non-tradable balance-only time. Single-coin kernels now apply the same rule to these internal
  gaps, and exact Rust now classifies the same row as non-tradable without triggering delist panic
  handling during mandatory validation. Only the GPU optimizer operation explicitly admits these
  internal gaps; retaining `optimize.backend: gpu` in a config does not relax normal backtest,
  suite, download, or reproduction validation. Coarser GPU candle intervals ignore complete NaN
  minutes inside mixed buckets, preserve all-gap buckets as non-tradable gaps, and retain any
  malformed non-gap price as a fail-closed aggregate. Gap rows must be strictly internal: the
  first-valid candle and any forced-delist endpoint remain valid finite prices, with the endpoint
  cutoff scaled to the prepared candle interval. Directional Metal kernels clear pending orders at
  a supported gap so they cannot fill stale pre-gap intents afterward. This enables
  account-equity and strategy-equity recovery
  metrics on such histories. Finite
  non-positive, partially invalid, or float32-unrepresentable prices remain fail-closed for GPU
  screening; HSL-enabled coins still require contiguous candles, and a forced-delist endpoint must
  remain representable because it supplies an executable close.

- Apple MPS single-coin EMA Anchor and Trailing Martingale optimization now mirrors exact Rust's
  forced-delist close when at least 1,400 prepared candles follow a coin's final valid candle,
  including adverse market slippage, directional price rounding, taker fees, panic-loss and fill
  accounting, position-duration finalization, pending-order clearing, and balance-only tail
  accounting. Multi-coin forced delists are covered by the expanded support below.

- Apple MPS multi-coin EMA Anchor and Trailing Martingale optimization now mirrors per-coin exact
  Rust forced delists for long-only, short-only, and fused long+short runs while another coin keeps
  the prepared timeline alive. Metal preserves ordinary-fill and order-generation chronology,
  closes long before short using adverse market slippage, directional price rounding, and taker
  fees, records panic-loss, realized-PnL, fill, duration, and HSL equity effects, and clears both
  sides' pending orders for the delisted coin. Forced-delist final candles that cannot be represented
  as finite positive float32 H/L/C fail before dispatch. Exact Rust remains authoritative.
  All-coins-ended tails and declared all-invalid gaps are covered by the expanded support above.

- Apple MPS HSL optimization now supports per-side
  `drawdown_worst_mean_1pct_strategy_eq_{long,short}` scoring and limits for EMA Anchor and
  Trailing Martingale across single-coin, one-sided multi-coin, and fused dual-side multi-coin
  topologies. The opt-in Metal proxy reduces full-resolution controller drawdowns to one worst
  sample per observed day, then applies a bounded logarithmic tail histogram; exact Rust
  retains each controller sample's timestamp for matching daily reduction, while exact validation
  and rolling drift gates remain authoritative.

- Apple MPS multi-coin EMA Anchor and Trailing Martingale optimization now supports staggered
  ordinary invalid candle tails when at least one prepared coin remains valid through the endpoint,
  and at least one candle whose packed float32 H/L/C values remain finite and positive inside the
  declared valid ranges covers every timestep after coverage begins. Tailed coins become
  non-tradable, contribute no unrealized PnL, and cannot leave stale orders blocking HSL, while
  portfolio and coin HSL controllers continue through the surviving timeline. Forced-delist tails
  are covered by the expanded support above, as are all-coins-ended tails and declared all-invalid
  gaps.

- Fixed live bot startup on Windows without symlink privileges by writing a visible pointer to the
  timestamped run log instead of failing while creating the stable log alias. Built-in monitor
  tooling resolves that pointer so the stable per-user path remains tail-able.

- Apple MPS single-coin HSL optimization now continues hard-stop sampling, rolling-PnL expiry,
  tier accounting, and restart checks through supported invalid candle tails. Tail candles remain
  non-tradable and cannot satisfy stale blocking orders. Multi-coin all-coins-ended tails are
  covered by the expanded support above.

- Apple MPS single-coin EMA Anchor and Trailing Martingale optimization now accepts invalid
  candles after the coin's final valid candle when HSL is disabled. Metal keeps shorter tails
  non-tradable and records balance-only account equity like exact Rust; the forced-delist boundary
  and multi-coin invalid-time behavior are covered by the expanded support above.

- Apple MPS multi-coin EMA Anchor optimization now supports any positive integer
  `backtest.candle_interval_minutes` for long-only, short-only, and fused long+short runs. Global
  and static per-coin minute spans, Forager spans, HSL decay, and cooldowns are converted to
  per-candle equivalents, while hourly windows and elapsed-day accounting retain exact Rust time
  semantics.

- Apple MPS multi-coin Trailing Martingale optimization now supports any positive integer
  `backtest.candle_interval_minutes` for long-only, short-only, and fused long+short runs. Global
  and static per-coin minute spans, Forager spans, HSL decay, and cooldowns are converted to
  per-candle equivalents, while hourly windows and elapsed-day accounting retain exact Rust time
  semantics.

- Hardened Bitunix live support: wallet balance now remains realized and stable across unrealized
  PnL changes, pending-order snapshots retain code-like venue transition statuses until
  authoritative absence, and forager candidates use native sharded one-minute Kline WebSockets
  with canonical REST startup and gap recovery, per-symbol silence detection, and symbol-scoped
  fallback when a subscription stalls or is rejected.

- Apple MPS single-coin optimization now matches exact Rust hourly volatility windows when an
  aggregated candle interval does not evenly divide an hour, retaining the boundary-crossing
  candle in the following hourly bucket instead of dropping it.

- Apple MPS single-coin EMA Anchor and Trailing Martingale optimization now supports any positive
  integer `backtest.candle_interval_minutes`. Minute-denominated strategy EMA spans, HSL EMA decay,
  and entry/HSL cooldowns are converted to per-candle equivalents for Metal while timestamp-based
  metrics and exact Rust validation retain elapsed-time semantics.

- Apple MPS multi-coin Trailing Martingale optimization now supports static per-coin
  `entry.ema_gate_mode` overrides. Initial entries and recursive reentries independently inherit
  or override their EMA gate for each coin and side, including fused long+short runs.

- Apple MPS Trailing Martingale optimization with market orders may now search entry and close
  retracement bounds that cross between recursive and trailing modes. Metal selects the mode per
  candidate and per coin, with sign-preserving float32 packing for positive underflow values;
  exact Rust validation and drift gates remain authoritative.

- Added weighted daily-series scoring and limits to Apple MPS optimization for
  `volume_pct_per_day_avg_w`, `equity_choppiness_w`, `equity_jerkiness_w`, and
  `exponential_fit_error_w`. The proxy applies Rust's ten trailing windows to its existing compact
  fill-volume and account-equity day series, excluding ambiguous partial cutoff-day volume rather
  than admitting pre-cutoff fills, while exact Rust validation and rolling drift gates remain
  authoritative.

- Added asymmetric long/short coin universes to non-suite dual-side Apple MPS multi-coin EMA
  Anchor and Trailing Martingale optimization. The proxy now preserves exact payload `entry_eligible`
  decisions as side-specific zero-exposure coin overrides, so Forager selection, dynamic wallet
  exposure, HSL, auto-unstuck, and one-way arbitration exclude the same side-ineligible coins as
  exact Rust without requiring matching long and short approved or ignored lists.

- Added baseline multi-coin Trailing Martingale ordinary market execution to Apple MPS
  optimization for long-only, short-only, fused long+short, hedge-mode, one-way, and compatible
  suite runs, with recursive or trailing entry and close mode selected per candidate and coin.
  Metal retains
  generation-time market intent, executes promoted orders on the next valid candle with adverse
  directional slippage and taker fees, applies executable-touch minimum sizing to short entries and
  all closes, and prices the strict portfolio entry cap at the market touch. For recursive closes,
  passive
  next-candle reachability alone decides whether to emit only the next close or the immutable
  recursive suffix; an emitted suffix promotes and minimum-sizes each merged group against its
  generation market before aggregate position trimming. Recursive entries retain immutable
  strategy sizing, require strict passive first-rung reachability before exposing their suffix,
  classify every emitted rung against the generation market, and preserve positive-cooldown
  single-entry staging. The portfolio entry gate globally orders first and recursive suffix rungs,
  removes the farthest orders with exact deterministic ties, and may retain one minimum-valid
  partial boundary strictly below the TWEL cap. Static coin overrides may select trailing or
  recursive entry and close mode independently. HSL and forager selection remain supported.
  Position- and total-exposure repair may now promote to market execution, resize against the
  executable-touch minimum before finalized reducer selection, and retain adverse slippage plus
  taker fees. Auto-unstuck now follows the same contract: the one globally selected reducer may be
  enabled by the side template or a static coin override, is resized at the executable touch while
  preserving the independently generated ordinary recursive ladder, and fills with adverse
  slippage and taker fees. Realized-loss gating now also composes with market execution: proxy
  screening projects adverse slippage and taker fees for ordinary closes and protective reducers,
  while exact Rust remains authoritative for the shared peak-balance allowance.

- Added baseline multi-coin EMA Anchor ordinary market execution to Apple MPS optimization for
  long-only, short-only, and fused long+short runs. The proxy retains generation-time entry and
  ordinary-close intent, executes it on the next valid candle with adverse directional slippage
  and taker fees, applies executable-touch minimum sizing, and prices total-exposure entry
  allocation at the market touch. HSL, auto-unstuck, TWEL repair, static unstuck overrides, and
  realized-loss gating may coexist with this path. Metal finalizes TWEL and unstuck candidates
  against ordinary closes, ranks their executable quantities globally across symbols and sides,
  advances only a rejected position to its fallback, and spends one shared loss allowance before
  gating ordinary closes. These generation-time decisions include market slippage and taker fees
  and remain attached to pending orders.

- Added single-coin Trailing Martingale recursive-close market execution to Apple MPS
  optimization. Exact passive next-candle expansion remains authoritative: a market-only next
  close does not reveal the recursive suffix, while an expanded immutable ladder classifies and
  executable-touch-sizes every emitted price group against its generation market snapshot.
  Pre-gate WEL reachability still controls expansion when that reducer is later loss-gated.
  Passive WEL quantity seeds later rungs before executable-touch resizing, and a same-price WEL
  merges into the following ordinary group even when TWEL or unstuck wins reducer selection.
  Expanded ladders independently re-finalize every WEL, TWEL, and unstuck request before reducer
  selection, retry the next finalized candidate when the preferred reducer is loss-gated, and gate
  promoted reducers and grid groups at their generation-time projected market price while retaining
  the next-candle price for fills. Reducer loss budgets retain their generation-time realized-PnL
  snapshot, and below-minimum reducers are removed while the ordinary ladder remains normalized and
  is reallocated when another close remains executable. Promoted grid groups and protective reducers
  retain canonical ordering,
  aggregate position trimming, quantity-relative minimum-size comparisons, adverse slippage, and
  taker fees. Entry and close optimizer bounds may cross the recursive/trailing mode boundary.

- Added single-coin Trailing Martingale recursive-entry market execution to Apple MPS optimization.
  Every immutable strategy-ladder rung is independently promoted against its generation market
  snapshot after the original passive rung is next-candle reachable, short entries are resized to
  the executable minimum, and executable ladder quantities stream through the strict
  total-exposure entry gate at their limit price or market touch. Strategy-ladder sizing retains
  its separate wallet-exposure allowance before that portfolio gate is applied; the retained
  nearest prefix ends at the first partially cropped portfolio-boundary rung. Entry optimizer
  bounds may cross the recursive/trailing mode boundary.

- Extended single-coin Trailing Martingale ordinary market-order optimization on Apple MPS to
  compatible HSL modes, one-sided minimum-effective-cost filtering, auto-unstuck, position- and
  total-exposure repair, and realized-loss gating. Market-promoted reducers are resized at
  executable touch before selection and aggregate allocation, and adverse slippage plus taker fees
  participate in the conservative loss projection.

- Added baseline ordinary market-order execution to Apple MPS optimization for single-coin
  Trailing Martingale, covering long, short, and dual-side near-touch trailing entries and closes,
  executable-touch minimum sizing, adverse slippage, and taker fees.

- Extended single-coin EMA Anchor ordinary market-order optimization on Apple MPS to compatible
  HSL modes, one-sided minimum-effective-cost filtering, auto-unstuck, total-exposure repair, and
  realized-loss gating. Market-promoted closes are resized at executable touch before reducer
  selection and aggregate/loss gating, with adverse slippage and taker fees included in projected
  loss. Active HSL panic closes participate in reducer ordering without consuming shared loss
  allowance, matching exact Rust's execution ordering.

- Added baseline ordinary market-order execution to Apple MPS optimization for single-coin EMA
  Anchor, including near-touch promotion, next-candle adverse slippage, taker fees, executable-touch
  sizing, directional and one-way modes, and compatible suites. Auto-unstuck, exposure repair,
  realized-loss gating, and Trailing Martingale were added in later slices; baseline multi-coin
  EMA Anchor support is now also available under the restrictions described above.

- Added Apple MPS suite scenario overrides for `live.hsl_signal_mode`. Coin, position-side, and
  unified HSL signal topology may now vary by scenario when each effective scenario passes the
  existing Metal topology and per-coin override checks.

- Expanded Apple MPS suite scenario overrides to the already-modeled taker-fee, market-order
  slippage, minimum-effective-cost filtering, and PnL-lookback execution settings. Every effective
  scenario is still scope-validated independently, so unsupported side/coin combinations fail
  before optimization.

- Hardened Apple MPS optimizer checkpoint/resume identity. GPU checkpoints now bind the complete
  fixed and tunable search shape and each prepared single-run or suite-scenario proxy execution
  contract, including prepared candle-value and timestamp hashes, starting balance, valid/trade
  windows, resolved fixed proxy parameters, liquidation and exposure policy, resolved market
  settings and fees, modeled execution settings, and effective NSGA-II population and variation
  controls. Resume now fails closed after any incompatible input change; older GPU checkpoints are
  intentionally invalidated.

- Added Apple MPS GPU optimizer support for dual-side multi-coin one-way mode in EMA Anchor and
  Trailing Martingale, with per-symbol initial-entry arbitration matching exact Rust.

- Added global strategy-equity recovery-distribution scoring and limits to Apple MPS optimization
  for EMA Anchor and Trailing Martingale across single-coin, one-sided multi-coin, and fused
  shared-account dual-side multi-coin runs. Strategy kernels emit opt-in candidate-relative hourly
  strategy-equity samples with mandatory initial and terminal/liquidation endpoints, and a bounded
  Metal postprocessor applies Rust's strict
  time-to-exceed, percentile, and worst-tail definitions for
  `strategy_eq_recovery_days_{mean,median,p95,p99,mean_worst_5pct,mean_worst_1pct}`. Exact Rust
  validation and rolling drift gates remain authoritative. Internal all-NaN H/L/C gaps
  contribute balance-only samples, and a bounded single-coin coin-HSL rolling-PnL overflow emits a
  conservative full-horizon recovery penalty. Independent dual-side multi-coin summaries remain
  fail closed
  because they cannot reconstruct one shared portfolio-equity curve.

- Added raw per-side HSL strategy-equity worst-drawdown scoring and limits to Apple MPS
  optimization for EMA Anchor and Trailing Martingale, across single-coin and multi-coin
  topologies. The bounded peak-and-maximum state is compiled into Metal only when either the long
  or short metric is requested; exact Rust validation and rolling drift gates remain authoritative.

- Added mean-worst-1% EMA-smoothed HSL strategy-equity drawdown scoring and limits to Apple MPS
  optimization for the account, long side, and short side. The Metal proxy uses an opt-in bounded
  logarithmic histogram so normal GPU runs do not pay the per-candidate state cost; exact Rust
  validation and rolling drift gates remain authoritative.

- Added per-side HSL strategy-equity peak-recovery scoring and limits to Apple MPS optimization.
  EMA Anchor and Trailing Martingale kernels now retain the longest strict time-to-exceed interval
  for each long and short HSL controller, including an unrecovered tail through the backtest end,
  and expose it in hours and days. Exact Rust validation remains authoritative.

- Fixed Apple MPS optimization halting when a converged or single-objective proxy front repeats an
  already exact-validated candidate. The backend now revalidates that actual current-front member
  (or waits for its in-flight exact job) instead of relabeling an off-front candidate, preserving
  the proxy-front drift gate and exact Rust authority while allowing the search to continue.

- Added worst EMA-smoothed HSL strategy-equity drawdown scoring and limits to Apple MPS
  optimization for the account, long side, and short side. Metal retains each controller's
  maximum across cooldown restarts and coin-mode controllers, then applies the same account-level
  `max(long, short)` reduction as exact Rust. Exact Rust validation remains authoritative.

- Accelerated HSL-disabled, one-sided single-coin Trailing Martingale optimization on Apple MPS
  with Rust-owned long-only and short-only Metal variants. The backend selects these variants
  automatically, logs the selected topology, and retains the generic kernel for dual-side or
  HSL-enabled runs. Exact Rust validation and the existing drift gates remain authoritative.

- Bounded Apple MPS proxy command buffers by candidate-candle workload so large populations and
  long histories cannot monopolize the shared display GPU in one Metal dispatch. The configured
  population and batch retain their optimization semantics while the backend transparently splits
  oversized dispatches, polls Ctrl+C between them, and retains the last complete ask/tell
  checkpoint if an in-progress generation is interrupted.

- Added fused dual-side multi-coin auto-unstuck screening to Apple MPS optimization for EMA Anchor
  and Trailing Martingale, including static per-coin overrides and compatible suites. The shared
  account kernel admits one global least-stuck candidate across both directional surfaces using
  exact Rust's price-difference, symbol-index, and long-before-short tie ordering, then applies the
  existing conservative realized-loss allowance. Exact Rust validation and drift gates remain
  authoritative.

- Enabled fused dual-side multi-coin exposure repair in Apple MPS optimization. EMA Anchor and
  Trailing Martingale now support per-side TWEL repair in shared-account runs, and Trailing
  Martingale also supports its per-position WEL reducer and static coin overrides. Each directional
  surface computes its exact-Rust-style action set from the same pre-fill shared account snapshot;
  exact Rust validation and the existing drift gates remain authoritative.

- Fixed finite `live.pnls_max_lookback_days` handling for single-coin `coin`-mode HSL on Apple
  MPS. EMA Anchor and Trailing Martingale screening now expire realized-PnL fill events with the
  same rolling-window rule as exact Rust instead of retaining an all-history peak; bounded GPU
  scratch overflow fails the affected candidate closed, and exact Rust remains authoritative.

- Added fused shared-account dual-side multi-coin Trailing Martingale screening to Apple MPS
  optimization. Long and short candidates now run in one portfolio kernel for unified, pside,
  and coin HSL, including shared event/tier metrics, directional PnL, coin overrides, forager
  selection, and the existing TM entry/close behavior. Exact Rust remains authoritative.

- Added fused shared-account dual-side multi-coin EMA Anchor screening to Apple MPS optimization.
  Unified, pside, and coin HSL now run inside one portfolio kernel, including per-coin HSL
  overrides and shared event/tier scoring metrics. Exact Rust remains authoritative.

- Added dual-side single-coin `unified` HSL to Apple MPS optimization for EMA Anchor and
  Trailing Martingale. Both Metal controllers now consume account-wide realized and unrealized
  PnL and require positions and blocking orders on both sides to be flat before ending a RED
  episode, matching exact Rust. Dual-side multi-coin `coin` and `unified` remain fail closed.

- Fixed exact Rust backtests so `unified` HSL confirms the whole account is flat before ending a
  RED episode. A position or blocking order on either side now prevents both unified controllers
  from halting or beginning cooldown, matching the documented live HSL scope.

- Added dual-side multi-coin `pside` HSL lifecycle and panic-loss scoring/limits to Apple MPS
  optimization. The proxy now reduces directional episode counters, durations, drawdowns,
  restarts, and absolute panic losses with the same aggregate formulas used by exact Rust, and
  replays HSL summaries through the conservative combined liquidation cutoff. Exact Rust remains
  authoritative. Account-normalized event losses and yellow/orange/RED time percentages remain
  fail closed until the proxy owns shared event-level equity and minute-level tier state.

- Added dual-side multi-coin HSL behavior to Apple MPS optimization for EMA Anchor and Trailing
  Martingale in `pside` signal mode. Each directional Metal controller owns its exact pside
  realized-plus-unrealized signal while the existing hedged portfolio reducer retains its
  conservative shared-equity liquidation cutoff; exact Rust validation remains authoritative.
  Dual-side multi-coin `coin` and `unified` modes, and dual-side multi-coin HSL scoring metrics,
  remain fail closed until a fused account controller can model their shared state.

- Added all ten canonical per-coin HSL setting overrides to one-sided multi-coin Apple MPS
  optimization for EMA Anchor and Trailing Martingale in `coin` signal mode. Each Metal coin
  controller now resolves its own enablement, thresholds, EMA span, cooldown, restart policy,
  tier ratios, orange behavior, and limit/market panic execution. Compatible suites may also
  supply scenario-local HSL coin overrides. Exact Rust validation remains authoritative;
  per-coin HSL overrides outside this topology fail closed.

- Added `coin` signal mode to one-sided multi-coin HSL Apple MPS optimization for EMA Anchor
  and Trailing Martingale. Metal now maintains an independent HSL episode per coin, including
  realized net PnL and fees, dynamic effective-slot scaling, warning/RED state, panic flattening,
  halt/restart lifecycle, and per-episode metrics. Directional outputs aggregate the same way as
  exact Rust: episode metrics are combined across coins while time-in-tier uses the worst active
  coin tier once per minute. Exact Rust validation remains authoritative; dual-side multi-coin
  `coin` mode remains fail closed.

- Added market panic-close execution to one-sided multi-coin HSL Apple MPS optimization for EMA
  Anchor and Trailing Martingale. The portfolio proxy now fills every panic close on the next
  tradable bar using directionally quantized close-price slippage and each coin's taker fee,
  matching the exact Rust backtest execution contract. Exact Rust validation remains
  authoritative.

- Added one-sided multi-coin HSL to Apple MPS optimization for EMA Anchor and Trailing
  Martingale in `unified` and `pside` signal modes, including compatible suites. Each Metal
  candidate now applies one shared-balance portfolio HSL controller across every coin on the
  enabled side, including warning tiers, RED entry blocking, limit panic flattening, cooldown
  restart, lifecycle metrics, and panic-loss metrics. Exact Rust validation remains authoritative;
  dual-side multi-coin HSL is currently limited to `pside` behavior without HSL scoring metrics.

- Added dual-side single-coin HSL to Apple MPS optimization for EMA Anchor and Trailing
  Martingale in `coin` and `pside` signal modes, including compatible suites. Metal now tracks
  realized net PnL independently for long and short while retaining shared account balance and
  exact Rust validation. Dual-side `unified` HSL remains fail closed until its documented
  account-wide flatten contract and exact-backtest finalization scope are reconciled. Dual-side
  multi-coin HSL currently supports only `pside` behavior without HSL scoring metrics.

- Consolidated the supported one-sided single-coin HSL Apple MPS screening behavior into one
  Rust-owned Metal controller shared by EMA Anchor and Trailing Martingale. Signal modes now use
  explicit unified, pside, and coin identities, with direct M3 trace conformance against the exact
  Rust HSL runtime. That foundation did not itself expand GPU support.

- Added directional close-fill PnL scoring and limits to Apple MPS optimization:
  `loss_profit_ratio_long`, `loss_profit_ratio_short`, `pnl_ratio_long_short`, and its
  `long_short_profit_ratio` alias. The proxy preserves long/short gross profit and loss separately
  across supported single- and multi-coin topologies and applies exact Rust neutral/cap formulas;
  exact Rust validation remains authoritative.

- Added account-equity peak-recovery hours and days scoring and limits to Apple MPS optimization
  for single-coin and one-sided multi-coin EMA Anchor and Trailing Martingale runs. The proxy
  reuses Metal's full-resolution completed peak-to-peak recovery accumulator; exact Rust
  validation remains authoritative.

- Added mean position-holding time and positions-held-per-day scoring and limits to Apple MPS
  optimization for single-coin and one-sided multi-coin EMA Anchor and Trailing Martingale runs.
  Metal records each completed position duration plus every open tail with constant-size sum and
  count accumulators; exact Rust validation remains authoritative.

- Added daily account-equity choppiness, jerkiness, and exponential-fit-error scoring and limits
  to Apple MPS optimization. The proxy applies the exact Rust formulas to its existing active daily
  closing-equity surface for supported EMA Anchor and Trailing Martingale topologies.

- Added peak-recovery day/hour scoring for strategy equity and realized PnL to Apple MPS
  optimization for single-coin and one-sided multi-coin EMA Anchor and Trailing Martingale runs.
  Metal tracks realized-PnL recovery intervals per candidate while exact Rust validation remains
  authoritative.

- Added active-symbol count and top-symbol fill-share scoring and limits to Apple MPS optimization
  for single-coin and one-sided multi-coin EMA Anchor and Trailing Martingale runs. Multi-coin
  kernels emit per-symbol fill counts only when either metric is requested, preserving the normal
  proxy's buffer and transfer cost; exact Rust validation remains authoritative.

- Added analyzed-start-anchored active fill-day count and ratio scoring and limits to Apple MPS
  optimization for single-coin and one-sided multi-coin EMA Anchor and Trailing Martingale runs.
  Metal counts distinct 24-hour fill buckets within the candidate's analyzed equity window; exact
  Rust validation remains authoritative, and dual-side multi-coin runs fail closed at the existing
  intraday shared-liquidation boundary.

- Added entry/close and long/short fill counts and daily rates, entry-to-close ratio, and
  per-configured-position-slot fill rates to Apple MPS optimization for single-coin and one-sided
  multi-coin EMA Anchor and Trailing Martingale runs. Metal records every proxy fill by role and
  side, while Python applies each candidate's configured active position-slot denominators using
  the exact Rust averaging contract. Exact Rust validation remains authoritative, and dual-side
  multi-coin runs fail closed at the existing intraday shared-liquidation boundary.

- Added `fills_count`, `fills_analysis_duration_days`, and `fills_per_day` scoring and limits to
  Apple MPS optimization for single-coin and one-sided multi-coin EMA Anchor and Trailing
  Martingale runs. The proxy reuses Metal's authoritative per-fill daily counts and the analyzed
  equity timestamp span; exact Rust validation and drift gates remain authoritative. Dual-side
  multi-coin runs fail closed because independent directional summaries cannot reconstruct the
  intraday shared-liquidation cutoff.

- Added weighted `adg_pnl_w`, `mdg_pnl_w`, `sharpe_ratio_pnl_w`, and
  `sortino_ratio_pnl_w` scoring and limits to Apple MPS optimization for single-coin and one-sided
  multi-coin runs. Metal counts every proxy fill, including multiple same-candle ladder fills, so
  the reducer can reproduce Rust's full-run minimum fill count and empty-suffix behavior across the
  ten weighted windows. Exact Rust validation and drift gates remain authoritative; dual-side
  multi-coin runs retain the existing fail-closed shared-liquidation boundary.

- Added `adg_pnl`, `mdg_pnl`, `sharpe_ratio_pnl`, and `sortino_ratio_pnl` scoring and limits to
  Apple MPS optimization. Metal emits each UTC fill day's realized balance change and last fill
  balance, matching Rust's collateral-agnostic daily PnL ratio contract for single-coin and
  one-sided multi-coin runs. Dual-side multi-coin runs remain fail closed because independent
  directional summaries cannot reconstruct an intraday shared-liquidation cutoff. Exact Rust
  validation and drift gates remain authoritative.

- Added the canonical USD gain, ADG, MDG, weighted ADG, and weighted MDG per-configured-exposure
  metrics for both long and short sides to Apple MPS optimization. They reuse the validated
  strategy-equity proxy reductions and divide by each candidate's effective side
  `total_wallet_exposure_limit`, including exact-last suite overrides; a zero-exposure side retains
  the CPU contract's zero value.

- Added the canonical USD account-equity scoring aliases for gain, ADG, MDG, Sharpe, Sortino,
  Omega, expected shortfall, Calmar, Sterling, worst drawdown, and worst-1% drawdown, including the
  available weighted variants, to Apple MPS optimization. With BTC collateral disabled, these
  aliases reuse the already validated strategy-equity proxy series while exact Rust metrics remain
  authoritative. Also added `exposure_ratio_usd` and `exposure_mean_ratio_usd` for single-coin and
  one-sided multi-coin runs; dual-side multi-coin runs fail closed because independent directional
  kernels cannot reconstruct net portfolio exposure.

- Added `total_wallet_exposure_max` and `total_wallet_exposure_mean` scoring and limits to Apple
  MPS optimization for EMA Anchor and Trailing Martingale across single-coin, one-sided multi-coin,
  and compatible suite runs. Metal samples absolute net long-minus-short exposure after each
  non-liquidating equity update, matching Rust's analysis series timing. Dual-side multi-coin runs
  fail closed because independent directional kernels cannot reconstruct the minute-level net
  portfolio exposure; exact Rust validation and drift gates remain authoritative.

- Added `entry_initial_balance_pct_long` and `entry_initial_balance_pct_short` scoring and limits
  to Apple MPS optimization for EMA Anchor and Trailing Martingale across single-coin,
  one-sided multi-coin, and compatible suite runs. Metal derives the value from each candidate's
  effective position count, total exposure, initial quantity, bounded or legacy excess allowance,
  and first-coin override precedence, while exact Rust validation remains authoritative. Dual-side
  multi-coin runs fail closed for these metrics because independent directional summaries cannot
  truncate their effective coin counts at shared portfolio liquidation.

- Added `position_unchanged_hours_max` and `position_unchanged_days_max` scoring and limits to
  Apple MPS optimization for EMA
  Anchor and Trailing Martingale across long, short, dual-side, single-coin, multi-coin, and
  compatible suite runs. Metal tracks the latest fill separately for each coin and position side,
  including the open tail to the final analyzed sample. Dual-side multi-coin runs reject both held-
  and unchanged-duration metrics because independent directional maxima cannot be truncated at a
  shared portfolio liquidation; exact Rust validation and drift gates remain authoritative.

- Added Apple MPS optimizer scoring and limits for `position_held_hours_max` and
  `peak_recovery_hours_strategy_eq`, including its legacy `peak_recovery_hours_hsl` alias. These
  are exact hour-denominated views of the already supported Rust-compatible duration metrics and
  require no additional Metal approximation. Dual-side multi-coin recovery metrics remain fail
  closed because independent directional summaries cannot reconstruct portfolio recovery.

- Added `loss_profit_ratio` scoring and limits to Apple MPS optimization for EMA Anchor and
  Trailing Martingale across long, short, and dual-side single-coin runs and one-sided multi-coin
  runs. Metal accumulates gross winning and losing close-fill PnL, excluding entry and close fees
  to match Rust's `Fill.pnl` analysis contract. Dual-side multi-coin runs remain fail closed because
  independent directional totals cannot be truncated at a shared portfolio liquidation. Exact
  Rust validation and drift gates remain authoritative.

- Added finite `live.pnls_max_lookback_days` support to the existing one-sided single-coin HSL
  Apple MPS optimizer path. Metal deliberately retains an all-history candidate-local peak as a
  conservative envelope over Rust's rolling peak, so it may trigger HSL early after an old peak
  expires but cannot hide a drawdown for that reason. Exact Rust validation and drift gates remain
  authoritative.

- Added HSL market panic-close execution to the supported one-sided single-coin EMA Anchor and
  Trailing Martingale Apple MPS optimizer path. The Metal proxy guarantees the persisted panic
  order on the next valid bar, uses that bar's close with directionally adverse configured
  slippage and price-step rounding, and charges the resolved taker fee. Resting-limit behavior is
  unchanged, while exact Rust validation and drift gates remain authoritative.

- Added resting-limit HSL panic-loss scoring and limit metrics to the supported one-sided
  single-coin Apple MPS optimizer path: panic-close loss sum/max, per-episode loss drawdown
  min/mean/max, and halt-to-restart equity loss. The proxy tags only HSL panic fills and retains
  exact Rust validation and drift gates as authoritative. Trailing Martingale panic closes now
  remain one exclusive full-position order and bypass the ordinary realized-loss gate, matching
  Rust instead of being reinterpreted as recursive close-grid rungs. Directional proxy filtering
  retains every HSL lifecycle and panic-loss accumulator; missing directional output and requests
  for these metrics on multi-coin kernels fail closed instead of substituting zeros.

- Added non-loss HSL lifecycle scoring and limit metrics to the supported one-sided single-coin
  Apple MPS optimizer path: trigger/restart counts and yearly rates, warning-tier occupancy, halt
  and flatten durations, trigger drawdown, and post-restart retriggers. Exact Rust validation and
  drift gates remain authoritative; HSL strategy-equity time-series metrics remain fail closed.

- Added the first HSL slice to Apple MPS optimization for one-sided single-coin EMA Anchor and
  Trailing Martingale runs, in both long and short directions and compatible suites. The Metal
  proxy models coin, pside, and unified drawdown signals; tunable RED threshold, EMA span, and
  cooldown; yellow/orange entry suppression; RED latching; resting-limit panic flattening;
  flat confirmation; positive-cooldown restart; zero-cooldown indefinite halt; cumulative
  no-restart peak tracking; effective coin-slot scaling; and terminal no-restart. The initial
  slice requires all-history PnL peaks and contiguous valid candles. Market panic execution,
  finite rolling PnL lookbacks, dual-side and multi-coin HSL, per-coin HSL overrides, and HSL
  strategy-equity time-series metrics remain fail closed.

- Added auto-unstuck to Apple MPS EMA Anchor and Trailing Martingale optimization for single-coin
  long-only, short-only, dual-side hedge/one-way, and compatible suite runs, plus one-sided
  multi-coin runs and suites with static per-coin overrides. The Metal proxy models EMA gating,
  one global least-stuck selector across the enabled portfolio, allowance-based loss sizing,
  exchange minimums, competition with WEL/TWEL and ordinary closes, and the realized-loss gate.
  Its all-history realized-PnL envelope is conservative relative to exact Rust's configured rolling
  lookback; exact validations and the existing classification, rank, and drift gates remain
  authoritative.

- Expanded Apple MPS optimizer scoring and limits with fill-gap mean, median, p95, and p99 hours.
  The proxy conservatively decodes its existing logarithmic inter-fill histogram at a float32-safe
  upper edge, adds exact leading and trailing gaps, and coalesces same-candle fills; exact
  Rust remains authoritative. Dual-side multi-coin runs keep these metrics fail closed because
  independent directional summaries cannot reconstruct portfolio fill timing.

- Added realized-loss gating to Apple MPS EMA Anchor and Trailing Martingale screening
  for long, short, hedge-mode, and one-way runs. Single-coin EMA Anchor tracks a conservative
  all-history peak-relative realized net-PnL budget, including maker fees, and blocks lossy ordinary
  or exposure-repair closes that exceed it. One-sided multi-coin EMA Anchor and Trailing
  Martingale allow only their single selected auto-unstuck reducer to consume a conservative
  realized-loss budget; other closes and dual-side multi-coin dispatches retain a stricter zero-loss
  proxy envelope, avoiding unsafe cross-dispatch loss-budget reservation and per-candle enumeration
  of TM's recursive 500-rung ladder. Multi-coin TM preserves the exact TWEL action set before loss
  screening, so a blocked reducer is not reallocated to another symbol, and screens reachable
  recursive close groups independently so later profitable rungs remain available when an earlier
  rung is blocked. Exact Rust remains authoritative for the configured rolling PnL lookback and
  allowance.

- Expanded Apple MPS optimizer scoring and limits with weighted strategy-equity MDG, Sharpe,
  Sortino, Omega, Calmar, and Sterling metrics. The proxy maps the exact optimizer's ten-subset
  averaging schedule onto its existing compact daily Metal summaries and skips these additional
  reductions when none of the weighted metrics is requested; exact Rust validation remains
  authoritative.

- Expanded Apple MPS optimizer scoring and limits with strategy-equity gain, Omega ratio,
  expected shortfall, Calmar ratio, Sterling ratio, and median underwater percentage. These
  metrics are reduced from the existing compact Metal equity summaries, while exact Rust
  backtests remain authoritative for persisted results and Pareto membership.

- Added EMA Anchor side-wide total-exposure repair to multi-coin Apple MPS optimization for
  long-only, short-only, and compatible suite runs. The Metal proxy models both
  `reduce_overweight` and `reduce_portfolio`, ranks every open position by projected adverse loss,
  uses the current eligible-position count and last valid delisted-coin price, and reserves the
  protective reducer before independently reachable ordinary EMA closes. Exact Rust validation
  and the existing classification, rank, and drift gates remain authoritative. Dual-side
  multi-coin repair remains fail closed until a shared-balance portfolio kernel can preserve exact
  cross-side sizing.

- Added EMA Anchor side-wide total-exposure repair to single-coin Apple MPS optimization for
  long-only, short-only, shared-balance dual-side, and compatible suite runs. The Metal proxy
  models the canonical TWEL reducer price and size, reserves the protective reducer before
  trimming the ordinary EMA close, and executes independently reachable closes in canonical
  order. Exact Rust validation and the existing classification, rank, and drift gates remain
  authoritative.

- Added Trailing Martingale side-wide total-exposure repair to the Apple MPS optimizer for
  single- and multi-coin long-only, short-only, and compatible suite runs. The Metal
  proxy models both `reduce_overweight` and `reduce_portfolio`, ranks repair candidates by projected
  adverse loss, applies exchange minimums and quantity steps, and lets the largest WEL/TWEL reducer
  compete before rebuilding the ordinary close ladder. Exact Rust validation and the existing
  classification, rank, and drift gates remain authoritative. Dual-side multi-coin exposure repair
  remains fail closed until a shared-balance portfolio kernel can preserve exact cross-side sizing.

- GPU optimization now latches Ctrl+C received during a native Metal dispatch,
  stops before another generation or exact-validation submission, saves a
  resumable checkpoint, and then cleans up optimizer workers and shared memory.

- Added Trailing Martingale per-position exposure repair to the Apple MPS optimizer for single- and
  multi-coin long-only, short-only, dual-side, and compatible suite runs. The Metal proxy models
  the canonical enable toggle and tunable threshold, gives the passive repair close precedence
  over normal strategy closes, reduces strictly below the allowance-adjusted WEL target, and
  honors static per-coin enable/threshold overrides. Exact Rust validation and the existing
  classification, rank, and drift gates remain authoritative. EMA Anchor position repair remains
  fail closed.

- Extended Apple MPS exposure-headroom support to multi-coin EMA Anchor and Trailing Martingale
  optimization, including long-only, short-only, dual-side hedge, suites, tunable allowance and
  TWEL-entry thresholds, and per-coin allowance percentage overrides under the globally configured
  bounded or legacy-raw mode. The Metal proxy now separates per-symbol allowed wallet exposure
  from the optional side-wide TWEL entry gate;
  exact Rust validation and the existing classification, rank, and drift gates remain
  authoritative.

- Added single-coin exposure-headroom policy support to the Apple MPS optimizer for EMA Anchor and
  Trailing Martingale, including long-only, short-only, dual-side, and compatible suite runs.
  Metal now models bounded and legacy-raw `we_excess_allowance_pct`, the
  `total_exposure_entry_gate_enabled` toggle, and `total_exposure_enforcer_threshold`; exact Rust
  backtests and the existing classification, rank, and drift gates remain authoritative.

- Added `backtest.filter_by_min_effective_cost` support across the Apple MPS optimizer's complete
  EMA Anchor and Trailing Martingale topology matrix: single- and multi-coin, long, short,
  dual-side, and compatible suites. The Metal proxy conservatively compares projected initial cost
  against the highest executable exchange minimum in each prepared coin window, using effective
  wallet-exposure limits, static per-coin overrides, a downward arithmetic bound for the float32
  projection, and the liquidation floor as a lower cash-balance bound while the whole portfolio is
  flat. Once any position is open, Metal rejects a candidate that exact Rust may admit, or an
  independently selected multicoin/dual-side candidate set has been generated, other flat slots
  fail closed because the proxy can no longer prove that the equity floor also bounds exact cash.
  This uncertainty is applied immediately where needed and remains sticky even if the proxy remains
  or becomes flat again; failing
  candidates are removed before Forager selection and one-way arbitration while every open
  position remains managed. Exact Rust retains its current-close rule and remains
  authoritative through the normal validation and drift gates. A finite positive liquidation
  threshold is required; concurrent-slot runs may halt when the conservative false negatives push
  proxy/exact rank agreement below the configured safety threshold.

- Added static per-coin Trailing Martingale overrides to single-side and dual-side multi-coin Apple
  MPS optimization and compatible suites. The Metal proxy consumes exact-last per-coin strategy,
  entry-cooldown, and wallet-exposure values; checkpoint identity records each resolved override
  matrix, while exact Rust backtests and the normal drift gates remain authoritative. Unsupported
  override leaves continue to fail before optimization begins.

- Added dual-side hedge-mode multi-coin Trailing Martingale optimization and compatible suites to
  the experimental Apple MPS backend. Each candidate receives independent long and short Metal
  screening dispatches which feed the existing conservative combined-equity proxy; exact Rust
  portfolio backtests and the normal classification, rank, and drift gates remain authoritative.
  One-way dual-side arbitration remains unsupported.

- Added long-only and short-only multi-coin Trailing Martingale optimization to the experimental
  Apple MPS backend, including compatible suites. A dedicated Rust-owned Metal kernel combines
  per-coin trailing-martingale state with the existing dynamic wallet-exposure and Forager
  portfolio model; exact Rust backtests remain authoritative and the existing constraint, rank,
  and drift gates fail closed on proxy disagreement.

- Added canonical combined multi-exchange datasets and per-coin source assignments to Apple MPS
  optimizer suites. Metal consumes the same prepared per-coin candles and market settings as exact
  Rust, and checkpoint identity now records each coin's resolved OHLCV and market-settings source.
  Individual-exchange scenarios fail closed if an assignment for one of their prepared coins
  selects another exchange.

- Fixed optimizer-suite exchange routing so an explicitly restricted scenario uses its requested
  individual exchange dataset even when only that exchange needed separate materialization. It no
  longer falls through to a combined dataset whose candles may come from another base exchange.

- Expanded Apple MPS optimizer suites with fail-closed scenario-local overrides for modeled
  runtime inputs: `coin_overrides`, starting balance, maker fee, liquidation threshold, Forager
  hysteresis, and hedge mode. Other non-bot overrides and per-coin source routing remain rejected.

- Added Apple MPS optimizer suites spanning exchanges, while retaining exactly one exchange per
  scenario and rejecting combined or per-coin source datasets.

- Added static per-coin EMA Anchor strategy, entry-cooldown, and wallet-exposure overrides to
  dual-side hedge-mode Apple MPS optimization, including compatible multi-coin suites.

- Added Apple MPS optimizer suite support for dual-side hedge-mode multi-coin EMA Anchor scenarios
  sharing one exchange and a consistent long/short topology.

- Added Apple MPS multi-coin EMA Anchor optimizer support for Forager score hysteresis, retaining
  flat incumbent candidates when challenger scores are only marginally better.

- Add dual-side hedge-mode multi-coin EMA-anchor optimization to the experimental Apple MPS
  backend. Metal screens long and short independently and combines their compact outputs into a
  conservative portfolio proxy, while unchanged exact Rust backtests remain authoritative and the
  existing constraint, rank, and drift gates fail closed on disagreement. Dual-side one-way mode,
  suites, coin overrides, and metrics requiring cross-side fill or recovery event streams remain
  explicitly unsupported in this slice.

- Add static per-coin overrides to experimental Apple MPS multi-coin EMA-anchor optimization.
  The enabled side may override EMA-anchor parameters, entry cooldown, and an explicit per-coin
  wallet-exposure limit. Metal applies those values after every candidate gene, matching exact
  Rust precedence; unsupported override leaves still fail closed, and checkpoint identity now
  includes the prepared effective override table.

- Add `-s/--save-selected` and `-f/--save-filtered` to `passivbot tool pareto` for copying the
  selected member or the post-limit member set, with fail-if-present destinations and a filtered
  export manifest.

- Add experimental Apple MPS optimization suites for the existing single-coin EMA-anchor and
  trailing-martingale scopes. Metal screens every candidate against each prepared scenario, while
  the canonical suite reducer, scenario-aware objectives, and limits select proxy candidates and
  unchanged exact Rust suite evaluations remain authoritative. This slice supports scenario date,
  coin, ignored-coin, and single-exchange selection. Scenario `bot.long`/`bot.short` overrides now
  retain exact last-write precedence by shadowing affected Metal candidate parameters per scenario
  and revalidating each effective scenario against the GPU scope; non-bot override paths,
  multi-exchange suites, multi-coin scenarios, and per-coin source assignments still fail closed.
  Effective external suite definitions, scenario filters, overrides, and resolved date windows are
  persisted and checked on resume.

- Extend experimental Apple MPS EMA-anchor suites to multi-coin scenarios on one shared exchange.
  Scenarios may select different coin subsets and independently dispatch to the single-coin or
  multicoin Metal kernel, while the canonical suite reducer and exact Rust validations remain
  authoritative. Multicoin suites require every effective scenario to share one enabled side, and
  each scenario revalidates `n_positions` against its own prepared coin count.

- Apply `optimize.fixed_runtime_overrides` to experimental Apple MPS candidates in the same order
  as the exact CPU optimizer. Fixed values shadow corresponding Metal search genes, participate in
  durable candidate hashing, and remain subordinate to later `optimize.enable_overrides`; the
  effective config still fails closed on unsupported GPU behavior. Config normalization now
  preserves documented user-defined dotted leaf paths instead of silently replacing them with
  schema defaults, rejects path aliases that collide or replace mappings, and validates exact
  finalized boundary configs before either optimizer backend starts. Fixed values that disable
  dependent trailing-martingale parameters remove and hash-canonicalize those dead GPU genes.

- Apply the V8 `optimize.enable_overrides` candidate contract in the experimental Apple MPS
  optimizer. `mirror_short_from_long` now mirrors each proxy candidate after anchor and tunable
  values are resolved, and `lossless_close_trailing` raises each trailing-martingale close
  threshold to its candidate retracement before Metal screening. Exact Rust remains authoritative;
  legacy trailing-grid override modes fail closed because their strategy is not supported by the
  GPU backend.

- Extend the experimental Apple MPS optimizer to anchored fine-tuning with `--start` plus
  `--fine-tune-params` for supported EMA-anchor and trailing-martingale scopes. The Metal proxy
  evolves the same discrete anchor id as exact Rust, applies each anchor's fixed optimizer-bound
  values before candidate tunables, and validates the full cross-anchor range so side enablement
  or unsupported risk behavior cannot be introduced silently.

- KuCoin private futures order websockets now discard the exact cached negotiated URL when the
  exchange expires its token, allowing `watch_orders` to obtain a fresh token instead of reusing
  the rejected URL indefinitely. The expected callback exception is reduced to a throttled warning
  while REST reconciliation continues normally.

- Add `passivbot tool compose-coin-overrides` to validate and combine a directory of single-coin
  configs into a lean unified config with minimal inline per-coin patches. The tool canonicalizes
  parameters belonging only to features disabled in every input, reports account-wide conflicts,
  merges approved coins with fail-closed resolved-market and cross-venue contract identity
  validation, supports selecting the master input, and can optionally retain that input's
  backtest and optimizer sections for fixed-override fine-tuning. Full output with coin overrides
  remains unsupported by the GPU optimizer backend.
- Add an experimental, purely additive Apple Silicon MPS optimizer backend for single-coin,
  EMA-anchor and trailing-martingale searches in long-only, short-only, and long+short modes.
  Single-side long-only or short-only multi-coin EMA-anchor searches are also supported for up to
  64 coins with shared balance, per-coin market and indicator state, dynamic wallet-exposure
  allocation, Forager
  selection, searchable Forager parameters and position count, strict tick-boundary fills, and
  compact unified-memory inputs guarded against excessive MPS allocation.
  Large NSGA-II populations run through strategy-specific Rust-owned Metal screening programs and
  feed feasible, diverse candidates into the unchanged exact Rust backtester; only exact results
  reach the Pareto store, and independent broad-probe rank checks halt on proxy drift. Dual-side
  screening preserves separate indicator/trailing/position state, a shared balance, Rust fill
  ordering, and hedge/one-way initial-side semantics. Multi-coin selection deliberately omits
  Forager score hysteresis and reselects on fills or effective position-count growth; exact Rust
  validation and rolling drift gates remain authoritative for this screening approximation.
  Recursive trailing-martingale entry and close ladders use immutable generation snapshots, merge
  equal-price closes, and fill every strictly crossed rung in Rust's canonical order. Other
  strategies, suites, multi-coin hedged or trailing-martingale runs, HSL, auto-unstuck,
  collateral, minimum-effective-cost filtering, market-order execution, incomplete candle tails,
  and unmodeled risk gates fail closed. Fused delta-form Metal
  EMA updates reduce long-horizon float32 path drift. Optimizer-limit feasibility disagreements
  feed aggregate, proxy-front, and broad-probe rolling constraint-agreement gates, retain exact
  Rust as the only authoritative classification, and persist per-limit proxy/exact diagnostics.
  Strict candle/order
  crossing comparisons are precomputed as
  integer price-tick boundaries, preventing float32 Metal prices from missing fills that exact Rust
  sees just beyond a decimal tick. Candle-derived EMA touches use Rust-compatible directional ticks.
  Trailing-martingale uses float64-derived directional ticks to choose the controlling raw/target
  value before Metal float32 can collapse them, then mirrors Rust's directional entry and nearest-
  tick close finalization. Raw-touch close minimum quantities are sized from the original float64
  price before that finalization, including their ordering relative to an aligned quantity step
  when float32 rounds both values together. Tick-aligned targets remain on their exchange tick,
  and partial final validation batches scale their reserved broad probes proportionally. True proxy-front
  membership is persisted independently from off-front probe eligibility, with the safety window
  and exact budget sized for a one-member proxy front; a generation with no novel front candidate
  fails closed instead of silently consuming that evidence budget. Resume also proves that its
  recovered evidence plus remaining exact budget can still activate the mandatory proxy-front
  gate, while truthful broad-probe scarcity remains admissible across restart and recovered probes
  continue feeding their independent gates; exact worker completions are durably consumed in
  submission order so that proof remains valid when workers finish out of order. Feasibility
  disagreements are evaluated by the independent constraint-agreement gates and excluded from rank
  correlation, preventing the same disagreement from being double-counted as arbitrary ordering of
  otherwise exact near-ties. Window, exact-budget, and fresh/resumed suffix checks reserve enough
  broad-probe capacity to retain eight rank-comparable samples whenever the current generations
  contain that many truthful off-front candidates. When a many-objective generation has fewer
  off-front candidates than requested, all available probes keep their true classification and
  diverse true-front candidates fill the unused exact slots; allocation shortfalls and recovery are
  logged instead of aborting the run. Existing CPU bot, backtest, and optimizer paths do not import
  or require the optional PyTorch dependency.
- Keep side-specific `approved_coins` authoritative in backtests and optimization so a coin
  approved only for long cannot open short entries, and a coin approved only for short cannot
  open long entries. Per-coin zero wallet-exposure overrides now retain the same entry-disable
  behavior after Rust derives runtime exposure budgets.
- Skip forager ranking and its feature requirements when each side's exact remaining candidate
  universe fits its remaining position slots, including when ineligible held positions consume
  slots. Python now scopes missing ranking-only inputs to Rust selection instead of making the
  whole symbol non-tradable. Current remote-enabled forager candidates may also bridge bounded,
  later-bracketed internal candle gaps under `live.max_active_candle_tail_gap_minutes` without
  depending on refresh timing inside one planning pass; compact transition diagnostics identify
  ranking-input continuity use and authoritative recovery, while cache-only candidates remain strict.
- Standardize suite reduction configuration on `reducer` across `backtest`, optimizer scoring,
  limits, CLI parsing, examples, and serialized configs. The former `aggregate`, `stat`, and
  `scenario_stat` spellings remain accepted as input aliases (plus legacy limit `field`),
  same-valued aliases collapse to `reducer`, conflicting aliases fail validation, and existing
  Pareto/suite result artifacts remain readable without rewriting their historical payload keys.
- Hash backtest cache arrays while writing their NPY artifacts, avoiding a second full-array read
  solely to build the cache manifest after multi-gigabyte cache publication.
- Speed up multi-coin backtest HLCV validation with bounded time-major scans, and report frame
  flush and valid-window validation timings separately during data preparation.
- Preserve the full configured exchange pool for combined optimizer and backtest suite datasets
  when every selected coin happens to use the same venue, so single-coin multi-exchange suites no
  longer reject valid unselected candidate venues as unavailable.
- Normalize nested suite scenario override documents to leaf config paths while keeping dynamic
  `coin_overrides` mappings atomic, preventing partial `live` or `bot` overrides from replacing the
  complete section during optimizer evaluation.
- Bound coin-mode HSL `always` restart reconstruction to its canonical fill-proven replay start.
  Older sparse fills still seed exact balance and position state, but discarded closed episodes no
  longer expand candle fetches, minute arrays, panic markers, or replay-event iteration;
  ambiguous evidence and strict restart policies retain the full configured lookback.
- Remove the persisted HSL replay cache and reconstruct every restart from authoritative fill/PnL
  history, candles, exchange state, config, and current time. Compact/sparse replay remains the
  performance path; obsolete replay-cache artifacts are ignored and no longer inspected by the
  cache doctor.

## v8.1.0 - 2026-08-10

- Scope coin-mode HSL fill-history readiness for `restart_after_red_policy=always` to the
  fill-proven held episode plus the flat-scope cooldown horizon. Recent fills for a currently flat
  scope, ambiguous held reconstruction, `threshold`/`never`, and pside/unified modes retain the full
  configured lookback; effective per-coin HSL settings are honored, pending/degraded PnL blockers
  use each pair's own active episode, and the requirement is rechecked after refresh so delayed or
  side-ambiguous fills fail closed. Coin finalization no longer requests unused account-wide PnL.

- Gate.io multi-currency futures balance events now publish bounded settle-currency
  composition diagnostics (wallet amount, available margin, reserved IM/order
  margin, unrealized PnL) and select the quote-matched futures-account row instead
  of blindly using `info[0]`. Trading wallet balance continues to reconstruct from
  available + position IM + order margin − unrealized PnL so resting-order
  reservations do not resize risk inputs.
- Prevent unrelated spot or unloaded-DEX rows in Hyperliquid's public `allMids` payload from
  aborting live ticker snapshots. Unknown exchange-returned identifiers are filtered at the
  connector boundary, while requested-market completeness and malformed known-market prices
  remain fail-closed.

- Allow the complete per-side HSL group in coin overrides when the global
  `live.hsl_signal_mode` is `"coin"`. Resolved per-coin HSL settings now drive both live
  supervision and Rust backtests; inline HSL patches fail in `pside` or `unified` mode, while HSL
  fields from complete override files are warned about and ignored outside coin mode. The signal
  mode remains global and cannot be changed by a coin override.

- Refine the coin-override policy: allow per-coin
  `bot.<side>.risk.entry_cooldown_minutes` and
  `bot.<side>.unstuck.ema_gating_enabled`, retain
  `bot.<side>.unstuck.loss_allowance_pct`, and stop allowing per-coin
  `bot.<side>.risk.we_excess_allowance_mode`. Configure the allowance mode globally.

- Make coin overrides explicit, typed patches instead of hydrated config diffs. File values now
  precede inline values without losing intentional resets to defaults, false, or zero; each resolved
  per-coin config is validated before use; normalized symbol collisions and strategy-kind mismatches
  are rejected; and configured override files fail closed when missing, unreadable, malformed, or
  invalid. Live and backtest consumers now resolve canonical grouped bot fields consistently.

- Refresh hardcoded schema defaults and the mirrored example config from the latest
  long-only trailing-martingale canon candidate: update `bot.long` parameters,
  `optimize.bounds`, and `live.approved_coins` (replace CRO with TON; reorder coin
  list). Backtest suite scenarios and unrelated defaults are unchanged.

- Stop inferring missing fill history from long periods without executions. Fill lookback
  coverage is now proven by successful exchange-endpoint traversal; only actual failed bounded
  fetches create unproven ranges, which remain retryable under the live execution loop's backoff
  until a successful response (including an empty response) clears them.
- TWEL-gated market entries and their final minimum-order guard use executable touch, and next-only
  short entry quantities are re-cropped after directional price quantization.
- Directionally quantize passive WEL auto-reducer prices away from off-tick executable touches so
  limit reducers cannot become crossing orders merely through price-step rounding.
- Resolve exchange market identifiers without lossy pre-normalization: exact CCXT symbols,
  native market IDs, multiplier-prefixed bases, and `exchange::<native-id>` identities now take
  precedence on every exchange. Convenience aliases that match multiple markets fail closed with
  their candidates instead of selecting an order-dependent first match, missing exact identifiers
  and namespaced HIP-3 aliases fail closed instead of normalizing to another market,
  suffix-bearing native IDs remain lossless,
  contradictory qualified source mappings are rejected, unqualified native IDs that identify
  different contracts across configured exchanges require venue scope, and ordinary convenience
  aliases are compared by underlying while scaled or exact inputs retain denomination identity;
  suite scenario identifiers
  are validated and reconciled after union, inception discovery is limited to requested/live
  venues and uses the offline fake timeline locally,
  source and approved-market aliases are coalesced before dataset preparation, conflicting
  duplicate coin and market-settings overrides are rejected consistently, unresolved delisted
  fill IDs preserve their raw historical symbol, resolver changes invalidate old HLCV and
  first-timestamp/inception caches now retain and revalidate resolved-symbol provenance, prepared
  HLCV cache keys track current resolved venue symbols
  after refreshing configured and source-only venue metadata,
  unknown combined-market inception cannot satisfy positive minimum-age rules,
  backtest-only coin sources cannot rewrite live approvals,
  invalid live identifiers disable only their own eligibility while unresolved ignored IDs retain
  only their own prior restrictions, `approved_coins="all"`
  unifies quote variants and venue denomination spellings under one underlying while exact
  scaled identifiers remain lossless, rejects unavailable exact source overrides for active coins,
  emits exchange-scoped identities for remaining collisions, and generated symbol maps plus durable
  per-exchange ambiguity tombstones remain deterministic across cache refresh order. Live coin
  overrides are rebuilt atomically so a failed metadata refresh cannot expose a partial override map.

- Resolve plain underlying names across exchange denomination conventions. Prefix forms such as
  `1000SHIB`, suffix forms such as `SHIB1000`, and Hyperliquid's `kSHIB` notation now share one
  denomination-aware identity when established by that venue's market convention. Numeric ticker
  affixes outside a recognized convention remain part of the asset name. A plain coin selects one
  active venue market deterministically, while exact identifiers continue to request a specific
  contract. Combined backtests keep market settings on the OHLCV denomination when an override
  venue uses a different scale.

- Reconstruct trailing extrema for positions older than an exchange's retained 1m candles with
  the shared 1m, 5m, 15m, then 1h historical-resolution ladder. Coarser candles are limited to
  the old leading prefix, their source counts and exact-1m boundary are logged, and a real recent
  1m suffix remains mandatory; missing or internally sparse recent data still makes only the
  affected trailing input unavailable.
- Move live trailing-input availability into Rust's side-scoped planning contract. Missing
  trailing extrema now suppress only entry or close branches that actually consume them while the
  other branch, other position side, panic, and independent reducers remain available. Remove the
  Python post-reconciliation exception matrix so stale orders absent from current Rust intent are
  cancelled normally.
- Make HSL restart price reconstruction portable across exchanges with limited candle retention.
  Replay now uses the finest available historical resolution in a fixed 1m, 5m, 15m, then 1h
  ladder for the older leading prefix, reports approximate source counts, and never uses coarser
  candles to conceal gaps inside the available 1m era. Fill-based episode boundaries, realized
  PnL, and fees remain exact.
- Route Bitget public OHLCV requests through the complete classic futures history endpoint even
  when the authenticated account uses UTA/Elite v3, preventing older available candles from being
  omitted from live EMA windows while retaining UTA routing for private account and order calls.
- Recognize repeated exclusive switching between complete order cohorts as live
  order-churn evidence. Alternating long/short or order-type intent can now use
  the existing account-wide far-order allowance without merging position-side
  identity; first appearances, isolated switches, uncertain history, and
  recently stable orders remain fail-open. Switching duration stops advancing
  when the current run begins, and a completed stable run prevents an older
  switching episode from being resurrected by one later transition while
  emitting the existing history-reset telemetry. Short switching intervals do
  not mask later sustained price or quantity drift, and ladder cardinality must
  remain consistent for every recurring cohort.
- Short market entries and promoted partial market closes are sized and trimmed from their
  executable touch so
  minimum-notional validation remains consistent across Rust planning, live execution, and
  backtesting, including after TWEL entry gating. Blocked
  loss-gate closes use that same execution price when validating their diagnostic exchange minimum.
  Limit-close minimums use each emitted limit price, drop below-minimum legs from mixed ladders before
  absorbing the remaining position dust, preserve an exact below-step remaining position when that is
  the only executable full close, and clamp short close prices to the minimum positive tick. Live
  execution-policy validation also canonicalizes only representation-noisy, tick-aligned
  submitted books to the same float Rust decodes from JSON.
  Off-tick trailing-strategy entry prices now quantize away from the spread in both next-only and
  expanded-grid paths so a passive bid is never rounded up and a passive ask is never rounded down.
- WEEX now recognizes exact structured error code `-1058` as a temporary per-symbol API-trading
  suspension. The affected symbol enters a configurable RAM-only cooldown (six hours by default):
  flat symbols use graceful stop, held symbols use TP-only while retaining close and panic
  management, and protective closes bypass failed entry-only leverage setup. The initial failure
  remains restart-budget-visible without charging skipped retry-backoff cycles; expiry retries
  automatically, each repeated qualifying response refreshes the deadline, and bot restart retries
  immediately. Cooldown duration validation is bounded so an extreme numeric value cannot overflow
  and mask the original exchange failure.
  The shared policy is available to future connectors only through their own exact exchange-code
  classifiers.
- Prevent symbols retained for existing positions or open-order reconciliation from becoming live
  forager candidates after removal from a side's approved set. Disapproved symbols now remain in
  graceful-stop or manual mode according to `live.auto_gs` while their existing state is managed.
- Reduce peak memory during combined candle preparation by releasing exchange-candidate frames
  after volume normalization and consuming selected frames as dense arrays are materialized.
- Scope live candle/EMA readiness to the Rust actions that consume it instead of deferring the
  entire planner cycle when one active symbol lacks a completed candle. Known missing EMA inputs
  remain explicit and value-free; backtests and unannotated Rust inputs stay strict, stale resting
  strategy orders are cancelled through normal Rust-authoritative reconciliation, entry and close
  strategy branches remain independent when their input needs differ, and independent panic,
  WEL, and TWEL reducers may continue when their own inputs are complete. The live producer-boundary
  validator accepts and strictly validates Rust's corresponding scoped-unavailability warning. A
  `NaN` returned by the completed-candle EMA API now follows that unavailable-input path instead of
  restarting the whole execution cycle; positive or negative infinity remains fatal.
- Harden combined-exchange HLCV preparation across independently downloaded datasets: equivalent
  full-range sources now follow configured exchange priority instead of total volume, robust
  complete-day median-log estimates replace arithmetic volume averaging, and underdetermined
  normalization fails loudly. `backtest.volume_normalization` now controls scaling and cache
  identity, while cache manifests and backtest dataset artifacts retain source-selection and
  normalization provenance.
- Reduce optimizer-suite startup time and peak memory by copying materialized candle datasets
  directly into shared memory instead of creating a redundant full-size intermediate array.
- Speed up combined backtest and optimizer-suite candle materialization by writing bounded
  time-major chunks instead of repeatedly sweeping the full memmap once per coin.
- Prevent unproven fill-history coverage from consuming the generic live restart budget; one reason-aware execution-loop backoff owns fill retries while planning remains fail-closed, and already-latched HSL RED supervision continues during coverage repair.
- Stop refetching every account surface when a known fill only gains authoritative PnL or revised fee evidence, while retaining confirmation for new source identities or structural fill changes; validate realized-PnL history once per Rust planning cycle instead of rescanning it for unstuck eligibility.
- Scope live fill-history readiness to enabled consumers: PnL risk keeps its configured lookback, entry cooldown proves only its structural-fill horizon, and bots without historical consumers use bounded recent ingestion.

- Live fill-history coverage now has one canonical verdict owned by
  `FillEventsManager`. Refresh, staged readiness, HSL replay, and realized-PnL
  consumers no longer duplicate cache/gap interpretation. Metadata that claims
  missing cached rows and malformed known-gap bounds fail closed and trigger
  repair or deferral, while confirmed empty windows remain valid.
- Prepared HLCV caches now key direction-agnostic candle data by the effective
  long/short coin union, allowing long-only, short-only, and both-side runs with
  matching data inputs to reuse one verified cache. Backtest artifacts report
  the current run's side membership separately from cache-build provenance, and
  optimizer data preparation derives reachable sides from optimization bounds
  so fixed-bound starts and Pareto restarts resolve consistently.
- Live Rust orchestrator output now fails fatally before diagnostics or reconciliation when its JSON
  (including duplicate keys, non-standard numeric constants, exponent-overflow floats, or decoder failures), required order
  batch, order fields, aggregate
  close quantity relative to the submitted position using representation-scale tolerance even for
  tiny contracts, conversion identities, complete per-symbol
  mode state, or required consumed diagnostic collections are missing or malformed. Diagnostic
  validation rejects impossible realized-loss blocks. Order-field validation includes overflowing
  numeric values, execution type
  inconsistent with the submitted market policy, order book, and near-touch threshold, and priority
  inconsistent with Rust's order and submitted-mode rule. It also rejects order families forbidden
  by the submitted mode or flat-side eligibility, rejects all orders for globally disabled sides,
  requires each flat-side entry batch to contain the strategy's valid initial family while
  preserving Rust's recursive initial-plus-grid ladders,
  rejects initial-normal entries for held trailing-strategy sides, multiple initial-partial entries,
  and multiple EMA Anchor entries for one symbol-side,
  enforces flat active-set caps, forced-normal active-slot reservations, and one-way initial-side
  exclusion, requires flat entries to agree
  with their active/allow-initial diagnostics, rejects foreign-strategy families and competing
  protective reducers, rejects protective reducers whose direct submitted unstuck, WEL, or TWEL
  enablement is disabled, requires panic reducers to close the full submitted position, rejects
  entry quantities below the submitted effective exchange minimum or
  outside the submitted quantity step, rejects close quantities below their effective minimum or
  outside the submitted quantity step except for an exact remaining position, rejects limit
  prices outside the submitted price step, requires diagnostic effective modes to match Rust's
  submitted mode/position/global-enable rule,
  requires order-type names to round-trip through Rust's
  canonical ID mapping, rejects legacy inflated-entry enum variants which have no Rust producer,
  and validates active-state diagnostics in both directions for ineligible
  sides and eligible managed positions, while preserving Rust's
  held-position DCA and configured HSL panic market closes as explicit Rust
  behavior (the latter is the protective exception to `live.market_orders_allowed`). The bot no longer
  converts a fabricated empty batch or usable subset which could cancel existing orders. Normal
  live calls emit a correlated failed-return event before propagating the error, and HSL RED
  supervisors no longer swallow fatal producer failures. Impossible loss-gate blocks are rejected
  when Rust bypasses that gate for panic reducers, the submitted policy disables the gate, or the
  block reports a different loss percentage than the submitted policy, and when the submitted
  side is flat, manual, panic, globally disabled, or uses a different strategy family. They are
  also rejected for reducer families disabled by their submitted unstuck, WEL, or TWEL gates,
  including the global auto-unstuck gate, or when their quantities violate the submitted close
  step, effective minimum, or position cap. Rust now
  also quantizes effective minimum quantities to the submitted quantity step without overshooting
  already aligned floating-point values, including valid quantities below ten decimal places, or
  collapsing a positive sub-step minimum to zero, including when an aligned step multiplication is
  represented one ULP below the exchange minimum, while quantities genuinely above a step still
  round up, preserves exchange-step precision in all shared Rust rounding paths while retaining
  ten-decimal cleanup for ordinary steps, and recomputes minimum entry quantity after the final
  strategy price is quantized so minimum-cost orders remain valid. Live validation scales cost
  tolerance to floating-point precision instead of admitting orders below tiny exchange minimum
  costs, and quantity-minimum
  tolerance is likewise restricted to floating representation error for both entries and partial
  closes. Market-entry minimum cost is validated against the submitted executable touch rather
  than the producer's reference price. Positive entry cooldowns retain Rust's single-entry staging
  rule after the time window expires, and positive trailing-martingale entry retracement retains
  Rust's single-entry staging rule even when cooldown is zero, canonical martingale entry and close
  families must match their submitted grid-versus-retracement branches, and EMA Anchor emits at
  most one entry and one close per symbol-side. EMA Anchor closes promoted to market execution are
  resized from the executable touch before aggregate-close and realized-loss gating. Canonical
  trailing-martingale retracement emits at
  most one trailing close per symbol-side, and close minimum-cost validation uses the emitted limit
  price or executable market touch. Held positions in enabled panic mode require Rust's
  full-position panic close. Loss-gate diagnostic prices must match the submitted exchange step.
  WEL and auto-unstuck limit prices are
  directionally quantized from off-tick book quotes before minimum sizing and
  loss projection. It quantizes EMA Anchor
  touch prices in the protective direction without moving float-noisy aligned touches, preserves
  positive bid and ask ticks below ten decimal places, keeps the minimum positive tick for short
  closes while suppressing long entries when no positive tick exists at or below the selected bid,
  restricts aligned-tick snapping to floating representation error, and keeps panic limit prices
  valid when the submitted top-of-book quote itself is off tick, including low-priced books at or
  near one price step and valid tiny increments, without skipping a tick because of ordinary
  floating-point noise. Quantity-step and price-step validation admit only floating representation
  error, so genuinely off-step values cannot hide inside a fixed fraction of the increment. Close validation rejects
  positions at or below Rust's final
  close-trimming dust threshold before applying the exact-position exception, and panic-limit
  prices must match Rust's exact one-tick protective formula for the submitted book. Full-position
  close checks tolerate only floating representation noise between Rust's step-rounded quantity
  and the submitted position. Entries are
  rejected when the submitted fill timestamp and cooldown make Rust's deterministic add-order gate
  active. The complete
  serialized diagnostic envelope now requires and
  validates every Rust warning variant. Enum-shaped producer fields fail fatally even when
  malformed as JSON arrays or objects, including loss-gate warning policies, and graceful-stop
  mode uses Rust's exact nonzero-position rule. Unstuck reducers must agree with Rust's submitted
  global auto-unstuck gate. Forager selections and flat active state must agree in both directions:
  selected symbols are flat and active, and every non-forced active flat symbol appears in the
  corresponding selection, while allowing Rust's later one-way tie-break to disable initial entry
  on one selected side.
- Live fill readiness now separates proven structural fill history from realized-PnL
  quality. Pending or synthetic PnL continues to block and repair before enabled HSL,
  auto-unstuck, or realized-loss logic can run, but no longer defers all fill-dependent
  planning when every authoritative-PnL consumer is disabled. Zero-exposure unstuck
  configurations follow Rust's disabled behavior, monitor snapshots leave disabled
  unstuck allowances unavailable instead of reading unsafe PnL, and coverage failures
  retain their coverage-specific diagnostics and retry classification.
- Live health-summary PnL now counts authoritative net realized PnL only.
  Pending and synthetic fill estimates remain visible in fill diagnostics but
  no longer create temporary fill-identity bookkeeping solely to correct the
  uptime metric after enrichment.
- Flat forager candidates can now persist proven-final public 1m WebSocket
  candles through the canonical candle path, reducing routine candle REST
  pressure while retaining REST for startup basis, gaps, reconnect recovery,
  prolonged silence, and periodic integrity audits. CCXT cache provenance and
  successor timestamps prove finality; each watcher session primes its first
  snapshot without persisting replayed rows, integrity audits force a REST
  overlap even when the WebSocket tail is fresh, and WebSocket silence never
  creates a synthetic no-trade candle. Changed values may correct an existing
  canonical timestamp, while extending the tail requires fresh-successor proof;
  shard persistence is read-verified before a WebSocket candle is exposed to
  cache and EMA readers, including on immutable legacy-backed days. Unstable
  streams cool down to REST-only maintenance before retrying automatically, and
  the subscription reconciler remains ready for runtime transitions into
  forager mode.
- Dynamic candle WebSocket removal now lets the owning watcher consume CCXT
  Pro's unsubscribe wake-up before cancellation, avoiding orphaned-future error
  spam, and reconciliation retires a removed symbol batch with one supported
  bulk unsubscribe. Bulk and singleton unsubscribe calls, watcher cancellation,
  and post-cancellation waits are hard-bounded even when connector coroutines
  suppress cancellation. Removed watchers remain owned until retirement returns,
  and abandoned watchers remain marked retiring until they actually stop.
  Internal bot restarts delegate the single maintainer cancellation to outer cleanup,
  then await teardown and close event,
  monitor, and exchange-client resources before constructing the replacement
  bot; cancellation-resistant tasks cannot extend cleanup beyond the bounded
  grace deadline, and an incomplete event-pipeline shutdown no longer permits a
  blocking monitor-publisher close to stall replacement. Correlated private
  incident records retain bounded full frame chains at normal log level while
  excluding exception text, locals, source lines, and credentials; hostile
  traceback accessors degrade only the optional diagnostic projection.
  Background forager refresh
  also distinguishes currently fetchable missing candles from verified
  no-trade continuity and retry-deferred gaps, preventing sparse KuCoin markets
  from repeatedly consuming REST budget while raw coverage remains honestly
  unavailable where proof is incomplete. Gap normalization preserves those
  proof- and retry-epoch-specific ranges so adjacent unverified or newly
  finalized minutes remain refreshable, using log-linear sweep normalization
  for large sparse histories.
- Coin-mode HSL startup now bounds and rebases `always`-policy held-pair
  replay at a fill-proven current episode plus any cooldown-linked predecessor
  episodes,
  so exchanges with limited recent 1m history do not strand a protected open
  position on irrelevant older closed episodes without carrying their realized
  PnL into the current episode. Startup failures also retain a
  correlated private bounded frame chain at normal log level while the console
  remains compact.
- Bitget UTA private order updates now use the native `holdSide` field for
  hedge position attribution. Hyperliquid briefly retries a sparse order-open
  event when a concurrent local create is still awaiting its exchange ID, then
  accepts it only through the existing exact acknowledged-ID contract. These
  fixes avoid unnecessary authoritative REST refreshes without weakening
  foreign or ambiguous order handling.
- Background forager candle refresh now charges its REST budget per attempted
  symbol/timeframe fetch instead of reserving a whole batch before execution.
  Wall-time or lock timeouts briefly defer only the affected surface, preventing
  one slow symbol from consuming the batch budget and starving other candidates.
  The remaining cycle time is also shared across the remaining selected surfaces,
  so sparse-history pagination cannot consume the entire wall-time allowance before
  later candidates receive a refresh attempt.
- Added production Bitunix USDT perpetual-futures support through a native signed REST and
  WebSocket connector, including complete market metadata and top-of-book coverage, live-candle
  pagination, hedge-mode order and position reconciliation, account configuration, realized-PnL
  fill events, long/short reduce-only order lifecycles, honest identifier-only market-order
  acknowledgements, strict complete open-order pagination, and bounded REST ticker snapshots when
  WebSockets are explicitly disabled. The connector rejects invalid order quantities and
  case-insensitive authentication-header collisions, isolates malformed order-detail rows, retains
  synthetic ticker provenance, refreshes public ticker subscriptions as markets change, times
  ticker-cache freshness locally, reconciles malformed private-WebSocket rows, sends the venue's
  JSON keepalive on idle private streams, supplies conservative documented VIP0 futures fees to
  live planning, retries transient native market-discovery failures, and uses native public REST
  market loading during cold-cache CLI startup instead of falling through to CCXT.
- Fills sharing a single millisecond are now ordered by the position chain the exchange reports
  with each fill instead of by arbitrary response order. Hyperliquid executions retain their
  individual `startPosition` boundaries, and older coalesced cache rows are expanded back into those
  components before reconstruction. Within the same timestamp cohort, a recovered close basis is
  propagated through a following add only when the raw close component explicitly reports PnL, so
  both the terminal size and VWAP can confirm the authoritative position without inventing a zero
  PnL or carrying an unproven basis across a history gap. If a position chain is ambiguous, trailing
  anchor selection keeps the existing fill order independently of mutable position state.
  Hyperliquid's recent-fill overlap counts timestamp cohorts and is not clamped forward by the
  time-based refresh checkpoint, preventing a same-millisecond execution burst or older cohort
  from excluding a late-arriving component. Legacy coalesced cache rows are split only when
  composite and canonical source identities, cohort fields, finite position-chain data, weighted
  price, quantity, PnL, and fees reconcile. An unreconciled aggregate triggers a recoverable cache
  quarantine and exchange rebuild before components are accepted.

- WEEX Futures orders now carry Passivbot's registered broker ID in the required
  `newClientOrderId` prefix while preserving Passivbot order-type markers for
  reconciliation and fill diagnostics.
- Retired the diagnostic-only Python planning-availability Cartesian product
  and its routine `planning.symbol_state` event. Snapshot provenance, actual
  planning deferrals, EMA degradation, Rust results, reconciliation, and
  initial-entry outcomes remain available through their canonical structured
  events; live trading and readiness enforcement are unchanged.
- Optimizer suites may now select the scoring basis independently for each objective. A scoring
  entry may set `scenario` to a named suite scenario, set it explicitly to `null` to use suite
  aggregation, and optionally set `aggregate` to `mean`, `min`, `max`, `std`, or `median`.
  Omitting `scenario` inherits `optimize.objective_scenario`; aggregate objectives without an
  explicit reducer inherit the metric-specific or default `backtest.aggregate` rule.
- Optimizer suite limits may now select a named `scenario` independently of scoring. Limits with
  an omitted or null `scenario` keep using suite aggregation and their explicit `stat` or
  `backtest.aggregate` fallback. Named-scenario limits use that scenario's metric value and reject
  an accompanying `stat`. Scenario labels are normalized consistently and retain generated labels
  after filtering. Named-scenario limits are validated against the active labels before data
  preparation, including when suite mode is disabled. The Pareto dashboard also applies forbidden
  `inside_range` bands and optimizer-compatible `auto` directions to scenario-specific limit
  columns, quotes scenario labels containing its boolean separator, honors disabled limits, and
  parses generated not-equal filters. Pareto CLI filtering likewise resolves `auto` limits with
  the optimizer's default metric directions. Iterative backtesting rejects unsupported
  scenario-specific limits before loading markets or preparing datasets.
- Optimizer starting configs may now be pre-filtered from stored Pareto metrics with
  `--filter-starting-configs` and optionally reduced with the same `anchors-farthest` selection as
  `pareto-compress` via `--compress-starting-configs N` (`--starting-configs-max N`). The optimizer
  warns that stored metrics are not verified against the new run and fails loudly when metric
  artifacts are missing, malformed, or all rejected. Active-scenario aggregate preselection matches
  suite runtime behavior by aggregating each metric over the selected scenarios that emitted it.
- Live EMA preparation now batches compatible spans per symbol and metric family, including bounded
  cache-only fallbacks for stale forager candidates, and complete candle windows bypass redundant
  Python gap reconstruction. A failed combined read retries each span through its primary reader
  before using bounded fallback, preserving shorter complete EMA windows. Batched cache-only
  fallbacks likewise validate coverage per span so a missing long-window prefix does not discard a
  complete shorter fallback. Metadata-only candle refreshes no longer invalidate otherwise
  identical open-tail projections, while candle-content and known-gap changes written by another
  bot process invalidate the affected cached projection. This removes repeated candle-window loads
  and projection recomputation. The final scalar EMA recurrence now runs in the Rust extension with
  the same sequential floating-point and non-finite-sample semantics, without changing
  completed-candle freshness, gap handling, or EMA math.
- KuCoin aggregate position-cycle PnL reconciliation is now idempotent across overlapping fill
  refreshes: a pending trade row no longer discards an already reconciled authoritative value, while
  a genuinely revised position-history total still updates the affected lifecycle.
- The in-memory order replacement churn gate now emits a compact account-wide admission-reason
  summary every ten minutes, distinguishing allowed near-market, stable/new, risk-critical, and
  allowance-backed creations from deferred candidates.
- Gate multi-currency futures balance now remains stable while resting orders reserve and release
  margin, preventing balance-driven ideal-order resizing and reconciliation churn. Passivbot
  reconstructs account margin balance from Gate's available, position-margin, and order-margin
  fields instead of treating available margin as equity.
- KuCoin 1m sparse-gap repair now immediately includes the nearest real candle on both sides of an
  unresolved interval, allowing one successful exchange payload to prove and materialize genuine
  no-trade minutes without first exhausting empty-range retries or waiting seven days. Empty,
  one-sided, malformed, or partially recovered responses are never converted into candles. Failed
  contextual verification preserves the unresolved gap and starts an independent bounded proof
  cooldown instead of consuming REST capacity on every candle read. Background forager
  refreshes also back off unchanged empty tails without ERROR spam while unexpected failures
  remain loud.
- Trailing extrema now use the configured bounded active-candle tail projection for a missing open
  1m tail after otherwise dense post-fill coverage. The temporary flat zero-volume rows are not
  persisted, delayed real candles replace them on the next cycle, and leading, internal, or
  over-limit gaps still make trailing state unavailable. Structured diagnostics identify the
  trailing consumer, symbol, position side, projection bounds, and consecutive fallback uses.

- Flat forager-selected symbols with resting entries now degrade to nontradable when required EMA
  inputs are temporarily unavailable, allowing normal reconciliation to cancel the stale entry
  instead of repeatedly crashing and restarting the whole live bot. Held positions and explicitly
  configured normal modes retain their strict required-input behavior. Bounded open-ended 1m gaps
  continue using provisional in-memory EMA projection even when retry metadata records the missing
  tail, so symbols remain tradable through the configured active-tail grace period and recover
  immediately when authoritative candles arrive. The same bounded open-tail policy now applies to
  stock perps instead of granting them an unbounded no-trade-tail exception. EMA reads also
  provisionally bridge unresolved gaps already bounded by later authoritative candles without
  persisting them, recompute when delayed real rows arrive, and refuse gaps wider than the live
  active-tail bound. Cache-only forager ranking carry-forward does not project unresolved internal
  gaps. Mixed fixed/forager sides retain strict readiness for the fixed side, and a dynamically
  managed resting entry is still cancelled if close, strategy, or ranking EMA degradation changes
  its side to the configured manual stop mode. A disabled opposite side no longer misclassifies an
  otherwise dynamic forager symbol as fixed-normal. Ranking degradation retires entries only on
  the affected side, while dynamic-forager eligibility survives the resulting symbol-level manual
  stop so a later identical ranking gap does not become an account-wide error. Failed or ambiguous
  cancellation attempts are retried only for the exact proven
  exchange/client order ID, including when an exchange ID appears after the client ID and after EMA
  recovery, without weakening ownership for orders first observed after the side enters manual or
  ordinary tp-only mode. Cache-only candidate close EMAs and completed-candle forager ranking
  metrics cannot reuse provisional active-strategy cache values; active-symbol quote-volume and
  log-range ranking reads are equally strict and remain separate from provisional strategy
  log-range inputs. Temporary bot-managed entry
  overrides such as HSL graceful-stop retain the same flat-symbol degradation behavior instead of
  promoting missing EMA inputs into an account-wide restart loop. Budget-derived forager ranking
  staleness also retains the active-tail grace period (10 minutes by default), so a large refresh
  budget cannot make flat candidates nontradable after only one or two missing completed candles.

- Supported CCXT private order streams now isolate malformed semantic rows from websocket
  transport health: unnormalizable rows are discarded with a bounded warning and force an
  authoritative account-state refresh, while valid rows in the same message are processed without
  reconnecting. Bitget side-attribution failures now use this path without logging raw payloads.
- Hyperliquid sparse private order updates now recover mandatory one-way position-side and
  close-only semantics by exact exchange order ID from the current authoritative REST open-order
  snapshot. Orders already resting at bot startup therefore avoid repeated semantic-rejection REST
  refreshes. A bounded recent copy covers terminal updates arriving just after reconciliation
  removes the order. Exchange-ID and client-ID aliases must agree, authoritative contradictions
  invalidate older cached semantics and cannot fall back to local acknowledgements, and
  snapshot-recovered rows still force account refresh because the snapshot proves semantics rather
  than local ownership. Missing, ambiguous, stale, and contradictory identities remain fail-closed.
- Binance's explicit `MarginModeAlreadySet` response is now treated as a successful configuration
  no-op at DEBUG instead of an ERROR; unknown margin-mode failures retain their existing loud
  handling.

- KuCoin private order updates now use the connector's actual exchange hedge mode for mandatory
  long/short attribution even when `live.hedge_mode=false` disables simultaneous strategy
  exposure, preventing valid updates without one-way `reduceOnly` metadata from reconnecting the
  watcher. Hedge-mode close-only effect now follows KuCoin's authoritative side plus position-side
  tuple when `reduceOnly` is omitted. KuCoin native higher-timeframe no-tick gaps are materialized
  only when bounded by real candles, absent from one successful raw payload, and no wider than the
  fixed 120-minute live connector policy, restoring required 1h volatility EMA readiness without
  allowing the simulation-only backtest gap setting to alter live behavior. Rejected or
  unidentifiable rows break payload continuity, expansion is bounded to the requested range, and
  later real candles deterministically replace persisted synthetic buckets. A later rejected real
  payload row evicts a cached sparse placeholder at the same timestamp; an unidentifiable rejection
  evicts cached placeholders between accepted page bounds or across the remaining requested range
  when those bounds are unavailable. Eviction also recomputes native-timeframe cache index bounds.

- Hyperliquid recent candle gaps now retry on a time-spaced schedule instead of
  exhausting the persistent-gap budget in consecutive live cycles while the
  venue may still publish an authoritative no-trade row. Accelerated retries are
  limited to bounded tail-sized gaps, and the retry decision now precedes ordinary
  present and tail-completion fetches without suppressing newly finalized candles
  beyond the deferred gap or repair of unrelated internal gaps. Deferred
  unverified rows remain absent from returned candle continuity rather than
  becoming synthetic zero-volume candles even when their retry is due or remote
  fetching is disabled. Targeted retries and day-coalesced historical fetches
  split around deferred ranges. Forced 1m candidate refreshes now detect partial
  pagination followed by an empty terminal page, allowing repeated failures to
  use the bounded in-memory retry delay without misclassifying complete
  overlap-pagination fetches. Persisted 1m rows trim or split stale known-gap
  metadata; unresolved remainders are deferred after partial recovery. Missing
  rows remain unavailable and are never fabricated by this recovery path, and
  large deferred ranges remain interval-based rather than expanding into one
  Python object per minute. Failed recent Hyperliquid persistent-gap retries
  retain the persistent retry cadence, and forced overlap refreshes split around
  deferred internal gaps. Gap retry metadata follows the manager's replay/live
  clock, partial historical pages flush their deferred index before propagating
  failure, and unresolved internal gaps keep dependent EMA windows unavailable.
  Overlap refreshes now stamp any attempted-but-unresolved known-gap remainder
  before later repair stages can retry it in the same request. Newly recorded
  1m gaps invalidate affected EMA/projection caches, while complete authoritative
  rows remain usable if stale gap metadata has not yet been trimmed.

- Binance and KuCoin private order streams now recover sparse Passivbot-owned
  hedge-mode updates only when the encoded client-order position side has an
  exact identity in this process's emitted-order registry, native position-side
  metadata is absent, and all supplied order identities agree with the same
  emitted record. Acknowledged identities remain registered while their orders
  are open, including orders resting longer than the normal foreign-writer
  lookback. Recovered updates force an authoritative account refresh without
  weakening strict REST open-order reconciliation. Genuine transport failures
  retain the existing bounded reconnect backoff.

- Forager monitor health now distinguishes approved candidates that are
  temporarily unrankable because volume/log-range or required candidate EMA
  inputs are unavailable from active-symbol trading degradation. Ranking-feature
  health is populated by the active Rust-orchestrator preparation path, candidate
  labels use the same market-age and effective-minimum-cost eligibility as live
  selection, all EMA health is cleared before a failed replacement bundle can
  leave stale state behind, candidates remain explicitly unrankable until a
  bundle completes, and active-symbol EMA failures retain their active
  degradation reason. Open-tail forager projection computes only close EMAs
  because ranking metrics must come from real candles, and latest-value EMA
  calculations avoid allocating an unused full series.

- Live trailing fill confirmation now accepts reconstructed position-price
  differences of at most one effective executable price tick to accommodate
  exchange rounding or truncation of sub-tick VWAPs, including Hyperliquid's
  significant-digit price ladder and its asymmetric spacing across powers of
  ten. Fill identity and reconstructed position size remain strict, the explicit
  position-opening replay retains its ordinary half-tick requirement, larger
  discrepancies remain fail-closed, and acceptance beyond the ordinary half-tick
  comparison emits one warning plus a structured diagnostic when confirmation
  clears.

- Live order-write failures now include bounded, sanitized exchange status, code,
  label, and reason fields when CCXT exposes a structured rejection payload through
  either an exception or a terminal result mapping, including OKX per-order
  `sCode`/`sMsg` details. Successful outcomes retain ordinary result summaries
  without misclassifying native success codes/messages as errors.
  Sensitive-marked values and long identifier-like tokens remain redacted; the
  existing bot restart error budget is unchanged.

- Gate.io now applies the configured leverage and margin mode before a symbol's
  first order creation. This refreshes Gate's leverage-derived position risk limit
  after contract risk-table changes instead of repeatedly failing valid orders
  with a zero risk limit. An hourly market refresh invalidates the configured
  marker when the effective leverage cap changes so the next entry refreshes
  the exchange configuration.
  A failed refresh blocks entries and advances the existing restart budget,
  while reduce-only closes remain eligible. Churn admission and the later
  configuration write now share one retry-eligibility timestamp so a backoff
  expiring mid-wave cannot introduce an unreserved signed action. Missing or
  invalid leverage-cap metadata is isolated to the affected symbol: entries
  remain pending and restart-visible, but market initialization and protective
  closes continue.

- Binance and Bitget private order updates now use the connector's actual exchange hedge mode for
  mandatory long/short attribution even when `live.hedge_mode=false` disables simultaneous
  strategy exposure. Valid hedge-account updates no longer enter the one-way normalization path
  and reconnect their watchers merely because native `reduceOnly` metadata is absent.

- Live trailing fill confirmation can now recover from a polluted historical
  `psize`/`pprice` residue when the exchange supplies an explicit position
  opening timestamp. Recovery is restricted to a single identified opening fill
  exactly matching both the opening and latest position-update timestamps, still
  requires its zero-state after-state to match the authoritative position, and
  emits an operator-visible fallback diagnostic only when confirmation actually
  clears. Multi-fill cohorts, later updates, proximity-only matches, and generic
  or update-only timestamps remain insufficient.

- KuCoin REST and private WebSocket clients now use IPv4 transport so API keys
  restricted to a host's stable public IPv4 address are not rejected when the
  host also has IPv6 connectivity.

- Simplified the live order-replacement churn gate to a recent Rust-ideal
  behavior filter. It now requires sustained monotonic price or quantity drift,
  measures stability from the current drift run, bounds the universal
  order-match tolerance to 0% through 1% (default 0.02%), rejects malformed Rust
  ideal orders before reconciliation, reuses the existing fresh market
  snapshot, performs one final churn-admission pass after exchange configuration
  and the fresh-market guard, then applies risk-first batch capacity only to
  admitted candidates. Removed
  account/config/list epochs, the wider tracking
  tolerance, flow-cost optimization, Hyperliquid request-budget reservations,
  and signed-action bookkeeping. Churn evidence remains economy-only: unmatched
  actual orders are still cancelled and near-market or risk-critical orders are
  never deferred. Symbols rotated out of the active universe release their RAM
  history even when position normalization retains flat zero-sized placeholders.

- Gate.io live configuration now accepts CCXT's `gate` exchange label as an alias,
  logs its normalization to Passivbot's canonical `gateio` identity, and consistently
  selects the dedicated Gate.io connector. Standalone market loading now also
  translates `gateio` to CCXT's renamed `gate` client without changing canonical
  cache, broker, event, or persisted-state paths. Gate's numeric REST account UID
  is normalized to the string required by CCXT Pro, while missing or invalid UIDs
  remain unavailable instead of becoming a cached `"None"` placeholder, preventing
  private order WebSocket reconnect loops after otherwise successful startup.

- Scope cancel-first create deferral to the symbol and position side of stale-order cancellations
  in hedge mode, or the whole symbol in one-way mode, while retaining conservative account-wide
  deferral for malformed unscoped cancellations.

- Monitor `state.latest.json` snapshots now refresh from a serialized background
  maintainer at least every five seconds, independently of successful planning
  cycles. Prolonged authoritative-data or planning degradation therefore remains
  visible without concurrent snapshot builds or any change to trading behavior.

- Live fill recovery now performs bounded historical refetches around degraded
  synthetic realized-PnL rows even when cache metadata already proves the configured
  lookback. Repair-only fetches preserve the incremental checkpoint and rotate through
  at most four independently bounded execution ranges per authoritative cycle. Bybit
  repair also advances a separately bounded closed-PnL update-time window so delayed
  records remain discoverable. Authoritative replacements are reported before later
  fallible refresh work; uptime health adds the full authoritative PnL for cached rows
  not previously counted by this process and applies a delta against the exact synthetic
  amount counted at runtime. Outstanding synthetic accounting is discarded after
  enrichment. Unresolved degraded rows defer risk planning with exponential backoff,
  retain pending/degraded PnL counts in structured cycle diagnostics, and do not consume
  the generic bot restart budget.

- WEEX REST and private WebSocket clients now use IPv4 transport so API keys
  bound to the host's public IPv4 address do not fail with `-1056 ILLEGAL_IP`
  when a dual-stack host would otherwise select IPv6.

- Documented Python 3.12 and 3.14 support, explicitly excluding Python 3.13 until its dependency
  set is supported and bounding package metadata to those validated minor versions, and fixed
  reentrant candle fetch-lock
  cleanup so a missing bookkeeping record no longer suppresses an exception raised by the
  protected operation. CI now builds the Rust extension and runs the Python suite on both Python
  3.12 and 3.14. Reentrancy is now restricted to the owning asyncio task, so parallel requests
  sharing one candle manager serialize same-symbol/timeframe fetches correctly. Installation
  examples now create the venv with the explicitly selected supported interpreter instead of
  assuming the system `python3` points to it.

- Reduced Hyperliquid account-state refresh churn by recovering websocket order-update semantics
  from Passivbot's own acknowledged order record when the exchange order id matches exactly.
  Every supplied native, unified, and client identity must agree, along with existing websocket
  side, raw side, raw status, position-side, and reduce-only metadata. This includes
  Hyperliquid-native `oid` and `cloid` identities, with `cloid` retained in acknowledged-order
  records. Recovered partial-fill updates force an authoritative refresh; ambiguous,
  contradictory, and foreign updates still fail closed.

- Backtests now treat a finite balance depleted by fills as liquidation before recomputing orders,
  so extreme optimizer candidates terminate normally instead of panicking an optimizer worker.
  Other invalid orchestrator inputs now propagate as backtest errors without unwinding Rust.

- Live fill confirmation now preserves the last successful exchange-refresh timestamp while
  loading or repairing local fill caches, widens fill-history fetches with bounded backoff when a
  position remains tied to a stale or mismatched fill, starting before the earliest available
  position/open timestamp even when that predates the configured PnL window. Affected trailing
  positions remain fail-closed until refreshed history reconstructs a post-fill state matching
  exchange state; price and quantity alone cannot prove a flat-to-position transition. Id-less
  fills use stable content-based identities rather than history-list indices. Position snapshots
  preserve distinct exchange opening times, while timestamp-free positions retry with
  progressively wider history windows outside the account-wide execution barrier, so only the
  affected trailing coin and position side remain nontradable between attempts.
  Widening starts only in background recovery after the required recent post-snapshot confirmation,
  tracks progress per coin and position side, and is capped by connector pagination capacity
  (two years on Bybit, otherwise one year) to avoid unbounded exchange pagination. Sparse Bybit
  trade history now traverses empty recent windows instead of stopping before older fills, while
  stale fill state remains part of the blocking authoritative refresh rather than being displaced
  by a background recovery.

- Bybit closed-PnL refreshes now cover requested history with explicit,
  contiguous sub-seven-day windows and cursor pagination inside each window.
  Sparse pages no longer create gaps in older realized-PnL history, and endpoint
  or pagination failures propagate instead of returning a partial result as a
  successful refresh.

- Unexpected PyMoo worker failures and process exits now abort optimization visibly instead of
  leaving the optimizer polling forever for a lost result.

- Live candle orchestration now reads bounded cache-only native 1h EMA carry-forward from the 1h
  index, requires a complete native window, isolates carried values from the active EMA cache, and
  applies the background refresher's surface-count staleness limit using the active live/replay
  clock. A minute-boundary open-tail projection is also retried before reusing a previous close
  EMA, avoiding transient candidate drops and unnecessary close-EMA fallback on WEEX and other
  exchanges without weakening active-symbol fail-closed behavior.

- Optimizer suites may set `optimize.objective_scenario` (or
  `--objective-scenario LABEL`) to score performance objectives on one named scenario while
  continuing to enforce limits against configured suite aggregates. Suite scenario labels must
  now be unique. Dataset preparation restricts exchange-specific preloads to the union of
  explicitly requested scenario coins when every assigned scenario names its coins. Resume
  validation rejects objective-scenario changes, including old results that predate the setting.

- `passivbot tool crash-finder` can discover ordered low-to-later-high pumps as well as
  high-to-later-low crashes via `--direction up|both`. Generated idiosyncratic stress scenarios
  may use `--scenario-force-normal adverse` to isolate long exposure during crashes and short
  exposure during pumps. CSV regeneration preserves all stored directions unless explicitly
  filtered with `--direction`.

- Canonical live-event payloads now make a bounded JSON-compatible copy at construction time,
  revalidate that copy at persistence boundaries, redact sensitive keys before retention, and
  record aggregate truncation metadata only when a limit applies. Event identity, routing, monitor
  envelope structure, and trading behavior are unchanged; legacy direct monitor history, order,
  and raw-fill storage remain outside this payload boundary.

- Event-bus degradation diagnostics now retain bounded exception types and canonical dropped-event
  labels only. Sink isolation, queue behavior, counters for registered events, timing, return
  values, event routing, and trading behavior are unchanged.

- Failed Rust-orchestrator return events now retain bounded exception types instead of persisted
  exception text. Existing timing and correlation metadata, event delivery, orchestration, and
  trading behavior are unchanged.

- Executor churn-admission and fresh-entry eligibility diagnostics now retain bounded exception
  types only. Existing admission/defer reasons, batch limits, fail-closed behavior, trace handling,
  event delivery isolation, scheduling, and trading behavior are unchanged.

- Ambiguous create diagnostics now retain bounded exception types only. Existing ambiguous
  bookkeeping, restart behavior, acknowledgement handling, event emission, and trading behavior
  are unchanged.

- Executor cancellation-priority/capacity and create/cancel acknowledgement diagnostics now
  retain bounded exception types only. Existing priority fallback, emitter isolation,
  acknowledgement/confirmation, and trading behavior are unchanged.

- Terminal and ambiguous execution-order events now retain bounded exception types instead of
  persisted exception text. Existing envelope correlation, result summaries, debug profiles,
  routing, executor behavior, exchange calls, and trading behavior are unchanged.

- Exchange-configuration refresh failure events now retain bounded exception types instead of
  exception text. Periodic and connector-local payloads continue to preserve their existing
  outcome and response metadata, best-effort delivery, and configuration control flow.

- Candle disk-load and persistence observer diagnostics now retain only code-owned stages, bounded
  exception types, and stable control-flow actions. Monitor persistence and cache-flush reporting
  remain independently best-effort; callback ordering, throttling, completed disk persistence/load
  behavior, event schemas, and trading behavior are unchanged.

- Startup monitor-publisher and live-event-pipeline installation diagnostics now retain bounded
  exception types and stable continuation actions without exception messages, unsafe class names,
  URLs, credentials, tokens, or tracebacks. Monitor enablement, pipeline installation, startup
  continuation, and trading behavior are unchanged.

- Order-sort market-price fetch failure diagnostics now retain bounded symbol context, a bounded
  exception type, and the stable `preserve_original_order` action. The existing empty-price
  fallback, original-order preservation, reconciliation, planning, and trading behavior are
  unchanged.

- Approved/ignored coin-list refresh failure diagnostics now retain only a bounded exception type
  and the stable `return_from_coin_list_refresh` action. Existing refresh return, list/universe and
  eligibility update semantics, scheduling, risk, HSL, and trading behavior are unchanged.

- Unrealized-PnL aggregation failure diagnostics now retain only a bounded exception type and the
  stable `return_zero` action. The existing immediate `0.0` fallback, price fetching, balance and
  equity handling, scheduling, risk, HSL, and trading behavior are unchanged.

- Shared live diagnostic-event fallback and diagnostic-step failures now log only their stable
  context and a bounded exception type. Their existing return values, isolation, routing, retry,
  scheduling, and trading behavior are unchanged.

- HSL history-replay candle-fetch and passive cache-observer diagnostics now retain bounded
  exception types only. Candle fallback and degraded replay outputs, retries, cache behavior, and
  trading state are unchanged.

- Reconciliation diagnostics for trace recording, order-churn evidence, and malformed open-order
  snapshots now retain only bounded exception types. This diagnostic-only redaction does not change
  order planning, churn availability, malformed-order guardrails, or exchange-action gating.

- Foreign-writer safety shutdown diagnostics now retain only bounded maintainer-cleanup failure
  types and a stable continuation action. Stop flags, cleanup invocation, and terminal
  foreign-writer stop behavior are unchanged.

- Redact observer-only balance and position refresh diagnostic fallbacks, including their shared
  market-snapshot structured and legacy projections, while preserving existing refresh
  continuation behavior.

- State-refresh position-change and progress-observer fallback diagnostics now retain only bounded
  exception types and stable actions. Balance handling, cancellation propagation, staged refresh
  cleanup, timing, fallback values, and event schemas are unchanged.

- Market-snapshot provider primary-fetch, missing-symbol retry, cache-sink, fallback, pre-create
  gate, event-emitter, and eligibility-trace diagnostics now retain only bounded exception types and
  stable actions. Snapshot fallback order, fail-closed order filtering, block attribution, cache-sink
  suppression, trace isolation, and exception cause chaining are unchanged.

- Hourly candle disk-audit, maintenance-cycle, and exchange-config event-emitter fallback
  diagnostics now retain only bounded exception types. Audit continuation, maintenance retry,
  market refresh, and exchange-config result behavior are unchanged.

- HSL replay cache and replay-lifecycle diagnostics now retain bounded exception types without
  exception messages, tracebacks, or unsafe exception class names. Cache write failures remain
  nonfatal, cache reuse still falls back to authoritative replay, and HSL readiness and protection
  behavior are unchanged.

- Fixed Bitget UTA open-order normalization recursively deriving position side from close-only
  effect and close-only effect from position side in effective one-way mode. UTA orders with an
  explicit `posSide` now derive close-only effect directly from the authoritative `side` plus
  `posSide` tuple.
- Fixed live order-churn evidence treating the execution loop's normal 30-second scheduled wait as
  a provenance gap, which could prevent the account-wide churn gate from activating for slowly
  moving EMA-based orders.
- Fixed WEEX V3 live reconciliation rejecting valid `COMBINED`-mode close orders when the response
  reported `reduceOnly=false`; WEEX close-only effect now follows its authoritative `side` plus
  `positionSide` action tuple. WEEX account equity is also normalized to realized wallet balance by
  excluding unrealized PnL, restoring live/backtest risk-input parity. Repeated unchanged churn
  deferrals and history-reset diagnostics remain durable in structured events while INFO output is
  summarized at most every five minutes.
- Configs using the retired `live.initial_entry_exec_max_market_dist_pct` now migrate automatically
  to the account-wide replacement-churn gate. Positive values preserve the market-distance
  threshold and hydrate the other new settings from canonical defaults; null and non-positive values
  preserve the old disabled intent through `order_replacement_churn_gate_activation_count = 0`.
  Explicit conflicting old and new settings still fail for manual resolution. This compatibility
  applies to V8 config shapes; legacy `pb_multi` input remains unsupported.
- Replaced the initial-entry-only market-distance posting gate with a strategy-agnostic,
  account-wide Rust-ideal churn-evidence gate. Moving distant entries and closes may be deferred
  after sustained create traffic, while market, risk-critical, and near-market orders remain
  allowance-exempt. On audited supported connectors, stale actual orders are removed in managed
  modes, malformed account-critical open-order snapshots block exchange writes, and any
  cancellation requests full authoritative confirmation while deferring ordinary creates in its
  affected symbol/position scope until Rust replans.
  One-way position-side and native close-only normalization is now deterministic across the
  supported connectors, including OKX long/short mode, KuCoin open orders, and Gate.io's native
  `is_reduce_only` field. Supported hedge-mode adapters no longer substitute client-order metadata
  for a missing exchange-native position side, and untrusted Hyperliquid WebSocket order rows
  trigger authoritative account-state refresh instead of reconnect churn. Churn distance is
  rechecked from a still-valid cached market snapshot after configuration before any create call. Malformed
  Rust ideal orders fail fatally before reconciliation or any exchange action; malformed
  actual-order identity and missing or malformed position sides retain their account-critical
  exchange-write barriers. Failed required margin-mode writes leave dependent creates pending
  instead of marking the symbol configured.
- Protective reducer arbitration now keeps the largest loss-admissible final absolute
  reduction among active panic, TWEL/WEL auto-reduce, and auto-unstuck intents for each position
  instead of using a fixed type priority or summing quantities. If the realized-loss gate blocks
  the largest non-panic intent, Passivbot tries the next-largest intent before giving up protective
  reduction. Equal-size ties keep panic first and otherwise prefer the closest-to-fill candidate,
  so a tiny auto-reduce can no longer suppress a materially larger unstuck close. Reducer sizing is
  finalized before its loss is checked, and a shared batch loss allowance is spent on finalized
  reducers largest-first rather than in symbol iteration order.
- Non-panic protective reducers may now coexist with compatible ordinary grid, trailing, or
  EMA-anchor closes for the same position. Passivbot still selects only one protective reducer,
  keeps panic close exclusive, reserves reducer quantity before trimming ordinary closes, and caps
  aggregate reduce-only quantity to the position in Rust planning and live reconciliation. The
  realized-loss gate evaluates the selected reducer first. This restores simultaneous unstuck plus
  trailing close behavior for `trailing_grid_v7` without a strategy-specific exemption.

- `trailing_grid_v7` with zero entry cooldown now preserves v7's simultaneous grid-entry ladder
  when a later trailing leg uses retracement. Positive entry cooldowns and canonical
  `trailing_martingale` retracement staging remain unchanged.

- Shutdown-stage failure diagnostics now retain bounded exception types alongside existing stage,
  task-count, timeout, and elapsed-time context without arbitrary exception messages, request URLs,
  response text, or credentials. Event-delivery and event-pipeline-close fallback logs use the
  same classification, as do per-maintainer cancellation and legacy cleanup failures.
  Maintainer cancellation, execution-loop waits, session closing, shutdown timing, and process
  control are unchanged.

- Candle remote-fetch callbacks, HLCV progress logs, archive fetch/day diagnostics, structured
  remote-call events, and fake-live candle traces now retain bounded exception types, URL hashes,
  parameter keys, stage, attempt, timing, and correlation without arbitrary exception messages,
  request URLs, or request-parameter values. Fetches, retry/backoff and rate-limit classification,
  archive availability, cache behavior, and trading behavior are unchanged.

- Local candle migration, cleanup, lock, index, disk/cache, health, inception-metadata, and
  deferred-index diagnostics now retain bounded exception types without arbitrary exception
  messages, repr values, or exception-value tracebacks. Cache contents, migration and cleanup
  behavior, lock handling, retries, fallbacks, and trading behavior are unchanged.

- Live candle completed-close fallback, startup/index/background warmup, forager refresh, active
  refresh, and refresh-cap diagnostics now retain bounded exception types without arbitrary
  exception messages or exception-value tracebacks. Warmup, cancellation, lock retry, refresh
  scheduling, fallback values, readiness, and trading behavior are unchanged.

- Live candle health-window, health-summary, trailing-fetch, freshness-readiness, and tail-gap
  diagnostics now retain bounded exception types without arbitrary exception messages. Health
  fallbacks, symbol/position-side availability, trailing deferral, readiness results, and trading
  behavior are unchanged.

- Fill-history refresh failure diagnostics now retain bounded exception types alongside existing
  source, coverage, retry, timing, count, and endpoint context without arbitrary exception
  messages or exception-value tracebacks. This includes exchange-specific fill fetchers, cache
  reads and doctor repair, staged remote-call events, startup/process handling, HSL flatten
  confirmation, and direct refresh callers. Validated numeric status/code and code-owned endpoint
  labels remain available. Publication and time-sync classification cannot replace the original
  refresh failure through hostile exception metadata, and wrapped timestamp failures remain
  recoverable through a bounded cause/context graph check. Code-owned recovery markers are scanned
  across complete exception text using bounded temporary chunks, preserving existing retry
  classification and caller-specific marker case rules without retaining that text. Legacy
  class-name recovery markers are inspected without projecting untrusted names. Exception
  propagation, fill accounting, refresh cadence, planning, risk, and trading behavior are unchanged.

- Best-effort live event emitter failure diagnostics, including HSL event emitters and the
  event-adjacent HSL coin-status human-log fallback, now retain bounded exception types instead of
  arbitrary exception messages that may contain request URLs, response text, or credentials.
  Event payloads, routing, sink isolation, retries, HSL behavior, and trading behavior are
  unchanged.

- Legacy monitor error events and WebSocket reconnect diagnostics now retain bounded exception
  classifications without arbitrary exception messages, request URLs, response text, or formatted
  exception-value tracebacks. Monitor error context is restricted to known code-owned
  classifications, and reconnect DEBUG output retains only bounded stack depth rather than frame
  labels or line values. Reconnect cadence, retry behavior, monitor persistence, and trading
  behavior are unchanged.

- The `ema.unavailable` and `ema.fallback_used` events and their dedicated legacy summaries now
  retain only code-owned reason classifications, bounded EMA/error types, symbols, spans, ages,
  and counts. Malformed typed values are normalized or omitted, adjacent EMA failure logs retain
  only exception type, and a bounded legacy warning remains when the structured event was not
  emitted. These paths no longer retain arbitrary exception or fallback-reason text; EMA
  calculation, fallback selection, candidate availability, and trading behavior are unchanged.

- Noncritical market-snapshot diagnostic skips now use the existing
  `market.snapshot_diagnostic_skipped` event as the sole normal warning when the structured console
  is available. The bounded event and legacy fallback retain only stable context and exception type;
  position/balance refresh behavior is unchanged, and arbitrary exception text is no longer stored
  or projected.

- Pre-create planning and market-snapshot skips now use the existing
  `execution.create_skipped` event as the sole normal warning when the structured console is
  available. The bounded event retains the stable reason, stage, count, symbols, and exception
  type without raw exception text; the legacy warning remains a fallback, and create filtering is
  unchanged.

- Large open-order snapshot deltas now emit one bounded `open_orders.snapshot_delta` INFO event per
  added or removed direction, with only the direction and order count. The event reaches structured,
  monitor, console, and text sinks with the `[order]` tag; reconciliation and order behavior are
  unchanged, and a bounded legacy INFO fallback remains when the structured console is unavailable.

- Incident bundle manifest and command-result summaries now retain the
  embedded live smoke report's text-log `scan_cost` metadata. The archive's
  full smoke report, log reads, matching, redaction, filtering, verdicts,
  exchange access, and live behavior are unchanged.

- `live-smoke-report` log scans now include bounded `scan_cost` metadata in
  full, summary, and brief output. The metadata reports elapsed time,
  successfully read files and selected tail lines, read methods, and known
  physical-versus-decoded bytes without changing log discovery, sequential
  reads, matching, redaction, window filtering, smoke verdicts, exchange
  access, or live behavior.

- Incident bundles now include bounded `scan_cost` metadata for their
  independent time-window event scan in the full report, manifest, and command
  result. The metadata uses the same elapsed-time, successful file/record,
  read-method, and physical-versus-decoded byte contract as the other live
  artifact reports; event matching, truncation, archive contents, verdicts,
  exchange access, and live behavior are unchanged.

- `live-smoke-report` now includes bounded monitor `scan_cost` metadata in
  full, summary, and brief output, matching the elapsed time, successfully
  read files and records, read methods, and physical-versus-decoded byte
  semantics exposed by the event query and performance report. This is
  diagnostic-only and does not change smoke verdicts, event selection,
  parsing, logs, process inspection, exchange access, or live behavior.

- `live-event-query` and `live-performance-report` now include bounded
  `scan_cost` metadata for elapsed artifact-scan time, successfully read files
  and records, read methods, and physical versus decoded byte totals. Explicit
  known flags prevent failed or unmeasurable reads from appearing complete;
  query selection, results, monitor data, exchange access, and live behavior
  are unchanged.

- EMA Anchor live monitor snapshots now mark trailing diagnostics as not
  applicable instead of requesting unrelated trailing-martingale parameters
  and repeatedly failing snapshot publication. The trailing diagnostics tool
  rejects snapshots that explicitly mark those diagnostics unsupported unless
  the operator selects explicit wizard mode.

- Gate.io live startup now selects CCXT 4.5.66's `gate` REST and WebSocket
  clients at session construction while retaining canonical `gateio` identity
  for user-facing configuration, market settings, caches, and logs.

- Window-bounded live smoke collection now accepts a monitor with only
  `current.ndjson` when the bounded monitor manifest proves that segment covers
  the requested window start; missing or invalid coverage evidence still fails
  closed.

- Generic staged balance refreshes now retain the actual exchange response for
  account data-packet provenance while carrying bounded balance composition as
  a separate diagnostic, including Binance's sequenced account cohort and
  legacy raw/normalized balance pairs. Packet hashes therefore reflect raw
  response changes without exposing raw payloads.

- HSL cooldown re-panic confirmation now remains protective after the original
  cooldown deadline until the exact scope-flattening fill is available. The
  confirmation path refreshes the fill cache and reconstructs durable fills
  from unambiguous order side and position side when no transient action field
  is present, including flattening fills stamped in the same exchange
  millisecond as the non-flat intervention snapshot.

- Hardened WEEX live reconciliation by requiring explicit combined/separated
  position mode and long/short open-order sides, and by adaptively splitting
  full fill-history windows so endpoint ordering cannot silently omit fills.
  Empty private order-channel heartbeat messages no longer trigger CCXT Pro's
  symbol resolver with an empty symbol set.

- Added live WEEX USDT perpetual-futures support through CCXT, including
  authenticated account state, simultaneous long/short order placement and
  cancellation, per-symbol combined-position/margin/leverage setup, live
  bid/ask pricing, positions, open orders, and fill/PnL ingestion. Because the
  V3 book ticker has no last-trade field, live last-price consumers use its
  top-of-book midpoint with the explicit `weex_book_ticker_mid` source label.
  WEEX historical 1m backtest-data downloading is not included.

- WEEX live 1m and 1h candle warmups now page through bounded historical
  windows before using the recent tail, so long EMA windows, trailing extrema,
  and HSL restart reconstruction are not silently truncated at 999 finalized
  candles. Indicator/trailing/HSL consumers fail closed on incomplete windows;
  WEEX bulk historical backtest downloading remains intentionally unsupported.

- Updated the live CCXT dependency from 4.5.48 to 4.5.66. WEEX uses a narrowly
  scoped compatibility handler for its documented successful configuration
  response, which upstream CCXT 4.5.66 otherwise raises as an exchange error.
  The Requests and aiohttp pins are aligned with CCXT 4.5.66's declared
  dependencies.

- Planning snapshot diagnostics now include a bounded completed-1m-candle
  freshness summary when that surface is required. The summary is derived only
  from the frozen planning signature and reports expected/real close ages plus
  bounded tail-gap fallback counts; `live-performance-report` validates and
  aggregates the same proof. Public row-count fields avoid secret-key redaction
  so the persisted evidence remains machine-readable. It does not reread
  candles or change planning, exchange access, orders, strategy, or risk
  behavior.

- `passivbot tool hsl-replay-benchmark` now reports exclusive timing profiles
  for fixture construction, replay internals, final-state projection, candidate
  and dense-reference runs, equivalence comparison, and residual orchestration.
  Compact-history benchmarks retain the dense-reference timing evidence they
  use for equivalence, while timeline-reference runs avoid double-counting the
  same execution. This is offline diagnostic output only; HSL state, replay
  behavior, exchange access, orders, and risk are unchanged.

- Trailing fill-confirmation watermarks now advance only when a fill fetch
  actually completes. Exhausted historical fill-gap retries still perform a
  recent-fill refresh, keeping trailing confirmation live while historical PnL
  coverage remains independently fail-closed.

- Trailing-position fill confirmation now treats exchange position-update
  timestamps as advisory and proves readiness with a successful post-position
  fill refresh, a new fill identity for runtime changes, and matching fill
  after-state. This prevents Bybit's later `updatedTime` from leaving valid
  trailing positions permanently nontradable. Repeated unavailable warnings are
  bounded, and monitor market state now exposes the actual planner tradability
  plus fill-confirmation predicates and watermarks.

- `passivbot tool runtime-attribution` now reconciles the producer's exact
  12-character lowercase-hex startup-log run-id prefix with one complete
  manifest or startup-event identity when exchange, user, prefix, and start time
  agree within two seconds. Ambiguous, incomplete, malformed, or out-of-bound
  observations remain separate. Monitor ingestion reads the canonical
  `_live_event` envelope while retaining legacy input compatibility; the
  read-only tool still does not contact exchanges or control bots.

- Shutdown evidence in live smoke reports now distinguishes complete and
  incomplete latest shutdown lifecycles per bot. Restart smoke validation uses
  distinct complete bots instead of aggregate event counts, so duplicate
  stopping or stopped events from one bot cannot satisfy a multi-bot restart
  gate; general smoke verdicts and live runtime behavior remain unchanged.

- Live smoke reports now include a bounded diagnostics-only event-pipeline
  integrity verdict from each bot's latest health snapshot. Cumulative drops,
  sink errors, and workers unexpectedly absent outside orderly pipeline shutdown
  are explicit attention evidence; existing smoke, process, trading, and
  top-level attention verdicts remain unchanged.

- Fake-live scenario-time callbacks now retain their original offline fake
  client through graceful shutdown. The final monitor snapshot can therefore
  complete after the bot releases its public-session reference, without a
  spurious `NoneType.now_ms` error or any change to live timekeeping, exchange
  sessions, trading behavior, or event payloads.

- Offline fake-live runs now retain the already-emitted redacted structured
  event envelopes as a run artifact. The coin-mode HSL RED regression uses that
  evidence to prove a panic fill is followed by an available planning snapshot
  without a post-panic `planning.unavailable` handoff, while leaving live event
  production, HSL behavior, exchange calls, orders, and risk unchanged.

- Hyperliquid `balance.changed` composition diagnostics now parse only proven
  unified-account `info.balances` coin and signed total fields from the
  already-fetched balance response. Non-unified payloads remain explicitly
  unavailable; malformed unified shapes remain diagnostic failures. The bounded
  rows add no exchange calls, valuation inference, scalar-balance changes,
  planning, order, or risk behavior, and raw connector payloads remain excluded.

- Binance `balance.changed` composition diagnostics now normalize only CCXT's
  documented unified `total`, `free`, `used`, and explicit `debt` maps from the
  already-fetched balance response. The bounded rows add no exchange calls,
  valuation inference, scalar-balance changes, planning, order, or risk
  behavior; raw connector payloads remain excluded.

- `balance.changed` now carries a bounded optional asset-composition diagnostic
  from the same authoritative balance response. The first connector parser is
  OKX: it reports only documented account-detail amount, USD value, unrealized
  PnL, explicit liability, collateral state, and field provenance. Equal-total
  collateral substitutions are durable through a separate composition
  signature, while console admission remains snapped-balance based and shows at
  most two sanitized assets. Generic paths report a stable unavailable
  diagnostic until their own parsers are added; balance calculation, API calls,
  refresh cadence, planning, orders, and risk are unchanged.

- Live forager background refresh now schedules the native 1h candle windows required by inactive
  candidates in addition to their 1m inputs. Candidate refresh budgeting operates per
  symbol/timeframe surface, bounds and rotates cache-health scans, honors configured warmup and
  stale-tail limits, interleaves 1m and 1h health checks while prioritizing cold 1m fetches, backs
  off unavailable native-1h leading-prefix retries only after successful fetches, keeps known-stale
  unfetched surfaces pending, and derives staleness from the actually refreshable universe. Forced
  native higher-timeframe reads bypass partial range-cache hits, preserve complete disk coverage
  when a retry returns only a partial range, keep incomplete ranges out of reusable EMA state, and
  invalidate the refreshed timeframe's EMA cache and overlapping range-cache entries. Backoff is
  scoped to the requested window and
  begins only after a nonempty fetch still proves its leading gap. Health-only scans do not consume
  fetch tokens, and native 1h work is scheduled only for nonzero strategy weights. Zero-budget
  cycles perform no fetches or per-symbol warmup computation even with open slots. This prevents
  cache-only planning from freezing the selection universe around incumbents whose 1h log-range
  cache alone remained fresh.

- Live startups now persist an immutable, non-secret runtime manifest and expose
  the same Python commit, config hash, embedded Rust source fingerprint, loaded
  Rust artifact hash, version, and run id through bounded startup events and
  monitor state. Newly discovered fill events retain which runtime first
  ingested them without falsely claiming that runtime created the order;
  refreshes preserve existing attribution and leave legacy fills unattributed.

- Added `passivbot tool runtime-attribution`, a bounded, read-only local report
  that correlates fill caches and monitor fill history with immutable runtime
  manifests, structured startup events, and legacy startup logs. It keeps
  recorded first-ingestion identity separate from non-proving producer-window
  candidates, leaves legacy fills unattributed, supports trailing-only and
  account/symbol/time filters, and can fail when selected fills lack recorded
  provenance without contacting exchanges or controlling bots.

- Full live smoke reports now retain a separately bounded sample of hard
  structured problem events, with authoritative total, retained, and truncated
  counts. Later warning-level attention can no longer hide every classification
  behind a nonzero hard-problem count. Concise summary, brief, and incident
  bundle smoke metadata now retain the same bounded hard-only evidence; the
  existing mixed latest-event sample and all runtime behavior remain unchanged.

- HSL replay now recognizes every ordered, fill-derived scope flatten as an
  episode boundary, including multiple close-and-reentry transitions within
  one replay minute and compact coin replay without historical unrealized PnL.
  RED cooldowns anchor to an episode-bounded flattening fill regardless of
  close order type; when an initial or cooldown-repanic flatten fill is
  temporarily unavailable, live supervision performs a rate-limited,
  episode-bounded fill refresh and otherwise defers instead of using stale
  pre-episode evidence, intervention entries, partial closes, or an invented
  current-time timestamp. Intra-minute replay also reconstructs the account
  balance at each ordered episode boundary so later fills cannot change an
  earlier episode's drawdown or no-restart outcome.

- `sink.degraded` events no longer retain raw sink exception text. They preserve
  the stable sink-failure reason, sink name, exception type, health counters,
  and pipeline timings while keeping request URLs, credentials, response data,
  and other exception-message content out of degraded-event and monitor sinks.

- `cycle.degraded` structured events no longer retain raw exception text or
  request URLs from generic execution-loop failures or fill-history coverage
  deferrals. A strict payload allow-list also drops nested spelling variants and
  unknown caller fields. Stable reason codes, bounded exception types, cycle
  correlation, safe operational details, and phase timings remain available;
  retry, recovery, restart, and trading behavior are unchanged.

- Added `passivbot tool live-restart-smoke-run`, an explicit local orchestrator
  that requires exact repository, Rust-source, supervisor-command, and target
  contracts before invoking the existing exact-pane graceful restart executor.
  After a successful restart it waits one caller-bounded observation interval
  and runs the existing bounded in-memory smoke collector over the exact
  restart-through-observation window. It emits aggregate restart counts and
  sanitized smoke evidence only, never pulls or builds code, SSHes, applies
  force escalation or broad process-pattern signals, or writes report files;
  post-action smoke failures leave the relaunched bots running and fail red for
  operator follow-up.

- Added `passivbot tool live-repository-prepare`, an explicit local executor
  that fetches only the pinned public canonical `origin/master`, requires exact
  caller-confirmed
  current and target commits, a tracked-clean `master` checkout with no Git
  operation in progress, and a hook-disabled true fast-forward before moving
  the worktree.
  It then verifies or rebuilds the Rust extension in a fresh bounded child
  process and requires an exact caller-confirmed source fingerprint before the
  checkout is restart-ready. It preserves untracked files and never SSHes,
  contacts exchanges, signals or starts live bots, force-checks out, or rolls
  back a target whose Rust preparation failed.

- Added `passivbot tool live-restart-smoke-evidence`, a pure fail-closed
  evaluator for already-generated full restart target and smoke JSON reports.
  It binds stable exact-target evidence and bounded post-restart monitor/log,
  shutdown, startup, and repository evidence to the expected head, supervisor
  fingerprint, and target count without executing report producers or
  controlling processes. Event and log windows retain and compare exact bounded
  epoch-millisecond values, and dropped hard-looking log evidence fails closed.

- Added `passivbot tool live-restart-smoke-collect` to collect the existing exact
  local target and bounded smoke reports in memory and immediately evaluate the
  sanitized restart evidence contract. It performs only local filesystem and
  bounded `tmux`/`ps`/`git` inventory reads; it does not write intermediate
  reports, pull or build code, contact exchanges, or control processes. Exact
  historical event windows select only managed rotation segments whose encoded
  intervals overlap the requested bounds, require retained predecessor coverage,
  and fail closed before content scanning when names, per-bot counts, global
  counts, or selected bytes exceed the bounded policy. Sanitized output reports
  aggregate selection completeness, counts, scan bytes, and code-owned issues.

- Live trailing restart reconciliation now preserves authoritative position-update
  timestamps from CCXT and raw exchange payloads and no longer
  treats position creation time as proof that fill history is current. Exchanges
  without an update timestamp remain fail-closed until a successful fill refresh
  after the position snapshot reports an after-state matching the live position.
  Unchanged position snapshots no longer associate a prefetched fill with the old
  position state, preventing a later real position change from waiting for an
  additional fill.

- Added `passivbot tool live-restart-executor`, an explicit local executor for
  exact tmux targets that already pass the bounded stable target report. It
  requires the expected Git commit, a tracked-clean checkout, an
  operator-confirmed Rust build-input fingerprint, a Rust extension stamped
  with that same fingerprint, a final rehash that detects ignored build-input
  drift during verification, the expected full supervisor-command fingerprint,
  and `--execute`,
  sends one Ctrl-C round only to verified panes, waits a bounded time for exact
  process exits, rechecks repository/runtime artifacts plus the private
  supervisor snapshot and pane/process identity before typing launch commands,
  and verifies stable replacements. It never pulls or builds code or applies
  an automatic force signal; partial or changed state fails closed for manual
  recovery.

- Exact live restart target sampling now binds pane/PID stability to an opaque
  fingerprint of the complete parsed supervisor command contract before
  report redaction or truncation, failing closed when that contract changes or
  is unavailable without exposing command content.

- Live trailing extrema now latch an affected position side unavailable as soon
  as an authoritative position change is observed, including the first
  position snapshot after restart, and remain unavailable until a newer fill
  identity at or after the authoritative position timestamp plus complete
  finalized 1m coverage is present. Fill refreshes that arrive before the
  matching position snapshot confirm against the prior snapshot's fill epoch.
  Side-scoped trailing failures and panic plans no longer suppress valid
  reconciliation on the unaffected hedge side.

- Live trailing extrema now reset independently per symbol and position side
  after every confirmed fill. Ordinary trailing closes are withheld and stale
  trailing orders retired until post-fill candles establish fresh extrema,
  while panic/HSL exits remain available. Trailing-martingale monitor
  diagnostics now use the Rust close formula and report its exact wallet
  exposure and volatility inputs.

- Exact live restart targets now classify whether the bot is a child of its
  tmux pane parent and therefore has a bounded candidate relaunch path after a
  required post-stop pane recheck. The report exposes only method/proof
  metadata, never the configured command, and restart plans require every
  resolved target to be relaunch-ready.

- `passivbot tool live-restart-smoke-plan` can now bind an explicitly
  confirmed tmux session into its pre-restart readiness phase with
  `--target-session-name`. The generated local-only target preflight requires
  2-5 stable identity samples, remains non-executing, and makes the missing
  target gate visible when no session is configured.

- Added bounded `--samples` and `--interval-s` stability checks to
  `passivbot tool live-restart-target-report`. Multi-sample reports fail if any
  local preflight is hard-red or if a window's canonical pane ID, pane PID,
  matched bot PID, or ownership proof changes during the sample window. Restart
  execution and process control remain unavailable.

- Added `passivbot tool live-restart-target-report`, a bounded local-only
  preflight that joins supervisor-config window names with canonical tmux pane
  IDs and proves ownership by matching each bot process PID or parent PID to
  its pane PID. It fails on missing, duplicate, unconfigured, or mismatched
  panes and never signals, starts, or controls a process, contacts a network
  or exchange, loads credentials, or writes files.

- Added `passivbot tool live-process-report`, a bounded local-only process
  sampler that does not read monitor events or text logs, access credential
  stores, contact networks or exchanges, control processes, or write files.
  It can compare the local process table with an optional supervisor config
  and sample process-state persistence/recovery using the existing smoke-report
  process contract. Use `--brief` for aggregate-only process, config, resource,
  state, and sampling counters without command, account, path, PID, or
  per-process rows.

- Live smoke reports can now opt into bounded repeated process-table sampling
  to distinguish observed, persistent, and recovered uninterruptible-sleep
  states while reporting stable PIDs, command/PID churn, and aggregate state
  observations. The default remains one process snapshot, the final sample
  retains the existing liveness/config verdict contract, and the sampler never
  signals or restarts a process.

- Added `passivbot tool trailing-inspect`, an offline one-shot explanation of effective
  `trailing_martingale` entry and close thresholds, retracements, and analytical prices from a
  config or explicit parameter overrides.

- V7 trailing-grid migration now disables the v8 TWEL entry gate when a non-positive v7 TWEL
  threshold disables its enforcer, collapses compatible v7 live/backtest warmup caps into the
  shared v8 field, and validates the complete canonical result before writing. The CLI now writes
  a durable JSON report by default while showing a concise action summary, and the new
  `compare-backtests` tool reports dataset, metric, equity-path, and fill-event differences between
  completed v7 and v8 artifacts without assigning a safety verdict.

- Live smoke reports now expose bounded cycle terminal-outcome health from
  existing `cycle.completed` and `cycle.degraded` events. The projection shows
  latest successful/degraded outcomes, successful completion after the latest
  observed degradation, elapsed time, order-change presence, and allowlisted
  phase maxima without copying arbitrary payload details or exception text.
  Existing hard-failure verdicts, cycle execution, exchange access, and trading
  behavior are unchanged.

- Live smoke reports now expose bounded latest-per-bot planning-snapshot health
  from existing `snapshot.built` events: required-surface and age coverage,
  market-snapshot freshness and missing counts, availability status counts,
  packet counts, and correlation IDs. Symbol lists, packet rows, raw references,
  hashes, values, and embedded availability records remain query-only; smoke
  verdicts, snapshot construction, exchange access, and trading behavior are
  unchanged.

- Live smoke reports now expose bounded latest-per-bot-and-kind health from
  existing `data_packet.updated` events: packet kinds, quality, freshness,
  source, revision, safe coverage counts, and warning/error counts. Raw packet
  references, hashes, timestamps, values, and warning/error text remain
  query-only; smoke verdicts, exchange access, and trading behavior are
  unchanged.

- Binance backtest OHLCV downloads now fill the current v2 store from checksum-verified Binance
  Vision monthly archives first, parallel daily archives second, and CCXT last. Monthly archives
  are limited to sufficiently incomplete published months, recent days stay on CCXT, existing
  valid rows are never overwritten, and CCXT repairs archive failures or real gaps within archive
  data. The legacy downloader path is not re-enabled.

- Live smoke reports now include bounded latest-per-bot planning-output health
  from existing `rust_orchestrator.returned` and `action.planned` events. The
  projection correlates cycle and remote-call IDs, Rust timing and order
  counts, planned order classifications, redacted symbol samples, truncation,
  and count mismatches without copying raw orders, quantities, prices, or
  hashes. Smoke verdicts, planning, exchange access, and trading behavior are
  unchanged.

- Staged-readiness smoke reports now include bounded latest-per-bot
  `entry.initial_eligibility` aggregates: evaluated and record totals, outcome
  and reason counts, truncation coverage, and correlation IDs. Raw per-symbol
  records remain query-only, and smoke verdicts, entry eligibility, exchange
  access, and trading behavior are unchanged.

- Staged-readiness smoke reports now include bounded latest-per-bot
  `planning.symbol_state` evidence across full, summary, brief, and selected
  output. The projection reports availability counts, unavailable reasons,
  order classes, surfaces, and redacted symbol samples without changing smoke
  verdicts, console output, planning, exchange access, or trading behavior.

- Time-bounded live smoke reports now scope monitor event-type inventory and
  sampled cycle IDs to the requested event window. Full-file validation and
  scanned-record counters remain unchanged.

- Live smoke reports now expose bounded existing
  `forager.eligibility_changed` evidence across full, summary, brief, and
  section-selective output, including approved/ignored add/remove counts,
  source kind, position side, and redacted symbol samples. The projection does
  not change smoke verdicts, console output, eligibility, or trading behavior.

- Staged-readiness smoke summaries now include bounded existing
  `planning.defer_summary` evidence such as defer counts, windows, required
  surfaces, retry mode, and redacted symbol samples. Registered optional
  section selectors also remain valid when their event family is naturally
  absent from the selected window.

- Live smoke reports now expose bounded existing
  `forager.feature_unavailable` evidence across full, summary, brief, and
  section-selective output. The projection does not change smoke verdicts,
  console output, forager selection, readiness, or trading behavior.

- Live configs may now define optional diagnostic `startup_phase_budgets` for
  canonical startup timing phases. Existing `bot.startup_timing` events carry
  configured elapsed and phase budgets, and live smoke reports prefer those
  explicit values over historical p95 projections. Live performance reports
  now retain bounded latest-lifecycle configured-budget assessments and
  aggregate their status. The budgets are reporting metadata only and do not
  gate startup, readiness, exchange access, or trading.

- `log-secret-inventory --summary` now emits bounded aggregate scan evidence
  without the per-file paths, ages, or hashes retained by the full report.

- The read-only log secret inventory now detects credential-like query
  parameters in scheme-less request paths and query fragments as well as full
  URLs, without returning matched values.

- Brief live smoke reports now summarize startup elapsed/phase budget status
  coverage, distinguishing unavailable and no-baseline assessments from
  within-budget phases without changing startup behavior or thresholds.

- A new read-only `log-secret-inventory` tool scans bounded current and rotated
  text-log input for secret-like artifact classes without returning matched
  values or source lines. Reports contain only aggregate class counts, bounded
  root-relative file identity, age, size, scan status, and stable hashes. The
  tool never rewrites, deletes, quarantines, copies, or uploads log artifacts.

- Successful maintainer cancellation summaries and hourly scheduler-jitter
  detail now remain available at DEBUG instead of adding routine INFO lines.
  Maintainer cancellation failures remain ERROR. Task cancellation, scheduler
  timing, exchange calls, and trading behavior are unchanged.

- Startup lifecycle output now has one normal human-ready signal: structured
  startup timing. `bot.started` keeps its structured console/text projection
  with a legacy fallback only when that projection is unavailable, while
  durable `bot.ready` stays out of console/text. Decorative READY banners,
  duplicate execution-loop startup lines, and routine warmup/maintainer detail
  no longer add INFO noise. Background candle-warmup success detail is DEBUG
  under `[candle]`; failure remains immediate ERROR. Startup timing, task
  scheduling, monitor persistence, and trading behavior are unchanged.

- The startup HSL safety warning now uses compact mode/budget/threshold wording
  so its complete deposit, withdrawal, balance-override, configuration-change,
  and history-reinterpretation warning fits the normal console record budget.
  Warning admission, HSL configuration, risk behavior, and trading behavior are
  unchanged.

- Approved/ignored coin membership changes now use one bounded structured
  console/text projection with per-side counts and symbol samples instead of
  unbounded legacy symbol lists. Existing structured event data, membership
  calculation, list mutation, and trading behavior are unchanged.

- Exchange clock-offset recovery warnings and durable time-sync diagnostics no
  longer retain raw exception text. They keep the stable exception class,
  source, recovery status, and bounded client outcome details within the normal
  rendered console record budget.

- Candle-fetch retry and exhaustion warnings now fit the normal rendered
  console record budget while retaining exchange, symbol/timeframe, attempt,
  elapsed time, exception class, and action. Structured diagnostics, redaction,
  retry behavior, candle readiness, and trading behavior are unchanged.

- Required-EMA-unavailable warnings now use a bounded structured console/text
  projection with counts, classified cause, nontradable-until-fresh action,
  and compact symbol/error/EMA identity. The complete structured payload,
  existing fifteen-minute human cadence, EMA readiness decisions, and trading
  behavior are unchanged.

- Periodic health summaries now use compact operator labels and separators so
  uptime, loop, position, balance, order/fill/error, resource, and slow-phase
  context fit the normal console record budget. Complete structured payloads,
  cadence, health calculations, and trading behavior are unchanged.

- Forager selection transitions now use a bounded operator summary with slot,
  selected/incumbent, hysteresis, reason, ranking, and replacement context.
  Complete structured and DEBUG diagnostics, transition cadence, Rust-owned
  selection behavior, and trading behavior are unchanged.

- Candle-health transition summaries now retain their aggregate counts and
  bounded missing-candle examples within the normal console record budget.
  Complete debug diagnostics, candle-health calculations, fetch behavior,
  readiness checks, and trading behavior are unchanged.

- Initial-entry distance-gate blocked events now retain every existing
  per-symbol structured/monitor record while limiting console/text and legacy
  fallback output to one bot-level representative per five minutes, with
  bounded active/suppressed counts. Cleared transitions remain immediate; entry
  eligibility, distance calculation, gate decisions, order creation, and
  trading behavior are unchanged.

- Close-EMA carry-forward warnings now use a bounded structured console/text
  projection with counts, symbol/span examples, age, fallback streak, and a
  classified reason. The existing fifteen-minute warning cadence, complete
  event payload, legacy fallback, EMA calculations, and trading behavior are
  unchanged.

- Slow staged account-refresh timings and periodic timing summaries now use a
  compact structured console/text projection. The existing ten-second detail
  threshold, complete event payloads, legacy fallback, refresh calls, and
  trading behavior are unchanged.

- HSL startup settings now use a compact INFO projection that retains the
  configured thresholds, spans, mode, tier ratios, actions, and restart policy
  within the normal console line budget. HSL validation, state reconstruction,
  risk decisions, and trading behavior are unchanged.

- Candle fetch-lock hold warnings now retain the affected exchange, symbol,
  timeframe, and compact local-holder identity/timing without repeating the
  deterministic lock path, duplicated owner scope, configured timeout, or
  implied action. Watchdog timing, lock behavior, and warning cadence are
  unchanged.

- CLI overrides of `live.approved_coins` now use a bounded startup log summary
  with per-side counts and three-symbol samples instead of printing the full
  old and new collections. Config application and non-target config-change
  logging are unchanged.

- Visible `trailing.status` console/text records now use a compact operator
  projection that fits the normal line budget while retaining status, position
  identity, mode, trigger gates, material threshold/retracement values, current
  price, and correlation. Structured and monitor payloads, visibility
  admission, cadence, trailing calculations, and trading behavior are
  unchanged.

- Five-minute `trailing.status` and `unstuck.status` snapshots now remain
  complete in structured and monitor sinks while normal console/text output is
  limited to first observations, qualitative or material numeric transitions,
  and hourly reminders. Unstuck allowance movement uses five-percent relative
  hysteresis; trailing ratio and price movement use explicit 0.05 percentage-
  point and 0.5-percent boundaries. Trading, risk, planning, and event cadence
  are unchanged.

- Material live memory snapshots now emit a bounded `resource.memory_snapshot`
  event and one compact operator line instead of a 457-525 character cache and
  task diagnostic. The complete bounded samples remain available in structured
  and monitor sinks, detailed diagnostics remain at DEBUG, and collection,
  cadence, admission, and trading behavior are unchanged.

- Execution-loop failures now use a bounded incident signature in normal
  console, monitor, and structured burst output instead of retaining raw
  exception text, request URLs, and unconditional tracebacks. Stack frames
  remain available at DEBUG without the exception value, and the error-budget
  summary is compact and tagged; retry, restart, and trading behavior are
  unchanged.

- Intermediate HSL coin-history replay progress now appears on the normal live
  console at most once every 30 seconds. The first progress update, completion,
  complete structured progress events, replay behavior, and safety readiness
  are unchanged.

- OKX per-symbol margin-mode/leverage configuration now emits bounded structured
  outcome events. Explicit already-configured responses stay available at DEBUG
  instead of appearing as routine INFO, while confirmed responses and failures
  retain their existing operator visibility and exchange behavior.

- Successful fill-refresh and fetcher-request timing detail now stays at DEBUG
  instead of appearing in the normal live console. Fetcher request errors,
  actual fills, degraded warnings, structured refresh summaries, refresh
  behavior, and failure propagation are unchanged.

- Completed staged account-refresh timing lines now stay at DEBUG unless the
  cohort takes at least ten seconds. Interesting sub-ten-second samples remain
  available as structured INFO events, and periodic timing summaries,
  readiness, exchange calls, and trading behavior are unchanged.

- Candle-fetch retry and exhaustion warnings now use bounded operator
  signatures instead of retaining raw request parameters and exception text.
  Their first occurrence is emitted even during the process's initial
  five-minute throttle window.
  Structured remote-call events omit raw exception text and replace explicit
  request URLs with a redacted marker plus stable hash; retries, callback
  invocation, and candle behavior are unchanged.

- Routine open-tail EMA projection-context aggregates now remain available at
  DEBUG instead of printing thousand-character diagnostics in the normal live
  console. Compact active-tail transitions and warnings, EMA projection,
  readiness, and trading behavior are unchanged.

- Rust-orchestrated forager selection changes now have one normal console/text
  owner: the producer's materiality-aware INFO summary. Their complete
  `forager.selection` events remain available in structured and monitor sinks
  without printing a second independently throttled summary. Python-filter
  selection events retain their existing console/text projection.

- Successful background forager candle-refresh completions now remain available
  at DEBUG instead of repeating in the normal live console. Refresh scheduling,
  wall-time-cap notices, failures, and candle behavior are unchanged.

- Successful candle-index maintenance now stays available at DEBUG under the
  `[candle]` tag instead of appearing as repeated `[boot]` INFO output after
  readiness. Index rebuild behavior and failure visibility are unchanged.

- Routine successful candle-warmup details now remain available at DEBUG
  instead of filling the normal live console. Startup readiness milestones,
  degraded/failure visibility, structured cache decisions, and warmup behavior
  are unchanged.

- Backtests now write `drawdown.png` alongside the existing summary plots,
  showing drawdown from the running peak of collateral-agnostic strategy equity.

- Raw-only wallet-balance jitter no longer appears on the normal live console
  when the hysteresis-snapped balance is unchanged. The complete
  `balance.changed` event remains available in structured, monitor, and durable
  text sinks; snapped changes and events without valid materiality metadata
  remain console-visible.

- Realized-loss gate blocks now use their structured warning as the sole normal
  console/text line when a live-event console sink is available. The legacy
  warning remains the fallback; gate decisions, diagnostics, and throttling are
  unchanged.

## v8.0.0 - 2026-07-14

- Restored the documented offline HSL restart smoke by pinning its fixture to
  the intended per-side HSL contract and updating its post-flatten drawdown
  assertion to the current panic-fill-anchored finalization semantics. Coin-mode
  fake-live RED handling now exercises the production protective supervisor
  instead of falling through normal planning while confirmations are pending.

- Detailed min-effective-cost entry blocks now use their structured events as
  the sole normal console/text lines when a live-event console sink is
  available. Legacy detail lines remain the fallback, while the distinct
  throttled aggregate summary and all gate behavior are unchanged.

- Initial-entry distance-gate blocked and cleared transitions now use their
  structured events as the sole normal console/text lines when a live-event
  console sink is available. The throttled legacy summaries remain the
  fallback; gate state, eligibility, and order filtering are unchanged.

- Ambiguous-cancel terminal states now use the structured execution warning as
  the sole normal console/text line when a live-event console sink is
  available, with an explicit full-account-confirmation cue in its compact
  projection. The legacy summary remains the fallback; cancellation and
  authoritative-confirmation behavior are unchanged.

- Execution-loop error bursts now use the structured `health.summary` console
  projection as the sole normal console/text line when a live-event console sink
  is available. The legacy warning remains the fallback when that projection or
  its emitter is unavailable; error thresholds, redaction, restart/backoff, and
  trading behavior are unchanged.

- Periodic health console/text output now projects the structured
  `health.summary` event as one compact line when the live event console is
  available, while retaining the same legacy fallback when it is disabled or
  absent. Health payloads now include quote, error-budget, bounded slow-phase,
  and correct RSS fields without changing trading behavior.

- Live fill console/text output now projects structured `fill.ingested` events,
  avoiding duplicate legacy lines when the structured console is available.
  Large fill batches emit one `fills.ingested_summary` console/text event while
  retaining every per-fill structured and monitor event; fill accounting,
  history, and PnL semantics are unchanged.

- Monitor retention now builds the same complete recursive inventory with one
  healthy-path `os.scandir` traversal and one explicit `DirEntry.stat` per
  visited entry, avoiding the duplicate directory scans performed by the prior
  `Path.rglob` path. A failed directory scan gets one bounded immediate retry;
  persistent errors still isolate that subtree. Cadence, symlink handling,
  protected paths, accounting, and deletion policy are unchanged.

- Periodic structured health summaries now split monitor retention work into
  fixed inventory, age-unlink, and byte-cap-unlink timings plus bounded work
  counts. Live smoke and performance reports project the same fields without
  changing retention cadence, policy, failure handling, report verdicts, or
  trading behavior.

- Balance-change console and text lines now show exact raw and snapped
  before/after transitions with signed deltas, equity, and source in one compact
  human-readable line. The complete structured event remains unchanged.

- Monitor retention pruning now inventories retained files once per due run and
  reuses the same size/mtime snapshot for age and byte-cap deletion. Retention
  cadence, protected files, recursive byte accounting, direct-only deletion,
  and oldest-first policy are unchanged.

- Position-change console and text lines now use a compact aligned transition
  with old/new size and price plus WE, base-WEL, effective-WEL, TWEL, and uPnL.
  The complete structured event remains unchanged for queries and incident
  reconstruction.

- Periodic structured health summaries now separate event-path manifest
  checkpoint and retention run counts and service totals/maxima within the
  existing inclusive monitor maintenance timing. Live smoke and performance
  reports project the fixed fields without changing persistence cadence,
  retention policy, event delivery, verdicts, or trading behavior.

- Monitor event and history appends now coalesce the best-effort manifest
  checkpoint to the existing snapshot cadence while forcing it at lifecycle
  and rotation boundaries. Startup also recovers the event sequence from a
  fixed-memory scan of checksummed current-segment recovery trailers when an
  unclean exit leaves the manifest stale, preventing sequence reuse or payload
  marker confusion without changing NDJSON append durability or trading
  behavior.

- Periodic structured health summaries now split real monitor-sink service time
  into fixed event-conversion, publisher lock-wait, rotation, persistence, and
  maintenance totals/maxima. Live smoke and performance reports project the
  same bounded fields without changing monitor persistence, event delivery,
  verdicts, or trading behavior.

- Periodic structured health summaries now attribute queued worker sink time to
  fixed structured and monitor sink classes with bounded per-window write
  counts and service totals/maxima. Live smoke and performance reports project
  the same fields without changing routing, delivery, verdicts, or trading
  behavior.

- Periodic structured health summaries now expose bounded event-pipeline queue
  wait and worker sink-service timing windows. Live smoke and performance
  reports project the processed counts, totals, and maxima without changing
  event delivery, smoke verdicts, or trading behavior.

- Reorganized AI-facing engineering documentation around a compact mandatory instruction set,
  explicit live/authenticated-operation approval boundaries, task-routed feature contracts and
  runbooks, generated live-event references, and warning-only documentation size/structure checks.
  The local fake-live harness is now explicitly distinguished from public-network and authenticated
  exchange validation. Temporary compatibility routes preserve active external review automations
  while they migrate to the canonical Markdown principles, validation matrix, and PR-review runbook.

- `live-performance-report` now correlates current-startup fill-cache loading
  with exact post-start fill-history coverage proof. Cache presence alone never
  claims proof, cache/proof ordering is explicit, and incomplete or absent
  lifecycle evidence remains unknown.

- Live order writes now emit bounded structured/monitor evidence immediately
  before concrete connector create and cancel calls. The events distinguish
  local connector-call arrival from pre-call submission intent and exchange
  acknowledgement without changing order payloads or execution behavior.

- `live-performance-report` now derives the current lifecycle's first
  connector-bound initial-entry eligibility milestone from
  `entry.initial_eligibility`. Blocked, candidate-free, protective-only, and
  malformed eligibility events do not claim fresh-entry readiness.

- Completed normal live order plans now emit one bounded structured/monitor
  `entry.initial_eligibility` event. It distinguishes fresh initial entries
  that were absent, already satisfied, blocked by an existing local gate,
  accompanied only by protective actions, or selected for the final
  connector-bound create batch immediately before invocation. The event
  observes existing reconciliation
  and execution decisions only; it does not add a gate, change order batches,
  or claim exchange acknowledgement.

- Startup readiness reporting now uses one canonical phase parser across
  performance and smoke consumers, rejects conflicting `phase`/legacy `stage`
  records, rejects stale rotated lifecycle data after any incomplete current
  tail, preserves bounded HSL replay context across sparse terminal events, and
  keeps prior restart samples available for smoke timing budgets.

- `live-performance-report` now derives bounded current-lifecycle startup
  milestones for the first cycle, first Rust call, and first submitted
  exchange write from existing structured events. Missing observations remain
  explicitly unknown; submitted write events do not claim connector success
  or fresh-entry eligibility.

- Capped rotated `live-performance-report` scans now keep each bot's startup
  readiness snapshot on the latest observed lifecycle even though recent-file
  selection reads `current.ndjson` before older segments. Historical aggregate
  startup distributions remain unchanged.

- Existing `bot.startup_timing` events now include bounded machine-readable
  readiness scope and trading-impact labels for account, HSL protective,
  execution-loop, first-market-state, and background-candle milestones. The
  best-effort active-candle phase remains timing-only. Live performance and
  smoke reports expose per-bot and aggregate readiness SLA timing without
  changing startup sequencing, readiness gates, exchange calls, or trading
  behavior.

- Coin-mode HSL replay progress now separates scanned candidate rows from
  applied state-update rows. Live performance and smoke reports use scan
  throughput for remaining-work estimates when available, retain explicit
  legacy applied-row fallback labeling for older events, use the dense upper
  bound for the generic active replay estimate while keeping required-pair
  estimates separate, preserve exact terminal candidate-work estimates, and
  keep legacy terminal events without candidate totals from reporting active
  remaining work. These report semantics do not change HSL replay ordering,
  readiness, or trading behavior.

- Isolated-only markets excluded from new entries by cross-margin preference
  now emit bounded per-side `config.market_compatibility` events while
  preserving the existing filter and warning behavior.

- Hyperliquid non-unified HIP-3 startup rejection now emits a bounded terminal
  `config.market_compatibility` event before preserving the existing fatal
  error. The event records only redacted account/capability counts and symbol
  samples and does not change market, margin, or startup policy.

- Unsupported configured markets now emit bounded
  `config.market_compatibility` structured and monitor events with list,
  position-side, count, redacted symbol sample, and stable reason context.
  Existing coin filtering and text-log warnings are unchanged.

- Added `passivbot tool pareto --scenario LABEL` for rebuilding and selecting from a
  scenario-specific nondominated sub-front of a suite optimization's saved aggregate Pareto
  members. Scenario metrics are used consistently for limits, objectives, and ranking, and output
  explicitly notes that candidates discarded from the original aggregate front are not recoverable.

- `passivbot tool live-performance-report` now retains each bot's explicit HSL
  protective-ready replay milestone and summarizes replay history formats,
  protective-ready elapsed time, and completed full-replay elapsed time from
  existing structured events. If the earlier milestone has rotated out of the
  selected event files, its aggregate elapsed value is recovered from the
  retained completion event. This is read-only reporting and does not change
  HSL startup or trading behavior.

- Cold coin-mode HSL history reconstruction now uses a private compact
  NumPy-backed replay payload instead of retaining the full nested per-minute,
  per-symbol timeline. Public balance/equity history output, pside/unified HSL,
  cache authority, episode/cooldown rules, and Rust risk math are unchanged.
  The offline replay benchmark can now separate held and background work, opt
  into a 30-day local-scale fixture, compare rich and compact history formats,
  and report Python allocation peaks.

- `passivbot tool live-smoke-report --processes` now includes bounded current
  live-process state counts, uninterruptible-sleep count, and CPU, memory, and
  RSS totals/maxima/reporting counts in full, summary, and brief output. These
  read-only fields remain observational and unavailable metrics stay null.

- Coin-mode HSL startup now reconstructs currently held pairs before declaring
  protective readiness, then continues cooldown-affected and remaining flat
  pairs in one shutdown-owned background replay. Initial entries remain
  blocked per coin and position side until that pair is reconstructed, while
  cancellations and panic/reduce-only protection remain available. Replay
  events and smoke reports distinguish protective-ready time from full replay.
  After protective readiness, replay now yields in smaller bounded chunks with
  a short cooperative pause so live exchange I/O is not starved; held-position
  reconstruction retains the faster startup cadence.

- Coin-mode HSL drawdown normalization now uses one Rust-owned live/backtest
  contract: account balance divided by the applicable slot count. TWEL still
  enables the side but no longer scales the HSL denominator, so increasing an
  exposure allowance cannot silently weaken the configured RED threshold.

- HSL RED episode finalization now uses one Rust-owned live/backtest contract
  for caller-supplied persistent no-restart peaks, restart policy, and cooldown
  deadlines. Coin-mode live restart now retains that no-restart peak like
  pside live and backtest instead of discarding it with the episode tracker.
  Python remains responsible for exchange/history proof and supplies the exact
  scope-flattening fill timestamp; backtests retain the exact configured
  deadline instead of extending sub-bar cooldowns to a full candle interval.

- Added `passivbot tool hsl-replay-benchmark`, a bounded offline benchmark for
  the current coin-HSL history initializer. It emits deterministic fixture and
  final-state hashes, explicit timeline-row and pair-row throughput, profiled
  stage timings, replay counters, and side-effect counters without contacting
  exchanges or reading/writing live cache and state artifacts.

- Approved and ignored forager-eligibility membership changes now emit bounded
  `forager.eligibility_changed` structured and monitor events. Each aggregate
  event identifies the list, add/remove operation, source kind, and per-side
  count with at most 12 sorted symbols; existing eligibility behavior and text
  logs are unchanged.

- `passivbot tool crash-finder` can now regenerate scenario suites from an existing
  `crash_clusters.csv` without rescanning local OHLCV data, emit market-wide/coin-focused/single-coin
  filtered suites, merge overlapping stress windows, and add per-coin forced-normal overrides for
  idiosyncratic non-market-wide crash stress scenarios, capped at two forced coins per scenario.
  When scanned range metadata is available, generated suites now drop coins with no cached data
  overlap in the scenario date window and omit targeted scenarios when no coins remain. Full
  discovery now efficiently groups 1m source rows into parameterized crash candles (`1h` by default)
  without rescanning the full minute array for every candle, while preserving the ordered
  high-to-later-low metric.

- Websocket reconnect attempts now emit bounded `websocket.reconnect`
  structured events with retry timing, fixed reason classification, text-log
  visibility, traceback cadence, and exception type. Existing reconnect timing,
  warning throttling, traceback logging, and exchange behavior are unchanged.

- Connector-local exchange-config failure logs in Binance, Bitget, Defx,
  Hyperliquid, KuCoin, and OKX, plus the parent per-symbol retry log, now keep
  bounded operation, symbol, retry, canonical known-code, and exception-type
  context without rendering arbitrary exception messages or partial API
  responses. Existing connector catches, fail-loud behavior, retries, and
  per-symbol success/failure handling are unchanged.

- Exchange-config success logs now use one bounded, value-safe formatter across
  the shared CCXT connector and Binance, Bitget, Bybit, KuCoin, and OKX. Raw API
  response values are replaced by canonical status, finite numeric leverage,
  bounded numeric code, or response type/presence labels; exchange calls and
  failure behavior are unchanged.

- Live executor create/cancel anomalies, including lower-level base/CCXT order
  write failures, no longer print raw order dictionaries, exchange responses,
  exception messages, or tracebacks. Existing bounded structured execution
  events remain authoritative; when their console projection is unavailable,
  fallback logs contain only safe action/symbol/type/reason context. Exchange
  behavior is unchanged.

- Live performance timing groups now expose their latest bounded report-safe
  canonical event IDs, including in `operation_durations` and
  `slowest_blockers`, so an operator can correlate a slow row with the
  structured event stream without exposing free-form event payloads. Existing
  legacy snapshot IDs are normalized consistently with `live-event-query`, and
  equal-timestamp samples use persistent event ordering.

- Backtests now warn when interior data gaps split a coin's history and real
  data outside the longest contiguous run is excluded from the backtest
  (previously silent). Stock-perps (`xyz:`) coins instead log their
  synthetic-flat-candle share at INFO level. The synthetic-candle backtesting
  model for stock perps (tradable flat candles during underlying-market
  closure) is now documented in docs/stock_perps.md with its accepted
  modeling caveats.

- Hardened OHLCV gap classification against transient exchange conditions. A
  persistent gap (missing tail, leading, or internal range) now gets a short
  one-hour re-verification window on its first observation and keeps the full
  seven-day window only once the identical gap is observed again at least 30
  minutes later — so an exchange publishing delay or partial response can no
  longer silently clip a coin's backtest data for a week. KuCoin pagination
  holes between pages are now recorded as expiring auto-detected gaps
  (retried on later fetches) instead of permanent no-trade gaps; holes inside
  a single exchange response remain verified no-trade minutes.

- Faster backtest startup on hlcvs cache hits: the multi-GB hlcvs artifact is
  now decompressed once instead of twice (manifest verification hands its
  arrays to the loader), array/chunk hashing no longer materializes a full
  copy of the data, and the OHLCV catalog reuses its sqlite connection instead
  of reconnecting per query. Cache formats, hashes, and outputs are unchanged;
  manifest verification now logs its elapsed time separately.

- Fixed Alpha Vantage stock-perps data provider misfiling candles by 4-5 hours:
  its US-Eastern timestamps were interpreted in the host's local timezone
  (DST-dependent) instead of America/New_York. Backtest data fetched with
  `tradfi.provider = "alphavantage"` before this fix should be re-downloaded.

- Live auto-unstuck emission is no longer gated in Python by whether an
  unstuck order is already resting on the exchange. Rust owns whether an
  unstuck ideal order is emitted from the realized-PnL cumsum facts; duplicate
  order risk now rides the same live reconciliation path as every other order
  type.

- HSL flat detection now uses a shared half-qty-step epsilon where symbol
  precision is available, including replay cache extension, pside/unified cache
  synthesis, current-episode proof, and coin replay episode transitions. This
  keeps dust below half a step from extending or restarting HSL episodes.

- Plan tracker: closed the Python-simplification item, the final open item
  of the risk/unstuck/HSL action plan. Removed live-path policy
  re-decisions: execution type, the redundant unstuck-suppression channel,
  and the per-cycle unstuck-allowance computation. The remaining
  Python-side order handling is documented as intentionally Python-owned
  reconciliation or live-only execution/data guards. Docs-only change.

- The live path no longer computes unstuck allowances for the Rust
  orchestrator input. Rust has always derived the unstuck allowance
  internally from the realized-pnl cumsum facts (risk.rs); the
  unstuck_allowance_long/short input fields were consumed only as a legacy
  fallback for the auto_unstuck_allowed flag, which live callers always set
  explicitly. The fields are now optional (serde defaults) and documented
  as legacy/diagnostic; live inputs and recorded planning snapshots omit
  them, removing the last per-cycle duplicate of the allowance formula from
  the hot path. The monitor still computes allowances on demand for
  diagnostics. Behavior unchanged.

- Optimizer defaults now keep HSL restarts enabled by setting
  `bot.long/short.hsl.restart_after_red_policy=always` in
  `optimize.fixed_runtime_overrides` instead of forcing
  `no_restart_drawdown_threshold=1`. This avoids permanent optimizer halts
  while preserving the live/default no-restart threshold values for configs
  that use `restart_after_red_policy=threshold`.

- Live unstuck-allowance inputs to the Rust orchestrator are no longer
  zeroed while an unstuck order is resting on the exchange. The allowance
  values are pure budget facts derived from fill history; suppression of
  new unstuck emission rides solely on the existing auto_unstuck_allowed
  flag, which the Rust orchestrator consumes as the sole gate. Behavior is
  unchanged (Rust emitted no unstuck either way); this removes a redundant
  second suppression channel that made the allowance inputs diverge from
  the backtest for reasons unrelated to budget.

- Live order conversion no longer re-decides execution type in Python. The
  Rust orchestrator is the single source of execution-type truth
  (`should_use_market_execution` owns the panic market-vs-limit choice from
  `hsl_panic_close_order_type`); the Python fallback that re-derived it for
  short order tuples was dead on every live path and is replaced by
  fail-loud validation - a tuple without a valid `limit`/`market`
  execution type now raises instead of silently defaulting to a limit
  order, which could have downgraded a panic market close.

- Plan tracker: closed the canonical HSL equity-history signal design item.
  The one-raw-per-minute data-store goal is realized by the shared
  authoritative timeline plus the cache-primitive store (pair matrices +
  per-pside account series) consumed by all three signal modes, with
  sample-parity tests at every trust boundary; the originally sketched five
  named dataframes are superseded by these primitives. Docs-only change.

- Plan tracker: closed the HSL replay performance/readiness item. All
  sub-items are implemented (persisted npz+manifest checkpoints with
  watermark extension for all three signal modes, fail-closed reuse gates,
  doctor coverage, phased timing evidence); dense per-minute replay
  stepping remains Python->PyO3 by explicit choice, amortized to
  first-boot-only by cache reuse, with batch vectorization noted as
  optional future work contingent on production startup timings.

- HSL pside/unified startup replay now attempts cache reuse before the full
  history fetch, completing the replay-cache arc for all signal modes. The
  gate shares the coin-mode core (fresh fill-coverage proof, strict
  write-time-proven expected metadata, account/pair watermark agreement,
  gap panic-fill rejection, watermark extension from exchange fills/candles,
  current-position reconciliation) and adds the pair-completeness proof the
  aggregate synthesis requires: any fill inside the covered window or the
  extension gap belonging to a pair that is not currently held (and thus
  not cached) rejects reuse, because per-pside unrealized/flatness
  aggregates are summed from cached pair matrices alone. Any rejection or
  unexpected error falls back to the authoritative full replay. End-to-end
  test proves the cache-fed unified boot reaches state identical to the
  full replay with the fetch provably skipped.

- HSL pside/unified startup replay now derives cooldown and no-restart
  evidence from canonical reconstructed RED episodes, matching the coin-mode
  behavior shipped earlier: an episode that crossed RED and was flattened by
  an ordinary (non-panic) close now latches its cooldown anchored at the
  scope-flattening fill (falling back to the flatten row minute when no fill
  evidence exists) and evaluates restart_after_red_policy/no-restart at that
  stop via the persistent cross-episode tracker. Previously such episodes
  were silently dropped, so a restart during an active cooldown resumed
  trading. RED-free ordinary flattenings now perform a plain episode reset
  (clearing the episode's RED memory) instead of carrying state into the
  next episode.

- HSL: new pure, unwired synthesis helper
  `_hsl_replay_pside_timeline_rows_from_cache` converts persisted held-pair
  matrices plus the schema-v5 account series into the aggregate timeline
  rows the pside/unified startup replay consumes, with fail-loud
  span/continuity/alignment checks. Parity tests prove the synthesized rows
  equal the authoritative history timeline field-for-field (long and short
  pairs, realized events on both sides, flatness transitions) and that
  contract-shaped rows drive the pside/unified initializer to a state
  identical to authoritative-shaped rows. The helper is not yet consumed:
  the pside/unified reuse gate is a follow-up slice, and it must prove from
  fills that cached pairs were the only pairs with in-window positions
  before trusting these aggregates.

- HSL replay cache schema v5: the persisted account-level realized-PnL
  series now carries per-minute per-pside deltas (`pnl_long`, `pnl_short`)
  alongside the account-level `pnl`, collected from the authoritative
  per-pside running totals during the history replay and reproduced exactly
  by the watermark-extension helper (which now requires an explicit position
  side on every extension fill and rejects the cache otherwise). This is
  groundwork for the future pside/unified cache-reuse gate, whose timeline
  synthesis needs per-pside realized PnL. Existing v4 caches fail schema
  validation and are rebuilt by the next full replay, by design.

- HSL pside/unified startup replay now persists the same write-only replay
  cache as coin mode after a successful replay (held-pair raw matrices plus
  the account-level realized-PnL series). The cache config digest includes
  the HSL signal mode, so caches written by one mode can never be reused by
  another, and cache-write failures only warn - they never affect the
  completed replay. The caches are not yet read back on pside/unified boot;
  the reuse gate for those modes is a follow-up slice.

- Live coin-HSL startup replay now derives cooldown and no-restart evidence
  from canonical reconstructed RED episodes, not only from bot-emitted panic
  order markers. An episode that crossed RED and was flattened by an ordinary
  (non-panic) close fill - including a manual close - now latches its
  cooldown anchored at the scope-flattening fill timestamp and evaluates the
  no-restart policy at that stop, exactly like a confirmed panic marker.
  Previously such episodes were silently reset with no stop accounting, so a
  restart during an active cooldown (or after a terminal-drawdown episode)
  would resume trading. RED-free ordinary flattenings keep the plain episode
  reset. The Rust backtest already finalizes cooldown/no-restart for such
  episodes (its per-episode tier latch keeps the stop path armed after the
  sample recovers); new Rust regression tests pin that parity for both the
  pside and coin scopes.

- Repaired three stale Rust hard_stop unit tests that pinned pre-B2.1/A2.2
  mode-override behavior and were never run by CI: ORANGE tp-only now forces
  flat sides too (A2.2), and RED only authorizes panic while the current
  sample is actively RED - a recovered sample downgrades to tp-only (B2.1
  red split). The repaired red test now pins both branches; no behavior
  changes.

- HSL startup now applies the clarified incomplete-history policy: with
  `restart_after_red_policy=always`, missing pre-episode fill coverage
  degrades to a loud warning when the coin scope's current-episode start is
  provable from covered fills (the `always` policy ignores historical
  no-restart evidence); `threshold` and `never` still require full
  configured lookback coverage. A new dangerous per-run CLI flag
  `--hsl-accept-incomplete-history` lets an operator explicitly start on
  incomplete evidence for any policy, with a critical startup banner and
  per-use critical logs warning that panic/cooldown/no-restart may be wrong.
  The override is enforced as per-run only: values persisted in config
  files are stripped at load time (with a critical log) before CLI
  overrides are applied, so it can never survive a restart. Corrupt
  (pending/degraded) PnL data still always hard-fails.

- HSL RED cooldown now anchors at the fill that actually flattened the
  affected scope, by any means, instead of the latest bot-emitted panic
  fill. If a position is finished off manually (or by any non-panic close)
  after the last panic fill, the cooldown window starts at that flattening
  fill rather than earlier, so cooldowns can no longer expire prematurely
  for manually-completed flattens.

- Backtest ORANGE `tp_only_with_active_entry_cancellation` now forces flat
  symbols in the affected HSL scope too, blocking initial entries exactly
  like live has since the A2.2 contract change; previously backtests allowed
  new initial entries during ORANGE for symbols without a position, so
  backtest results could overstate entry activity near the orange tier.

- HSL panic orders are now authorized only while the CURRENT drawdown sample
  is in RED (`red_active_now`), in both live and backtest. Previously a
  latched RED episode kept emitting panic closes until the scope was flat
  even after the drawdown recovered; now a recovered sample pauses panic
  emission for the remainder of the episode while entries stay blocked
  (`tp_only_with_active_entry_cancellation`), and panic resumes if RED
  re-activates. Flat-scope stop finalization, cooldown, and no-restart
  accounting are unchanged and still use the episode's RED evidence.

- The HSL no-restart (permanent halt) trigger now evaluates
  `max(drawdown_raw, drawdown_ema)` against
  `hsl_no_restart_drawdown_threshold` in both live and backtest, instead of
  raw drawdown only. The permanent halt is intentionally conservative: it now
  also trips on sustained smoothed damage even when the instantaneous
  drawdown at the stop sample has partially recovered. The RED/panic-now
  trigger is unchanged (`min(raw, ema)` crossing `hsl_red_threshold`).
- `passivbot tool live-smoke-report` now summarizes existing cache load, flush,
  and warmup-decision events as `cache_health` in full/summary output and
  `cache` in brief output.
- Live coin-mode HSL startup can now reuse its persisted replay cache: when
  the cached series pass every trust gate (proven fill coverage at write and
  load time, config digest identity, watermark agreement, gap extension from
  exchange fills/candles, and current-position reconciliation), the bot
  replays from the cache plus the gap instead of re-fetching the full
  lookback. Any gate failure falls back to the full exchange-derived replay;
  the cache never becomes authoritative trading state, and a fresh VPS
  reconstructs identical decisions.
- HSL-enabled startup and live-config preflight now surface a history
  reinterpretation caveat and point operators to a dedicated HSL risks doc for
  deposits, withdrawals, balance overrides, and HSL config changes.
- Rust close-reducer pruning now keeps only the closest-to-fill reducer when
  multiple same-priority protective reducers target one coin+pside in the same
  ideal-order batch; ordinary grid/trailing close ladders remain preserved.
- Bounded `we_excess_allowance_mode` now treats non-positive or non-finite base
  WEL as zero allowed exposure and non-positive/non-finite TWEL as zero excess
  headroom instead of falling back to the raw excess percentage.
- Rust protective reducers now suppress lower-priority same-position ordinary
  close orders in the same ideal-order batch, so panic, TWEL/WEL auto-reduce,
  and auto-unstuck no longer stack with grid/trailing closes for one coin+pside.
- Rust WEL auto-reduce now takes priority over same-position auto-unstuck
  reducers in the same ideal-order batch, matching the documented reducer
  priority before auto-unstuck is admitted.
- Rust TWEL auto-reduce now takes priority over same-position auto-unstuck
  reducers in the same ideal-order batch, preventing two reducer closes from
  stacking on one coin+pside when portfolio exposure enforcement is active.
- Rust TWEL `reduce_overweight` auto-reduce now uses the dynamic currently
  tradable slot count when deciding which positions are overweight, matching
  dynamic WEL sizing instead of the configured `n_positions` floor. If no
  symbols are eligible for new entries but positions remain open, TWEL repair
  falls back to the held-position count so protective reduce-only closes can
  still be emitted.
- `passivbot tool live-config-preflight` now reports
  `balance_hysteresis_snap_pct` and warns when it is invalid or above `0.05`,
  where snapped-balance entry sizing/gating can diverge noticeably from
  raw-balance exposure repair near risk boundaries.
- Entry ladder throttling now uses `entry_cooldown_minutes` as the single
  control: full simultaneous ladders are allowed only when
  `entry_cooldown_minutes = 0.0` and entry retracement is disabled. Any
  positive cooldown, including fractional sub-minute values, stages at most one
  position-adding entry order and blocks further adds until the exact cooldown
  window expires.
- Live and backtest HSL runtime paths now require normalized
  `live.hsl_signal_mode` instead of silently treating a missing raw key as
  `unified`; raw-config diagnostics now report the schema default `coin`.
- The Rust orchestrator JSON boundary now rejects invalid account/risk globals
  such as non-positive raw balance, negative realized-loss limits, and negative
  unstuck allowances before risk gates or order planning can silently skip.
- HSL/risk/unstuck config validation now clamps HSL EMA spans below `1.0`
  during config preparation and rejects malformed HSL, risk, and unstuck
  numeric inputs at the Rust orchestrator JSON boundary before order planning.
- HSL panic close execution now preserves side-local
  `bot.long/short.hsl.panic_close_order_type` in live and backtests; configuring
  one side as `market` no longer market-promotes panic closes for the other side
  when that side is configured as `limit`.
- HSL restart behavior after RED is now controlled by explicit
  `bot.long/short.hsl.restart_after_red_policy` values: `threshold` preserves
  the previous no-restart-threshold behavior, `never` makes any RED terminal,
  and `always` restarts after cooldown while disabling the no-restart safety
  latch for that HSL scope.
- Live TWEL auto-reduce now honors configured
  `risk_twel_enforcer_policy` when building Rust orchestrator payloads instead
  of always falling back to `reduce_overweight`, aligning live behavior with
  backtests for configs using `reduce_portfolio`.
- Live coin-mode HSL now computes slot drawdown from configured
  `n_positions` and current raw balance only, so TWEL or excess allowance no
  longer makes the configured RED drawdown threshold tolerate a larger
  percentage loss. Compared with the previous TWEL-scaled denominator, this
  makes TWEL > 1 coin-HSL stops trigger sooner and TWEL < 1 stops trigger later.
- DEAP optimizer generation evaluations now honor
  `optimize.max_pending_starting_evals_per_cpu`, bounding queued offspring
  evaluations with the same memory-control cap used for starting seeds.
- Pymoo optimizer starting configs now reuse their precomputed seed evaluations
  during initial population setup instead of backtesting the same seed vectors a
  second time.
- Optimizer Pareto storage now checks candidate/front dominance in a single
  pass, reducing per-candidate overhead without changing Pareto semantics.
- Optimizer vector-shape extraction now rejects empty `config.optimize.bounds`
  instead of generating key paths that fail later without matching bounds.
- Compressed `all_results.bin` optimizer history now preserves deleted keys
  during replay, preventing stale fields such as prior candidate errors from
  leaking into later entries.
- Pareto limit filters now fail loudly when a configured limit metric is missing
  instead of silently retaining candidates that cannot be checked.
- Suite optimizer workers now close lazy-slicing master shared-memory attachments when the
  evaluator is cleaned up, avoiding attachment churn across evaluations.
- Pareto pruning now rejects non-finite objective matrices before selecting
  required extremes, preventing NaN values from being retained as best/worst axes.
- Pareto bootstrap now uses non-empty scoring metadata from existing entries
  before rebuilding the front, preventing legacy unscored files from forcing
  minimize-all dominance for scored results.
- Suite scenarios now reject unknown scenario fields before running, catching
  typos such as `coin` instead of silently ignoring them.
- Resumed pymoo optimizer checkpoints now refresh the active problem,
  termination target, and checkpoint callback before continuing, so increasing
  `optimize.iters` on resume takes effect.
- Optimizer overrides now reject unknown `optimize.enable_overrides` names before
  the run starts, and `forward_tp_grid` / `backward_tp_grid` now reorder
  `trailing_grid_v7` close-grid markup bounds as intended.
- Anchored fine-tune optimizer seed conversion now preserves each anchor's
  original id when an earlier starting seed is skipped.
- Optimizer SIGINT handling now safely no-ops before a worker pool exists and
  terminates an active pool without referencing backend-local shutdown state.
- `passivbot optimize --suite-config` now enables suite mode when `--suite` is
  omitted, while explicit `--suite n` still disables suite mode.
- Partial suite override files that define scenarios without `aggregate` now
  preserve the base config's `backtest.aggregate` instead of resetting to mean.
- Optimizer stepped bounds now stay on the configured grid for fractional steps
  such as `0.25`, `0.125`, and `0.0025`, avoiding off-grid candidate values in
  DEAP, pymoo repair, seed conversion, and result hashing.
- Fixed DEAP optimizer candidate recording so duplicate-guard perturbations and
  evaluated starting seeds keep the fitness attached to the actual evaluated
  parameter vector.
- Suite optimizer context preparation now matches suite-runner exchange and
  coin-universe setup, and fails loudly when a scenario cannot be prepared
  instead of silently dropping scenarios or falling back to other exchanges.
- Added optional `optimize.seed`; the default `null` randomizes optimizer
  population and worker RNGs, including replacing pymoo's previous fixed
  default seed, while an integer seed opts into deterministic seeding for
  diagnostics.
- Optimizer Pareto recording now fails loudly on corrupt existing Pareto files
  or invalid objective payloads instead of silently skipping store errors or
  pruning files that were never loaded.
- The `cache` live-event debug profile now enriches existing cache load,
  flush, and warmup-decision events with bounded key/count/source metadata
  without changing default event payloads or console output.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now exposes the
  embedded restart plan's compact timeout-escalation ladder summary in the
  returned report and bundle manifest.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now exposes the
  embedded restart plan's compact warning and issue summaries in the returned
  report and bundle manifest.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now exposes the
  embedded restart plan's compact process-signal safety and execution-policy
  summaries in the returned report and bundle manifest.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now exposes the
  embedded restart plan's planned smoke and follow-up incident-bundle command
  summaries in the returned report and bundle manifest.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now exposes the
  embedded restart plan's smoke/performance section filters in the returned
  report and bundle manifest summary.
- `passivbot tool live-incident-bundle --restart-smoke-plan` now passes
  `--performance-section` filters into the embedded restart plan's planned
  failure-bundle command when performance sections are selected.
- `passivbot tool live-restart-smoke-plan` now supports
  `--performance-section`, passing selected performance-report sections to the
  planned failure incident-bundle command.
- `passivbot tool live-smoke-report --section` now accepts base smoke metadata
  selectors such as `repository`, `monitor`, and `event_window`, so repeated
  smoke loops can request checkout or scan-window evidence directly.
- `passivbot tool live-incident-bundle --performance-report` now supports
  `--performance-section`, so embedded performance evidence can be scoped to
  selected top-level sections while keeping common metadata.
- `passivbot tool live-restart-smoke-plan` now includes
  `live-incident-bundle --performance-report` in its planned failure evidence
  command, so restart-smoke incident bundles capture bounded performance timing
  and readiness summaries by default.
- `passivbot tool live-incident-bundle --performance-report` now embeds an
  opt-in `live-performance-report` artifact and compact manifest summary using
  compatible bundle time, bot/exchange/user, debug-profile, and event-file
  bounds.
- `passivbot tool live-performance-report` now supports `--debug-profile`, so
  performance summaries can be scoped to events enriched by one live-event
  debug profile.
- `passivbot tool live-incident-bundle` now supports `--debug-profile`, passing
  the first-class debug-profile filter through its embedded event, problem-event,
  and time-window reports plus the bundle manifest.
- `passivbot tool live-event-query` now supports `--debug-profile`, a
  first-class filter for events whose structured data has a matching
  `debug_profile` value.
- The `startup` live-event debug profile now enriches existing startup timing
  events with bounded phase, timing, and details-shape metadata without
  changing default event payloads or console output.
- The `state` live-event debug profile now enriches existing state refresh
  timing and progress events with bounded plan, pending-surface, and slowest
  surface metadata without changing default event payloads or console output.
- The `forager` live-event debug profile now enriches existing forager
  selection and feature-unavailable events with bounded count/key-shape
  metadata without changing default event payloads or console output.
- `passivbot tool live-smoke-report --brief` now includes allowlisted
  `latest_data` for risk-event groups, exposing compact state such as HSL mode
  transitions and unstuck over-budget summaries while still excluding raw
  balances, drawdown internals, and per-side allowance details.
- `passivbot tool live-smoke-report --summary` and `--brief` now include
  bounded dropped-unparsed log samples when `--log-window-unparsed-policy drop`
  suppresses contextless hard or attention-looking log lines.
- `passivbot tool live-event-query` now emits a warning issue when filtered
  current-only queries skip rotated monitor event segments, making empty
  incident queries less likely to be mistaken for complete history.
- `passivbot tool live-incident-bundle` now includes the compact time-window
  query summary in `manifest.json`, so archived bundles expose matched-event,
  truncation, and scan-bound evidence without opening `time_window_report.json`.
- `passivbot tool live-incident-bundle` now includes the bundle-level result
  verdict in `manifest.json`, so archived incident bundles expose total
  `ok`/`hard_failures` without opening the command output.
- `passivbot tool live-incident-bundle` now includes top-level smoke verdict
  fields in `manifest.json`, making the bundle manifest self-contained for
  `ok`, `attention`, and smoke failure/count triage.
- `passivbot tool live-incident-bundle` now includes bounded repository and
  monitor smoke summaries in the returned report and `manifest.json`, making
  checkout cleanliness and monitor event-count context visible in bundle-level
  triage.
- `passivbot tool live-incident-bundle` now includes bounded text-log and
  event-window smoke summaries in `manifest.json`, so bundle manifests can
  explain log-sourced hard or attention verdicts without opening the embedded
  full smoke report.
- `passivbot tool live-incident-bundle` now includes the bounded process
  smoke summary in the returned report and manifest, making missing,
  duplicate, or unexpected live-bot process evidence visible without opening
  the embedded full smoke report.
- `passivbot tool live-incident-bundle` now includes smoke verdict source
  breakdowns and recovered problem-event counts in the returned report and
  manifest, making red or attention incidents easier to attribute without
  opening the embedded full smoke report.
- `passivbot tool live-incident-bundle` now includes the bounded
  `problem_events` smoke summary in the returned report and manifest, including
  hard and non-hard problem-event type histograms for quicker incident triage.
- `passivbot tool live-smoke-report --summary` and `--brief` now split
  structured problem-event type counts into hard and non-hard histograms,
  making mixed smoke attention easier to triage without opening grouped event
  rows.
- `passivbot tool live-smoke-report --summary` and `--brief` now include
  `problem_events.non_hard`, making non-fatal structured attention easier to
  distinguish from hard problem events at a glance.
- `passivbot tool live-smoke-report --section` and
  `passivbot tool live-incident-bundle --smoke-section` now accept the brief
  smoke-summary names such as `fill_refresh`, `hsl_replay`, and
  `remote_calls` as aliases for their embedded full-report sections, reducing
  CLI friction when moving between brief smoke output and incident bundles.
- `passivbot tool live-incident-bundle` now includes bounded data-plane smoke
  summaries for remote calls, account-critical remote calls, fill refresh,
  startup timings, and HSL replay in the returned report and manifest, making
  common exchange/data-readiness and startup-latency evidence visible without
  opening the full embedded smoke report.
- `passivbot tool live-incident-bundle` now includes bounded operational smoke
  summaries for exchange config refresh, staged readiness, event-pipeline
  health, and shutdown events in the returned report and manifest, so common
  live-smoke attention sources are visible without opening the full embedded
  smoke report.
- `passivbot tool live-incident-bundle` now includes the bounded EMA-readiness
  smoke summary in the returned report and manifest, so incident bundles expose
  current EMA unavailable reason counts without requiring operators to open the
  full embedded smoke report.
- `passivbot tool live-incident-bundle` now includes a bounded risk-event
  smoke summary in the returned report and manifest, so incident bundles expose
  HSL RED/cooldown/raw-red context without requiring operators to open the full
  embedded smoke report.
- `passivbot tool live-smoke-report --brief` now includes bounded risk
  attention groups, prioritizing HSL RED/cooldown/raw-red and risk panic-mode
  context even when newer routine risk status events would otherwise bury them
  in latest-event ordering.
- `passivbot tool live-smoke-report --brief` now includes bounded latest risk
  event samples, making HSL RED/cooldown/mode-change smoke context visible
  without dumping verbose risk event payloads.
- `passivbot tool live-smoke-report --brief` now includes bounded hard and
  attention log-match samples, so hard smoke verdicts can be attributed without
  rerunning the larger summary report.
- `passivbot tool live-smoke-report --brief` now adds dense and required HSL
  replay max remaining-work aggregates alongside the existing primary
  remaining-work max fields.
- `passivbot tool live-smoke-report --brief` now distinguishes dense and
  required HSL replay remaining-work estimates in active replay samples, so
  dense pair replay is not hidden when required-position replay is already
  complete.
- `passivbot tool live-smoke-report --brief` now includes bounded active HSL
  replay samples with bot, stage, symbol, elapsed age, progress, and remaining
  work estimates, making long-running HSL startup replay easier to attribute.
- `passivbot tool live-smoke-report` now summarizes staged-readiness reason
  codes, defer reasons, contexts, and bounded max timing fields in concise
  output, making current-epoch planning delays easier to attribute without a
  separate event query.
- `passivbot tool live-smoke-report --brief` now includes bounded slowest
  remote-call latency samples, making slow exchange/account/candle surfaces
  visible without dumping the full summary.
- `passivbot tool live-smoke-report` now includes bounded EMA-readiness symbol
  samples by unavailable reason in summary and brief output, so operators can
  see which symbols are affected without a separate event query.
- `passivbot tool live-smoke-report` now summarizes HSL raw-red-pending
  targets in concise risk output, including bounded red-proximity and
  EMA-gap-to-red percentages without exposing raw drawdown internals.
- `passivbot tool live-smoke-report` now names staged-readiness missing and
  invalid surfaces in summary and brief output, so issues like stale
  `completed_candles` are visible without a separate event query.
- `passivbot tool live-smoke-report` now lists active HSL cooldown targets in
  concise risk summaries, so RED cooldown symbols are visible even when they do
  not have current drawdown-distance metrics.
- `passivbot tool live-smoke-report` now includes normalized HSL red-proximity
  percentages in concise closest-to-red risk summaries, making current HSL
  proximity visible without exposing raw drawdown-space thresholds.
- `passivbot tool live-smoke-report` now summarizes failed remote-call
  reasons, surfaces, kinds, and error types in concise smoke output, making
  transient exchange failures easier to identify without a separate event
  query.
- `passivbot tool live-smoke-report` now attaches the timestamped log context
  line to unparseable traceback/error matches, making hard text-log matches
  easier to attribute without changing smoke verdict policy.
- `passivbot tool live-smoke-report` now summarizes EMA-readiness unavailable
  reasons and candidate error types in concise smoke output, making
  `cache_only_fetch_failed` vs `never_fetched_cache_only` visible without a
  separate event query.
- Forager monitor readiness is now scoped to the exact symbols evaluated by the
  latest EMA bundle, optional cache-only metric gaps are classified as
  candidate health failures, and open-tail projection preserves log-range
  inputs required by held or explicitly normal strategies. Completed-candle
  forager ranking metrics now have a distinct Rust input channel so coincident
  strategy and ranking spans cannot reuse projected values for coin selection.
- `passivbot tool live-smoke-report` now reports timestamp/nonce
  `cycle.degraded` events recovered by a subsequent successful
  `exchange.time_sync` event as recovered problem events instead of hard smoke
  failures, while unrecovered timestamp/nonce errors remain hard.
- Coin-mode HSL startup reconstruction now limits candle-price replay and
  strict historical UPnL validation to current-position symbols and historical
  panic-close cooldown symbols. Flat non-panic historical fill symbols no
  longer block startup or force broad candle replay, while runtime coin-HSL
  still evaluates them from fill history after startup.
- `passivbot tool live-smoke-report` now labels active HSL startup replay
  groups as stale or long-running when existing monitor events show no recent
  progress or prolonged replay elapsed time, making startup-blocked bots easier
  to spot without changing trading behavior.
- Low-balance exposure-increasing create skips now appear through the
  structured live event console as `execution.create_skipped` summaries, and
  the legacy `[balance] too low` line is only a fallback when that path is
  unavailable or disabled.
- Legacy balance and position-change console lines are now suppressed when the
  structured live event console path is active, leaving `balance.changed` and
  `position.changed` projections as the default operator output while
  preserving legacy lines as fallback.
- Flat coin-mode HSL cooldown finalizations now emit their
  `hsl.red_triggered` event as informational instead of critical when no
  exchange close was needed, so smoke reports no longer treat cooldown-only
  flat symbols as hard panic failures.
- Legacy order-wave complete/settled console lines are now suppressed when the
  structured live event console path is active, leaving structured execution
  wave summaries as the default operator output while preserving the legacy
  lines as fallback.
- Legacy unstuck status/selection console lines are now suppressed when the
  structured live event console path is active, leaving the structured
  `[unstuck]` projection as the default operator output while preserving a
  fallback if that path is disabled.
- Legacy startup timing console lines are now suppressed when the structured
  live event console path is active, leaving the structured `[boot]` projection
  as the default operator output while preserving a fallback if that path is
  disabled.
- Startup timing events now appear in the live event console projection by
  default, making account-ready, candle-ready, HSL-ready, market-ready, and
  startup-ready phase durations visible from the structured event stream.
- Live event console summaries are now enabled by default for `passivbot live`;
  set `logging.live_event_console=false` or `PASSIVBOT_LIVE_EVENT_CONSOLE=0`
  to opt out while legacy console logs are still being migrated.
- Improved opt-in live event console summaries for trailing and unstuck
  positions, including threshold/retracement prices and unstuck selection
  details from existing structured events.
- Added `--level` filtering to `passivbot tool live-event-query`, so operator
  event, timeline, trace-summary, order-trace, and cycle-trace reports can be
  scoped by live-event severity.
- `passivbot tool live-smoke-report --brief` now includes the structured event
  window `enabled` flag, matching the full report and making unwindowed brief
  smoke output explicit.
- `passivbot tool live-smoke-report --brief` now includes bounded text-log
  window counters, making it clear when hard/attention log counts came from a
  time-windowed scan and how many log lines were skipped.
- `passivbot tool live-smoke-report --summary` and `--brief` now expose
  existing startup timing evidence, making slow restart phases visible in the
  concise smoke-loop projections.
- Added opt-in `passivbot tool live-event-query --event-tail-lines` to bound
  monitor event parsing for repeated recent-window queries while leaving full
  event validation as the default.
- Added opt-in `passivbot tool live-smoke-report --event-tail-lines` to bound
  monitor event parsing for repeated recent-window smoke checks while leaving
  full monitor-event validation as the default.
- Added structured `hsl.raw_red_pending` diagnostics when HSL raw drawdown is
  already beyond red but EMA-confirmed drawdown has not crossed yet, helping
  operators spot pending RED risk without changing trading behavior.
- HSL history replay now ignores historical `close_panic_*` markers that cannot
  be confirmed as RED by reconstructed HSL metrics at the marker timestamp, so
  an old or erroneous panic fill does not recreate RED cooldown or supervisor
  state on restart.
- `passivbot tool live-smoke-report --processes` now performs a read-only
  local config check for running/expected live commands and reports a hard
  smoke failure when account-level HSL (`unified`/`pside`) is combined with an
  active balance override.
- `passivbot tool live-config-preflight` now flags active
  `balance_override` plus account-level HSL signal modes (`unified`/`pside`)
  before live startup, including an optional `--balance-override` argument for
  preflighting runs that will pass `-bo`.
- Added a live HSL safety guard: `hsl_signal_mode=unified` or `pside` now
  fails before account-level equity replay when `balance_override` is active,
  preventing synthetic historical peaks from triggering false RED panic orders
  until an explicit HSL baseline/checkpoint exists. HSL history replay also
  zero-anchors realized-PnL timeline fields at the configured lookback boundary
  so replayed peaks match the live runtime lookback contract.
- Added root-level `passivbot -V` and `passivbot --version` output.
- Added `hsl_replay_health` summaries to
  `passivbot tool live-smoke-report`, so smoke reports show active,
  completed, and failed HSL startup replay state from existing
  `hsl.replay.*` events.
- Added structured `risk.entry_cooldown_delta_anchored` events when live
  entry cooldown is anchored from an exchange position-size increase, including
  cases where the legacy text warning is throttled.
- Updated the canonical v8 trailing-martingale default config profile, including
  the 41-coin universe, per-coin HSL signal mode, refreshed optimizer
  scoring/limits/bounds, `bot.long.risk.n_positions = 5`, and
  `entry.ema_gate_mode = "all"` for default-reliant configs.
- Added `--tag` filtering to `passivbot tool live-event-query`, so operator
  event, timeline, trace-summary, order-trace, and cycle-trace reports can be
  scoped by structured live-event tags.
- Corrected `passivbot tool live-performance-report` `snapshot_to_rust`
  correlation so planning snapshot epochs are no longer mistaken for live
  cycle IDs; legacy snapshot events now use the latest preceding snapshot in
  the same bot/restart scope and expose match counters.
- Added `operation_durations` summaries to
  `passivbot tool live-performance-report`, collating existing startup, cycle,
  state-refresh, remote-call, HSL replay, cache, decision-boundary,
  input-staleness, execution, and shutdown timing groups into one bounded
  trading-impact-ranked table without adding new live events or exchange calls.
- Added `forager_ema_readiness` summaries to
  `passivbot tool live-performance-report`, deriving bounded forager selection,
  forager feature-unavailable, EMA unavailable, and EMA fallback evidence from
  existing events without exposing raw EMA errors, top-score payloads, account
  values, or cache paths.
- Added `cache_warmup` summaries to
  `passivbot tool live-performance-report`, deriving bounded warm-cache reuse,
  cold-path, candle cache load, and candle cache flush evidence from existing
  cache events without exposing raw cache paths or payloads.
- Added `hsl_replay_profile` summaries to
  `passivbot tool live-performance-report`, deriving bounded HSL replay
  work/progress and startup-blocking timing context from existing
  `hsl.replay.*` events.
- Added snapshot surface and market-snapshot age breakdowns to
  `passivbot tool live-performance-report`, using bounded metadata from
  existing `snapshot.built` events without exposing market prices or raw
  payloads.
- Added `execution_timing` summaries to
  `passivbot tool live-performance-report`, deriving bounded exchange-action
  latency groups from existing order-wave, create/cancel, and confirmation
  events without exposing raw order payloads.
- Added `shutdown_latency` summaries to
  `passivbot tool live-performance-report`, projecting existing lifecycle
  shutdown events into per-stage and total shutdown timing groups without
  copying shutdown error text.
- Added `resource_pressure` summaries to
  `passivbot tool live-performance-report`, projecting whitelisted
  `health.summary` process and event-pipeline fields with count, min, mean,
  median, p95, max, and latest values without raw account or financial
  payloads.
- Added `exchange_config_refresh` summaries and elapsed timing groups to
  `passivbot tool live-performance-report`, projecting existing structured
  refresh success/failure events without copying raw exchange error text or
  making new exchange calls.
- Exchange-config refresh summaries in `live-smoke-report` and
  `live-performance-report` now distinguish historical failures from each
  bot's latest status and count recovered bots after a later successful refresh.
- Improved cold `passivbot backtest` materialization by batching legacy OHLCV
  imports by month, vectorizing chunk writes, staging HLCV cache writes with
  rollback on publish failure, and honoring Ctrl+C between expensive
  materializer/cache stages.
- Added explicit hard-failure and attention source breakdowns to
  `passivbot tool live-smoke-report`, so red or attention smokes identify
  monitor parse errors, invalid rows, structured events, log matches, and
  process liveness contributions without changing verdict logic.
- Added risk/HSL log-match counters to `passivbot tool live-smoke-report`, so
  CRITICAL risk-state log lines can be distinguished from non-risk hard log
  matches without changing smoke verdict logic.
- Added event-pipeline health summaries to
  `passivbot tool live-smoke-report`, projecting existing `health.summary`
  queue/drop/sink-error counters into full, summary, and brief reports.
- Added bounded staged-readiness health summaries to
  `passivbot tool live-smoke-report`, projecting existing staged
  `cycle.degraded` events into latest missing/invalid surface counts and
  completed-candle mismatch evidence.
- Added bounded EMA readiness health summaries to
  `passivbot tool live-smoke-report`, projecting existing `ema.unavailable`
  events into latest candidate/unavailable counts plus reason/error evidence.
- Added no-extra-call `exchange_surface_health` summaries to
  `passivbot tool ticker-endpoint-probe`, interpreting already-recorded endpoint
  outcomes into exchange/user-level notes for open-orders fallback, time-sync
  support, fill-history pagination, and OHLCV tail shape.
- Added no-extra-call endpoint latency health summaries to
  `passivbot tool ticker-endpoint-probe`, derived from existing probe outcomes.
- Added opt-in bounded fill-history pagination sampling to
  `passivbot tool ticker-endpoint-probe` via `--fill-history-pages`, while
  preserving the default single-call `fetch_my_trades(first symbol)` behavior.
- Added no-extra-call rate-limit pressure estimates to
  `passivbot tool ticker-endpoint-probe`, derived from existing probe outcomes
  and CCXT rate-limit metadata.
- Added no-extra-call fill-history sample health summaries to
  `passivbot tool ticker-endpoint-probe`, derived from the existing
  `fetch_my_trades(first symbol)` probe result without raw trade/order ids.
- Added no-extra-call 1m candle freshness health summaries to
  `passivbot tool ticker-endpoint-probe`, derived from the existing OHLCV tail
  probe results.
- Added read-only `fetch_time` clock-skew health summaries to
  `passivbot tool ticker-endpoint-probe`, with `--skip-time-sync` for operators
  who want to omit the extra time-sync call.
- Added process-signal safety guidance to
  `passivbot tool live-restart-smoke-plan`, warning future restart automation
  away from broad `pkill -f`/`pgrep -f` live-bot matches and toward exact tmux
  panes or exact canonical process rows.
- Added report-only startup phase budget projections to
  `passivbot tool live-smoke-report`, comparing latest startup timings with
  prior local p95 baselines from existing monitor events.
- Hardened forager active-symbol EMA readiness by allowing required
  qv/log-range ranking features to carry forward bounded cached real-candle EMA
  values for active/normal symbols during fill handoff.
- Added optional `--compare` diff reporting to
  `passivbot tool live-config-preflight` for local, read-only HSL, universe,
  forager, identity, and cache-setting changes between two configs.
- Added config-only cache readiness/root-hint reporting to
  `passivbot tool live-config-preflight`, including derived compare-mode
  readiness deltas without scanning cache artifacts or enforcing startup
  policy.
- Added report-only warm-cache readiness evidence to
  `passivbot tool cache-integrity-doctor`, derived from already-scanned local
  candle, fill, and HSL/risk cache metadata.
- Added interior/boundary candle gap summaries to
  `passivbot tool cache-integrity-doctor` and its report-only warm-cache
  readiness projection, clarifying leading missing rows and trailing shortfall
  gaps without repair or startup enforcement.
- Added fill-cache and HSL/risk-state metadata summaries to
  `passivbot tool cache-integrity-doctor`, including local fill
  `pnl_contract` compatibility counts and coverage timestamps.
- Hardened recent live-ops tooling and debug-profile diagnostics by redacting
  shareable path fields consistently, keeping Rust debug sample construction
  best-effort, and scoping EMA debug enrichment to the `ema` profile only.
- Hardened read-only live-ops tools so `live-config-preflight` and
  `hsl-startup-preview` resolve both grouped and flat bot-side config keys, and
  HSL preview output keeps allowlisted event details scalar-only.
- Added a `fills` live-event debug profile with bounded fill refresh and fill
  ingestion shape metadata, without raw fill/source payloads or default console
  changes.
- Added an `hsl` live-event debug profile with bounded HSL event key, metric
  key, and latch/cooldown state-shape metadata, without changing default HSL
  events, console output, or trading behavior.
- Added an `execution` live-event debug profile with bounded order-wave,
  order-write, and confirmation key-shape metadata, without raw order payloads
  or default console changes.
- Added v2 candle coverage windows and suspicious interior gap samples to
  `passivbot tool cache-integrity-doctor`, derived only from local `.valid.npy`
  cache artifacts.
- Added `passivbot tool live-restart-smoke-plan` for read-only dry-run restart
  smoke planning from a tmuxp-style supervisor config, with explicit
  non-execution metadata and rejected execution flags.
- Added `passivbot tool hsl-startup-preview` for read-only offline HSL
  startup previews from config and local monitor events, with explicit
  unavailable fields for current drawdown and panic-order prediction.
- Added `logging.live_event_debug_profiles` and
  `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` for opt-in structured live-event
  enrichment, starting with bounded Rust orchestrator input/output samples.
- Added the `ema` live-event debug profile, which enriches structured
  `ema.unavailable` events with bounded parsed readiness detail without
  changing console output or trading behavior.
- Added the `remote_calls` live-event debug profile, which enriches structured
  remote-call events with bounded payload-shape and correlation details without
  adding raw payloads or console output.
- Added the `candles` live-event debug profile enrichment for candle tail and
  disk-coverage events, exposing bounded key-shape and timing/counter details
  without raw candle payloads or console output.
- Added `passivbot tool live-config-preflight` for read-only offline summaries
  of risk-relevant live config facts before startup.
- Added shutdown-event summaries to `passivbot tool live-smoke-report`, including
  `bot.stopping`, shutdown stage, and `bot.stopped` events in full, summary, and
  brief reports.
- Added structured `exchange.time_sync` live events for CCXT timestamp/nonce
  recovery diagnostics without changing recovery behavior or console volume.
- Added structured `fills.refresh_summary` startup cache-ready events for fill
  history cache load diagnostics without adding console noise.
- Added `passivbot tool live-smoke-report --brief` for top-level VPS smoke
  counters without event groups or log match details.
- Added periodic console status lines for coin-mode HSL positions, including
  distance to RED, drawdown, slot budget, realized PnL peak, and unrealized PnL.
- Added `passivbot tool live-smoke-report --summary` for concise smoke evidence
  that keeps high-signal process, log, problem-event, risk, repository, and
  remote-call health fields without emitting the full verbose report.
- Added account-critical remote-call health summaries to
  `passivbot tool live-smoke-report`, isolating balance, position, and
  open-order endpoint health from broader candle/fill traffic.
- Added top-level success, failure, and throttle totals to remote-call health
  summaries in `passivbot tool live-smoke-report`.
- Added remote-call health rollups to `passivbot tool live-smoke-report`,
  grouping successes, failures, throttles, latency, and affected symbols by
  bot/component/kind/surface.
- Added remote-call elapsed-time summaries to `passivbot tool live-smoke-report`
  so slow exchange/API calls can be inspected even when they eventually succeed.
- Added repository branch/head metadata to `passivbot tool live-smoke-report`
  so VPS smoke evidence records the deployed code revision without counting
  local untracked artifacts as dirty.
- Added grouped problem-event summaries to `passivbot tool live-smoke-report`
  so repeated structured degradation can be inspected by bot, event type,
  reason, and hard/non-hard status without reading every event sample.
- Changed `passivbot tool live-smoke-report --log-window-unparsed-policy drop`
  to skip contextless unparsed log lines inside time-windowed scans, avoiding
  stale traceback matches when the tail starts in the middle of an old
  traceback.
- Added structured `state.refresh_timing` and `state.refresh_progress` live
  events for staged authoritative refresh timing/progress diagnostics, with
  bounded smoke-report context for slow pending refresh surfaces.
- Added live-event trace summary and order-trace sections to
  `passivbot tool live-incident-bundle` event reports, with
  `--no-trace-report` for compact bundles.
- Added `passivbot tool cache-integrity-doctor` for read-only local cache smoke
  reports covering root presence, file counts/sizes, and empty or corrupt
  JSON/NDJSON/NPY artifacts.
- Improved `passivbot backtest --help-all` descriptions for high-impact
  runtime/config overrides, including plot groups, suite aggregation,
  HLCV dataset replay modes, HSL modes, and TWEL/WEL policy flags.
- Added `passivbot tool live-event-query --cycle-trace` for offline cycle
  reconstruction grouped by cycle id, including bounded timeline samples,
  aggregate event summaries, and nested order traces.
- Added `passivbot tool live-event-query --order-trace` for offline order
  lifecycle reconstruction grouped by order wave and action ids.
- Added startup phase timing baselines to `passivbot tool live-smoke-report`,
  showing latest phase timings with rolling median/p95 context from local
  monitor events.
- Added `passivbot tool live-event-query --trace-summary` for compact aggregate
  summaries of matched live event traces, including event types, statuses,
  reason codes, ids, symbols, and order-wave coverage.
- Improved event-projected live console summaries for cycle/order execution
  events, including compact wave, order, confirmation, and Rust planning
  details without increasing console event volume.
- Added structured `health.summary` live events for periodic health and
  resource summaries without adding console noise.
- Added process/system/event-pipeline resource-pressure fields to structured
  `health.summary` events, including load average, open file count, queue depth,
  event drops, and sink error counters when available.
- Added `passivbot tool live-incident-bundle` for collecting local monitor
  events, smoke summaries, redacted log excerpts, monitor snapshots, config
  hashes, runtime metadata, and bounded event segments into a tarball.
- Added `passivbot tool live-event-query` filters and timeline ids for bot id,
  snapshot id, plan id, action id, and remote call group id so operators can
  trace non-cycle live event scopes.
- Added `passivbot tool live-smoke-report` for local smoke-test summaries from
  monitor event NDJSON and recent text logs.
- Added `passivbot tool live-event-query` filters for order wave id, remote
  call id, symbol, position side, reason code, and status, plus optional
  timeline rows for matched structured live events.
- Added structured `cache.load.completed` live events for candle disk-cache load
  summaries.
- Throttled repeated `cache.load.completed` live events per symbol/timeframe and
  added `suppressed_count` so warmup/HSL replay does not flood monitor storage.
- Added throttled structured `cache.flush.completed` live events for candle
  disk-cache write summaries.
- Added structured `risk.mode_changed` live events for HSL runtime forced-mode
  changes such as panic, graceful-stop, tp-only, and clear transitions.
- Added off-console structured `unstuck.status` and `unstuck.selection` live
  events alongside existing `[unstuck]` logs, and included them in
  `live-smoke-report` risk-event summaries.
- Added structured `hsl.red_triggered` live events for HSL stop finalization
  paths that reconstruct or finalize RED state without a fresh threshold-crossing
  sample.
- Added structured `bot.startup_timing` live events for startup phase timing
  diagnostics.
- Added structured `cache.warmup_decision` live events for candle warmup cache
  reuse/cold-path summaries.
- Added `passivbot tool live-event-query --event-type`/`--kind` filters for
  inspecting specific structured live events without grepping monitor files.
- Added structured `candle.tail_projected` live events for open-tail EMA
  projection decisions, preserving per-symbol candle-tail context without
  default console noise.
- Added structured `candle.coverage_checked` live events for required candle
  disk-coverage audits, including bounded missing-span summaries.
- Added structured `fills.refresh_summary` live events for fill/PnL refresh
  timing, coverage, retry, and failure summaries without exposing raw fill ids.
- Reduced default console/file noise for candidate-only forager EMA and
  open-tail projection diagnostics; detailed per-symbol internals remain in
  structured/debug events while active-symbol failures still fail loudly.
- Changed `passivbot tool live-event-query` directory scans to inspect
  `current.ndjson` segments by default; use `--include-rotated` for full
  rotated history validation.
- Added `live.limit_order_create_max_market_dist_pct` with a default of `0.8`
  so live skips limit-order creations far outside fresh market price bands
  instead of repeatedly submitting exchange-invalid deep orders.
- Added `passivbot tool live-event-query` to validate monitor event NDJSON and
  reconstruct one live event chain by `cycle_id`.
- Added staged live shutdown progress events/logs, made candle fetch-lock waits
  abort promptly once shutdown is requested, and shortened the post-cancel
  background execution-loop grace from 5s to 1s.
- Fixed Bitget UTA / Elite close-order placement by omitting the one-way-only
  `reduceOnly` flag from hedge-mode v3 orders that already send `posSide`.
- Fixed Bitget UTA / Elite open-order normalization so hedge-mode close orders
  keep their exchange-reported `side` instead of being misread as entries.
- Fixed Rust extension freshness detection for Python abi3 builds so
  `passivbot_rust.abi3.so` artifacts are discovered, source-stamped, and reused
  instead of triggering repeated rebuilds followed by stale-extension failures.
- Fixed v8 backtests/optimizer runs so candidates with depleted raw wallet
  balance terminate through the normal liquidation path and emit incomplete
  `backtest_completion_ratio` metrics instead of crashing coin-HSL slot-budget
  evaluation.
- Implemented the v8 TWEL policy contract: TWEL entry gating is now controlled
  separately from TWEL auto-reduce, entry gating uses the capped thresholded
  portfolio cap, and TWEL auto-reduce supports `reduce_overweight` and
  `reduce_portfolio` policies while remaining subject to the realized-loss gate.
  Manual and panic exposure now counts toward same-side TWEL measurement while
  remaining excluded from TWEL auto-reduce candidate selection.
- Fixed live forager EMA readiness for flat approved-universe and transient
  forager-selected symbols: missing close/required EMA data now marks the flat
  symbol nontradable for that planning cycle instead of restarting the execution
  loop, while explicit normal symbols and held/open-order symbols remain
  fail-loud.
- Fixed two live restart/minute-boundary edge cases: required 1m log-range EMA
  loads now retry bounded open-tail projection when a fresh one-candle tail
  appears after projection precompute without clearing candidate-only forager
  qv/log-range maps to `None`, and coin-HSL balance/equity replay now emits
  explicit zero coin-UPnL for replay-proven flat symbols with realized history.
- Hardened live forager promotion readiness: newly selected normal forager
  symbols now get targeted candle warmup before normal order planning, and
  missing required forager ranking EMAs still fail loudly for active/normal
  symbols instead of silently making them nontradable.
- Fixed live HSL restart replay so historical drawdown threshold crossings no
  longer create a fresh RED panic after recovery; startup now panics only when
  current drawdown is RED or when exchange-derived panic/cooldown markers
  reconstruct an active prior HSL stop.
- Hardened v8 live startup after overnight VPS probes: deterministic coin-HSL
  validation errors stop as terminal startup failures instead of restart loops,
  stale candle fetch locks now include owner diagnostics and local hold-timeout
  warnings, partial fill-history gap repairs persist correctly, true secondary
  forager symbols with unavailable required EMA inputs are marked nontradable
  until fresh data is available, active/normal EMA inputs remain fail-loud, and
  Gateio history replay uses single-fetch concurrency by default.
- Hardened v8 live restart behavior by clearing successfully retried empty
  fill-history gaps, failing loudly on ambiguous coin-HSL carry-in replay,
  avoiding duplicate coin-HSL startup replay, keeping active/normal forager
  EMA inputs fail-loud, and summarizing close-EMA fallbacks.
- Reduced v8 live startup noise and CPU pressure by summarizing flat forager
  candidate EMA readiness failures, optimizing coin-HSL restart replay, adding
  coin-HSL replay progress logs, and suppressing known websocket timeout futures.
- Reduced live Kucoin fill-history churn by keeping old synthetic PnL records out of
  routine/latest repair windows, and made flat forager candidates with unavailable
  required EMA volatility inputs non-tradable for that planning cycle instead of
  restarting the execution loop.
- Fixed v8 live fill-history startup/restart behavior so unproven
  `pnls_max_lookback_days` coverage triggers a blocking lookback refresh and
  retry/defer instead of sending neutral PnL inputs or repeatedly restarting the
  execution loop.
- Reduced repeated live fill-history repair work when coverage remains blocked
  by the same unresolved gap, and made live execution-loop retry delays respond
  promptly to shutdown.
- Hardened live candle EMA inputs by filtering invalid OHLCV rows at ingestion and
  preventing a leading non-finite candle sample from poisoning log-range EMAs.
- Added `strategy_eq_underwater_pct_mean` and `strategy_eq_underwater_pct_median`
  backtest metrics for average and median daily-worst strategy-equity drawdown.
- Added `bot.<side>.strategy.trailing_martingale.entry.ema_gate_mode` with
  `disabled`, `all`, `initial`, and `reentry` modes for controlling which entry
  orders are EMA-gated. The fixed enum is not optimized; one-way flat
  long-vs-short tie-breaking still requires EMA bands even when emitted entry
  EMA gating is disabled.
- Added `bot.<side>.unstuck.ema_gating_enabled` as a fixed auto-unstuck toggle.
  When false, auto-unstuck skips the EMA trigger while keeping loss allowance,
  exposure threshold, and sizing checks intact.
- Changed the v8 default backtest candle interval to 1 minute and added
  `bot.<side>.risk.we_excess_allowance_mode`. V8 defaults to bounded excess
  allowance; migrated v7 trailing-grid configs also force v7-absent entry
  cooldowns to `0.0`, warn when v7 raw excess allowance would be clamped, and
  report inserted v8 defaults for review.
- Coin overrides can now set `bot.<side>.unstuck.loss_allowance_pct`. When an
  overridden coin+side is selected for auto-unstucking, Rust uses that percentage
  in the existing account-wide allowance formula while preserving the one-position
  global unstuck selection behavior.
- Added Bitget UTA / Elite copy-trading account support with v3 API routing for
  balance, orders, and fill-event history while keeping classic Bitget accounts
  on the existing v2/mix paths.
- Fixed Hyperliquid balance on unified/portfolio-margin accounts. The unified
  `total[USDC]` payload is the cross-margined account *equity* (it already
  includes perp unrealized PnL for core and every HIP-3 dex), but Passivbot was
  using it directly as `balance`, then recomputing `equity = balance + uPNL` —
  double-counting unrealized PnL. Balance now subtracts the exchange-reported
  uPNL across all perp positions (core + HIP-3), matching the non-unified path
  and the Passivbot `balance = equity - uPNL` contract. Missing/invalid uPNL on
  a counted position hard-fails rather than defaulting.
- Added `backtest.market_settings` overrides for historical/rebranded market metadata, including
  exchange-specific overrides before Rust backtests receive market parameters; backtests now warn
  and default missing `c_mult` to 1.0 instead of hard-failing.
- Fixed live `[pos]` logging so short position size increases are labeled as
  `added` and short size decreases as `reduced`, matching exposure magnitude
  instead of signed numeric ordering.
- Fixed live ignored-coin handling so ignored symbols are sent to the Rust
  orchestrator as `graceful_stop`, preventing new initial entries after a
  previously open ignored position becomes fully flat.
- Added a v8 TWEL/total exposure enforcer policy-contract plan for the future
  portfolio governor redesign, based on the known v7 threshold/refill behavior
  but without changing current v8 runtime behavior.
- Hardened v8 live-safety review follow-ups: ambiguous order-create responses are
  remembered before retry, protective panic bypasses stale normal-mode filters
  while requiring fresh account-critical balance/position/order state, PnL risk
  gates require explicit fill-history coverage including coin HSL, Bitget keeps
  multiple fills per order, and OKX net-mode accounts fail loudly.
- Added optimizer polish bounds via `--polish-pct`/`--polish-bounds-pct`, which narrows
  existing optimize bounds around the current config values while preserving positive steps.
  `--polish-bounds-mode` can now choose the default clamped behavior, allow tunable
  polished bounds to escape the original bounds, or expand fixed bounds too.
- Fixed Pareto-member replay drift when a reusable HLCV cache contains more warmup than
  the selected config's own indicators require. Backtests now preserve the optimizer's
  bounds-aware warmup window and requested-start trade floor, so replaying an optimizer
  Pareto JSON matches its recorded metrics when the same dataset is used.
- Hardened v8 audit follow-ups: live HSL cooldowns now reset from flat-confirmed
  panic fills, suite metric medians are real/fail-loud, malformed foreign
  client-order ids decode to `unknown`, partial OHLCV fetches no longer bless
  stale gaps/chunk rows, live realized-loss gate zero values and fee metadata are
  preserved, trailing-anchor-unavailable symbols keep existing orders untouched,
  and unsupported live fill-event exchanges now fail with an explicit startup error.
- Fixed v8 backtests so delisted open positions are realized at the last valid candle, and
  next-candle close-ladder peeking expands recursive close grids when any ladder rung can fill.
- Fixed live v8 trailing state handling so missing fill anchors or candle failures preserve
  the last known trailing bundle and mark affected symbols non-tradable for the planning cycle.
- Tightened optimizer fail-loud behavior: non-finite scenario metrics now invalidate the
  candidate instead of scoring as zero, median limit stats are emitted, malformed
  `optimize.limits` fail config loading, and fatal optimizer exceptions exit non-zero.
- Fixed fill-event attribution edge cases: Bitget hedge-mode bare close fills now map
  buy closes to shorts and sell closes to longs, fill normalization no longer falls back
  to raw client ids or long-side defaults on helper import failures, and Bybit refreshes
  avoid double-counting overlapping coalesced execution ids.
- Tightened OHLCV fetch failure handling so exhausted CCXT retries now fail instead of
  masquerading as an empty page that can persist a false trailing-unavailable gap.
- Fixed live HSL no-restart latching to preserve the persistent stop-episode peak
  across auto-restart cooldowns and restart history replay, matching v8 backtest behavior.
- Tightened Bitget fill normalization so ambiguous side/position-side payloads fail
  loudly instead of defaulting fills to the long side.
- Backtests now use exchange-derived per-coin maker/taker fees by default, while
  `backtest.maker_fee_override` and `backtest.taker_fee_override` remain explicit
  global overrides and are exposed as visible backtest/optimize CLI flags.
- Suite backtests and optimizer suites now reject asymmetric per-side approved/ignored
  coin lists instead of silently converting them to a long/short union.
- Live execution now skips both order cancellations and creations while the raw wallet
  balance is below the configured threshold, avoiding entry-grid cancellation from a
  transient near-zero balance snapshot.
- Live order reconciliation now blocks a symbol and requests a full account refresh
  when an open order snapshot is malformed, instead of dropping the bad actual order
  and creating a duplicate.
- Rust live/backtest orchestration now rejects missing or invalid exchange metadata
  and requires the realized-loss gate parameter instead of accepting neutral serde/PyO3
  defaults.
- Exchange configuration and test doubles now fail loudly on unsafe setup gaps:
  Binance/Bitget/KuCoin hedge-mode failures propagate, KuCoin order side inference
  prefers explicit hedge-side payloads, custom endpoint override errors raise, and
  the fake exchange rejects invalid reduce-only orders instead of silently clipping them.
- OHLCV cache integrity handling now retries expired persistent gaps after the documented
  seven-day horizon, avoids stealing active fetch locks by unlinking lock files, serializes
  v2 chunk writes with per-chunk locks, and no longer wipes corrupt chunks before a remote
  repair succeeds.
- Backtest HLCV preparation now preserves real-row validity through source-dir/direct fetches,
  dataset overrides, and archive day imports, preventing edge-filled listing/delisting gaps
  from becoming tradable candles.
- Changed optimizer candidate canonicalization so disabled trailing-martingale close
  retracement params collapse to bounded canonical values before evaluation, duplicate
  detection, and Pareto/result persistence.
- Fixed suite backtests so scenario data preparation always includes the base
  `live.approved_coins` universe even when other scenarios define explicit
  coin subsets, and so `coin_overrides.<coin>.live.forced_mode_<side>=normal`
  is carried into Rust backtests as a forced normal active slot.
- Tightened fail-loud handling for live cancellations, current fill-event caches, and
  single-exchange HLCV preparation: unexpected cancel failures now propagate through
  restart/error handling, unreadable current fill-cache day files fail cache loading, and
  per-coin HLCV fetch errors no longer silently shrink the requested backtest universe.
- Added HSL `coin` signal mode, which tracks per-coin realized drawdown plus current UPnL
  against the configured slot budget and panic-closes only the affected `coin+pside`. Live
  uses configured `n_positions`; backtests use configured `n_positions` in fixed-WEL mode and
  the effective tradability-aware denominator when `dynamic_wel_by_tradability=true`.
- Hardened HSL `coin` restart reconstruction and backtest artifacts: live replay now restores
  active RED panic state from per-coin history, and coin-mode backtests emit side strategy-equity
  and drawdown series with one sample per bar.
- Fixed backtest HSL setup so enabling HSL on one side no longer implicitly enables the disabled
  opposite side through the common HSL config.
- Hardened live coin-HSL restart replay so open positions and panic/cooldown history require
  exchange-derived per-coin timeline PnL, panic-flatten markers are reconstructed per coin, and
  active cooldown intervention/residue state survives restart.
- Exposed `live.hsl_signal_mode` on the backtest/optimize CLI as `--hsl-signal-mode`,
  so HSL signal mode can be changed without editing the config file.
- Added HSL backtest metrics for per-event panic-close realized-loss drawdown severity:
  min, mean, and max loss as a fraction of equity before each panic-close episode.
- Reduced suite-optimizer seed-evaluation memory pressure by passing lazy-sliced coin
  columns to Rust as active indices instead of materializing per-worker HLCV coin-subset
  copies.
- Fixed live Hyperliquid `xyz:*` stock-perp EMA reads during off-hours/no-trade
  tails by allowing stock-perp-only flat zero-volume tail candles from the last
  real close, while preserving fail-loud behavior when no real candle seed exists.
- Tightened optimizer starting-config semantics: seed and fine-tune anchor values outside
  `optimize.bounds` are clamped with aggregated source/key logging, while base-config runtime
  policy fields such as HSL/unstuck boolean toggles now win over anchor configs.
- Added a Metric/Metric Correlations table to `passivbot tool pareto-analyze`,
  limited to metrics already shown in Metric Distributions, and wrote the full
  selected metric-correlation set to `metric_correlations.csv` when using
  `--output-dir`.
- Changed pymoo NSGA-III `population_size: null` to use a default population
  budget of `500` while auto-selecting the finest compatible reference-direction
  grid, so adding objectives no longer drops the per-generation population
  because of Das-Dennis grid jumps.
- Added `passivbot tool ohlcvs-doctor` to audit v2 OHLCV chunk caches and
  rebuild `caches/ohlcvs/catalog.sqlite` metadata from copied `data/` chunks.
- Capped loss/profit-ratio analysis metrics at a finite value for losing-only
  backtests while keeping no-PnL runs neutral, preventing optimizer scoring on
  `loss_profit_ratio` from failing after JSON/Python metric aggregation.
- Capped `risk_we_excess_allowance_pct` by the side's `total_wallet_exposure_limit`
  before per-position sizing, WEL enforcement, unstuck, threshold weighting, and
  min-effective-cost projections use it, so `n_positions = 1` no longer allows
  per-symbol exposure above TWEL through excess allowance.
- Changed v8 optimizer fine-tuning so combining `--fine-tune-params` with `--start`
  treats the starting configs as fixed-parameter anchors, letting one run tune selected
  params across multiple Pareto candidates while preserving plain `--start` as seed-only.
- Canonicalized live fill-event accounting: cached fills now store gross `pnl`,
  signed quote-currency `fee_paid`, fee-quality metadata, and a
  `gross_pnl_quote_fee_best_effort_v2` cache contract. Non-quote fees are
  converted when a fresh ticker is available, otherwise estimated from reported
  fee rates or `live.fee_pct_fallback`; every fill is sanity-checked against
  `live.fee_pct_sanity_abs_max`.
- Fee-policy warnings now deduplicate repeated overlapping-refresh examples
  and include the original rejected fee ratio/source when sanity replacement
  uses `live.fee_pct_fallback`.
- Live realized-loss gates, unstuck allowances, fill health summaries, and
  backtest rolling realized-PnL risk windows now use net realized PnL
  (`pnl + fee_paid`) consistently. KuCoin positions-history net cycle PnL is
  converted back to gross close-fill PnL before reconciliation, and
  legacy/missing-contract caches are repaired when safe or quarantined and
  rebuilt automatically from exchange fills on startup.
- Fixed live bots so non-shutdown `asyncio.CancelledError` failures from CCXT
  account-state or candle fetches are logged, counted, and routed through the
  existing restart/backoff path instead of silently exiting without countdown.
- Fixed live orchestrator order calculation so live bots no longer require
  `backtest.market_order_slippage_pct`; backtest-only market slippage remains
  confined to backtest simulation.
- Backtest and optimizer runs now automatically clean stale `caches/ohlcvs/materialized/`
  scratch payloads while preserving materialized directories locked by active processes.
- `live.custom_endpoints_path` is now part of the canonical config schema, so normalized
  live configs preserve endpoint override files instead of dropping the documented setting.
- Updated user-facing docs for current CLI logging flags, custom endpoint setup,
  backtest exchange naming, suite exchange expansion, uncovered tool commands, and
  current Forager/indicator wording.
- Changed the v8 strategy runtime to use Rust-owned `trailing_martingale` and `ema_anchor`
  strategy parameters end-to-end, with no production fallback bridge from removed v7
  `trailing_grid` fields.
- Added deprecated v8 compatibility strategy kind `trailing_grid_v7` plus
  `passivbot tool migrate-config-v7` for explicitly converting v7 trailing-grid configs into
  canonical v8 shape without reinterpreting them as `trailing_martingale`.
- Fixed live v8 EMA warmup sizing to fail loudly on malformed strategy/forager span values
  instead of silently shrinking the warmup window and risking missing orchestrator EMA inputs.
- Hardened `trailing_martingale` close recursion against non-finite close prices before
  sorting recursive close ladders.
- Updated the canonical v8 schema defaults and mirrored example config to the new
  `trailing_martingale` long-only `n_positions = 4` profile at
  `configs/examples/default_trailing_martingale_long.json`.
- Fixed Hyperliquid `xyz:*` stock-perp backtest/optimizer startup so explicit
  `backtest.ohlcv_source_dir` data can use the direct source-dir preparation path when
  strict local v2 materialization is unavailable.
- Added optimizer `--resume` checkpoint recovery safeguards: resume now requires a
  readable checkpoint plus prior `all_results.bin` metadata, rejects changed optimizer
  search domains before appending results, and exits non-zero on fatal optimizer errors.
- Changed explicit `backtest.ohlcv_source_dir` backtest/optimizer runs to read that
  caller-managed OHLCV tree directly instead of first importing it into PB7's v2 raw
  `caches/ohlcvs` store; the final prepared HLCV cache is still written normally.

## v7.12.0 - 2026-05-27

- Changed backtest/optimizer HLCV preparation to treat normal market availability limits as coverage metadata: late coin starts and unavailable tails are logged and persisted in artifacts instead of aborting the whole run; large internal gaps are repaired or excluded from the tradable window so synthetic spans do not become tradable. Corruption, malformed candles, missing BTC benchmark data, and no tradable candles still fail loudly.
- Final `caches/hlcvs_data/` caches now require valid manifests and old manifest-less final caches rebuild by default; explicit override datasets require valid manifests/checksums.
- Added per-coin HLCV coverage metadata to materialized datasets, including requested range, valid start/end, leading/trailing missing minutes, internal gap counts/windows, and synthetic fill count/source.
- Capped omega-ratio analysis metrics at a finite value when a backtest has positive returns with no losing days, and reports flat/no-movement windows as `0.0`, preventing optimizer scoring metrics from disappearing during JSON/Python aggregation.
- The `v8` branch is versioned as the next major release, `v8.0.0`.
- Increased the pymoo NSGA3 auto reference-direction cap from `330` to `500`, giving 9-objective auto-population optimizer runs `495` reference directions instead of `165`.
- Fixed v8 strategy min-effective-cost gating so live and backtest use the active strategy's initial sizing parameter instead of legacy flat `BotParams.entry_initial_qty_pct`.
- Fixed flat shared bot keys to override grouped defaults during config canonicalization, and changed flat strategy coin overrides to fail loudly instead of being silently discarded.
- Added a live-only entry cooldown guard that can anchor `entry_cooldown_minutes` from exchange-observed position increases when fill-event data is temporarily delayed.
- Fixed optimizer/backtest HLCV universe preparation for canonical v8 grouped bot config, so side enablement reads `bot.<side>.risk.total_wallet_exposure_limit` and `bot.<side>.risk.n_positions` without requiring runtime flat aliases.
- Fixed strict v2 HLCV materialization so a leading invalid prefix is accepted as pre-inception when the first valid candle matches authoritative first-timestamp metadata, even if an older persistent gap starts inside that prefix.
- Fixed strict v2 HLCV gap cleanup so overlapping persistent pre-inception records no longer crash materialization with a SQLite unique-constraint error while normalizing authoritative first-candle boundaries.
- Fixed suite HLCV preparation so individual exchange datasets use the date windows of the scenarios that consume them instead of inheriting the global combined-suite window.
- Fixed `position_held_*` and `position_unchanged_*` backtest metrics so still-open positions are measured through the backtest end timestamp instead of stopping at the last fill.
- Changed optimizer `fixed_params` and `--fine_tune_params` to v8 dotted config-path selectors, with path-prefix matching such as `long.strategy` for `bot.long.strategy.<active_strategy>.*` and sorted multi-line logs showing each selector expansion in compact dotted form.
- Extended optimizer `fixed_params` and `--fine_tune_params` selectors to match config-path suffixes, so leaf selectors such as `we_excess_allowance_pct` expand to all matching long/short bounds while still avoiding partial-substring matches.
- Added fill-activity backtest analysis metrics covering fill counts, per-day rates, long/short and entry/close splits, no-fill gap durations, slot-normalized activity, active fill days, analysis duration, active symbols, and top-symbol fill concentration.
- Fixed `passivbot tool pareto -o/--objectives` so stored fill-activity metrics such as `fills_gap_p95_hours` can be used for candidate selection even when they were not part of the optimizer run's original `optimize.scoring`.
- Added `passivbot tool pareto-compress` for selecting a compact, non-destructive representative subset from a Pareto front, with optional copied JSON output and a selection manifest. When writing to a non-empty output directory, the tool now leaves unrelated files in place and overwrites only selected output filenames plus `selection.json`.
- Changed optimizer scoring and limit handling to fail loudly when a configured metric is absent from backtest analysis instead of silently treating it as zero or no violation.
- Restored `backtest_completion_ratio` in backtest analysis and optimizer suite metrics so default optimizer limits can reject early-stopped backtests without failing on a missing metric.
- Replaced the v7 `trailing_grid` strategy schema with the v8 `trailing_martingale` strategy. Entries and closes now use unified threshold/retracement parameters with 1h/1m volatility scaling; entries also support wallet-exposure scaling, while closes support additive wallet-exposure threshold shifts for recursive reduce ladders.
- Bumped the canonical config schema to `v8.0.0`, added shared dynamic distance multiplier logic for `trailing_martingale` and `ema_anchor`, changed `ema_anchor` inventory sensitivity to signed wallet-exposure ratio, and added explicit runtime toggles for the position exposure enforcer, total exposure enforcer, and auto-unstuck.
- Renamed timeframe-specific EMA span config fields to use explicit `1m` / `1h` suffixes, including forager `volume_ema_span_1m` / `volatility_ema_span_1m`, trailing martingale `volatility_ema_span_1m` / `volatility_ema_span_1h`, and ema anchor volatility span fields.
- Changed v8 risk handling so manual-mode positions are outside bot-managed active slots and bot-scope TWEL accounting, renamed user-facing WEL/TWEL enforcer config keys to `position_exposure_enforcer_*` / `total_exposure_enforcer_*`, and gave total exposure enforcement a second reduction pass that can trim least-stuck bot-scope positions below their per-position floor when required to bring total exposure back under the configured threshold.
- Added canonical strategy-equity recovery-duration metrics: `strategy_eq_recovery_days_mean`, `strategy_eq_recovery_days_median`, `strategy_eq_recovery_days_p95`, `strategy_eq_recovery_days_p99`, `strategy_eq_recovery_days_mean_worst_5pct`, `strategy_eq_recovery_days_mean_worst_1pct`, and `strategy_eq_recovery_days_max`; `peak_recovery_days_strategy_eq` remains as a backwards-compatible alias for the max.
- Changed pymoo NSGA-II optimization so `optimize.population_size: null` now auto-resolves to `250`, avoiding startup failures when `optimize.pymoo.algorithm: "auto"` selects NSGA-II for small objective sets.
- Added backtest `entry_interval_hours_mean`, `entry_interval_hours_median`, `entry_interval_hours_p95`, `entry_interval_hours_p99`, and `entry_interval_hours_max` analysis metrics, measuring gaps between normal initial entries per coin and side.
- Fixed CCXT live startup so malformed metadata on unrelated ineligible exchange markets no longer blocks the bot, while executable symbols still fail loudly when required qty sizing metadata is missing.
- Fixed Bybit UTA live balance parsing so Passivbot uses account equity minus perpetual UPNL as raw balance instead of double-applying UPNL from collateral `usdValue` fields.
- Fixed KuCoin aggregate realized-PnL enrichment so positions-history rows are reconciled as cycle observations against reconstructed fill lifecycles only when unambiguous, preventing rapid or delayed position cycles from being assigned to the wrong close fill while ambiguous rows stay synthetic and refreshable.
- Changed staged live bounded active 1m tail gaps to project provisional no-trade EMA inputs for close, quote-volume, and log-range instead of carrying forward latest-real EMA values; projected rows and EMA values are not persisted or reused once real candles arrive.
- Live fill events now synthesize missing realized PnL from canonical fill history when exchange enrichment remains unavailable, with explicit synthetic/degraded provenance and later authoritative replacement when enriched data is fetched.
- Added exponential backoff while live account refresh is blocked by pending realized-PnL enrichment, so stale KuCoin close fills no longer trigger continuous fill-history polling while PnL-dependent logic remains blocked.
- Fixed OHLCV v2 backtest/data-download fetches so newly downloaded rows are written directly to `caches/ohlcvs` instead of repopulating deprecated `caches/ohlcv` daily shards.
- Cleaned up `passivbot tool generate-mcap-list` startup output by routing through its normal CLI entrypoint and suppressing noisy symbol-map lock maintenance warnings.
- Fixed live fill refreshes so cached close fills with pending realized PnL keep extending the incremental refresh window until exchange enrichment catches up, KuCoin positions-history enrichment uses a bounded delayed-record lookahead, and pending-PnL account refresh blocks no longer burn the generic restart budget.
- Fixed backtest/data-downloader startup when the legacy `caches/ohlcv` path is a dangling symlink after moving to the v2 `caches/ohlcvs` store.
- Increased the pymoo NSGA3 auto reference-direction cap from `330` to `500`, giving 9-objective auto-population optimizer runs `495` reference directions instead of `165`.
- Restored `backtest_completion_ratio` in backtest analysis and optimizer suite metrics so default optimizer limits can reject early-stopped backtests without failing on a missing metric.
- Changed optimizer scoring and limit handling to fail loudly when a configured metric is absent from backtest analysis instead of silently treating it as zero or no violation.
- Fixed `passivbot tool pareto -o/--objectives` so stored fill-activity metrics such as `fills_gap_p95_hours` can be used for candidate selection even when they were not part of the optimizer run's original `optimize.scoring`.
- Fixed `position_held_*` and `position_unchanged_*` backtest metrics so still-open positions are measured through the backtest end timestamp instead of stopping at the last fill.
- Added `passivbot tool pareto-analyze` and `passivbot tool pareto-compress` for inspecting Pareto-front metric/config distributions and selecting compact representative subsets with optional copied JSON output. When `pareto-compress` writes to a non-empty output directory, it leaves unrelated files in place and overwrites only selected output filenames plus `selection.json`.

## v7.11.0 - 2026-05-13

- Fixed backtests with asymmetric `approved_coins` so long-only and short-only coin lists remain side-specific, disabled sides no longer inflate HLCV data preparation, and dynamic WEL-by-tradability counts side-eligible coins separately.
- Fixed Rust extension auto-rebuild coordination so simultaneous bot startups share one compile, waiters re-check freshness after the lock, stale lock timeouts fail closed, and stale shadow artifacts are no longer stamped as current.
- Live fill events now distinguish detected fills from realized-PnL enrichment: close fills whose exchange PnL details are not yet available log `pnl=pending`, block PnL-dependent logic until enriched, and emit an enrichment log once the authoritative PnL arrives.
- Fixed TWEL auto-reduce dead zones where positions sitting at raw per-position WEL could block reductions even though `risk_twel_enforcer_threshold < 1.0` required total exposure below TWEL.
- Changed the HSL config default `live.hsl_signal_mode` to `unified`, making account-level strategy drawdown the canonical HSL signal while keeping `pside` available for side-local HSL tuning, and clarified that HSL RED waits for all positions on that side to be fully closed rather than waiting for PnL recovery.
- Added `passivbot tool merge-paretos` for combining two or more Pareto run/front directories into capped long/short starting-config sets.
- Changed optimizer `fixed_params` and `--fine_tune_params` from exact-only bounds keys to literal bounds-key selectors, with sorted multi-line logs showing each selector expansion and the resulting fixed/tunable bounds.
- Changed no-path `passivbot tool pareto` discovery to choose the lexicographically latest `optimize_results/<run>/pareto` directory containing at least one `*.json` candidate instead of using directory modified time.
- Fixed Gate.io live order creation with current CCXT/Gate.io by passing Passivbot custom ids as `clientOrderId`, letting CCXT emit Gate.io's required `t-`-prefixed order `text` while preserving the embedded Passivbot order-type marker.
- Fixed live foreign-writer detection so a bot's own freshly acknowledged orders can be recognized by exchange order id, canonical Passivbot custom id, or a strict recent order fingerprint instead of relying only on raw client-id string equality.
- Fixed OHLCV v2 planning so persistent gaps are not bypassed by sparse store bounds, and single-exchange backtest preparation no longer attempts the same v2 local path twice before falling back.
- Hyperliquid HIP-3 ticker fallback now uses dex inference from market metadata instead of relying only on CCXT `info.hip3`, and ticker probe coin resolution now accepts HIP-3 aliases such as `SP500`, `xyz:SP500`, and `XYZ-SP500`.
- Hyperliquid live startup now detects and logs account abstraction mode, treats `portfolioMargin` as unified-compatible for HIP-3/non-standard perps, limits non-unified `dexAbstraction` accounts to vanilla perps, and adds `passivbot tool hyperliquid-abstraction-probe`.
- Hyperliquid staged live state now uses one coherent positions+balance snapshot without the earlier exchange-specific HIP-3 reserve-reconciliation layer, reducing `REST`/`REST+open_orders` balance oscillation and order-size churn.
- Live candle fetching now applies the configured candle fetch delay inside `CandlestickManager` before each CCXT OHLCV call, reducing paginated startup/refresh bursts across exchanges instead of only sleeping after symbol-level refresh loops.
- Live background candle warmup now runs as lower-priority work by default, with one-symbol concurrency and minimum pacing, so broad cache catch-up is less likely to compete with account-state refreshes and order execution.
- Live fill refreshes now distinguish routine recent-fill checks from explicit account-state confirmations: routine incremental refreshes default to a narrower `live.fills_recent_overlap_minutes=10.0`, while confirmation refreshes keep `live.fills_confirmation_overlap_minutes=60.0`.
- Staged live routine fill refreshes now prefetch in a single-flight background lane after the initial fill stamp, so ordinary minute-boundary fill checks no longer block account refresh/order planning unless fills fall behind or an explicit confirmation is pending.
- Live forager INFO logs now report actual selected-set/slot changes, hysteresis replacements, and periodic heartbeats; rank-only score movement is kept at DEBUG to reduce log noise.
- Live active-candle refresh gaps that look like one-candle exchange publication lag now log at INFO with slower throttling instead of recurring WARNING lines; larger/actionable gaps still warn.
- Live initial entry creations now have an optional executor-side market-distance gate (`live.initial_entry_exec_max_market_dist_pct`, default `0.005`) to avoid posting far-from-market EMA-drifting initial orders. Blocked initial entries are visible at INFO when first blocked or when price/qty drift exceeds `live.order_match_tolerance_pct`.
- Increased the default `live.forager_score_hysteresis_pct` from `0.005` to `0.02` to reduce forager selection flip-flop observed in live multi-exchange testing.
- Tightened the default live OHLCV budget from `30` to `24` fetches/minute and increased default `live.recv_window_ms` from `5000` to `10000` to reduce public-data pressure and Binance timestamp drift rejects.
- Detailed per-symbol min-effective-cost entry blocks now log at INFO at most once per hour per symbol/side, summary counts are INFO with explicit blocked/detailed/suppressed totals, and unchanged repeated blocks are suppressed to DEBUG to keep normal logs actionable.
- Routine empty fill-refresh timing, KuCoin empty fill-history fetch chatter, and clean fast order-settlement confirmations now log at DEBUG; INFO is reserved for new fills, blocking fill confirmations, very slow fill refreshes, non-open-order confirmation changes, or slow settlement.
- Live staged completed-candle freshness now keys the ledger by required completed minute per symbol instead of mutable cache internals, reducing unnecessary safe deferrals after background candle refreshes improve the cache.
- Live websocket reconnect logs are now throttled: early reconnects and persistent reconnect storms remain visible at WARNING, while repeated reconnect chatter and tracebacks move to DEBUG.
- Live forager candle refresh now prioritizes symbols with positions or open orders and budgets forager-only active symbols with `live.max_ohlcv_fetches_per_minute`, reducing broad approved-coin OHLCV refresh bursts.
- Live forager candle refresh now has a wall-time cap (`live.max_forager_candle_refresh_seconds`, default `45`) so best-effort broad candidate candle catch-up yields and retries later instead of monopolizing runtime on slow or sparse exchanges.
- Staged live active-symbol candle freshness now tolerates bounded open-ended 1m tail gaps with `live.max_active_candle_tail_gap_minutes` (default `10`), carrying forward the latest real candle/EMA state with warning-visible diagnostics instead of blocking immediately; gaps beyond the threshold still block the affected trading-critical candle surface.
- Added `passivbot tool ticker-probe`, a read-only exchange capability probe for `fetch_ticker`, `fetch_tickers(symbols)`, `fetch_tickers()`, and optional top-of-book data to support separating live price truth from candle fetching.
- Added `passivbot tool ticker-endpoint-probe`, a multi-user read-only CCXT timing probe for ticker variants, bids/asks, order book, 1m OHLCV tail behavior, market metadata, and private account-state endpoints.
- Staged live order planning now sources bid/ask/last from a dedicated market snapshot provider before falling back to candle-manager last prices, moving current price truth out of incomplete candle paths.
- Live authoritative refresh now always uses the staged account-state pipeline. The legacy live refresh path and `live.authoritative_refresh_mode` config selector were removed; older branches such as `v7.10` remain the comparison point for legacy behavior.
- Live market snapshots now cache all valid symbols returned by a bulk ticker response and coalesce concurrent cache misses behind one in-flight `fetch_tickers()` request, reducing redundant remote calls during staged planning.
- Live market snapshots now strictly retry missing symbols with the exchange's symbol-scoped ticker endpoint before failing, avoiding unnecessary execution-loop aborts when a bulk ticker response omits a few requested symbols.
- Live market snapshots now use an explicit ticker strategy: broad `fetch_tickers()` remains the default, Hyperliquid keeps its custom `allMids` path, and Bitget defaults to `fetch_tickers(symbols)` because Bitget's broad CCXT ticker response can omit requested USDC perp symbols.
- Live foreign-writer detection now treats create-order timeouts/errors as ambiguous bot-owned attempts for a bounded recent window, preventing the bot from falsely flagging its own orders as foreign when the exchange accepted an order but the create call timed out.
- Staged live execution now records account, candle, and market-data freshness in an explicit ledger and blocks new order creation for a symbol when a bot-created order disappears before a follow-up account-state refresh can rule out a fill.
- Staged live order planning now hard-fails before Rust order calculation if the current authoritative epoch is missing required account, completed-candle, or market-snapshot freshness stamps.
- Staged live order sorting now preserves the original deterministic order with a visible warning when market prices are unavailable, instead of silently assigning missing prices a neutral distance.
- CandlestickManager now has a completed-candle-only contract: compatibility latest-close helpers no longer fetch tickers, current-minute OHLCV, or persist in-progress candles; live current price reads use market snapshots instead.
- Live candle health diagnostics now report required completed-candle coverage for active symbols, including 1m/15m/1h freshness, missing spans, known gaps, and runtime synthetic candles, with detailed output at `DEBUG` and `INFO` only when interesting.
- Live forager candle budgeting now ranks refresh candidates by latest completed-candle staleness, keeps position/open-order symbols outside non-critical budget limits, and supports `live.max_forager_candle_staleness_minutes` to cap acceptable eligible-coin staleness.
- Live startup now performs only a minimal trading-ready candle warmup for symbols with positions/open orders before entering the main loop, then runs broad approved-coin candle catch-up in a cancellable background task. Set `live.defer_broad_candle_warmup=false` to keep the old blocking broad warmup behavior.
- Live shutdown now interrupts candle/EMA warmup and cancels a stuck execution loop before closing exchange sessions, reducing Ctrl-C/shutdown hangs during broad market-data refresh.
- Live startup warmup now reuses already-fresh 1m candle cache windows when local coverage and refresh metadata prove the required completed-candle range, with `live.force_cold_startup=true` available to force the existing cold fetch path.
- Removed the deprecated broad `live.price_distance_threshold` setting. Rust-owned order generation and the live `order_match_tolerance_pct` replacement tolerance now define normal order placement/churn behavior; stale `price_distance_threshold` keys are stripped during config normalization. Use `live.initial_entry_exec_max_market_dist_pct` for the narrower live-only initial-entry posting economy gate.
- Live startup logs one-shot readiness timings for account state, active candles, optional HSL history replay, first market refresh, startup readiness, and broad candle warmup completion.
- Rust order orchestration now emits only the next most-likely flat entry order for live symbols without a position, while preserving full entry-grid expansion once a position exists and preserving backtest next-candle expansion behavior.
- Added `live.forager_score_hysteresis_pct` to keep already-selected flat forager coins when challenger scores are only marginally better, reducing selection flip-flop in live, backtest, and optimizer.
- Live forager diagnostics now include Rust-owned selection score logs: `INFO` reports selected/incumbent coins and top scores only on selection/hysteresis changes or periodic heartbeat, while `DEBUG` includes top-score component detail.
- Live config validation now requires `live.max_n_cancellations_per_batch > live.max_n_creations_per_batch`, making the intended cancel-before-create batch capacity contract explicit at config load.
- Fixed OHLCV v2 local preparation so sparse invalid v2 windows are repaired from existing legacy daily shards first, then fetched with exact intraday ranges instead of triggering full-range archive downloads or collapsing same-day repair windows to empty fetches.

## v7.10.0 - 2026-04-22

- Updated the hardcoded schema defaults and mirrored example config to a new trailing-grid `n_positions = 7` profile from `tmp/candidate.json`; the canonical example file is now `configs/examples/default_trailing_grid_long_npos7.json`. Default approved coins, suite scenarios, optimizer bounds, and optimizer scoring/limit templates were refreshed with canonical `*_strategy_eq` metric names and day-based duration metrics while keeping backtest defaults at `candle_interval_minutes = 1`, `end_date = "now"`, and `suite_enabled = false`.
- Removed inflated grid re-entry behavior from current live/backtest/runtime paths. Grid re-entries are now always normal-or-cropped, config loading strips deprecated `bot.{long,short}.entry_grid_inflation_enabled` flags after warning when they were set `true`, and legacy inflated order-type ids remain decodable for historical fills and live restart compatibility.
- Added day-denominated backtest analysis metrics mirroring the existing duration metrics: high exposure, peak recovery, position held, and position unchanged outputs now keep their `*_hours*` fields and also expose equivalent `*_days*` fields.
- Backtest `drawdown_worst_mean_1pct` and `drawdown_worst_mean_1pct_strategy_eq` now compute drawdowns from the full-resolution equity curve first, then average the worst 1% of daily worst drawdowns. This better distinguishes isolated max-drawdown spikes from sustained drawdown regimes.
- Backtest BTC collateral is now initialized at the first active trading step instead of at the beginning of EMA warmup data, so warmup-period BTC price movement no longer changes starting account equity.
- Added `strategy_equity` to backtest `balance_and_equity.csv.gz` artifacts so the collateral-agnostic strategy-equity curve is available alongside balance and USD/BTC equity.
- Added backtest artifact helpers for loading a run's config, analysis, fills, balance/equity data, HLCVs, timestamps, BTC/USD prices, and market settings into notebooks, plus a single-coin fill plot helper backed by the loaded artifact data.
- Added `passivbot tool inspect-ohlcvs` for diagnosing the v2 OHLCV store under `caches/ohlcvs/`. The tool can summarize catalog counts and symbols, or inspect one symbol's bounds, chunk validity, persistent gaps, and recent fetch attempts.
- Renamed collateral-agnostic strategy-equity analysis metrics to canonical `*_strategy_eq` names and deprecated the old `*_strategy_pnl_rebased` / `*_hsl` metric names as input aliases. New `analysis.json` output uses canonical names, while optimizer, Pareto, limits, aggregate config, and visibility filters still resolve old stored result keys. `peak_recovery_hours_pnl` now uses net realized PnL (`pnl + fee_paid`) and includes the open tail from the last realized-PnL peak to the end of the backtest.
- Fixed suite-mode limit semantics so `passivbot optimize` and `passivbot tool pareto` now resolve omitted `stat=` the same way: explicit `stat=` still wins, otherwise both defer to `backtest.aggregate.<metric>`, then `backtest.aggregate.default`, then `mean`. This removes the old optimizer-only behavior where `>` silently implied `min` and `<` silently implied `max`.
- Fixed GateIO 1m OHLCV backfills older than GateIO's recent public history window. Passivbot now clips unsupported GateIO fetches, records those spans as unavailable, and avoids repeated `Candlestick too long ago` API failures; use `backtest.ohlcv_source_dir` or another candle source for older GateIO backtests.
- Reduced optimizer startup memory pressure when warming from large starting-config sets. Starting configs now stream into quantization instead of being fully materialized up front, and pymoo worker evaluations now reuse per-worker evaluator state plus metrics-only backtests instead of serializing full evaluator payloads and full backtest histories for every candidate.
- Upgraded the pinned `ccxt` dependency from `4.5.22` to `4.5.48` and added a dedicated CCXT upgrade validation workflow with live snapshot capture/diff tooling plus offline contract fixtures for upgrade drift.
- Fixed backtest `pnls_max_lookback_days` rolling realized-PnL reconstruction to match live semantics exactly: both now derive peak/current PnL stats from the active lookback window only by filtering in-window fills and recomputing cumulative realized PnL from that filtered sequence. This fixes overstated auto-unstuck allowance and related risk gating drift caused by the old rebased rolling-peak implementation.
- Fixed all-zero `forager_score_weights` configs to normalize to EMA-readiness-only ranking consistently across Python config prep, Rust selection, and optimizer inputs instead of drifting into ambiguous fallback behavior.
- Stopped hydrating omitted `config.bot.{long,short}` fields from schema-tuned bot defaults in legacy/current configs. Newly omitted feature-style params now hydrate to explicit off/compatibility values with config logs, sparse disabled sides remain loadable, legacy `n_closes` and `min_markup` aliases are preserved, and the Rust parser now fails loudly instead of silently supplying bot-key fallbacks.
- Hyperliquid live balance reconciliation no longer republishes bot-managed resting-order reserve after `fetch_open_orders()`. This removes the old `REST`/`REST+open_orders` balance oscillation path that could trigger self-induced order-size churn.
- Live balance/equity replay now skips unsupported historical fill symbols that have no current position, and coin-mode HSL restart reconstruction accepts realized-only rows only when fill replay proves that coin side is flat. This avoids restart loops from stale delisted/unsupported history while preserving hard failures for open or ambiguous risk inputs.
- New/generated live configs now enable bounded text log rotation by default while preserving explicit `logging.rotation = false` in existing configs.
- Fixed OHLCV cache backfills so earlier requested ranges are no longer silently suppressed just because later shards already exist on disk. CandlestickManager now separates earliest observed cached candles from authoritative exchange-history lower bounds, migrates stale legacy `pre_inception` gaps out of old indexes, and warns when a requested span is clipped by an authoritative start boundary.
- Live bots now watch for newer Passivbot-managed open orders they did not emit during the current runtime and stop after repeated detections within a rolling window. This ignores manual/non-Passivbot orders and older inherited orders, reducing the chance of two Passivbot instances silently competing on the same account indefinitely.
- Staged live bots now route orchestrator latest-price reads through `CandlestickManager`, and `CandlestickManager.get_last_prices()` now uses cheap cache hits plus one bulk ticker snapshot when safe before any per-symbol fallback. This materially reduces staged live market-data call bursts on exchanges like Bybit.
- Live runtime shutdown is now cleaner: Ctrl-C and stop-signal paths stop execution sooner, await cancelled maintainer tasks during shutdown, exit restart cooldowns promptly, and classify Bybit `110001 / order not exists or too late to cancel` as the expected benign cancel race instead of logging a noisy error traceback.
- Fixed CLI `live.approved_coins` / `live.ignored_coins` file overrides so live reload keeps the original file path in `_coins_sources` instead of freezing the first parsed snapshot. Mid-run edits to `-s path/to/file` coin lists now take effect correctly.
- Fixed optimizer Pareto artifact persistence so saved `pareto/*.json` candidates now preserve the exact evaluated bot parameter values instead of being re-rounded again inside `ParetoStore`. This restores replay fidelity between `passivbot tool pareto` selections and standalone `passivbot backtest` runs of the selected file.
- Fixed `passivbot optimize/backtest -cim/--candle-interval-minutes` type handling so integral values stay integers through the Python/Rust backtest boundary. This fixes crashes like `TypeError: 'float' object cannot be interpreted as an integer` when using `-cim 2`.
- Hyperliquid non-unified (`dexAbstraction`) accounts now hard-fail if any HIP-3/non-standard perp symbol appears in effective `approved_coins` or live exchange state. Those symbols now require `unifiedAccount` mode instead of being partially skipped or partially supported.

## v7.9.1 - 2026-04-13
- Removed the legacy `python src/downloader.py ...` entrypoint. Use `passivbot download ...` for OHLCV cache warming.
- Added formal top-level `config_version` schema tagging starting at `v7.9.0`. Canonical defaults and the mirrored example config now carry the schema version, older configs log a migration attempt during load, and the loader upgrades them to the current schema version.
- Backtests now read `market_orders_allowed`, `market_order_near_touch_threshold`, and `pnls_max_lookback_days` from `config.live` only. `config.backtest` no longer accepts those fields, which avoids silent drift between live and backtest behavior.
- Pre-v7.9 backtests did not correctly observe `pnls_max_lookback_days`, and they also did not simulate ordinary market-order execution. v7.9+ treats both as backtest correctness fixes rather than preserving bug-compatibility via migrated `backtest` overrides.
- `live.pnls_max_lookback_days` now uses one consistent contract across live risk logic, HSL, plotting, and backtests: `0` means the minimal effective lookback for that path's native sampling resolution, positive numbers mean that many rolling days, and `"all"` means full available history. Full-history live fill refreshes also stay incremental once the cache is warm instead of forcing a full refetch every cycle.
- `passivbot optimize --help-all` now exposes fixed per-side bot runtime overrides for `hsl_enabled`, `hsl_orange_tier_mode`, and `hsl_panic_close_order_type` without making them optimizer dimensions, and `optimize.bounds` now rejects trying to tune those non-numeric bot fields.
- Restored `backtest.visible_metrics` for standalone backtest terminal output filtering. `null` now shows optimize-derived metrics, `[]` shows all, and explicit lists add extra metrics without affecting the full saved `analysis.json`.
- Fixed `CCXTBot.create_ccxt_sessions()` using the generic exchange name (e.g. `binance`) instead of the futures-specific CCXT id (`binanceusdm`). This caused `load_markets()` to unnecessarily fetch COIN-margined markets from `dapi.binance.com`, and a timeout on that endpoint would cascade-fail all symbol trade fetches and open order updates.
- Fixed `BinanceFetcher._fetch_symbol_trades` sending future `endTime` (now+1h) and using a tight 7-day safety margin (0.1%), causing Binance `-4181 "Invalid start time"` errors for symbols with sparse trades. Removed the +1h extension and widened the margin to 1%.
- Hyperliquid live sizing now compensates for missing cross-margin reserve in `fetch_balance()`: HIP-3 stock-perp positions can restore their hidden `marginUsed`, and Passivbot-managed resting non-reduce-only entry orders can restore reserved margin on both HIP-3 and flat standard perps. This prevents the bot from misreading its own reserved margin as equity loss and churning order sizes in cancel/replace loops, while still ignoring external/manual orders.
- Backtest/optimizer HLCV dataset caches under `caches/hlcvs_data/` now use descriptive directory names with exchange, coin label/count, actual dataset date range, and the cache hash suffix. Existing legacy hash-only cache directories still load unchanged.
- Config validation now hard-fails invalid `bot.long.unstuck_ema_dist <= -1.0` and `bot.short.unstuck_ema_dist >= 1.0` instead of silently disabling auto-unstuck with a non-positive EMA trigger price. The same guard now rejects optimize bounds that would generate those invalid values.
- Fixed Bybit `closed-pnl` pagination storms that caused retCode:10006 rate-limit errors every ~15 minutes. Fill lookback coverage is now derived from `FillEventsManager` cache metadata instead of a session-local flag, so once an open-ended lookback has been checked successfully the bot reuses incremental refreshes across restarts even when the early lookback window legitimately contains no fills.
- Applied exchange-aware EMA bundle pacing in `_load_orchestrator_ema_bundle`. Strict exchanges use the configured inter-symbol delay to avoid hour-boundary candle bursts, while exchanges with zero pacing keep the original concurrent `asyncio.gather` behavior instead of being globally serialized.
- Added random jitter (0–120s) to the hourly `init_markets` cycle so multiple bots on the same VPS don't fire heavy API bursts simultaneously.
- `passivbot live` now persists logs to a timestamped file under `logs/` by default, using `config.logging` for the on/off switch and file-rotation settings, and also refreshes `logs/{user}.log` as a stable alias to the current run for monitor tooling. This makes the built-in live workflow self-logging without needing `run_with_logging.py`.
- Added a canonical live-container runtime contract around `Dockerfile_live`, a thin `container/entrypoint.sh` wrapper, env-generated `api-keys.json` support, env-driven config overrides, and a documented Compose/Railway deployment path that reuses the normal `passivbot live` CLI instead of maintaining platform-specific baked configs.
- Restored `passivbot live --user` / `-u` as the curated shorthand for `live.user`, so existing live-run workflows using `-u account_name` work again and the alias is visible in the default live help output.
- `passivbot live -h` now shows a curated shorthand for `live.pnls_max_lookback_days` as `--pnls-max-lookback-days` / `-pmld` in the default help output instead of exposing it only via `--help-all` and the raw dotted config flag, and the flag now accepts either a non-negative float or `"all"`.
- Added `passivbot tool pareto`, a CLI Pareto front explorer that filters JSON Pareto members with optimizer-style limit expressions, defaults to the newest local `optimize_results/.../pareto` when no path is given, accepts either a run dir or `pareto/` dir, and selects a single candidate using knee, reference-point, ideal-point, weighted utility, lexicographic, or outranking methods with optional shortlist and JSON output. It now also shows the retained front's ideal point, and `-o` / `--objectives` can use stored metrics outside the original `optimize.scoring` list when their min/max direction is known.
- Changed `passivbot tool pareto` to default to the `ideal` selection method instead of `knee`.
- Fixed backtest post-processing for zero-fill runs. When a period produces no fills but still has equity samples, balance/equity resampling now keeps a `DatetimeIndex` and no longer crashes during analysis/plot generation with larger `backtest.balance_sample_divider` values.
- Fixed first-ohlcv timestamp cache handling for newly listed coins. Cached `0.0` entries are now treated as unresolved and refreshed, so optimize/backtest candle downloads correctly clamp fetch start to the coin's actual listing history instead of wasting time paging from much earlier dates.
- Fixed optimizer/backtest liquidation reporting to use an explicit Rust-provided `analysis.liquidated` flag instead of inferring liquidation from `drawdown_worst`, avoiding false positives after runs that made a new equity peak before hitting the liquidation floor.
- Added trade-level backtest metrics for completed positions: `win_rate`, `win_rate_w`, and `trade_loss_{max,mean,median}`. These measure completed-trade outcomes from open-to-flat realized PnL and normalize loss metrics by balance at trade open.
- Added optimizer-facing backtest ratio metrics `paper_loss_ratio`, `paper_loss_mean_ratio`, `exposure_ratio`, and `exposure_mean_ratio`, plus weighted `_w` variants. These measure growth relative to unrealized equity-vs-balance drawdown and actual wallet exposure.
- `live.approved_coins` now supports explicit per-side `"all"` entries such as `{"long": ["BTC"], "short": "all"}`. Missing or explicit empty side values now stay disabled instead of being backfilled from schema defaults. `live.empty_means_all_approved` is no longer part of the canonical config shape; older configs still migrate with a deprecation warning, and globally empty legacy inputs are converted to `approved_coins = "all"`.

### Upgrade Notes
- Reinstall after pulling this release. `passivbot` now validates the active environment and the loaded Rust extension more aggressively, so stale editable installs or stale shell shims are more likely to fail loudly instead of continuing silently. Use `python3 -m pip install -e .` for live-only setups or `python3 -m pip install -e ".[full]"` for backtest/optimize setups, and rebuild with `maturin develop --release` if needed.
- `optimize.backend` now defaults to `pymoo`, so optimization users need the full install profile with the new `pymoo` dependency.
- `configs/template.json` is no longer the canonical starting point. Use `configs/examples/default_trailing_grid_long_npos7.json` or omit the config path to start from the in-code defaults in `src/config/schema.py`.
- The local monitor publisher now ships enabled by default in the canonical schema. Set `monitor.enabled = false` if you do not want snapshot/event files written under `monitor/`.
- `live.max_realized_loss_pct` now defaults to `1.0`, which effectively disables the realized-loss gate unless you set a tighter value explicitly.

### Added
- **Pymoo optimizer backend** - Optimization can now run with `optimize.backend: pymoo` in addition to DEAP, with shared backend dispatch and dedicated backend coverage.
- **Pymoo NSGA-III config is now live** - `optimize.pymoo.algorithm`, nested `optimize.pymoo.shared.*`, and NSGA-III reference-direction settings are now actually honored at runtime, with auto-sized NSGA-III reference directions and `"auto"` per-variable mutation probability support.
- **Repro and sync sidecar tools** - Added `src/repro_harness.py`, `src/analysis_visibility.py`, `src/tools/capture_optimize_memory.py`, root-level `sync_tar.py`, and `vpssync.sh` for replay/debug/VPS workflows.
- **Standalone trailing diagnostics explorer** - Added `src/tools/trailing_diagnostics.py` plus reusable helpers for recomputing next-entry and next-close trailing behavior from `config + monitor snapshot` or manual inputs.
- **HSL events per-year metrics** - Backtest HSL analysis now also exports `hard_stop_triggers_per_year` and `hard_stop_restarts_per_year` so runs with different date ranges can be compared more directly without losing the absolute trigger/restart counts.
- **Fake-live exchange harness for HSL replay** - Added a deterministic `fake` exchange, `src/tools/run_fake_live.py`, and scenario-driven tests/docs so live HSL RED, cooldown restart, terminal halt, and cooldown-position policies can be replayed locally against scripted candles and manual interventions.
- **Opt-in live monitor publisher** - Added a local monitor publisher with on-disk snapshots, event streams, and retained fill/price/candle history, plus basic live bot integration for startup, balance, order, fill, and shutdown events.
- **Read-only monitor relay** - Added a local `monitor-relay` tool exposing monitor snapshots and streamed event/history tails over HTTP and websocket, including recent-message replay on connect.
- **Browser monitor dashboard** - The monitor relay now also serves `GET /dashboard` with a read-only web dashboard that bootstraps from `/snapshot`, stays live via `/ws`, shows summary/focus/positions/trailing/forager/unstuck/recent activity panels, and supports quick focus changes by clicking symbol-bearing rows.
- **Monitor web wrapper** - Added `passivbot tool monitor-web` to reuse or launch the local relay and keep the browser dashboard available from one command.
- **Terminal monitor TUI** - Added a local `monitor-tui` tool consuming the relay for current-state panels, live recent activity, focus cycling, pause/resume, and screen dumps.
- **Monitor dev wrapper** - Added a `monitor-dev` helper that reuses or launches the local relay and opens the terminal monitor with the newest bot log tailed by default.

### Changed
- **Optimizer scoring now has explicit min/max goals** - `optimize.scoring` is normalized to `{metric, goal}` entries, optimizer engines receive minimization-space values internally, and user-facing logging/Pareto tools now show raw metric values with named objectives instead of signed `w_i` fields. Legacy string-list scoring configs and legacy Pareto result files remain readable.
- **Config loading now uses a canonical staged pipeline** - Defaults now come only from in-code schema, omitted CLI configs instantiate schema defaults directly, `load_config()` / `format_config()` normalize to canonical user-facing keys without leaking runtime `filter_*` aliases, runtime aliasing moved into explicit compilation helpers, and the named example profile now lives at `configs/examples/default_trailing_grid_long_npos7.json`.
- **Realized-loss gate now ships disabled by default** - `live.max_realized_loss_pct` now defaults to `1.0`, so the gate is opt-in unless the operator explicitly chooses a tighter peak-relative realized-loss floor.
- **Executable min-cost filtering now matches actual order sizing** - `filter_by_min_effective_cost` now uses the executable minimum entry qty after `qty_step` rounding instead of raw `min_qty/min_cost` metadata, and CCXT markets reporting nonpositive `min_qty` now clamp it to `qty_step`. This fixes GateIO symbols such as `SOL/USDT:USDT` being admitted when the smallest executable order would exceed the intended initial entry size.
- **BTC-denominated backtest metrics now always use BTC equity** - `*_btc` metrics are now computed from BTC-denominated balance/equity even when `backtest.btc_collateral_cap = 0`, instead of mirroring the USD analysis. This makes metrics like `adg_btc` and `gain_btc` informative as BTC-relative performance measures for cash-collateral runs as well.
- **ADG terminal smoothing simplified** - Backtest `gain`/`adg` now smooth the terminal value by taking the mean of the last up to 3 daily equity samples instead of running an EMA over the full daily-equity series. This preserves end-of-run drawdown smoothing while reducing computation.
- **Pymoo NSGA-III population defaults are now auto-sized** - `optimize.population_size: null` now means “use the NSGA-III reference-direction count” for pymoo/NSGA-III runs, and template/config defaults now leave that field null instead of forcing a fixed 500/1000 population.
- **Unified `passivbot` CLI added** - Passivbot now installs a `passivbot` command with subcommands such as `passivbot live`, `passivbot backtest`, `passivbot optimize`, `passivbot download`, and `passivbot tool ...`. Existing direct script entrypoints like `python3 src/main.py ...` remain supported for backwards compatibility.
- **CLI help is now task-oriented by default** - `passivbot live -h`, `passivbot backtest -h`, and `passivbot optimize -h` now show curated, grouped common flags by default, while `--help-all` exposes the full advanced/raw override surface.
- **Install profiles split into `live`, `full`, and `dev`** - `pip install -e .` now targets a lightweight live-trading environment, while `pip install -e ".[full]"` adds backtesting/optimization/tooling dependencies and `pip install -e ".[dev]"` adds contributor-focused docs/lint extras on top.
- **Equity hard-stop config moved under `bot.common`** - Shared HSL settings now live at `bot.common.equity_hard_stop_loss`, with config formatting migrating legacy `live.equity_hard_stop_loss` inputs and optimizer bounds to the new location.
- **Live HSL cooldown interventions are now configurable** - RED cooldown no longer blocks the runtime in one wait path. Live now keeps the halt active while enforcing `live.hsl_position_during_cooldown_policy` (`panic`, `normal`, `manual`, `tp_only`, or `graceful_stop`) until cooldown expires or trading is resumed.
- **Browser monitor is now multi-bot first-class** - The web dashboard now consumes the multiplexed relay feed directly, shows a dense overview for all active bots in one page, and lets operators switch focused bot detail views without separate relay instances or per-bot dashboard sessions.
- **Monitor relay presence is now sticky** - Auto-discovered bots now degrade from `active` to `stale` before being pruned, and the browser overview keeps a stable bot order instead of reshuffling on every freshness blip.
- **HSL cooldown contracts are now documented explicitly** - Added a dedicated cooldown-contract reference covering RED replay, restart, and cooldown-position intervention behavior so operator/runtime expectations are easier to verify against logs.

### Fixed
- **Backtest rolling `pnls_max_lookback_days` peaks now actually expire** - Backtest risk consumers such as auto-unstuck and the realized-loss gate no longer compare the current rolling realized-PnL window against a stale all-time maximum of that rolling series. The Rust backtest now ages rolling realized-PnL state out by time even during fill droughts and uses the true in-window peak/current pair for `pnls_max_lookback_days > 0`.
- **Exchange config refresh now retries per symbol** - Live bots no longer mark exchange-config updates as complete when a symbol fails or hits a rate limit; failed symbols now back off and retry while successful symbols continue to progress.
- **Live forager key mapping** - Live runtime now consistently reads canonical `forager_*` config keys while still exporting Rust orchestrator payload fields under the internal `filter_*` names expected at the Python/Rust boundary.
- **Pymoo optimizer now records results incrementally during each generation** - Completed pymoo evaluations are now drained in the main process as workers finish, immediately written to `all_results.bin` / Pareto storage, and stripped from the generation payload before pymoo continues. This improves progress visibility and avoids retaining full metrics payloads until the entire generation completes.
- **Optimizer multiprocessing now works under the unified CLI on spawn-based platforms** - `passivbot optimize ...` no longer fails at pool startup with a pickling error for the SIGINT worker initializer when launched through the unified CLI on macOS/Python spawn multiprocessing.
- **CLI now guards against wrong-environment `passivbot` launches** - When `VIRTUAL_ENV` or `CONDA_PREFIX` is active but the resolved `passivbot` command is running under a different Python interpreter, the console entrypoint now re-execs into the active environment's `passivbot` script when available, or fails loudly with explicit mismatch diagnostics and install guidance instead of silently running the wrong install.
- **Startup exchange-config timeout handling** - Live startup now gives CCXT exchange sessions a 30s default timeout and retries `update_exchange_config()` on transient network/request timeouts during `init_markets()`, reducing cold-boot failures without suppressing non-retryable errors.
- **Hyperliquid HIP-3 margin-mode detection for `XYZ-...` symbols** - Hyperliquid stock perps exposed by CCXT as `XYZ-...` or `XYZ:...` now correctly force isolated margin mode, preventing erroneous cross-margin config calls that could lead to repeated duplicate entry submissions on stock-perp markets such as `XYZ100`.
- **Hyperliquid HIP-3 state sync for positions and open orders** - Hyperliquid stock-perp positions and open orders now use dex-scoped CCXT queries for HIP-3 symbols instead of relying only on the default `fetch_balance()` / global open-orders routes. This fixes bots repeatedly re-entering because filled HIP-3 positions or resting HIP-3 orders were invisible to local state reconciliation.
- **Hyperliquid HIP-3 isolated trading disabled for now** - Passivbot now treats Hyperliquid HIP-3 as cross-only for live trading until isolated-margin support is properly designed. Cross-capable HIP-3 markets remain tradable in cross mode, isolated-only HIP-3 markets are skipped with warnings, and existing isolated HIP-3 positions or open orders fail startup loudly instead of running in a risky partial-support mode.
- **Stock-perp source-dir resolution in HLCV preparation** - Hyperliquid stock-perp backtests now resolve source-dir symbols against loaded market metadata instead of failing on cache-map casing mismatches such as `xyz:AAPL` vs `XYZ-AAPL/USDC:USDC`.
- **Editable-install Rust freshness checks now find `maturin develop` outputs reliably** - The stale-extension safety check now detects root-level `site-packages/passivbot_rust...so` installs created by `maturin develop`, so `passivbot ...` no longer loops on “stale even after recompilation” while still using an old `src/passivbot_rust...so` shadow copy.
- **Backtest HSL panic execution and metrics export** - Account-level RED panic now forces panic mode on all symbols/sides in Rust backtests, `panic_close_order_type="market"` is simulated as next-bar taker execution instead of limit-only behavior, and `hard_stop_*` analysis metrics are exported once as shared metrics rather than duplicated into `_usd`/`_btc` variants.
- **Rust-owned market-vs-limit execution intent** - Rust orchestrator now decides whether eligible non-panic orders should be emitted as `limit` or `market` using one shared near-touch threshold and market-crossing rules. Live now consumes that Rust execution intent directly, and backtests use the same intent for guaranteed market fills with slippage and taker fees.
- **Backtest taker-fee execution for market fills** - Backtest market executions now charge taker fees instead of maker fees, respect optional `backtest.taker_fee_override`, and record a `liquidity` column (`maker` / `taker`) in `fills.csv`. Simulated market fills remain guaranteed once selected, with execution price shifted by `backtest.panic_market_slippage_pct`.
- **Backtest HSL drawdown visualization** - Backtests now output `hard_stop_drawdown.png` alongside the existing summary plots when account-level HSL is enabled. The new plot shows raw drawdown, EMA-smoothed drawdown, the active HSL trigger score, tier thresholds, and RED-threshold proximity over time. `--disable_plotting` also supports a dedicated `hard_stop` plot group.
- **Backtest HSL EMA span fallback** - Backtests no longer fail when `bot.common.equity_hard_stop_loss.ema_span_minutes` is smaller than `backtest.candle_interval_minutes`. Sub-interval spans now fall back to a one-sample EMA, which disables smoothing and makes the HSL score follow raw drawdown.
- **HSL no-restart threshold semantics** - Values of `bot.common.equity_hard_stop_loss.no_restart_drawdown_threshold` below `red_threshold` are now clamped up to `red_threshold` in live, backtest, and optimizer flows. Stop events now treat `drawdown_raw >= no_restart_drawdown_threshold` as terminal, so setting both thresholds equal makes the first RED halt non-restarting.
- **Backtest HSL analysis metrics expanded and clarified** - Added account-level HSL metrics for yellow/orange/red time share, RED halt duration, trigger drawdown, panic-close realized loss, flatten time, and restart-to-retrigger rate. Also renamed the old ambiguous halt-loss metric to `hard_stop_halt_to_restart_equity_loss_pct`.
- **HLCV fetch logging and cache-root hygiene** - CCXT candle fetch progress logs now include the actual returned candle range (`first`/`last`) instead of only the requested `since`, and CandlestickManager now quarantines invalid root-level daily shard files or `index.json` debris found directly under `caches/ohlcv/{exchange}/{timeframe}` so mixed/corrupt cache roots stop masquerading as symbol data.

## v7.8.4 - 2026-03-06

### Changed
- **Dual balance routing (raw vs hysteresis-snapped)** - Live and orchestrator flows now carry both `balance_raw` (raw wallet balance) and `balance` (hysteresis-snapped balance). Sizing/order-shaping paths use snapped balance, while risk/accounting paths use raw balance (including realized-loss gate peak/floor checks, TWEL entry/auto-reduce gating, and auto-unstuck allowance calculations). This applies consistently across live and backtest via Rust orchestrator input.
- **WEL denominator behavior split by mode** - Live now uses a hard fixed denominator for per-symbol WEL (`total_wallet_exposure_limit / config.bot.{pside}.n_positions`), removing runtime denominator drift from open-position count. Backtests now expose `backtest.dynamic_wel_by_tradability` (default `true`): when enabled, WEL uses tradability-aware denominator growth (`min(n_positions, n_tradable_max)`) based on coins with real candles, and does not shrink after delistings; when disabled, backtests use the same fixed denominator as live.
- **Bulk price fetch for Hyperliquid** - `calc_ideal_orders` now uses a single `allMids` API call to get prices for all symbols instead of individual `get_current_close` calls per symbol (1 call vs ~70). Falls back to per-symbol fetches for non-Hyperliquid exchanges or on error.
- **Sequential margin mode setting for Hyperliquid** - Margin mode and leverage API calls are now sequential with a small delay instead of being fired in parallel, reducing API burst on coin changes.
- **Equity hard-stop framework (live+backtest)** - Added nested equity hard-stop config (now under `bot.common.equity_hard_stop_loss`) with threshold, EMA span in minutes, configurable yellow/orange tier ratios, orange mode selector, panic close order type, plus Rust drawdown/tier state machine module, backtest rolling-peak enforcement using `pnls_max_lookback_days`, and live runtime hooks for tier tracking/latching with RED supervisory flatten-until-confirmed-flat behavior.

### Fixed
- **Bybit fill-event qty inflation on duplicate pages** - `BybitFetcher` now deduplicates `fetch_my_trades` rows by exec id before canonicalization/coalescing, preventing duplicate pagination rows from inflating canonical `qty`, `fees`, and close PnL.
- **Balance peak drift in wrong direction under hysteresis** - Peak reconstruction (`balance + (pnl_cumsum_max - pnl_cumsum_last)`) previously used hysteresis-snapped balance in some paths. Since snapped balance can stay stale while `pnl_cumsum_last` changes fill-by-fill, this made reconstructed peak drift down after profits and up after losses. Peak/PnL-accuracy-sensitive paths now use raw balance (`balance_raw`) consistently.
- **Pytest Rust-module bootstrap fallback** - Test bootstrap now tries the project venv `passivbot_rust` package before falling back to the lightweight stub when tests are launched outside the venv, reducing false failures from missing/incorrect Rust module resolution.
- **`max_ohlcv_fetches_per_minute` ignored when forager slots open** - The rate limit config was only applied when all position slots were full. With open slots (the common case), all candidate symbols were fetched without rate limiting, causing 429 errors on Hyperliquid.
- **Hyperliquid positions+balance double fetch** - `fetch_positions` and `fetch_balance` now share a single API call via a dedup lock instead of making two identical `clearinghouseState` requests per execution cycle.
- **Thundering herd on minute boundary** - `get_candles` no longer force-refreshes all symbols simultaneously when a new minute boundary crosses. A 1-candle staleness tolerance prevents the TTL override that caused all symbols to fetch at once.
- **Candle refresh TTLs aligned to 1-minute finalization** - Active candle refresh TTL raised from 10s to 60s and EMA close TTL from 30s to 60s, matching the actual 1-minute candle finalization interval.
- **Boot stagger for multi-bot setups** - Added `boot_stagger_seconds` config (default 30s for Hyperliquid) to randomize startup delay, preventing simultaneous API bursts when multiple bots share the same IP.
- **Warmup and refresh fetch pacing** - Added configurable `warmup_fetch_delay_ms` (default 200ms for Hyperliquid) with delays between individual symbol fetches during warmup, forager refresh, and active candle refresh loops.
- **Exponential backoff on 429 errors** - WebSocket `watch_orders` uses exponential backoff (up to 30s) on rate limit errors. Execution loop backs off 5s on `RateLimitExceeded`. Hourly `init_markets` catches rate limits with 10s recovery.
- **Fill events pagination abort on repeated rate limits** - `HyperliquidFetcher` now aborts after 5 consecutive rate limit retries with exponential backoff instead of retrying indefinitely.
- **EMA bundle and active candle sweep abort on rate limit** - Both `_load_orchestrator_ema_bundle` and `update_ohlcvs_1m_for_actives` skip remaining symbols when the CandlestickManager's global rate limit backoff is active.
- **Live close-EMA failure handling in orchestrator feed** - `_load_orchestrator_ema_bundle()` no longer silently drops failed/non-finite close EMA spans. It now fails loudly when no prior EMA exists, and otherwise reuses the last successfully computed close EMA for that exact symbol/span with explicit `[ema]` warning logs (including reason, age, and consecutive fallback count).
- **Required 1h log-range EMA handling in orchestrator feed** - `_load_orchestrator_ema_bundle()` now fails loudly when required `h1` log-range spans (from `entry_volatility_ema_span_hours`) are missing or non-finite, instead of deferring to downstream Rust `MissingEma` errors.
- **EMA bundle fetch stability under lock contention** - Orchestrator EMA bundle loading now fetches per-symbol spans serially and drains all symbol task outcomes before re-raising, reducing same-symbol candle-lock contention and eliminating unretrieved sibling-task exception noise.

### Added
- **Fill events doctor tool** - Added `src/tools/fill_events_doctor.py` to audit cached fill events and auto-repair known Bybit duplicate-fill anomalies without requiring exchange API calls. Bybit startup now runs doctor by default (can be disabled with `PASSIVBOT_FILL_EVENTS_DOCTOR=off`).

## v7.8.3 - 2026-02-24

### Added
- **Global realized-loss gate for close orders** - Added `live.max_realized_loss_pct` (default `0.05`) to block any close order (including WEL/TWEL auto-reduce and unstuck) that would realize losses beyond a peak-balance-relative threshold. Panic closes remain exempt. Live bot now emits `[risk]` warnings when orders are blocked by this gate.

### Fixed
- **False-positive stale Rust extension after identical rebuild** - `sync_installed_extension_into_src()` now updates the local `src/` `.so` mtime when its content (SHA256) already matches the installed site-packages build. Previously the old mtime was preserved, causing `check_and_maybe_compile` to report the extension as stale in a loop even though the binary was current.
- **Peak recovery hours PnL metric** - `peak_recovery_hours_pnl` now computes directly from fill events using gross PnL with strict peak detection (`>` instead of `>=`), instead of reconstructing a cumulative series over the equity index. Fixes inaccurate recovery times when fills were sparse relative to the equity series.
- **Combined OHLCV normalization source selection** - Volume normalization in combined backtests now uses each coin's OHLCV source exchange (`ohlcv_source`) instead of the market-settings exchange when `backtest.market_settings_sources` differs from OHLCV routing.
- **Config template/format preservation** - Added `live.enable_archive_candle_fetch` to the template defaults and ensured `backtest.market_settings_sources` is preserved during config formatting.
- **Live no-fill minute EMA continuity** - When finalized 1m candles are missing because no trades occurred, live runtime now materializes synthetic zero-candles in memory (not on disk), preventing avoidable `MissingEma` loop errors on illiquid symbols. If real candles arrive later, they overwrite synthetic runtime candles and invalidate EMA cache automatically.
- **Suite base scenario inherited all scenario coins** - Scenarios without explicit `coins` (e.g. the `"base"` scenario) fell back to `master_coins` — the union of every scenario's coin list — instead of the original `approved_coins` from the config. Now `apply_scenario` falls back to `base_coins` (the config's `approved_coins`) when a scenario omits its own coin list.
- **Aggregate methods ignored in optimizer scoring and Pareto analysis** - `calc_fitness` always looked up the `_mean` stat for every scoring metric, ignoring the `backtest.aggregate` config (e.g. `"high_exposure_hours_max_long": "max"`). The optimizer now overrides `flat_stats` with correctly aggregated values before computing objectives. The standalone `pareto_store.py` script reads the aggregate config for suite-metric extraction and limit filtering while leaving stored objectives unchanged.
- **Backtest HLCV cache reuse across configs** - Configs that differ only in trading parameters (EMA spans, warmup ratio) now share the same HLCV cache slot. Previously, different EMA spans produced different `warmup_minutes`, which was included in the cache hash, causing unnecessary re-downloads. The cache now uses a ratchet-up strategy: warmup sufficiency is checked at load time, and the cache is overwritten only when a larger warmup is needed.
- **Backtest cache warmup downgrade guard** - Cache saves now keep the highest recorded `warmup_minutes` for a cache slot and skip writes that would downgrade it, reducing refetch churn when multiple runs touch the same cache concurrently.

## v7.8.2 - 2026-02-09

### Added
- **Configurable candle interval** - New `backtest.candle_interval_minutes` setting (default 1) aggregates 1m candles to coarser intervals (e.g., 5m) for faster backtests and optimizer iterations. EMA alphas are automatically adjusted for the interval. Trade-off: intra-interval fill ordering is lost.
- **High-exposure duration metrics** - New backtest metrics `high_exposure_hours_{mean,max}_{long,short}` measuring continuous durations where total wallet exposure exceeded its daily average. Available for optimization scoring and limit checks.
- **Total wallet exposure plot** - Backtests now output `total_wallet_exposure.png` showing long TWE (positive, blue) and short TWE (negative, red) over time.
- **External OHLCV source dir** - New `backtest.ohlcv_source_dir` config option to load 1m candle data from a pre-populated directory tree before falling back to exchange archives. Supports both `.npy` and `.npz` file formats.

### Fixed
- **OHLCV source-dir fallback behavior** - Non-contiguous source-dir candle data now falls back to CandlestickManager instead of propagating gappy series into downstream strict continuity checks.

### Fixed
- **Short-only exposure metrics** - `total_wallet_exposure_max` and related metrics now use absolute values, correctly reporting exposure magnitude for short-only configs where `twe_net` is negative.
- **Timestamp day bucketing** - Backtest analysis now initializes daily bucketing from the first timestamp, preventing a phantom first-day sample when using aggregated candle intervals.
- **Forager fills plots with aggregated candles** - `fills_plots` now use the effective candle stream from the executed backtest, keeping fills aligned when `backtest.candle_interval_minutes > 1`.

### Changed
- **Template config tuning** - Updated `configs/template.json` optimization bounds/scenarios and backtest defaults (`btc_collateral_cap`, `maker_fee_override`, optimize limits).

## v7.8.1 - 2026-02-07

### Fixed
- **Gate.io cache cutoff** - Set `GATEIO_CACHE_CUTOFF_DATE` to 2026-02-07 so stale Gate.io caches are quarantined on startup.

## v7.8.0 - 2026-02-07

### Fixed
- **Live bot candle cache** - Rebuilds candlestick index metadata for the required warmup ranges on startup, preventing stale `index.json` metadata from suppressing candle refreshes.
- **Windows backtest startup** - Avoids importing `resource` at module load, preventing crashes on Windows during backtest/optimizer startup.
- **Legacy cache migration** - Migration now runs once globally and covers all exchanges on first init (not just the first exchange to start), and legacy data is resolved relative to the cache root to avoid unintended copies.
- **Combined OHLCV selection** - `market_settings_sources` no longer expands OHLCV candidates; combined data now uses `backtest.exchanges` plus forced coin sources only.

### Changed
- **Logging** - Reduced INFO/WARNING noise (unsupported market notices now INFO with `[config]`, hedge-mode success logs moved to DEBUG, Bitget OHLCV limit probes moved to DEBUG, KuCoin PnL discrepancy warnings further throttled, large zero-candle warnings now only trigger above 1000). Added `[order]` tag to order plan summaries and extra context for MissingEma errors.

## v7.7.1 - 2026-02-07

### Added
- **Stock perps (HIP-3) support** - Hyperliquid stock perpetuals are now supported, including symbol normalization and routing in combined mode.
- **Pareto host** - Added a lightweight host mode for serving Pareto outputs.

### Fixed
- **Combined HLCV prep** - Fixed `orig_coins` NameError during combined data preparation.

### Changed
- **Logging refinements** - Further reduced INFO noise and improved context across rounds 8–10.
- **Agent docs** - Updated guidance and pitfalls documentation for cross-platform portability.

## v7.7.0 - 2026-01-26

### Fixed
- **Bybit: Missing PnL on some close fills** - Fixed pagination bug in `BybitFetcher._fetch_positions_history()` that caused closed-pnl records to be skipped when >100 records existed in a time window. Now uses hybrid pagination: cursor-based for recent records (no gaps), time-based sliding window for older records.

### Added
- **Fill events now include psize/pprice** - Each fill event is annotated with position size (`psize`) and VWAP entry price (`pprice`) after the fill. Values are computed using a two-phase algorithm and persisted to cache for all exchanges.
- **Logging best practices documentation** - New `docs/ai/log_analysis_prompt.md` with comprehensive logging guidelines, level definitions, and improvement tracking.
- **Exchange API quirks documentation** - New `docs/ai/exchange_api_quirks.md` documenting known exchange-specific limitations and workarounds.
- **Debugging case studies** - New `docs/ai/debugging_case_studies.md` with detailed debugging sessions as reference.

### Changed
- **Logging improvements (7 rounds of refinement)**:
  - Standardized log tags: `[memory]`, `[warmup]`, `[hourly]`, `[fills]`, `[mapping]`, `[candle]`, `[ranking]`, `[mode]`
  - Moved routine API/cache messages from INFO to DEBUG level (CCXT fetch details, cache updates)
  - Moved CCXT API payloads from DEBUG to TRACE level
  - EMA ranking logs now throttled to every 5 minutes (was every cycle)
  - Mode changes throttled to 2 minutes per symbol (reduces forager oscillation noise)
  - KucoinFetcher PnL discrepancy warnings throttled to 1 hour with delta-based deduplication
  - WebSocket reconnection now logs explicit `[ws] reconnecting...` messages
  - Strict mode gaps changed from WARNING to DEBUG (expected for illiquid markets)
  - Persistent gaps changed from WARNING to INFO with throttling
  - Zero-candle synthesis warnings aggregated and rate-limited
- **PnL tracking now uses FillEventsManager exclusively** - Legacy `update_pnls` path removed. FillEventsManager provides more accurate fill tracking with proper event deduplication, canonical schemas, and exchange-specific fetchers for all supported exchanges.
- Fill events are now stored in `caches/fill_events/{exchange}/{user}/` instead of the old `caches/{exchange}/{user}_pnls.json` format. Existing legacy cache files are ignored; FillEventsManager will rebuild from exchange API on first run.
- Unstuck allowances now computed from FillEventsManager data instead of legacy pnls list.
- Trailing position change timestamps now derived from FillEventsManager events.

### Removed
- `--shadow-mode` CLI flag (no longer needed; FillEventsManager is production-ready)
- `live.pnls_manager_shadow_mode` config option
- Legacy `init_pnls`, `update_pnls`, `fetch_pnls` methods in passivbot.py
- Legacy `init_fill_events`, `update_fill_events`, `fetch_fill_events` methods (dead code)
- Shadow mode comparison logging (`_compare_pnls_shadow`, etc.)

### Migration Notes
- **No action required** - FillEventsManager automatically fetches and caches fill data
- Old `{user}_pnls.json` cache files can be safely deleted after upgrading
- If using custom exchange configurations, ensure the exchange's fill fetcher is supported (Binance, Bybit, Bitget, GateIO, Hyperliquid, KuCoin, OKX)

## v7.6.2 - 2026-01-20

### Fixed
- One-way mode now respects disabled sides when choosing initial entry side, preventing a disabled side from blocking entries.
- Startup banner now dynamically calculates width to prevent misaligned borders.
- Bybit leverage/margin mode "not modified" errors now handled gracefully instead of logging full tracebacks.
- Large warmup spans (>2 days) now properly trigger gap-filling via CCXT even when end_ts touches present, fixing issue where thousands of zero-candles were synthesized for historical gaps.
- Windows compatibility: cache folder names now replace `:` with `_` on Windows or when `WINDOWS_COMPATIBILITY=1` env var is set (#547, thanks @FelixJongleur42). **Note:** Existing Windows caches will be orphaned and re-downloaded.
- Pareto dashboard: fixed JavaScript callback errors when switching between tabs (#550, thanks @646826).

### Changed
- Config modification logs now prefixed with `[config]` for easier filtering (e.g., `[config] changed live.user bybit_01 -> gateio_01`).
- Zero-candle synthesis logs are now rate-limited to at most once per minute per symbol, reducing log spam.
- Zero-candle logs now include human-readable UTC timestamps showing which candles were synthesized (e.g., `synthesized 3 zero-candles at 2026-01-19T22:15 to 2026-01-19T22:17`).
- Synthetic candles are now tracked at runtime; when real data arrives for a previously-synthetic timestamp, the EMA cache is automatically invalidated and will be recomputed on next cycle.
- FillEventsManager logs now prefixed with `[fills]` for easier filtering; verbose refresh logs consolidated into single summary line (e.g., `[fills] refresh: events=1311 (+1) | persisted 2 days (2026-01-19, 2026-01-20)`).
- BybitFetcher residual PnL warnings reduced to debug level with compact summary (was logging all order IDs every cycle at WARNING level).
- Health summary now includes realized PnL sum when fills > 0 (e.g., `fills=3 (pnl=+12.50)`).
- Startup banner now shows "TWEL" (Total Wallet Exposure Limit) instead of "Exposure" to clarify it's a limit, not current exposure; long+short mode shows both limits (e.g., `TWEL: L:125% S:85%`).
- Synthetic candle replacement logs now prefixed with `[candle]` for easier filtering.

### Added
- `openpyxl` added to `requirements-live.txt` (required for Bitget archive XLSX parsing).
- `CandlestickManager.needs_ema_recompute(symbol)`: check if EMAs should be recomputed due to synthetic→real data replacement.
- `CandlestickManager.clear_synthetic_tracking(symbol)`: clear synthetic timestamp tracking after warmup completes.
- `live.warmup_jitter_seconds` (default 30): random delay before warmup to prevent API rate limit storms when multiple bots start simultaneously.
- `live.max_concurrent_api_requests` (default null): optional global concurrency limit for CCXT API calls via CandlestickManager's network semaphore.
- `backtest.maker_fee_override` (default null): optional backtest/optimizer maker fee override (part-per-one) to replace exchange-derived fees.
- `live.enable_archive_candle_fetch` (default false): opt-in to use exchange archive data for candle fetching in live bots; disabled by default to avoid potential timeout issues. Backtester always enables archive fetching regardless of this setting.

## v7.6.1 - 2026-01-03

### Testing
- Added comprehensive test coverage for HLCV preparation module (16 tests covering 1,017 lines of production code)
- Added comprehensive orchestrator integration tests (19 tests for order accuracy, edge cases, multi-symbol coordination)
- Added warmup utilities test coverage (20 tests for EMA warmup calculations and edge cases)
- Improved Rust stub in conftest.py with correct parameter signatures and orchestrator JSON API support
- Total: 55 new tests, bringing test suite from ~420 to 477 passing tests

## v7.6.0 - 2026-01-03

### Added
- Shared Pareto core (`pareto_core.py`) with constraint-aware dominance, crowding, and extreme-preserving pruning; reused by ParetoStore.
- Canonical suite metrics payload now shared by backtest and optimizer; suite summaries include the same schema as Pareto members.
- Targeted Pareto tests to ensure consistency.
- KuCoin exchange-config regression tests covering hedge-mode setup and leverage/margin configuration (guards CCXT upgrades).
- Pareto explorer: added configurable “Closest config metrics” dropdown so users can choose which metrics are shown in the Closest Config table, defaulting to scoring/limit metrics.
- `live.balance_override` setting/CLI flag to pin balance to a fixed value instead of fetching from the exchange (off by default).
- Fill events manager: added Gate.io support via ccxt trade fetcher.
- Rust build pipeline: pre-import staleness checks with skip/force/fail flags, shared helpers, and a `scripts/check_rust_extension.py` reporter; added tests for staleness detection.
- Rust compile flow now less noisy in normal operation (debug lock prints removed); compile attempts still logged when rebuilding.
- Balance hysteresis now applied centrally in core bot update_balance; exchange fetch_balance implementations return raw balances.
- Added configurable `live.balance_hysteresis_snap_pct` (default 0.02); set 0.0 to disable balance hysteresis entirely.
- Optimizer: bounds now support optional step size `[low, high, step]` for grid-based optimization; stepped parameters stay on-grid through sampling, crossover/mutation, and Pareto storage.
- Live: added `live.candle_lock_timeout_seconds` to control how long CandlestickManager waits for per-symbol candle locks when multiple bot instances share the same cache (default 10s).
- Rust orchestrator JSON API for unified order planning across live and backtest.
- Backtest HLCV preparation pipeline now routes through CandlestickManager with shared warmup utilities.

### Changed
- Backtest fills now include signed `wallet_exposure` and `twe_long`/`twe_short`/`twe_net` (replacing the previous `total_wallet_exposure` fill column).
- Pareto explorer: default metrics for X/Y/histogram, scenario comparison, param scatter, correlation heatmap, and Closest Config now derive from `config.optimize.scoring` and `config.optimize.limits` instead of first-alphabetical metrics; Closest Config table no longer shows raw *_mean/_min/_max/_std stat columns by default.
- Suite summaries are leaner: redundant metric dumps removed; canonical metrics schema persisted alongside per-scenario timing.
- Pareto pruning preserves per-objective extremes when enforcing max size.
- Hyperliquid combined balance/position caching test isolated stubs to avoid polluting the rest of the suite.
- Separated `fetch_positions` and `fetch_balance` responsibilities across all exchange wrappers (each now returns only positions or only balance) and added `update_positions_and_balance()` helper in the core bot to refresh both concurrently.
- `update_positions_and_balance()` now runs balance and positions concurrently, logs position changes after both complete, and then emits a single balance-change event so equity logging always uses fresh positions.
- KuCoin `get_order_execution_params` now aligns with the latest CCXT payload requirements so orders always include the correct margin/position parameters after the CCXT upgrade.
- Added Pareto regression test to ensure per-metric extremes remain present after front pruning.
- Metric adg_pnl now includes fees paid, effectively making it net pnl instead of gross pnl.
- Risk management docs refreshed and consolidated; new notes on unstucking, WEL/TWEL enforcers, and conditional stop-loss concepts.
- Balance updates now keep the previous value on fetch failures (no more transient zero balances); warnings are logged and the standard restart-on-errors flow handles persistent issues.
- EMA log spam reduced: volume/log-range EMA summaries only emit when rankings change, keeping live logs quieter.
- Suite configuration is canonical under `backtest.suite` for both backtesting and optimizer runs; `optimize.suite` (if present) is ignored and removed during config normalization.
- Live orchestrator compare mode now derives all EMA inputs from a single per-symbol candle snapshot (1m + 1h), reducing redundant candle-lock contention and false compare failures in multi-bot deployments.
- Live order generation now runs exclusively through the Rust orchestrator; legacy Python order planning paths are removed.
