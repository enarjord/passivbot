# Development history after v8.1.0

Historical detail for changes after the v8.1.0 tag through
[be47aed12](https://github.com/enarjord/passivbot/commit/be47aed12).
These entries were consolidated into the [Unreleased changelog](https://github.com/enarjord/passivbot/blob/master/CHANGELOG.md#unreleased).
They describe successive implementation steps; later entries can supersede earlier restrictions
or defaults. For current capabilities and upgrade requirements, use the changelog and the
[optimizer guide](optimizing.md), [configuration guide](configuration.md), and
[release status](releases.md). This is an archived ledger, not a published release or an active
place to append new changes.

## Detailed entries (newest first)

- GPU optimization uses bounded temporal replay for long-history, single-coin Trailing Martingale
  with both sides enabled, allowing more concurrent candidates while preserving replay state,
  per-dispatch work limits, and interruption checkpoints.

- GPU optimization reuses full-history screened seed metrics in the initial population, including
  after checkpoint resume, avoiding duplicate replay while preserving exact validation. Small
  two-sided MPS batches use unchunked replay when they fit the dispatch work limit.

- Risk-input recovery now stops after `live.risk_input_max_attempts` failed attempts (default 10)
  per recovery episode, without entering the full-bot restart loop. Each failed attempt logs its
  count, limit, safe balance diagnostics, and retry delay; the first and final failures include
  bounded tracebacks. Changing failure reasons or passing an early check does not renew the budget.
  Ordinary trading resumes after successful recovery; valid HSL protection remains available while
  waiting. Invalid balances are never substituted and required history is never discarded.

- Trailing-martingale price EMA spans now live at
  `bot.<side>.strategy.trailing_martingale.entry.ema_span_0/1` (schema v8.4.0).
  Existing configs, bounds, coin files/inline overrides, and scenario paths migrate without
  changing their horizons. Explicit new paths win on conflicts, with warnings. Previous CLI
  paths and optimizer selectors remain accepted. Entry-only optimizer selectors now include
  these spans; coupled unstuck search, CPU/GPU execution, and warmup use the new paths.
  Shared volatility spans and other strategies keep their existing locations.

- Coin HSL no longer clears live protection or repeatedly reconstructs history when a delayed
  ordinary flatten falls before the bounded replay window. Successful empty-window replays now
  establish the proven episode boundary before calculating current risk, preventing discarded
  losses from triggering RED while retaining re-entry fees (including validated same-millisecond
  re-entries) and required cooldown/no-restart history.
- Add `couple_unstuck_ema_spans` to `optimize.enable_overrides` to search strategy and unstuck
  EMA spans together on CPU or Apple MPS, including effective coin/scenario strategy overrides.
  Coupled search omits redundant unstuck span genes and saves explicit horizons for ordinary
  replay. Independent search remains the default; changing the option requires a fresh run.

- Auto-unstuck now owns independent `bot.<side>.unstuck.ema_span_0` / `ema_span_1`, including
  coin overrides and CPU optimizer bounds. Schema v8.3.0 migrates missing spans from each coin's
  effective strategy to preserve saved trading behavior, with warnings where exact migration is
  impossible (including the formerly coupled optimizer search). Hard-coded defaults and the default
  example include the new spans. Active-gate warmup, backtest EMA updates, and independent monitor
  triggers use them. Missing live unstuck-only EMAs defer unstucking while preserving strategy inputs.
  Apple MPS screening supports the independent spans across both supported strategies, including
  coin overrides, aggregated candles, and chunked replay. Start a fresh GPU optimizer run after
  upgrading; older screening checkpoints use an incompatible parameter layout.

- GPU optimization automatically queues two validation batches (or twice the worker count,
  whichever is larger), allowing proxy screening and exact evaluation to overlap. Explicit queue
  sizes remain unchanged. Drift calibration now compares unpenalized objective values, keeping
  constraint penalties from obscuring disagreement. The fill-gap time-weighted mean uses streamed
  squared gaps instead of histogram bounds. Existing GPU checkpoints require a fresh run.
- Pymoo mutation exposes separate `mutation_prob` (per individual) and
  `mutation_prob_per_variable` controls on CPU and GPU. Both default to `"auto"`, preserving
  historical mutation intensity. The old, misleading `mutation_prob_var` config key migrates to
  `mutation_prob`; existing numeric settings keep their effective meaning.

- KuCoin partial closes now contribute realized losses to auto-unstuck and other PnL
  risk inputs before the whole position closes. Existing trade-derived closes mislabeled
  as authoritative are automatically backed up and reconstructed on cache load; the
  fill-events doctor can also inspect or repair them. Reconciled position-history PnL
  is preserved, and incomplete reconstruction remains unavailable to PnL risk checks.

- KuCoin market-age discovery uses a valid millisecond history range, fixing
  `Parameter 'from' must be milliseconds` errors and restoring eligibility for
  new forager entries when the configured minimum market age is met.

- Live startup skips unavailable coin overrides with a bounded notice and retries resolution on
  market refresh. Inactive-market overrides and protection of existing positions remain intact;
  an empty eligible market set no longer admits unsupported approved coins.
- Unexpected failures that abort a bot run show bounded traceback frames, the failing phase,
  and stop/restart action at normal console logging levels, including failures before bot
  construction. Known missing market keys are shown
  without exposing raw exception payloads or locals. Repeated startup failures share traceback
  suppression across restarts, with periodic counts and a recovery notice after successful startup.

- Retain Bitget Classic and UTA fills without client order IDs when fetching mixed
  bot and external execution history. Previously, omitted external closes could
  corrupt reconstructed positions/PnL and leave trailing confirmation pending.
  Existing incomplete caches require a history refresh covering the omitted fills.

- Restored `-ltwel`/`-stwel` and `-lnp`/`-snp` CLI shortcuts for long/short total wallet
  exposure limits and position counts in live and backtest commands. They now target the grouped
  `bot.<side>.risk` fields, take precedence over supported flat config aliases, and appear in
  command help.

- GPU suite optimization shares identical prepared MPS market tensors across scenarios, reducing
  memory use and repeated preparation. Long-history single-side Trailing Martingale replays use
  smaller threadgroups for higher throughput while retaining bounded, interruptible dispatches.

- Apple MPS multi-coin Trailing Martingale optimization uses bounded history chunks when long
  datasets would otherwise restrict candidate parallelism. Replays preserve their full strategy
  and metric state between dispatches, retain the configured work limit, and check interruption
  between chunks. Profiling reports temporal dispatch counts and maximum dispatch duration.

- `compose-coin-overrides --override-params` pins selected groups or leaves using fine-tune-style
  dotted selectors, retaining equal values while unselected fields inherit the master. Custom
  selection takes precedence over lean/verbose mode and rejects selectors matching no allowed
  input fields.

- `compose-coin-overrides --override-mode verbose` pins all allowed per-coin values, including
  values equal to the master and inactive-feature settings, so later master edits do not change
  those values. The default `lean` mode retains minimal patches. Composition no longer mutates
  caller-owned configs, rejects overwriting the selected master or a single-coin input, and publishes output
  atomically with protection against concurrent file creation. Reports now show position-count
  changes made during composition.

- `compose-coin-overrides --master-config` accepts an external JSON/HJSON master, including a
  multi-coin baseline. The input directory determines the composed coin set; the master supplies
  global values and the baseline for generated overrides.

- Added `position_held_time_weighted_mean_hours` to backtest analysis and CPU/GPU optimizer
  scoring: a duration-weighted mean of per-coin/side holding episodes, including still-open
  positions. Minimize it to penalize long holds using the full duration distribution.

- Added rolling-harmonic ADG, time-integrated ADG, and positive-gain-participation strategy-equity
  metrics to exact backtest analysis and CPU/GPU optimizer scoring.

- `compose-coin-overrides --include-backtest-optimize` now retains the master's GPU optimizer
  settings instead of rejecting multi-coin configs with static coin overrides. GPU-specific
  compatibility checks still run when starting the optimizer.

- Backtest and optimizer results now expose `n_days` and effective UTC analysis dates in
  existing metrics payloads, preserving per-exchange and per-scenario windows. Saved backtest
  configs include current `metrics` or `suite_metrics`; `fills_analysis_duration_days` remains
  available as a compatibility alias with the same exact/CPU scoring, suite reducer, and GPU
  exact-only policy. Both duration names default to maximizing duration in shorthand scoring.
  Standalone artifacts preserve non-finite diagnostics separately from finite metric statistics.
  Saved suite configs retain the effective exchange defaults from external suite overrides.

- Concurrent OHLCV writers now lock monthly initialization through catalog publication, preserving
  both candle bodies and validity masks. Materialized scratch allocation and pruning use a persistent
  advisory lock, preventing simultaneous owners while owner metadata is being initialized.
  Stop older backtest/optimizer workers before sharing their cache with the new lock protocol;
  ambiguous legacy locks require explicit cleanup after those workers have stopped.

- Backtests reject non-finite H/L/C inside declared valid candle windows, including all-NaN
  gaps in direct or prepared datasets. Held positions require an available valuation candle;
  missing data no longer removes their unrealized PnL from equity or risk samples. Unheld
  symbols may retain unavailable rows outside their listing windows.

- Optimizer resume validates fixed policy, resolved coin and scenario overrides, prepared candle
  and market-setting content, transitive evaluator Python sources/dependency versions, and the
  verified Rust source and binary
  before reusing fitness. Every reconstructed result and GPU seed checkpoint carries matching
  evidence; legacy results without historical identities require a fresh run. Candidate values
  and CPU worker counts may vary, while complete GPU settings and rebuilt Rust binaries remain
  strict resume inputs. Suite dates and override files are resolved before evaluation and recording.

- Gate.io and KuCoin fill-history refreshes now reject unfinished pagination instead of marking
  partial or failed fetches as complete coverage for realized-PnL risk checks. A traversal that
  completes on its final allowed request remains valid.

- OKX now reads all pending-order pages before reconciling the account, preventing duplicate
  orders when more than 100 orders are open. Failed or stalled pagination rejects the snapshot.

- External NumPy candle files no longer permit pickle objects, and local archive extraction
  explicitly filters unsafe paths and links on every supported Python version.

- Fill accounting now preserves explicitly reported zero fees and fully resolved zero-sum fee
  lists while retaining missing or malformed amounts for fallback accounting, including after
  fill coalescing and reload of older cached fee estimates. Non-quote fee conversion refreshes
  expired ticker quotes and retries temporary lookup failures without discarding a fresh quote
  because another fill cannot use it. Legacy Hyperliquid aggregates with unknown component fees
  require cache repair instead of silently omitting those fees from reconciliation.

- The monitor relay rejects foreign or malformed browser WebSocket origins before reading or
  sending account snapshots, while preserving the dashboard and originless native clients.

- Wrapped config documents now strip persisted `hsl_accept_incomplete_history` overrides before
  normalization and apply CLI overrides to the wrapped payload, so the HSL coverage waiver remains
  an explicit per-run CLI choice after reload.

- Live order creation now rechecks account invalidation after exchange configuration and quote
  refreshes and immediately before each connector call, preventing orders based on a superseded
  plan when a private fill update arrives. Partial-fill updates also invalidate account state when
  they match a recently created order.

- Backtest weighted equity metrics now include fill-free trailing windows and carry the last
  actual balance into their equity-versus-balance calculations, preventing open losing positions
  from receiving artificially favorable scores. Equity peak-recovery metrics now include an
  unrecovered drawdown through the final sample.

- Equity Hard Stop Loss now resets RED-free episodes at the fill that flattens the configured
  coin, position-side, or unified scope, including a re-entry in the same minute. Live replay,
  exact backtests, and Apple MPS screening retain closing PnL and fees before resetting episode
  drawdown, while preserving RED-triggered cooldowns and persistent no-restart limits. Closing
  fills that first trigger RED finalize before same-step re-entry. Ambiguous required episode
  evidence defers startup replay and ordinary planning while already-latched RED supervision and
  panic protection for active cooldown positions remain available. Resting cooldown entries are
  cancelled according to the configured policy, preserving orders after a proven manual-policy
  intervention and graceful-stop adds to held positions. Terminal no-restart scopes also cancel
  resting entries. Normal-policy restart overrides and cooldown expiry retain the new episode's
  baseline and entry fees before replaying later exact boundaries; live and restart paths retain
  losses incurred before the next observation. Proven ordinary RED closes and panic closes follow
  the same restart rules, including proven same-millisecond interventions, with terminal no-restart
  protection taking precedence. Coin replay uses the account balance at each boundary, excluding
  later fills and fees on other symbols. Offline fake-live
  scenarios retain complete fill history when their simulated dates differ from wall time.

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

## Release-section correction

The following change landed after the v8.1.0 tag but was previously filed under that release.
It belongs to the post-v8.1.0 development history.

- Resolve plain underlying names across exchange denomination conventions. Prefix forms such as
  `1000SHIB`, suffix forms such as `SHIB1000`, and Hyperliquid's `kSHIB` notation now share one
  denomination-aware identity when established by that venue's market convention. Numeric ticker
  affixes outside a recognized convention remain part of the asset name. A plain coin selects one
  active venue market deterministically, while exact identifiers continue to request a specific
  contract. Combined backtests keep market settings on the OHLCV denomination when an override
  venue uses a different scale.
