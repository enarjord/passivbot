# Changelog

All notable user-facing changes will be documented in this file.

## Unreleased

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
