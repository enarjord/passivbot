# Strategy Runtime Contracts

## Canonical And Legacy Strategy Schemas

The canonical strategy kind is `trailing_martingale`. The released v7 `trailing_grid` schema is not
aliased to it. Do not add compatibility migrations, duplicate strategy names, or silent shims for
removed fields unless the user explicitly asks for released-version compatibility.

Only `trailing_martingale` and `ema_anchor` are supported. The v7 strategy, dedicated migration
command and `pb_multi` compatibility path are removed. Normal loading accepts the final v9
schema only. Released v8 conversion is explicit and reported; conflicting aliases and unknown
fields fail before hydration. Do not infer runtime compatibility from historical release notes.

Removed v7 trailing-grid concepts:

- `entry_trailing_grid_ratio`
- `close_trailing_grid_ratio`
- `close_grid_markup_start`
- `close_grid_markup_end`
- linear `markup_start` to `markup_end` TP grids
- separate entry grid spacing vs trailing threshold knobs

Canonical config path:

```text
bot.<side>.strategy.trailing_martingale
```

Removed v7 config path (rejected by normal loading):

```text
bot.<side>.strategy.trailing_grid_v7
```

Optimizer selector contract:

- `optimize.fixed_params` and `--fine_tune_params` use dotted config-path selectors.
- `long.*` / `short.*` selectors are aliases for `bot.long.*` / `bot.short.*`.
- Selectors match path segments by prefix or suffix, not partial substring. Use `long.strategy`
  to match the whole active strategy subtree, `long.strategy.close` to match only
  `bot.long.strategy.<active_strategy>.close.*`, or a leaf selector such as
  `we_excess_allowance_pct` to match every bound ending with that parameter name.
- `*` is allowed as a one-segment wildcard, for example `*.strategy.close`.
- Authored `fixed_runtime_overrides` and scenario patches use literal full canonical paths;
  optimizer selector shorthand does not authorize alternate configuration spellings.
- Do not use flattened underscore selector names such as `long_entry_*` in current user-facing docs
  or agent instructions.
- When `--fine_tune_params` is combined with `--start`, the starting configs are anchor configs:
  non-tuned optimizer-bound bot params are fixed from the selected anchor, while the fine-tune
  selectors remain tunable. Base-config policy fields, including boolean toggles such as
  `bot.<side>.hsl.enabled`, still win over anchors. Anchor and seed values outside
  `optimize.bounds` are clamped into bounds with aggregated source/key logging. Without
  `--fine_tune_params`, `--start` remains seed-only.

Seed evaluation contract:

- Shared ingestion normalizes, clamps, quantizes and deduplicates starting configs.
  Canonical CPU preparation resolves bounds, anchors, fixed/runtime/scenario overrides and
  effective side-enable views before compact GPU request encoding.
- CPU optimizers evaluate every deduplicated seed with Rust CPU backtests. GPU optimization
  evaluates every deduplicated seed and initial parent with its authoritative CUDA service.
  Seed work is additional to the evolutionary `optimize.iters` budget.
- Explicitly enabled `optimize.prefilter` checks may instead complete a seed or offspring with
  a graded constraint and signed numerical penalty defaults, without replay. Such records carry
  `backtest_performed=false`; they are terminal optimizer evaluations, not simulator evidence.
  Disabled checks perform no strategy probing. Rust owns hypothetical geometry and grid sizing;
  Python owns normalized policy violations and ordinary evolutionary F/G transport.
- Partial scenario-screening observations may select later offspring for full-suite evaluation;
  they must never supply complete fitness, result records or Pareto membership.
- Incomplete GPU checkpoints own normalized CPU search/cohort state and anchors, without
  device handles or borrowed-memory names. Resume does not require original seed files.

Timeframe-specific EMA spans use explicit horizon suffixes in canonical config names. Use `_1m`
for 1-minute candle inputs and `_1h` for 1-hour candle inputs, for example
`volatility_ema_span_1m`, `volatility_ema_span_1h`, `forager_volume_ema_span_1m`, and
`forager_volatility_ema_span_1m`. Generic strategy EMA spans such as `ema_span_0` and `ema_span_1`
remain unsuffixed because their timeframe comes from the strategy's base candle stream.

## Forager Readiness Stages

Live health distinguishes three stages:

1. The candidate universe contains approved, age-eligible, cost-eligible symbols.
2. A rankable candidate has the required real-candle quote-volume and log-range
   features.
3. Each Rust-plannable action has the strategy, trailing, and risk inputs it
   consumes; availability may differ by position side and entry/close branch.

A flat candidate with unavailable ranking features remains visible as a forager
candidate but is marked `forager.rankable=false`; this is distinct from an
active symbol losing a required strategy input. The Rust payload still fails
closed for an unrankable candidate. Monitor health must preserve that stage
distinction so operators do not mistake a bounded candidate-ranking exclusion
for an exchange market becoming intrinsically nontradable. Before the current
EMA bundle completes—or after a replacement bundle fails—eligible candidates
remain visible with `forager.rankable=false` and
`ema_bundle_unevaluated`; monitor health must not infer readiness from empty
failure sets. Bundle readiness is scoped to the exact evaluated symbol set, so
a newly age- or cost-eligible candidate remains unevaluated until a subsequent
bundle includes it. Forager candidate ranking continues to use completed-candle
metrics; held and explicitly normal symbols nevertheless retain any open-tail
log-range projections required by their active strategy. Live orchestration
passes completed-candle forager metrics separately from strategy EMA maps so a
shared span cannot leak a projected strategy value into coin ranking.

## Trailing Martingale Semantics

Trailing-martingale price spans belong to `entry.ema_span_0/1`. Final v9 loading rejects the old
strategy-root leaves. The explicit released-v8 converter moves them before independently
authored file/inline patches merge and rejects conflicting values within one authored layer.
Public optimizer bounds share the nested paths; internal optimizer keys and Metal columns retain
`ema_span_0/1`. Entry subtree selectors include the horizons. EMA-anchor price spans retain their
strategy-root paths. Volatility horizons remain shared by entries and closes.

Entries and closes use threshold/retracement fields.

- `retracement_base_pct <= 0.0`: trailing disabled, use passive recursive limit-order behavior.
- `retracement_base_pct > 0.0`: threshold is the required excursion, retracement is the confirmation move.
- Trailing extrema reset after any fill for the same coin+pside.
- Passivbot tracks trailing state itself from candle inputs; exchange-native trailing order types are not part of the core contract.

Live trailing availability is a Rust planning input, not a Python reconciliation policy. Python
sets `trailing_available=false` only for the affected coin+position-side while retaining a
structurally valid but inert bundle. Rust omits only an entry or close branch whose active strategy
configuration actually consumes trailing extrema. The other branch, the other position side,
panic, and independent risk reducers continue when their own inputs are complete. The resulting
Rust ideal set remains authoritative: reconciliation cancels resting orders absent from that set
and never preserves them merely because trailing reconstruction is pending.

Entry thresholds and retracements are multiplicative distances. Positive volatility or wallet
exposure weights widen them.

`bot.<side>.strategy.trailing_martingale.entry.ema_gate_mode` controls which entry orders are
capped/floored by the EMA entry band:

- `disabled`: no emitted entry order is capped/floored by EMA. Flat initial entries rest at current
  best bid/ask; re-entries use normal threshold/retracement logic.
- `all`: initial entries, partial initial entries, and re-entries are capped/floored by EMA.
- `initial`: normal initial and partial initial entries are capped/floored by EMA. Re-entries are
  not EMA capped/floored. This is the default and preserves the previous canonical behavior.
- `reentry`: flat initial entries are not capped/floored by EMA; re-entries are capped/floored by
  EMA.

The value is fixed config, not an optimizer parameter. In one-way mode, if both sides are flat and
both long and short are otherwise eligible, the long-vs-short tie-break still uses the EMA entry
band distance even when `ema_gate_mode = "disabled"`. Candles and EMA bands are therefore required
for that tie-break, and missing EMA inputs must fail loudly.

Close thresholds are additive so they can intentionally cross through break-even or negative markup
as wallet exposure rises. Close retracement is volatility-weighted and intentionally has no
wallet-exposure modifier.

TM grid and trailing re-entry/close distance targets use a fused base-plus-distance expression.
A representation residual is corrected only when checked decimal arithmetic proves exact venue
alignment; genuine off-grid targets keep directional rounding. Quantity-unit recovery requires exact
decimal divisibility below 2^53 units and distinguishable neighbors in the operation's direction.
Ambiguous counts, unproven alignment, and checked-arithmetic overflow keep physical-unit rounding.
Near-half quantity products use checked decimal operand arithmetic, including products rounded
onto the boundary. Unproven products retain physical-unit rounding.
Venue rounding converts the chosen integer count directly to its decimal step value, avoiding
another physical-value rescaling. Nearest rounding resolves quotient ties with checked decimal
arithmetic when possible; directional rounding only recovers provably aligned inputs and keeps
physical rounding for other intermediate values.

Compound trailing-price targets retain the two effective threshold/retracement operands through
an exact decimal venue-alignment proof. For example, `43.2 * (1 + 0.1 - 0.09375)` proves `43.47`
before the subtraction introduces a binary residual. Only exact divisibility overrides rounding;
non-aligned pairs retain the existing composed-f64 distance and fused physical target. A failed
compound proof must not retry alignment using the already-rounded sum; a physical target equal
to the canonical venue value remains exact independently of any distance proof. Genuine
authored offsets remain directional inputs, not candidates for tolerance snapping.

GPU TM replay transports original decimal DDF and base threshold/retracement operands from Rust
as finite integer limbs, separately from the ordinary float32 parameters. Checked unit arithmetic
resolves near-half quantities only for uniquely recoverable counts below 2^24. Static price-distance
proofs additionally require a passive initial-fill tick lineage; partial closes retain that lineage,
while every weighted increase, market fill, and full flatten invalidates it. Recursive snapshots
and temporal replay preserve the same proof state. Per-modifier masks preserve original-precision nonzero values and explicit coin pins, so an
effective static group remains provable when pinned zeros replace nonzero global modifiers. Coefficients outside 60 bits, decimal exponents outside [-18, 18], checked
integer overflow, weighted prices, and dynamic distances retain fused physical float32 arithmetic.
Non-aligned rational price proofs additionally decline remainders within a conservative f64
operand/FMA/tick-conversion envelope, where Rust's fused rounding may cross the decimal boundary.
Exact aligned proofs and authored offsets outside that envelope retain their existing decision.
These bounded proofs do not establish general GPU numerical acceptance or native migration readiness.

Auto-unstuck has its own EMA trigger toggle:
`bot.<side>.unstuck.ema_gating_enabled`. It defaults to `true`. When false, auto-unstuck skips the
EMA trigger/readiness check but still requires `unstuck.enabled`, loss allowance, exposure
threshold, close sizing, and valid market/exchange inputs.

V9 auto-unstuck uses one continuous account-wide net-realized loss memory, with
`bot.global.unstuck.memory_ema_span_minutes` slow decay and per-fill clipping. Entry/exit fees,
ordinary closes, HSL closes and signed rebates all contribute in deterministic economic
order. UPNL and external balance changes do not. HSL episode transitions never reset it.
Rust computes allowance as `max(0, p * (B + D) - D)` using the explicit binary32/FMA
graph in [the risk contract](../../v9_risk_model.md). Selection, effective TWEL,
exposure thresholds and close sizing retain their existing policies. The global
`bot.global.loss_gate` consumes the same ordered facts using its independently configured
`memory_ema_span_minutes` and separate drawdown. It uses `p=max_drawdown_ratio`,
without TWEL multiplication; panic exemption and close-selection policy remain. Gate-only
configs still produce gate memory. Each enabled filter consumes each fill once and the
economic ledger records it once.
Config schema v9.0.0 retires its separate rolling window. Active old-schema gates
require explicit semantic migration; zero and all-history windows are not EMA spans.
Live realized-memory bootstrap discloses an unknown prefix and residual attenuation;
a finite suffix is not proof of infinite-tail equality. Required malformed or missing
realized facts are fatal/deferred at their owning boundary, never an invented zero D.

`bot.<side>.unstuck.ema_span_0` and `ema_span_1` independently define the unstuck price EMA
band, using the same base candle stream and unrounded geometric-mean third span as the strategy
band. Long eligibility uses the upper band; short eligibility uses the lower band. Strategy EMA
changes must not change the unstuck band. Missing required unstuck EMAs follow the existing scoped
input-unavailable contract; disabled gating does not require them or extend warmup. Live loading
requests unstuck-only spans for held sides separately, preserving usable strategy spans when an
unstuck horizon is unavailable. Live warmup uses that same held/static eligibility and does not
expand flat forager candidates to an unused unstuck horizon. Monitor bands are independently
available for each consumer.

Explicit released-v8 conversion materializes missing unstuck price spans from the effective active
strategy before default hydration, including coin overrides and external-file/inline precedence.
Explicit unstuck spans win. Converted optimizer bounds copy fixed legacy bounds, or freeze at the migrated starting values for
varying legacy ranges unless supplied; a warning explains that independent genes cannot preserve
the old coupled search. New defaults expose tunable spans.
Supported shared-account Metal kernels model a separate price EMA band for unstuck.
It uses the same seeded recurrence, floating horizons and candle-interval scaling as exact Rust.
Coin overrides win over candidate globals, and temporal replay persists the independent band.

The opt-in optimizer override `couple_unstuck_ema_spans` derives each side/coin's unstuck pair
from its effective strategy, after mirroring and fixed/scenario overrides. It removes redundant
unstuck span genes and materializes explicit runtime spans in saved candidates and scenarios.
GPU packing preserves the dependency on candidate strategy genes while retaining strategy coin
pins. This is optimizer configuration finalization, not a live/backtest coupling mode.

## Backtest Limit Fill Buffer

`backtest.limit_order_fill_buffer_pct` is a simulation-only ratio, defaulting to `0.0`.
Limit buys require `low < price * (1 - buffer)` and limit sells require
`high > price * (1 + buffer)`. Equality never fills. The buffer changes eligibility only:
filled limit orders retain the original order price and maker fees. Market execution bypasses it.
The actual fill and next-candle ladder-expansion hint share the same Rust predicate; raw candles
used for indicators, trailing extrema, equity, and risk remain unchanged. Live inputs do not
forward this setting. CPU and GPU optimization treat it as fixed evaluation policy. GPU screening
encodes buffered strict crossings as immutable integer tick boundaries during data preparation; kernels retain
the same fill comparisons and original candle/touch inputs. Prepared-data cache keys and
checkpoint execution identity include the buffer.
Thresholds above all representable order ticks saturate at the maximum tick: this preserves
all-sell/no-buy eligibility without rejecting large valid buffers. Executable price range checks
remain unchanged.
The native backtest payload loader accepts an absent buffer as `0.0` for payloads from before
this setting existed; explicit invalid values still fail. Canonical Python loading always supplies
the field. Orchestrator next-candle hints likewise default absence to zero and reject non-finite
values or ratios outside `[0, 1)` before order construction.

## Live/Backtest Market Slippage Boundary

`backtest.market_order_slippage_pct` is a backtest simulation knob only. Live orchestrator input
must not read or forward it; live callers intentionally omit `market_order_slippage_pct` and rely
on Rust's `0.0` serde default for live loss projections. Live market-order execution uses the
current bid/ask snapshot plus exchange-side behavior, controlled by `live.market_orders_allowed`
and `live.market_order_near_touch_threshold`.

## Close Recursion Contract

Close orders are computed recursively when the close threshold depends on wallet exposure:

1. Compute the next close from current position exposure.
2. Size it up to `close.qty_pct`.
3. Simulate that close as filled.
4. Recompute wallet-exposure ratio.
5. Repeat until the position is exhausted or the ladder is complete.

If `close.retracement_base_pct <= 0.0` and `close.threshold_we_weight == 0.0`, recursive closes all
have the same price. Rust intentionally emits one full-position close in that case; `close.qty_pct`
is effectively moot because multiple same-price slices would be redundant.

## GPU Market Execution Price Preparation

Native GPU preparation asks Rust to choose market SELL and BUY tick counts from the original
f64 candle closes, f64 slippage and venue price steps before packing replay inputs as f32.
The helper uses the CPU market-price rounding and minimum-price rules. Every active result
must prove a unique positive int32 tick count at the canonical decimal step; ambiguous or
out-of-range results fail preparation. Zero is reserved for inactive candle/coin cells and
must never become an active execution price.

Shared Metal/CUDA replay consumes those immutable counts for ordinary market fills, HSL
panic fills and delisting closes at their declared execution indices. Touch eligibility,
quantity sizing, marking and fee policy retain their existing inputs. GPU state and decoded
physical prices remain f32, so matching tick decisions do not establish exact financial
trajectory equality for every configuration.

## Close Reducer Compatibility

For each coin and position side, Rust selects at most one protective reducer per ideal-order batch.
Active panic, TWEL/WEL exposure-repair, and auto-unstuck intents are consolidated by keeping the
largest loss-admissible absolute reduction after final position/minimum sizing, not by summing
their quantities. If the realized-loss gate blocks the largest non-panic intent, Rust tries the
next-largest intent. This prevents a small auto-reduce from suppressing a materially larger unstuck
close while retaining a smaller safety fallback when the larger close would exceed the loss
budget. A full-position HSL panic is therefore largest and remains exclusive; equal-size ties keep
panic first and otherwise prefer the closest-to-fill candidate.

A selected non-panic reducer may coexist with ordinary grid, trailing, or EMA-anchor closes. Its
quantity is reserved first; if aggregate close quantity would exceed the position, ordinary closes
are trimmed furthest-from-fill first against the remaining quantity. Aggregate reduce-only quantity
must never exceed the position after quantity-step and effective-minimum handling. The cumulative
realized-loss gate participates in this allocation, checks the final reducer size, and spends the
shared batch allowance on final reducer quantities largest-first rather than in symbol iteration
order. It then evaluates ordinary closes after the selected reducers. Live reconciliation reapplies
the same aggregate cap against current exchange position size, still trimming ordinary closes
before the reducer if the position shrank after planning.
This contract is shared by both supported strategy kinds.

## Source Of Truth

Rust owns strategy dispatch and order behavior. If Python docs, adapters, or tests imply old
`trailing_grid` behavior, update those surfaces to match Rust rather than adding Python-side
behavior patches.

## Validation

- Rust unit tests cover strategy dispatch, entries, closes, recursion, risk, and unstuck behavior.
- Python tests use the verified real extension when asserting Rust output.
- Live/backtest parity tests compare optimized behavior with a simple reference contract.
- Config schema tests reject removed fields and strategy kinds before hydration; released-v8
  conversion tests cover explicit transformations, conflicts and file/inline precedence.
- Behaviorally relevant changes include a bounded real backtest smoke.

## Key Code And Tests

- `passivbot-rust/src/orchestrator.rs`
- `passivbot-rust/src/entries.rs`
- `passivbot-rust/src/closes.rs`
- `tests/test_orchestrator_json_api.py`
- `tests/test_orchestrator_integration.py`
- `tests/test_auto_unstuck_allowance.py`

## Adaptive entry timing and unilateralness

Rust owns signed RMS directionality, its use as a lower-is-better Forager component,
and additive cooldown minutes. The shared side-level float span lives under
`forager.unilateralness_ema_span_1m`. Both runtimes use the same last
`ceil(20 * span) + 1` completed one-minute closes with zero-seeded exponential weights.
Live replays the window. CPU uses rolling aggregate stacks with amortized constant
work per candle, agreeing with replay within floating-point roundoff; all-flat windows
are exactly zero. No RAM-only indicator state is needed for restart. Missing history is absent,
never a neutral score. CPU candle intervals other than one minute are rejected
when an RMS consumer is enabled. Dormant weights on a statically disabled side do
not request RMS history, allocate CPU trackers, or restrict candle intervals. Optimizer
dataset preflight checks finalized per-coin eligibility and overrides across reachable
boundary configs. Entry-ineligible sides do not restrict intervals or RMS activation.

Canonical loading and cleaning expose missing unilateralness weight/span and
adaptive cooldown weight/minimum-duration optimizer bounds as fixed ranges at
the side's configured values (using normalized Forager weights). A missing maximum
duration bound is hydrated only for a finite ceiling; null remains unbounded.
Explicit bounds retain their authority. Visibility must not open a search range or
change optimizer candidate normalization or warmup for unchanged configurations.

Forager may carry a complete cached observation only within the existing candidate
age budget, without feeding invented flat returns. Cooldown requires a current
completed window. Their values remain separate in the input envelope. Missing
RMS is scoped to the consuming entry or ranking branch; closes remain independent.
Live `unilateralness_unavailable` carries separate `current` and `forager` span
lists. RMS transport failures must not set the generic missing-strategy-input
permission. A valid cached ranking value does not satisfy current cooldown input.
CPU history requests include RMS only for reachable consumers; scoring alone needs no
history when a known eligible side universe fits fixed slots with dynamic WEL disabled,
or has at most one coin. Shared trade activation does not wait for RMS.
Per-coin `warmup_minutes_source` distinguishes producer-stamped `history` from
`activation` budgets. Only known history metadata may be replaced by a candidate's
non-RMS activation budget; optimizer activation stamps and untyped external budgets
remain authoritative. Rust's zero global warmup retains its automatic non-RMS
fallback. Python sends the minimum positive budget when it explicitly computes a
zero activation budget for an adverse-RMS consumer.
The backtest marks exact spans still warming from known listing history in
`unilateralness_warmup_spans`. A missing score at a marked span defers only required
ranking or the side/order branch consuming adverse cooldown. If any compared
candidate lacks required RMS, defer the whole ranking decision, not just that
candidate; never select from a history-ready subset. Closes and unrelated
entries remain independent. Once enough closes exist the marker is removed.
Unmarked missing inputs, unrelated required inputs and invalid scores remain fatal.
The Python output validator evaluates the submitted cooldown inputs through the
same pure Rust policy, rather than trusting a producer-echoed duration.

Cooldown uses nonnegative additive weights for existing exposure ratio (without a
unit cap) and adverse signed RMS. Apply floor/ceiling once after composition.
A validated policy whose base/floor already reaches its ceiling is constant and
requires no modifier inputs. Inactive adverse modifiers impose no RMS history or
interval requirement; optimizer bounds must account for nonconstant corners.
Enabled weights require a finite ceiling; structural fill coverage, pair activation
and restart anchors use that horizon even at base zero. Live approval gates initial
entries, not DCA on held graceful-stop positions. Keep held-side cooldown history and
adverse inputs conservatively; effective trading modes remain Rust's decision.
Current effective duration,
not base alone, governs elapsed-time gating and entry-ladder staging. At effective
zero retain existing strategy rules. Partial fills remain ordinary increasing fills.
All new weights default to zero. Metal/CUDA screening supports these settings for
EMA Anchor and Trailing Martingale, including directional and fused portfolios.
GPU RMS uses a finite exponential recurrence, reads the expiring return from immutable
candles, and periodically rebuilds to bound float32 subtraction error. Replay checkpoints
preserve each candidate/coin/side accumulator; exact Rust validation remains authoritative.
The GPU parameter-layout revision invalidates older optimizer checkpoints.
