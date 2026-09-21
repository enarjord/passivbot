# Revised HSL implementation checkpoints

This temporary implementation record follows the
[reviewed design](hsl_best_effort_redesign.md). Legacy HSL remains the trading default;
the revised numerical kernel below has no live, backtest or optimizer controller caller.

## Shared numerical kernel

`passivbot-rust/src/hsl_revised.rs` owns the pure batch drawdown calculation. Its
`hsl_revised_signal` Python binding accepts ordered `(timestamp_ms, realized, unrealized)`
currency observations, a positive budget, fractional EMA span, RED threshold and optional
entry reference. The JSON result includes equity, running peaks, raw drawdown, EMA and
strict threshold decisions for every observation. This is an experimental comparison
surface, not a complete HSL engine or an alternative execution authority.

The kernel implements current-budget rebasing, raw-drawdown EMA with its first value as
seed, same-minute replacement from the prior-minute baseline, and known boundary peaks.
A supplied minimal-history reference seeds the peak without creating a zero EMA sample.
It has no journal, cache, persistent EMA or prior-decision input. Callers remain responsible
for scope aggregation, valid current observations, clipping, reconstructing history and
controller lifecycle. A singleton without an entry reference is just a mathematical
one-point series, not the complete no-history estimator.

Subtract common realized/unrealized offsets before combining currency deltas to retain
small representable changes. Nonpositive historical peaks use the reference impairment
value of one. Opposite deltas that individually overflow are combined with power-of-two
scaling before saturation so their representable residual is not discarded.
Arithmetic outside finite float range saturates with
`numeric_range_approximation=true`; the current endpoint is always restored to the budget.
This is explicit approximation, not a finite-precision claim of exactness for extreme
history. Invalid minimum inputs, nonfinite observations and reversed timestamps raise
rather than returning a healthy decision. The later best-effort reconstructor must turn
damaged historical fields into documented estimates before calling this numeric kernel.

## Validation and remaining work

The real-extension parity suite compares the kernel with the independent Decimal oracle:
hand-computed losses/recoveries, strict threshold equality, fractional spans, singleton
references, nonpositive peaks, repeated within-minute samples, common currency offsets,
40 deterministic 100-observation paths, finite-range behavior and input rejection.
Existing legacy Rust tests run alongside these cases.

## Pair history and candle projection

`hsl_revised_history.rs` implements the independent pair reconstruction rules: select
canonical fill revisions before window clipping, retain known sequence and deterministic
ambiguous cohorts, walk signed quantities backward from current exposure, clamp impossible
historical quantities, then reconstruct average basis and net realized cashflows forward.
Known realized PnL and signed fees survive missing quantity/price components. Observed current
size, basis and mark replace the endpoint. Linear and inverse contract units remain explicit.
The output includes historical samples, canonical fill cashflows/quantity transitions and
diagnostic reasons; estimated flats are not automatically lifecycle proof.

The experimental `hsl_revised_history` JSON binding accepts one normalized pair and an
already normalized historical price grid. Malformed historical numeric fill components are
represented as null, independently of their usable PnL/fee fields. It is not an exchange
payload parser: identities, timestamps, revision metadata and actual sequence provenance
are supplied by factual normalization. The later snapshot layer must quarantine impossible
post-capture revisions and isolate fills after the position anchor before this primitive
selects revisions. This primitive alone does not establish freshness or coherent acquisition.
An absent grid returns the current endpoint and independent cashflow trace; it does not
discard the cashflows or claim zero historical realized losses.

`hsl_revised_prices.rs` supplies a bounded minute-close grid. It selects the finest available
1m/5m/15m/1h source, expands complete coarse candles along the reviewed zigzag, then fills
interior/trailing gaps and the missing leading segment within lookback. A real 1m candle
needs only its valid close. Coarse candles require complete, consistent positive OHLC and
their entire interval inside lookback. Future/incomplete sources are excluded. Empty history
is an explicit result for the minimal construction, not an evaluation failure. Generated
rows disclose source resolution, source candle end and whether the price was carried.

Conflicting equal-resolution contributions at a timestamp are excluded at that resolution;
uncontested coarser data can still supply the minute. Further duplicates cannot resolve the
conflict, and input permutation does not change the result. This explicitly extends the
reference helper's requirement to normalize conflicts before sampling: the estimator chooses
an available uncontested source rather than failing the whole history. With no such source,
the normal in-window filling/minimal-history rule applies. These estimated rows never enter
the factual candle store. The price generator rejects intervals longer than 90 days to bound
allocation; production's separate enabled-HSL minimum of one day remains a config obligation.

These components remain disconnected from trading. Reconstruction parity cases include
complete/missing/corrected fills, local clamping, known cashflows, malformed historical fields,
both contract types and position sides, deterministic fault/repair tapes, actual offline fake
exchange cashflows, extreme finite values, causal coarse prices, source conflicts and gap
provenance. The next layer must compose these primitives across immutable scoped snapshots
and lifecycle boundaries without adding an exact-history readiness gate.

## Pure controller replay

`hsl_revised_controller.rs` replays GREEN/RED permissions over supplied scope-level
numerical episodes. Its experimental JSON binding accepts a complete trace ending at the
current evaluation time; stale final samples are rejected instead of retaining expired
cooldowns. It implements `always`/`never`, zero cooldown, `panic`/`normal` interventions,
partial-exit continuity and lookback clipping, with no previous-controller-state input.
Every episode uses currency PnL on the same cumulative basis and rebases against the
common current endpoint. Episode boundaries reset peaks and EMA, not the equity currency
offset; completed episodes may end at nonpositive historical equity. Minimal-history entry
references are accepted only for the current exposed singleton, never a completed/flat
episode, and future trace observations are
rejected rather than clipped. A numerical RED that remains reconstructible in the trace survives subsequent recovery;
corrected or expired evidence can remove it on the next independent replay.

A supported flatten ends an episode only after its final risk observation. Cooldown starts
at that actual flatten, not at a partial fill or retry. An estimated flat cannot be marked
as a supported boundary. A surviving generic flat does not prove an expired numerical
crossing was an HSL stop; unreconstructible decisions remain void. The input builder must
still establish exchange-derived boundaries and any explicit stop provenance. The binding
is a pure trace component, not proof that the supplied trace was acquired correctly.

The independent Python controller oracle plus real-extension parity covers intervention
fees, repeated stops, residual exposure, same-minute boundary order, exact cooldown/window
edges, corrections, missing flat evidence, very long EMA spans and deterministic generated
multi-episode traces. No live loop, backtester, optimizer or execution adapter invokes this
component yet. Scope composition and lifecycle evidence construction remain required.

## Scoped snapshot preparation and candle-free estimates

`hsl_revised_snapshot.rs` prepares selected coin, side or unified pairs from immutable
normalized observations. Current balance/position/mark captures must be usable. Historical
fill defects remain approximation diagnostics: quarantine causally impossible revisions
before canonical selection, isolate fills after the observed position, and retain usable
cashflows and current-anchored numerical history independently of lifecycle eligibility.

Supported flatten rows retain their canonical consumed-prefix lengths, including sequenced
same-time flat/reopen events. Cross-pair timestamp cohorts have no invented global ordering.
Unknown quantities, contradictory transitions and unsupported capture order can withhold a
lifecycle boundary without withholding the numeric estimate. An old damaged prefix does
not permanently taint a later clean suffix. Scope quality uses only selected pair inputs.
Compensated quantity summation avoids drift across repeated partial fills. Cancellation
residuals within eight scaled floating-point epsilons are disclosed as
quantity roundoff rather than contradictory exposure; larger mismatches remain estimated.
The scale includes the reconstructed suffix and resets at zero. Rounding alone does not
certify a lifecycle flat: an optional positive exchange `quantity_step` must be available,
and the entire error allowance must be below half that quantum. Otherwise the rounded
history remains usable but the boundary is uncertain (`quantity_precision_unavailable`).
Actual current exposure is never snapped away. Experimental HSL JSON fields use exact float parsing
so a supplied finite position/cashflow does not change before reconstruction sees it;
legacy JSON parsing is unchanged. Missing precision is not a signal veto.
These rows are evidence for the later episode composer, not controller permissions by themselves.

`hsl_revised_candle_free.rs` composes the approved all-candles-absent estimate across selected
pairs. It preserves known/estimated net realized cashflows and their in-window peak, nets
currency gains and losses before division, and combines them with current UPNL. The peak
reference contributes no fabricated past EMA sample. Coin zero-slot scopes remain inactive;
side/unified budgets use the raw balance. Inactivity never bypasses current-input validation.
Pair and scoped currency sums use a fixed-size exact binary accumulator, retaining all
exponent levels and rounding only when a public float value is read. Gross cashflows and
fees remain separate terms across scope composition, so an already rounded per-pair sum
cannot erase a small fee when it later cancels another pair. Unrepresentable readouts
saturate with diagnostics; the underlying accumulator retains the full sum for later
cancellation. Peak/current cashflow differences and their combination with current UPNL
are formed before rounding as well. Allocation and per-add work are bounded independently
of tape length. The helper refuses to discard usable historical
prices to obtain a singleton result. It is not the mixed-price scope dispatcher.

Parity covers the independent boundary fixtures, corrections and source timing, generated
historical damage, real offline fake-exchange partial/final fills and current cashflows,
scope isolation, contract units, extreme currency sums and no-candle realized-loss cases.
Full fake-live orchestration and live/backtest/optimizer callers are still outstanding.

Still required before offline completion:

- Snapshot-aware composition of the Rust history/price primitives, candle-free/mixed-price
  signals and shared coin/pside/unified adapters, checked against the independent fixtures.
- Complete exchange-reconstructible controller replay, interventions, cooldown/never
  expiry, known flatten samples and execution permissions.
- Startup-only engine selection, config migration and optimizer/backtest integration.
- Source-matched extension parity through full offline fake-live scenarios, partial fills,
  current-state changes, replay recovery and cache-free restart.
- Canonical contracts, migration documentation and the live-validation/rollback checklist.

No actual live testing or legacy replacement is part of offline completion. Those are
subsequent stages with separate authorization.

## Staged configuration boundary

`config.hsl_revised` owns explicit migration and engine-specific hydration.
`live.hsl_engine` defaults to legacy; revised is a startup-only shared selection.
Enabled revised restart choices and unified portfolio authority cannot come from
implicit defaults. CLI/scenario/effective coin configuration and optimizer candidates
are revalidated, removed paths/metrics fail before pruning, and portfolio bounds map
to the portfolio block. The saved-fitness contract includes engine and intervention.

Current runtime guards reject every revised mode before live credential lookup or
backtest/optimizer preparation; they must be replaced with real mode-specific adapters
as integration lands. Configuration support does not establish full runtime readiness.
The scope composer, execution/backtest/optimizer integration, full fake-live testing,
and final live validation/rollback checklist remain outstanding.

Candle-free evaluation retains its peak-to-current currency loss through division
by budget plus loss. It does not round the loss away by subtracting two absolute
equities. The one actual observation seeds raw drawdown and EMA equally, including
when the loss is smaller than a budget ULP. Signal settings are validated even for
an inactive zero-slot coin scope; inactivity only removes the budget division.

## Factual history transport

`live.hsl_revised_inputs` copies a manager's current canonical fill batch and raw
candle-resolution arrays into immutable Rust input records. It does not consume
manager-derived position size/basis, infer ordering from IDs, merge successive
snapshots, synthesize a fill, clip history, resample prices or decide risk.
The manager's canonical batch is replaced on each capture; its resolved rows use
revision zero within that batch. Execution sequence remains unknown until a
producer supplies an explicit supported sequence contract.

Native contract quantities remain separate from contract multipliers. Missing or
mismatched normalized multipliers make only the historical quantity unavailable.
The adapter consumes the manager's existing canonical contract, including its
optional defaults for supplied PnL completeness and native contract units. It does
not reinterpret omitted raw optional metadata as corrupt canonical observations
or add raw-presence certificates to otherwise usable history. Current market
multipliers remain independently required and must match the normalized units.
Pending PnL placeholders stay missing; usable current-contract estimates and signed
fees survive with diagnostics. Unknown accounting contracts cannot supply gross or
fee amounts. Unattributed/undated records are disclosed rather than assigned to an
invented pair or time. Output contains only needed scalar observations, not raw
exchange payloads, runtime provenance or mutable manager references.

Candle inputs use the manager's `ts/o/h/l/c` fields and their actual resolution and
capture time. Rust retains ownership of completeness, causal clipping, conflicts,
resampling and in-window carrying. These are transport helpers; no runtime caller
is activated yet. Offline tests include the real fill manager over the fake exchange,
contract quantities, fees, both sides, partial closes and cache-free reconstruction.
