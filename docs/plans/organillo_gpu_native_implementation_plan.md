# Organillo historical eligibility implementation plan

This historical design plan describes the optional eligibility input. The current behavior is
specified in the [feature contract](../ai/features/organillo.md) and
[user guide](../organillo.md).

Target branch: `codex/gpu-native-optimizer`.

The design began at public base `27610e9c6f180cc78d18c829b20ab8c7b893241d`, which included
native GPU completion accounting. Metal and CUDA hardware qualification are separate: results
on one backend do not establish parity on the other.

## Goal and scope

Import externally computed historical coin eligibility into standalone CPU backtests, CPU
optimization, and native GPU optimization as a small, optional dataset input.

Passivbot validates and consumes a supplied CSV. All market-cap, volume, category, ranking, and
selection-rule calculations remain outside Passivbot. Eligibility is fixed per prepared scenario,
shared by all optimizer candidates, and is not an optimizer gene.

When no CSV is supplied, preserve ordinary backtest and optimization behavior and eliminate
eligibility work from simulation. Do not add live behavior, a new scheduler, CPU validation of
normal GPU evaluations, or a runtime CPU fallback.

## Input contract

- Add one optional CSV path under `backtest`; finalize its name during implementation. Absence
  disables Organillo. An explicitly supplied invalid or unreadable file raises before simulation.
- First column: `date`, using `YYYY-MM-DD`. Remaining columns: canonical coin identities mapped
  deterministically to the prepared exchange's candle columns. No fuzzy matching or silent drops
  of unresolved or ambiguous identities.
- Dates are unique, strictly increasing daily UTC dates. A row applies from 00:00 UTC inclusive
  until the next 00:00 UTC exclusive.
- Every eligibility cell is exactly `0` or `1`. Reject empty, nonbinary, malformed, and ragged
  rows, duplicate headers, and missing required dates or coin coverage. Do not fill missing
  eligibility with a default or implicitly carry it forward.
- Require complete coverage for the operational trading period, including any execution candle
  consumed by the simulator. Warmup-only candles do not authorize entries or contribute earlier
  eligibility counts.
- The enabled candidate universe is the union of eligible coins over the requested period,
  subject to existing approved/ignored and side restrictions. Resolve this consistently with
  current universe preparation; document the interaction explicitly. Loading the union must
  not make future-only admissions affect earlier eligibility, sizing, or readiness.
- Errors identify the file and, where applicable, row, date, column, and expected format.
- Provide a small standalone validation tool that calls the same validator as backtest/optimizer
  preparation. It validates formatting, identity mapping, and coverage, not the external rules
  or the historical correctness of the decisions.

## Trading and WEL semantics

`0` imposes graceful stop: existing positions remain fully managed, including configured DCA,
closes, unstuck, and protective risk behavior. Once flat, no new position may open. `1` removes
the Organillo restriction and restores ordinary eligibility through Forager, strategy, and risk
controls. It does not force entry or override another mode, exclusion, or safety restriction.

Eligibility is separate from candle validity and exchange tradability. Keep indicator history
and held-position management active when a coin becomes ineligible. Do not treat a zero as a
delisting, fabricate candles, or force a close.

Apply the mask to the existing per-side tradability/eligibility count used by
`update_n_positions_and_wallet_exposure_limits`:

1. Count coins satisfying existing candle tradability, side entry eligibility, nonzero side WEL,
   and the current Organillo mask.
2. With `backtest.dynamic_wel_by_tradability=true`, preserve the existing non-shrinking peak
   count: `peak_s = max(peak_s, eligible_count_s)` and
   `effective_slots_s = min(configured_slots_s, peak_s)`.
3. For automatically allocated WEL, use the existing `TWEL_s / effective_slots_s` policy,
   including its zero-count handling. Preserve explicit per-coin WEL overrides.
4. With dynamic WEL disabled, retain the configured denominator. Organillo still restricts
   initial entries, but does not introduce dynamic resizing.

The peak is a maximum simultaneous eligible count, not the number of distinct coins ever
selected. A constant-size rotating selection must not accumulate a larger denominator.
Retired held coins must continue receiving appropriate runtime budgets; filtering them out of
the count must not leave their management using stale budgets. All-zero rows must not stop
position management, reset the peak, or cause division by zero.

Counts and budgets remain simulation-owned because side restrictions, readiness, and candidate
parameters can differ. Only the external yes/no schedule is precomputed.

For timing, use eligibility effective at an initial entry's execution candle. Reconcile pending
initial orders before they can fill on a newly ineligible candle. Preserve permitted DCA for a
held position and prevent a same-candle close from being followed by a prohibited new position.
Implement the same event ordering in CPU and GPU; explicitly test UTC boundaries and the
existing order-at-k / execution-at-k+1 convention.

## Narrow implementation sequence

1. **Canonical preparation and validation.** Add the optional config input, shared validator,
   CLI validation entry point, and one-time timestamp/coin alignment. Fail before worker/service
   startup for malformed input. Keep eligibility helpers off the ordinary disabled preparation
   path where possible.
2. **Compact transport.** Keep OHLCV layout unchanged. Prepare a daily `uint8` matrix and compact
   candle-to-schedule indexing alongside candles. Borrow immutable shared arrays across CPU
   workers and candidates. Extend `PreparedGpuDataset`, `NativeDatasetRegistry`, and service
   attachment/lifetime handling with optional schedule references; transfer when a dataset
   becomes resident, never per candidate. Preserve source-column and time-slice mapping.
3. **Rust consumer.** Add the mask to initial-entry eligibility and the existing WEL accounting,
   preserving held-position management and both fresh and cached order/orchestrator paths.
   Dispatch once into a disabled path with eligibility branches compiled away or the existing
   loop retained. Avoid a new portfolio-wide warmup gate for historical admissions.
4. **GPU consumer.** Apply the same contract in native shared-account EMA Anchor and Trailing
   Martingale replay, covering single-coin, directional, and fused long/short execution. Include
   temporal chunks and factual HSL replay/restarts. Use a compile-time feature specialization
   so the disabled kernel has no mask reads, eligibility branches, or added replay state.
5. **Identity and integration.** Include enabled schedule content, alignment semantics, and
   selected scenario slices in prepared-data and saved evaluation identity. Preserve explicit
   expected hashes during scenario overrides; changed content must not reuse checkpoint fitness.
   Wire standalone backtests, CPU optimizers, suites, and native GPU service through the same
   prepared input. Keep the current GPU ceiling of 64 distinct coins per scenario; reject a
   larger historical union explicitly rather than expanding capacity or falling back.
6. **Documentation and validation.** Add the feature contract, user CSV example, effective-time
   and WEL explanation, and an Unreleased changelog entry when implementing. Complete the
   acceptance gates below before publication.

Relevant target-branch integration points:

- `passivbot-rust/src/backtest.rs`: Rust eligibility, order timing, runtime WEL budgets.
- `src/optimization/gpu/datasets.py` and `src/optimization/native_datasets.py`: immutable
  schedule descriptors, mapping, ownership, and registration.
- `src/optimization/gpu/service.py`, `src/optimization/gpu/mps_kernel.py`, and their shared
  shader sources: native GPU replay transport and compile-time ablation. Current file names
  retain MPS terminology. The Apple Metal and CUDA execution paths require separate hardware
  qualification.
- `src/optimization/prepared_dataset_identity.py` and evaluation-contract consumers: replay
  and saved-fitness identity.
- Existing `passivbot tool gpu-parity`: offline development comparison; never called by normal
  optimization.

## Correctness and CPU/GPU parity acceptance

Use identical immutable candles, metadata, timestamps, resolved configs, and schedules for both
engines. Rebuild and fingerprint-verify the Rust extension. Run the native Metal backend through the actual
GPU service, not a retained legacy diagnostic path. Repeat hardware qualification on CUDA
when available; Metal results do not establish CUDA hardware parity.

Required fixtures cover:

- Disabled input and enabled all-ones input on an otherwise identical universe.
- `1 -> 0 -> 1` for flat and held coins, including DCA, closes, and re-admission through Forager.
- Pending initial orders at UTC boundaries and close-then-entry behavior within a candle.
- All-zero days, never-selected coins, staggered admission, constant-size rotation, and later
  removals. Prove future-only coins do not change earlier slots, WEL, or global readiness.
- A newly admitted coin with a long indicator warmup while another coin retires. Existing
  positions must remain managed and stale initial orders must not fill.
- Both dynamic-WEL settings; separate long/short eligibility; explicit WEL overrides; ignored
  coins and existing forced modes; competition for limited Forager slots.
- Coin subsets/reordering, scenario overrides, time slices, and non-one-minute candle intervals.
- Both GPU strategies, single-coin and multicoin, hedged and one-way, chunk boundaries, factual
  HSL replay, and service dataset eviction/reload.
- Invalid CSVs, missing files/coverage, mismatched transport dimensions, nonbinary buffers,
  changed-content resume rejection, and an over-limit GPU historical union.

Acceptance criteria:

- Eligibility decisions, per-side counts/peaks, effective slots, schedule timing, and categorical
  order/mode decisions match exactly in targeted fixtures. No prohibited initial entry occurs.
- Numerical WEL, fills, balances, and requested metrics match the existing scoped CPU/GPU
  numerical policy. Do not widen tolerances to hide an eligibility or event-order mismatch.
  Include fixtures away from numerical thresholds so behavioral differences are unambiguous.
- Held-position management continues through zero rows; no forced closes or portfolio-wide
  pauses are introduced by eligibility alone.
- Disabled output equals the baseline within each engine on identical inputs. Enabled all-ones
  output equals disabled output on a matching prepared universe and operational period.
- Validation failures occur before simulation with actionable messages. No silent fallback,
  fabricated eligibility, or reuse of fitness from a different schedule is permitted.
- Focused Rust/Python tests, native GPU parity checks, and bounded offline CLI integration
  checks pass. GPU unavailable means GPU acceptance remains outstanding, not implicitly passed.

## Disabled-path performance acceptance

Structural acceptance comes first:

- No CSV access, eligibility parsing, schedule hashing, schedule allocation, extra universe
  preparation, or eligibility-related shared-memory/device transfers without a supplied CSV.
- No added per-candle or per-coin eligibility branch/read in disabled Rust simulation.
- No added eligibility arguments/buffers, replay-state storage, or eligibility instructions in
  the disabled compiled GPU kernel. Compare preprocessed/compiled code against the baseline;
  source-level guards alone do not establish ablation.
- Existing dataset residency, request packing, batching, candidate search, scoring, and normal
  checkpoint paths acquire no eligibility work when disabled. Preserve disabled identity shape
  where possible; ordinary implementation-version invalidation remains separate.

After implementation, measure an exact baseline and changed head using the same hardware,
dependency/runtime versions, verified Rust artifacts, frozen inputs, seeds, candidate counts,
metrics, and execution settings. Use paired, alternating repeated runs and record variability.
Separate cold preparation/compilation from warm steady-state execution.

Measure standalone CPU backtests, CPU optimization, and native GPU optimization. Include small
runs sensitive to setup overhead and representative long multicoin runs. Cover preparation time,
evaluation throughput, end-to-end completion time/latency, peak host/device memory, and transfer
counts/bytes. Use fixed-candidate cohorts for direct throughput comparisons and bounded optimizer
runs for integration; distinguish engine numerical differences from within-engine regressions.

Acceptance requires unchanged disabled results, no added eligibility allocations/transfers, and
no reproducible slowdown beyond measured baseline noise. Predeclare the run count and detection
margin from baseline variability; report confidence bounds and the smallest regression the
measurements can exclude. If evidence is inconclusive, gather more evidence rather than declare
zero cost. Literal zero slowdown cannot be established from one timing run.

Measure enabled overhead separately: one-time validation/preparation, schedule memory and device
upload, and replay cost. These measurements must not substitute for disabled-path acceptance.

## Validation evidence

Record baseline/change SHAs, runtime and Rust fingerprints, fixture/data/schedule identities,
resolved public test configurations, test and parity results, compiled-path ablation evidence,
and the paired performance report. Keep private inputs and results outside the tracked tree.
Any incomplete CPU/GPU parity or performance gate remains an explicit blocker to accepting the
implementation as complete.
