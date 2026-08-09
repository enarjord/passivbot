# Coin Overrides Overhaul Plan

## Objective

Make per-coin configuration overrides deterministic, validated, and shared by live and
backtest execution. Deliver the work in small, independently reviewable pull requests
instead of extending the current feature branch indefinitely.

## Agreed policy

- [x] Pause further expansion of PR #1462 while the common override foundation is built.
- [x] Keep `bot.<pside>.unstuck.loss_allowance_pct` overridable.
- [x] Remove `bot.<pside>.risk.we_excess_allowance_mode` from the override surface.
- [x] Add `bot.<pside>.unstuck.ema_gating_enabled` to the override surface.
- [x] Add `bot.<pside>.risk.entry_cooldown_minutes` to the override surface.
- [x] Make the complete `bot.<pside>.hsl.*` group overridable when the global
  `live.hsl_signal_mode` is `"coin"`.
- [x] Do not add any new `live.*` overrides in this series.

## Cross-PR invariants

- A coin override is an explicit, typed patch. Missing values never become overrides
  through default hydration.
- Precedence is deterministic: global configuration, then file-based coin patch, then
  inline coin patch.
- An explicitly supplied value remains meaningful even when it equals a schema default,
  is false, or is zero.
- Every effective per-coin configuration is validated before trading or simulation uses
  it. Invalid overrides fail with the coin, source, and parameter path identified.
- Live and backtest execution consume the same resolved per-coin configuration.
- Coin-symbol normalization collisions are rejected rather than silently overwriting one
  patch with another.
- Strategy-kind mismatches are rejected until mixed per-coin strategy kinds are explicitly
  designed and supported.
- Missing, unreadable, malformed, or invalid configured override files fail closed.
- Parsing and runtime compilation are idempotent; repeated preparation must not lose patch
  provenance or change the result.
- Documentation examples use only public repository fixtures under `configs/examples/`.

## PR 1: canonical override-patch foundation

### Design and implementation

- [x] Inventory every current override producer and consumer in live, backtest, config
  loading, and runtime compilation.
- [x] Define one canonical internal representation for a per-coin patch, including source
  metadata needed for actionable errors.
- [x] Extract only explicitly present values from override inputs; do not infer patches by
  diffing hydrated configurations.
- [x] Canonicalize supported legacy input spellings exactly once at the parse boundary.
- [x] Centralize file and inline precedence in one resolver.
- [x] Validate patch types and values, then validate the fully resolved per-coin config.
- [x] Reject normalized symbol collisions and strategy-kind mismatches.
- [x] Make configured override-file failures fatal and actionable.
- [x] Preserve the currently documented override allowlist in this PR; policy changes are
  reserved for PR 2 and PR 3.
- [x] Route live and backtest callers through the same resolver.
- [x] Update `docs/coin_overrides.md` to describe the actual foundation contract.
- [x] Add an `Unreleased` changelog entry for user-visible correctness changes.

### Regression coverage

- [x] Cover every currently allowed leaf for both long and short sides.
- [x] Cover file-only, inline-only, and file-plus-inline precedence.
- [x] Cover explicit resets to global/default values, `false`, and zero.
- [x] Cover canonical and supported legacy input paths.
- [x] Cover unknown and disallowed paths without silently accepting misspellings.
- [x] Cover missing, unreadable, malformed, and structurally invalid files.
- [x] Cover normalized-symbol collisions.
- [x] Cover strategy-kind mismatches.
- [x] Cover booleans supplied as strings, nulls, non-finite numbers, wrong scalar types,
  and cross-field validation failures.
- [x] Cover repeated preparation/compilation and confirm idempotence.
- [x] Cover equivalent live and backtest resolution results.

### Validation and release gate

- [x] Run focused override and config tests.
- [x] Run the broader Python test suite required by `docs/ai/validation.md`.
- [x] Rebuild and verify the Rust extension before Rust-backed validation.
- [x] Run at least one real backtest with no coin patch and one with a public coin patch;
  verify unchanged baseline behavior and the intended per-coin delta.
- [x] Run the deterministic offline fake-live harness with file and inline overrides,
  including a restart scenario.
- [x] Inspect the exact diff and staged paths for public-repository safety.
- [x] Open a regular ready-for-review PR.
- [x] Request Codex review; first pass reviewed `abb40a4840e64a834a50077540d0a2d5c2894313`.
- [x] Address actionable findings, add regressions, rerun proportional validation, and
  request re-review until no unresolved actionable findings remain.
- [x] Require all CI checks green on the current head.
- [x] Merge the reviewed current head and mark this section complete.

## PR 2: override policy and entry cooldown

### Implementation

- [x] Branch from the merged PR 1 result.
- [x] Express the allowlist as a maintainable parameter policy rather than scattered
  conditionals.
- [x] Remove `bot.<pside>.risk.we_excess_allowance_mode` from allowed overrides with an
  actionable migration error.
- [x] Keep `bot.<pside>.unstuck.loss_allowance_pct` allowed.
- [x] Add `bot.<pside>.unstuck.ema_gating_enabled`.
- [x] Add `bot.<pside>.risk.entry_cooldown_minutes`.
- [x] Update reference documentation and the changelog.

### Validation and release gate

- [x] Add table-driven policy tests for every added, retained, and removed parameter.
- [x] Prove independent long/short behavior and file/inline precedence.
- [x] Run focused and broader Python tests.
- [x] Rebuild and verify the Rust extension.
- [x] Run real backtest comparisons that exercise entry cooldown and unstuck EMA gating.
- [x] Run deterministic offline fake-live scenarios that exercise the new parameters.
- [x] Open a regular ready-for-review PR.
- [x] Request Codex review; clean pass reviewed
  `145b497315e65934870d0aec381d2316403b803d`.
- [x] Address findings and repeat review plus validation until the current head is green.
- [x] Merge the reviewed current head and mark this section complete.

## PR 3: conditional per-coin HSL group

### Implementation

- [x] Branch from the merged PR 2 result.
- [x] Allow the complete `bot.<pside>.hsl.*` group only when the effective global
  `live.hsl_signal_mode` is `"coin"`.
- [x] Define and test precedence when the signal mode and HSL fields come from different
  sources.
- [x] Reject HSL coin patches in non-coin signal modes with actionable errors.
- [x] Validate every resolved HSL cross-field invariant.
- [x] Update HSL and coin-override documentation and the changelog.

### Validation and release gate

- [x] Add table-driven coverage for every HSL field on both sides.
- [x] Cover global, file, and inline signal-mode transitions and invalid combinations.
- [x] Run focused and broader Python tests.
- [x] Rebuild and verify the Rust extension.
- [x] Run real HSL backtests with public fixtures and compare expected behavior.
- [x] Run deterministic offline fake-live HSL scenarios, including restart behavior.
- [x] Open a regular ready-for-review PR and request Codex review; first pass reviewed
  `295abf6221006e8e1cbc0a73519438d670a6b57b`.
- [x] Address the five first-pass findings in
  `c151cbb44642228204dc8e47e765093d7d0f60f3`, add regressions, and rerun the full
  Python suite, full Rust library suite, fake-live tests, and controlled HSL backtests.
- [ ] Address findings and repeat review plus validation until the current head is green.
- [ ] Merge the reviewed current head and mark this section complete.

## Reviewer loop used for every PR

1. [ ] Refresh the target branch and establish the exact target and head SHAs.
2. [ ] Run author validation on the exact proposed head.
3. [ ] Request Codex review and inspect thread-level resolution state.
4. [ ] Classify each finding against code, tests, and the documented contract.
5. [ ] Fix actionable findings and add regression coverage. If a finding is incorrect,
   reply with evidence and leave it open for reviewer reconsideration.
6. [ ] Rerun proportional validation and request another review on the new head.
7. [ ] Repeat until CI is green and no actionable review thread remains unresolved.
8. [ ] Merge only the reviewed head, then begin the next PR from the updated target branch.

## Progress

- [x] Audit completed and parameter-policy decisions confirmed.
- [x] Foundation branch created from current `origin/master`.
- [x] Execution checklist persisted.
- [x] PR 1 implementation and author validation complete.
- [x] PR 1 under review.
- [x] PR 1 merged as `75116d1954c19e2bf54ef8313c18d332815a00e1`.
- [x] PR 2 under review.
- [x] PR 2 merged as `9cc3cad58c566364d0ef5e01e9ef6457132246ea`.
- [x] PR 3 first-pass findings addressed; current-head re-review pending.
- [ ] PR 3 merged.
