# Pymoo Backend Handoff

Detailed handoff for implementing `pymoo` as an additional optimizer backend in Passivbot, using PR `#574` as a starting point without replacing the existing DEAP backend immediately.

## Goal

Add a `pymoo` optimizer backend that can live alongside the current DEAP backend for a transition period.

Constraints:

- Preserve current DEAP behavior as the reference contract.
- Do not merge a `pymoo` port by replacing DEAP first and fixing semantics later.
- Preserve Passivbot-specific optimizer semantics:
  - stepped bounds
  - hard-fail behavior on trading-critical/backtest input errors
  - single-objective configs
  - existing output structure and downstream analysis tools

## Why This Approach

PR `#574` has useful modularization work, but it is not safe as an immediate replacement because it changes optimizer behavior in ways that are meaningful for Passivbot.

Confirmed risks in the PR:

- `src/optimization/bounds.py` drops the third bound element and treats stepped params as continuous.
- `src/optimization/problem.py` catches evaluator exceptions and converts them into huge penalty objectives instead of failing the run.
- `src/optimize.py` assumes `n_obj >= 2`, which breaks single-objective optimization configs still supported elsewhere in the repo.

The right move is:

1. Reuse the good structure from the PR.
2. Keep DEAP as the default and the behavioral baseline.
3. Introduce `pymoo` behind an explicit backend switch.
4. Prove parity on the important semantics before considering DEAP removal.

## Branching

Create a fresh feature branch. Do not implement this work on top of the current local dirty branch.

Suggested branch name:

- `feature/pymoo-backend-alongside-deap`

## Recommended Source Material

Read these first:

1. `AGENTS.md`
2. `docs/ai/principles.yaml`
3. `docs/ai/error_contract.md`
4. `docs/ai/code_review_prompt.md`
5. `docs/ai/commands.md`

Then inspect:

1. Current `master` optimizer flow in `src/optimize.py`
2. PR `#574`
3. Current optimizer tests under `tests/optimization/`
4. `src/pareto_store.py`
5. `src/tools/pareto_dash.py`

## High-Level Strategy

Keep one shared optimizer entrypoint and split only the algorithm-specific execution.

Target shape:

- shared front-end in `src/optimize.py`
- backend-neutral helpers in `src/optimization/`
- backend-specific runners in a new backend package

Suggested structure:

- `src/optimization/backends/__init__.py`
- `src/optimization/backends/deap_backend.py`
- `src/optimization/backends/pymoo_backend.py`

The shared front-end should continue to own:

- config loading and formatting
- CLI handling
- suite preparation
- shared-memory dataset setup
- results directory naming
- starting-config loading
- backend selection

The backend layer should own:

- algorithm object creation
- population initialization
- mutation/crossover wiring
- generation/evaluation loop
- backend-specific termination logic

## What To Reuse From PR #574

These modules are good starting points, but should be adapted rather than copied blindly:

- `src/optimization/problem.py`
- `src/optimization/evaluator.py`
- `src/optimization/callback.py`
- `src/optimization/output.py`
- `src/optimization/repair.py`

Likely useful ideas from the PR:

- explicit `Problem` object
- explicit evaluator objects for single and suite modes
- separate callback/output modules
- modular optimizer package layout

Do not inherit these semantics unchanged:

- ignoring stepped bounds
- swallowing evaluator exceptions
- assuming all optimization is many-objective
- replacing the top-level optimizer flow wholesale

## Backend Selection Design

Add an explicit backend selector.

Preferred config key:

- `config.optimize.backend`

Allowed values:

- `deap`
- `pymoo`

Preferred behavior:

- default to `deap`
- allow CLI override via something like `--optimizer-backend deap|pymoo`
- centralize validation in config formatting/parsing, not downstream

This makes rollout reversible and lets both backends run from the same config family.

## Implementation Phases

## Phase 1: Backend Skeleton Without Behavior Changes

Goal: introduce backend separation while preserving current DEAP behavior.

Tasks:

1. Extract current DEAP algorithm flow from `src/optimize.py` into `src/optimization/backends/deap_backend.py`.
2. Keep DEAP path behaviorally identical.
3. Update `src/optimize.py` to dispatch to the selected backend.
4. Add backend selection tests.

Acceptance criteria:

- existing DEAP tests still pass
- running with no backend specified behaves exactly like today
- `--optimizer-backend deap` is explicit and equivalent to default

## Phase 2: Land Pymoo Backend Behind the Switch

Goal: add a non-default `pymoo` backend without removing DEAP.

Tasks:

1. Create `src/optimization/backends/pymoo_backend.py`.
2. Reuse the PR’s modular `Problem`, `Evaluator`, `Callback`, and `Output` ideas.
3. Make the backend callable from the shared entrypoint.
4. Keep all output paths compatible with current tools:
   - `optimize_results/...`
   - `pareto/`
   - `all_results.bin`

Acceptance criteria:

- `--optimizer-backend pymoo` runs
- DEAP remains default
- output files are readable by existing tooling

## Phase 3: Restore Passivbot Semantics In The Pymoo Backend

This phase is the important one. Do not skip it.

### 3A. Stepped Bounds

Passivbot currently supports bounds of the form:

- `[low, high]`
- `[low, high, step]`

The `pymoo` backend must preserve this semantics.

Recommended options, in order:

1. Integer-index encoding for stepped parameters
2. Exact step-snapping repair/operator logic

Preferred approach:

- represent stepped params in an internal parameter schema
- for stepped params, optimize an integer index `0..n_steps`
- convert between optimizer-space and config-space explicitly

Why this is preferred:

- exact grid preservation
- no floating drift
- integer-like params such as `n_positions` remain valid by construction
- easier parity against old DEAP behavior

Needed work:

- introduce a backend-neutral parameter schema module
- distinguish:
  - continuous params
  - stepped real params
  - integer params
- provide:
  - config -> optimizer vector
  - optimizer vector -> config
  - random initialization
  - validation

If repair-based snapping is used instead:

- snapping must be exact
- tests must prove all sampled/mutated values stay on-grid
- integer-like params must never become fractional in config-space

### 3B. Error Handling

The `pymoo` backend must not convert evaluator errors into giant objective vectors.

Required behavior:

- if required backtest/evaluator input fails, raise immediately
- let the optimization run fail loudly
- do not continue with synthetic penalty scores for hard failures

Action:

- remove broad catch-and-penalize behavior from the PR-derived `Problem`
- keep exceptions visible with actionable context

### 3C. Single-Objective Support

Current Passivbot config and tests still support single-objective scoring.

Required behavior:

- `optimize.scoring = ["adg"]` must keep working

Implementation options:

1. Use a `pymoo` algorithm that supports one objective cleanly in this backend.
2. Special-case the `pymoo` path while preserving the shared API.

Do not:

- silently reject single-objective configs repo-wide
- require users to pad fake objectives

### 3D. Stable Output Contract

The `pymoo` backend must produce data that existing Passivbot tooling can read.

Required invariants:

- `pareto/` JSON structure remains compatible
- `all_results.bin` remains compatible enough for `pareto_dash.py`
- objective naming is stable
- best-config selection works for both backends

The PR’s callback/output logic can be reused if it preserves existing schema expectations.

## Phase 4: Backend Parity Tests

Add tests that compare semantics, not exact Pareto sets.

Key idea:

- evolutionary search will not produce identical fronts across libraries
- the goal is semantic parity, not bit-for-bit identity

Add tests for:

1. stepped bound preservation
2. integer-like parameter preservation
3. single-objective execution
4. hard-fail behavior on evaluator exceptions
5. limit handling
6. starting-config ingestion
7. result file schema compatibility
8. suite mode basic execution

Important comparison style:

- same config
- same dataset fixture or mock evaluator
- same seeded starting configs
- compare validity and contract adherence
- do not assert identical optimization trajectories

## Detailed File Plan

Suggested changes:

### `src/optimize.py`

Keep as the shared CLI/orchestration entrypoint.

Refactor responsibilities:

- parse config and CLI
- choose backend
- prepare data/shared memory
- construct evaluator context
- call selected backend runner
- perform common cleanup

Avoid:

- embedding all algorithm logic directly here

### `src/optimization/backends/deap_backend.py`

Lift current DEAP logic here with minimal behavior changes.

Own:

- DEAP creator/toolbox setup
- DEAP population generation
- DEAP evaluation loop
- DEAP hall-of-fame/Pareto flow

### `src/optimization/backends/pymoo_backend.py`

Own:

- `pymoo` problem construction
- algorithm selection
- initialization
- termination
- callback wiring

This module should not own data prep or config parsing.

### `src/optimization/parameter_space.py`

Recommended new module.

Purpose:

- one canonical optimizer parameter schema for both backends

Suggested responsibilities:

- parse optimize bounds into structured parameter specs
- preserve step semantics
- map config <-> optimizer vector
- validate vectors and configs

This avoids duplicating bound semantics across DEAP and `pymoo`.

### `src/optimization/problem.py`

Keep only if it remains narrowly focused on `pymoo`.

Required edits versus PR:

- no broad exception swallowing
- integrate with the shared parameter schema
- ensure scoring/limits still map correctly

### `src/optimization/repair.py`

Only keep if it enforces real Passivbot semantics.

If using integer-index encoding for stepped params, this module may become much simpler or only apply to continuous significant-digit rounding.

### `src/optimization/callback.py`

Keep if it writes outputs in a backend-neutral schema.

Prefer naming and structure that do not imply `pymoo`-only ownership if the module becomes generally useful.

## Testing Plan

Minimum targeted tests:

1. Existing optimizer tests that should continue to pass for DEAP.
2. New backend-selection tests.
3. New `pymoo`-specific tests for:
   - stepped params remain on-grid
   - evaluator exceptions abort
   - one-objective configs run
   - output schema remains compatible

Recommended additional regression tests:

1. A small fixture with:
   - stepped float param
   - integer-like param
   - continuous param
2. A suite fixture with at least two scenarios.
3. A failure fixture where evaluator raises.

Smoke runs when relevant:

- one-coin short-window optimization under DEAP
- same config under `pymoo`
- compare:
  - no invalid fractional integer params
  - no off-grid stepped values
  - no schema breakage

## Suggested Acceptance Criteria Before Switching Default

Do not switch default from DEAP to `pymoo` until all are true:

1. `pymoo` preserves stepped-bound semantics.
2. `pymoo` preserves hard-fail behavior for evaluator/backtest failures.
3. `pymoo` supports single-objective configs.
4. `pymoo` output is readable by existing Passivbot analysis tooling.
5. Targeted parity tests exist and pass.
6. Real smoke runs show sane output on representative configs.

Only after that:

1. consider switching default to `pymoo`
2. leave DEAP available for one more transition window
3. remove DEAP later in a separate cleanup PR

## What Not To Do

- Do not merge a `pymoo` rewrite that replaces DEAP immediately.
- Do not accept silent behavior drift for stepped bounds.
- Do not catch evaluator exceptions and continue the optimization run.
- Do not push config fallback logic downstream.
- Do not change result schemas casually; downstream tools already depend on them.
- Do not judge parity by “the tests pass and `pymoo` runs once.”

## Open Design Decisions

The implementing agent should resolve these explicitly:

1. Parameter encoding:
   - integer-index encoding for stepped params
   - or exact repair/snap model
2. Single-objective `pymoo` strategy:
   - special algorithm
   - or backend-level wrapper logic
3. Best-config selection:
   - shared helper for both backends
   - or backend-specific selection rule with shared output schema

These decisions should be documented in code comments or a short follow-up note in this plan file if the implementation diverges from the preferred approach.

## Recommended Execution Order For A Fresh Agent

1. Create fresh feature branch.
2. Extract current DEAP flow into a backend module with no behavior change.
3. Add backend switch with DEAP default.
4. Add `pymoo` backend behind the switch.
5. Implement shared parameter schema preserving stepped semantics.
6. Remove evaluator exception swallowing from the `pymoo` path.
7. Add single-objective support.
8. Add targeted parity and regression tests.
9. Run targeted tests and smoke runs.
10. Stop with DEAP still default unless explicitly asked to switch default.

## Deliverable Expectation For The Fresh Agent

A good first implementation branch should end with:

- DEAP still available and still default
- `pymoo` runnable behind explicit selection
- stepped bounds preserved
- hard-fail semantics preserved
- single-objective support preserved
- targeted parity tests added

That is the correct intermediate state. Replacing DEAP should be a later, separate decision.
