# Config Loader Redesign Handoff

## Purpose

Redesign Passivbot's config loading / parsing / migration / hydration / validation flow so it is:

- predictable
- auditable
- explicit about every transformation
- easy to evolve as the schema grows
- aligned with the future Rust strategy runtime / schema redesign

This handoff is written for a fresh agent with no prior thread context.

## Current Problem Summary

The current config system has outgrown incremental patching.

Observed issues:

1. There are effectively two schema/default sources:
   - `src/config_utils.py:get_template_config()`
   - `configs/template.json`
   These drift and produce misleading add/remove logs when loading configs.

2. `load_config()` / `format_config()` mix too many responsibilities:
   - flavor detection
   - legacy migration
   - default hydration
   - validation
   - target/runtime adaptation
   - internal alias injection
   - pruning / cleanup

3. Internal runtime alias keys are added into the same object as user config, then later removed.
   This creates noisy and misleading logs such as "Removed unused key ..." for keys the user never
   actually wrote.

4. Backward compatibility logic is scattered through `config_utils.py` and keeps growing.

5. Logging is too procedural and leaf-oriented.
   - useful semantic changes are mixed with internal bookkeeping
   - loading the official example/template config can emit surprising churn

6. The current shape is poorly aligned with the future direction in:
   - [rust_strategy_runtime_refactor.md](/Users/eiriknarjord/repos/passivbot-3/docs/plans/rust_strategy_runtime_refactor.md)

## What The User Wants

These points were explicitly requested and should be treated as design constraints:

1. Single source of truth for defaults/schema.
2. Example configs should not be used as hidden schema/default sources.
3. Backward compatibility transforms should live in a dedicated, structured area.
4. One clean orchestrator should call a sequence of pure helper transforms.
5. Raw user config must be preserved unmodified as `_raw`.
6. There should be target-specific cleanup/projection functions:
   - live
   - backtest
   - optimize
   - monitor
   - others as needed
7. Config handling must be explicit and noisy in the right way:
   - every meaningful rename
   - every section added from defaults
   - every value changed
   - but avoid flooding logs with irrelevant leaf noise

## Non-Goals

1. Do not rewrite unrelated runtime logic.
2. Do not change trading behavior as part of the refactor except where config normalization bugs are
   currently causing wrong behavior.
3. Do not attempt full Rust-driven config generation in phase 1.
4. Do not remove all legacy support immediately.
   - Keep existing truly needed legacy migration support.
   - Isolate it so it stops bloating the main loader.

## Recommended High-Level Design

Split the config system into these layers:

1. Schema / defaults
2. Parsing
3. Migration
4. Canonical normalization
5. Validation
6. Target projection
7. Runtime compilation
8. Logging / transform reporting

### 1. Schema / Defaults

One canonical in-code source of truth.

Requirements:

- no behavioral dependence on `configs/template.json`
- stable hardcoded defaults in one module
- target-independent canonical shape
- suitable as the base for help/docs generation later

Recommended file:

- `src/config/schema.py`

Likely contents:

- canonical default config tree
- helper metadata for sections/paths
- maybe schema constants / enums

### 2. Parsing

Keep file parsing separate from normalization.

Recommended file:

- `src/config/parse.py`

Responsibilities:

- load hjson/json
- preserve exact raw content as Python object
- no migration
- no aliasing
- no defaults

### 3. Migration

All backward compatibility belongs here.

Recommended package:

- `src/config/migrations/`

Suggested files:

- `__init__.py`
- `detect.py`
- `legacy_pb_multi.py`
- `legacy_v7.py`
- `renames.py`
- `versions.py`

Responsibilities:

- detect flavor/version
- migrate old config shapes to canonical shape
- emit structured transform records
- no runtime alias injection

Important:

- versioned migrations are preferred over ad hoc "if key exists then mutate" scattered everywhere
- add explicit `config_version` support if feasible in phase 1

### 4. Canonical Normalization

Normalize only to canonical user-facing config.

Recommended file:

- `src/config/normalize.py`

Responsibilities:

- hydrate missing defaults from schema
- coerce canonical types
- normalize canonical section structures
- normalize section-local details such as:
  - monitor config
  - pymoo config
  - bot-side canonical params

This layer must not create internal runtime aliases.

### 5. Validation

Validation should be explicit and separate from normalization.

Recommended file:

- `src/config/validate.py`

Responsibilities:

- ensure required canonical fields are present
- enforce domain/range constraints
- fail loudly with actionable paths/messages

Examples:

- required monitor booleans
- forager span constraints
- HSL constraints
- optimizer nested config constraints

### 6. Target Projection

Project canonical config to the target runtime shape.

Recommended file:

- `src/config/project.py`

API sketch:

```python
def project_config(config: dict, target: str) -> dict:
    ...
```

Targets to support:

- `canonical`
- `live`
- `backtest`
- `optimize`
- `monitor`

Rules:

- projections prune unrelated sections
- canonical config remains intact
- projections should be deterministic and pure

Examples:

- live projection may drop optimize/backtest-heavy sections
- optimize projection may drop monitor/logging-only sections if not needed
- monitor projection may be tiny

### 7. Runtime Compilation

Compile projected config into runtime/internal form for consumers.

Recommended file:

- `src/config/runtime_compile.py`

Responsibilities:

- create internal alias keys only when needed by runtime boundaries
- prepare Rust/orchestrator payload fields
- prepare backtest-specific compiled fields

This is where keys like:

- `forager_volume_ema_span` -> internal/runtime `filter_volume_ema_span`

should be created if still needed.

They must not be added to canonical config during `load_config()`.

### 8. Logging / Transform Reporting

Recommended file:

- `src/config/transform_log.py`

Current `_transform_log` concept is useful and should be kept.

Improve it by distinguishing:

- semantic user-visible transforms
- internal compile/runtime transforms

Suggested transform record shape:

```python
{
    "stage": "migrate|normalize|validate|project|compile",
    "action": "rename|add_section|add_default|remove_unused|update_value|normalize",
    "path": "optimize.pymoo.algorithm",
    "old": ...,
    "new": ...,
    "detail": "...",
}
```

### Logging Policy

CLI logging should be coalesced and user-meaningful.

Good examples:

- `Added missing optimize section from defaults`
- `Renamed 4 legacy bot forager keys under bot.long`
- `Normalized optimize.pymoo.algorithm: auto`
- `Updated live.approved_coins from scalar override to side dict`

Avoid logging every leaf added under a whole missing section unless debug mode is enabled.

Detailed leaf transforms can still live in `_transform_log`.

## Proposed Public API

Keep `config_utils.py` as a thin compatibility facade initially.

Recommended new APIs:

```python
def load_raw_config(path: str) -> dict: ...

def normalize_config(
    raw_config: dict,
    *,
    base_config_path: str = "",
    live_only: bool = False,
    verbose: bool = True,
) -> dict: ...

def validate_config(config: dict) -> None: ...

def project_config(config: dict, target: str) -> dict: ...

def compile_runtime_config(config: dict, runtime: str) -> dict: ...

def load_config(path: str, live_only: bool = False, verbose: bool = True) -> dict:
    # compatibility facade
    ...
```

Target behavior:

- `load_config()` returns canonical config, not runtime-aliased config
- runtime consumers explicitly request projection/compile helpers

## Example Config Files Policy

Current `configs/template.json` should stop being the implicit behavior anchor.

Recommended:

1. Move user-facing examples under `configs/examples/`
2. Rename examples to describe behavior, not role

Examples:

- `configs/examples/diversified_long_only_top20.json`
- `configs/examples/hsl_single_coin_btc.json`
- `configs/examples/optimize_many_objective_pymoo.json`

Important:

- examples are examples
- defaults/schema come from code only

## Concrete Phased Rollout

### Phase 0: Design Lock

Deliverables:

- this plan reviewed and accepted
- agree that canonical schema source is code, not `template.json`
- agree on target split:
  - canonical
  - project
  - compile

### Phase 1: Single Source Of Truth

Goal:

- eliminate schema drift between `get_template_config()` and `configs/template.json`

Deliverables:

1. Create `src/config/schema.py`
2. Move current canonical defaults there
3. Make `config_utils.get_template_config()` a thin wrapper or compatibility shim
4. Move `configs/template.json` to `configs/examples/...`
5. Add a test that canonical defaults and example config are not implicitly coupled

Acceptance criteria:

- loading the official example config does not emit bogus add/remove churn
- no duplicate schema/default sources remain

### Phase 2: Extract Migrations

Goal:

- isolate backward compatibility logic

Deliverables:

1. Create `src/config/migrations/`
2. Move:
   - flavor detection
   - legacy renames
   - old optimize/bot/live field migrations
3. Add structured migration reports
4. Keep `_raw` unchanged

Acceptance criteria:

- `config_utils.py` no longer contains the growing pile of legacy migration helpers
- legacy test coverage remains green

### Phase 3: Canonical Normalize + Validate Split

Goal:

- make normalize and validate separate pure stages

Deliverables:

1. `normalize.py`
2. `validate.py`
3. move pymoo/monitor/bot canonical normalization here
4. validation errors reference canonical paths

Acceptance criteria:

- canonical normalization does not create internal alias keys
- validation failures are deterministic and path-specific

### Phase 4: Target Projection

Goal:

- support focused, pruned configs per runtime target

Deliverables:

1. `project.py`
2. target projections for:
   - live
   - backtest
   - optimize
   - monitor
3. tests verifying each target contains only what it needs

Acceptance criteria:

- live config can be projected without optimize/backtest noise
- optimize config can be projected without monitor noise

### Phase 5: Runtime Compilation

Goal:

- move internal aliasing / Rust boundary prep out of canonical loading

Deliverables:

1. `runtime_compile.py`
2. move runtime alias injection here
3. update consumers to call compile helpers explicitly

Acceptance criteria:

- canonical config no longer shows internal alias churn
- runtime/backtest behavior remains unchanged

### Phase 6: Logging Cleanup

Goal:

- user-visible logs reflect meaningful config changes, not internal churn

Deliverables:

1. coalesced transform summary logger
2. structured `_transform_log`
3. optional verbose/debug leaf-level dump

Acceptance criteria:

- loading a canonical current config produces short, comprehensible logs
- loading an old config produces explicit migration logs

## Testing Strategy

Add tests by layer, not only end-to-end.

### Unit Tests

1. schema default shape tests
2. migration tests per legacy format
3. normalize tests
4. validate tests
5. project tests
6. runtime compile tests

### Regression Tests

Specifically cover current pain points:

1. loading the official example config should not log phantom added/removed keys
2. canonical config should not contain internal alias keys
3. runtime compile may contain internal keys if required
4. `_raw` remains byte-for-byte semantically unchanged from parse result
5. target projections prune unrelated sections

### Golden / Snapshot Tests

Recommended for:

- migration transform logs
- projected config outputs
- example profile normalization

## Invariants To Preserve

1. `_raw` must persist the unmodified parsed config object.
2. Canonical config paths must be stable and user-facing.
3. Runtime alias keys must not leak into canonical config.
4. Config validation must fail loudly on required invalid inputs.
5. No silent fallback to example file contents.
6. Backward compatibility remains explicit and test-covered.

## Open Questions / Decisions To Confirm

These should be decided before coding deeply:

1. Should `load_config()` return canonical config only, or canonical + projected target by default?
   Recommended: canonical only.

2. Should target projection happen inside command entrypoints rather than the loader?
   Recommended: yes.

3. Should `config_version` be mandatory for new files?
   Recommended:
   - phase 1: optional, inferred for old configs
   - phase 2+: write it into new saved/example configs

4. Should user-facing example configs be generated from schema defaults plus named overrides?
   Recommended: yes, eventually.

5. Should transform logging have levels:
   - summary
   - detailed
   Recommended: yes.

## Relationship To Rust Strategy Runtime Refactor

This redesign should prepare for the future split described in:

- [rust_strategy_runtime_refactor.md](/Users/eiriknarjord/repos/passivbot-3/docs/plans/rust_strategy_runtime_refactor.md)

Important alignment points:

1. Do not keep growing one monolithic Python-side bot schema normalizer.
2. Design the new config system so strategy/shared-engine separation can slot in later.
3. Keep canonical config independent from Rust boundary payload quirks.
4. Make future strategy param specs pluggable rather than hardcoded throughout `config_utils.py`.

## Recommended First Implementation Slice

If starting immediately, the best first slice is:

1. introduce `src/config/schema.py`
2. make it the only schema/default source
3. move `configs/template.json` to `configs/examples/...`
4. stop injecting internal alias keys during canonical load
5. keep existing `config_utils.load_config()` API as a facade while routing through the new schema

Why this first:

- it removes the most visible current confusion
- it is high leverage
- it does not require finishing the whole redesign before user value appears

## Short Checklist For The Implementing Agent

1. Read:
   - `AGENTS.md`
   - `docs/ai/principles.yaml`
   - `docs/ai/error_contract.md`
   - this file
   - `docs/plans/rust_strategy_runtime_refactor.md`
2. Confirm current branch and recent commits.
3. Start with phase 1 only.
4. Keep runtime behavior stable.
5. Add regression tests before broad refactors.
6. Do not mix strategy-schema redesign into phase 1 implementation.

## Suggested Deliverable Sequence

1. PR/commit 1:
   - schema source extraction
   - example config move/rename
   - tests for schema drift removal

2. PR/commit 2:
   - migration module extraction
   - compatibility tests

3. PR/commit 3:
   - canonical/project/compile separation
   - target projections

4. PR/commit 4:
   - logging cleanup
   - docs/help updates

## Final Recommendation

Do not keep patching the current monolithic `config_utils.py` as the long-term answer.

The right move is:

- a phased refactor
- one schema source of truth
- canonical config distinct from runtime-compiled config
- isolated backward compatibility
- target-specific projections
- structured, semantic transform logging

This redesign is justified now and will reduce future churn, especially as the Rust-side strategy
runtime and schema model become more modular.
