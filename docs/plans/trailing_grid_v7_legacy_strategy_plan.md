# Trailing Grid V7 Legacy Strategy Plan

## Status

Draft plan for adding v7 trailing-grid compatibility to v8 as an explicit strategy kind.

This plan is a deliberate compatibility carve-out from the earlier v8 clean-break direction. It does
not propose converting v7 configs to `trailing_martingale`, and it does not propose maintaining a
separate v7 branch. The goal is to let v7 trailing-grid users move onto the v8 runtime with one repo,
one virtual environment, and one operational surface while keeping v7 trading behavior isolated and
deprecatable.

## Summary

Add a new v8 strategy kind:

```text
trailing_grid_v7
```

Implement it as a normal Rust strategy module alongside `trailing_martingale` and `ema_anchor`.
Then add a migration helper that turns legacy v7 configs into canonical v8 configs using:

```text
live.strategy_kind = "trailing_grid_v7"
bot.<side>.strategy.trailing_grid_v7
```

The migration is structural. It preserves the old optimized parameters under a legacy strategy
subtree instead of interpreting them as v8 `trailing_martingale` parameters.

## Motivation

The alternative to this plan is some period of separate v7 and v8 support. That creates two kinds of
ongoing cost:

1. Maintainer cost:
   - identifying which v8 fixes/features must also be backported to v7
   - resolving conflicts across a large and moving diff
   - testing two release lines

2. Operator cost:
   - keeping separate repo clones
   - keeping separate virtual environments
   - running or switching between separate bot installs on small VPSs

V8 already has a strategy runtime abstraction. Using that abstraction for a `trailing_grid_v7`
compatibility strategy puts the burden in one bounded implementation area instead of spreading it
across branch maintenance and operator deployment.

## Goals

1. Let a v7 trailing-grid user migrate to a v8-shaped config without re-optimizing immediately.

2. Preserve v7 trailing-grid order behavior in Rust:
   - `entry_trailing_grid_ratio`
   - `close_trailing_grid_ratio`
   - `close_grid_markup_start`
   - `close_grid_markup_end`
   - linear markup-grid close behavior
   - v7 grid/trailing split behavior

3. Keep v8's strategy runtime model intact:
   - strategy kinds are registered in Rust
   - strategy parameter metadata is exposed by Rust
   - live and backtest use the same Rust strategy dispatch path
   - optimizer bounds resolve through the active strategy metadata

4. Keep the compatibility surface narrow and deprecatable:
   - users can see they are using `trailing_grid_v7`
   - docs can mark it as transitional
   - later removal is a strategy-kind removal, not a schema archaeology exercise

5. Avoid lossy conversion:
   - do not translate v7 parameters into `trailing_martingale`
   - do not silently reinterpret optimized v7 values as different v8 semantics

## Non-Goals

1. Do not maintain a second full v7 config universe inside v8.

2. Do not make `trailing_martingale` accept v7 aliases.

3. Do not add a broad "v7 mode" that changes unrelated v8 runtime behavior.

4. Do not promise support for every pre-v7 historical config shape.

5. Do not keep `trailing_grid_v7` indefinitely without a deprecation policy.

6. Do not re-add v7 strategy fields to shared `BotParams` unless unavoidable. Strategy-only fields
   should live in strategy params.

## Naming

Recommended strategy kind:

```text
trailing_grid_v7
```

Reasoning:

- `v7` is too broad and sounds like an entire platform/runtime mode.
- `legacy_v7` is acceptable, but it hides the actual behavior being preserved.
- `trailing_grid_v7` says exactly what the strategy is: the v7 trailing-grid behavior running
  under the v8 strategy runtime.

No short alias should be added initially. If an alias is later desired, it should be explicit and
documented as a compatibility alias, not discovered accidentally through normalization.

## Canonical Migrated Config Shape

A migrated v7 config should become a normal v8 config:

```json
{
  "config_version": "v8.0.0",
  "live": {
    "strategy_kind": "trailing_grid_v7"
  },
  "bot": {
    "long": {
      "risk": {
        "n_positions": 7,
        "total_wallet_exposure_limit": 1.5,
        "position_exposure_enforcer_threshold": 0.99,
        "total_exposure_enforcer_threshold": 0.985,
        "we_excess_allowance_pct": 0.1
      },
      "forager": {},
      "hsl": {},
      "unstuck": {},
      "strategy": {
        "trailing_grid_v7": {
          "ema_span_0": 385.0,
          "ema_span_1": 620.0,
          "entry": {
            "grid_double_down_factor": 1.39,
            "grid_spacing_pct": 0.02312,
            "grid_spacing_we_weight": 0.6766,
            "grid_spacing_volatility_weight": 17.8,
            "initial_ema_dist": 0.0078,
            "initial_qty_pct": 0.0122,
            "trailing_double_down_factor": 1.0,
            "trailing_grid_ratio": -0.32,
            "trailing_retracement_pct": 0.01498,
            "trailing_retracement_we_weight": 4.958,
            "trailing_retracement_volatility_weight": 37.9,
            "trailing_threshold_pct": 0.00215,
            "trailing_threshold_we_weight": 4.243,
            "trailing_threshold_volatility_weight": 15.2,
            "volatility_ema_span_hours": 1909.0
          },
          "close": {
            "grid_markup_start": 0.01041,
            "grid_markup_end": 0.00241,
            "grid_qty_pct": 0.88,
            "trailing_grid_ratio": -0.07,
            "trailing_qty_pct": 0.89,
            "trailing_retracement_pct": 0.00413,
            "trailing_threshold_pct": 0.0125
          }
        }
      }
    }
  }
}
```

The exact nested names may change during implementation, but the important contract is:

- legacy strategy fields move under `bot.<side>.strategy.trailing_grid_v7`
- shared policy fields move under the existing v8 shared groups
- inactive strategy subtrees may be omitted
- the migration output is canonical v8 shape, not old v7 JSON preserved internally

## Migration Contract

Add an explicit migration helper. Suggested CLI shape:

```bash
passivbot tool migrate-config-v7 input_v7.json output_v8_trailing_grid_v7.json
```

The helper should:

1. Load a v7 config.

2. Detect v7 by `config_version` major version or by the presence of old trailing-grid fields.

3. Build a v8 config from the current v8 template.

4. Set:

```text
config_version = "v8.0.0"
live.strategy_kind = "trailing_grid_v7"
```

5. Move v7 shared bot fields into v8 grouped bot sections:
   - risk
   - forager
   - hsl
   - unstuck

6. Move v7 trailing-grid fields into:

```text
bot.<side>.strategy.trailing_grid_v7
```

7. Move v7 optimize bounds for trailing-grid fields into:

```text
optimize.bounds.<side>.strategy.trailing_grid_v7
```

8. Preserve relevant live/backtest/optimize/logging/monitor fields already supported by v8.

9. Preserve coin overrides by moving legacy flat strategy override keys into the legacy strategy
   subtree.

10. Emit a transform report:
    - source config path
    - source version
    - destination strategy kind
    - moved fields
    - dropped unsupported fields
    - fields requiring manual review

The helper should fail loudly on ambiguous or unsupported inputs. It should not silently fill
trading-critical strategy parameters with neutral defaults when the source config intended a value.

## Rust Implementation

Add a new strategy module:

```text
passivbot-rust/src/strategies/trailing_grid_v7.rs
```

Register it through the existing strategy runtime:

- `passivbot-rust/src/strategies/mod.rs`
- `passivbot-rust/src/strategies/registry.rs`
- `passivbot-rust/src/strategies/spec.rs`
- `passivbot-rust/src/python.rs` strategy-param extraction

Recommended Rust param shape:

```rust
pub struct TrailingGridV7Params {
    pub ema_span_0: f64,
    pub ema_span_1: f64,
    pub entry: TrailingGridV7EntryParams,
    pub close: TrailingGridV7CloseParams,
}
```

Do not put the old strategy-only fields back into shared `BotParams` unless an existing shared
runtime function truly requires it. The legacy module should own the old strategy parameters.

The module should port v7 behavior deliberately:

- v7 `calc_next_entry_long/short` grid/trailing split
- v7 `calc_grid_entry_long/short`
- v7 `calc_trailing_entry_long/short`
- v7 `calc_next_close_long/short` grid/trailing split
- v7 `calc_grid_close_long/short` markup-start/end behavior
- v7 `calc_trailing_close_long/short`

Shared behavior should stay shared:

- WEL/TWEL enforcement
- realized-loss gates
- unstuck behavior
- HSL mode/risk gating
- order sorting and trimming
- trailing bundle state input
- live/backtest orchestration

The strategy module should produce the same order types where possible. If any order type names need
new variants or old variants have changed meaning, define that explicitly in the implementation
plan before coding.

## Strategy Metadata And Optimizer Bounds

Rust strategy metadata must describe `trailing_grid_v7` fully:

- strategy kind name
- per-side defaults
- parameter config paths
- optimizer keys
- default optimize bounds

The old flat v7 optimize keys may remain user-facing for the migration helper input, but canonical
v8 output should store bounds under the active strategy subtree. The flattened optimizer adapter may
still expose keys such as:

```text
long_close_grid_markup_start
long_close_grid_markup_end
long_entry_trailing_grid_ratio
long_close_trailing_grid_ratio
```

Those keys should be valid only when `live.strategy_kind = "trailing_grid_v7"`.

For `trailing_martingale`, the old v7 keys should remain invalid.

## Python Config Pipeline

Integrate with the existing staged config pipeline instead of adding an alternate loader.

Relevant modules:

- `src/config/schema.py`
- `src/config/migrations/`
- `src/config/strategy.py`
- `src/config/strategy_spec.py`
- `src/config/optimize_bounds.py`
- `src/config/param_paths.py`
- `src/config/overrides.py`
- `src/config/runtime_compile.py`
- `src/config/validate.py`

Expected changes:

1. Strategy discovery should include `trailing_grid_v7` through Rust metadata.

2. `get_template_config()` may include defaults for `trailing_grid_v7`, but inactive strategy
   subtrees should be pruned from user-facing prepared configs as they are today.

3. Config validation should require a non-empty active strategy subtree for
   `trailing_grid_v7`.

4. Coin overrides should allow legacy strategy fields only under:

```text
coin_overrides.<coin>.bot.<side>.strategy.trailing_grid_v7
```

5. Old flat v7 override keys may be accepted by the migration helper, but normal v8 loading should
   not silently accept them for non-legacy strategies.

6. Dotted path selectors should resolve through the same active-strategy metadata path as the other
   v8 strategies.

## Live Runtime

Live must continue to build Rust orchestrator input through the strategy runtime path.

Areas that need attention:

- `_strategy_params_to_rust_dict()`
- `_bot_params_to_rust_dict()`
- `is_trailing()`
- initial-entry sizing helpers
- orchestrator EMA/warmup span discovery
- coin override handling
- min-effective-cost gating

Any live helper that currently assumes `trailing_martingale` nested `entry`/`close` fields must
either:

1. become strategy-kind-aware, or
2. call Rust/Python strategy metadata helpers instead of inspecting hardcoded field paths.

Do not add Python-side order behavior. Python should prepare complete inputs; Rust should calculate
orders.

## Backtest Runtime

Backtest should run `trailing_grid_v7` through the same Rust `new_with_strategy_params` /
`parse_strategy_params` path as the other strategies.

Required behavior:

- the active strategy kind is passed to Rust through backtest params
- per-coin strategy params are built from canonical migrated config shape
- EMA span requirements are derived from strategy params
- volatility/log-range requirements match v7 behavior
- live and backtest use equivalent strategy params for the same config

## Optimizer

Optimizer support is part of the compatibility contract. A migrated v7 optimized config should be
able to continue optimizing the v7 strategy kind.

Required behavior:

- optimize bounds for v7 fields resolve only under `trailing_grid_v7`
- fine-tune selectors work against canonical v8 paths
- old v7 flat bounds can be migrated into canonical nested bounds
- start/anchor config compatibility checks compare strategy kind and reject mismatches
- `trailing_martingale`-specific optimizer overrides do not accidentally run on
  `trailing_grid_v7`

Existing overrides such as `lossless_close_trailing`, `forward_tp_grid`, and `backward_tp_grid`
need explicit handling:

- either restrict them to `trailing_martingale`, or
- define separate `trailing_grid_v7` behavior

Silent no-op behavior is not acceptable for optimizer overrides that users request explicitly.

## Documentation

Docs should describe this as deprecated compatibility, not as v7 becoming the default v8 path.

Update user-facing docs only when implementation begins:

- `docs/configuration.md`
- `docs/config_workflow.md`
- examples under `configs/examples/`
- `CHANGELOG.md`

Suggested wording:

```text
V8's canonical strategy is trailing_martingale. For transition, v8 also includes
trailing_grid_v7, a deprecated compatibility strategy preserving v7 trailing-grid behavior. The
migration helper can convert v7 configs into v8-shaped configs using trailing_grid_v7. It does not
convert v7 configs into trailing_martingale.
```

Update AI docs only after the implementation policy is settled:

- `docs/ai/decisions.md`
- `docs/ai/features/strategy_runtime.md`

Those docs currently encode the clean-break decision. This plan should supersede that only after
the project explicitly chooses to implement this compatibility strategy.

## Tests And Validation

Minimum test matrix:

1. Strategy metadata:
   - Rust exposes `trailing_grid_v7` in `get_strategy_kinds()`
   - Rust exposes defaults, config paths, and optimize bounds
   - Python strategy metadata consumes the Rust spec

2. Config migration:
   - canonical v7 example migrates to v8 shape
   - `live.strategy_kind` becomes `trailing_grid_v7`
   - old strategy fields move to the legacy strategy subtree
   - shared fields move to grouped v8 bot sections
   - old bounds move to canonical nested bounds
   - unsupported fields are reported, not silently dropped

3. Config loading:
   - migrated config loads
   - migrated config validates
   - non-legacy v8 configs still reject old v7 trailing-grid fields

4. Rust behavior parity:
   - focused unit tests compare representative v7 order outputs against `trailing_grid_v7`
   - cover long and short
   - cover grid-only, trailing-only, trailing-first split, and grid-first split
   - cover `close_grid_markup_start/end` ordering, including backward TP grids

5. Backtest smoke:
   - migrated v7 example runs through v8 backtest
   - no missing strategy params
   - no unknown optimize bounds

6. Live/orchestrator smoke:
   - orchestrator input accepts `trailing_grid_v7`
   - live strategy params include all required spans and strategy fields
   - `is_trailing()` reports true when v7 trailing entry or close behavior is active

7. Optimizer smoke:
   - migrated v7 optimize config builds an optimization shape
   - legacy bounds resolve under `trailing_grid_v7`
   - `trailing_martingale` configs still reject v7-only bounds

## Rollout

Phase 1: Planning and exact field map

- freeze the list of supported v7 strategy fields
- decide the canonical nested names
- decide default bounds
- decide migration helper command name and output behavior

Phase 2: Rust strategy kind

- add param structs
- add strategy metadata
- add dispatch
- port v7 order logic
- add Rust unit parity tests

Phase 3: Python config and migration

- add migration helper
- wire strategy metadata through config defaults/bounds
- update path resolution and overrides
- add config migration tests

Phase 4: Live/backtest/optimizer validation

- run migrated example through backtest
- run optimizer shape/smoke
- validate orchestrator input generation
- update user docs and changelog

Phase 5: Deprecation policy

- mark `trailing_grid_v7` deprecated from its first release
- document that it exists to bridge v7 users onto v8 infrastructure
- decide whether removal is time-based or version-based after observing adoption

## Deprecation Policy

The strategy should be introduced as deprecated compatibility, not as a first-class long-term
strategy direction.

Recommended policy:

- keep `trailing_grid_v7` for at least one stable v8 release cycle after introduction
- do not add new features to it unless required for live/backtest parity or serious safety issues
- keep bug fixes limited to parity, correctness, and trading-critical safety
- encourage new optimization work to target `trailing_martingale` or newer v8 strategies

## Open Questions

1. Should the migration helper be CLI-only, or should config loading detect v7 and print an
   actionable error pointing to the helper?

2. Should migrated configs keep a breadcrumb such as:

```json
"migration": {
  "source_config_version": "v7.12.0",
  "source_strategy": "trailing_grid",
  "target_strategy": "trailing_grid_v7"
}
```

3. Which v7 fields are explicitly unsupported because they are not strategy behavior?

4. Should examples include one migrated v7 config, or should examples stay focused on canonical v8
   strategies with migration documented separately?

5. Should `trailing_grid_v7` support all old optimizer override names, or should the migration
   helper reject old overrides that do not have a clear legacy strategy meaning?
