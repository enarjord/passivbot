# Nested Config Schema Sketch

This is a rough future-work sketch for moving from the current mostly-flat per-side config layout to a more structured nested schema.

## Motivation

The current `bot.{long,short}` layout is manageable, but it is getting crowded:

- close parameters, entry parameters, HSL parameters, forager parameters, and risk parameters all live side-by-side
- optimizer bounds and config docs have to mirror that flat namespace
- related parameters are harder to scan and reason about together

A nested schema could improve:

- readability
- discoverability
- grouping of related behavior
- long-term maintainability as more subsystems gain structured config

## Non-Goals

- Do not mix this migration with active trading-logic changes.
- Do not let downstream consumers handle both old and new schemas ad hoc.
- Do not introduce local fallback defaults in runtime consumers.

This should be treated as a schema migration project, not a piecemeal refactor.

## Possible Shape

Example direction for `bot.long` / `bot.short`:

```json
{
  "bot": {
    "long": {
      "close": {},
      "entry": {},
      "forager": {},
      "hsl": {},
      "risk": {}
    },
    "short": {
      "close": {},
      "entry": {},
      "forager": {},
      "hsl": {},
      "risk": {}
    }
  }
}
```

Likely groups:

- `close`
- `entry`
- `forager`
- `hsl`
- `risk`

## Recommended Migration Strategy

1. Define one canonical public schema.
2. Keep one centralized config migration/hydration layer in `format_config()`.
3. Convert the nested public schema into one stable internal runtime schema before downstream consumption.
4. Make Python and Rust consumers read only the normalized internal schema.
5. Remove legacy-schema support only after migration tooling and docs are in place.

The key point is that schema compatibility belongs in config formatting code, not in runtime consumers.

## Internal Representation Choice

Two reasonable options:

1. Nested public schema -> normalized flat internal runtime schema
2. Nested public schema -> nested internal runtime schema

Recommendation:

Start with option 1.

Reasons:

- much smaller migration surface
- existing Python and Rust consumers remain simpler to adapt
- optimizer and orchestrator paths can migrate incrementally behind one normalization boundary
- easier rollback if needed

Once the public schema is stable, the internal runtime schema can be reconsidered later.

## Main Work Areas

### 1. Config Formatting

- define nested canonical template
- add centralized migration from flat legacy keys
- add validation for nested groups
- preserve current guarantees about required params and centralized default insertion

### 2. Optimizer Bounds

Need a clear representation for nested parameters.

Likely approach:

- keep `optimize.bounds` flat even if `bot` becomes nested
- map bounds like `long_forager_score_weights_volume` to nested config paths centrally

This keeps optimizer UX stable while avoiding nested bound syntax complexity.

### 3. CLI / Override Paths

Decide whether override syntax becomes:

- `bot.long.forager.volume_ema_span`

or whether overrides continue to target normalized internal keys.

Recommendation:

Support nested override paths at the public schema boundary, then normalize centrally.

### 4. Rust/Python Boundary

- Rust should keep consuming one explicit normalized schema
- PyO3 extraction should not become a second migration layer
- Python should not translate nested/flat variants ad hoc before Rust calls

### 5. JSON API / Backtest Inputs

Any persisted config snapshots, orchestrator JSON payloads, or other external interfaces need an explicit compatibility decision:

- either canonicalize before serialization
- or version the schema explicitly

## Risks

- schema drift between docs, template, optimizer, and runtime
- hidden consumer assumptions about flat keys
- overloading one change with both schema and behavior changes
- long compatibility tail if multiple schemas are tolerated in too many places

## Suggested Rollout

1. Write a concrete schema spec.
2. Add migration tests for flat -> nested canonicalization.
3. Add nested template/config docs.
4. Keep internal runtime schema stable at first.
5. Update CLI overrides and optimizer mapping.
6. Migrate user-facing configs/templates/examples.
7. Remove old public keys only after a deliberate deprecation window.

## Recommended Preconditions

Before doing this migration, it would help to first finish:

- Rust-owned forager/coin-selection cleanup
- strict config-consumer cleanup
- broader audit of downstream `dict.get(..., default)` patterns for required params

That reduces the chance of schema migration interacting badly with silent fallback behavior.
