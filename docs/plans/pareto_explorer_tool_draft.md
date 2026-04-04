# Pareto Explorer Tool Draft

## Goal

Add a new user-facing CLI tool for selecting a single "best" candidate from a Pareto front
directory of JSON artifacts.

Primary entrypoint:

```bash
passivbot tool pareto <pareto_dir> [options]
```

Recommended alias set:

- `passivbot tool pareto`
- `passivbot tool pareto-explorer`

The tool should:

1. Load Pareto members from a directory of `.json` files.
2. Optionally filter the front with CLI limit expressions using the same syntax family as
   optimizer CLI limits.
3. Select one candidate using a named decision method.
4. Print a concise summary of:
   - selected file path / hash
   - method used
   - candidates loaded vs retained after filtering
   - objective values for the selected candidate
   - method-specific score / rationale

## Scope

This tool is a selector/explorer, not a mutator:

- read-only
- no pruning / rewriting
- no web UI
- no optimizer reruns
- no artifact conversion

It should be built from scratch, but may reuse small helper patterns from:

- `src/pareto_store.py`
- `src/tools/pareto_dash.py`
- `src/config/scoring.py`
- `src/config/limits.py`
- `src/limit_utils.py`

## Inputs

Accepted path forms:

1. direct Pareto directory:
   - `optimize_results/.../pareto`
2. optimize run directory containing `pareto/`
   - `optimize_results/.../`

Expected artifact format:

- one `.json` per Pareto member
- new metrics schema preferred:
  - `metrics.objectives`
  - `metrics.stats`
  - `suite_metrics` for suite runs
- legacy objective keys (`w_i`) should still be readable when practical

## Objective Surface

Selection methods should operate on the run's configured scoring objectives by default.

Objective decoding rules:

1. Read scoring spec from `optimize.scoring`.
2. Decode `metrics.objectives` into raw user-facing values using the stored goal (`min` / `max`).
3. Key the decoded objective map by canonical metric name.

Default method surface:

- all scoring objectives in stored order

Optional narrowing:

- support `--objectives metric_a,metric_b,...` to restrict the active objective set for methods
  where that is meaningful

## Limit Filtering

The tool should support limit syntax modeled after optimizer CLI limits:

- `--limit 'adg_strategy_pnl_rebased>0.0'`
- `--limit 'drawdown_worst_hsl<=0.35'`
- `--limit 'loss_profit_ratio outside_range [0.05,0.7]'`
- `--limits '[{...}, {...}]'`

Recommended semantics:

1. `--limit` expressions are keep-conditions, matching current optimizer CLI and `pareto_store.py`
   user expectations.
2. `--limits` accepts canonical list payloads using internal `penalize_if` semantics.
3. Filtering happens before candidate selection.

Metric resolution order for filtering:

1. explicit `stat=...` on the limit, if provided
2. objective raw value, if the metric is one of the scoring objectives
3. suite aggregated metric value, if present
4. `mean` stat fallback from flattened stats

This is more intuitive for Pareto artifact exploration than optimizer-internal fallback rules
that vary by comparison direction.

## Selection Methods

### 1. `knee`

Purpose:

- choose a balanced compromise point without requiring user weights

Implementation target:

- normalize all active objectives to utility space `[0, 1]`, where `1` is best
- build anchor points from the per-objective best candidates
- score each candidate by distance above the anchor hyperplane
- if anchor geometry is degenerate, fall back to maximin utility

CLI:

- `-m knee`

### 2. `reference`

Purpose:

- choose the candidate closest to user-specified aspiration targets

Implementation target:

- repeatable `--target metric=value`
- active objectives default to target metrics
- minimize weighted Euclidean distance to the normalized target point

CLI:

- `-m reference`
- `--target adg_strategy_pnl_rebased=0.0012`
- `--target drawdown_worst_hsl=0.28`

### 3. `ideal`

Purpose:

- choose the candidate closest to the observed ideal point on the current front

Implementation target:

- normalize to utility space
- minimize weighted Euclidean distance to the all-best point

CLI:

- `-m ideal`

### 4. `utility`

Purpose:

- choose via weighted scalarization once explicit preferences are known

Implementation target:

- normalize to utility space
- maximize weighted mean / weighted sum
- default equal weights when omitted

CLI:

- `-m utility`
- `--weight adg_strategy_pnl_rebased=4`
- `--weight drawdown_worst_hsl=2`

### 5. `lexicographic`

Purpose:

- enforce strict priority ordering between objectives

Implementation target:

- sort candidates by prioritized objective list in order
- use active objective order from `--priority` if given
- otherwise fall back to stored scoring order

CLI:

- `-m lexicographic`
- `--priority adg_strategy_pnl_rebased,drawdown_worst_hsl,peak_recovery_hours_hsl`

### 6. `outranking`

Purpose:

- choose by pairwise wins/losses instead of a direct scalar score

Implementation target:

- simplified PROMETHEE-II style net flow
- normalize to utility space
- compare each candidate pairwise across weighted objectives
- select highest net flow

CLI:

- `-m outranking`
- optional `--weight ...` support

## Method Ranking For Passivbot

Ranked by practical usefulness for Passivbot's typical 5+ objective fronts:

1. `reference`
   - best fit when the operator already knows acceptable ADG / drawdown / recovery targets
   - most operationally useful for screening optimizer results into a desired regime
2. `knee`
   - best default when no explicit targets exist
   - tends to pick balanced compromise configs rather than extreme specialists
3. `ideal`
   - simple and robust
   - good default fallback when `reference` targets are not available
4. `lexicographic`
   - strong fit when the user has strict priorities
   - less suitable as a general default for high-dimensional balanced selection
5. `utility`
   - useful but fragile
   - sensitive to weights and normalization assumptions
6. `outranking`
   - worth offering, but lowest priority for everyday Passivbot usage
   - harder to explain and less transparent than `reference` / `knee` / `ideal`

Recommended default:

- `knee`

Recommended practical operator workflow:

1. use `--limit` / `--limits` to enforce must-have constraints
2. use `reference` if you know your target regime
3. otherwise use `knee`
4. use `lexicographic` only when strict objective priority is intentional

## CLI Design

Suggested shape:

```bash
passivbot tool pareto <path> \
  --method knee \
  --limit 'drawdown_worst_hsl<=0.35' \
  --limit 'adg_strategy_pnl_rebased>0.0'
```

```bash
passivbot tool pareto <path> \
  --method reference \
  --target adg_strategy_pnl_rebased=0.001 \
  --target drawdown_worst_hsl=0.25 \
  --weight adg_strategy_pnl_rebased=2
```

```bash
passivbot tool pareto <path> \
  --method lexicographic \
  --priority adg_strategy_pnl_rebased,drawdown_worst_hsl,peak_recovery_hours_hsl
```

Core flags:

- optional positional `path` (defaults to newest `optimize_results/.../pareto`)
- `-m`, `--method`
- `-l`, `--limit`
- `--limits`
- `--objectives`
- `--weight`
- `--target`
- `--priority`

Optional presentation flags:

- `--show-top N` to print ranked shortlist, not only the top candidate
- `--json` to print machine-readable result metadata

These presentation flags are nice-to-have, not required for phase 1.

## Output Contract

Default human-readable output should include:

1. resolved Pareto directory
2. number of candidates loaded
3. number retained after limits
4. selection method and method-specific summary
5. selected file path / filename
6. objective values for the selected candidate

Example shape:

```text
Pareto directory: optimize_results/.../pareto
Loaded candidates: 182
Retained after limits: 47
Method: knee
Selected: optimize_results/.../pareto/abc123.json
Score: 0.183241

Objectives:
  adg_strategy_pnl_rebased (max): 0.00121
  drawdown_worst_hsl (min): 0.274
  peak_recovery_hours_hsl (min): 188.0
  ...
```

## Tests

Add targeted tests for:

1. path resolution (`run_dir` vs `pareto_dir`)
2. objective decoding from Pareto JSON
3. limit filtering with `--limit` keep-condition syntax
4. each selection method on a synthetic front:
   - `reference`
   - `knee`
   - `ideal`
   - `utility`
   - `lexicographic`
   - `outranking`
5. CLI parser wiring and aliases

Synthetic fixtures should use 3 objectives with known geometry so method choices are stable.

## Docs

User-facing docs to update after implementation:

- `docs/tools.md`
- `docs/optimizing.md`
- `CHANGELOG.md`

The docs should explain:

- what the tool does
- when to use each method
- why `knee` is the default
- that limits are applied before method selection
