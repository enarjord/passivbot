# Optimization Campaigns

Read this before planning, running, resuming, or advancing an optimization campaign. The goal is a
useful set of qualified configurations, with evidence for their tradeoffs and a reproducible path
to the result. A long-running process or an attractive GPU Pareto front is an intermediate output.
The final deliverable is a diverse qualified Pareto front, not a single "optimal" config: multiple
objectives have tradeoffs, and the user chooses among them.

Follow [AGENTS.md](../../../AGENTS.md) for authority and publication boundaries. This runbook does
not grant remote-process or live-trading authority. Use [Optimizing](../../optimizing.md) for CLI
semantics, [Backtesting](../../backtesting.md) for data preparation, and the
[metric contract](../features/backtest_metrics.md) for units, clocks, and unavailable values.
The [long-only template recipe](../case_studies/long_only_template_optimization.md) applies these
principles to a concrete configuration style.

## Start With A Campaign Brief

Recover these facts from the user's request and available artifacts before asking questions. Ask
only for consequential missing choices; make routine implementation decisions within the agreed
scope. An existing campaign brief is context, not continuing authority for another host or run.

| Record | Questions it answers |
|---|---|
| Intended artifact | General template, coin-specific config, portfolio, or a family of tradeoffs? |
| Fixed policy | Which sides, strategy, risk mechanisms, exposure, position count, and execution settings must survive every stage and export? |
| Search space | Which authored bounds may vary, which values are anchors, and which changes need discussion? |
| Objectives and limits | Which canonical metrics, directions, scenario reducers, and hard feasibility conditions define success? |
| Evaluation | Frozen date cutoff, history, universe, venues, costs, candle interval, warmup, metric definition, and EW half-life? |
| Resources and authority | Backend, available machine, process scope, resource budget, and permitted data downloads? |
| Progression and finish | Stage budgets, assessment checkpoints, qualification tests, and what constitutes a finished result? |

Keep this brief and the run ledger private unless their exact contents are approved for publication.
To request a future campaign concisely, the user can point to this runbook, name a recipe or desired
style, provide a config, and specify the machine and any changed constraints. The agent should fill
in the procedure, rather than requiring the user to restate it.

## Separate Policy, Search, And Evaluation

1. **Policy defines the intended bot.** Set booleans and runtime settings in the actual bot config;
   fix numeric policy coordinates in the bounds too. Check global inheritance, explicit side pins,
   coin overrides, and scenario patches. Inspect effective values, not just one JSON subtree.
2. **Search defines allowed suggestions.** Preserve the original authored bounds separately from
   fine-tuned or polished copies. Missing authored v9 bounds fix coordinates; removing `-ft` does
   not restore omitted ranges. Derived overrides can still change effective values. Search
   normalized participating Forager weights together, or freeze the group.
3. **Evaluation defines what a score means.** Bind results to data, evaluator/runtime identity,
   costs, dates, scenarios, metric policy, readiness, and fixed runtime settings. A directory name
   or matching metric names alone does not establish comparability.

Use the canonical loader and `passivbot tool config-explain` to inspect policy. Review selector
expansion, actual tunable dimensions, anchor clamping, and effective population in startup logs.
Fix contradictions between starting values, fixed bounds, and the user's intent before spending a
large budget. Do not silently replace user-authored bounds with recommended defaults.

`optimize.fixed_runtime_overrides` changes evaluated runtime values without changing stored bot
values. Prefer encoding a required final policy in the bot itself. If evaluation-only overrides
are intentional, label them and independently backtest the exported config's actual policy.

### Fresh Seeds Versus Resume

Use `--resume` for interruption recovery under a compatible evaluation and search contract. Let the
owner's compatibility checks decide; do not edit identities or transplant fitness to bypass them.
An incompatible checkpoint remains useful as a source of candidate configurations.

Use a fresh run seeded with prior configs when changing evaluation inputs. This includes expanded
history, a later cutoff, a changed suite/reducer, costs, limits/objectives, candle interval, EW
half-life, backend precision, or evaluator implementation. Old metrics may guide a diverse seed
sample; they do not become the new stage's fitness. See the full
[resume identity contract](../../optimizing.md) for exact exclusions and implementation checks.

Bounds deserve special care: indicator bounds affect maximum prepared history and can affect the
shared optimizer trade-activation budget. Broad search, fine-tune, and polish can therefore have
different effective readiness even on the same candles. Treat a derived readiness change as an
evaluation change. Never infer numerical drift from mismatched trade-start policies.

## Establish A Trustworthy Baseline

- Pin source revision, dependencies, loaded Rust artifact/source identity, and GPU precision/runtime
  when applicable. Isolate the campaign from unrelated runs; do not upgrade an active evaluator in
  place. See [Rust extension verification](rust_extension.md).
- Prepare the complete requested history, including maximum warmup, BTC benchmark support, listing
  edges, aliases, market settings, and venue selection. Freeze an explicit cutoff instead of `now`.
- Prefer verified caches. Strict offline preparation requires complete compatible provenance as
  well as candle files. If it refuses an edge, repair or refresh the genuine inputs; do not invent
  listing dates, pre-inception flags, prices, or an offline-provenance claim.
- Verify a copied dataset and its manifest, including aliases/source bindings and timestamp range.
  Different loader routes can carry different metadata. An array that loads is not proof that the
  optimizer accepted the intended source identity. Preserve validation when investigating a route
  mismatch; do not modify factual cache metadata to make it pass.
- Check the owner's cache identity before reusing history: equivalent date spellings can hash
  differently on some revisions. Complete candle caches can still trigger public market-metadata
  refreshes. Copy genuine metadata with its timestamps intact; never fabricate freshness or identity.
- Preflight RAM, swap, shared memory, VRAM, scratch/history growth, and competing workloads. Retain
  resource guards. Run a bounded smoke and a standalone Rust CPU baseline before a large search.

A small GPU/CPU fixture checks that the runtime works on that fixture. It does not qualify the
actual multi-coin, long-history search. Record the baseline's actual dates, completion, activity,
and risk, including failures or unavailable metrics.

## Progress Through Increasing Difficulty

| Stage | Purpose | Evidence for advancing |
|---|---|---|
| Focused adaptation | Fine-tune new or uncertain parameter families around a credible baseline on a recent manageable window. Skip if no such anchor exists. | Useful feasible diversity, diminishing improvements, and representative CPU checks. |
| Broad discovery | Restore the authored full search space and freshly evaluate diverse previous seeds. | Stable useful tradeoffs after enough completed evaluations for the dimensionality. |
| History expansion | Introduce older regimes, preparing genuine additional history at each step. | Seeds survive broader replay; targeted search addresses weaknesses without abandoning the policy. |
| Scenario robustness | Add approved coin subsets, time regimes, and realistic execution/cost stress. Freeze scenario definitions and aggregation. | Complete-suite fitness, acceptable per-scenario behavior, and a viable diverse pool. |
| Confirmation | Use a different RNG seed and, where useful, local polish around several tradeoffs. | Little material new improvement and no persistent local fragility. |
| Final qualification | CPU/Rust evaluate the proposed final pool under one frozen final contract. | Rebuilt feasible CPU Pareto membership and complete declared checks. |

These are decision gates, not a requirement to exhaust a preset iteration count. Assess completed
full evaluations, not wall time alone. Distinguish attempts, rejected prefilters, partial screening,
seed evaluations, and full results; all do not consume or count toward budgets in the same way.
See [optimizer seed and screening semantics](commands.md#optimize).

At each assessment compare feasible count/share, front spread, objective extremes, useful interior
tradeoffs, and change in a fixed-reference convergence measure such as hypervolume. Keep reference
point, directions, scaling, and evaluation policy unchanged for that comparison. Clipping points
for this diagnostic must not remove them from the actual Pareto front. More front members alone
does not establish better solutions; a small aggregate gain can hide a useful new extreme.

Choose a minimum discovery budget and a later comparison interval in advance. A small percentage
gain over another interval can be a practical plateau signal, supported by representative metrics
and CPU checks. Record the chosen heuristic; it is not a proof of convergence. Continue when
productive, advance at diminishing returns, and report any budget-limited uncertainty.

If a larger dataset exceeds hardware capacity, change the search plan explicitly. Bounded-window
GPU discovery plus continuous full-history CPU qualification is one option. Separate windows do
not reproduce a continuous account path. Do not silently shorten history, change precision,
coarsen candles, or drop scenarios and report the original test as completed.

## Preserve Diversity And Apply Limits Deliberately

Retain objective extremes and normalized-space interior diversity instead of seeding only the top
CAGR config. Use `passivbot tool pareto-compress` where appropriate, and preserve CPU-tested leaders
even if GPU sorting later drops them. Budget for full GPU evaluation of each deduplicated seed that
passes configured prefilters; compress redundant points instead of importing a very large front.

Keep seed directories restricted to accepted config/result inputs. Selection reports, manifests,
status, and prose belong outside them: `--start` traverses directories. Compression output includes
`selection.json`; separate that report before treating the directory as a pure config seed pool.
When building config-only seeds, use the canonical result extractor, retain the next stage's bounds
and fixed policy, and remove historical fitness. Do not accidentally carry fine-tuned fixed bounds
into a supposed full-parameter stage. Recheck canonical normalization/clamping before launch.

Use limits to penalize objectively unusable configurations or enforce explicit user requirements,
never to encode the agent's preferences or bias the front toward favored tradeoffs. Inspect activity
and objective support to diagnose unusable behavior; a weak value on one objective alone does not
justify excluding a valid tradeoff. Keep the front diverse.

Before tightening a limit, estimate how much useful diversity survives, including extremes and
interior choices. If the feasible pool empties, diagnose the conflict or search weakness; do not
present the least-penalized infeasible member as a qualified front member. Apply a changed limit in
a new declared stage with fresh evaluation. Filtering old records is exploratory selection, not replay.

For suites, inspect the effective coin intersection and nonempty coverage of every scenario. An
average can conceal a disastrous subset; retain per-scenario diagnostics and suitable worst-case
limits. Supported scenario screening can reduce offspring cost, but only completed full-suite
results count as final fitness. Scenarios used to steer the search are training evidence, even if
introduced later. Keep genuinely unused holdouts or forward checks separate.

Some revisions require symmetric suite approval lists even for a disabled side. If so, use a shared
data universe, keep the unwanted side explicitly disabled, and verify effective side activation in
every prepared context. An approval list alone does not enable a side.

Record how the universe was selected: extending history for today's approved coins does not remove
selection or survivorship bias. If sides or coins were explored independently, replay the combined
portfolio with its intended shared-account and one-way/hedge policy; isolated returns cannot be
added to establish joint performance.

## CPU Qualification Of GPU Candidates

Native GPU optimization supplies its own fitness; independent CPU validation is not automatic.
Check representative extremes and a balanced point early, after material improvements, at stage
gates, and for the final pool. Compare feasibility and ordering as well as numeric deltas.
If feature participation is a requirement, inspect executed activity as well as enabled flags;
for example, an HSL trigger is not proof that an HSL close executed.

Distinguish two questions when readiness differs:

- A **matched-evaluation CPU comparison** uses the same data, trade activation, costs, and other
  execution policies as the GPU result to diagnose numerical/path differences.
- A **canonical candidate CPU backtest** evaluates the actual proposed config, including its own
  readiness. This tests the artifact being proposed for use.

Use owner preparation/payload APIs for diagnostic matching; keep factual cache files unchanged.
Report activation effects separately from remaining GPU/CPU differences. Preserve raw metrics,
native statuses, support counts, actual termination, intended cutoff, and liquidation state.
Never replace invalid/undefined metrics with favorable finite defaults or loosen tolerances to
declare success. See [GPU parity tools](../../gpu_parity.md).

At final qualification evaluate every proposed member, prior CPU leaders, and enough near-front
candidates to address observed ranking differences. Recompute feasibility and non-dominance from
comparable CPU values. Do not relabel the GPU front as CPU-verified after sampling only a few points.

## Monitor, Recover, And Hand Off

Maintain a compact private ledger: source/runtime/data identities, effective config and command,
active run/process, budgets, seeds, frozen stage snapshots, decisions, validation, and next action.
Back up checkpoints, result streams, session identity, and exact front membership; retain hashes.
An append-only local backup can contain dropped members, so directory count alone is not current
membership. Use the canonical loader for compressed result streams rather than assuming each
serialized item is a complete independent record.

Monitor progress timestamps/bars, completed evaluations, full-result growth, checkpoint freshness,
feasibility, front spread, GPU utilization/temperature/memory, RAM/swap, and errors. A checkpoint can
age during a long physical replay; compare it with advancing bars before declaring a stall.
Completion percentages and GPU utilization alone cannot establish useful search progress.

Advance routine stages within the authorized scope. On resource pressure, repeated nonprogress,
or an evaluation-identity problem, preserve evidence and gracefully checkpoint/pause the affected
campaign process. Verify process ownership and command before signalling it; preserve unrelated
runs. Notify meaningful improvements, stage changes, completion, failure, or needed input, and stay
quiet on unchanged healthy state.

Performance research remains secondary. Start with existing logs and inexpensive passive counters;
measure observer cost and distinguish cold preparation from warm replay. High GPU utilization is
not an occupancy or bandwidth measurement. Intrusive profiling, synchronization instrumentation,
extra GPU jobs, and tuning experiments belong on an isolated frozen workload when resources are
available. Do not change the active campaign's code, precision, or clock/power settings as a bonus
experiment. Compare throughput only for equivalent work and verify correctness before adoption.

Consider the campaign complete when the declared final tests are finished, the retained CPU front
passes policy and limits, useful diversity remains, and confirmation offers little material gain.
Deliver the full Pareto configs, a metric table, representative tradeoff annotations,
baseline and scenario comparisons, reproduction evidence, and remaining limitations. If no pool
qualifies, report the unmet requirements and preserve the search evidence; that is not a validated
template. Optimization completion does not authorize deployment.

## Implementation Entry Points

| Question | Owning code |
|---|---|
| Selector expansion, anchors, polish, and starting-config handling | [optimize.py](../../../src/optimize.py) |
| Search-bound history and shared activation | [optimization/warmup.py](../../../src/optimization/warmup.py) |
| Prepared inputs and standalone CPU payloads | [backtest.py](../../../src/backtest.py) |
| Diverse selection and canonical config extraction | [pareto_compress.py](../../../src/pareto_compress.py), [config/result_records.py](../../../src/config/result_records.py) |
| Metric statuses and definitions | [metrics_schema.py](../../../src/metrics_schema.py), [core_metrics.rs](../../../passivbot-rust/src/core_metrics.rs) |
