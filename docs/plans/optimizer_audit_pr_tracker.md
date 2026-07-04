# Optimizer Audit PR Tracker

Tracks the v8 optimizer audit follow-up work as targeted pull requests. Each
implementation PR should stay narrow, wait for current-head Claude and Hermes
approval, and merge only after both reviewers approve the same head.

## Review Gate

- Claude approval must mention the PR's current head SHA.
- Hermes approval must mention the PR's current head SHA.
- Reviewer findings are addressed on the same PR before merge, and the follow-up
  fix delta must itself receive current-head Claude and Hermes greenlight before
  merging.
- Orthogonal follow-up PRs may proceed while earlier PRs wait for reviews.

## Merged

- [x] #1044 - Fix optimizer grid quantization for non-power-of-ten steps.
- [x] #1045 - Propagate DEAP evaluated vectors back to parent process.
- [x] #1046 - Harden Pareto store bootstrap and recorder failures.
- [x] #1047 - Make suite optimizer scenario drops fail loud.
- [x] #1048 - Add optional optimizer seed control; default remains random.
- [x] #1049 - Validate optimizer overrides and verify v7 TP grid overrides.
- [x] #1050 - Preserve anchored fine-tune seed anchor ids.
- [x] #1051 - Fix optimizer pool shutdown on KeyboardInterrupt.
- [x] #1052 - Treat suite config as suite mode.
- [x] #1053 - Preserve suite override aggregate semantics.
- [x] #1054 - Reject unknown suite scenario keys.
- [x] #1055 - Honor pymoo resume runtime state changes.
- [x] #1056 - Preserve deleted keys in compressed all_results deltas.
- [x] #1057 - Fail loud on unresolved Pareto limit metrics.
- [x] #1058 - Correct high-priority optimizer documentation drift.
- [x] #1059 - Close suite evaluator master attachments.
- [x] #1060 - Remove broken unused opt_utils helpers.
- [x] #1061 - Reject empty optimizer bounds in config adapter.
- [x] #1062 - Reject non-finite Pareto prune objectives.
- [x] #1063 - Clean duplicate ExchangeDataset metadata.
- [x] #1064 - Remove unused PymooRecorderCallback.
- [x] #1065 - Remove unused OptimizeOutput.
- [x] #1066 - Remove unused is_anchored_shape_key.
- [x] #1067 - Use scored Pareto metadata during bootstrap.
- [x] #1068 - Remove unused ParetoStore bounds/helper cleanup path.
- [x] #1069 - Remove duplicate optimize argparse import.
- [x] #1070 - Remove unused ParetoPoint dataclass.
- [x] #1071 - Install DEAP evaluator once per worker.
- [x] #1072 - Avoid forced optimizer data pre-copy.
- [x] #1073 - Stream resume result validation.
- [x] #1074 - Bound DEAP pending offspring evaluations.
- [x] #1075 - Remove unused opt_utils helpers.
- [x] #1076 - Correct optimizer limit stat docs.

## Open PRs

- [ ] #1077 - Track optimizer audit PR progress.
  - Status: mergeable.
  - Gate: waiting for Claude and Hermes review on the current PR head after
    tracker updates.
- [ ] #1078 - Reuse pymoo starting seed evaluations.
  - Status: mergeable.
  - Gate: rebased onto current `v8` after #1074-#1076 merged; waiting for fresh
    Claude and Hermes review on head
    `1bd99a86c9ceb10937fa37e077cb412ff1a00981`.
- [ ] #1079 - Check Pareto dominance in one pass.
  - Status: mergeable.
  - Gate: rebased onto current `v8` after #1074-#1076 merged; waiting for fresh
    Claude and Hermes review on head
    `f35daa3a93ca54fc85d00a323eabe2e400327f4e`.
- [ ] #1080 - Remove unused all_results seed skip branch.
  - Status: mergeable.
  - Gate: waiting for Claude and Hermes review on head
    `1826e53bc1ca6f5bf6333a282545defcc6c59c13`.

## Remaining Or Paused Audit Items

- [ ] ParetoStore add-entry performance.
  - Current note: exact objective-vector lookup is already in place, and #1079
    covers the safe dominance-pass cleanup. Remaining heavier work is
    synchronous JSON write/profiling-driven behavior and should not be changed
    without measurement.
- [ ] ParetoStore.get_front deletion claim.
  - Current note: tests still exercise the method; not treated as dead without
    either removing test need or finding production replacement.

## Out Of Scope For Optimizer Audit PRs

- #1043 - `codex/v8-smoke-cache-health` is unrelated to this optimizer audit
  tracker and remains ignored here unless explicitly requested.
