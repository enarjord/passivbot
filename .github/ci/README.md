# Python CI shards

The main workflow calls `python-tests.yml` separately for Python 3.12 and 3.14.
Each version runs eight shards. Most files are assigned by taking the SHA-256
digest of their repository-relative POSIX path as an integer modulo the shard
count. The eight-shard matrix uses a small set of whole-file overrides in
`ci_sharding.py` based on [measured runtimes on both Python versions](https://github.com/enarjord/passivbot/actions/runs/43056104234).
When rebalancing, compare the slowest shards on both versions and include the
extra checks on shard zero in the comparison.

`ci_pytest_shard` is opt-in. It performs normal pytest collection, including the
existing default marker filter, before choosing a shard. Every item in a file
stays together, and the selected items retain their original collection order.
Ordinary local `pytest` commands are unchanged. To run one shard locally:

```bash
PYTHONPATH=.github/ci:src python -m pytest -p ci_pytest_shard \
  --ci-shard-index=0 --ci-shard-count=8 \
  --ci-shard-manifest=/tmp/shard-0.json
```

Both shard options are required together. Invalid or empty shards are errors.
The existing test-suite flags and exclusions remain in effect; sharding adds no
skips. Each shard has a 90-minute timeout, builds and verifies its native
extension, and caps numeric-library threads at one. Documentation checks and
release CLI smokes run on shard zero of each Python version.

Each shard uploads its full default-selected collection and selected node IDs.
The per-version coverage job requires every shard (including documentation and
CLI checks) to succeed, then verifies all eight manifests: the Python version,
shard indices/count, identical ordered full collections, correct file assignment,
original selected order, and complete/disjoint coverage. Missing, duplicate,
empty or inconsistent manifests fail the gate. Verification uses only Python's
standard library:

```bash
python .github/ci/ci_sharding.py /tmp/collection \
  --shard-count=8 --python-version=3.12
```

The required checks retain the names `Python 3.12`, `Python 3.14` and `Rust`.
Each Python aggregate runs even after failure, cancellation or skipping, and only
accepts success from its own version's reusable workflow. It cannot be satisfied
by the other Python version. The Rust job is unchanged.

## Comparing runtimes

Compare completed runs with the same Python versions and test layout. Use
`created_at` to `started_at` for initial job wait and `started_at` to
`completed_at` for job runtime. Dependent jobs' initial wait includes time
waiting on their prerequisites. The workflow's trigger-to-completion time
includes both waiting and execution.

```bash
gh api "repos/enarjord/passivbot/actions/runs/$RUN_ID" \
  --jq '[.created_at, .updated_at, .conclusion] | @tsv'
gh api "repos/enarjord/passivbot/actions/runs/$RUN_ID/jobs?per_page=100" \
  --jq '.jobs[] | [.name, .created_at, .started_at, .completed_at, .conclusion] | @tsv'
```

Each Python shard logs `CI_TEST_FILE` JSON records sorted by total test time.
The records sum pytest's setup, call, and teardown durations by file. They
exclude collection and CI setup. With parallel pytest workers, summed test time
can exceed the pytest step's wall time. Use job step timestamps for wall time.

```bash
gh api "repos/enarjord/passivbot/actions/jobs/$JOB_ID/logs" \
  | sed -n 's/^.* CI_TEST_FILE //p'
```
