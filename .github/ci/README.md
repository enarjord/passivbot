# Python CI shards

The main workflow calls `python-tests.yml` separately for Python 3.12 and 3.14.
Each version runs four shards. A file is assigned by taking the SHA-256 digest
of its repository-relative POSIX path as an integer modulo the shard count.
Each shard runs two pytest-xdist workers with `--dist=loadfile`, which keeps
the cases from each file together. Workers must collect identical ordered test
IDs. The controller verifies that every selected test reports completion once
before accepting a successful run.

`ci_pytest_shard` is opt-in. It performs normal pytest collection, including the
existing default marker filter, before choosing a shard. The manifest preserves
the original collection order. To run one shard locally:

```bash
PYTHONPATH=.github/ci:src python -m pytest -p ci_pytest_shard \
  --ci-shard-index=0 --ci-shard-count=4 \
  --ci-shard-manifest=/tmp/shard-0.json
```

Both shard options are required together. Invalid or empty shards are errors.
The existing test-suite flags and exclusions remain in effect; sharding adds no
skips. Each shard has a 90-minute timeout, builds and verifies its native
extension, and caps numeric-library threads at one. Documentation checks and
release CLI smokes run on shard zero of each Python version.

Each shard uploads its full default-selected collection and selected node IDs.
The per-version Python check requires every shard (including documentation and
CLI checks) to succeed, then verifies all four manifests: the Python version,
shard indices/count, identical ordered full collections, correct file assignment,
original selected order, and complete/disjoint coverage. Missing, duplicate,
empty or inconsistent manifests fail the gate. Verification uses only Python's
standard library:

```bash
python .github/ci/ci_sharding.py /tmp/collection \
  --shard-count=4 --python-version=3.12
```

The required checks retain the names `Python 3.12`, `Python 3.14` and `Rust`.
Each Python check runs even after failure, cancellation or skipping. It requires
its own version's shard workflow and manifest verification to succeed.

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
