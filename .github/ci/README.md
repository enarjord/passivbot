# Python CI shards

The main workflow calls `python-tests.yml` separately for Python 3.12 and 3.14.
Each version runs four shards. Assign whole files in descending order of their
estimated worker time to the shard with the lowest estimated load. Break ties
by path and shard index. The estimate for shard zero includes its extra checks.
The timing snapshot affects assignment only. Pytest collection determines the
tests to run, including new files, which start with a one-second estimate.
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
  --ci-shard-weights=.github/ci/python-test-durations.json \
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
  --shard-count=4 --python-version=3.12 \
  --weights=.github/ci/python-test-durations.json
```

The required checks retain the names `Python 3.12`, `Python 3.14` and `Rust`.
Each Python check runs even after failure, cancellation or skipping. It requires
its own version's shard workflow and manifest verification to succeed.

## Comparing runtimes

Compare layouts on the same commit, Python versions, runner class, and worker
count. Omit both weights arguments to use the original SHA-256 path assignment.
Use
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

The initial timing snapshot uses successful four-shard runs
[43280872788](https://github.com/enarjord/passivbot/actions/runs/43280872788),
[43272080676](https://github.com/enarjord/passivbot/actions/runs/43272080676),
[43269237654](https://github.com/enarjord/passivbot/actions/runs/43269237654),
[43261910453](https://github.com/enarjord/passivbot/actions/runs/43261910453),
[43252373336](https://github.com/enarjord/passivbot/actions/runs/43252373336), and
[43136896484](https://github.com/enarjord/passivbot/actions/runs/43136896484).
To refresh it, take the median observed test-file time per Python version from
several successful runs. Round to integer milliseconds with a minimum of one.
Set shard zero's extra worker milliseconds to twice the median wall time of its
documentation, tagged migration, release CLI, and installed wheel steps. Two
workers share the shard, and these steps occupy the runner outside pytest.
