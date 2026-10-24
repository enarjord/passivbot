# Python CI shards

The main workflow calls `python-tests.yml` separately for Python 3.12 and 3.14.
Each version runs eight shards. A test file belongs to the shard given by
`int.from_bytes(sha256(relative_posix_path.encode("utf-8")).digest(), "big") % 8`.
The path is relative to the repository root; the entire digest is used, without
salts or timing tables.

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
