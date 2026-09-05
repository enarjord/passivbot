"""Version the code that produces optimizer fitness, independently of repository commits."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from rust_utils import verify_loaded_runtime_extension

# These modules own runtime config, candidate payloads, simulation orchestration,
# and fitness/constraint calculation. Live connectors, docs, and tests are excluded.
_EVALUATOR_FILES = (
    "backtest.py",
    "config_utils.py",
    "limit_utils.py",
    "metrics_schema.py",
    "optimize.py",
    "optimize_suite.py",
    "optimizer_overrides.py",
    "opt_utils.py",
    "pure_funcs.py",
    "suite_runner.py",
    "warmup_utils.py",
    "tools/iterative_backtester.py",
)
_EVALUATOR_PACKAGES = ("config", "optimization")


def python_evaluator_source_fingerprint(source_root: Path) -> str:
    paths = {source_root / name for name in _EVALUATOR_FILES}
    for package in _EVALUATOR_PACKAGES:
        paths.update((source_root / package).rglob("*.py"))
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def evaluation_implementation_identity() -> dict:
    """Freeze the verified evaluator identity once per optimizer process."""
    runtime = verify_loaded_runtime_extension()
    rust_source = runtime.get("runtime_compiled_source_stamp")
    rust_artifact = runtime.get("runtime_compiled_sha256")
    if not rust_source or not rust_artifact:
        raise RuntimeError(
            "Optimizer evaluation identity requires a source-stamped Rust extension; "
            "rebuild and verify the extension before starting or resuming"
        )
    return {
        "version": 1,
        "python_source_sha256": python_evaluator_source_fingerprint(
            Path(__file__).resolve().parents[1]
        ),
        "rust_source_sha256": rust_source,
        "rust_artifact_sha256": rust_artifact,
    }
