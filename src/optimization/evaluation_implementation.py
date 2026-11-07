"""Identify optimizer evaluator code and its installed Python environment."""

from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import sys

from rust_utils import verify_loaded_runtime_extension


def _hash_sources(source_root, paths):
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def evaluation_implementation_identity() -> dict:
    """Freeze the verified evaluator implementation and environment once per process."""
    runtime = verify_loaded_runtime_extension()
    rust_source = runtime.get("runtime_compiled_source_stamp")
    rust_artifact = runtime.get("runtime_compiled_sha256")
    if not rust_source or not rust_artifact:
        raise RuntimeError(
            "Optimizer evaluation identity requires a source-stamped Rust extension; "
            "rebuild and verify the extension before starting or resuming"
        )
    source_root = Path(__file__).resolve().parents[1]
    # A conservative manifest is cheaper than walking every AST and resolving
    # providers for dynamic imports on each CLI start. Hash source contents and
    # every installed distribution, so an unrelated change can invalidate reuse
    # but a dynamically selected module/dependency cannot escape the identity.
    paths = set(source_root.rglob("*.py"))
    distributions = []
    for distribution in metadata.distributions():
        text = distribution.read_text("METADATA")
        if text is None:
            text = distribution.read_text("PKG-INFO")
        if text is None:
            raise RuntimeError("Installed distribution has no reproducible metadata")
        distributions.append({
            "name": distribution.metadata["Name"], "version": distribution.version,
            "metadata_sha256": sha256(text.encode()).hexdigest(),
        })
    distributions.sort(key=lambda item: (item["name"], item["version"], item["metadata_sha256"]))
    return {
        "version": 2,
        "python_source_sha256": _hash_sources(source_root, paths),
        "python_runtime": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "python_dependencies": {"distributions": distributions},
        "rust_source_sha256": rust_source,
        "rust_artifact_sha256": rust_artifact,
    }
