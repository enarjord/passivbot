"""Deterministic whole-file pytest partitioning and collection coverage checks."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


FORMAT_VERSION = 1
UNKNOWN_FILE_WORKER_MS = 1000


def validate_shard(index, count):
    if type(count) is not int or count <= 0:
        raise ValueError("shard count must be a positive integer")
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("shard index must be an integer in [0, shard count)")


def validate_test_path(path):
    if (not isinstance(path, str) or not path or "\\" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path):
        raise ValueError("test paths must be canonical repository-relative POSIX paths")


def shard_for_path(path, count):
    validate_shard(0, count)
    validate_test_path(path)
    return int.from_bytes(hashlib.sha256(path.encode("utf-8")).digest(), "big") % count


def load_weights(path, python_version):
    """Load the timing snapshot for the Python version running the tests."""
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = snapshot.get(python_version) if isinstance(snapshot, dict) else None
    if not isinstance(weights, dict) or set(weights) != {
        "shard_zero_extra_worker_ms", "files_ms"
    }:
        raise ValueError(f"invalid shard timing snapshot for Python {python_version}")
    extra = weights["shard_zero_extra_worker_ms"]
    files = weights["files_ms"]
    if type(extra) is not int or extra < 0 or not isinstance(files, dict) or not files:
        raise ValueError("invalid shard timing weights")
    for file, duration in files.items():
        validate_test_path(file)
        if type(duration) is not int or duration <= 0:
            raise ValueError(f"invalid test file duration for {file}")
    return weights


def selected_nodeids(collected, index, count, *, weights=None):
    validate_shard(index, count)
    paths = {nodeid.split("::", 1)[0] for nodeid in collected}
    if weights is None:
        assignment = {path: shard_for_path(path, count) for path in paths}
    else:
        for path in paths:
            validate_test_path(path)
        durations = {
            path: weights["files_ms"].get(path, UNKNOWN_FILE_WORKER_MS)
            for path in paths
        }
        loads = [weights["shard_zero_extra_worker_ms"]] + [0] * (count - 1)
        assignment = {}
        # Seed each shard with a file, even when shard zero's smoke work
        # outweighs the entire selected suite (for example with pytest -k).
        empty_shards = set(range(count))
        for path in sorted(paths, key=lambda file: (-durations[file], file)):
            shard = min(empty_shards or range(count), key=lambda shard: (loads[shard], shard))
            assignment[path] = shard
            loads[shard] += durations[path]
            empty_shards.discard(shard)
    return [nodeid for nodeid in collected
            if assignment[nodeid.split("::", 1)[0]] == index]


def verify_manifests(directory, *, count, python_version, weights=None):
    validate_shard(0, count)
    paths = sorted(Path(directory).glob("*.json"))
    if len(paths) != count:
        raise ValueError(f"expected {count} shard manifests, found {len(paths)}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_index = {}
    collected = None
    for manifest in manifests:
        if not isinstance(manifest, dict) or set(manifest) != {
            "format_version", "python_version", "shard_index", "shard_count",
            "collected", "selected",
        }:
            raise ValueError("invalid shard manifest fields")
        if type(manifest["format_version"]) is not int or manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported shard manifest format")
        index = manifest["shard_index"]
        validate_shard(index, manifest["shard_count"])
        if manifest["shard_count"] != count or manifest["python_version"] != python_version:
            raise ValueError("shard count or Python version differs from the requested gate")
        if index in by_index:
            raise ValueError(f"duplicate shard index {index}")
        for key in ("collected", "selected"):
            values = manifest[key]
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{key} must be a nonempty list of node IDs")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {key} node IDs")
        if collected is None:
            collected = manifest["collected"]
        elif manifest["collected"] != collected:
            raise ValueError("shards did not collect the same ordered default test suite")
        expected = selected_nodeids(manifest["collected"], index, count, weights=weights)
        if manifest["selected"] != expected:
            raise ValueError(f"shard {index} has missing, extra, misassigned, or reordered tests")
        by_index[index] = manifest
    # All distinct in-range shard indices are present. Recheck their combined
    # membership rather than relying only on the partition function's contract.
    selected = [nodeid for manifest in by_index.values() for nodeid in manifest["selected"]]
    if len(selected) != len(collected) or set(selected) != set(collected):
        raise ValueError("shards must cover every default-selected test exactly once")
    return len(collected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    try:
        weights = load_weights(args.weights, args.python_version) if args.weights else None
        total = verify_manifests(args.directory, count=args.shard_count,
                                 python_version=args.python_version, weights=weights)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified {args.shard_count} shards: {total} tests on Python {args.python_version}")


if __name__ == "__main__":
    main()
