"""Deterministic whole-file pytest partitioning and collection coverage checks."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


FORMAT_VERSION = 1


def validate_shard(index, count):
    if type(count) is not int or count <= 0:
        raise ValueError("shard count must be a positive integer")
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("shard index must be an integer in [0, shard count)")


def shard_for_path(path, count):
    validate_shard(0, count)
    if (not isinstance(path, str) or not path or "\\" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path):
        raise ValueError("test paths must be canonical repository-relative POSIX paths")
    return int.from_bytes(hashlib.sha256(path.encode("utf-8")).digest(), "big") % count


def selected_nodeids(collected, index, count):
    validate_shard(index, count)
    return [nodeid for nodeid in collected
            if shard_for_path(nodeid.split("::", 1)[0], count) == index]


def verify_manifests(directory, *, count, python_version):
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
        expected = selected_nodeids(manifest["collected"], index, count)
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
    args = parser.parse_args()
    try:
        total = verify_manifests(args.directory, count=args.shard_count,
                                 python_version=args.python_version)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified {args.shard_count} shards: {total} tests on Python {args.python_version}")


if __name__ == "__main__":
    main()
